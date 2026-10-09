"""One attempt per SWE-bench task; durable starts prevent re-sampling on resume."""

import argparse
import asyncio
import fcntl
import hashlib
import json
import time
from pathlib import Path
from urllib.parse import urlsplit

import httpx

from adaptive_branching.src.swe.agent import AgentConfig
from adaptive_branching.tools.swe.run_harbor_benchmark import (
    JsonlRecorder,
    bind_model_session,
    create_model_session,
    delete_model_session,
    load_samples,
)

BUDGETS = {"rl": AgentConfig(), "large": AgentConfig(max_turns=250, max_tokens=32768, model_context=262144)}
AGENTS = {
    name: "adaptive_branching.src.swe.verified_harbor_agent:Verified" + suffix + "Agent"
    for name, suffix in [("rl", "Small"), ("large", "Large")]
}
BENCHMARKS = {"verified": (500, "4.0.3"), "lite": (300, "5.0.2"), "multilingual": (300, "5.0.2")}
LITE_IDENTITY = {"dataset": "SWE-bench_Lite", "dataset_repo": "SWE-bench/SWE-bench_Lite", "dataset_split": "test"}
NORMAL_ENDINGS = {"Submitted", "TurnLimit", "ContextLimit", "FormatLimit", "LengthTruncated"}


class DispatchGuard:
    """Pause new attempts after infrastructure failures while draining active work."""

    def __init__(self, directory, error_limit=3):
        if not isinstance(directory, Path) or not directory.is_dir():
            raise ValueError("dispatch guard requires an existing output directory")
        if type(error_limit) is not int or error_limit < 1:
            raise ValueError("error_limit must be a positive integer")
        self.path = directory / "PAUSE.json"
        self.error_limit = error_limit
        self.consecutive_errors = 0

    async def wait(self):
        while self.path.exists():
            if not self.path.is_file():
                raise ValueError(f"pause marker must be a file: {self.path}")
            await asyncio.sleep(1)

    def record(self, row):
        if not isinstance(row, dict) or not isinstance(row.get("instance_id"), str) or not row["instance_id"]:
            raise ValueError("dispatch guard requires a result with an instance ID")
        error = row.get("error")
        if error is not None and (not isinstance(error, str) or not error):
            raise ValueError("result error must be nonempty text or None")
        self.consecutive_errors = self.consecutive_errors + 1 if error else 0
        if self.consecutive_errors >= self.error_limit and not self.path.exists():
            receipt = {
                "reason": "consecutive_execution_errors",
                "count": self.consecutive_errors,
                "last_instance": row["instance_id"],
                "last_error": error,
                "paused_at": time.time(),
            }
            self.path.write_text(json.dumps(receipt, indent=2) + "\n")
            print("DISPATCH_PAUSED " + json.dumps(receipt), flush=True)


def score_response(body, instance, budget, *, benchmark="verified"):
    """Keep official raw results, but match RL's terminal-length failure rule."""
    if benchmark not in BENCHMARKS:
        raise ValueError(f"unknown benchmark: {benchmark}")
    if not isinstance(body, dict) or not isinstance(instance, str) or not instance or budget not in BUDGETS:
        raise ValueError("score_response requires a response, instance ID and known budget")
    if body.get("exit_status") not in NORMAL_ENDINGS:
        raise ValueError(f"unsuccessful execution: {body.get('exit_status')!r}")
    reward, report = body.get("reward"), body.get("eval_report")
    if type(reward) not in (int, float) or reward not in (0, 1) or not isinstance(report, dict):
        raise ValueError("missing binary verifier result")
    if report.get("instance_id") != instance or report.get("official_harness_version") != BENCHMARKS[benchmark][1]:
        raise ValueError("wrong official verifier instance/version")
    # Lite and Verified contain overlapping IDs; version alone is not identity.
    # The dataset marker prevents dispatch to an accidentally reused task set.
    if benchmark == "lite" and report.get("dataset") != LITE_IDENTITY["dataset"]:
        raise ValueError("wrong Lite verifier dataset")
    if type(report.get("reward")) not in (int, float) or report["reward"] != reward:
        raise ValueError("verifier reward mismatch")
    official = report.get("official_report")
    if not isinstance(official, dict) or not official:
        raise ValueError("missing official report")
    if instance in official:
        if not isinstance(official[instance], dict):
            raise ValueError("official instance report must be an object")
        if benchmark in ("lite", "multilingual") and official[instance].get("infra_failure", False):
            raise ValueError(f"official verifier reported infrastructure failure: {official[instance]}")
        resolved = official[instance].get("resolved")
        if type(resolved) is not bool or int(resolved) != reward:
            raise ValueError("official resolved flag disagrees with reward")
    elif not (
        reward == 0
        and official.get("schema_version") == 2
        and official.get("total_instances") == official.get("submitted_instances") == 1
        and official.get("submitted_ids") == official.get("empty_patch_ids") == [instance]
        and all(official.get(key) == [] for key in ("resolved_ids", "error_ids", "incomplete_ids"))
    ):
        raise ValueError("invalid empty-patch report")
    metrics = body.get("agent_metrics")
    if not isinstance(metrics, dict) or type(metrics.get("turns")) is not int:
        raise ValueError("missing turn telemetry")
    if not 0 <= metrics["turns"] <= BUDGETS[budget].max_turns:
        raise ValueError("turn budget violated")
    return 0 if benchmark in ("verified", "lite") and body["exit_status"] == "LengthTruncated" else int(reward)


def read_records(path):
    if not isinstance(path, Path):
        raise TypeError("journal path must be a Path")
    records = {}
    if path.exists():
        for line in path.read_text().splitlines():
            row = json.loads(line)
            instance = row["instance_id"]
            if not isinstance(instance, str) or not instance or instance in records:
                raise ValueError(f"invalid/duplicate instance in {path}: {instance!r}")
            records[instance] = row
    return records


def verifier_timeout_failure(body, instance):
    """Count an evidenced official 1800-second test timeout as zero.

    Missing reports or generic transport errors alone never establish a timeout.
    The original response and on-disk trial remain unchanged for audit.
    """
    if not isinstance(body, dict) or not isinstance(instance, str) or not instance:
        raise ValueError("response object and nonempty instance required")
    if body.get("exit_status") not in NORMAL_ENDINGS or not body.get("trial_dir"):
        return None
    trial = Path(body["trial_dir"])
    if not trial.is_absolute() or not trial.is_dir():
        raise ValueError(f"invalid verifier trial path: {trial}")
    stdout = trial / "verifier/test-stdout.txt"
    predictions = trial / "verifier/predictions.json"
    if not stdout.is_file() or not predictions.is_file():
        return None
    marker = f"{instance}: Test timed out after 1800 seconds."
    if marker not in stdout.read_text():
        return None
    rows = json.loads(predictions.read_text())
    if (
        not isinstance(rows, list)
        or len(rows) != 1
        or not isinstance(rows[0], dict)
        or rows[0].get("instance_id") != instance
        or not isinstance(rows[0].get("model_patch"), str)
    ):
        raise ValueError(f"timeout prediction identity mismatch: {predictions}")
    return dict(
        policy="official_test_timeout_is_failure",
        timeout_seconds=1800,
        score=0,
        evidence=str(stdout),
        evidence_sha256=hashlib.sha256(stdout.read_bytes()).hexdigest(),
        patch_sha256=hashlib.sha256(rows[0]["model_patch"].encode()).hexdigest(),
    )


def summary(records, total=500):
    if type(total) is not int or total <= 0 or not isinstance(records, dict) or len(records) > total:
        raise ValueError("invalid summary denominator or records")
    if any(type(row.get("score")) is not int or row["score"] not in (0, 1) for row in records.values()):
        raise ValueError("summary requires binary scores")
    solved = sum(row["score"] for row in records.values())
    return {
        "completed": len(records),
        "total": total,
        "solved": solved,
        "score": solved / total,
        "errors": sum(bool(row.get("error")) for row in records.values()),
        "missing": total - len(records),
    }


def verifier_summary(records, total, budget, benchmark):
    """Report official outcomes over the full denominator, without reward shaping."""
    summary(records, total=total)  # Validate the journal and denominator first.
    if budget not in BUDGETS or benchmark not in BENCHMARKS:
        raise ValueError("known budget and benchmark required")
    correct = truncated_correct = test_timeouts = 0
    for instance, row in records.items():
        if row.get("error"):
            continue  # Infrastructure failures remain explicit in the regular summary.
        adjudication = row.get("adjudication")
        if adjudication is not None:
            if (
                not isinstance(adjudication, dict)
                or adjudication.get("policy") != "official_test_timeout_is_failure"
                or row["score"] != 0
            ):
                raise ValueError(f"invalid verifier timeout adjudication: {instance}")
            test_timeouts += 1
            continue
        body = row.get("response")
        checked_score = score_response(body, instance, budget, benchmark=benchmark)
        if checked_score != row["score"]:
            raise ValueError(f"persisted score disagrees with response: {instance}")
        correct += body["reward"]
        truncated_correct += int(body["reward"] == 1 and body["exit_status"] == "LengthTruncated")
    return dict(
        verifier_correct=correct,
        verifier_accuracy=correct / total,
        length_truncated_verifier_correct=truncated_correct,
        verifier_test_timeouts=test_timeouts,
    )


async def evaluate(args, *, transport=None):
    if args.budget not in BUDGETS or type(args.concurrency) is not int or not 1 <= args.concurrency <= 128:
        raise ValueError("invalid budget/concurrency")
    if type(args.timeout) is not int or args.timeout <= 0:
        raise ValueError("timeout must be positive")
    if not isinstance(args.model, str) or not args.model.strip():
        raise ValueError("model must be nonempty")
    for name in ("harbor", "session"):
        url = urlsplit(getattr(args, name))
        if (url.scheme not in {"http", "https"} or not url.hostname or url.username or url.password
                or url.query or url.fragment):
            raise ValueError(f"{name} must be an absolute HTTP(S) URL without credentials, query, or fragment")
    benchmark = getattr(args, "benchmark", "verified")
    if benchmark not in BENCHMARKS:
        raise ValueError(f"unknown benchmark: {benchmark}")
    total, _ = BENCHMARKS[benchmark]
    samples = load_samples(args.prompts)
    if len(samples) != total:
        raise ValueError(f"{benchmark} requires exactly {total} unique tasks, got {len(samples)}")
    if benchmark == "multilingual" and any(s["metadata"].get("dataset") != "SWE-bench_Multilingual" for s in samples):
        raise ValueError("Multilingual requires correctly labeled prompts")
    if benchmark == "lite" and any(
        any(sample["metadata"].get(key) != value for key, value in LITE_IDENTITY.items()) for sample in samples
    ):
        raise ValueError("Lite requires correctly labeled dataset/repository/test-split prompts")
    ids = {s["metadata"]["instance_id"] for s in samples}
    args.output.mkdir(parents=True, exist_ok=True)
    guard = DispatchGuard(args.output, getattr(args, "pause_after_errors", 3))
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    config.update(
        prompt_sha256=hashlib.sha256(args.prompts.read_bytes()).hexdigest(),
        agent=AGENTS[args.budget],
        budget_config=vars(BUDGETS[args.budget]),
        attempts=1,
        denominator=total,
        length_truncated_score=0 if benchmark in ("verified", "lite") else "official_verifier",
        temperature=1.0,
    )
    receipt = args.output / "config.json"
    if receipt.exists() and json.loads(receipt.read_text()) != config:
        raise ValueError("existing evaluation configuration differs")
    receipt.write_text(json.dumps(config, indent=2) + "\n")
    starts = read_records(args.output / "started.jsonl")
    results = read_records(args.output / "results.jsonl")
    if not set(results) <= set(starts) <= ids:
        raise ValueError("journal contains unknown or unstarted results")
    recorder = JsonlRecorder(args.output / "results.jsonl")
    starter = JsonlRecorder(args.output / "started.jsonl")
    # A process crash after dispatch is ambiguous. Conservatively count zero;
    # never create a second trajectory for a durably started instance.
    for instance in starts.keys() - results.keys():
        row = {"instance_id": instance, "score": 0, "error": "InterruptedAttempt", "attempt": 1}
        await recorder.append(row)
        results[instance] = row
    semaphore = asyncio.Semaphore(args.concurrency)
    fatal_error = asyncio.Event()
    limits = httpx.Limits(max_connections=args.concurrency * 2, max_keepalive_connections=args.concurrency)
    async with httpx.AsyncClient(timeout=args.timeout, limits=limits, transport=transport) as client:

        async def run_one(sample):
            instance = sample["metadata"]["instance_id"]
            if instance not in ids:
                raise ValueError("unknown task")
            async with semaphore:
                if fatal_error.is_set():
                    return
                await guard.wait()
                row = {
                    "instance_id": instance,
                    "attempt": 1,
                    "started_at": time.time(),
                    "score": 0,
                    "error": None,
                    "response": None,
                }
                await starter.append({"instance_id": instance, "started_at": row["started_at"], "attempt": 1})
                session = None
                try:
                    async with asyncio.timeout(args.timeout):
                        session = await create_model_session(client, args.session)
                        budget = BUDGETS[args.budget]
                        request = bind_model_session(
                            {
                                "instance_id": instance,
                                "model": "openai/" + args.model,
                                "api_key": "dummy",
                                "agent_name": AGENTS[args.budget],
                                "max_seq_len": budget.model_context,
                                "max_turns": budget.max_turns,
                                "context_reserve_tokens": None,
                                "force_submit_on_limit": False,
                                "run_verifier": True,
                                "sampling_params": {
                                    "temperature": 1.0,
                                    "max_tokens": budget.max_tokens,
                                    "chat_template_kwargs": {"enable_thinking": True},
                                },
                            },
                            session,
                        )
                        row["request"] = request
                        response = await client.post(args.harbor.rstrip("/") + "/run", json=request)
                        response.raise_for_status()
                        row["response"] = response.json()
                        row["score"] = score_response(row["response"], instance, args.budget, benchmark=benchmark)
                except (httpx.HTTPError, TimeoutError, ValueError, TypeError, KeyError) as exc:
                    # Preserve the cause before applying the selected failure policy.
                    row["error"] = f"{type(exc).__name__}: {exc}"
                    if benchmark in ("verified", "lite") and row["response"] is not None:
                        adjudication = verifier_timeout_failure(row["response"], instance)
                        if adjudication is not None:
                            row["original_error"] = row["error"]
                            row["error"] = None
                            row["score"] = 0
                            row["adjudication"] = adjudication
                finally:
                    if session is not None:
                        try:
                            await delete_model_session(client, session)
                        except httpx.HTTPError as exc:
                            row["cleanup_error"] = f"{type(exc).__name__}: {exc}"
                row["finished_at"] = time.time()
                await recorder.append(row)
                results[instance] = row
                if (row["error"] or row.get("cleanup_error")) and not getattr(args, "keep_going", False):
                    fatal_error.set()
                    raise RuntimeError(
                        f"SWE evaluation failed for {instance}; result saved in {args.output / 'results.jsonl'}: "
                        f"{row['error'] or row['cleanup_error']}"
                    )
                guard.record(row)
                progress = summary(results, total=total)
                progress.update(verifier_summary(results, total, args.budget, benchmark))
                print("PROGRESS " + json.dumps(progress), flush=True)

        await asyncio.gather(*(run_one(s) for s in samples if s["metadata"]["instance_id"] not in starts))
    result = summary(results, total=total)
    result.update(verifier_summary(results, total, args.budget, benchmark))
    if result["completed"] != total:
        raise AssertionError(f"not all {total} tasks accounted for")
    if result["errors"] and not getattr(args, "keep_going", False):
        raise RuntimeError(f"evaluation contains {result['errors']} infrastructure failures; inspect {args.output}")
    (args.output / "DONE.json").write_text(json.dumps(result, indent=2) + "\n")
    print("EVALUATION_DONE " + json.dumps(result), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", choices=BENCHMARKS, default="verified")
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--budget", choices=BUDGETS, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--harbor", required=True)
    parser.add_argument("--session", required=True)
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--timeout", type=int, default=24000)
    parser.add_argument("--pause-after-errors", type=int, default=3)
    parser.add_argument("--keep-going", action="store_true", help="record infrastructure failures and continue")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "runner.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        asyncio.run(evaluate(args))


if __name__ == "__main__":
    main()
