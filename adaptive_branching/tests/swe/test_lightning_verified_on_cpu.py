import asyncio
import importlib
import json
import os
import shlex
import subprocess
import sys
from types import SimpleNamespace

import httpx
import pytest

from adaptive_branching.tests.swe.test_lightning_integration_on_cpu import load_bridge
from adaptive_branching.tools.swe import run_lightning_verified as runner


def response(instance="task-0", *, reward=1, status="Submitted", turns=1):
    return {
        "reward": reward,
        "exit_status": status,
        "agent_metrics": {"turns": turns},
        "eval_report": {
            "instance_id": instance,
            "official_harness_version": "4.0.3",
            "reward": reward,
            "official_report": {instance: {"resolved": bool(reward)}},
        },
    }


@pytest.mark.parametrize("status", sorted(runner.NORMAL_ENDINGS))
@pytest.mark.parametrize("reward", [0, 1])
def test_score_uses_official_outcome_and_length_rule(status, reward):
    assert runner.score_response(response(status=status, reward=reward), "task-0", "rl") == (
        0 if status == "LengthTruncated" else reward
    )


@pytest.mark.parametrize(
    "turns,budget,valid",
    [
        (100, "rl", True),
        (101, "rl", False),
        (250, "large", True),
        (251, "large", False),
        (-1, "rl", False),
        (0, "rl", True),
    ],
)
def test_score_turn_boundaries(turns, budget, valid):
    if valid:
        assert runner.score_response(response(turns=turns), "task-0", budget) == 1
    else:
        with pytest.raises(ValueError):
            runner.score_response(response(turns=turns), "task-0", budget)


@pytest.mark.parametrize(
    "change",
    [
        dict(exit_status="TimeLimitExceeded"),
        dict(reward=True),
        dict(reward=None),
        dict(eval_report={}),
        dict(agent_metrics={}),
    ],
)
def test_invalid_results_fail_validation(change):
    with pytest.raises(ValueError):
        runner.score_response({**response(), **change}, "task-0", "rl")


def test_summary_keeps_full_denominator_and_empty():
    assert runner.summary({}) == dict(completed=0, total=500, solved=0, score=0, errors=0, missing=500)
    assert runner.summary({"one": {"score": 1}})["score"] == 1 / 500
    with pytest.raises(ValueError):
        runner.summary({"one": {"score": True}})
    with pytest.raises(ValueError):
        runner.summary({}, total=0)


def test_verifier_summary_keeps_truncated_success_and_full_denominator():
    assert runner.verifier_summary({}, 500, "rl", "verified")["verifier_accuracy"] == 0
    records = {
        "task-0": dict(score=0, error=None, response=response(status="LengthTruncated")),
        "task-1": dict(score=1, error=None, response=response("task-1")),
        "task-2": dict(score=0, error="transport failed", response=None),
        "task-3": dict(score=0, error=None, adjudication=dict(policy="official_test_timeout_is_failure")),
    }
    result = runner.verifier_summary(records, 500, "rl", "verified")
    assert result == dict(
        verifier_correct=2, verifier_accuracy=2 / 500, length_truncated_verifier_correct=1, verifier_test_timeouts=1
    )
    assert runner.verifier_summary({"task-1": records["task-1"]}, 1, "rl", "verified")["verifier_accuracy"] == 1


@pytest.mark.parametrize(
    "invalid", ["denominator", "budget", "benchmark", "missing-response", "score", "reward", "adjudication"]
)
def test_verifier_summary_fails_on_invalid_evidence(invalid):
    record = dict(score=1, error=None, response=response())
    total, budget, benchmark = 500, "rl", "verified"
    if invalid == "denominator":
        total = 0
    elif invalid == "budget":
        budget = "unknown"
    elif invalid == "benchmark":
        benchmark = "unknown"
    elif invalid == "missing-response":
        record.pop("response")
    elif invalid == "score":
        record["score"] = 0
    elif invalid == "reward":
        record["response"]["reward"] = True
    elif invalid == "adjudication":
        record["adjudication"] = dict(policy="unknown")
    with pytest.raises(ValueError):
        runner.verifier_summary({"task-0": record}, total, budget, benchmark)


def test_dispatch_guard_validation_and_success_reset(tmp_path):
    for limit in [0, -1, True]:
        with pytest.raises(ValueError):
            runner.DispatchGuard(tmp_path, limit)
    with pytest.raises(ValueError):
        runner.DispatchGuard(tmp_path / "missing")
    guard = runner.DispatchGuard(tmp_path, 2)
    for row in [{}, {"instance_id": "x", "error": 1}]:
        with pytest.raises(ValueError):
            guard.record(row)
    guard.record({"instance_id": "a", "error": "infra"})
    guard.record({"instance_id": "b", "error": None})
    guard.record({"instance_id": "c", "error": "infra"})
    assert not guard.path.exists()
    guard.record({"instance_id": "d", "error": "infra"})
    assert json.loads(guard.path.read_text())["count"] == 2


@pytest.mark.asyncio
async def test_dispatch_guard_waits_until_operator_removes_pause(tmp_path):
    guard = runner.DispatchGuard(tmp_path, 1)
    await guard.wait()
    guard.record({"instance_id": "a", "error": "infra"})
    waiter = asyncio.create_task(guard.wait())
    await asyncio.sleep(0)
    assert not waiter.done()
    guard.path.unlink()
    await asyncio.wait_for(waiter, timeout=2)
    guard.path.mkdir()
    with pytest.raises(ValueError):
        await guard.wait()


@pytest.mark.asyncio
async def test_infra_pause_preserves_results_and_never_retries(tmp_path):
    prompts = tmp_path / "prompts.jsonl"
    prompts.write_text(
        "".join(json.dumps({"prompt": "fix", "metadata": {"instance_id": f"task-{i}"}}) + "\n" for i in range(500))
    )
    args = SimpleNamespace(
        prompts=prompts,
        output=tmp_path / "results",
        budget="rl",
        model="base",
        harbor="http://harbor",
        session="http://session",
        concurrency=2,
        timeout=20,
        keep_going=True,
        pause_after_errors=3,
    )
    attempts = []

    async def handler(request):
        if request.url.path == "/health":
            return httpx.Response(200, json={"session_server_instance_id": "server"})
        if request.url.path == "/sessions":
            return httpx.Response(200, json={"session_id": "session"})
        if request.method == "DELETE":
            return httpx.Response(200, json={})
        instance = json.loads(request.content)["instance_id"]
        attempts.append(instance)
        await asyncio.sleep(0)
        if int(instance.split("-")[1]) < 3:
            raise httpx.ReadError("infrastructure failure")
        return httpx.Response(200, json=response(instance))

    task = asyncio.create_task(runner.evaluate(args, transport=httpx.MockTransport(handler)))
    try:

        async def wait_for_pause():
            while not (args.output / "PAUSE.json").exists():
                await asyncio.sleep(0.01)

        await asyncio.wait_for(wait_for_pause(), timeout=5)
        await asyncio.sleep(0.05)  # Let already-dispatched work drain.
        count = len(attempts)
        assert 3 <= count <= 4
        assert len((args.output / "results.jsonl").read_text().splitlines()) == count
        await asyncio.sleep(0.05)
        assert len(attempts) == count
        (args.output / "PAUSE.json").unlink()
        result = await asyncio.wait_for(task, timeout=10)
        assert result["completed"] == 500 and result["errors"] == 3 and result["solved"] == 497
        assert len(attempts) == len(set(attempts)) == 500
    finally:
        if not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)


@pytest.mark.parametrize("concurrency", [1, 64, 128])
def test_all_500_once_errors_zero_and_resume_without_resampling(tmp_path, concurrency):
    prompts = tmp_path / "prompts.jsonl"
    prompts.write_text(
        "".join(json.dumps({"prompt": "fix", "metadata": {"instance_id": f"task-{i}"}}) + "\n" for i in range(500))
    )
    args = SimpleNamespace(
        prompts=prompts,
        output=tmp_path / "results",
        budget="large",
        model="base",
        harbor="http://harbor",
        session="http://session",
        concurrency=concurrency,
        timeout=20,
        keep_going=True,
    )
    attempts = []

    def handler(request):
        if request.url.path == "/health":
            return httpx.Response(200, json={"session_server_instance_id": "server"})
        if request.url.path == "/sessions":
            return httpx.Response(200, json={"session_id": "session"})
        if request.method == "DELETE":
            return httpx.Response(200, json={})
        assert request.url.path == "/run"
        body = json.loads(request.content)
        instance = body["instance_id"]
        attempts.append(instance)
        assert body["max_turns"] == 250 and body["sampling_params"]["max_tokens"] == 32768
        assert body["force_submit_on_limit"] is False
        if instance == "task-0":
            raise httpx.ReadError("network failure")
        return httpx.Response(
            200, json=response(instance, status="TimeLimitExceeded" if instance == "task-1" else "Submitted")
        )

    transport = httpx.MockTransport(handler)
    result = asyncio.run(runner.evaluate(args, transport=transport))
    assert result == dict(
        completed=500,
        total=500,
        solved=498,
        score=498 / 500,
        errors=2,
        missing=0,
        verifier_correct=498,
        verifier_accuracy=498 / 500,
        length_truncated_verifier_correct=0,
        verifier_test_timeouts=0,
    )
    assert len(attempts) == len(set(attempts)) == 500
    assert asyncio.run(runner.evaluate(args, transport=transport)) == result
    assert len(attempts) == 500
    # Simulate a lost final result after a durable start. Resume counts zero.
    rows = (args.output / "results.jsonl").read_text().splitlines()
    (args.output / "results.jsonl").write_text("\n".join(rows[:-1]) + "\n")
    result = asyncio.run(runner.evaluate(args, transport=transport))
    assert result["solved"] == 497 and result["errors"] == 3 and len(attempts) == 500
    args.budget = "rl"
    with pytest.raises(ValueError, match="configuration differs"):
        asyncio.run(runner.evaluate(args, transport=transport))


@pytest.mark.parametrize("concurrency", [0, -1, 129, True, 1.5])
def test_invalid_concurrency_fails_before_dispatch(concurrency):
    args = SimpleNamespace(budget="large", concurrency=concurrency)
    with pytest.raises(ValueError, match="invalid budget/concurrency"):
        asyncio.run(runner.evaluate(args))


def test_bridge_budgets_and_synthetic_baseline(monkeypatch, tmp_path):
    load_bridge(monkeypatch)
    name = "adaptive_branching.src.swe.verified_harbor_agent"
    monkeypatch.delitem(sys.modules, name, raising=False)
    module = importlib.import_module(name)
    monkeypatch.setitem(sys.modules, name, module)
    extra = {"OPENAI_API_BASE": "http://model/v1"}
    large = module.VerifiedLargeAgent(tmp_path / "logs", "openai/base", max_seq_len=262144, extra_env=extra)
    assert large.config == runner.BUDGETS["large"]
    small = module.VerifiedSmallAgent(tmp_path / "logs", "openai/base", extra_env=extra)
    assert small.config == runner.BUDGETS["rl"]
    with pytest.raises(ValueError):
        module.VerifiedLargeAgent(tmp_path, "openai/base", extra_env=extra)
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    (repo / "source.py").write_text("before\n")
    (repo / ".gitignore").write_text("ignored.py\n")
    (repo / "ignored.py").write_text("tracked despite gitignore\n")
    subprocess.run(["git", "-C", str(repo), "add", "."], check=True)
    subprocess.run(["git", "-C", str(repo), "add", "-f", "ignored.py"], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@invalid",
            "commit",
            "-qm",
            "old history",
        ],
        check=True,
    )
    subprocess.run(["git", "-C", str(repo), "update-ref", "refs/harbor/image-baseline", "HEAD"], check=True)
    script = module.PREPARE_BASELINE.replace("cd /testbed", "cd " + shlex.quote(str(repo)))
    script = script.replace("command -v timeout", "true")  # macOS test host lacks GNU timeout.
    script = script.replace("/tmp/verified-baseline-files", str(tmp_path / "baseline-files"))
    subprocess.run(["bash", "-c", script], check=True, capture_output=True)
    assert (repo / "source.py").read_text() == "before\n"
    assert "ignored.py" in subprocess.check_output(["git", "-C", str(repo), "ls-files"], text=True).splitlines()
    assert subprocess.check_output(["git", "-C", str(repo), "log", "--format=%s"], text=True).strip() == (
        "Evaluation baseline"
    )
    (repo / "source.py").write_text("after\n")
    diff = subprocess.check_output(["git", "-C", str(repo), "diff", "refs/harbor/image-baseline"], text=True)
    assert "-before" in diff and "+after" in diff
    with pytest.raises(TypeError):
        asyncio.run(small.setup(object()))


@pytest.mark.parametrize("files", [1, 128])
def test_baseline_disables_background_git_even_with_aggressive_global_config(monkeypatch, tmp_path, files):
    load_bridge(monkeypatch)
    name = "adaptive_branching.src.swe.verified_harbor_agent"
    monkeypatch.delitem(sys.modules, name, raising=False)
    module = importlib.import_module(name)
    monkeypatch.setitem(sys.modules, name, module)
    repo = tmp_path / "repo"
    repo.mkdir()
    global_config = tmp_path / "gitconfig"
    global_config.write_text("[gc]\n\tauto = 1\n\tautoDetach = true\n[maintenance]\n\tauto = true\n")
    env = dict(os.environ, GIT_CONFIG_GLOBAL=str(global_config), GIT_CONFIG_NOSYSTEM="1")
    subprocess.run(["git", "init", "-q", str(repo)], env=env, check=True)
    for i in range(files):
        (repo / f"source-{i}.py").write_text(f"before-{i}\n")
    subprocess.run(["git", "-C", str(repo), "add", "."], env=env, check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@invalid",
            "-c",
            "gc.auto=0",
            "-c",
            "maintenance.auto=false",
            "commit",
            "-qm",
            "original",
        ],
        env=env,
        check=True,
        capture_output=True,
    )
    subprocess.run(["git", "-C", str(repo), "update-ref", "refs/harbor/image-baseline", "HEAD"], env=env, check=True)
    before = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD^{tree}"], env=env)
    script = module.PREPARE_BASELINE.replace("cd /testbed", "cd " + shlex.quote(str(repo)))
    script = script.replace("command -v timeout", "true").replace(
        "/tmp/verified-baseline-files", str(tmp_path / "files")
    )
    trace = tmp_path / "trace.jsonl"
    subprocess.run(["bash", "-c", script], env=dict(env, GIT_TRACE2_EVENT=str(trace)), check=True, capture_output=True)
    events = [json.loads(line) for line in trace.read_text().splitlines()]
    assert events and not any(
        event.get("event") == "child_start" and any(arg in {"gc", "maintenance"} for arg in event.get("argv", []))
        for event in events
    ), events
    assert subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD^{tree}"], env=env) == before
    saved = tmp_path / "saved-git"
    (repo / ".git").rename(saved)
    assert not (repo / ".git").exists()
    saved.rename(repo / ".git")
    (repo / "source-0.py").write_text("after\n")
    diff = subprocess.check_output(["git", "-C", str(repo), "diff", "refs/harbor/image-baseline"], env=env, text=True)
    assert "-before-0" in diff and "+after" in diff


def test_official_timeout_requires_actual_trial_evidence(tmp_path):
    trial = tmp_path / "trial"
    verifier = trial / "verifier"
    verifier.mkdir(parents=True)
    body = dict(exit_status="TurnLimit", trial_dir=str(trial))
    assert runner.verifier_timeout_failure(body, "task-0") is None
    (verifier / "predictions.json").write_text(json.dumps([dict(instance_id="task-0", model_patch="patch")]))
    stdout = verifier / "test-stdout.txt"
    stdout.write_text("task-0: Test timed out after 1800 seconds.")
    evidence = runner.verifier_timeout_failure(body, "task-0")
    assert evidence["score"] == 0 and evidence["timeout_seconds"] == 1800
    assert evidence["evidence_sha256"] and evidence["patch_sha256"]
    assert runner.verifier_timeout_failure({**body, "exit_status": "AgentInfrastructureFailure"}, "task-0") is None
    stdout.write_text("task-0: Test timed out after 900 seconds.")
    assert runner.verifier_timeout_failure(body, "task-0") is None
    stdout.write_text("task-0: Test timed out after 1800 seconds.")
    for predictions in ([], [{}], [dict(instance_id="wrong", model_patch="")], [1], [dict(instance_id="task-0")]):
        (verifier / "predictions.json").write_text(json.dumps(predictions))
        with pytest.raises(ValueError, match="identity mismatch"):
            runner.verifier_timeout_failure(body, "task-0")
    for value in ("", None):
        with pytest.raises(ValueError):
            runner.verifier_timeout_failure(body, value)
    with pytest.raises(ValueError, match="invalid verifier trial"):
        runner.verifier_timeout_failure({**body, "trial_dir": str(tmp_path / "missing")}, "task-0")
