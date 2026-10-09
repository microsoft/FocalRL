#!/usr/bin/env python3
"""Run resumable R2E or SWE-bench full-outcome evaluations through Harbor."""

import argparse
import asyncio
import json
import math
import os
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx

# Model budget exhaustion is a completed attempt, not an infrastructure retry.
# Keep the verifier's outcome even when the final generation was truncated.
_USABLE_EXIT_STATUSES = {"Submitted", "LimitsExceeded", "RepeatedFormatError", "LengthTruncated"}


@dataclass(frozen=True)
class BenchmarkJob:
    job_id: str
    task_index: int
    sample_index: int
    seed: int
    metadata: dict[str, Any]


@dataclass(frozen=True)
class ModelSession:
    server_url: str
    session_id: str
    server_instance_id: str

    @property
    def model_base_url(self) -> str:
        return f"{self.server_url}/sessions/{self.session_id}/v1"

    @property
    def heartbeat_target(self) -> str:
        return urlparse(self.server_url).netloc


def _positive_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return value


def load_samples(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    samples: list[dict[str, Any]] = []
    instance_ids: set[str] = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            sample = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON at {path}:{line_number}") from exc
        if not isinstance(sample, dict):
            raise TypeError(f"{path}:{line_number} must contain an object")
        prompt = sample.get("prompt")
        metadata = sample.get("metadata")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(f"{path}:{line_number} prompt must be a non-empty string")
        if not isinstance(metadata, dict):
            raise TypeError(f"{path}:{line_number} metadata must be an object")
        instance_id = metadata.get("instance_id")
        if not isinstance(instance_id, str) or not instance_id.strip():
            raise ValueError(f"{path}:{line_number} metadata.instance_id must be a non-empty string")
        if instance_id in instance_ids:
            raise ValueError(f"duplicate instance_id {instance_id!r} at {path}:{line_number}")
        instance_ids.add(instance_id)
        samples.append(sample)
    if not samples:
        raise ValueError(f"prompt file contains no samples: {path}")
    return samples


def build_jobs(samples: list[dict[str, Any]], *, repeats: int, base_seed: int) -> list[BenchmarkJob]:
    _positive_int(repeats, "repeats")
    if isinstance(base_seed, bool) or not isinstance(base_seed, int) or base_seed < 0:
        raise ValueError(f"base_seed must be a non-negative integer, got {base_seed!r}")
    if not samples:
        raise ValueError("samples must not be empty")
    jobs: list[BenchmarkJob] = []
    for task_index, sample in enumerate(samples):
        metadata = sample["metadata"]
        instance_id = metadata["instance_id"]
        for sample_index in range(repeats):
            jobs.append(
                BenchmarkJob(
                    job_id=f"{instance_id}::sample-{sample_index}",
                    task_index=task_index,
                    sample_index=sample_index,
                    seed=base_seed + task_index * repeats + sample_index,
                    metadata=dict(metadata),
                )
            )
    return jobs


def load_completed(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    if not path.is_file():
        raise ValueError(f"output path is not a file: {path}")
    completed: dict[str, dict[str, Any]] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON at {path}:{line_number}") from exc
        if not isinstance(record, dict):
            raise TypeError(f"{path}:{line_number} must contain an object")
        job_id = record.get("job_id")
        response = record.get("response")
        if not isinstance(job_id, str) or not job_id:
            raise ValueError(f"{path}:{line_number} has invalid job_id")
        if job_id in completed:
            raise ValueError(f"duplicate completed job_id {job_id!r} at {path}:{line_number}")
        try:
            validate_usable_response(response)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{path}:{line_number} is not a valid completed result") from exc
        if not isinstance(record.get("request_config"), dict):
            raise TypeError(f"{path}:{line_number} request_config must be an object")
        if not isinstance(record.get("request"), dict):
            raise TypeError(f"{path}:{line_number} request must be an object")
        completed[job_id] = record
    return completed


def build_request_config(
    job: BenchmarkJob,
    *,
    model: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    max_seq_len: int,
    max_turns: int = 250,
    context_reserve_tokens: int = 32768,
) -> dict[str, Any]:
    if not model.strip():
        raise ValueError("model must be non-empty")
    if not math.isfinite(temperature) or temperature < 0:
        raise ValueError("temperature must be finite and non-negative")
    if not math.isfinite(top_p) or not 0 < top_p <= 1:
        raise ValueError("top_p must be in (0, 1]")
    _positive_int(max_tokens, "max_tokens")
    _positive_int(max_seq_len, "max_seq_len")
    _positive_int(max_turns, "max_turns")
    _positive_int(context_reserve_tokens, "context_reserve_tokens")
    if context_reserve_tokens >= max_seq_len:
        raise ValueError("context_reserve_tokens must be smaller than max_seq_len")
    return {
        **job.metadata,
        "instance_id": job.metadata["instance_id"],
        "model": f"openai/{model}",
        "api_key": "dummy",
        "agent_name": "mini-swe-agent",
        "max_seq_len": max_seq_len,
        "max_turns": max_turns,
        "context_reserve_tokens": context_reserve_tokens,
        "sampling_params": {
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "seed": job.seed,
        },
    }


async def create_model_session(client: httpx.AsyncClient, server_url: str) -> ModelSession:
    server_url = server_url.rstrip("/")
    parsed = urlparse(server_url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("session_server_url must be an absolute HTTP(S) URL")

    health_response = await client.get(f"{server_url}/health")
    health_response.raise_for_status()
    server_instance_id = health_response.json().get("session_server_instance_id")
    if not isinstance(server_instance_id, str) or not server_instance_id:
        raise ValueError("session server health has no session_server_instance_id")

    create_response = await client.post(f"{server_url}/sessions")
    create_response.raise_for_status()
    session_id = create_response.json().get("session_id")
    if not isinstance(session_id, str) or not session_id:
        raise ValueError("session server returned an invalid session_id")
    return ModelSession(server_url, session_id, server_instance_id)


def bind_model_session(request_config: dict[str, Any], session: ModelSession) -> dict[str, Any]:
    return {
        **request_config,
        "base_url": session.model_base_url,
        "session_server_id": session.heartbeat_target,
        "session_server_instance_id": session.server_instance_id,
    }


async def collect_model_session(client: httpx.AsyncClient, session: ModelSession) -> dict[str, Any]:
    session_url = f"{session.server_url}/sessions/{session.session_id}"
    try:
        response = await client.get(session_url)
        response.raise_for_status()
        trace = response.json()
        if not isinstance(trace, dict):
            raise TypeError("session trace must be an object")
        records = trace.get("records")
        if not isinstance(records, list) or not records:
            raise ValueError("session trace contains no model records")
        if not isinstance(trace.get("metadata"), dict):
            raise TypeError("session trace metadata must be an object")
        return trace
    finally:
        await delete_model_session(client, session)


async def delete_model_session(client: httpx.AsyncClient, session: ModelSession) -> None:
    session_url = f"{session.server_url}/sessions/{session.session_id}"
    for attempt in range(3):
        try:
            response = await client.delete(session_url)
            response.raise_for_status()
            return
        except httpx.HTTPError:
            if attempt == 2:
                raise
            await asyncio.sleep(2**attempt)


def validate_usable_response(body: Any) -> dict[str, Any]:
    if not isinstance(body, dict):
        raise TypeError(f"Harbor response must be an object, got {type(body).__name__}")
    if body.get("exit_status") not in _USABLE_EXIT_STATUSES:
        raise ValueError(f"Harbor trial is not usable: {body.get('exit_status')!r}")
    for key in ("trial_id", "trial_dir"):
        if not isinstance(body.get(key), str) or not body[key].strip():
            raise ValueError(f"completed Harbor response has invalid {key}: {body.get(key)!r}")
    reward = body.get("reward")
    if isinstance(reward, bool) or not isinstance(reward, (int, float)) or not math.isfinite(float(reward)):
        raise ValueError(f"completed Harbor response has invalid reward: {reward!r}")
    if reward not in (0.0, 1.0):
        raise ValueError(f"completed Harbor full-outcome reward must be binary, got {reward!r}")
    for key in ("agent_metrics", "eval_report"):
        if not isinstance(body.get(key), dict):
            raise TypeError(f"completed Harbor response has invalid {key}: {body.get(key)!r}")
    report = body["eval_report"]
    report_reward = report.get("reward")
    if isinstance(report_reward, bool) or not isinstance(report_reward, (int, float)):
        raise TypeError(f"completed Harbor response has invalid eval_report.reward: {report_reward!r}")
    if float(report_reward) != float(reward):
        raise ValueError(f"Harbor reward mismatch: response={reward!r}, eval_report={report_reward!r}")
    return body


class JsonlRecorder:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = asyncio.Lock()

    async def append(self, record: dict[str, Any]) -> None:
        if not isinstance(record, dict) or not record:
            raise ValueError("record must be a non-empty object")
        encoded = (json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n").encode()
        async with self._lock:
            fd = os.open(self.path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o644)
            try:
                written = os.write(fd, encoded)
                if written != len(encoded):
                    raise OSError(f"short write to {self.path}: {written}/{len(encoded)} bytes")
                os.fsync(fd)
            finally:
                os.close(fd)


class Progress:
    def __init__(self, *, total: int, progress_every: int) -> None:
        self.total = _positive_int(total, "total")
        self.progress_every = _positive_int(progress_every, "progress_every")
        self.started = time.monotonic()
        self.finished = 0
        self.submitted = 0
        self.failed = 0
        self.attempts = 0
        self.active = 0
        self.max_active = 0
        self.rewards: list[float] = []
        self.exit_statuses: Counter[str] = Counter()
        self._lock = asyncio.Lock()

    async def attempt_started(self) -> None:
        async with self._lock:
            self.attempts += 1
            self.active += 1
            self.max_active = max(self.max_active, self.active)

    async def attempt_finished(self) -> None:
        async with self._lock:
            self.active -= 1
            if self.active < 0:
                raise RuntimeError("active request count became negative")

    async def job_finished(self, response: dict[str, Any] | None) -> None:
        async with self._lock:
            self.finished += 1
            if response is None:
                self.failed += 1
                self.exit_statuses["RunnerFailure"] += 1
            else:
                self.submitted += 1
                self.rewards.append(float(response["reward"]))
                self.exit_statuses[str(response["exit_status"])] += 1
            if self.finished % self.progress_every == 0 or self.finished == self.total:
                elapsed = time.monotonic() - self.started
                print(
                    "PROGRESS "
                    + json.dumps(
                        {
                            "active": self.active,
                            "attempts": self.attempts,
                            "elapsed_seconds": round(elapsed, 3),
                            "failed": self.failed,
                            "finished": self.finished,
                            "max_active": self.max_active,
                            "submitted": self.submitted,
                            "total": self.total,
                            "trials_per_minute": round(60 * self.finished / elapsed, 3) if elapsed else 0.0,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )

    def summary(self) -> dict[str, Any]:
        elapsed = time.monotonic() - self.started
        return {
            "attempts": self.attempts,
            "elapsed_seconds": round(elapsed, 3),
            "exit_statuses": dict(sorted(self.exit_statuses.items())),
            "failed": self.failed,
            "max_active": self.max_active,
            "mean_reward": sum(self.rewards) / len(self.rewards) if self.rewards else None,
            "submitted": self.submitted,
            "total": self.total,
            "trials_per_minute": round(60 * self.finished / elapsed, 3) if elapsed else 0.0,
        }


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


async def run_benchmark(
    args: argparse.Namespace,
    *,
    transport: httpx.AsyncBaseTransport | None = None,
) -> dict[str, Any]:
    for name in (
        "repeats",
        "concurrency",
        "max_attempts",
        "request_timeout",
        "max_tokens",
        "max_seq_len",
        "max_turns",
        "context_reserve_tokens",
    ):
        _positive_int(getattr(args, name), name)
    if args.context_reserve_tokens >= args.max_seq_len:
        raise ValueError("context_reserve_tokens must be smaller than max_seq_len")
    if args.limit is not None:
        _positive_int(args.limit, "limit")
    if args.output.resolve() == args.error_output.resolve():
        raise ValueError("output and error_output must be different files")
    if not args.server_url.startswith(("http://", "https://")):
        raise ValueError("server_url must be an absolute HTTP(S) URL")
    if not args.session_server_url.startswith(("http://", "https://")):
        raise ValueError("session_server_url must be an absolute HTTP(S) URL")

    samples = load_samples(args.prompt_data)
    if args.limit is not None:
        samples = samples[: args.limit]
    jobs = build_jobs(samples, repeats=args.repeats, base_seed=args.base_seed)
    completed = load_completed(args.output)
    request_configs = {
        job.job_id: build_request_config(
            job,
            model=args.model,
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
            max_seq_len=args.max_seq_len,
            max_turns=args.max_turns,
            context_reserve_tokens=args.context_reserve_tokens,
        )
        for job in jobs
    }
    known_job_ids = set(request_configs)
    unexpected = set(completed) - known_job_ids
    if unexpected:
        raise ValueError(f"output contains jobs outside the requested benchmark: {sorted(unexpected)[:3]}")
    for job_id, record in completed.items():
        if record["request_config"] != request_configs[job_id]:
            raise ValueError(f"completed job {job_id!r} was produced with a different request configuration")
    pending = [job for job in jobs if job.job_id not in completed]
    if not pending:
        return {
            "already_completed": len(completed),
            "failed": 0,
            "submitted": 0,
            "total": 0,
        }

    recorder = JsonlRecorder(args.output)
    error_recorder = JsonlRecorder(args.error_output)
    progress = Progress(total=len(pending), progress_every=args.progress_every)
    semaphore = asyncio.Semaphore(args.concurrency)
    limits = httpx.Limits(
        max_connections=args.concurrency,
        max_keepalive_connections=args.concurrency,
    )
    timeout = httpx.Timeout(float(args.request_timeout))

    async with httpx.AsyncClient(limits=limits, timeout=timeout, transport=transport) as client:

        async def run_job(job: BenchmarkJob) -> bool:
            request_config = request_configs[job.job_id]
            for attempt in range(1, args.max_attempts + 1):
                started_at = _timestamp()
                started = time.monotonic()
                response_body: dict[str, Any] | None = None
                request_payload: dict[str, Any] | None = None
                error: str | None = None
                try:
                    async with semaphore:
                        await progress.attempt_started()
                        try:
                            session = await create_model_session(client, args.session_server_url)
                            request_payload = bind_model_session(request_config, session)
                            try:
                                response = await client.post(
                                    f"{args.server_url.rstrip('/')}/run", json=request_payload
                                )
                                response.raise_for_status()
                                decoded = response.json()
                                if not isinstance(decoded, dict):
                                    raise TypeError(f"Harbor response must be an object, got {type(decoded).__name__}")
                                response_body = decoded
                                validate_usable_response(response_body)
                            finally:
                                await delete_model_session(client, session)
                        finally:
                            await progress.attempt_finished()
                except (httpx.HTTPError, json.JSONDecodeError, TypeError, ValueError) as exc:
                    error = f"{type(exc).__name__}: {exc}"
                elapsed = time.monotonic() - started
                base_record = {
                    "attempt": attempt,
                    "elapsed_seconds": round(elapsed, 3),
                    "finished_at": _timestamp(),
                    "instance_id": job.metadata["instance_id"],
                    "job_id": job.job_id,
                    "metadata": job.metadata,
                    "request": request_payload,
                    "request_config": request_config,
                    "sample_index": job.sample_index,
                    "seed": job.seed,
                    "started_at": started_at,
                }
                if error is None:
                    assert response_body is not None
                    await recorder.append({**base_record, "response": response_body})
                    await progress.job_finished(response_body)
                    return True
                await error_recorder.append({**base_record, "error": error, "response": response_body})
                if attempt < args.max_attempts:
                    await asyncio.sleep(min(30, 2 ** (attempt - 1)))
            await progress.job_finished(None)
            return False

        succeeded = await asyncio.gather(*(run_job(job) for job in pending))

    summary = {
        **progress.summary(),
        "already_completed": len(completed),
        "requested_jobs": len(jobs),
        "successful_this_run": sum(succeeded),
    }
    print("SUMMARY " + json.dumps(summary, sort_keys=True), flush=True)
    if not all(succeeded):
        raise RuntimeError(f"{len(succeeded) - sum(succeeded)} benchmark jobs exhausted all attempts")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt-data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--error-output", type=Path, required=True)
    parser.add_argument("--server-url", default="http://127.0.0.1:11200")
    parser.add_argument(
        "--session-server-url",
        default=os.environ.get("MILES_SESSION_SERVER_URL", "http://127.0.0.1:30001"),
    )
    parser.add_argument("--model", default="Qwen3.5-4B")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--repeats", type=int, default=4)
    parser.add_argument("--concurrency", type=int, default=128)
    parser.add_argument("--max-attempts", type=int, default=2)
    parser.add_argument("--request-timeout", type=int, default=12000)
    parser.add_argument("--base-seed", type=int, default=20260830)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--max-seq-len", type=int, default=163840)
    parser.add_argument("--max-turns", type=int, default=250)
    parser.add_argument("--context-reserve-tokens", type=int, default=32768)
    parser.add_argument("--progress-every", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    asyncio.run(run_benchmark(parse_args()))


if __name__ == "__main__":
    main()
