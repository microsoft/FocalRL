import argparse
import asyncio
import json
import sys

import httpx
import pytest

from adaptive_branching.tools.swe import run_harbor_benchmark


def _sample(instance_id="r2e-pandas-deadbeef"):
    return {
        "prompt": "Fix the bug.",
        "metadata": {
            "instance_id": instance_id,
            "repo_name": "pandas",
            "docker_image": "example/image:tag",
        },
    }


def _eval_report(reward=0.0):
    return {
        "reward": reward,
        "actual": {"test_fix": "FAILED" if reward == 0.0 else "PASSED"},
        "expected": {"test_fix": "PASSED"},
    }


def _session_trace(session_id="session-1"):
    return {
        "session_id": session_id,
        "records": [{"request": {"input_ids": [1]}, "response": {"choices": []}}],
        "metadata": {"accumulated_token_ids": [1]},
    }


def test_load_samples_and_build_jobs(tmp_path):
    path = tmp_path / "prompts.jsonl"
    path.write_text(json.dumps(_sample()) + "\n", encoding="utf-8")

    samples = run_harbor_benchmark.load_samples(path)
    jobs = run_harbor_benchmark.build_jobs(samples, repeats=2, base_seed=100)

    assert [job.job_id for job in jobs] == [
        "r2e-pandas-deadbeef::sample-0",
        "r2e-pandas-deadbeef::sample-1",
    ]
    assert [job.seed for job in jobs] == [100, 101]


def test_cli_defaults_use_final_horizon_and_timeout(monkeypatch, tmp_path):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_harbor_benchmark.py",
            "--prompt-data",
            str(tmp_path / "prompts.jsonl"),
            "--output",
            str(tmp_path / "results.jsonl"),
            "--error-output",
            str(tmp_path / "errors.jsonl"),
        ],
    )

    args = run_harbor_benchmark.parse_args()

    assert args.max_seq_len == 163840
    assert args.max_turns == 250
    assert args.context_reserve_tokens == 32768
    assert args.max_tokens == 8192
    assert args.request_timeout == 12000
    assert args.server_url == "http://127.0.0.1:11200"
    assert args.session_server_url == "http://127.0.0.1:30001"
    assert args.concurrency == 128


def test_load_samples_rejects_bad_and_duplicate_rows(tmp_path):
    path = tmp_path / "prompts.jsonl"
    path.write_text("not-json\n", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid JSON"):
        run_harbor_benchmark.load_samples(path)

    path.write_text("\n".join(json.dumps(_sample()) for _ in range(2)), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate instance_id"):
        run_harbor_benchmark.load_samples(path)


def test_build_request_config_and_bind_session():
    job = run_harbor_benchmark.build_jobs([_sample()], repeats=1, base_seed=7)[0]
    config = run_harbor_benchmark.build_request_config(
        job,
        model="Qwen3.5-4B",
        temperature=0.8,
        top_p=0.95,
        max_tokens=8192,
        max_seq_len=65536,
    )
    session = run_harbor_benchmark.ModelSession("http://127.0.0.1:31000", "session-1", "server-1")
    payload = run_harbor_benchmark.bind_model_session(config, session)

    assert payload["base_url"] == "http://127.0.0.1:31000/sessions/session-1/v1"
    assert payload["session_server_id"] == "127.0.0.1:31000"
    assert payload["session_server_instance_id"] == "server-1"
    assert payload["model"] == "openai/Qwen3.5-4B"
    assert payload["max_turns"] == 250
    assert payload["context_reserve_tokens"] == 32768
    assert payload["sampling_params"] == {
        "temperature": 0.8,
        "top_p": 0.95,
        "max_tokens": 8192,
        "seed": 7,
    }
    response = {
        "exit_status": "Submitted",
        "trial_id": "trial-1",
        "trial_dir": "/trials/trial-1",
        "reward": 0.0,
        "agent_metrics": {},
        "eval_report": _eval_report(),
    }
    assert run_harbor_benchmark.validate_usable_response(response) is response

    response["exit_status"] = "LimitsExceeded"
    assert run_harbor_benchmark.validate_usable_response(response) is response

    response["exit_status"] = "RepeatedFormatError"
    assert run_harbor_benchmark.validate_usable_response(response) is response


def test_collect_model_session_deletes_empty_session():
    requests = []

    def handler(request):
        requests.append((request.method, request.url.path))
        if request.method == "GET":
            return httpx.Response(200, json={"records": [], "metadata": {}})
        return httpx.Response(204)

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            session = run_harbor_benchmark.ModelSession("http://session.test", "empty", "server-1")
            with pytest.raises(ValueError, match="no model records"):
                await run_harbor_benchmark.collect_model_session(client, session)

    asyncio.run(run())
    assert requests == [
        ("GET", "/sessions/empty"),
        ("DELETE", "/sessions/empty"),
    ]


def test_delete_model_session_succeeds_without_retry(monkeypatch):
    attempts = 0
    sleeps = []

    def handler(request):
        nonlocal attempts
        assert request.method == "DELETE"
        assert request.url.path == "/sessions/session-1"
        attempts += 1
        return httpx.Response(204)

    async def fake_sleep(delay):
        sleeps.append(delay)

    async def run():
        monkeypatch.setattr(run_harbor_benchmark.asyncio, "sleep", fake_sleep)
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            session = run_harbor_benchmark.ModelSession("http://session.test", "session-1", "server-1")
            await run_harbor_benchmark.delete_model_session(client, session)

    asyncio.run(run())
    assert attempts == 1
    assert sleeps == []


def test_delete_model_session_retries_twice_then_succeeds(monkeypatch):
    attempts = 0
    sleeps = []

    def handler(request):
        nonlocal attempts
        attempts += 1
        return httpx.Response(502 if attempts < 3 else 204)

    async def fake_sleep(delay):
        sleeps.append(delay)

    async def run():
        monkeypatch.setattr(run_harbor_benchmark.asyncio, "sleep", fake_sleep)
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            session = run_harbor_benchmark.ModelSession("http://session.test", "session-1", "server-1")
            await run_harbor_benchmark.delete_model_session(client, session)

    asyncio.run(run())
    assert attempts == 3
    assert sleeps == [1, 2]


def test_delete_model_session_raises_after_three_failures(monkeypatch):
    attempts = 0
    sleeps = []

    def handler(request):
        nonlocal attempts
        attempts += 1
        return httpx.Response(502)

    async def fake_sleep(delay):
        sleeps.append(delay)

    async def run():
        monkeypatch.setattr(run_harbor_benchmark.asyncio, "sleep", fake_sleep)
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            session = run_harbor_benchmark.ModelSession("http://session.test", "session-1", "server-1")
            with pytest.raises(httpx.HTTPStatusError):
                await run_harbor_benchmark.delete_model_session(client, session)

    asyncio.run(run())
    assert attempts == 3
    assert sleeps == [1, 2]


@pytest.mark.parametrize(
    "response",
    [
        {},
        {"exit_status": "AgentError"},
        {
            "exit_status": "Submitted",
            "trial_id": "trial-1",
            "trial_dir": "",
            "reward": 0.0,
            "agent_metrics": {},
            "eval_report": {},
        },
        {
            "exit_status": "Submitted",
            "trial_id": "trial-1",
            "trial_dir": "/trial",
            "reward": True,
            "agent_metrics": {},
            "eval_report": {},
        },
        {
            "exit_status": "Submitted",
            "trial_id": "trial-1",
            "trial_dir": "/trial",
            "reward": 0.0,
            "agent_metrics": {},
            "eval_report": {},
        },
        {
            "exit_status": "Submitted",
            "trial_id": "trial-1",
            "trial_dir": "/trial",
            "reward": 0.0,
            "agent_metrics": {},
            "eval_report": {**_eval_report(), "reward": 1.0},
        },
    ],
)
def test_validate_submitted_response_rejects_incomplete_response(response):
    with pytest.raises((TypeError, ValueError)):
        run_harbor_benchmark.validate_usable_response(response)


@pytest.mark.parametrize(
    "report",
    [
        {"reward": 0.0, "actual": {}, "expected": {"test_b": "PASSED"}},
        {
            "reward": 0.0,
            "instance_id": "django__django-1",
            "official_harness_version": "4.0.3",
            "official_report": {"resolved": False},
        },
    ],
)
def test_validate_submitted_response_accepts_dataset_specific_report(report):
    response = {
        "exit_status": "Submitted",
        "trial_id": "trial-1",
        "trial_dir": "/trial",
        "reward": 0.0,
        "agent_metrics": {},
        "eval_report": report,
    }

    assert run_harbor_benchmark.validate_usable_response(response) is response


def test_load_completed_rejects_duplicate_or_failed_records(tmp_path):
    path = tmp_path / "results.jsonl"
    response = {
        "exit_status": "Submitted",
        "trial_id": "trial-1",
        "trial_dir": "/trial",
        "reward": 0.0,
        "agent_metrics": {},
        "eval_report": _eval_report(),
    }
    record = {
        "job_id": "job-1",
        "request": {"model": "openai/Qwen3.5-4B"},
        "request_config": {"model": "openai/Qwen3.5-4B"},
        "response": response,
        "session": _session_trace(),
    }
    path.write_text(json.dumps(record) + "\n" + json.dumps(record) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate completed job_id"):
        run_harbor_benchmark.load_completed(path)

    record["response"]["exit_status"] = "AgentError"
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="not a valid completed result"):
        run_harbor_benchmark.load_completed(path)


def test_jsonl_recorder_appends_complete_lines(tmp_path):
    path = tmp_path / "records.jsonl"
    recorder = run_harbor_benchmark.JsonlRecorder(path)

    asyncio.run(recorder.append({"job_id": "a"}))
    asyncio.run(recorder.append({"job_id": "b"}))

    assert [json.loads(line)["job_id"] for line in path.read_text().splitlines()] == ["a", "b"]


def test_run_benchmark_noops_when_all_jobs_are_complete(tmp_path):
    prompts = tmp_path / "prompts.jsonl"
    output = tmp_path / "results.jsonl"
    prompts.write_text(json.dumps(_sample()) + "\n", encoding="utf-8")
    job = run_harbor_benchmark.build_jobs([_sample()], repeats=1, base_seed=20260830)[0]
    request_config = run_harbor_benchmark.build_request_config(
        job,
        model="Qwen3.5-4B",
        temperature=0.8,
        top_p=0.95,
        max_tokens=8192,
        max_seq_len=65536,
    )
    output.write_text(
        json.dumps(
            {
                "job_id": "r2e-pandas-deadbeef::sample-0",
                "request": {
                    **request_config,
                    "base_url": "http://127.0.0.1:31000/sessions/session-1/v1",
                    "session_server_id": "127.0.0.1:31000",
                    "session_server_instance_id": "server-1",
                },
                "request_config": request_config,
                "response": {
                    "exit_status": "Submitted",
                    "trial_id": "trial-1",
                    "trial_dir": "/trial",
                    "reward": 0.0,
                    "agent_metrics": {},
                    "eval_report": _eval_report(),
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    args = argparse.Namespace(
        prompt_data=prompts,
        output=output,
        error_output=tmp_path / "errors.jsonl",
        server_url="http://127.0.0.1:11200",
        session_server_url="http://127.0.0.1:31000",
        model="Qwen3.5-4B",
        limit=1,
        repeats=1,
        concurrency=1,
        max_attempts=1,
        request_timeout=1,
        base_seed=20260830,
        temperature=0.8,
        top_p=0.95,
        max_tokens=8192,
        max_seq_len=65536,
        max_turns=250,
        context_reserve_tokens=32768,
        progress_every=1,
    )

    summary = asyncio.run(run_harbor_benchmark.run_benchmark(args))

    assert summary == {"already_completed": 1, "failed": 0, "submitted": 0, "total": 0}


def _args(tmp_path, *, repeats=2, max_attempts=1):
    prompts = tmp_path / "prompts.jsonl"
    prompts.write_text(json.dumps(_sample()) + "\n", encoding="utf-8")
    return argparse.Namespace(
        prompt_data=prompts,
        output=tmp_path / "results.jsonl",
        error_output=tmp_path / "errors.jsonl",
        server_url="http://harbor.test",
        session_server_url="http://session.test",
        model="Qwen3.5-4B",
        limit=1,
        repeats=repeats,
        concurrency=2,
        max_attempts=max_attempts,
        request_timeout=10,
        base_seed=20260830,
        temperature=0.8,
        top_p=0.95,
        max_tokens=8192,
        max_seq_len=65536,
        max_turns=250,
        context_reserve_tokens=32768,
        progress_every=1,
    )


class _MockServices:
    def __init__(self, harbor_response):
        self.harbor_response = harbor_response
        self.harbor_requests = []
        self.deleted_sessions = []
        self.session_count = 0

    async def __call__(self, request):
        if request.url.host == "session.test":
            if request.url.path == "/health":
                return httpx.Response(200, json={"session_server_instance_id": "server-1"})
            if request.method == "POST" and request.url.path == "/sessions":
                self.session_count += 1
                return httpx.Response(200, json={"session_id": f"session-{self.session_count}"})
            session_id = request.url.path.rsplit("/", 1)[-1]
            if request.method == "DELETE":
                self.deleted_sessions.append(session_id)
                return httpx.Response(204)
        if request.url.host == "harbor.test" and request.url.path == "/run":
            payload = json.loads(request.content)
            self.harbor_requests.append(payload)
            await asyncio.sleep(0.01)
            return httpx.Response(200, json=self.harbor_response(payload, len(self.harbor_requests)))
        raise AssertionError(f"unexpected request: {request.method} {request.url}")


def test_run_benchmark_records_two_submitted_trials(tmp_path):
    def response(payload, _attempt):
        seed = payload["sampling_params"]["seed"]
        return {
            "exit_status": "Submitted",
            "trial_id": f"trial-{seed}",
            "trial_dir": f"/trials/trial-{seed}",
            "reward": float(seed % 2),
            "agent_metrics": {"turns": 1},
            "eval_report": _eval_report(float(seed % 2)),
        }

    args = _args(tmp_path)
    services = _MockServices(response)
    summary = asyncio.run(run_harbor_benchmark.run_benchmark(args, transport=httpx.MockTransport(services)))

    records = [json.loads(line) for line in args.output.read_text().splitlines()]
    assert summary["submitted"] == 2
    assert summary["failed"] == 0
    assert summary["max_active"] == 2
    assert sorted(request["sampling_params"]["seed"] for request in services.harbor_requests) == [
        20260830,
        20260831,
    ]
    assert len(records) == 2
    assert all(record["response"]["exit_status"] == "Submitted" for record in records)
    assert all(record["request"]["model"] == "openai/Qwen3.5-4B" for record in records)
    assert all("/sessions/session-" in record["request"]["base_url"] for record in records)
    assert all("session" not in record for record in records)
    assert sorted(services.deleted_sessions) == ["session-1", "session-2"]
    assert not args.error_output.exists()


def test_run_benchmark_retries_non_submitted_response(tmp_path):
    def response(_payload, attempt):
        if attempt == 1:
            return {"exit_status": "AgentError", "reward": 0.0}
        return {
            "exit_status": "Submitted",
            "trial_id": "trial-ok",
            "trial_dir": "/trials/trial-ok",
            "reward": 0.0,
            "agent_metrics": {},
            "eval_report": _eval_report(),
        }

    args = _args(tmp_path, repeats=1, max_attempts=2)
    services = _MockServices(response)
    summary = asyncio.run(run_harbor_benchmark.run_benchmark(args, transport=httpx.MockTransport(services)))

    errors = [json.loads(line) for line in args.error_output.read_text().splitlines()]
    assert len(services.harbor_requests) == 2
    assert summary["attempts"] == 2
    assert summary["submitted"] == 1
    assert errors[0]["response"]["exit_status"] == "AgentError"
    assert "session" not in errors[0]
    assert sorted(services.deleted_sessions) == ["session-1", "session-2"]


@pytest.mark.parametrize("exit_status", ["LengthTruncated", "LimitsExceeded", "RepeatedFormatError"])
@pytest.mark.parametrize("reward", [0.0, 1.0])
def test_model_limits_keep_verifier_outcome_without_retry_and_resume(tmp_path, exit_status, reward):
    def response(_payload, _attempt):
        return {
            "exit_status": exit_status,
            "trial_id": "trial-limit",
            "trial_dir": "/trials/trial-limit",
            "reward": reward,
            "agent_metrics": {},
            "eval_report": {
                "reward": reward,
                "instance_id": "r2e-pandas-deadbeef",
                "official_harness_version": "4.0.3",
                "official_report": {"resolved": bool(reward)},
            },
        }

    args = _args(tmp_path, repeats=1, max_attempts=2)
    services = _MockServices(response)
    transport = httpx.MockTransport(services)
    summary = asyncio.run(run_harbor_benchmark.run_benchmark(args, transport=transport))

    assert len(services.harbor_requests) == summary["attempts"] == 1
    assert summary["submitted"] == summary["requested_jobs"] == 1
    assert summary["failed"] == 0
    assert summary["mean_reward"] == reward
    assert summary["exit_statuses"] == {exit_status: 1}
    assert not args.error_output.exists()
    assert services.deleted_sessions == ["session-1"]
    records = run_harbor_benchmark.load_completed(args.output)
    assert len(records) == 1
    assert next(iter(records.values()))["response"]["reward"] == reward

    resumed = asyncio.run(run_harbor_benchmark.run_benchmark(args, transport=transport))
    assert resumed["already_completed"] == 1
    assert len(services.harbor_requests) == 1


@pytest.mark.parametrize("reward", [None, True, -1, 0.5, 2, float("nan"), float("inf")])
def test_full_outcome_rejects_nonbinary_reward(reward):
    response = {
        "exit_status": "LengthTruncated",
        "trial_id": "trial-limit",
        "trial_dir": "/trials/trial-limit",
        "reward": reward,
        "agent_metrics": {},
        "eval_report": {"reward": reward},
    }
    with pytest.raises(ValueError, match="reward"):
        run_harbor_benchmark.validate_usable_response(response)
