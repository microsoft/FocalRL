"""Lite registration, dataset identity and one-attempt journal regression tests."""

import asyncio
import json
from types import SimpleNamespace

import httpx
import pytest

from adaptive_branching.tests.swe.test_lightning_verified_on_cpu import response
from adaptive_branching.tools.swe import run_lightning_verified as runner


def lite_response(instance="task-0", **kwargs):
    result = response(instance, **kwargs)
    result["eval_report"]["dataset"] = "SWE-bench_Lite"
    result["eval_report"]["official_harness_version"] = "5.0.2"
    return result


def lite_args(tmp_path, *, count=300, metadata=None):
    prompts = tmp_path / "prompts.jsonl"
    prompts.write_text(
        "".join(
            json.dumps(
                {"prompt": "fix", "metadata": {**runner.LITE_IDENTITY, **(metadata or {}), "instance_id": f"task-{i}"}}
            )
            + "\n"
            for i in range(count)
        )
    )
    return SimpleNamespace(
        benchmark="lite",
        prompts=prompts,
        output=tmp_path / "results",
        budget="rl",
        model="test",
        harbor="http://harbor",
        session="http://session",
        concurrency=64,
        timeout=20,
        keep_going=True,
    )


@pytest.mark.parametrize("status", sorted(runner.NORMAL_ENDINGS))
@pytest.mark.parametrize("reward", [0, 1])
def test_lite_uses_verified_length_rule(status, reward):
    assert runner.score_response(lite_response(status=status, reward=reward), "task-0", "rl", benchmark="lite") == (
        0 if status == "LengthTruncated" else reward
    )


@pytest.mark.parametrize(
    "change",
    [
        {"dataset": None},
        {"dataset": "SWE-bench_Verified"},
        {"official_harness_version": "4.0.3"},
        {"instance_id": "wrong"},
        {"official_report": {"task-0": {"resolved": True, "infra_failure": True}}},
        {"official_report": {"task-0": None}},
    ],
)
def test_lite_rejects_wrong_report_identity_and_infra(change):
    result = lite_response()
    result["eval_report"].update(change)
    with pytest.raises(ValueError):
        runner.score_response(result, "task-0", "rl", benchmark="lite")


def test_lite_accepts_official_empty_patch_report():
    result = lite_response(reward=0)
    result["eval_report"]["official_report"] = {
        "schema_version": 2,
        "total_instances": 1,
        "submitted_instances": 1,
        "submitted_ids": ["task-0"],
        "empty_patch_ids": ["task-0"],
        "resolved_ids": [],
        "error_ids": [],
        "incomplete_ids": [],
    }
    assert runner.score_response(result, "task-0", "rl", benchmark="lite") == 0


@pytest.mark.parametrize("count", [0, 1, 299, 301])
def test_lite_rejects_missing_extra_tasks(tmp_path, count):
    with pytest.raises(ValueError):
        asyncio.run(runner.evaluate(lite_args(tmp_path, count=count)))
    assert not (tmp_path / "results").exists()


@pytest.mark.parametrize(
    "metadata",
    [
        {"dataset": "SWE-bench_Verified"},
        {"dataset": None},
        {"dataset_repo": "wrong"},
        {"dataset_split": "train"},
    ],
)
def test_lite_rejects_wrong_prompt_identity_before_dispatch(tmp_path, metadata):
    with pytest.raises(ValueError, match="correctly labeled"):
        asyncio.run(runner.evaluate(lite_args(tmp_path, metadata=metadata)))
    assert not (tmp_path / "results").exists()


def test_lite_rejects_duplicate_tasks(tmp_path):
    args = lite_args(tmp_path)
    lines = args.prompts.read_text().splitlines()
    lines[-1] = lines[0]
    args.prompts.write_text("\n".join(lines) + "\n")
    with pytest.raises(ValueError, match="duplicate"):
        asyncio.run(runner.evaluate(args))


def test_lite_all_300_once_resume_preserves_budget_and_errors(tmp_path):
    args = lite_args(tmp_path)
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
        assert body["max_turns"] == 100 and body["max_seq_len"] == 81920
        assert body["sampling_params"]["max_tokens"] == 12288
        assert body["sampling_params"]["chat_template_kwargs"] == {"enable_thinking": True}
        if instance == "task-0":
            raise httpx.ReadError("infrastructure failure")
        status = "LengthTruncated" if instance == "task-1" else "Submitted"
        return httpx.Response(200, json=lite_response(instance, status=status))

    transport = httpx.MockTransport(handler)
    result = asyncio.run(runner.evaluate(args, transport=transport))
    assert result == dict(
        completed=300,
        total=300,
        solved=298,
        score=298 / 300,
        errors=1,
        missing=0,
        verifier_correct=299,
        verifier_accuracy=299 / 300,
        length_truncated_verifier_correct=1,
        verifier_test_timeouts=0,
    )
    assert len(attempts) == len(set(attempts)) == 300
    config = json.loads((args.output / "config.json").read_text())
    assert config["denominator"] == 300 and config["length_truncated_score"] == 0
    assert asyncio.run(runner.evaluate(args, transport=transport)) == result
    assert len(attempts) == 300
