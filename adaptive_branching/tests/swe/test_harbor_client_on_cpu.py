# Modified for anonymous release: deployment identifiers replaced.
import asyncio
import json
from types import SimpleNamespace

import httpx
import pytest

from adaptive_branching.src.swe import harbor_client, reward
from adaptive_branching.src.swe.tool_metrics import extract_swe_tool_metrics
from miles.rollout.generate_utils.openai_endpoint_utils import compute_samples_from_openai_records
from miles.rollout.session.session_types import SessionRecord
from miles.utils.types import Sample


def _one_tool_metrics(code=0):
    return extract_swe_tool_metrics(
        [
            {"role": "assistant", "extra": {"actions": [{"tool_call_id": "x"}]}},
            {"role": "tool", "tool_call_id": "x", "extra": {"returncode": code}},
        ]
    )


def test_tool_metrics_promotion_clears_stale_retry_data():
    old = harbor_client._promoted_tool_metrics(_one_tool_metrics())
    old.update(harbor_client._promoted_tool_metrics({}))
    assert old["agent_tool_unit_count"] is None
    assert old["agent_swe_tool_returncode_zero_rate"] is None
    assert old["agent_swe_tool_metrics_available"] == 0
    assert harbor_client._promoted_tool_metrics(extract_swe_tool_metrics([]))["agent_swe_tool_result_count"] == 0


@pytest.mark.parametrize(
    "key,value",
    [
        ("agent_swe_tool_timeout_count", -1),
        ("agent_swe_tool_timeout_count", 0.5),
        ("agent_swe_tool_timeout_count", 1),
        ("agent_swe_tool_result_count", 2),
        ("agent_tool_call_count", 2),
        ("agent_tool_unit_success_count", 0),
        ("agent_tool_unit_success_rate", 1.1),
        ("agent_swe_tool_metrics_available", True),
        ("agent_swe_tool_returncode_zero_rate", float("nan")),
    ],
)
def test_tool_metrics_promotion_rejects_inconsistent_values(key, value):
    metrics = _one_tool_metrics()
    metrics[key] = value
    with pytest.raises(ValueError):
        harbor_client._promoted_tool_metrics(metrics)


@pytest.mark.parametrize(
    "error,field",
    [
        (httpx.ConnectError("refused"), "connect_error"),
        (httpx.ConnectTimeout("timeout"), "timeout"),
        (httpx.ReadTimeout("timeout"), "timeout"),
        (asyncio.TimeoutError(), "timeout"),
        (httpx.ReadError("reset"), "transport_error"),
        (
            httpx.HTTPStatusError(
                "bad gateway", request=httpx.Request("POST", "http://harbor/run"), response=httpx.Response(502)
            ),
            "http_error",
        ),
    ],
)
def test_run_reports_request_failure_and_clears_tool_metrics(monkeypatch, error, field):
    async def post(*args):
        raise error

    monkeypatch.setattr(harbor_client, "_post_trial", post)
    result = asyncio.run(harbor_client.run("http://router/session", None, metadata={"instance_id": "task"}))
    assert result[f"agent_swe_harbor_{field}_count"] == 1
    assert result["agent_swe_harbor_request_count"] == 1
    assert result["agent_swe_harbor_request_success_count"] == 0
    assert result["agent_excluded_from_training"] is True
    assert result["agent_tool_unit_success_rate"] is None


def test_run_promotes_tool_metrics_after_success(monkeypatch):
    async def post(*args):
        return {
            "reward": 0,
            "exit_status": "Submitted",
            "eval_report": {"reward": 0},
            "agent_metrics": _one_tool_metrics(1),
        }

    monkeypatch.setattr(harbor_client, "_post_trial", post)
    result = asyncio.run(harbor_client.run("http://router/session", None, metadata={"instance_id": "task"}))
    assert result["agent_tool_unit_success_rate"] == 1
    assert result["agent_swe_tool_returncode_zero_rate"] == 0
    assert result["agent_swe_harbor_request_success_count"] == 1
    assert result["agent_excluded_from_training"] is False


def test_default_trial_timeout_exceeds_harbor_agent_timeout(monkeypatch):
    monkeypatch.delenv("HARBOR_TRIAL_TIMEOUT_SECONDS", raising=False)

    assert harbor_client._trial_timeout_seconds() == 12000


@pytest.mark.parametrize("exit_status", [None, "", False, 1])
def test_trial_status_helpers_reject_invalid_status(exit_status):
    with pytest.raises(ValueError, match="exit_status"):
        harbor_client._failed_trial(exit_status)
    with pytest.raises(ValueError, match="exit_status"):
        harbor_client._is_trainable_response(exit_status=exit_status, eval_report={"reward": 1}, reward=1)


@pytest.fixture(autouse=True)
def clean_client_env(monkeypatch):
    harbor_client._client = None
    for name in (
        "FORCE_EXCLUDE",
        "AGENT_MODEL_NAME",
        "HARBOR_ADMIN_SECRET",
        "HARBOR_SERVER_URL",
        "HARBOR_PROXY_URL",
        "HARBOR_TRIAL_TIMEOUT_SECONDS",
        "MILES_ROUTER_EXTERNAL_HOST",
        "MILES_ROUTER_EXTERNAL_PORT",
        "SWE_AGENT_MAX_TURNS",
        "SWE_CONTEXT_RESERVE_TOKENS",
        "SWE_MAX_TOKENS_PER_TURN",
    ):
        monkeypatch.delenv(name, raising=False)
    yield
    if harbor_client._client is not None:
        asyncio.run(harbor_client._client.aclose())
        harbor_client._client = None


def test_get_client_configures_harbor_proxy(monkeypatch):
    captured = {}

    def transport_factory(**kwargs):
        captured.update(kwargs)
        return httpx.MockTransport(lambda request: httpx.Response(200, request=request))

    monkeypatch.setenv("HARBOR_PROXY_URL", "http://127.0.0.1:1055")
    monkeypatch.setattr(harbor_client.httpx, "AsyncHTTPTransport", transport_factory)

    harbor_client._get_client()

    assert captured["proxy"] == "http://127.0.0.1:1055"
    assert captured["socket_options"]


def test_get_client_rejects_invalid_harbor_proxy(monkeypatch):
    monkeypatch.setenv("HARBOR_PROXY_URL", "127.0.0.1:1055")
    with pytest.raises(ValueError, match="HARBOR_PROXY_URL"):
        harbor_client._get_client()


@pytest.mark.parametrize("proxy", ["", "http://127.0.0.1:1055"])
@pytest.mark.parametrize("limit", [1, 128, 512])
def test_get_client_applies_limit_to_real_transport_pool(monkeypatch, proxy, limit):
    monkeypatch.setenv("HARBOR_PROXY_URL", proxy)
    monkeypatch.setattr(harbor_client, "_MAX_HARBOR_CONNECTIONS", limit)
    client = harbor_client._get_client()
    assert client._transport._pool._max_connections == limit
    assert client._transport._pool._max_keepalive_connections == limit
    assert harbor_client._get_client() is client


@pytest.mark.parametrize("limit", [0, -1, True])
def test_get_client_rejects_invalid_connection_limit(monkeypatch, limit):
    monkeypatch.setattr(harbor_client, "_MAX_HARBOR_CONNECTIONS", limit)
    with pytest.raises(ValueError, match="_MAX_HARBOR_CONNECTIONS"):
        harbor_client._get_client()
    assert harbor_client._client is None


def test_build_request_carries_task_session_and_sampling(monkeypatch):
    monkeypatch.setenv("AGENT_MODEL_NAME", "Qwen3.5-4B")
    request = harbor_client.build_request(
        base_url="http://node-0:30000/sessions/session-1",
        request_kwargs={"temperature": 0.8, "top_p": 0.95, "seed": 7},
        metadata={
            "instance_id": "r2e-pandas-deadbeef",
            "session_server_id": "node-0:30000",
            "session_server_instance_id": "server-1",
            "max_seq_len": "65536",
        },
    )

    assert request == {
        "instance_id": "r2e-pandas-deadbeef",
        "session_server_id": "node-0:30000",
        "session_server_instance_id": "server-1",
        "max_seq_len": 65536,
        "base_url": "http://node-0:30000/sessions/session-1/v1",
        "model": "openai/Qwen3.5-4B",
        "agent_name": "mini-swe-agent",
        "sampling_params": {"temperature": 0.8, "top_p": 0.95, "seed": 7},
    }


def test_build_request_rewrites_router_host(monkeypatch):
    monkeypatch.setenv("MILES_ROUTER_EXTERNAL_HOST", "127.0.0.1")
    request = harbor_client.build_request(
        base_url="http://node-0:30000/sessions/session-1",
        request_kwargs={},
        metadata={"instance_id": "r2e-pandas-deadbeef", "session_server_id": "node-0:30000"},
    )

    assert request["base_url"] == "http://127.0.0.1:30000/sessions/session-1/v1"
    assert request["session_server_id"] == "127.0.0.1:30000"


def test_build_request_rewrites_router_host_and_port(monkeypatch):
    monkeypatch.setenv("MILES_ROUTER_EXTERNAL_HOST", "127.0.0.1")
    monkeypatch.setenv("MILES_ROUTER_EXTERNAL_PORT", "31000")

    request = harbor_client.build_request(
        base_url="http://node-0:30000/sessions/session-1",
        request_kwargs={},
        metadata={
            "instance_id": "r2e-pandas-deadbeef",
            "session_server_id": "node-0:30000",
        },
    )

    assert request["base_url"] == "http://127.0.0.1:31000/sessions/session-1/v1"
    assert request["session_server_id"] == "127.0.0.1:31000"


@pytest.mark.parametrize("port", ["0", "65536"])
def test_build_request_rejects_invalid_router_external_port(monkeypatch, port):
    monkeypatch.setenv("MILES_ROUTER_EXTERNAL_HOST", "127.0.0.1")
    monkeypatch.setenv("MILES_ROUTER_EXTERNAL_PORT", port)

    with pytest.raises(ValueError, match="MILES_ROUTER_EXTERNAL_PORT"):
        harbor_client.build_request(
            base_url="http://node-0:30000/sessions/session-1",
            request_kwargs={},
            metadata={"instance_id": "r2e-pandas-deadbeef"},
        )


def test_build_request_carries_turn_and_context_reserve(monkeypatch):
    monkeypatch.setenv("SWE_AGENT_MAX_TURNS", "250")
    monkeypatch.setenv("SWE_CONTEXT_RESERVE_TOKENS", "32768")
    monkeypatch.setenv("SWE_MAX_TOKENS_PER_TURN", "8192")

    request = harbor_client.build_request(
        base_url="http://node-0:30000/sessions/session-1",
        request_kwargs={"max_tokens": 8192},
        metadata={"instance_id": "r2e-pandas-deadbeef", "max_seq_len": 163840},
    )

    assert request["max_turns"] == 250
    assert request["context_reserve_tokens"] == 32768
    assert request["sampling_params"]["max_tokens"] == 8192


def test_build_request_maps_local_horizon_to_total_replay_limit(monkeypatch):
    request = harbor_client.build_request(
        base_url="http://node-0:30000/sessions/session-1",
        request_kwargs={},
        metadata={
            "instance_id": "r2e-pandas-deadbeef",
            "ab_local_rollout": True,
            "agent_max_turns": 10,
            "swe_replay_path": "/trials/source/replay.json",
            "swe_replay_n_calls": 37,
        },
    )

    assert request["replay_path"] == "/trials/source/replay.json"
    assert request["max_turns"] == 47
    assert request["run_verifier"] is False
    assert request["force_submit_on_limit"] is False


def test_build_request_does_not_send_controller_only_metadata():
    request = harbor_client.build_request(
        base_url="http://node-0:30000/sessions/session-1",
        request_kwargs={},
        metadata={
            "instance_id": "r2e-pandas-deadbeef",
            "ab_swe_golden_patch": "private patch",
            "ab_branch_spec": {"private": True},
        },
    )

    assert "ab_swe_golden_patch" not in request
    assert "ab_branch_spec" not in request


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("SWE_AGENT_MAX_TURNS", "0"),
        ("SWE_CONTEXT_RESERVE_TOKENS", "-1"),
        ("SWE_MAX_TOKENS_PER_TURN", "0"),
    ],
)
def test_build_request_rejects_nonpositive_limits(monkeypatch, name, value):
    monkeypatch.setenv(name, value)
    with pytest.raises(ValueError, match=name):
        harbor_client.build_request(
            base_url="http://node-0:30000/sessions/session-1",
            request_kwargs={},
            metadata={"instance_id": "r2e-pandas-deadbeef", "max_seq_len": 163840},
        )


def test_build_request_rejects_reserve_that_consumes_context(monkeypatch):
    monkeypatch.setenv("SWE_CONTEXT_RESERVE_TOKENS", "163840")
    with pytest.raises(ValueError, match="smaller than"):
        harbor_client.build_request(
            base_url="http://node-0:30000/sessions/session-1",
            request_kwargs={},
            metadata={"instance_id": "r2e-pandas-deadbeef", "max_seq_len": 163840},
        )


@pytest.mark.parametrize(
    "metadata",
    [{}, {"instance_id": ""}, {"instance_id": 3}],
)
def test_build_request_rejects_missing_task_id(metadata):
    with pytest.raises(ValueError, match="instance_id"):
        harbor_client.build_request(base_url="http://node:30000/session/1", request_kwargs={}, metadata=metadata)


def test_run_returns_trace_identity_and_excludes_failed_trials(monkeypatch):
    async def fake_post(url, payload):
        assert url == "http://harbor:11200/run"
        assert payload["instance_id"] == "r2e-pandas-deadbeef"
        return {
            "reward": 0,
            "exit_status": "AgentError",
            "exit_status_detail": "AgentError",
            "eval_report": {},
            "agent_metrics": {"turns": 2},
            "trial_dir": "/trials/run-1",
        }

    monkeypatch.setenv("HARBOR_SERVER_URL", "http://harbor:11200/")
    monkeypatch.setattr(harbor_client, "_post_trial", fake_post)

    result = asyncio.run(
        harbor_client.run(
            base_url="http://node:30000/sessions/one",
            prompt="ignored",
            metadata={"instance_id": "r2e-pandas-deadbeef"},
        )
    )

    assert {
        k: v
        for k, v in result.items()
        if k not in harbor_client.SWE_PROMOTED_TOOL_METRIC_KEYS | harbor_client.SWE_HARBOR_METRIC_KEYS
    } == {
        "reward": 0.0,
        "exit_status": "AgentError",
        "exit_status_detail": "AgentError",
        "eval_report": {},
        "agent_metrics": {"turns": 2},
        "agent_turns": 2,
        "agent_excluded_from_training": True,
        "agent_context_reserve_hit": False,
        "agent_max_turns_hit": False,
        "agent_forced_final_answer_reason": None,
        "agent_last_finish_reason": None,
        "trial_dir": "/trials/run-1",
    }


@pytest.mark.parametrize("verifier_reward", [0, 1])
def test_run_excludes_unmarked_full_limits_even_with_verified_reward(monkeypatch, verifier_reward):
    async def fake_post(url, payload):
        return {
            "reward": verifier_reward,
            "exit_status": "LimitsExceeded",
            "eval_report": {"reward": verifier_reward},
            "agent_metrics": {"turns": 250},
            "trial_id": "trial-1",
        }

    monkeypatch.setattr(harbor_client, "_post_trial", fake_post)
    result = asyncio.run(
        harbor_client.run(
            base_url="http://node:30000/sessions/one",
            prompt="ignored",
            metadata={"instance_id": "r2e-pandas-deadbeef"},
        )
    )

    assert result is not None
    assert result["exit_status"] == "LimitsExceeded"
    assert result["reward"] == verifier_reward
    assert result["agent_excluded_from_training"] is True


def test_run_excludes_limits_exceeded_without_verifier_reward(monkeypatch):
    async def fake_post(url, payload):
        return {
            "reward": 0,
            "exit_status": "LimitsExceeded",
            "eval_report": {},
            "agent_metrics": {"turns": 250},
        }

    monkeypatch.setattr(harbor_client, "_post_trial", fake_post)
    result = asyncio.run(
        harbor_client.run(
            base_url="http://node:30000/sessions/one",
            prompt="ignored",
            metadata={"instance_id": "r2e-pandas-deadbeef"},
        )
    )

    assert result is not None
    assert result["agent_excluded_from_training"] is True


@pytest.mark.parametrize("local", [False, True])
@pytest.mark.parametrize("verifier_reward", [None, 0, 1])
def test_length_is_trainable_without_verifier_or_trajectory_download(monkeypatch, local, verifier_reward):
    async def fake_post(url, payload):
        assert url.endswith("/run")
        return {
            "exit_status": "LengthTruncated",
            "reward": verifier_reward or 0,
            "eval_report": {} if verifier_reward is None else {"reward": verifier_reward},
            "agent_metrics": {"turns": 1},
        }

    async def unexpected_download(*args):
        pytest.fail("length has a known zero training reward and does not need PRM messages")

    monkeypatch.setattr(harbor_client, "_post_trial", fake_post)
    monkeypatch.setattr(harbor_client, "get_trial_messages", unexpected_download)
    metadata = {"instance_id": "r2e-pandas-deadbeef"}
    if local:
        metadata.update(
            ab_local_rollout=True, agent_max_turns=10, swe_replay_path="/replay.json", swe_replay_n_calls=0
        )
    result = asyncio.run(harbor_client.run("http://node:30000/sessions/one", "ignored", metadata=metadata))
    assert result["agent_last_finish_reason"] == "length"
    assert result["agent_excluded_from_training"] is False
    assert result["reward"] == (verifier_reward or 0)  # raw verifier result stays available for auditing


@pytest.mark.parametrize("previous_limit", [None, "context_reserve", "max_turns"])
@pytest.mark.parametrize("previous_finish_reason", [None, "length"])
@pytest.mark.parametrize(
    "exit_status, finish_reason, verifier_reward, expected_score, excluded",
    [
        ("Submitted", "stop", 1, 1, False),
        ("Submitted", "tool_calls", 0, 0, False),
        ("LengthTruncated", "length", 1, 0, False),
        ("AgentError", "stop", 0, 0, True),
    ],
)
def test_group_resample_uses_current_finish_reason(
    monkeypatch,
    previous_limit,
    previous_finish_reason,
    exit_status,
    finish_reason,
    verifier_reward,
    expected_score,
    excluded,
):
    sample = Sample(
        index=186,
        group_index=23,
        prompt="issue",
        status=Sample.Status.TRUNCATED if previous_finish_reason == "length" else Sample.Status.COMPLETED,
        metadata={
            "instance_id": "r2e-pandas-deadbeef",
            "agent_last_finish_reason": previous_finish_reason,
            "agent_excluded_from_training": False,
            "agent_context_reserve_hit": previous_limit == "context_reserve",
            "agent_max_turns_hit": previous_limit == "max_turns",
            "agent_forced_final_answer_reason": previous_limit,
            "agent_metrics": {"agent_forced_final_answer_reason": previous_limit},
        },
    )
    group = [sample, Sample(group_index=23, status=Sample.Status.ABORTED)]
    # The collector resets every member when one member aborts, retaining metadata.
    for member in group:
        member.reset_for_retry()
    assert sample.status == Sample.Status.ABORTED
    assert sample.metadata["agent_last_finish_reason"] == previous_finish_reason

    async def fake_post(url, payload):
        assert url.endswith("/run")
        return {
            "exit_status": exit_status,
            "reward": verifier_reward,
            "eval_report": {"reward": verifier_reward},
        }

    monkeypatch.setattr(harbor_client, "_post_trial", fake_post)
    agent_metadata = asyncio.run(
        harbor_client.run("http://node:30000/sessions/new", sample.prompt, metadata=sample.metadata)
    )
    record = SessionRecord(
        timestamp=0,
        method="POST",
        path="/v1/chat/completions",
        status_code=200,
        request={"input_ids": [1, 2]},
        response={"choices": [{"finish_reason": finish_reason, "meta_info": {"output_token_logprobs": [[-0.1, 3]]}}]},
    )
    samples = compute_samples_from_openai_records(
        SimpleNamespace(), sample, [record], SimpleNamespace(decode=lambda ids: str(ids))
    )
    assert len(samples) == 1
    current = samples[0]
    current.metadata.update(agent_metadata)  # Same merge as agentic_tool_call.generate.
    result = asyncio.run(reward.reward_func(None, current))

    assert result["score"] == expected_score
    assert current.metadata["agent_last_finish_reason"] == ("length" if exit_status == "LengthTruncated" else None)
    assert current.status == (Sample.Status.TRUNCATED if finish_reason == "length" else Sample.Status.COMPLETED)
    assert current.remove_sample is excluded
    assert current.metadata["agent_excluded_from_training"] is excluded
    assert current.metadata["agent_context_reserve_hit"] is False
    assert current.metadata["agent_max_turns_hit"] is False
    assert current.metadata["agent_forced_final_answer_reason"] is None
    assert current.tokens == [1, 2, 3] and current.loss_mask == [1]


@pytest.mark.parametrize(
    "metrics",
    [
        {"agent_context_reserve_hit": True},
        {"agent_forced_final_answer_reason": "context_reserve"},
        {"agent_forced_final_answer_reason": "max_turns"},
    ],
)
def test_length_during_forced_submission_remains_excluded(monkeypatch, metrics):
    async def fake_post(url, payload):
        return {"exit_status": "LengthTruncated", "reward": 0, "agent_metrics": metrics}

    monkeypatch.setattr(harbor_client, "_post_trial", fake_post)
    result = asyncio.run(
        harbor_client.run(
            "http://node:30000/sessions/one",
            "ignored",
            metadata={"instance_id": "r2e-pandas-deadbeef"},
        )
    )
    assert result["agent_excluded_from_training"] is True


def test_run_keeps_local_replay_without_verifier(monkeypatch):
    async def fake_post(url, payload):
        assert payload["run_verifier"] is False
        return {
            "reward": 0,
            "exit_status": "LimitsExceeded",
            "eval_report": {},
            "agent_metrics": {"turns": 47},
            "trial_dir": "/trials/local-1",
        }

    async def fake_messages(trial_dir):
        assert trial_dir == "/trials/local-1"
        return [{"role": "user", "content": "prefix"}, {"role": "assistant", "content": "repair"}]

    monkeypatch.setattr(harbor_client, "_post_trial", fake_post)
    monkeypatch.setattr(harbor_client, "get_trial_messages", fake_messages)
    result = asyncio.run(
        harbor_client.run(
            base_url="http://node:30000/sessions/one",
            prompt="ignored",
            metadata={
                "instance_id": "r2e-pandas-deadbeef",
                "ab_local_rollout": True,
                "agent_max_turns": 10,
                "swe_replay_path": "/trials/source/replay.json",
                "swe_replay_n_calls": 37,
            },
        )
    )

    assert result["agent_excluded_from_training"] is False
    assert result["agent_turns"] == 10
    assert result["agent_last_finish_reason"] is None
    assert result["agent_max_turns_hit"] is False
    assert result["messages"][-1]["role"] == "assistant"


@pytest.mark.parametrize("local", [False, True])
@pytest.mark.parametrize("exit_status", ["Submitted", "LimitsExceeded", "LengthTruncated", "BadRequestError"])
@pytest.mark.parametrize("score", [0, 1])
@pytest.mark.parametrize("marker", ["flag", "reason"])
@pytest.mark.parametrize("reason", ["context_reserve", "max_turns"])
def test_forced_limit_excludes_every_final_status_without_artifact_or_prm(
    monkeypatch, local, exit_status, score, marker, reason
):
    flag = "agent_context_reserve_hit" if reason == "context_reserve" else "agent_max_turns_hit"
    metrics = {flag: True} if marker == "flag" else {"agent_forced_final_answer_reason": reason}

    async def fake_post(url, payload):
        assert url.endswith("/run")
        return {
            "exit_status": exit_status,
            "reward": score,
            "eval_report": {"reward": score},
            "agent_metrics": metrics,
        }

    async def unexpected_download(*args):
        pytest.fail("context-excluded Local samples do not need a PRM trajectory download")

    monkeypatch.setattr(harbor_client, "_post_trial", fake_post)
    monkeypatch.setattr(harbor_client, "get_trial_messages", unexpected_download)
    metadata = {"instance_id": "r2e-pandas-deadbeef"}
    if local:
        metadata.update(
            ab_local_rollout=True, agent_max_turns=15, swe_replay_path="/replay.json", swe_replay_n_calls=0
        )
    result = asyncio.run(harbor_client.run("http://node:30000/sessions/one", "ignored", metadata=metadata))
    assert result["agent_context_reserve_hit"] is (reason == "context_reserve")
    assert result["agent_max_turns_hit"] is (reason == "max_turns")
    assert result["agent_excluded_from_training"] is True
    assert result["reward"] == score  # Preserve raw outcome for audit.


@pytest.mark.parametrize("turns, completion, excluded", [(1, 375, False), (1, 376, True), (15, 376, False)])
def test_old_local_runtime_context_stop_is_detected_without_restarting_harbor(
    monkeypatch, caplog, turns, completion, excluded
):
    monkeypatch.setenv("SWE_CONTEXT_RESERVE_TOKENS", "32768")

    async def fake_post(url, payload):
        return {
            "exit_status": "LimitsExceeded",
            "reward": 0,
            "agent_metrics": {"turns": turns},
            "trial_dir": "/trials/local-old",
        }

    async def fake_messages(trial_dir):
        return [
            {
                "role": "assistant",
                "extra": {
                    "response": {
                        "usage": {
                            "prompt_tokens": 229000,
                            "completion_tokens": completion,
                        }
                    }
                },
            }
        ]

    monkeypatch.setattr(harbor_client, "_post_trial", fake_post)
    monkeypatch.setattr(harbor_client, "get_trial_messages", fake_messages)
    result = asyncio.run(
        harbor_client.run(
            "http://node:30000/sessions/one",
            "ignored",
            metadata={
                "instance_id": "r2e-pandas-deadbeef",
                "ab_local_rollout": True,
                "agent_max_turns": 15,
                "swe_replay_path": "/replay.json",
                "swe_replay_n_calls": 0,
                "max_seq_len": 262144,
            },
        )
    )
    assert result["agent_context_reserve_hit"] is excluded
    assert result["agent_excluded_from_training"] is excluded
    assert ("reconstructed runtime guard" in caplog.text) is excluded


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1, 0.5, 2])
def test_context_exclusion_does_not_hide_invalid_verifier_reward(monkeypatch, value):
    async def fake_post(url, payload):
        return {"exit_status": "Submitted", "reward": value, "agent_metrics": {"agent_context_reserve_hit": True}}

    monkeypatch.setattr(harbor_client, "_post_trial", fake_post)
    with pytest.raises(ValueError, match="finite and binary"):
        asyncio.run(
            harbor_client.run("http://node:30000/sessions/one", "ignored", metadata={"instance_id": "r2e-test"})
        )


def test_generated_agent_turns_handles_missing_and_replay_prefix():
    assert harbor_client._generated_agent_turns({}) is None
    assert harbor_client._generated_agent_turns({"turns": 0}) == 0
    assert harbor_client._generated_agent_turns({"turns": 6}, replay_prefix_calls=4) == 2


@pytest.mark.parametrize(
    ("metrics", "prefix"),
    [({"turns": True}, 0), ({"turns": -1}, 0), ({"turns": 3}, 4)],
)
def test_generated_agent_turns_rejects_invalid_counts(metrics, prefix):
    with pytest.raises(ValueError):
        harbor_client._generated_agent_turns(metrics, replay_prefix_calls=prefix)


def test_trajectory_and_replay_helpers_use_harbor_api(monkeypatch):
    calls = []

    async def fake_post(url, payload):
        calls.append((url, payload))
        if url.endswith("/trajectory"):
            return {"messages": [{"role": "assistant", "content": "step"}], "assistant_turns": 1}
        return {
            "replay_path": "/trials/source/agent/local-replay-turn-1.json",
            "prefix_messages": [{"role": "user", "content": "issue"}],
            "n_calls": 0,
        }

    monkeypatch.setenv("HARBOR_SERVER_URL", "http://harbor:11200")
    monkeypatch.setattr(harbor_client, "_post_trial", fake_post)

    messages = asyncio.run(harbor_client.get_trial_messages("/trials/source"))
    replay = asyncio.run(harbor_client.create_trial_replay("/trials/source", 1))

    assert messages[0]["content"] == "step"
    assert replay["n_calls"] == 0
    assert calls == [
        ("http://harbor:11200/trajectory", {"trial_dir": "/trials/source"}),
        ("http://harbor:11200/replay", {"trial_dir": "/trials/source", "selected_turn": 1}),
    ]


@pytest.fixture(params=["trajectory", "replay"])
def artifact_api(request, monkeypatch):
    monkeypatch.setenv("HARBOR_SERVER_URL", "http://harbor:11200")
    monkeypatch.setenv("HARBOR_ADMIN_SECRET", "test-secret")
    messages = [{"role": "user", "content": "issue"}]
    if request.param == "trajectory":
        return SimpleNamespace(
            call=lambda: harbor_client.get_trial_messages("/trials/source"),
            path="/trajectory",
            payload={"trial_dir": "/trials/source"},
            body={"messages": messages},
            expected=messages,
        )
    body = {"replay_path": "/trials/source/agent/local-replay-turn-1.json", "prefix_messages": messages, "n_calls": 0}
    return SimpleNamespace(
        call=lambda: harbor_client.create_trial_replay("/trials/source", 1),
        path="/replay",
        payload={"trial_dir": "/trials/source", "selected_turn": 1},
        body=body,
        expected=body,
    )


@pytest.mark.parametrize("success_attempt", [1, 10, None])
@pytest.mark.parametrize("failure", [502, 503, 504, httpx.ConnectError, httpx.RemoteProtocolError, httpx.ReadTimeout])
def test_artifact_request_retries_only_the_same_request(artifact_api, monkeypatch, caplog, success_attempt, failure):
    calls = []
    sleeps = []

    def respond(request):
        calls.append(request)
        if len(calls) == success_attempt:
            return httpx.Response(200, json=artifact_api.body)
        if isinstance(failure, int):
            return httpx.Response(failure)
        raise failure("temporary connection failure", request=request)

    async def fake_sleep(seconds):
        sleeps.append(seconds)

    harbor_client._client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
    monkeypatch.setattr(harbor_client.asyncio, "sleep", fake_sleep)
    if success_attempt is None:
        with pytest.raises(httpx.HTTPStatusError if isinstance(failure, int) else failure):
            asyncio.run(artifact_api.call())
    else:
        assert asyncio.run(artifact_api.call()) == artifact_api.expected

    attempts = success_attempt or 10
    assert len(calls) == attempts
    assert sleeps == [1] * (attempts - 1)
    for request in calls:
        assert request.url.path == artifact_api.path
        assert json.loads(request.content) == artifact_api.payload
        assert request.headers["Authorization"] == "Bearer test-secret"
    failures = attempts - (success_attempt is not None)
    assert len(caplog.records) == failures
    if failures:
        assert f"attempt {failures}/10" in caplog.records[-1].message
        assert "trial_dir=/trials/source" in caplog.text
        assert "test-secret" not in caplog.text


@pytest.mark.parametrize(
    "status, body",
    [(400, {}), (401, {}), (403, {}), (404, {}), (429, {}), (500, {}), (200, b"not json"), (200, b"[]"), (200, b"{}")],
)
def test_artifact_request_fails_fast_on_status_or_invalid_data(artifact_api, monkeypatch, status, body):
    calls = []

    def respond(request):
        calls.append(request)
        return httpx.Response(status, content=body) if isinstance(body, bytes) else httpx.Response(status, json=body)

    async def unexpected_sleep(seconds):
        pytest.fail("permanent errors must not be retried")

    harbor_client._client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
    monkeypatch.setattr(harbor_client.asyncio, "sleep", unexpected_sleep)
    with pytest.raises((httpx.HTTPStatusError, ValueError, TypeError)):
        asyncio.run(artifact_api.call())
    assert len(calls) == 1


def test_artifact_timeout_cancels_request_before_retry(artifact_api, monkeypatch):
    calls = 0
    cancelled = 0
    sleeps = []
    assert harbor_client._ARTIFACT_REQUEST_TIMEOUT_SECONDS == 60
    monkeypatch.setattr(harbor_client, "_ARTIFACT_REQUEST_TIMEOUT_SECONDS", 0.01)

    async def hang(request):
        nonlocal calls, cancelled
        assert calls == cancelled
        calls += 1
        try:
            await asyncio.Event().wait()
        finally:
            cancelled += 1

    async def fake_sleep(seconds):
        sleeps.append(seconds)

    harbor_client._client = httpx.AsyncClient(transport=httpx.MockTransport(hang))
    monkeypatch.setattr(harbor_client.asyncio, "sleep", fake_sleep)
    with pytest.raises(asyncio.TimeoutError):
        asyncio.run(artifact_api.call())
    assert calls == cancelled == 10
    assert sleeps == [1] * 9


def test_artifact_request_preserves_external_cancellation(artifact_api, monkeypatch):
    calls = []

    async def unexpected_sleep(seconds):
        pytest.fail("cancellation must not be retried")

    async def cancel_request():
        started = asyncio.Event()

        async def hang(request):
            calls.append(request)
            started.set()
            await asyncio.Event().wait()

        harbor_client._client = httpx.AsyncClient(transport=httpx.MockTransport(hang))
        task = asyncio.create_task(artifact_api.call())
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    monkeypatch.setattr(harbor_client.asyncio, "sleep", unexpected_sleep)
    asyncio.run(cancel_request())
    assert len(calls) == 1


@pytest.mark.parametrize("path", ["/run", "/flush", "/sessions"])
def test_artifact_retry_rejects_other_endpoints(path):
    with pytest.raises(ValueError, match="retries are not allowed"):
        asyncio.run(harbor_client._post_trial_artifact(f"http://harbor:11200{path}", {}))
    assert harbor_client._client is None


@pytest.mark.parametrize("trial_dir", ["", " ", None])
def test_artifact_helpers_reject_invalid_trial_dir(trial_dir):
    with pytest.raises(ValueError, match="trial_dir"):
        asyncio.run(harbor_client.get_trial_messages(trial_dir))
    with pytest.raises(ValueError, match="trial_dir"):
        asyncio.run(harbor_client.create_trial_replay(trial_dir, 1))
    assert harbor_client._client is None


@pytest.mark.parametrize("selected_turn", [0, -1, True, 1.5])
def test_replay_rejects_invalid_selected_turn(selected_turn):
    with pytest.raises(ValueError, match="selected_turn"):
        asyncio.run(harbor_client.create_trial_replay("/trials/source", selected_turn))
    assert harbor_client._client is None


def test_run_does_not_retry_gateway_error():
    calls = []

    def respond(request):
        calls.append(request)
        return httpx.Response(502)

    harbor_client._client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
    result = asyncio.run(
        harbor_client.run(
            base_url="http://node:30000/sessions/one",
            prompt="ignored",
            metadata={"instance_id": "r2e-pandas-deadbeef"},
        )
    )
    assert result["agent_excluded_from_training"] is True
    assert result["exit_status"] == "HarborTransportError"
    assert len(calls) == 1
    assert calls[0].url.path == "/run"


def test_run_rejects_mismatched_verifier_reward(monkeypatch):
    async def fake_post(url, payload):
        return {
            "reward": 1,
            "exit_status": "Submitted",
            "eval_report": {"reward": 0},
            "agent_metrics": {},
        }

    monkeypatch.setattr(harbor_client, "_post_trial", fake_post)
    with pytest.raises(ValueError, match="does not match"):
        asyncio.run(
            harbor_client.run(
                base_url="http://node:30000/sessions/one",
                prompt="ignored",
                metadata={"instance_id": "r2e-pandas-deadbeef"},
            )
        )


def test_run_maps_transport_failure_to_dropped_sample(monkeypatch):
    async def fail(url, payload):
        request = httpx.Request("POST", url)
        raise httpx.ConnectError("offline", request=request)

    monkeypatch.setattr(harbor_client, "_post_trial", fail)
    result = asyncio.run(
        harbor_client.run(
            base_url="http://node:30000/sessions/one",
            prompt="ignored",
            metadata={"instance_id": "r2e-pandas-deadbeef"},
        )
    )
    assert {
        k: v
        for k, v in result.items()
        if k not in harbor_client.SWE_PROMOTED_TOOL_METRIC_KEYS | harbor_client.SWE_HARBOR_METRIC_KEYS
    } == {
        "reward": 0.0,
        "exit_status": "HarborTransportError",
        "eval_report": {},
        "agent_metrics": {},
        "agent_excluded_from_training": True,
        "agent_context_reserve_hit": False,
        "agent_max_turns_hit": False,
        "agent_forced_final_answer_reason": None,
    }


def test_abort_flushes_only_the_current_session(monkeypatch):
    captured = {}

    async def fake_flush(url, payload, headers):
        captured.update(url=url, payload=payload, headers=headers)

    monkeypatch.setattr(harbor_client, "_flush_trials", fake_flush)
    monkeypatch.setenv("HARBOR_SERVER_URL", "http://harbor:11200/")
    monkeypatch.setenv("HARBOR_ADMIN_SECRET", "secret")

    asyncio.run(harbor_client.abort(SimpleNamespace(session_server_instance_id="server-1")))

    assert captured == {
        "url": "http://harbor:11200/flush",
        "payload": {"session_server_instance_id": "server-1"},
        "headers": {"Authorization": "Bearer secret"},
    }


def test_flush_trials_retries_transport_failures(monkeypatch):
    attempts = 0
    sleeps = []

    class FakeClient:
        async def post(self, url, json, headers):
            nonlocal attempts
            attempts += 1
            if attempts < 3:
                raise httpx.ConnectError("offline")
            return httpx.Response(200, request=httpx.Request("POST", url))

    async def fake_sleep(seconds):
        sleeps.append(seconds)

    monkeypatch.setattr(harbor_client, "_get_client", FakeClient)
    monkeypatch.setattr(harbor_client.asyncio, "sleep", fake_sleep)

    asyncio.run(harbor_client._flush_trials("http://harbor/flush", {"id": "one"}, None))

    assert attempts == 3
    assert sleeps == [1, 1]


@pytest.mark.parametrize("proxy", ["", "http://127.0.0.1:1055"])
def test_default_shared_pool_has_512_connections(monkeypatch, proxy):
    monkeypatch.setenv("HARBOR_PROXY_URL", proxy)
    client = harbor_client._get_client()
    assert client._transport._pool._max_connections == 512
    assert client._transport._pool._max_keepalive_connections == 512


def test_trial_and_artifact_requests_reuse_one_client(monkeypatch):
    transports = []
    received = []

    def respond(request):
        received.append(request.url.path)
        return httpx.Response(200, json={"path": request.url.path})

    def transport_factory(**kwargs):
        assert kwargs["limits"].max_connections == 512
        transport = httpx.MockTransport(respond)
        transports.append(transport)
        return transport

    monkeypatch.setattr(harbor_client.httpx, "AsyncHTTPTransport", transport_factory)

    async def exercise():
        for endpoint in ("/run", "/trajectory", "/replay"):
            assert await harbor_client._post_trial(f"http://harbor:11200{endpoint}", {}) == {"path": endpoint}

    asyncio.run(exercise())
    assert received == ["/run", "/trajectory", "/replay"]
    assert len(transports) == 1


@pytest.mark.parametrize("endpoint", ["/trajectory", "/replay"])
@pytest.mark.parametrize("limit, n_trials", [(1, 1), (2, 1), (512, 192)])
def test_artifact_progress_depends_on_shared_pool_headroom(monkeypatch, endpoint, limit, n_trials):
    """Exercise real sockets, including 192 held /run requests in the 512 pool."""
    assert 0 < n_trials <= limit
    monkeypatch.setattr(harbor_client, "_MAX_HARBOR_CONNECTIONS", limit)
    # Production uses Linux TCP keepalive constants. Leave the real HTTP pool
    # intact but omit OS-specific socket tuning in this portable loopback test.
    transport_type = httpx.AsyncHTTPTransport

    def portable_transport(**kwargs):
        assert "socket_options" in kwargs and "limits" in kwargs
        kwargs["socket_options"] = None
        return transport_type(**kwargs)

    monkeypatch.setattr(harbor_client.httpx, "AsyncHTTPTransport", portable_transport)

    async def exercise():
        started = asyncio.Event()
        release = asyncio.Event()
        handlers = set()
        received = []

        async def serve(reader, writer):
            handlers.add(asyncio.current_task())
            try:
                header = (await reader.readuntil(b"\r\n\r\n")).decode("ascii")
                method, path, _ = header.splitlines()[0].split()
                assert method == "POST" and path in {"/run", endpoint}
                fields = dict(line.split(": ", 1) for line in header.splitlines()[1:] if line)
                payload = json.loads(await reader.readexactly(int(fields["Content-Length"])))
                assert payload == {"trial_dir": "/trial/source"}
                received.append(path)
                if path == "/run":
                    if received.count("/run") == n_trials:
                        started.set()
                    await release.wait()
                body = json.dumps({"path": path}).encode()
                writer.write(
                    b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nConnection: close\r\n"
                    + f"Content-Length: {len(body)}\r\n\r\n".encode()
                    + body
                )
                await writer.drain()
            finally:
                writer.close()
                await writer.wait_closed()

        server = await asyncio.start_server(serve, "127.0.0.1", 0, backlog=512)
        assert len(server.sockets) == 1
        url = f"http://127.0.0.1:{server.sockets[0].getsockname()[1]}"
        trials = [
            asyncio.create_task(harbor_client._post_trial(url + "/run", {"trial_dir": "/trial/source"}))
            for _ in range(n_trials)
        ]
        try:
            await asyncio.wait_for(started.wait(), 5)
            request = harbor_client._post_trial(url + endpoint, {"trial_dir": "/trial/source"})
            if limit > n_trials:
                assert await asyncio.wait_for(request, 2) == {"path": endpoint}
                assert received == ["/run"] * n_trials + [endpoint]
            else:
                with pytest.raises(asyncio.TimeoutError):
                    await asyncio.wait_for(request, 0.1)
                assert received == ["/run"] * n_trials
            assert all(not trial.done() for trial in trials), "artifact request must not cancel or restart trials"
            release.set()
            assert await asyncio.wait_for(asyncio.gather(*trials), 5) == [{"path": "/run"}] * n_trials
            # An artifact waiting on a full pool must work once /run releases its connection.
            assert await asyncio.wait_for(
                harbor_client._post_trial(url + endpoint, {"trial_dir": "/trial/source"}), 2
            ) == {"path": endpoint}
        finally:
            release.set()
            await asyncio.wait_for(asyncio.gather(*trials), 5)
            if harbor_client._client is not None:
                await harbor_client._client.aclose()
                harbor_client._client = None
            server.close()
            await server.wait_closed()
            await asyncio.gather(*handlers)

    asyncio.run(exercise())
