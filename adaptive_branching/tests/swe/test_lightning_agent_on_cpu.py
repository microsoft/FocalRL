import asyncio
import itertools
import json
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from adaptive_branching.src.swe import agent, lightning_reference as ref, thinking_protocol as protocol
from adaptive_branching.src.swe.lightning_generate import constrain_sample
from adaptive_branching.src.swe.lightning_reward import score_sample

_CALL_IDS = itertools.count()


def completion(command="true", *, content=None, finish="stop", prompt=100, output=10):
    return {
        "choices": [
            {
                "message": {
                    "content": content or "",
                    "tool_calls": (
                        []
                        if content is not None
                        else [
                            {
                                "id": f"call-{next(_CALL_IDS)}",
                                "type": "function",
                                "function": {"name": "bash", "arguments": json.dumps({"command": command})},
                            }
                        ]
                    ),
                },
                "finish_reason": finish,
            }
        ],
        "usage": {"prompt_tokens": prompt, "completion_tokens": output},
    }


def run_loop(replies, outputs=(), config=None):
    seen = []
    replies, outputs = iter(replies), iter(outputs)

    async def query(messages):
        seen.append([dict(message) for message in messages])
        reply = next(replies)
        if isinstance(reply, BaseException):
            raise reply
        return reply

    async def execute(command, timeout):
        result = next(outputs)
        if isinstance(result, BaseException):
            raise result
        return result

    episode = asyncio.run(agent.run_episode("Fix issue", query, execute, config=config))
    return episode, seen


def test_prompts_have_one_source_and_keep_restrictions():
    root = Path(protocol.__file__).parent
    for name in ("SYSTEM_PROMPT", "INSTANCE_PROMPT"):
        value = getattr(protocol, name)
        assert value == (root / "thinking_prompts" / (name.lower() + ".txt")).read_text()
        assert not hasattr(ref, name)
        assert not (root / "lightning_reference" / (name.lower() + ".txt")).exists()
    assert "git is disabled" in protocol.INSTANCE_PROMPT
    assert "network is disabled" in protocol.INSTANCE_PROMPT
    assert "Do NOT install packages" in protocol.INSTANCE_PROMPT
    assert "Do NOT modify test-harness" in protocol.INSTANCE_PROMPT
    assert "patch.txt" not in protocol.INSTANCE_PROMPT


@pytest.mark.parametrize(
    "kwargs",
    [{"max_turns": 0}, {"max_tokens": True}, {"command_timeout": -1}, {"observation_chars": 1}, {"max_tokens": 81920}],
)
def test_config_rejects_invalid_budgets(kwargs):
    with pytest.raises(ValueError):
        agent.AgentConfig(**kwargs)


@pytest.mark.parametrize("code", [0, 1])
def test_submission_first_output_line_ends_even_nonzero(code):
    episode, calls = run_loop([completion("echo " + ref.SUBMIT_MARKER)], [("\n " + ref.SUBMIT_MARKER + "\n", code)])
    assert episode.stop_reason == "Submitted" and episode.turns == 1 and len(calls) == 1
    assert episode.metrics()["agent_swe_tool_submit_count"] == 1
    assert calls[0][0]["content"] == protocol.SYSTEM_PROMPT
    assert calls[0][1]["content"] == protocol.INSTANCE_PROMPT.format(problem_statement="Fix issue")


def test_cap_has_no_forced_query_and_keeps_observation():
    episode, calls = run_loop([completion()], [("done", 0)], agent.AgentConfig(max_turns=1))
    assert episode.stop_reason == "TurnLimit" and len(calls) == 1
    assert episode.messages[-1]["content"] == ref.render_observation(0, "done", 6000)
    assert episode.metrics()["agent_max_turns_hit"] is True
    assert episode.metrics()["agent_context_limit_hit"] is False


def test_context_overflow_does_not_append_failed_turn():
    episode, calls = run_loop([completion(), agent.ContextOverflow()], [("ok", 0)])
    assert episode.stop_reason == "ContextLimit" and episode.turns == 1
    assert len(episode.messages) == 4 and len(calls) == 2
    assert episode.metrics()["agent_context_limit_hit"] is True
    assert episode.metrics()["agent_context_reserve_hit"] is False
    assert episode.metrics()["agent_max_turns_hit"] is False


def test_three_format_errors_end_without_shell():
    episode, calls = run_loop([completion(content="bad")] * 3)
    assert episode.stop_reason == "FormatLimit" and episode.format_errors == 3
    assert len(calls) == 3 and len(episode.tool_events) == 0
    assert calls[1][-1]["content"].startswith(protocol.format_error_message(0, "stop"))


@pytest.mark.parametrize("call_shape", ["none", "valid", "broken_args", "missing_id"])
def test_single_turn_length_stops_before_any_tool_or_followup(call_shape):
    body = completion("echo " + ref.SUBMIT_MARKER, finish="length")
    if call_shape == "none":
        body["choices"][0]["message"]["tool_calls"] = []
    elif call_shape == "broken_args":
        body["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"] = "{"
    elif call_shape == "missing_id":
        del body["choices"][0]["message"]["tool_calls"][0]["id"]
    episode, calls = run_loop([body])
    assert episode.stop_reason == "LengthTruncated" and episode.turns == 1
    assert episode.format_errors == 0 and not episode.tool_events
    assert len(calls) == 1 and len(episode.messages) == 3
    assert episode.messages[-1]["tool_calls"] == body["choices"][0]["message"]["tool_calls"]
    assert episode.metrics()["agent_last_finish_reason"] == "length"


def test_valid_action_resets_consecutive_format_errors():
    episode, _ = run_loop(
        [completion(content="bad"), completion(), completion(content="bad"), completion()],
        [("ok", 0), (ref.SUBMIT_MARKER, 0)],
    )
    assert episode.stop_reason == "Submitted" and episode.format_errors == 2


@pytest.mark.parametrize(
    "command",
    [
        "git log",
        "cat .git/config",
        "ls /opt/agl_tmp",
        "curl https://x",
        "python -m pip install x",
        "python -c 'import requests; requests.get(url)'",
        "echo yes > conftest.py",
    ],
)
def test_forbidden_actions_never_reach_shell(command):
    episode, _ = run_loop([completion(command)], config=agent.AgentConfig(max_turns=1))
    assert episode.blocked_actions == 1 and not episode.tool_events
    assert episode.messages[-1]["content"] == ref.render_observation(1, ref._forbidden_action(command), 6000)


@pytest.mark.parametrize(
    "command",
    [
        'cd /testbed && if git init 2>&1; then echo "git works"; else echo "git not accessible"; fi',
        'if ! git status; then true; fi',
        'while git status; do true; done',
        'until git status; do true; done',
        'if false; then true; elif /usr/bin/git init; then true; fi',
    ],
)
def test_git_in_shell_condition_never_reaches_executor(command):
    episode, _ = run_loop([completion(command)], config=agent.AgentConfig(max_turns=1))
    assert episode.metrics()["agent_lightning_blocked_actions"] == 1
    assert "git is disabled" in episode.messages[-1]["content"]


@pytest.mark.parametrize("command", ["", "true", "echo git init", "if test -f src.py; then cat src.py; fi"])
def test_git_condition_guard_allows_non_git_commands(command):
    assert ref._forbidden_action(command) is None


@pytest.mark.parametrize("error", [httpx.ConnectError("unreachable"), RuntimeError("docker exec broken"), asyncio.CancelledError()])
def test_infra_or_cancellation_propagates(error):
    with pytest.raises(type(error)):
        run_loop([completion()], [error])


def test_timeout_and_nonzero_are_distinct_from_transport_failure():
    episode, _ = run_loop([completion(), completion()], [("fail", 1), ("[timed out after 120s]", 124)], agent.AgentConfig(max_turns=2))
    metrics = episode.metrics()
    assert metrics["agent_swe_tool_returncode_nonzero_count"] == 1
    assert metrics["agent_swe_tool_timeout_count"] == 1
    assert metrics["agent_tool_unit_success_rate"] == 0.5


@pytest.mark.parametrize("size", [0, 1, 5999, 6000, 6001, 10000])
def test_observation_matches_upstream(size):
    text = "x" * size
    episode, _ = run_loop([completion()], [(text, 0)], agent.AgentConfig(max_turns=1))
    assert episode.messages[-1]["content"] == ref.render_observation(0, text, 6000)


@pytest.mark.parametrize("change", [{"choices": []}, {"usage": {}}, {"usage": {"prompt_tokens": -1, "completion_tokens": 1}}])
def test_malformed_model_response_fails(change):
    body = completion()
    body.update(change)
    with pytest.raises(ValueError):
        agent.validate_completion(body)


def test_missing_tool_id_is_serving_configuration_failure():
    body = completion()
    body["choices"][0]["message"]["tool_calls"] = [{"function": {}}]
    with pytest.raises(ValueError, match="IDs"):
        agent.validate_completion(body)


def test_http_sampling_and_pause_retry(monkeypatch):
    payloads = []
    response_body = completion()

    def handler(request):
        payloads.append(json.loads(request.content))
        if len(payloads) == 1:
            return httpx.Response(429, text="gateway paused")
        return httpx.Response(200, json=response_body)

    async def sleep(seconds):
        assert seconds == 5

    monkeypatch.setattr(agent.asyncio, "sleep", sleep)

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await agent.query_model(client, "http://model/chat/completions", "model", [{"role": "user"}], agent.AgentConfig())

    assert asyncio.run(run()) == response_body
    assert len(payloads) == 2 and payloads[0] == payloads[1]
    assert payloads[0]["max_tokens"] == 12288 and payloads[0]["temperature"] == 1
    assert payloads[0]["chat_template_kwargs"] == {"enable_thinking": True}
    assert payloads[0]["tools"] == protocol.TOOLS
    assert payloads[0]["tool_choice"] == "auto"
    assert payloads[0]["parallel_tool_calls"] is True


@pytest.mark.parametrize(
    "status,text,error",
    [
        (400, "maximum context length exceeded", agent.ContextOverflow),
        (
            400,
            "The input (82000 tokens) is longer than the model's context length (81920 tokens).",
            agent.ContextOverflow,
        ),
        (
            400,
            "The input (81,920 tokens) is longer than the model's context length (81,920 tokens).",
            agent.ContextOverflow,
        ),
        (400, "input is longer than expected tool arguments", httpx.HTTPStatusError),
        (
            500,
            "The input (82000 tokens) is longer than the model's context length (81920 tokens).",
            httpx.HTTPStatusError,
        ),
        (400, "invalid model", httpx.HTTPStatusError),
        (500, "server failed", httpx.HTTPStatusError),
    ],
)
def test_only_known_context_errors_are_budget_termination(status, text, error):
    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(status, text=text))) as client:
            await agent.query_model(client, "http://model/chat/completions", "model", [{}], agent.AgentConfig())

    with pytest.raises(error):
        asyncio.run(run())


def result(status="TurnLimit", raw=0, infra=False):
    return {
        "exit_status": status,
        "reward": raw,
        "eval_report": {"reward": raw, **({"infrastructure_error": "failed"} if infra else {})},
        "agent_metrics": {"turns": 100, "agent_lightning_max_prompt_tokens": 64000},
    }


@pytest.mark.parametrize("status", ["Submitted", "TurnLimit", "ContextLimit", "FormatLimit"])
@pytest.mark.parametrize("raw", [0, 1])
def test_capped_and_format_ended_episodes_keep_actual_verifier_reward(status, raw):
    decoded = agent.decode_result(result(status, raw))
    assert decoded["reward"] == raw and decoded["agent_excluded_from_training"] is False
    assert decoded["agent_context_limit_hit"] is (status == "ContextLimit")
    assert decoded["agent_max_turns_hit"] is (status == "TurnLimit")
    assert decoded["agent_context_reserve_hit"] is False


@pytest.mark.parametrize("status,infra", [("AgentInfrastructureFailure", False), ("TurnLimit", True), ("AgentError", False)])
def test_infra_results_never_become_trainable_zero(status, infra):
    assert agent.decode_result(result(status, infra=infra))["agent_excluded_from_training"] is True


def sample(raw=1, turns=100, prompt=64000, evaluation=False):
    return SimpleNamespace(
        metadata={
            "reward": raw,
            "agent_metrics": {"turns": turns, "agent_lightning_max_prompt_tokens": prompt},
            "swe_lightning_evaluation": evaluation,
            "swe_generated_format_audit": {"version": 2, "spans": 1, "repeated_spans": 0, "invalid_spans": 0, "format_failures": {}},
            "exit_status": "TurnLimit",
        },
        remove_sample=False,
        status=SimpleNamespace(name="COMPLETED"),
    )


@pytest.mark.parametrize(
    "raw,turns,prompt,evaluation,expected",
    [
        (1, 80, 50000, False, 1),
        (1, 90, 57000, False, 0.9),
        (1, 100, 64000, False, 0.8),
        (1, 100, 80000, True, 1),
        (0, 100, 80000, False, 0),
    ],
)
def test_reward_penalties_match_reference_and_eval_is_raw(raw, turns, prompt, evaluation, expected):
    s = sample(raw, turns, prompt, evaluation)
    reward = score_sample(s)
    assert reward["score"] == pytest.approx(expected) and reward["acc"] == (raw == 1)
    assert not s.remove_sample


def test_reward_exclusion_and_missing_eval_flag():
    s = sample()
    s.metadata["agent_excluded_from_training"] = True
    assert score_sample(s)["judge_raw"] == "excluded_from_training" and s.remove_sample
    s = sample()
    del s.metadata["swe_lightning_evaluation"]
    with pytest.raises(ValueError):
        score_sample(s)


def training_sample(prompt, response):
    return SimpleNamespace(
        tokens=[1] * (prompt + response),
        response_length=response,
        loss_mask=[1] * response,
        rollout_log_probs=[-0.1] * response,
        reward=1.0,
        metadata={},
        remove_sample=False,
    )


@pytest.mark.parametrize("prompt,response", [(10, 65537), (65537, 1), (10, 81910), (1, 1)])
@pytest.mark.parametrize("evaluation", [False, True])
def test_full_training_row_preserved(prompt, response, evaluation):
    s = training_sample(prompt, response)
    original = {name: getattr(s, name) for name in ("tokens", "loss_mask", "rollout_log_probs")}
    assert constrain_sample(s, evaluation=evaluation) is s
    assert len(s.tokens) == prompt + response and s.response_length == response
    assert not s.remove_sample and s.reward == 1.0
    assert s.metadata == {"swe_lightning_evaluation": evaluation}
    for name, value in original.items():
        assert getattr(s, name) is value
        assert len(value) == (prompt + response if name == "tokens" else response)


@pytest.mark.parametrize("prompt,response", [(0, 0), (1, 0), (1, 1)])
def test_no_trainable_tokens_excluded(prompt, response):
    s = training_sample(prompt, response)
    s.loss_mask = [0] * response
    constrain_sample(s, evaluation=False)
    assert s.remove_sample and s.metadata["agent_excluded_from_training"]


def test_total_context_overflow_fails_without_mutation():
    s = training_sample(10, 81911)
    with pytest.raises(ValueError, match="exceeding model_context=81920"):
        constrain_sample(s, evaluation=False)
    assert len(s.tokens) == 81921 and s.response_length == 81911
    assert len(s.loss_mask) == len(s.rollout_log_probs) == 81911 and s.reward == 1.0
    assert s.metadata == {}


@pytest.mark.parametrize(
    "field,value",
    [
        ("response_length", -1),
        ("response_length", True),
        ("response_length", 3),
        ("loss_mask", []),
        ("rollout_log_probs", []),
        ("metadata", None),
    ],
)
def test_invalid_training_row_fails(field, value):
    s = training_sample(1, 1)
    setattr(s, field, value)
    with pytest.raises(ValueError):
        constrain_sample(s, evaluation=False)


def test_invalid_evaluation_flag_fails():
    with pytest.raises(ValueError):
        constrain_sample(training_sample(1, 1), evaluation=1)


def test_model_context_boundary_before_tool_execution():
    limit = agent.AgentConfig().model_context
    episode, _ = run_loop([completion(ref.SUBMIT_MARKER, prompt=limit - 10, output=10)], [(ref.SUBMIT_MARKER, 0)])
    assert episode.stop_reason == "Submitted"
    # No shell result provided: any accidental tool execution would also fail.
    with pytest.raises(ValueError, match="exceeding model_context"):
        run_loop([completion("true", prompt=limit - 10, output=11)])


def test_request_pins_agent_runtime_without_mutating_old_client(monkeypatch):
    from adaptive_branching.src.swe import harbor_client

    monkeypatch.delenv("SWE_CONTEXT_RESERVE_TOKENS", raising=False)
    metadata = {"instance_id": "r2e-x", "max_seq_len": 131072}
    new = agent.build_request("http://localhost/session/123", {}, metadata)
    old = harbor_client.build_request(base_url="http://localhost/session/123", request_kwargs={}, metadata=metadata)
    assert new["agent_name"] == agent.HARBOR_AGENT and old["agent_name"] == "mini-swe-agent"
    assert new["max_seq_len"] == 81920 and new["max_turns"] == 100
    assert new["force_submit_on_limit"] is False and new["context_reserve_tokens"] is None
    assert metadata["max_seq_len"] == 131072


def test_empty_completion_is_format_feedback_not_infra():
    body = completion()
    body["choices"][0]["message"]["content"] = None
    body["choices"][0]["message"]["tool_calls"] = []
    episode, _ = run_loop([body] * 3)
    assert episode.stop_reason == "FormatLimit"
    assert episode.format_errors == 3


def test_success_clears_retry_infra_flag():
    metadata = {"agent_function_failed": True, "agent_excluded_from_training": True}
    metadata.update(agent.decode_result(result("Submitted", 1)))
    assert metadata["agent_function_failed"] is False
    assert metadata["agent_excluded_from_training"] is False


@pytest.mark.parametrize("arguments", [{"command": "printf ok"}, '{"command": "printf ok"}'])
def test_native_arguments_and_original_history_preserved(arguments):
    body = completion()
    body["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"] = arguments
    original = json.loads(json.dumps(body))
    episode, _ = run_loop([body], [("ok", 0)], agent.AgentConfig(max_turns=1))
    call = original["choices"][0]["message"]["tool_calls"][0]
    assert episode.messages[2]["tool_calls"] == [call]
    assert episode.messages[3]["role"] == "tool"
    assert episode.messages[3]["tool_call_id"] == call["id"]
    assert body == original


@pytest.mark.parametrize(
    "arguments",
    [
        None,
        [],
        {},
        {"command": ""},
        {"command": " "},
        {"command": 1},
        {"command": "x\x00"},
        {"command": "true", "extra": 1},
        '{"command":',
        "[]",
    ],
)
def test_invalid_arguments_are_model_format_failure_not_infra(arguments):
    body = completion()
    body["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"] = arguments
    episode, queries = run_loop([body])
    assert episode.stop_reason == "FormatLimit" and episode.format_errors == 1
    assert len(queries) == 1 and not episode.tool_events
    assert episode.messages[-1]["role"] == "tool"


@pytest.mark.parametrize("field,value", [("name", "python"), ("name", None)])
def test_unknown_function_cannot_execute(field, value):
    body = completion()
    body["choices"][0]["message"]["tool_calls"][0]["function"][field] = value
    episode, _ = run_loop([body])
    assert episode.stop_reason == "FormatLimit" and not episode.tool_events


@pytest.mark.parametrize("count", [1, 2, 20])
def test_multiple_calls_execute_and_return_individual_results(count):
    body = multi_completion([f"printf {i}" for i in range(count)])
    calls = body["choices"][0]["message"]["tool_calls"]
    executed = []
    queries = []

    async def query(messages):
        queries.append(json.loads(json.dumps(messages)))
        return body if len(queries) == 1 else completion("echo " + ref.SUBMIT_MARKER)

    async def execute(command, timeout):
        executed.append(command)
        return (ref.SUBMIT_MARKER, 0) if command.startswith("echo") else (command, 0)

    episode = asyncio.run(agent.run_episode("fix", query, execute))
    assert episode.stop_reason == "Submitted" and episode.format_errors == 0 and episode.turns == 2
    assert executed[:-1] == [f"printf {i}" for i in range(count)]
    assert [m["tool_call_id"] for m in queries[1][3:]] == [c["id"] for c in calls]
    assert all(m["role"] == "tool" for m in queries[1][3:])
    assert [m["content"] for m in queries[1][3:]] == [ref.render_observation(0, f"printf {i}", 6000) for i in range(count)]


def multi_completion(commands):
    assert isinstance(commands, list) and commands
    body = completion()
    body["choices"][0]["message"]["tool_calls"] = [completion(command)["choices"][0]["message"]["tool_calls"][0] for command in commands]
    body["choices"][0]["finish_reason"] = "tool_calls"
    return body


@pytest.mark.parametrize("submit_at", [0, 1, 2])
def test_submission_skips_later_calls_but_closes_each_tool_id(submit_at):
    body = multi_completion(["first", "second", "third"])
    outputs = [("ok", 0)] * submit_at + [(ref.SUBMIT_MARKER, 0)]
    episode, _ = run_loop([body], outputs)
    assert episode.stop_reason == "Submitted" and episode.turns == 1
    results = episode.messages[3:]
    assert len(results) == 3
    assert [m["tool_call_id"] for m in results] == [c["id"] for c in body["choices"][0]["message"]["tool_calls"]]
    assert all("Not executed" in m["content"] for m in results[submit_at + 1 :])
    assert episode.metrics()["agent_swe_tool_submit_count"] == 1


def test_blocked_and_failed_calls_do_not_block_other_commands():
    body = multi_completion(["git log", "false", "true"])
    episode, _ = run_loop([body], [("failed", 1), ("ok", 0)], agent.AgentConfig(max_turns=1))
    assert episode.turns == 1 and episode.blocked_actions == 1 and episode.format_errors == 0
    assert len(episode.messages[3:]) == 3
    assert "git is disabled" in episode.messages[3]["content"]
    assert "failed" in episode.messages[4]["content"] and "ok" in episode.messages[5]["content"]
    assert episode.metrics()["agent_swe_tool_returncode_nonzero_count"] == 1


def test_invalid_later_arguments_prevent_partial_batch_execution():
    body = multi_completion(["true", "false"])
    body["choices"][0]["message"]["tool_calls"][1]["function"]["arguments"] = "{"
    episode, _ = run_loop([body])
    assert episode.stop_reason == "FormatLimit" and not episode.tool_events
    assert len(episode.messages[3:]) == 2
    assert all("No calls in this batch were executed" in m["content"] for m in episode.messages[3:])


def test_transport_failure_in_batch_propagates_without_later_execution():
    body = multi_completion(["true", "false", "echo never"])
    with pytest.raises(httpx.ConnectError):
        run_loop([body], [("ok", 0), httpx.ConnectError("lost sandbox")])


@pytest.mark.parametrize("calls", ["bad", [None], [{"id": ""}], [{"id": "x"}, {"id": "x"}]])
def test_broken_serving_envelopes_fail_fast(calls):
    body = completion()
    body["choices"][0]["message"]["tool_calls"] = calls
    with pytest.raises(ValueError):
        run_loop([body])


def test_fenced_prose_is_never_an_action():
    episode, _ = run_loop([completion(content="```bash\necho forbidden\n```")] * 3)
    assert episode.stop_reason == "FormatLimit" and not episode.tool_events


def test_content_is_not_executed_when_native_call_exists():
    body = completion("true")
    body["choices"][0]["message"]["content"] = "```bash\nexit 99\n```"
    executed = []

    async def query(messages):
        return body

    async def execute(command, timeout):
        executed.append(command)
        return "ok", 0

    asyncio.run(agent.run_episode("fix", query, execute, config=agent.AgentConfig(max_turns=1)))
    assert executed == ["true"]


@pytest.mark.parametrize("raw", [0, 1])
@pytest.mark.parametrize("evaluation", [False, True])
def test_length_zero_reward_keeps_verifier_result_and_trainable_sample(raw, evaluation):
    s = sample(raw=raw, evaluation=evaluation)
    s.metadata.update(agent.decode_result(result("LengthTruncated", raw)))
    s.status = SimpleNamespace(name="TRUNCATED")
    score = score_sample(s)
    assert score == {"score": 0.0, "acc": False, "pred": "LengthTruncated", "judge_raw": "single_call_length"}
    assert not s.remove_sample and not s.metadata["agent_excluded_from_training"]
    assert s.metadata["reward"] == s.metadata["agent_lightning_raw_reward"] == raw
    assert s.metadata["agent_last_finish_reason"] == "length"
    assert s.metadata["agent_single_call_length_penalized"]
    assert s.metadata["agent_lightning_shaping_penalty"] == 0


def test_length_cannot_override_infrastructure_exclusion():
    s = sample()
    s.metadata.update(agent.decode_result(result("LengthTruncated", 0, infra=True)))
    score = score_sample(s)
    assert s.remove_sample and score["judge_raw"] == "excluded_from_training"


@pytest.mark.parametrize("raw", [0, 1])
@pytest.mark.parametrize("evaluation", [False, True])
def test_format_zero_reward_keeps_raw_verifier_result(raw, evaluation):
    s = sample(raw=raw, evaluation=evaluation)
    s.metadata.update(agent.decode_result(result("FormatLimit", raw)))
    score = score_sample(s)
    assert score == {"score": 0.0, "acc": False, "pred": "FormatLimit", "judge_raw": "format_limit"}
    assert s.metadata["reward"] == s.metadata["agent_lightning_raw_reward"] == raw
    assert s.metadata["agent_lightning_shaping_penalty"] == raw
    assert s.metadata["agent_format_failure_penalized"] and not s.remove_sample


def test_format_failure_does_not_override_infra_exclusion():
    s = sample()
    s.metadata.update(agent.decode_result(result("FormatLimit", 1, infra=True)))
    assert score_sample(s)["judge_raw"] == "excluded_from_training" and s.remove_sample


def test_length_requires_consistent_finish_metadata():
    s = sample()
    s.metadata["exit_status"] = "LengthTruncated"
    with pytest.raises(ValueError, match="requires agent_last_finish_reason"):
        score_sample(s)


def test_actual_input_overflow_http_ends_episode_and_keeps_verifier_reward():
    model_calls = []
    body = completion("true")

    def handler(request):
        model_calls.append(json.loads(request.content))
        if len(model_calls) == 1:
            return httpx.Response(200, json=body)
        return httpx.Response(
            400,
            json={"error": {"message": "The input (82000 tokens) is longer than the model's context length (81920 tokens)."}},
        )

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:

            async def query(messages):
                return await agent.query_model(client, "http://model/chat/completions", "test", messages, agent.AgentConfig())

            async def execute(command, timeout):
                assert command == "true"
                return "ok", 0

            return await agent.run_episode("fix", query, execute)

    episode = asyncio.run(run())
    assert episode.stop_reason == "ContextLimit" and episode.turns == 1
    assert len(model_calls) == 2 and all(c["max_tokens"] == 12288 for c in model_calls)
    response = result(episode.stop_reason, 1)
    response["agent_metrics"] = episode.metrics()
    s = sample(evaluation=True)
    s.metadata.update(agent.decode_result(response))
    assert score_sample(s)["score"] == 1 and not s.remove_sample
    assert s.metadata["agent_context_limit_hit"] is True
