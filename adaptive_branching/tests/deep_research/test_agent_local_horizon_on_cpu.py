"""CPU-only tests for the short local-process agent horizon."""

from __future__ import annotations

import os
from types import SimpleNamespace

import pytest

# agent.py constructs its tool adapters at import time; CPU tests replace all
# network calls but still need syntactically valid adapter configuration.
os.environ.setdefault("LLM_API_BASE", "http://localhost:4141/v1")
os.environ.setdefault("LLM_API_KEY", "test-key")
os.environ.setdefault("BROWSER_LLM_MODEL", "test-model")

from adaptive_branching.src.deep_research import agent


def _response(message, finish_reason):
    payload = {
        "choices": [{"message": message, "finish_reason": finish_reason}],
        "usage": {"completion_tokens": 1},
    }
    return SimpleNamespace(status_code=200, text="", json=lambda: payload)


def _error_response(status_code, text):
    return SimpleNamespace(status_code=status_code, text=text)


def _tool_message(turn):
    return {
        "role": "assistant",
        "content": f"research turn {turn}",
        "tool_calls": [
            {
                "id": f"call-{turn}",
                "type": "function",
                "function": {"name": "search", "arguments": '{"query":"q"}'},
            }
        ],
    }


@pytest.mark.parametrize("policy", ["random", "entropy_delta_max"])
@pytest.mark.asyncio
async def test_branch_capture_preserves_history_and_original_entropy(monkeypatch, policy):
    from adaptive_branching.tests.deep_research.test_branch_sampling_on_cpu import meta

    monkeypatch.setenv("AB_BRANCH_SELECTION", policy)
    monkeypatch.setenv("AB_LOCAL_REWARD_MODE", "terminal")
    requests = []

    async def post(client, url, payload, **kwargs):
        requests.append(payload.copy())
        turn = len(requests)
        message = _tool_message(turn) if turn <= 3 else {"role": "assistant", "content": "answer"}
        data = {"choices": [{"message": message, "finish_reason": "tool_calls" if turn <= 3 else "stop",
                            "meta_info": meta()}], "usage": {"completion_tokens": 1}}
        return SimpleNamespace(status_code=200, json=lambda: data)

    async def tools(calls):
        return ([{"role": "tool", "tool_call_id": calls[0]["id"], "content": calls[0]["id"]}], {})

    monkeypatch.setattr(agent, "_post_chat_completion", post)
    monkeypatch.setattr(agent, "_execute_tool_calls", tools)
    result = await agent.run("http://unused/v1", "question", metadata={
        "agent_max_turns": 5, "agent_history_mode": "keep5", "agent_keep_tool_results": 1,
        "ab_entropy_vocab_size": 100})
    original_tools = [m["content"] for m in result["ab_branch_history"] if m["role"] == "tool"]
    assert original_tools == ["call-1", "call-2", "call-3"]
    assert any(m.get("content") == "Tool result is omitted to save tokens." for m in result["messages"])
    if policy == "entropy_delta_max":
        assert all(p["top_logprobs"] == 10 and p["return_meta_info"] for p in requests)
        assert set(result["ab_branch_entropies"]) == {"1", "2", "3", "4"}
    else:
        assert all("top_logprobs" not in p for p in requests)
        assert result["ab_branch_entropies"] == {}


@pytest.mark.asyncio
async def test_missing_entropy_fails_but_terminal_child_does_not_request_it(monkeypatch):
    from adaptive_branching.src.deep_research.branch_sampling import BranchDataError

    monkeypatch.setenv("AB_BRANCH_SELECTION", "entropy_delta_max")
    monkeypatch.setenv("AB_LOCAL_REWARD_MODE", "terminal")
    requests = []

    async def post(client, url, payload, **kwargs):
        requests.append(payload)
        return _response({"role": "assistant", "content": "answer"}, "stop")

    monkeypatch.setattr(agent, "_post_chat_completion", post)
    with pytest.raises(BranchDataError, match="turn 1"):
        await agent.run("http://unused/v1", "question", metadata={"ab_entropy_vocab_size": 100})
    result = await agent.run("http://unused/v1", "question", metadata={"ab_local_rollout": True})
    assert "top_logprobs" not in requests[-1]
    assert "ab_branch_history" not in result


def test_responses_adapter_preserves_function_call_history_and_usage():
    payload = agent._responses_payload(
        {
            "model": "gpt-5.6-sol",
            "messages": [
                {"role": "system", "content": "system"},
                {"role": "user", "content": "question"},
                _tool_message(1),
                {"role": "tool", "tool_call_id": "call-1", "content": "result"},
            ],
            "tools": agent.TOOLS,
            "tool_choice": "auto",
            "max_tokens": 100,
            "reasoning_effort": "none",
            "chat_template_kwargs": {"clear_thinking": False},
            "top_k": 40,
            "logprobs": False,
        }
    )

    assert payload["reasoning"] == {"effort": "none"}
    assert payload["max_output_tokens"] == 100
    assert payload["input"][-2] == {
        "type": "function_call",
        "call_id": "call-1",
        "name": "search",
        "arguments": '{"query":"q"}',
    }
    assert payload["input"][-1] == {
        "type": "function_call_output",
        "call_id": "call-1",
        "output": "result",
    }
    assert payload["tools"][0]["name"] == "search"
    assert "chat_template_kwargs" not in payload
    assert "top_k" not in payload
    assert "logprobs" not in payload

    normalized = agent._chat_data_from_responses(
        {
            "status": "completed",
            "output": [
                {
                    "type": "function_call",
                    "call_id": "call-2",
                    "name": "fetch_url",
                    "arguments": '{"url":["https://example.com"],"purpose":"p"}',
                }
            ],
            "usage": {"input_tokens": 10, "output_tokens": 4, "total_tokens": 14},
        }
    )

    assert normalized["choices"][0]["finish_reason"] == "tool_calls"
    assert normalized["choices"][0]["message"]["tool_calls"][0]["id"] == "call-2"
    assert normalized["usage"]["completion_tokens"] == 4

    next_payload = agent._responses_payload(
        {
            "model": "gpt-5.6-sol",
            "messages": [
                normalized["choices"][0]["message"],
                {"role": "tool", "tool_call_id": "call-2", "content": "fetched"},
            ],
        }
    )
    assert next_payload["input"][0] == normalized["choices"][0]["message"]["_responses_output"][0]
    assert next_payload["input"][1]["type"] == "function_call_output"


@pytest.mark.asyncio
async def test_responses_agent_runs_tool_call_then_returns_answer(monkeypatch):
    calls = []
    responses = [
        {
            "status": "completed",
            "output": [
                {
                    "type": "function_call",
                    "call_id": "call-1",
                    "name": "search",
                    "arguments": '{"query":["test"]}',
                }
            ],
            "usage": {"input_tokens": 10, "output_tokens": 2, "total_tokens": 12},
        },
        {
            "status": "completed",
            "output": [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "final answer"}],
                }
            ],
            "usage": {"input_tokens": 15, "output_tokens": 3, "total_tokens": 18},
        },
    ]

    async def fake_post(client, url, payload, headers=None):
        calls.append((url, payload))
        data = responses.pop(0)
        return SimpleNamespace(status_code=200, text="", json=lambda: data)

    async def fake_execute(tool_calls):
        assert tool_calls[0]["function"]["name"] == "search"
        return ([{"role": "tool", "tool_call_id": "call-1", "content": "search result"}], {})

    monkeypatch.setattr(agent, "_post_chat_completion", fake_post)
    monkeypatch.setattr(agent, "_execute_tool_calls", fake_execute)

    result = await agent.run(
        "http://localhost:4141/v1",
        "question",
        request_kwargs={"model": "gpt-5.6-sol", "max_tokens": 100, "reasoning_effort": "none"},
        metadata={"agent_api_mode": "responses", "agent_max_turns": 5, "agent_return_messages": True},
    )

    assert [url for url, _ in calls] == [
        "http://localhost:4141/v1/responses",
        "http://localhost:4141/v1/responses",
    ]
    assert calls[1][1]["input"][-1] == {
        "type": "function_call_output",
        "call_id": "call-1",
        "output": "search result",
    }
    assert result["agent_turns"] == 2
    assert result["agent_final_answer"] == "final answer"
    assert result["agent_session_tokens"] == 18


@pytest.mark.asyncio
async def test_responses_agent_captures_complete_heterogeneous_raw_responses(monkeypatch):
    first = {
        "id": "response-1",
        "status": "completed",
        "copilot_usage": {"total_nano_aiu": 7},
        "output": [
            {"type": "reasoning", "content": [], "encrypted_content": "opaque"},
            {
                "type": "function_call",
                "call_id": "call-1",
                "name": "search",
                "arguments": '{"query":["test"]}',
            },
        ],
        "usage": {"input_tokens": 10, "output_tokens": 4, "total_tokens": 14},
    }
    second = {
        "id": "response-2",
        "status": "completed",
        "output": [
            {"type": "reasoning", "content": [], "encrypted_content": "opaque-2"},
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "final answer"}],
            },
        ],
        "usage": {"input_tokens": 15, "output_tokens": 6, "total_tokens": 21},
    }
    responses = [first, second]

    async def fake_post(client, url, payload, headers=None):
        del client, url, payload, headers
        data = responses.pop(0)
        return SimpleNamespace(status_code=200, text="", json=lambda: data)

    async def fake_execute(tool_calls):
        assert tool_calls[0]["function"]["name"] == "search"
        return ([{"role": "tool", "tool_call_id": "call-1", "content": "search result"}], {})

    monkeypatch.setattr(agent, "_post_chat_completion", fake_post)
    monkeypatch.setattr(agent, "_execute_tool_calls", fake_execute)

    result = await agent.run(
        "http://localhost:4141/v1",
        "question",
        request_kwargs={"model": "gpt-5.6-sol", "max_tokens": 100, "reasoning_effort": "medium"},
        metadata={
            "agent_api_mode": "responses",
            "agent_max_turns": 5,
            "agent_return_messages": True,
            "agent_capture_raw_responses": True,
        },
    )

    trace = result["agent_raw_responses"]
    assert [entry["request_index"] for entry in trace] == [1, 2]
    assert [entry["agent_turn"] for entry in trace] == [1, 2]
    assert all(entry["request_kind"] == "agent_turn" for entry in trace)
    assert trace[0]["response"] == first
    assert trace[0]["response"]["output"][0]["type"] == "reasoning"
    assert trace[0]["response"]["output"][1]["type"] == "function_call"
    assert trace[1]["response"] == second
    assert trace[1]["response"]["output"][1]["content"][0]["text"] == "final answer"


@pytest.mark.asyncio
async def test_raw_response_capture_rejects_chat_api_before_network(monkeypatch):
    async def fail_post(*_args, **_kwargs):
        pytest.fail("chat request must not start with raw Responses capture enabled")

    monkeypatch.setattr(agent, "_post_chat_completion", fail_post)

    with pytest.raises(ValueError, match="requires agent_api_mode='responses'"):
        await agent.run(
            "http://localhost:30000/v1",
            "question",
            metadata={"agent_api_mode": "chat", "agent_capture_raw_responses": True},
        )


@pytest.mark.parametrize("finish_reason", ["abort", "length"])
@pytest.mark.asyncio
async def test_incomplete_finish_reason_stops_before_tool_execution(monkeypatch, finish_reason):
    calls = []

    async def fake_post(client, url, payload, headers=None):
        calls.append(payload)
        return _response(_tool_message(1), finish_reason)

    async def fail_execute(tool_calls):
        pytest.fail(f"tool calls must not execute after finish_reason={finish_reason}: {tool_calls}")

    monkeypatch.setattr(agent, "_post_chat_completion", fake_post)
    monkeypatch.setattr(agent, "_execute_tool_calls", fail_execute)

    result = await agent.run(
        "http://localhost:30000/v1",
        "question",
        metadata={"agent_max_turns": 5, "agent_return_messages": True},
    )

    assert len(calls) == 1
    assert result["agent_turns"] == 1
    assert result["agent_last_finish_reason"] == finish_reason
    assert result["agent_finished"] is False
    assert result["agent_tool_call_count"] == 0
    assert result["messages"][-1]["role"] == "assistant"


@pytest.mark.asyncio
async def test_local_horizon_stops_without_forced_final(monkeypatch):
    calls = []
    responses = [_response(_tool_message(turn), "tool_calls") for turn in range(1, 6)]

    async def fake_post(client, url, payload, headers=None):
        calls.append(payload)
        return responses.pop(0)

    async def fake_execute(tool_calls):
        call_id = tool_calls[0]["id"]
        return ([{"role": "tool", "tool_call_id": call_id, "content": "result"}], {})

    monkeypatch.setattr(agent, "_post_chat_completion", fake_post)
    monkeypatch.setattr(agent, "_execute_tool_calls", fake_execute)

    result = await agent.run(
        "http://localhost:30000/v1",
        "question",
        metadata={
            "agent_max_turns": 5,
            "agent_force_final_on_max_turns": False,
            "agent_return_messages": True,
        },
    )

    assert len(calls) == 5
    assert result["agent_turns"] == 5
    assert result["agent_local_horizon_hit"] is True
    assert result["agent_max_turns_hit"] is False
    assert result["agent_forced_final_answer"] is False
    assert result["agent_last_finish_reason"] == "local_horizon"
    assert result["messages"][-1]["role"] == "tool"


@pytest.mark.asyncio
async def test_natural_end_precedes_local_horizon(monkeypatch):
    calls = []

    async def fake_post(client, url, payload, headers=None):
        calls.append(payload)
        return _response({"role": "assistant", "content": "natural answer"}, "stop")

    monkeypatch.setattr(agent, "_post_chat_completion", fake_post)

    result = await agent.run(
        "http://localhost:30000/v1",
        "question",
        metadata={
            "agent_max_turns": 5,
            "agent_force_final_on_max_turns": False,
            "agent_return_messages": True,
        },
    )

    assert len(calls) == 1
    assert result["agent_turns"] == 1
    assert result["agent_local_horizon_hit"] is False
    assert result["agent_last_finish_reason"] == "stop"
    assert result["agent_final_answer"] == "natural answer"
    assert result["agent_tool_unit_count"] == 0
    assert result["agent_tool_unit_success_count"] == 0
    assert result["agent_tool_unit_success_rate"] == 1.0


@pytest.mark.parametrize(
    "content",
    [
        "",
        "<｜DSML｜unknown_broken_boundary>",
        '<｜DSML｜invoke name="search">',
        "<｜DSML｜tool_code>\n<search>query</search>",
        '<search search_query="query" />',
        'Tool:search\nArguments: {"query": "query"}',
        "web_search: query",
    ],
)
def test_textual_tool_call_detector_covers_deepseek_variants(content):
    assert agent._looks_like_unparsed_tool_call(content)


@pytest.mark.asyncio
async def test_opt_in_textual_tool_call_retry_resamples_without_polluting_history(monkeypatch):
    calls = []
    responses = [
        _response({"role": "assistant", "content": "<｜DSML｜tool_code><search>bad 1</search>"}, "stop"),
        _response({"role": "assistant", "content": '<search search_query="bad 2" />'}, "stop"),
        _response({"role": "assistant", "content": "Tool:search\nArguments: {}"}, "stop"),
        _response({"role": "assistant", "content": "natural answer"}, "stop"),
    ]

    async def fake_post(client, url, payload, headers=None):
        calls.append(payload)
        return responses.pop(0)

    monkeypatch.setattr(agent, "_post_chat_completion", fake_post)

    result = await agent.run(
        "http://localhost:30000/v1",
        "question",
        metadata={
            "agent_max_turns": 5,
            "agent_return_messages": True,
            "agent_tool_call_text_retries": 3,
        },
    )

    assert len(calls) == 4
    assert all(call["messages"] == calls[0]["messages"] for call in calls)
    assert result["agent_tool_call_text_retry_count"] == 3
    assert result.get("agent_tool_call_text_retry_exhausted", False) is False
    assert result["agent_turns"] == 1
    assert result["agent_final_answer"] == "natural answer"
    assert [message["content"] for message in result["messages"] if message["role"] == "assistant"] == [
        "natural answer"
    ]


@pytest.mark.asyncio
async def test_textual_tool_call_retry_is_disabled_by_default(monkeypatch):
    calls = []

    async def fake_post(client, url, payload, headers=None):
        calls.append(payload)
        return _response({"role": "assistant", "content": "<tool_call>query</tool_call>"}, "stop")

    monkeypatch.setattr(agent, "_post_chat_completion", fake_post)

    result = await agent.run(
        "http://localhost:30000/v1",
        "question",
        metadata={"agent_max_turns": 5, "agent_return_messages": True},
    )

    assert len(calls) == 1
    assert "agent_tool_call_text_retry_count" not in result
    assert "agent_tool_call_text_retry_exhausted" not in result
    assert result["agent_final_answer_tool_call_like"] is True


@pytest.mark.asyncio
async def test_local_context_guard_stops_without_forced_final(monkeypatch):
    calls = []

    async def fake_post(client, url, payload, headers=None):
        calls.append(payload)
        response = _response(_tool_message(1), "tool_calls")
        response.json = lambda: {
            "choices": [{"message": _tool_message(1), "finish_reason": "tool_calls"}],
            "usage": {"total_tokens": 1},
        }
        return response

    async def fake_execute(tool_calls):
        call_id = tool_calls[0]["id"]
        return ([{"role": "tool", "tool_call_id": call_id, "content": "result"}], {})

    monkeypatch.setenv("AGENT_CONTEXT_RESERVE_TOKENS", "10")
    monkeypatch.setattr(agent, "_post_chat_completion", fake_post)
    monkeypatch.setattr(agent, "_execute_tool_calls", fake_execute)

    result = await agent.run(
        "http://localhost:30000/v1",
        "question",
        metadata={
            "agent_max_turns": 5,
            "agent_force_final_on_context_reserve": False,
            "agent_return_messages": True,
            "max_seq_len": 10,
        },
    )

    assert len(calls) == 1
    assert result["agent_turns"] == 1
    assert result["agent_context_reserve_hit"] is True
    assert result["agent_local_horizon_hit"] is True
    assert result["agent_forced_final_answer"] is False
    assert result["agent_last_finish_reason"] == "local_context_reserve"


@pytest.mark.asyncio
async def test_context_length_400_recovers_with_capped_forced_final(monkeypatch):
    calls = []
    normal = _response(_tool_message(1), "tool_calls")
    normal.json = lambda: {
        "choices": [{"message": _tool_message(1), "finish_reason": "tool_calls"}],
        "usage": {"total_tokens": 100},
    }
    context_error = _error_response(
        400,
        (
            "Requested token count exceeds the model's maximum context length of 2,000 tokens. "
            "You requested a total of 33,568 tokens: 800 tokens from the input messages and "
            "32,768 tokens for the completion."
        ),
    )
    forced = _response({"role": "assistant", "content": "recovered final"}, "stop")
    responses = [normal, context_error, forced]

    async def fake_post(client, url, payload, headers=None):
        calls.append(payload)
        return responses.pop(0)

    async def fake_execute(tool_calls):
        call_id = tool_calls[0]["id"]
        return ([{"role": "tool", "tool_call_id": call_id, "content": "large result"}], {})

    monkeypatch.setenv("AGENT_CONTEXT_RESERVE_TOKENS", "500")
    monkeypatch.setattr(agent, "_post_chat_completion", fake_post)
    monkeypatch.setattr(agent, "_execute_tool_calls", fake_execute)

    result = await agent.run(
        "http://localhost:30000/v1",
        "question",
        request_kwargs={
            "max_tokens": 32768,
            "chat_template_kwargs": {"clear_thinking": False},
        },
        metadata={"agent_max_turns": 5, "agent_return_messages": True, "max_seq_len": 2000},
    )

    assert len(calls) == 3
    assert all(call["chat_template_kwargs"] == {"clear_thinking": False} for call in calls)
    assert calls[-1]["max_tokens"] == 176
    assert result["agent_context_length_recovery"] is True
    assert result["agent_context_length_recovery_input_tokens"] == 800
    assert result["agent_context_length_recovery_server_max_tokens"] == 2000
    assert result["agent_context_reserve_hit"] is True
    assert result["agent_forced_final_answer"] is True
    assert result["agent_forced_final_answer_reason"] == "context_reserve"
    assert result["agent_final_answer"] == "recovered final"


@pytest.mark.asyncio
async def test_default_max_turn_behavior_still_forces_final_answer(monkeypatch):
    calls = []
    responses = [
        _response(_tool_message(1), "tool_calls"),
        _response({"role": "assistant", "content": "forced final"}, "stop"),
    ]

    async def fake_post(client, url, payload, headers=None):
        calls.append(payload)
        return responses.pop(0)

    async def fake_execute(tool_calls):
        call_id = tool_calls[0]["id"]
        return ([{"role": "tool", "tool_call_id": call_id, "content": "result"}], {})

    monkeypatch.setattr(agent, "_post_chat_completion", fake_post)
    monkeypatch.setattr(agent, "_execute_tool_calls", fake_execute)

    result = await agent.run(
        "http://localhost:30000/v1",
        "question",
        request_kwargs={"chat_template_kwargs": {"clear_thinking": False}},
        metadata={"agent_max_turns": 1, "agent_return_messages": True},
    )

    assert len(calls) == 2
    assert all(call["chat_template_kwargs"] == {"clear_thinking": False} for call in calls)
    assert result["agent_max_turns_hit"] is True
    assert result["agent_local_horizon_hit"] is False
    assert result["agent_forced_final_answer"] is True
    assert result["agent_forced_final_answer_reason"] == "max_turns"
    assert result["agent_final_answer"] == "forced final"


@pytest.mark.asyncio
async def test_keep5_continuation_counts_total_turns_and_strips_old_results(monkeypatch):
    import copy

    prefix = [{"role": "system", "content": "system"}, {"role": "user", "content": "question"}]
    for turn in range(1, 3):
        prefix.extend(
            [_tool_message(turn), {"role": "tool", "tool_call_id": f"call-{turn}", "content": f"result {turn}"}]
        )
    calls = []

    async def fake_post(_client, _url, payload, **_kwargs):
        calls.append(copy.deepcopy(payload))
        return _response(_tool_message(3) if len(calls) == 1 else {"role": "assistant", "content": "answer"}, "stop")

    async def fake_execute(_calls):
        return [{"role": "tool", "tool_call_id": "call-3", "content": "new result"}], {}

    monkeypatch.setattr(agent, "_post_chat_completion", fake_post)
    monkeypatch.setattr(agent, "_execute_tool_calls", fake_execute)
    result = await agent.run(
        base_url="http://unused",
        prompt=prefix,
        request_kwargs={"max_tokens": 20},
        metadata={
            "agent_max_turns": 3,
            "agent_turn_offset": 2,
            "agent_history_mode": "keep5",
            "agent_keep_tool_results": 1,
        },
    )
    assert len(calls) == 2  # one remaining action, then forced final
    assert calls[0]["messages"] == prefix
    assert "tools" not in calls[1]
    assert result["agent_turns"] == 3
    assert result["agent_max_turns_hit"]
    assert result["agent_continuation_prefix_turns"] == 2
    assert result["messages"][3]["content"] == agent.TOOL_RESULT_OMITTED_PLACEHOLDER
    assert result["messages"][5]["content"] == agent.TOOL_RESULT_OMITTED_PLACEHOLDER


@pytest.mark.asyncio
async def test_continuation_restores_context_guard_before_first_call(monkeypatch):
    calls = []

    async def fake_post(_client, _url, payload, **_kwargs):
        calls.append(payload)
        return _response({"role": "assistant", "content": "answer"}, "stop")

    monkeypatch.setattr(agent, "_post_chat_completion", fake_post)
    monkeypatch.setenv("AGENT_CONTEXT_RESERVE_TOKENS", "2000")
    prefix = [
        {"role": "user", "content": "question"},
        _tool_message(1),
        {"role": "tool", "tool_call_id": "call-1", "content": "result"},
    ]
    result = await agent.run(
        base_url="http://unused",
        prompt=prefix,
        metadata={
            "agent_max_turns": 3,
            "agent_turn_offset": 1,
            "agent_initial_session_tokens": 8000,
            "max_seq_len": 10000,
        },
    )
    assert len(calls) == 1 and "tools" not in calls[0]
    assert result["agent_context_reserve_hit"]
    assert result["agent_turns"] == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("offset", [-1, True, 3, 1])
async def test_continuation_rejects_invalid_offset(offset):
    with pytest.raises(ValueError):
        await agent.run(
            base_url="http://unused",
            prompt="question",
            metadata={
                "agent_max_turns": 3,
                "agent_turn_offset": offset,
            },
        )
