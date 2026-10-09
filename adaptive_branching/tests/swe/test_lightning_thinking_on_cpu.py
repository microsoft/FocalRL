"""Thinking/action separation, history retention, and training-token regressions."""

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from jinja2 import Environment

from adaptive_branching.src.swe import agent, lightning_reference as reference, thinking_protocol as protocol


def completion(content="", reasoning="Inspect the implementation.", finish="tool_calls", command="printf ok"):
    return {
        "choices": [
            {
                "message": {
                    "content": content,
                    "reasoning_content": reasoning,
                    "tool_calls": (
                        []
                        if finish == "length"
                        else [
                            {
                                "id": "call-1",
                                "type": "function",
                                "function": {
                                    "name": "bash",
                                    "arguments": {"command": command},
                                },
                            }
                        ]
                    ),
                },
                "finish_reason": finish,
            }
        ],
        "usage": {"prompt_tokens": 100, "completion_tokens": 30},
    }


def test_runtime_prompts_remove_thought_but_preserve_restrictions():
    assert "THOUGHT" not in protocol.SYSTEM_PROMPT + protocol.INSTANCE_PROMPT
    assert protocol.INSTANCE_PROMPT.count("{problem_statement}") == 1
    for rule in ["## Important Boundaries", "## Recommended Workflow", "## Environment Details", "## Submission"]:
        assert rule in protocol.INSTANCE_PROMPT
    assert "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT" in protocol.INSTANCE_PROMPT
    assert "Do NOT modify test-harness or config files" in protocol.INSTANCE_PROMPT


@pytest.mark.parametrize("reasoning", ["", None, "inspect\n", "```bash\nexit 99\n```"])
def test_reasoning_is_kept_in_history_and_never_executed(reasoning):
    replies = iter([completion(reasoning=reasoning), completion(command="echo " + reference.SUBMIT_MARKER)])
    calls, executed = [], []

    async def query(messages):
        calls.append(json.loads(json.dumps(messages)))
        return next(replies)

    async def execute(command, timeout):
        executed.append(command)
        return ("ok" if len(executed) == 1 else reference.SUBMIT_MARKER), 0

    episode = asyncio.run(agent.run_episode("fix", query, execute))
    assert episode.stop_reason == "Submitted" and episode.output_tokens == 60
    assert executed == ["printf ok", "echo " + reference.SUBMIT_MARKER]
    assert calls[1][2]["reasoning_content"] == reasoning
    assert episode.messages[2]["reasoning_content"] == reasoning
    assert calls[1][2]["content"] == ""
    assert calls[1][3]["role"] == "tool"
    assert calls[1][3]["tool_call_id"] == "call-1"


@pytest.mark.parametrize("content", ["", None])
def test_thinking_exhausts_budget_without_action_and_stops(content):
    responses = iter([completion(content, "unfinished reasoning", "length")])
    calls = []

    async def query(messages):
        calls.append(messages)
        return next(responses)

    async def execute(command, timeout):
        raise AssertionError("length output must not execute a tool")

    episode = asyncio.run(agent.run_episode("fix", query, execute))
    assert episode.turns == 1 and episode.format_errors == 0 and len(calls) == 1
    assert episode.stop_reason == "LengthTruncated"
    assert episode.messages[-1]["reasoning_content"] == "unfinished reasoning"


@pytest.mark.parametrize("reasoning", [[], {}, 1, True])
def test_malformed_reasoning_fails_fast(reasoning):
    with pytest.raises(ValueError, match="reasoning_content"):
        agent.validate_completion(completion(reasoning=reasoning))


def test_unparsed_think_block_cannot_supply_an_action():
    with pytest.raises(ValueError, match="reasoning parser"):
        agent.validate_completion(completion("<think>```bash\nexit 99\n```</think>"))


@pytest.mark.parametrize("n_actions,finish", [(0, "stop"), (0, "length"), (2, "stop"), (100, "length")])
def test_format_repair_does_not_reintroduce_thought(n_actions, finish):
    assert "THOUGHT" not in protocol.format_error_message(n_actions, finish)
    assert ("output token limit" if finish == "length" else "native bash") in protocol.format_error_message(
        n_actions, finish
    )


@pytest.mark.parametrize("n_actions,finish", [(-1, "stop"), (True, "stop"), (0, "invalid")])
def test_format_repair_rejects_invalid_inputs(n_actions, finish):
    with pytest.raises(ValueError):
        protocol.format_error_message(n_actions, finish)


def test_format_feedback_requires_native_tools():
    assert "one or more native bash tool calls" in protocol.format_error_message(0, "tool_calls")


def test_qwen_template_preserves_reasoning_after_user_observation():
    template_path = (
        Path(__file__).resolve().parents[3] / "miles/utils/chat_template_utils/templates/qwen3.5_fixed.jinja"
    )
    template = Environment().from_string(template_path.read_text())
    messages = [
        {"role": "system", "content": protocol.SYSTEM_PROMPT},
        {"role": "user", "content": "fix"},
        {"role": "assistant", **completion(reasoning="UNIQUE_REASONING")["choices"][0]["message"]},
        {"role": "tool", "tool_call_id": "call-1", "content": "command succeeded"},
    ]
    rendered = template.render(
        messages=messages, tools=protocol.TOOLS, add_generation_prompt=True, enable_thinking=True, clear_thinking=False
    )
    assert "<think>\nUNIQUE_REASONING\n</think>" in rendered
    assert "<tool_call>\n<function=bash>\n<parameter=command>\nprintf ok" in rendered
    assert "<tool_response>\ncommand succeeded" in rendered
    assert rendered.endswith("<|im_start|>assistant\n<think>\n")


def test_reasoning_and_action_tokens_both_receive_loss():
    from miles.rollout.generate_utils.openai_endpoint_utils import compute_samples_from_openai_records
    from miles.rollout.session.session_types import SessionRecord
    from miles.utils.types import Sample

    raw = "<think>\nanalysis\n</think>\n\n<tool_call>\n<function=bash>\n<parameter=command>\nprintf ok\n</parameter>\n</function>\n</tool_call>"
    ids = list(raw.encode())
    response = completion()
    response["choices"][0]["meta_info"] = {"output_token_logprobs": [[-0.1, t] for t in ids]}
    record = SessionRecord(
        timestamp=0,
        method="POST",
        path="/v1/chat/completions",
        status_code=200,
        request={"input_ids": [1, 2]},
        response=response,
    )
    tokenizer = SimpleNamespace(decode=lambda tokens: bytes(tokens).decode())
    samples = compute_samples_from_openai_records(
        SimpleNamespace(), Sample(index=0, prompt="fix"), [record], tokenizer
    )
    assert len(samples) == 1
    sample = samples[0]
    assert sample.tokens == [1, 2] + ids
    assert sample.response == raw and sample.response_length == len(ids)
    assert sample.loss_mask == [1] * len(ids)
    assert sample.rollout_log_probs == [-0.1] * len(ids)


def test_multi_tool_history_and_training_masks_roundtrip():
    from miles.rollout.generate_utils.openai_endpoint_utils import compute_samples_from_openai_records
    from miles.rollout.generate_utils.sample_utils import merge_samples
    from miles.rollout.session.session_types import SessionRecord
    from miles.utils.types import Sample

    template_path = (
        Path(__file__).resolve().parents[3] / "miles/utils/chat_template_utils/templates/qwen3.5_fixed.jinja"
    )
    template = Environment().from_string(template_path.read_text())
    records, outputs, prompts = [], [], []

    def render(messages, generation):
        assert isinstance(messages, list) and messages
        return template.render(
            messages=messages,
            tools=protocol.TOOLS,
            add_generation_prompt=generation,
            enable_thinking=True,
            clear_thinking=False,
        )

    async def query(messages):
        prompt = render(messages, True)
        body = completion(command="printf first" if not records else "echo " + reference.SUBMIT_MARKER)
        response = body["choices"][0]["message"]
        if not records:
            second = completion(command="printf second")["choices"][0]["message"]["tool_calls"][0]
            second["id"] = "call-2"
            response["tool_calls"].append(second)
        response["role"] = "assistant"
        full = render(messages + [response], False)
        assert full.startswith(prompt)
        raw_output = full[len(prompt) :]
        if records:
            assert prompt.startswith(prompts[0] + outputs[0])
            assert prompt.count("<tool_response>") == 2
        prompts.append(prompt)
        outputs.append(raw_output)
        ids = list(raw_output.encode())
        body["choices"][0]["meta_info"] = {"output_token_logprobs": [[-0.1, t] for t in ids]}
        records.append(
            SessionRecord(
                timestamp=0,
                method="POST",
                path="/v1/chat/completions",
                status_code=200,
                request={"input_ids": list(prompt.encode())},
                response=body,
            )
        )
        return body

    async def execute(command, timeout):
        return (reference.SUBMIT_MARKER if command.startswith("echo") else command), 0

    episode = asyncio.run(agent.run_episode("fix", query, execute))
    assert episode.stop_reason == "Submitted" and episode.turns == 2
    tokenizer = SimpleNamespace(decode=lambda ids: bytes(ids).decode())
    accumulated = list((prompts[-1] + outputs[-1]).encode())
    samples = compute_samples_from_openai_records(
        SimpleNamespace(), Sample(index=0, prompt="fix"), records, tokenizer, accumulated_token_ids=accumulated
    )
    merged = merge_samples(samples, tokenizer)
    response_ids = merged.tokens[len(prompts[0].encode()) :]
    train_ids = [t for t, m in zip(response_ids, merged.loss_mask, strict=True) if m]
    masked_ids = [t for t, m in zip(response_ids, merged.loss_mask, strict=True) if not m]
    assert bytes(train_ids).decode() == "".join(outputs)
    assert bytes(masked_ids).decode().count("<tool_response>") == 2
    assert len(merged.loss_mask) == merged.response_length == len(merged.rollout_log_probs)
