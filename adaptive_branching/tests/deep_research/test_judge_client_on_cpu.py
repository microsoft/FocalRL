from __future__ import annotations

import pytest

from adaptive_branching.src.deep_research.judge_client import (
    JudgeClient,
    _chat_sampling_params,
    _parse_json_object,
    _validation_retry_prompt,
)


def test_parse_json_object_accepts_plain_and_fenced_json():
    assert _parse_json_object('{"state":"correct"}') == {"state": "correct"}
    assert _parse_json_object('```json\n{"state":"wrong"}\n```') == {"state": "wrong"}


def test_parse_json_object_rejects_truncated_json_instead_of_repairing_it():
    with pytest.raises(ValueError, match="invalid JSON"):
        _parse_json_object('{"state":"correct"')


def test_validation_retry_prompt_reports_error_without_silent_repair():
    prompt = _validation_retry_prompt("original task", ValueError("selected_turn must be in [1, 8], got 9"))
    assert prompt.startswith("original task\n")
    assert "selected_turn must be in [1, 8], got 9" in prompt
    assert "corrected JSON object only" in prompt
    with pytest.raises(ValueError, match="original_user"):
        _validation_retry_prompt("", ValueError("bad"))


def test_chat_sampling_params_accepts_qwen_recommended_thinking_values():
    config = {
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": 20,
        "min_p": 0.0,
        "presence_penalty": 0.0,
        "repetition_penalty": 1.0,
    }
    assert _chat_sampling_params(config) == config


@pytest.mark.parametrize(
    ("key", "bad_value"),
    [
        ("temperature", 0),
        ("top_p", 1.1),
        ("top_k", 0),
        ("min_p", -0.1),
        ("presence_penalty", 2.1),
        ("repetition_penalty", 0),
    ],
)
def test_chat_sampling_params_rejects_invalid_values(key, bad_value):
    config = {
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": 20,
        "min_p": 0.0,
        "presence_penalty": 0.0,
        "repetition_penalty": 1.0,
    }
    config[key] = bad_value
    with pytest.raises(ValueError, match=key):
        _chat_sampling_params(config)


@pytest.mark.asyncio
@pytest.mark.parametrize("enable_thinking", [None, True, False])
async def test_chat_completion_passes_explicit_sampling_parameters(monkeypatch, enable_thinking):
    class FakeCompletions:
        def __init__(self):
            self.kwargs = None

        async def create(self, **kwargs):
            self.kwargs = kwargs
            message = type("Message", (), {"content": '{"ok":true}'})()
            choice = type("Choice", (), {"finish_reason": "stop", "message": message})()
            return type("Response", (), {"choices": [choice]})()

    completions = FakeCompletions()
    config = {
        "model": "Qwen/Qwen3.5-4B",
        "max_tokens": 32768,
        "reasoning_effort": "",
        "base_url": "http://unused.invalid/v1",
        "api_key": "test-only",
        "timeout": 1,
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": 20,
        "min_p": 0.0,
        "presence_penalty": 0.0,
        "repetition_penalty": 1.0,
    }
    if enable_thinking is not None:
        config["enable_thinking"] = enable_thinking
    monkeypatch.setattr("adaptive_branching.src.deep_research.judge_client.judge_settings", lambda _: config)
    fake_client = type(
        "FakeClient",
        (),
        {"chat": type("Chat", (), {"completions": completions})()},
    )()
    monkeypatch.setattr("openai.AsyncOpenAI", lambda **_: fake_client)
    client = JudgeClient(config_section="locator")
    assert client.enable_thinking is enable_thinking

    assert await client._complete("system", "user") == '{"ok":true}'
    expected_extra = {"top_k": 20, "min_p": 0.0, "repetition_penalty": 1.0}
    if enable_thinking is not None:
        expected_extra["chat_template_kwargs"] = {"enable_thinking": enable_thinking}
    assert completions.kwargs == {
        "model": "Qwen/Qwen3.5-4B",
        "messages": [{"role": "system", "content": "system"}, {"role": "user", "content": "user"}],
        "max_tokens": 32768,
        "temperature": 0.6,
        "top_p": 0.95,
        "presence_penalty": 0.0,
        "extra_body": expected_extra,
    }


@pytest.mark.parametrize("value", ["false", "true", "", 0, 1, None, [], {}])
def test_client_rejects_malformed_thinking_before_network(monkeypatch, value):
    config = {"model": "Qwen/Qwen3.5-9B", "max_tokens": 128, "enable_thinking": value}
    monkeypatch.setattr("adaptive_branching.src.deep_research.judge_client.judge_settings", lambda _: config)
    with pytest.raises(ValueError, match="enable_thinking"):
        JudgeClient(config_section="locator")


@pytest.mark.parametrize("value", [True, False])
def test_client_rejects_thinking_for_responses_api(monkeypatch, value):
    config = {"model": "gpt-test", "max_tokens": 128, "enable_thinking": value}
    monkeypatch.setattr("adaptive_branching.src.deep_research.judge_client.judge_settings", lambda _: config)
    with pytest.raises(ValueError, match="Responses API"):
        JudgeClient(config_section="locator")


@pytest.mark.asyncio
async def test_complete_json_retries_validation_with_explicit_feedback(monkeypatch, capsys):
    client = object.__new__(JudgeClient)
    client.max_retries = 2
    prompts = []
    responses = iter(['{"selected_turn":9}', '{"selected_turn":8}'])

    async def fake_complete(_system, user):
        prompts.append(user)
        return next(responses)

    monkeypatch.setattr(client, "_complete", fake_complete)

    def validate(row):
        selected_turn = row["selected_turn"]
        if not 1 <= selected_turn <= 8:
            raise ValueError("selected_turn must be in [1, 8], got 9")
        return selected_turn

    result = await client.complete_json(
        "system",
        "choose within range",
        required_keys=("selected_turn",),
        tag="locator_test",
        validate=validate,
    )
    assert result == 8
    assert prompts[0] == "choose within range"
    assert "selected_turn must be in [1, 8], got 9" in prompts[1]
    assert "RETRY tag=locator_test next_attempt=2/2" in capsys.readouterr().out
