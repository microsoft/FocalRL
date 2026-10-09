from __future__ import annotations

import asyncio

import pytest

from adaptive_branching.src.deep_research import value_cliff_locator as event_locator


def _messages(*contents: str):
    return [{"role": "user", "content": "question"}, *({"role": "assistant", "content": text} for text in contents)]


def test_value_cliff_prefix_is_immediately_before_selected_action():
    messages = [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "first"},
        {"role": "assistant", "content": "selected", "tool_calls": []},
        {"role": "tool", "content": "observation"},
        {"role": "assistant", "content": "later"},
    ]
    before, after = event_locator.build_value_cliff_prefixes(messages, 2)
    assert before[-1]["content"] == "first"
    assert after[-2]["content"] == "selected"
    assert after[-1]["content"] == "observation"
    with pytest.raises(ValueError, match="within"):
        event_locator.build_value_cliff_prefixes(messages, 3)


def test_locator_prompt_balances_failed_and_successful_trace_budget():
    prompt, metadata = event_locator.build_value_cliff_locator_prompt(
        question="q",
        ground_truth="answer",
        failed_messages=_messages("f" * 1000, "final"),
        successful_messages=_messages("s" * 100),
        trace_max_chars=500,
    )
    assert "Select exactly one Failed Turn in [1, 1]" in prompt
    assert "failed_trace_truncated=true" in prompt
    assert "successful_trace_truncated=false" in prompt
    assert metadata["failed_assistant_turns"] == 2


def test_locator_uses_current_schema_and_retained_horizon(monkeypatch):
    monkeypatch.setenv("AB_EVENT_HIDDEN_MAX_TURNS", "5")
    captured = {}

    class FakeClient:
        async def complete_json(self, system, prompt, *, required_keys, validate, **kwargs):
            captured.update(system=system, prompt=prompt, required_keys=required_keys)
            return validate(
                {
                    "selected_turn": 2,
                    "value_drop_reason": "Turn 2 adopted unsupported X.",
                    "recovery_rubric": {
                        "avoid_error": "Stop relying on unsupported X.",
                        "redirect": "Test evidence that distinguishes X from Y.",
                    },
                }
            )

    result = asyncio.run(
        event_locator.locate_value_cliff(
            FakeClient(),
            question="original question",
            ground_truth="Y",
            positive_messages=_messages("verified Y"),
            negative_messages=_messages("background", "adopt X", "wrong final"),
            tag="test",
            trace_max_chars=10_000,
        )
    )
    assert result.event["event_turn"] == 2
    assert result.event["recovery_rubric"]["redirect"].startswith("Test evidence")
    assert result.diagnostics == {
        "policy": "value_cliff_rubric_locator_v6_20260822",
        "stage_status": "selected",
    }
    assert "exactly five new assistant turns" in captured["system"]
    assert tuple(captured["required_keys"]) == ("selected_turn", "value_drop_reason", "recovery_rubric")


@pytest.mark.parametrize("value", ["", "0", "-1", "x"])
def test_retained_hidden_max_turns_fails_fast(monkeypatch, value):
    monkeypatch.setenv("AB_EVENT_HIDDEN_MAX_TURNS", value)
    with pytest.raises(ValueError, match="AB_EVENT_HIDDEN_MAX_TURNS"):
        event_locator.value_cliff_local_horizon()
