import pytest

from adaptive_branching.src.deep_research.judge_token_budget import (
    JudgeTokenBudget,
    local_tokenizer,
    shrink_observations,
)


class Tokenizer:
    def apply_chat_template(self, messages, **kwargs):
        assert kwargs == dict(tokenize=True, add_generation_prompt=True, enable_thinking=True, return_dict=False)
        return [1] * (sum(len(m["content"]) for m in messages) + 20)


def budget(limit=1000):
    return JudgeTokenBudget(Tokenizer(), max_input_tokens=limit, context_length=20000, max_output_tokens=100)


def test_fitting_preserves_short_inputs_and_exact_boundary():
    assert budget(24).fit("sys", "u") == ("u", 24, 24)


def test_shortening_preserves_turns_actions_and_rubric():
    protected = "## Question\nQ\n<recovery_rubric>EXACT</recovery_rubric>\n"
    action = "### Failed Turn 17\n[assistant] ACTION\n[tool_call:search] ARGUMENTS\n"
    tail = "\n### Successful Turn 8\n[assistant] FINAL\n</local_continuation>"
    text = protected + action + "[following_observation:search] " + "z" * 4000 + tail
    result, before, after = budget().fit("sys", text)
    assert before > 1000 and after <= 1000
    assert protected in result and action in result and tail in result
    assert "JUDGE_TOKEN_BUDGET" in result


def test_reasoning_can_shrink_but_assistant_content_cannot():
    assert len(shrink_observations("[reasoning] " + "x" * 1000, 0.5)) < 1000
    text = "[assistant] " + "x" * 2000
    assert shrink_observations(text, 0.5) == text
    with pytest.raises(ValueError, match="protected content"):
        budget().fit("sys", text)


@pytest.mark.parametrize("system,user", [("", "u"), ("s", ""), (None, "u")])
def test_empty_prompts_fail(system, user):
    with pytest.raises(ValueError, match="nonempty"):
        budget().fit(system, user)


@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_invalid_limits_fail(value):
    with pytest.raises(ValueError, match="positive integer"):
        budget(value)


def test_context_overflow_and_missing_tokenizer_fail():
    with pytest.raises(ValueError, match="exceeds"):
        JudgeTokenBudget(Tokenizer(), max_input_tokens=100, context_length=110, max_output_tokens=11)
    with pytest.raises(ValueError, match="local checkpoint"):
        local_tokenizer("/nonexistent/local-judge-tokenizer")


@pytest.mark.parametrize("text,ratio", [("", 0.5), ("text", 0), ("text", 1), (None, 0.5)])
def test_bad_shrink_inputs_fail(text, ratio):
    with pytest.raises(ValueError):
        shrink_observations(text, ratio)


@pytest.mark.asyncio
async def test_client_checks_budget_before_network_and_after_retry(monkeypatch):
    from adaptive_branching.src.deep_research.judge_client import JudgeClient

    client = object.__new__(JudgeClient)
    client.token_budget = budget(2000)
    client.max_retries = 2
    seen = []

    async def complete(system, user):
        assert client.token_budget.count(system, user) <= 2000
        seen.append(user)
        return "{}" if len(seen) == 1 else '{"ok":true}'

    monkeypatch.setattr(client, "_complete", complete)
    result = await client.complete_json("sys", "[reasoning] " + "x" * 4000, ("ok",), "test", lambda x: x["ok"])
    assert result is True and len(seen) == 2
    assert "Required correction" in seen[1]
