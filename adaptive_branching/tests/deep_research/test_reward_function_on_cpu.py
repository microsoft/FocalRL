"""CPU-only tests for the LLM-judge reward function.

These tests mock AsyncOpenAI so they never touch the network or a GPU. They
verify the reasoning_effort env wiring and the empty-choices fallback, which are
the parts that silently broke when a gpt-5.x / reasoning model was configured.

Run: python -m pytest adaptive_branching/tests/test_reward_function_on_cpu.py -q
"""

from __future__ import annotations

import sys
import types

import pytest

from adaptive_branching.src.deep_research import reward_function


class _Sample:
    def __init__(self, prompt, response, label, metadata=None, status="completed", response_length=None, tokens=None):
        self.prompt = prompt
        self.response = response
        self.label = label
        self.metadata = metadata or {}
        self.remove_sample = False
        self.status = status
        self.response_length = response_length
        self.tokens = tokens


def _make_response(content, *, choices_empty=False):
    """Build an object accepted by both mocked OpenAI response APIs."""
    if choices_empty:
        return types.SimpleNamespace(choices=[], output_text="", status="completed", incomplete_details=None)
    msg = types.SimpleNamespace(content=content)
    choice = types.SimpleNamespace(message=msg, finish_reason="stop")
    return types.SimpleNamespace(choices=[choice], output_text=content, status="completed", incomplete_details=None)


class _FakeCompletions:
    def __init__(self, script):
        # script: list of responses (or Exceptions) to return per call
        self._script = list(script)
        self.calls = []  # records kwargs of each create() call

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        item = self._script.pop(0) if self._script else _make_response("Correct")
        if isinstance(item, Exception):
            raise item
        return item


class _FakeClient:
    def __init__(self, script):
        self.chat = types.SimpleNamespace(completions=_FakeCompletions(script))
        self.responses = _FakeCompletions(script)


@pytest.fixture
def patched_openai(monkeypatch):
    """Patch AsyncOpenAI (imported lazily inside reward_func) and return a holder
    whose `.client` is set after reward_func instantiates it."""
    holder = {}

    def factory_maker(script):
        def factory(*args, **kwargs):
            holder["client"] = _FakeClient(script)
            holder["init_kwargs"] = kwargs
            return holder["client"]

        return factory

    fake_openai_module = types.ModuleType("openai")

    def install(script):
        fake_openai_module.AsyncOpenAI = factory_maker(script)
        monkeypatch.setitem(sys.modules, "openai", fake_openai_module)

    holder["install"] = install
    return holder


def _set_env(monkeypatch, **overrides):
    monkeypatch.setenv("LLM_JUDGE_URL", "http://localhost:4141/v1")
    monkeypatch.setenv("LLM_JUDGE_KEY", "test-only-unused")
    monkeypatch.setenv("LLM_JUDGE_MODEL", "claude-opus-4.6")
    for k in (
        "LLM_JUDGE_REASONING_EFFORT",
        "LLM_JUDGE_MAX_TOKENS",
        "FORCE_EXCLUDE",
        "AGENT_EXCLUDE_POSITIVE_REPEATED_SEARCH",
        "AGENT_REPEAT_SEARCH_MIN_RATE",
        "AGENT_EXCLUDE_POSITIVE_REPEATED_FETCH_URL",
        "AGENT_REPEAT_FETCH_URL_MIN_RATE",
        "AGENT_SOFT_OVERLONG_ENABLE",
        "AGENT_SOFT_OVERLONG_START_TOKENS",
        "AGENT_SOFT_OVERLONG_HARD_TOKENS",
        "AGENT_SOFT_OVERLONG_PENALTY_SCALE",
        "AGENT_SOFT_OVERLONG_POSITIVE_ONLY",
        "AB_LOCAL_ROLLOUT_ENABLE",
        "AB_LOCAL_REWARD_MODE",
        "AB_EVENT_HIDDEN_MAX_TURNS",
    ):
        monkeypatch.delenv(k, raising=False)
    for k, v in overrides.items():
        if v is None:
            monkeypatch.delenv(k, raising=False)
        else:
            monkeypatch.setenv(k, v)


@pytest.mark.asyncio
async def test_outcome_settings_come_from_judge_yaml(monkeypatch, patched_openai):
    _set_env(monkeypatch)
    patched_openai["install"]([_make_response("correct: yes")])
    sample = _Sample("What is 2+2?", "<answer>four</answer>", "4")
    out = await reward_function.reward_func(None, sample)
    assert out["acc"] is True and out["score"] == 1.0, out
    create_kwargs = patched_openai["client"].responses.calls[0]
    assert create_kwargs["model"] == "gpt-5.6-sol"
    assert create_kwargs["reasoning"] == {"effort": "medium"}
    assert create_kwargs["max_output_tokens"] == 16384


@pytest.mark.asyncio
async def test_reasoning_effort_passed_when_set(monkeypatch, patched_openai):
    _set_env(monkeypatch, LLM_JUDGE_REASONING_EFFORT="low")
    patched_openai["install"]([_make_response("correct: yes")])
    sample = _Sample("What is 2+2?", "<answer>four</answer>", "4")
    out = await reward_function.reward_func(None, sample)
    assert out["acc"] is True, out
    create_kwargs = patched_openai["client"].responses.calls[0]
    assert create_kwargs["reasoning"] == {"effort": "medium"}, create_kwargs


@pytest.mark.asyncio
async def test_legacy_reasoning_env_does_not_override_judge_yaml(monkeypatch, patched_openai):
    _set_env(monkeypatch, LLM_JUDGE_REASONING_EFFORT="none")
    patched_openai["install"]([_make_response("correct: yes")])
    sample = _Sample("q", "<answer>a</answer>", "a")
    out = await reward_function.reward_func(None, sample)
    assert out["acc"] is True
    assert patched_openai["client"].responses.calls[0]["reasoning"] == {"effort": "medium"}


@pytest.mark.asyncio
async def test_incorrect_verdict(monkeypatch, patched_openai):
    _set_env(monkeypatch, LLM_JUDGE_REASONING_EFFORT="low")
    patched_openai["install"]([_make_response("correct: no")])
    sample = _Sample("Speed of light?", "<answer>3e8 m/s</answer>", "299792458 m/s")
    out = await reward_function.reward_func(None, sample)
    assert out["acc"] is False and out["score"] == 0.0, out


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("content", "expected"),
    [
        ('{"correct": "yes", "confidence": 100}', True),
        ('```json\n{"correct": false, "confidence": 100}\n```', False),
    ],
)
async def test_json_verdict_is_parsed(monkeypatch, patched_openai, content, expected):
    _set_env(monkeypatch, LLM_JUDGE_REASONING_EFFORT="low")
    patched_openai["install"]([_make_response(content)])
    sample = _Sample("q", "candidate", "gold")

    out = await reward_function.reward_func(None, sample)

    assert out["acc"] is expected
    assert out["score"] == float(expected)
    assert out.get("judge_error") is None


@pytest.mark.asyncio
async def test_natural_wrong_answer_stays_trainable(monkeypatch, patched_openai):
    _set_env(monkeypatch, LLM_JUDGE_REASONING_EFFORT="low")
    patched_openai["install"]([_make_response("correct: no")])
    sample = _Sample("q", "wrong", "right", {"agent_last_finish_reason": "stop"})
    out = await reward_function.reward_func(None, sample)
    assert out["acc"] is False and out["score"] == 0.0, out
    assert sample.remove_sample is False
    assert sample.metadata["agent_excluded_from_training"] is False


@pytest.mark.asyncio
async def test_single_call_length_is_trainable_zero_and_skips_outcome_judge(monkeypatch, patched_openai):
    _set_env(monkeypatch, LLM_JUDGE_REASONING_EFFORT="low")
    patched_openai["install"]([_make_response("correct: yes")])
    length_sample = _Sample(
        "q1",
        "partial output",
        "gold1",
        {"agent_last_finish_reason": "length"},
        response_length=22000,
    )
    natural_sample = _Sample(
        "q2",
        "complete answer",
        "gold2",
        {"agent_last_finish_reason": "stop"},
    )

    results = await reward_function.reward_func(None, [length_sample, natural_sample])

    assert len(patched_openai["client"].responses.calls) == 1
    assert results[0] == {
        "score": 0.0,
        "acc": False,
        "pred": "partial output",
        "judge_raw": "single_call_length",
    }
    assert length_sample.remove_sample is False
    assert length_sample.metadata["agent_excluded_from_training"] is False
    assert length_sample.metadata["agent_single_call_length_penalized"] is True
    assert results[1]["score"] == 1.0
    assert natural_sample.remove_sample is False


@pytest.mark.asyncio
async def test_length_with_context_reserve_remains_excluded(monkeypatch, patched_openai):
    _set_env(monkeypatch, LLM_JUDGE_REASONING_EFFORT="low", FORCE_EXCLUDE="none")
    patched_openai["install"]([_make_response("correct: no")])
    sample = _Sample(
        "q",
        "partial output",
        "gold",
        {
            "agent_last_finish_reason": "length",
            "agent_context_reserve_hit": True,
        },
        response_length=22000,
    )

    result = await reward_function.reward_func(None, sample)

    assert result["score"] == 0.0
    assert sample.remove_sample is True
    assert sample.metadata["agent_excluded_from_training"] is True
    assert sample.metadata["agent_single_call_length_penalized"] is False


@pytest.mark.asyncio
async def test_forced_wrong_answer_is_excluded(monkeypatch, patched_openai):
    _set_env(monkeypatch, LLM_JUDGE_REASONING_EFFORT="low")
    patched_openai["install"]([_make_response("correct: no")])
    sample = _Sample(
        "q",
        "wrong",
        "right",
        {
            "agent_forced_final_answer": True,
            "agent_forced_final_answer_reason": "context_reserve",
            "agent_last_finish_reason": "stop",
        },
    )
    out = await reward_function.reward_func(None, sample)
    assert out["acc"] is False and out["score"] == 0.0, out
    assert sample.remove_sample is True
    assert sample.metadata["agent_excluded_from_training"] is True


@pytest.mark.asyncio
async def test_force_exclude_none_keeps_forced_wrong_trainable(monkeypatch, patched_openai):
    _set_env(monkeypatch, LLM_JUDGE_REASONING_EFFORT="low", FORCE_EXCLUDE="None")
    patched_openai["install"]([_make_response("correct: no")])
    sample = _Sample(
        "q",
        "wrong",
        "right",
        {
            "agent_forced_final_answer": True,
            "agent_forced_final_answer_reason": "context_reserve",
            "agent_last_finish_reason": "stop",
        },
    )
    out = await reward_function.reward_func(None, sample)
    assert out["acc"] is False and out["score"] == 0.0, out
    assert sample.remove_sample is False
    assert sample.metadata["agent_excluded_from_training"] is False


@pytest.mark.asyncio
async def test_forced_correct_answer_stays_trainable(monkeypatch, patched_openai):
    _set_env(monkeypatch, LLM_JUDGE_REASONING_EFFORT="low")
    patched_openai["install"]([_make_response("correct: yes")])
    sample = _Sample(
        "q",
        "right",
        "right",
        {
            "agent_forced_final_answer": True,
            "agent_forced_final_answer_reason": "max_turns",
            "agent_last_finish_reason": "stop",
        },
    )
    out = await reward_function.reward_func(None, sample)
    assert out["acc"] is True and out["score"] == 1.0, out
    assert sample.remove_sample is False
    assert sample.metadata["agent_excluded_from_training"] is False


@pytest.mark.asyncio
async def test_force_exclude_all_excludes_forced_correct(monkeypatch, patched_openai):
    _set_env(monkeypatch, LLM_JUDGE_REASONING_EFFORT="low", FORCE_EXCLUDE="ALL")
    patched_openai["install"]([_make_response("correct: yes")])
    sample = _Sample(
        "q",
        "right",
        "right",
        {
            "agent_forced_final_answer": True,
            "agent_forced_final_answer_reason": "context_reserve",
            "agent_last_finish_reason": "stop",
        },
    )
    out = await reward_function.reward_func(None, sample)
    assert out["acc"] is True and out["score"] == 1.0, out
    assert sample.remove_sample is True
    assert sample.metadata["agent_excluded_from_training"] is True


@pytest.mark.asyncio
async def test_invalid_force_exclude_mode_raises(monkeypatch, patched_openai):
    _set_env(monkeypatch, LLM_JUDGE_REASONING_EFFORT="low", FORCE_EXCLUDE="sometimes")
    patched_openai["install"]([_make_response("correct: yes")])
    sample = _Sample("q", "right", "right")
    with pytest.raises(ValueError, match="FORCE_EXCLUDE"):
        await reward_function.reward_func(None, sample)


@pytest.mark.asyncio
async def test_positive_repeated_search_filter_default_off(monkeypatch, patched_openai):
    _set_env(monkeypatch, LLM_JUDGE_REASONING_EFFORT="low")
    patched_openai["install"]([_make_response("correct: yes")])
    sample = _Sample(
        "q",
        "right",
        "right",
        {
            "agent_search_query_count": 4,
            "agent_search_query_repeat_count": 2,
            "agent_last_finish_reason": "stop",
        },
    )
    out = await reward_function.reward_func(None, sample)
    assert out["acc"] is True and out["score"] == 1.0, out
    assert sample.remove_sample is False
    assert sample.metadata["agent_excluded_from_training"] is False
    assert sample.metadata["agent_search_query_repeat_rate"] == 0.5
    assert sample.metadata["agent_positive_repeated_search_excluded"] is False


@pytest.mark.asyncio
async def test_positive_repeated_search_filter_excludes_when_enabled(monkeypatch, patched_openai):
    _set_env(
        monkeypatch,
        LLM_JUDGE_REASONING_EFFORT="low",
        AGENT_EXCLUDE_POSITIVE_REPEATED_SEARCH="1",
        AGENT_REPEAT_SEARCH_MIN_RATE="0.5",
    )
    patched_openai["install"]([_make_response("correct: yes")])
    sample = _Sample(
        "q",
        "right",
        "right",
        {
            "agent_search_query_count": 4,
            "agent_search_query_repeat_count": 2,
            "agent_last_finish_reason": "stop",
        },
    )
    out = await reward_function.reward_func(None, sample)
    assert out["acc"] is True and out["score"] == 1.0, out
    assert sample.remove_sample is True
    assert sample.metadata["agent_excluded_from_training"] is True
    assert sample.metadata["agent_positive_repeated_search_excluded"] is True


@pytest.mark.asyncio
async def test_positive_repeated_search_filter_default_rate_is_02(monkeypatch, patched_openai):
    _set_env(
        monkeypatch,
        LLM_JUDGE_REASONING_EFFORT="low",
        AGENT_EXCLUDE_POSITIVE_REPEATED_SEARCH="1",
    )
    patched_openai["install"]([_make_response("correct: yes")])
    sample = _Sample(
        "q",
        "right",
        "right",
        {
            "agent_search_query_count": 4,
            "agent_search_query_repeat_count": 1,
            "agent_last_finish_reason": "stop",
        },
    )
    out = await reward_function.reward_func(None, sample)
    assert out["acc"] is True and out["score"] == 1.0, out
    assert sample.metadata["agent_search_query_repeat_rate"] == 0.25
    assert sample.remove_sample is True
    assert sample.metadata["agent_positive_repeated_search_excluded"] is True


@pytest.mark.asyncio
async def test_repeated_search_negative_answer_stays_trainable(monkeypatch, patched_openai):
    _set_env(
        monkeypatch,
        LLM_JUDGE_REASONING_EFFORT="low",
        AGENT_EXCLUDE_POSITIVE_REPEATED_SEARCH="1",
        AGENT_REPEAT_SEARCH_MIN_RATE="0.5",
    )
    patched_openai["install"]([_make_response("correct: no")])
    sample = _Sample(
        "q",
        "wrong",
        "right",
        {
            "agent_search_query_count": 4,
            "agent_search_query_repeat_count": 2,
            "agent_last_finish_reason": "stop",
        },
    )
    out = await reward_function.reward_func(None, sample)
    assert out["acc"] is False and out["score"] == 0.0, out
    assert sample.remove_sample is False
    assert sample.metadata["agent_excluded_from_training"] is False
    assert sample.metadata["agent_positive_repeated_search_excluded"] is False


@pytest.mark.asyncio
async def test_positive_repeated_fetch_url_filter_default_off(monkeypatch, patched_openai):
    _set_env(monkeypatch, LLM_JUDGE_REASONING_EFFORT="low")
    patched_openai["install"]([_make_response("correct: yes")])
    sample = _Sample(
        "q",
        "right",
        "right",
        {
            "agent_fetch_url_count": 16,
            "agent_fetch_url_repeat_count": 5,
            "agent_last_finish_reason": "stop",
        },
    )
    out = await reward_function.reward_func(None, sample)
    assert out["acc"] is True and out["score"] == 1.0, out
    assert sample.remove_sample is False
    assert sample.metadata["agent_excluded_from_training"] is False
    assert sample.metadata["agent_fetch_url_repeat_rate"] == 5 / 16
    assert sample.metadata["agent_positive_repeated_fetch_url_excluded"] is False


@pytest.mark.asyncio
async def test_positive_repeated_fetch_url_filter_excludes_when_enabled(monkeypatch, patched_openai):
    _set_env(
        monkeypatch,
        LLM_JUDGE_REASONING_EFFORT="low",
        AGENT_EXCLUDE_POSITIVE_REPEATED_FETCH_URL="1",
        AGENT_REPEAT_FETCH_URL_MIN_RATE="0.25",
    )
    patched_openai["install"]([_make_response("correct: yes")])
    sample = _Sample(
        "q",
        "right",
        "right",
        {
            "agent_fetch_url_count": 16,
            "agent_fetch_url_repeat_count": 4,
            "agent_last_finish_reason": "stop",
        },
    )
    out = await reward_function.reward_func(None, sample)
    assert out["acc"] is True and out["score"] == 1.0, out
    assert sample.remove_sample is True
    assert sample.metadata["agent_excluded_from_training"] is True
    assert sample.metadata["agent_fetch_url_repeat_rate"] == 0.25
    assert sample.metadata["agent_positive_repeated_fetch_url_excluded"] is True


@pytest.mark.asyncio
async def test_repeated_fetch_url_negative_answer_stays_trainable(monkeypatch, patched_openai):
    _set_env(
        monkeypatch,
        LLM_JUDGE_REASONING_EFFORT="low",
        AGENT_EXCLUDE_POSITIVE_REPEATED_FETCH_URL="1",
        AGENT_REPEAT_FETCH_URL_MIN_RATE="0.25",
    )
    patched_openai["install"]([_make_response("correct: no")])
    sample = _Sample(
        "q",
        "wrong",
        "right",
        {
            "agent_fetch_url_count": 16,
            "agent_fetch_url_repeat_count": 5,
            "agent_last_finish_reason": "stop",
        },
    )
    out = await reward_function.reward_func(None, sample)
    assert out["acc"] is False and out["score"] == 0.0, out
    assert sample.remove_sample is False
    assert sample.metadata["agent_excluded_from_training"] is False
    assert sample.metadata["agent_positive_repeated_fetch_url_excluded"] is False


@pytest.mark.asyncio
async def test_soft_overlong_penalizes_correct_answer(monkeypatch, patched_openai):
    _set_env(
        monkeypatch,
        LLM_JUDGE_REASONING_EFFORT="low",
        AGENT_SOFT_OVERLONG_ENABLE="1",
        AGENT_SOFT_OVERLONG_START_TOKENS="100",
        AGENT_SOFT_OVERLONG_HARD_TOKENS="200",
        AGENT_SOFT_OVERLONG_PENALTY_SCALE="0.5",
        AGENT_SOFT_OVERLONG_POSITIVE_ONLY="1",
    )
    patched_openai["install"]([_make_response("correct: yes")])
    sample = _Sample(
        "q",
        "right",
        "right",
        {"agent_session_tokens": 150, "agent_last_finish_reason": "stop"},
    )
    out = await reward_function.reward_func(None, sample)
    assert out["acc"] is True
    assert out["score"] == pytest.approx(0.75)
    assert sample.remove_sample is False
    assert sample.metadata["agent_soft_overlong_penalty"] == pytest.approx(-0.25)
    assert sample.metadata["agent_soft_overlong_fraction"] == pytest.approx(0.5)
    assert sample.metadata["agent_soft_overlong_applied"] is True


@pytest.mark.asyncio
async def test_soft_overlong_clamps_at_hard_without_filtering(monkeypatch, patched_openai):
    _set_env(
        monkeypatch,
        LLM_JUDGE_REASONING_EFFORT="low",
        AGENT_SOFT_OVERLONG_ENABLE="1",
        AGENT_SOFT_OVERLONG_START_TOKENS="100",
        AGENT_SOFT_OVERLONG_HARD_TOKENS="200",
        AGENT_SOFT_OVERLONG_PENALTY_SCALE="0.75",
        AGENT_SOFT_OVERLONG_POSITIVE_ONLY="1",
    )
    patched_openai["install"]([_make_response("correct: yes")])
    sample = _Sample(
        "q",
        "right",
        "right",
        {"agent_session_tokens": 250, "agent_last_finish_reason": "stop"},
    )
    out = await reward_function.reward_func(None, sample)
    assert out["acc"] is True
    assert out["score"] == pytest.approx(0.25)
    assert sample.remove_sample is False
    assert sample.metadata["agent_soft_overlong_penalty"] == pytest.approx(-0.75)
    assert sample.metadata["agent_soft_overlong_fraction"] == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_soft_overlong_positive_only_keeps_wrong_score_unchanged(monkeypatch, patched_openai):
    _set_env(
        monkeypatch,
        LLM_JUDGE_REASONING_EFFORT="low",
        AGENT_SOFT_OVERLONG_ENABLE="1",
        AGENT_SOFT_OVERLONG_START_TOKENS="100",
        AGENT_SOFT_OVERLONG_HARD_TOKENS="200",
        AGENT_SOFT_OVERLONG_PENALTY_SCALE="0.75",
        AGENT_SOFT_OVERLONG_POSITIVE_ONLY="1",
    )
    patched_openai["install"]([_make_response("correct: no")])
    sample = _Sample("q", "wrong", "right", {"agent_last_finish_reason": "stop"}, response_length=250)
    out = await reward_function.reward_func(None, sample)
    assert out["acc"] is False
    assert out["score"] == 0.0
    assert sample.remove_sample is False
    assert sample.metadata["agent_soft_overlong_tokens"] == 250
    assert sample.metadata["agent_soft_overlong_penalty"] == 0.0
    assert sample.metadata["agent_soft_overlong_applied"] is False


@pytest.mark.asyncio
async def test_soft_overlong_config_is_ignored_when_disabled(monkeypatch, patched_openai):
    _set_env(
        monkeypatch,
        LLM_JUDGE_REASONING_EFFORT="low",
        AGENT_SOFT_OVERLONG_ENABLE="0",
        AGENT_SOFT_OVERLONG_START_TOKENS="100",
        AGENT_SOFT_OVERLONG_HARD_TOKENS="200",
        AGENT_SOFT_OVERLONG_PENALTY_SCALE="0.75",
        AGENT_SOFT_OVERLONG_POSITIVE_ONLY="1",
    )
    patched_openai["install"]([_make_response("correct: yes")])
    sample = _Sample("q", "right", "right", {"agent_session_tokens": 250})
    out = await reward_function.reward_func(None, sample)
    assert out["acc"] is True
    assert out["score"] == 1.0
    assert "agent_soft_overlong_penalty" not in sample.metadata


@pytest.mark.asyncio
async def test_agent_function_failure_is_excluded(monkeypatch, patched_openai):
    _set_env(monkeypatch, LLM_JUDGE_REASONING_EFFORT="low")
    patched_openai["install"]([_make_response("correct: no")])
    sample = _Sample("q", "partial", "right", {"agent_function_failed": True})
    out = await reward_function.reward_func(None, sample)
    assert out["acc"] is False and out["score"] == 0.0, out
    assert sample.remove_sample is True
    assert sample.metadata["agent_excluded_from_training"] is True


@pytest.mark.asyncio
async def test_truncated_sample_is_excluded(monkeypatch, patched_openai):
    _set_env(monkeypatch, LLM_JUDGE_REASONING_EFFORT="low")
    patched_openai["install"]([_make_response("correct: no")])
    sample = _Sample("q", "partial", "right", status="truncated")
    out = await reward_function.reward_func(None, sample)
    assert out["acc"] is False and out["score"] == 0.0, out
    assert sample.remove_sample is True
    assert sample.metadata["agent_excluded_from_training"] is True


@pytest.mark.asyncio
async def test_empty_choices_retries_then_fails(monkeypatch, patched_openai):
    """Exhausted judge retries exclude the sample instead of creating a trainable negative."""
    _set_env(monkeypatch, LLM_JUDGE_REASONING_EFFORT="low")
    patched_openai["install"]([_make_response(None, choices_empty=True) for _ in range(5)])
    sample = _Sample("q", "<answer>a</answer>", "a")
    out = await reward_function.reward_func(None, sample)
    assert out["acc"] is False and out["score"] == 0.0, out
    assert out["judge_error"] is True
    assert sample.remove_sample is True
    assert sample.metadata["agent_excluded_from_training"] is True
    assert sample.metadata["outcome_judge_failed"] is True
    assert "empty_output_text" in sample.metadata["outcome_judge_error"]
    assert len(patched_openai["client"].responses.calls) == 5


@pytest.mark.asyncio
async def test_empty_then_recovers(monkeypatch, patched_openai):
    """First call empty (reasoning), second returns a verdict -> success."""
    _set_env(monkeypatch, LLM_JUDGE_REASONING_EFFORT="low")
    patched_openai["install"]([_make_response(None, choices_empty=True), _make_response("correct: yes")])
    sample = _Sample("q", "<answer>a</answer>", "a")
    out = await reward_function.reward_func(None, sample)
    assert out["acc"] is True and out["score"] == 1.0, out
    assert len(patched_openai["client"].responses.calls) == 2


@pytest.mark.asyncio
async def test_plain_response_is_judged(monkeypatch, patched_openai):
    """BrowseComp judge extracts answers from plain text; <answer> is not required."""
    _set_env(monkeypatch)
    patched_openai["install"]([_make_response("correct: yes")])
    sample = _Sample("q", "no answer tag here", "a")
    out = await reward_function.reward_func(None, sample)
    assert out["acc"] is True and out["score"] == 1.0, out
    assert len(patched_openai["client"].responses.calls) == 1


@pytest.mark.asyncio
async def test_missing_judge_key_fails_fast(monkeypatch, patched_openai):
    _set_env(monkeypatch)
    monkeypatch.delenv("LLM_JUDGE_KEY")
    patched_openai["install"]([_make_response("correct: yes")])
    sample = _Sample("q", "<answer>a</answer>", "a")
    with pytest.raises(ValueError, match="LLM_JUDGE_KEY"):
        await reward_function.reward_func(None, sample)


@pytest.mark.asyncio
async def test_hidden_subproblem_group_bypasses_outcome_judge(monkeypatch, patched_openai):
    _set_env(monkeypatch)
    patched_openai["install"]([AssertionError("outcome judge must not be called")])

    async def fake_process_judge(samples):
        return [{"score": 1.0, "acc": True, "pred": "RIGHT", "event_rubric_state": "redirected"} for _ in samples]

    monkeypatch.setattr(reward_function, "score_value_cliff_group", fake_process_judge)
    samples = [
        _Sample(
            "natural pre-event prefix",
            "continuation",
            "RIGHT",
            {"ab_local_rollout": True, "ab_local_reward_mode": "v6_prm"},
        )
        for _ in range(2)
    ]

    out = await reward_function.reward_func(None, samples)

    assert [item["score"] for item in out] == [1.0, 1.0]
    assert "client" not in patched_openai


@pytest.mark.asyncio
async def test_full_outcome_helper_uses_canonical_prompt_for_hidden_local_sample(monkeypatch, patched_openai):
    _set_env(monkeypatch)
    patched_openai["install"]([_make_response("correct: yes")])
    sample = _Sample(
        "prefix messages",
        "final answer",
        "ORIGINAL GOLD",
        {
            "ab_local_rollout": True,
            "ab_local_reward_mode": "terminal",
            "ab_original_question": "original question",
            "agent_last_finish_reason": "stop",
        },
    )

    out = await reward_function.score_full_outcome_group(None, [sample])

    assert out[0]["acc"] is True
    prompt = patched_openai["client"].responses.calls[0]["input"]
    assert prompt == reward_function.LLM_JUDGE_PROMPT_TEMPLATE.format(
        question="original question",
        response="final answer",
        correct_answer="ORIGINAL GOLD",
    )


@pytest.mark.asyncio
async def test_full_group_runs_only_event_locator(monkeypatch, patched_openai):
    _set_env(monkeypatch)
    patched_openai["install"]([_make_response("correct: yes"), _make_response("correct: no")])
    called = []

    async def fake_event_locator(args, samples, results):
        called.append((args, samples, results))

    monkeypatch.setattr(reward_function, "annotate_value_cliff_branches", fake_event_locator)
    samples = [_Sample("question", "gold", "gold"), _Sample("question", "wrong", "gold")]

    out = await reward_function.reward_func(None, samples)

    assert [item["score"] for item in out] == [1.0, 0.0]
    assert len(called) == 1
    assert called[0][1] is samples
    assert called[0][2] == out
