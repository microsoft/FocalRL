from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from adaptive_branching.src.deep_research import value_cliff_reward


def _sample(index: int, *, reward_mode: str = "v6_prm", natural_finish: bool = False):
    prefix = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "original question"},
        {"role": "assistant", "content": "background"},
    ]
    continuation = [
        {"role": "assistant", "content": f"replacement action {index}"},
        {"role": "tool", "content": "first observation"},
        {"role": "assistant", "content": "use the distinction"},
    ]
    return SimpleNamespace(
        index=index,
        group_index=8,
        prompt="original question",
        response=f"replacement action {index}",
        label="ORIGINAL GOLD MUST STAY HIDDEN",
        remove_sample=False,
        status="completed",
        metadata={
            "ab_local_reward_mode": reward_mode,
            "ab_event_recovery_rubric": {
                "avoid_error": "Stop relying on unsupported X.",
                "redirect": "Test evidence that distinguishes X from Y.",
            },
            "ab_local_prefix_message_count": len(prefix),
            "agent_max_turns": 5,
            "messages": [*prefix, *continuation],
            "agent_last_finish_reason": "stop" if natural_finish else "local_horizon",
            "agent_local_horizon_hit": not natural_finish,
        },
    )


def _rubric_verdict(score: float):
    avoid = score >= 0.5
    redirect = score == 1.0
    return {
        "avoid_error_met": avoid,
        "redirect_met": redirect,
        "score": score,
        "evidence_turns": [1] if avoid or redirect else [],
        "reason": "Visible behavior supports the ordinal score.",
    }


def test_v6_maps_prm_to_zero_half_one(monkeypatch):
    verdicts = iter([_rubric_verdict(0.0), _rubric_verdict(0.5), _rubric_verdict(1.0)])

    class FakeClient:
        def __init__(self, **kwargs):
            assert kwargs == {"config_section": "local_prm"}

        async def complete_json(self, system, prompt, *, validate, **kwargs):
            assert "at most five replacement" in system
            assert "ORIGINAL GOLD MUST STAY HIDDEN" not in prompt
            return validate(next(verdicts))

    monkeypatch.setattr(value_cliff_reward, "JudgeClient", FakeClient)
    samples = [_sample(1), _sample(2), _sample(3)]
    results = asyncio.run(value_cliff_reward.score_value_cliff_group(samples))
    assert [result["score"] for result in results] == [0.0, 0.5, 1.0]
    assert [result["event_rubric_state"] for result in results] == [
        "avoid_failed",
        "avoid_only",
        "redirected",
    ]
    assert all(sample.metadata["ab_event_local_group_informative"] for sample in samples)


def test_v6_single_call_length_is_trainable_zero_and_skips_only_its_judge(monkeypatch):
    verdicts = iter([_rubric_verdict(0.5), _rubric_verdict(1.0)])
    judge_calls = []

    class FakeClient:
        def __init__(self, **kwargs):
            assert kwargs == {"config_section": "local_prm"}

        async def complete_json(self, system, prompt, *, validate, **kwargs):
            judge_calls.append(kwargs["tag"])
            return validate(next(verdicts))

    monkeypatch.setattr(value_cliff_reward, "JudgeClient", FakeClient)
    samples = [_sample(1), _sample(2), _sample(3)]
    samples[1].metadata["agent_last_finish_reason"] = "length"
    samples[1].metadata["agent_local_horizon_hit"] = False

    results = asyncio.run(value_cliff_reward.score_value_cliff_group(samples))

    assert len(judge_calls) == 2
    assert [result["score"] for result in results] == [0.5, 0.0, 1.0]
    assert results[1] == {
        "score": 0.0,
        "acc": False,
        "pred": "",
        "judge_raw": "single_call_length",
        "event_rubric_state": "avoid_failed",
    }
    assert all(not sample.remove_sample for sample in samples)
    assert samples[1].metadata["agent_excluded_from_training"] is False
    assert samples[1].metadata["agent_single_call_length_penalized"] is True
    assert all(sample.metadata["ab_event_local_group_informative"] for sample in samples)


def test_v6p_routes_natural_finishes_to_outcome_and_unfinished_to_v6_rubric(monkeypatch):
    rubric_verdicts = iter([_rubric_verdict(1.0), _rubric_verdict(0.5), _rubric_verdict(0.0)])
    rubric_indices = []
    outcome_indices = []

    class FakeClient:
        def __init__(self, **kwargs):
            assert kwargs == {"config_section": "local_prm"}

        async def complete_json(self, *args, validate, tag, **kwargs):
            rubric_indices.append(int(tag.rsplit("_s", 1)[1]))
            return validate(next(rubric_verdicts))

    async def fake_full_outcome(samples):
        outcome_indices.extend(sample.index for sample in samples)
        return [
            {"score": 1.0, "acc": True, "pred": "gold", "judge_raw": "yes"},
            {"score": 0.0, "acc": False, "pred": "wrong", "judge_raw": "no"},
        ]

    monkeypatch.setattr(value_cliff_reward, "JudgeClient", FakeClient)
    monkeypatch.setattr(value_cliff_reward, "_score_full_outcome_samples", fake_full_outcome)
    samples = [
        _sample(1, reward_mode="v6p", natural_finish=True),
        _sample(2, reward_mode="v6p", natural_finish=True),
        _sample(3, reward_mode="v6p"),
        _sample(4, reward_mode="v6p"),
        _sample(5, reward_mode="v6p"),
    ]

    results = asyncio.run(value_cliff_reward.score_value_cliff_group(samples))

    assert outcome_indices == [1, 2]
    assert rubric_indices == [3, 4, 5]
    assert [result["score"] for result in results] == [1.0, 0.0, 1.0, 0.5, 0.0]
    assert [result["acc"] for result in results] == [True, False, True, False, False]
    assert [result["event_rubric_state"] for result in results] == [
        "outcome_correct",
        "outcome_wrong",
        "redirected",
        "avoid_only",
        "avoid_failed",
    ]
    assert all(
        sample.metadata["ab_event_rubric_judge_version"] == value_cliff_reward.VALUE_CLIFF_V6P_REWARD_VERSION
        for sample in samples
    )


def test_v6p_single_call_length_is_zero_and_skips_both_judges(monkeypatch):
    rubric_calls = []
    outcome_indices = []

    class FakeClient:
        def __init__(self, **kwargs):
            assert kwargs == {"config_section": "local_prm"}

        async def complete_json(self, *args, validate, tag, **kwargs):
            rubric_calls.append(tag)
            return validate(_rubric_verdict(0.5))

    async def fake_full_outcome(samples):
        outcome_indices.extend(sample.index for sample in samples)
        return [{"score": 1.0, "acc": True, "pred": "gold", "judge_raw": "yes"}]

    monkeypatch.setattr(value_cliff_reward, "JudgeClient", FakeClient)
    monkeypatch.setattr(value_cliff_reward, "_score_full_outcome_samples", fake_full_outcome)
    samples = [
        _sample(1, reward_mode="v6p", natural_finish=True),
        _sample(2, reward_mode="v6p"),
        _sample(3, reward_mode="v6p"),
    ]
    samples[1].metadata["agent_last_finish_reason"] = "length"
    samples[1].metadata["agent_local_horizon_hit"] = False

    results = asyncio.run(value_cliff_reward.score_value_cliff_group(samples))

    assert outcome_indices == [1]
    assert len(rubric_calls) == 1
    assert [result["score"] for result in results] == [1.0, 0.0, 0.5]
    assert results[1]["judge_raw"] == "single_call_length"
    assert samples[1].metadata["agent_single_call_length_penalized"] is True
    assert all(not sample.remove_sample for sample in samples)


def test_v6p_natural_finish_uses_full_outcome_at_configured_horizon(monkeypatch):
    outcome_indices = []

    class ForbiddenClient:
        def __init__(self, **kwargs):
            raise AssertionError("a natural finish at the configured horizon must not use the local rubric")

    async def fake_full_outcome(samples):
        outcome_indices.extend(sample.index for sample in samples)
        return [{"score": 1.0, "acc": True, "pred": "gold", "judge_raw": "yes"}]

    monkeypatch.setattr(value_cliff_reward, "JudgeClient", ForbiddenClient)
    monkeypatch.setattr(value_cliff_reward, "_score_full_outcome_samples", fake_full_outcome)
    sample = _sample(1, reward_mode="v6p", natural_finish=True)
    prefix_count = sample.metadata["ab_local_prefix_message_count"]
    sample.metadata["messages"] = [
        *sample.metadata["messages"][:prefix_count],
        *({"role": "assistant", "content": f"local action {turn}"} for turn in range(1, 7)),
    ]
    sample.metadata["agent_max_turns"] = 6

    results = asyncio.run(value_cliff_reward.score_value_cliff_group([sample]))

    assert outcome_indices == [1]
    assert results[0]["score"] == 1.0
    assert results[0]["event_rubric_state"] == "outcome_correct"
    assert sample.metadata["ab_event_local_natural_finish"] is True
    assert sample.metadata["ab_event_rubric_judge"]["reward_source"] == "full_outcome"


def test_v6_context_reserve_still_excludes_complete_group(monkeypatch):
    samples = [_sample(1), _sample(2)]
    samples[0].metadata["agent_last_finish_reason"] = "local_context_reserve"
    samples[0].metadata["agent_context_reserve_hit"] = True

    class ForbiddenClient:
        def __init__(self, **kwargs):
            raise AssertionError("judge must not be constructed")

    monkeypatch.setattr(value_cliff_reward, "JudgeClient", ForbiddenClient)
    results = asyncio.run(value_cliff_reward.score_value_cliff_group(samples))

    assert all(result["judge_error"] for result in results)
    assert all(sample.remove_sample for sample in samples)
    assert all("agent_context_reserve_hit" in sample.metadata["ab_event_rubric_judge_error"] for sample in samples)


def test_v7_single_call_length_is_trainable_zero_and_skips_only_its_judge(monkeypatch):
    verdicts = iter([_rubric_verdict(0.5), _rubric_verdict(1.0)])
    judge_calls = []

    class FakeClient:
        def __init__(self, **kwargs):
            assert kwargs == {"config_section": "local_prm"}

        async def complete_json(self, system, prompt, *, validate, **kwargs):
            judge_calls.append(kwargs["tag"])
            return validate(next(verdicts))

    monkeypatch.setattr(value_cliff_reward, "JudgeClient", FakeClient)
    samples = [_sample(1, reward_mode="v7_prm"), _sample(2, reward_mode="v7_prm"), _sample(3, reward_mode="v7_prm")]
    samples[1].metadata["agent_last_finish_reason"] = "length"
    samples[1].metadata["agent_local_horizon_hit"] = False

    results = asyncio.run(value_cliff_reward.score_value_cliff_group(samples))

    assert len(judge_calls) == 2
    assert [result["score"] for result in results] == [0.25, 0.0, 0.5]
    assert results[1]["event_rubric_state"] == "avoid_failed"
    assert all(not sample.remove_sample for sample in samples)
    assert samples[1].metadata["agent_single_call_length_penalized"] is True


def test_group_rejects_mixed_or_unknown_reward_modes():
    samples = [_sample(1), _sample(2, reward_mode="v7_prm")]
    with pytest.raises(ValueError, match="mixes reward modes"):
        asyncio.run(value_cliff_reward.score_value_cliff_group(samples))
    with pytest.raises(ValueError, match="unsupported"):
        asyncio.run(value_cliff_reward.score_value_cliff_group([_sample(1, reward_mode="legacy")]))


def test_terminal_routes_every_replay_to_full_outcome(monkeypatch):
    captured = []

    async def fake_full_outcome(samples):
        captured.extend(sample.index for sample in samples)
        return [
            {"score": float(sample.index == 1), "acc": sample.index == 1, "pred": "", "judge_raw": "ok"}
            for sample in samples
        ]

    monkeypatch.setattr(value_cliff_reward, "_score_full_outcome_samples", fake_full_outcome)
    samples = [_sample(1, reward_mode="terminal", natural_finish=True), _sample(2, reward_mode="terminal")]
    results = asyncio.run(value_cliff_reward.score_value_cliff_group(samples))
    assert captured == [1, 2]
    assert [result["score"] for result in results] == [1.0, 0.0]


def test_prm_structural_failure_excludes_complete_group(monkeypatch):
    samples = [_sample(1), _sample(2)]
    samples[0].metadata["agent_max_turns"] = 0

    class ForbiddenClient:
        def __init__(self, **kwargs):
            raise AssertionError("judge must not be constructed")

    monkeypatch.setattr(value_cliff_reward, "JudgeClient", ForbiddenClient)
    results = asyncio.run(value_cliff_reward.score_value_cliff_group(samples))
    assert all(result["judge_error"] for result in results)
    assert all(sample.remove_sample for sample in samples)
    assert all("invalid agent_max_turns" in sample.metadata["ab_event_rubric_judge_error"] for sample in samples)


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        ({"natural_finished": True, "outcome_correct": True}, (1.0, "outcome_correct")),
        ({"natural_finished": True, "outcome_correct": False}, (0.0, "outcome_wrong")),
        (
            {"natural_finished": False, "avoid_error_met": True, "redirect_met": True},
            (0.5, "redirected"),
        ),
        (
            {"natural_finished": False, "avoid_error_met": True, "redirect_met": False},
            (0.25, "avoid_only"),
        ),
        (
            {"natural_finished": False, "avoid_error_met": False, "redirect_met": True},
            (0.0, "avoid_failed"),
        ),
    ],
)
def test_v7_reward_mapping_is_code_defined(kwargs, expected):
    assert value_cliff_reward.map_value_cliff_outcome4_reward(**kwargs) == expected


def test_v7_routes_natural_and_unfinished_replays_separately(monkeypatch):
    rubric_verdicts = iter([_rubric_verdict(1.0), _rubric_verdict(0.5), _rubric_verdict(0.0)])
    rubric_indices = []
    outcome_indices = []

    class FakeClient:
        def __init__(self, **kwargs):
            assert kwargs == {"config_section": "local_prm"}

        async def complete_json(self, *args, validate, tag, **kwargs):
            rubric_indices.append(int(tag.rsplit("_s", 1)[1]))
            return validate(next(rubric_verdicts))

    async def fake_full_outcome(samples):
        outcome_indices.extend(sample.index for sample in samples)
        return [
            {"score": 1.0, "acc": True, "pred": "gold", "judge_raw": "yes"},
            {"score": 0.0, "acc": False, "pred": "wrong", "judge_raw": "no"},
        ]

    monkeypatch.setattr(value_cliff_reward, "JudgeClient", FakeClient)
    monkeypatch.setattr(value_cliff_reward, "_score_full_outcome_samples", fake_full_outcome)
    samples = [
        _sample(1, reward_mode="v7_prm", natural_finish=True),
        _sample(2, reward_mode="v7_prm", natural_finish=True),
        _sample(3, reward_mode="v7_prm"),
        _sample(4, reward_mode="v7_prm"),
        _sample(5, reward_mode="v7_prm"),
    ]
    results = asyncio.run(value_cliff_reward.score_value_cliff_group(samples))
    assert outcome_indices == [1, 2]
    assert rubric_indices == [3, 4, 5]
    assert [result["score"] for result in results] == [1.0, 0.0, 0.5, 0.25, 0.0]
    assert [result["event_rubric_state"] for result in results] == [
        "outcome_correct",
        "outcome_wrong",
        "redirected",
        "avoid_only",
        "avoid_failed",
    ]


def test_v7_outcome_judge_failure_excludes_complete_group(monkeypatch):
    async def fake_full_outcome(samples):
        return [{"score": 0.0, "acc": False, "judge_raw": "bad", "judge_error": True}]

    monkeypatch.setattr(value_cliff_reward, "_score_full_outcome_samples", fake_full_outcome)
    sample = _sample(1, reward_mode="v7_prm", natural_finish=True)
    results = asyncio.run(value_cliff_reward.score_value_cliff_group([sample]))
    assert results[0]["judge_error"] is True
    assert sample.remove_sample is True
