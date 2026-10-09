from __future__ import annotations

import pytest

from adaptive_branching.src.deep_research import value_cliff_rubric
from adaptive_branching.tools.deep_research import score_value_cliff_rubric_locator, score_value_cliff_rubric_replays


def _locator_verdict(selected_turn=2):
    return {
        "selected_turn": selected_turn,
        "value_drop_reason": "The action adopted an unsupported candidate.",
        "recovery_rubric": {
            "avoid_error": "Do not continue treating the unsupported candidate as established.",
            "redirect": "Directly test evidence that distinguishes it from the viable candidate.",
        },
    }


def test_offline_tools_use_the_shared_value_cliff_prompt_suite():
    assert score_value_cliff_rubric_locator.RUBRIC_LOCATOR_SYSTEM == value_cliff_rubric.VALUE_CLIFF_LOCATOR_SYSTEM
    assert score_value_cliff_rubric_replays.RUBRIC_REPLAY_JUDGE_SYSTEM == value_cliff_rubric.VALUE_CLIFF_JUDGE_SYSTEM
    assert score_value_cliff_rubric_locator.SCORER_VERSION == value_cliff_rubric.VALUE_CLIFF_LOCATOR_VERSION
    assert score_value_cliff_rubric_replays.SCORER_VERSION == value_cliff_rubric.VALUE_CLIFF_JUDGE_VERSION


def test_prompt_horizon_is_explicit_and_dynamic():
    assert "exactly five new assistant turns" in value_cliff_rubric.value_cliff_locator_system(5)
    assert "at most five replacement" in value_cliff_rubric.value_cliff_judge_system(5)
    assert "exactly 7 new assistant turns" in value_cliff_rubric.value_cliff_locator_system(7)
    assert "at most 7 replacement" in value_cliff_rubric.value_cliff_judge_system(7)
    for invalid in (0, -1, True, "5"):
        with pytest.raises(ValueError, match="positive integer"):
            value_cliff_rubric.value_cliff_judge_system(invalid)


def test_local_reward_mode_defaults_to_v6_and_rejects_unknown(monkeypatch):
    monkeypatch.delenv(value_cliff_rubric.LOCAL_REWARD_MODE_ENV, raising=False)
    assert value_cliff_rubric.configured_local_reward_mode() == value_cliff_rubric.V6_PRM_REWARD_MODE
    monkeypatch.setenv(value_cliff_rubric.LOCAL_REWARD_MODE_ENV, value_cliff_rubric.V7_PRM_REWARD_MODE)
    assert value_cliff_rubric.configured_local_reward_mode() == value_cliff_rubric.V7_PRM_REWARD_MODE
    monkeypatch.setenv(value_cliff_rubric.LOCAL_REWARD_MODE_ENV, value_cliff_rubric.V6P_REWARD_MODE)
    assert value_cliff_rubric.configured_local_reward_mode() == value_cliff_rubric.V6P_REWARD_MODE
    monkeypatch.setenv(value_cliff_rubric.LOCAL_REWARD_MODE_ENV, "unknown")
    with pytest.raises(ValueError, match=value_cliff_rubric.LOCAL_REWARD_MODE_ENV):
        value_cliff_rubric.configured_local_reward_mode()


def test_locator_and_judge_validators_enforce_boundaries_and_ordinal_mapping():
    locator = value_cliff_rubric.validate_value_cliff_locator_verdict(_locator_verdict(), assistant_turns=3)
    assert locator["selected_turn"] == 2
    with pytest.raises(ValueError, match="selected_turn"):
        value_cliff_rubric.validate_value_cliff_locator_verdict(_locator_verdict(selected_turn=3), assistant_turns=3)

    for avoid, redirect, score in (
        (False, False, 0.0),
        (False, True, 0.0),
        (True, False, 0.5),
        (True, True, 1.0),
    ):
        verdict = value_cliff_rubric.validate_value_cliff_judge_verdict(
            {
                "avoid_error_met": avoid,
                "redirect_met": redirect,
                "score": score,
                "evidence_turns": [1] if avoid or redirect else [],
                "reason": "Visible behavior supports the verdict.",
            },
            local_turns=1,
        )
        assert verdict["score"] == score

    invalid = {
        "avoid_error_met": True,
        "redirect_met": False,
        "score": 1.0,
        "evidence_turns": [1],
        "reason": "Inconsistent score.",
    }
    with pytest.raises(ValueError, match="score must be 0.5"):
        value_cliff_rubric.validate_value_cliff_judge_verdict(invalid, local_turns=1)
