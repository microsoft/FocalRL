from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from adaptive_branching.src.deep_research import value_cliff_online as event_online
from adaptive_branching.src.deep_research.value_cliff_locator import ValueCliffLocalizationResult


def _messages(final: str):
    return [
        {"role": "user", "content": "original question"},
        {"role": "assistant", "content": "collect background"},
        {"role": "tool", "content": "background evidence"},
        {"role": "assistant", "content": "adopt WRONG"},
        {"role": "assistant", "content": final},
    ]


def _sample(index: int, final: str):
    return SimpleNamespace(
        index=index,
        group_index=7,
        label="GOLD",
        prompt=[{"role": "user", "content": "original question"}],
        remove_sample=False,
        metadata={"messages": _messages(final), "agent_last_finish_reason": "stop"},
    )


def _localization():
    return ValueCliffLocalizationResult(
        event={
            "event_type": "value_cliff",
            "event_turn": 2,
            "event_summary": "Turn 2 adopted unsupported WRONG.",
            "value_drop_reason": "Turn 2 adopted unsupported WRONG.",
            "recovery_rubric": {
                "avoid_error": "Stop relying on WRONG.",
                "redirect": "Test evidence that distinguishes WRONG from GOLD.",
            },
        },
        diagnostics={"policy": "value_cliff_rubric_locator_v6_20260822", "stage_status": "selected"},
    )


@pytest.fixture(autouse=True)
def _local_env(monkeypatch):
    monkeypatch.setenv("AB_LOCAL_ROLLOUT_ENABLE", "1")
    monkeypatch.setenv("AB_LOCAL_MAX_GROUPS_PER_FULL_GROUP", "8")
    monkeypatch.setenv("AB_EVENT_HIDDEN_MAX_TURNS", "5")
    monkeypatch.setenv("AB_LOCAL_REWARD_MODE", "v6_prm")


def test_configuration_keeps_hidden_horizon_and_rejects_invalid_budget(monkeypatch):
    assert event_online.local_horizon_max_turns() == 5
    monkeypatch.setenv("AB_LOCAL_MAX_GROUPS_PER_FULL_GROUP", "-1")
    with pytest.raises(ValueError, match="nonnegative"):
        event_online.local_group_budget()


@pytest.mark.parametrize("reward_mode", ["v6_prm", "v6p", "v7_prm", "terminal"])
def test_annotation_uses_fixed_value_cliff_prefix_and_reward_mode(monkeypatch, reward_mode):
    monkeypatch.setenv("AB_LOCAL_REWARD_MODE", reward_mode)
    monkeypatch.setattr(event_online, "_make_locator_client", object)

    async def fake_locate(*args, **kwargs):
        return _localization()

    monkeypatch.setattr(event_online, "locate_value_cliff", fake_locate)
    positive = _sample(1, "correct")
    negative = _sample(2, "wrong")
    asyncio.run(
        event_online.annotate_value_cliff_branches(
            SimpleNamespace(),
            [positive, negative],
            [{"acc": True}, {"acc": False}],
        )
    )
    spec = negative.metadata["ab_branch_spec"]
    assert spec["branch_type"] == "value_cliff_local"
    assert spec["reward_mode"] == reward_mode
    assert spec["event_turn"] == 2
    assert spec["prefix_messages"][-1]["content"] == "background evidence"
    assert all("adopt WRONG" not in str(message) for message in spec["prefix_messages"])
    assert spec["recovery_rubric"]["avoid_error"] == "Stop relying on WRONG."
    info = positive.metadata["ab_event_selection_info"]
    assert info["ab_locator_attempted_count"] == 1
    assert info["ab_event_branch_count"] == 1


def test_disabled_local_rollout_does_not_construct_locator(monkeypatch):
    monkeypatch.setenv("AB_LOCAL_ROLLOUT_ENABLE", "0")
    monkeypatch.setattr(event_online, "_make_locator_client", lambda: pytest.fail("must not run"))
    asyncio.run(
        event_online.annotate_value_cliff_branches(
            SimpleNamespace(),
            [_sample(1, "correct"), _sample(2, "wrong")],
            [{"acc": True}, {"acc": False}],
        )
    )


def test_missing_matched_polarity_records_skip_without_locator(monkeypatch):
    monkeypatch.setattr(event_online, "_make_locator_client", lambda: pytest.fail("must not run"))
    samples = [_sample(1, "wrong"), _sample(2, "wrong")]
    asyncio.run(
        event_online.annotate_value_cliff_branches(SimpleNamespace(), samples, [{"acc": False}, {"acc": False}])
    )
    assert samples[0].metadata["ab_event_selection_info"]["ab_skipped_reason"] == "missing_positive_or_negative"


def test_annotation_rejects_cardinality_and_invalid_metadata():
    sample = _sample(1, "wrong")
    with pytest.raises(ValueError, match="one result per sample"):
        asyncio.run(event_online.annotate_value_cliff_branches(SimpleNamespace(), [sample], []))
    sample.metadata = "invalid"
    with pytest.raises(TypeError, match="metadata"):
        asyncio.run(event_online.annotate_value_cliff_branches(SimpleNamespace(), [sample], [{"acc": False}]))
