import asyncio
import copy

import pytest

from miles.utils.types import Sample

from adaptive_branching.src.swe import value_cliff_online


def _messages(first: str) -> list[dict]:
    return [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "issue"},
        {"role": "assistant", "content": first, "extra": {"actions": []}},
        {"role": "tool", "content": "observation"},
        {"role": "assistant", "content": "later", "extra": {"actions": []}},
    ]


@pytest.mark.parametrize(
    "positive_status,negative_status,budget,negative_count",
    [
        ("Submitted", "Submitted", 1, 1),
        ("TurnLimit", "TurnLimit", 2, 7),
        ("ContextLimit", "FormatLimit", 8, 7),
        ("FormatLimit", "ContextLimit", 2, 1),
        ("LengthTruncated", "LengthTruncated", 8, 7),
        (None, None, 2, 7),
        ("Submitted", "TurnLimit", 0, 7),
    ],
)
def test_mixed_group_attaches_a_harbor_replay_branch(
    monkeypatch, positive_status, negative_status, budget, negative_count
):
    monkeypatch.setenv("AB_LOCAL_ROLLOUT_ENABLE", "1")
    monkeypatch.setenv("AB_LOCAL_MAX_GROUPS_PER_FULL_GROUP", str(budget))
    monkeypatch.setenv("AB_EVENT_HIDDEN_MAX_TURNS", "10")
    monkeypatch.setattr(
        value_cliff_online,
        "judge_settings",
        lambda section: {"trace_max_chars": 10000, "max_concurrency": 4},
    )

    trajectories = {
        "/trials/positive": _messages("inspect the relevant distinction"),
        "/trials/negative": _messages("assume the wrong lifecycle"),
    }

    async def fake_messages(trial_dir):
        return trajectories[trial_dir]

    async def fake_replay(trial_dir, selected_turn):
        assert trial_dir == "/trials/negative"
        assert selected_turn == 1
        return {
            "replay_path": "/trials/negative/agent/local-replay-turn-1.json",
            "prefix_messages": trajectories[trial_dir][:2],
            "n_calls": 0,
        }

    class FakeClient:
        def __init__(self, *, config_section):
            assert config_section == "locator"

        async def complete_json(self, _system, _prompt, *, required_keys, tag, validate):
            assert required_keys
            assert tag in {f"swe_value_cliff_g7_s{i}" for i in range(2, 2 + min(budget, negative_count))}
            return validate(
                {
                    "selected_turn": 1,
                    "value_drop_reason": "the first action commits to the wrong lifecycle",
                    "recovery_rubric": {
                        "avoid_error": "stop treating the current loop as test-owned",
                        "redirect": "distinguish a new loop from an externally owned loop",
                    },
                }
            )

    monkeypatch.setattr(value_cliff_online, "get_trial_messages", fake_messages)
    monkeypatch.setattr(value_cliff_online, "create_trial_replay", fake_replay)
    monkeypatch.setattr(value_cliff_online, "JudgeClient", FakeClient)
    positive = Sample(
        prompt="Fix the loop lifecycle.",
        group_index=7,
        index=1,
        metadata={
            "reward": 1,
            "exit_status": positive_status,
            "trial_dir": "/trials/positive",
            "ab_swe_golden_patch": "diff --git a/loop.py b/loop.py",
            "agent_excluded_from_training": False,
        },
    )
    negative = Sample(
        prompt="Fix the loop lifecycle.",
        group_index=7,
        index=2,
        metadata={
            "reward": 0,
            "exit_status": negative_status,
            "trial_dir": "/trials/negative",
            "ab_swe_golden_patch": "diff --git a/loop.py b/loop.py",
            "agent_excluded_from_training": False,
        },
    )

    negatives = [copy.deepcopy(negative) for _ in range(negative_count)]
    for index, sample in enumerate(negatives, 2):
        sample.index = index
    # A truncated success has a zero training result but remains verifier-positive.
    results = [{"score": 0.0, "acc": False}] + [{"score": 0.0, "acc": False}] * negative_count
    asyncio.run(
        value_cliff_online.annotate_swe_value_cliff_branches(
            None,
            [positive, *reversed(negatives)],
            results,
        )
    )

    assert "ab_branch_spec" not in positive.metadata
    selected = min(budget, negative_count)
    for sample in negatives[:selected]:
        assert sample.metadata["swe_replay_n_calls"] == 0
        assert sample.metadata["ab_branch_spec"]["event_turn"] == 1
        assert sample.metadata["ab_branch_spec"]["reward_mode"] == "v6_prm"
    assert all("ab_branch_spec" not in s.metadata for s in negatives[selected:])
    info = positive.metadata["ab_event_selection_info"]
    assert info["ab_event_branch_count"] == selected
    assert info["ab_local_budget"] == budget
    assert info["ab_eligible_positive_count"] == 1
    assert info["ab_eligible_negative_count"] == negative_count
    assert info["ab_selection_policy"] == "swe_raw_verifier_mixed_v2"


@pytest.mark.parametrize("rewards", [[], [0], [1], [0] * 8, [1] * 8])
def test_non_mixed_group_does_not_call_harbor(monkeypatch, rewards):
    monkeypatch.setenv("AB_LOCAL_ROLLOUT_ENABLE", "1")

    async def unexpected(_trial_dir):
        raise AssertionError("Harbor should not be queried for a non-mixed group")

    monkeypatch.setattr(value_cliff_online, "get_trial_messages", unexpected)
    sample = Sample(
        prompt="issue",
        index=0,
        metadata={"reward": 0, "exit_status": "Submitted", "agent_excluded_from_training": False},
    )

    samples = []
    for index, raw in enumerate(rewards):
        item = copy.deepcopy(sample)
        item.index = index
        item.metadata["reward"] = raw
        samples.append(item)
    asyncio.run(
        value_cliff_online.annotate_swe_value_cliff_branches(
            None,
            samples,
            [{"score": raw, "acc": bool(raw)} for raw in rewards],
        )
    )

    for item in samples:
        assert item.metadata["ab_event_selection_info"]["ab_locator_attempted_count"] == 0


@pytest.mark.parametrize("flag", ["agent_excluded_from_training", "agent_function_failed", "remove", "aborted"])
@pytest.mark.parametrize("excluded_raw", [0, 1])
def test_excluded_outcomes_cannot_supply_either_side(monkeypatch, flag, excluded_raw):
    monkeypatch.setenv("AB_LOCAL_ROLLOUT_ENABLE", "1")

    async def unexpected(*args):
        raise AssertionError("Excluded outcomes must not reach Harbor")

    monkeypatch.setattr(value_cliff_online, "get_trial_messages", unexpected)
    excluded = Sample(index=0, metadata={"reward": excluded_raw})
    if flag == "remove":
        excluded.remove_sample = True
    elif flag == "aborted":
        excluded.status = Sample.Status.ABORTED
    else:
        excluded.metadata[flag] = True
    valid = Sample(index=1, metadata={"reward": 1 - excluded_raw})
    asyncio.run(value_cliff_online.annotate_swe_value_cliff_branches(None, [excluded, valid], [{}, {}]))
    assert valid.metadata["ab_event_selection_info"]["ab_locator_attempted_count"] == 0


@pytest.mark.parametrize("raw", [None, True, "0", 0.5, -1, 2])
def test_invalid_verifier_label_fails_before_locator(monkeypatch, raw):
    monkeypatch.setenv("AB_LOCAL_ROLLOUT_ENABLE", "1")
    sample = Sample(index=9, metadata={"reward": raw})
    with pytest.raises(ValueError, match="sample_index=9.*binary verifier reward"):
        asyncio.run(value_cliff_online.annotate_swe_value_cliff_branches(None, [sample], [{}]))


def test_result_length_mismatch_fails(monkeypatch):
    monkeypatch.setenv("AB_LOCAL_ROLLOUT_ENABLE", "1")
    with pytest.raises(ValueError, match="one full-outcome result"):
        asyncio.run(value_cliff_online.annotate_swe_value_cliff_branches(None, [Sample(metadata={})], []))


def test_missing_trial_path_fails_with_sample_context(monkeypatch):
    monkeypatch.setenv("AB_LOCAL_ROLLOUT_ENABLE", "1")
    samples = [Sample(index=i, metadata={"reward": i, "ab_swe_golden_patch": "patch"}) for i in (0, 1)]
    with pytest.raises(ValueError, match="sample_index=1: missing trial_dir"):
        asyncio.run(value_cliff_online.annotate_swe_value_cliff_branches(None, samples, [{}, {}]))
