from __future__ import annotations

import asyncio
import copy
import math
import random
from types import SimpleNamespace

import pytest

from adaptive_branching.src.deep_research import branch_sampling as branching
from adaptive_branching.src.deep_research import value_cliff_online as online
from miles.utils.types import Sample


def history(turns=4):
    messages = [{"role": "user", "content": "question"}]
    for turn in range(1, turns + 1):
        messages.extend(
            [
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": f"call-{turn}",
                            "type": "function",
                            "function": {"name": "search", "arguments": '{"query":"q"}'},
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": f"call-{turn}", "content": f"result-{turn}"},
            ]
        )
    messages.append({"role": "assistant", "content": "answer"})
    return messages


def parent(index=0, acc=True, turns=4):
    messages = history(turns)
    return Sample(
        index=index,
        group_index=3,
        prompt="question",
        label="answer",
        status=Sample.Status.COMPLETED,
        reward={"acc": acc, "score": float(acc)},
        metadata={
            "messages": messages,
            "ab_branch_history": copy.deepcopy(messages),
            "ab_branch_entropies": {str(t): {"entropy": float(t)} for t in range(1, turns + 1)},
        },
    )


def meta(tokens=1, sampled=0):
    return {
        "output_token_logprobs": [[math.log(0.05), sampled, None] for _ in range(tokens)],
        "output_top_logprobs": [[[math.log(0.05), t, None] for t in range(10)] for _ in range(tokens)],
        "weight_version": "7",
    }


@pytest.fixture
def branch_env(monkeypatch):
    monkeypatch.setenv("AB_BRANCH_SELECTION", "random")
    monkeypatch.setenv("AB_BRANCH_SEED", "42")
    monkeypatch.setenv("AB_LOCAL_ROLLOUT_ENABLE", "1")
    monkeypatch.setenv("AB_LOCAL_REWARD_MODE", "terminal")


@pytest.mark.parametrize("acc", [False, True])
def test_homogeneous_outcomes_branch_without_locator(branch_env, monkeypatch, acc):
    monkeypatch.setattr(online, "_make_locator_client", lambda: pytest.fail("locator must not be used"))
    samples = [parent(i, acc=acc) for i in range(8)]
    asyncio.run(online.annotate_value_cliff_branches(None, samples, [s.reward for s in samples]))
    chosen = [s for s in samples if s.metadata.get("ab_branch_spec")]
    assert len(chosen) == 1
    info = chosen[0].metadata["ab_branch_spec"]
    assert info["reward_mode"] == "terminal"
    assert info["prefix_messages"][-1]["role"] == "tool"
    assert info["selection"]["eligible_parents"] == 8
    original = copy.deepcopy(info)
    asyncio.run(online.annotate_value_cliff_branches(None, samples, [s.reward for s in samples]))
    assert chosen[0].metadata["ab_branch_spec"] == original


def test_candidates_exclude_root_and_final():
    assert branching.candidate_positions(history(3)) == [(3, 2), (5, 3)]
    assert branching.candidate_positions([]) == []
    assert branching.candidate_positions(history(1)) == []
    with pytest.raises(ValueError):
        branching.candidate_positions([None])


def test_no_candidates_and_excluded_parent(branch_env):
    samples = [parent(0, turns=1), parent(1)]
    samples[1].remove_sample = True
    asyncio.run(online.annotate_value_cliff_branches(None, samples, [s.reward for s in samples]))
    assert not any(s.metadata.get("ab_branch_spec") for s in samples)
    assert samples[0].metadata["ab_branch_selection"]["skip_reason"] == "no_eligible_parent"


def test_missing_history_and_mixed_group_fail(branch_env):
    sample = parent()
    sample.metadata.pop("ab_branch_history")
    with pytest.raises(branching.BranchDataError, match="history"):
        asyncio.run(online.annotate_value_cliff_branches(None, [sample], [sample.reward]))
    another = parent(1)
    another.group_index = 4
    with pytest.raises(ValueError, match="mixed"):
        branching.annotate_terminal_branch([sample, another], [], "random")


def test_keep5_reconstructed_at_selected_point(branch_env):
    sample = parent(turns=9)
    sample.metadata.update(agent_history_mode="keep5", agent_keep_tool_results=5)
    # Select turn 7; at this point result 1 is omitted but result 2 remains visible.
    sample.metadata["ab_branch_entropies"]["7"]["entropy"] = 100.0
    branching.annotate_terminal_branch([sample], [(0, sample, sample.reward, history(9))], "entropy_delta_max")
    prefix = sample.metadata["ab_branch_spec"]["prefix_messages"]
    tool_results = [m["content"] for m in prefix if m["role"] == "tool"]
    assert tool_results == ["Tool result is omitted to save tokens."] + [f"result-{t}" for t in range(2, 7)]
    assert sample.metadata["ab_branch_history"][2]["content"] == "result-1"
    prefix[-1]["content"] = "mutation"
    assert sample.metadata["ab_branch_history"][12]["content"] == "result-6"


@pytest.mark.parametrize("tokens,selected,count", [(1, 0, 10), (20, 10, 11), (21, 0, 10)])
def test_entropy_window_and_union(tokens, selected, count):
    result = branching.entropy_score(meta(tokens, selected), 100)
    assert result["tokens"] == min(tokens, 20)
    assert result["entropy"] == pytest.approx(-count * 0.05 * math.log(0.05) * min(tokens, 20) / math.log(100))
    assert result["weight_version"] == "7"


@pytest.mark.parametrize(
    "kind", ["empty", "missing", "length", "top_count", "duplicate", "mass", "nan", "id", "inconsistent"]
)
def test_invalid_entropy_fails(kind):
    data = meta()
    if kind == "empty":
        data["output_token_logprobs"] = []
    if kind == "missing":
        data.pop("output_top_logprobs")
    if kind == "length":
        data["output_top_logprobs"] *= 2
    if kind == "top_count":
        data["output_top_logprobs"][0].pop()
    if kind == "duplicate":
        data["output_top_logprobs"][0][1][1] = 0
    if kind == "mass":
        for row in data["output_top_logprobs"][0]:
            row[0] = math.log(0.2)
        data["output_token_logprobs"][0][0] = math.log(0.2)
    if kind == "nan":
        data["output_token_logprobs"][0][0] = float("nan")
    if kind == "id":
        data["output_token_logprobs"][0][1] = 100
    if kind == "inconsistent":
        data["output_token_logprobs"][0][0] = math.log(0.01)
    with pytest.raises(branching.BranchDataError):
        branching.entropy_score(data, 100)


def test_entropy_negative_delta_earliest_tie():
    scores = {"1": {"entropy": 10}, "2": {"entropy": 8}, "3": {"entropy": 8}}
    assert branching.choose_position([(5, 3), (3, 2)], "entropy_delta_max", random.Random(0), scores) == (3, 2)
    assert branching.choose_position([(3, 2)], "random", random.Random(0), {}) == (3, 2)


@pytest.mark.parametrize("bad", [None, "bad", float("inf"), -1, True])
def test_invalid_entropy_selection(bad):
    with pytest.raises(branching.BranchDataError):
        branching.choose_position([(3, 2)], "entropy_delta_max", random.Random(), {"1": {"entropy": bad}})


def test_policy_seed_and_supply_validation(branch_env, monkeypatch):
    assert branching.selection_policy() == "random"
    assert branching.seed_for_group(3) == branching.seed_for_group(3)
    assert branching.seed_for_group(3) != branching.seed_for_group(4)
    branching.validate_branch_supply(0, 16)
    with pytest.raises(RuntimeError, match="cannot fill"):
        branching.validate_branch_supply(16, 16)
    with pytest.raises(ValueError):
        branching.validate_branch_supply(0, 0)
    with pytest.raises(ValueError):
        branching.seed_for_group(-1)
    monkeypatch.setenv("AB_BRANCH_SEED", "-1")
    with pytest.raises(ValueError):
        branching.seed_for_group(3)
    monkeypatch.setenv("AB_LOCAL_REWARD_MODE", "v6_prm")
    with pytest.raises(ValueError):
        branching.selection_policy()
    monkeypatch.setenv("AB_BRANCH_SELECTION", "invalid")
    with pytest.raises(ValueError):
        branching.selection_policy()


def test_generation_cost_includes_repeated_calls_and_unknown_usage():
    known = SimpleNamespace(response={"usage": {"prompt_tokens": 20, "completion_tokens": 3}})
    unknown = SimpleNamespace(response={"error": "timeout"})
    assert branching.generation_cost([known, known, unknown]) == {
        "calls": 3,
        "prompt_tokens": 40,
        "completion_tokens": 6,
        "unknown_usage_calls": 1,
    }
    assert branching.generation_cost([])["calls"] == 0
    with pytest.raises(ValueError):
        branching.generation_cost([SimpleNamespace(response={"usage": {}})])
    with pytest.raises(ValueError):
        branching.generation_cost(None)


def test_collector_terminal_children_and_dedup(branch_env, monkeypatch):
    from adaptive_branching.tests.deep_research.test_fully_async_value_cliff_collector_on_cpu import _import_collector

    collector = _import_collector(monkeypatch)
    sample = parent()
    asyncio.run(online.annotate_value_cliff_branches(None, [sample], [sample.reward]))
    buffer = SimpleNamespace(sample_group_index=10, sample_index=80)
    args = SimpleNamespace(n_samples_per_prompt=8)
    groups = collector._build_local_groups(args, buffer, [sample])
    assert len(groups) == 1 and len(groups[0]) == 8
    assert {s.group_index for s in groups[0]} == {10}
    assert [s.index for s in groups[0]] == list(range(80, 88))
    for child in groups[0]:
        assert child.metadata["ab_local_reward_mode"] == "terminal"
        assert child.metadata["agent_force_final_on_max_turns"] is True
        assert "ab_event_recovery_rubric" not in child.metadata
        assert "ab_branch_history" not in child.metadata
        assert child.prompt[-1]["role"] == "tool"
    assert collector._build_local_groups(args, buffer, [sample]) == []
    assert collector._build_local_groups(args, buffer, groups[0]) == []
    groups[0][0].prompt[-1]["content"] = "mutation"
    assert groups[0][1].prompt[-1]["content"] != "mutation"
