"""Independent terminal-ORM groups from random or offline entropy-selected prefixes.

These are controlled branching baselines, not reproductions of tree advantages.
No network calls, locator, or PRM are involved in selection.
"""

from __future__ import annotations

import copy
import hashlib
import math
import os
import random
from typing import Any

STRATEGIES = {"value_cliff", "random", "entropy_delta_max"}
BRANCH_TYPE = "sampled_terminal"
TOKEN_WINDOW = 20
TOP_LOGPROBS = 10


class BranchDataError(ValueError):
    """Do not silently downgrade broken entropy capture to an ordinary rollout."""


def selection_policy() -> str:
    policy = os.getenv("AB_BRANCH_SELECTION", "value_cliff")
    if policy not in STRATEGIES:
        raise ValueError(f"invalid AB_BRANCH_SELECTION={policy!r}")
    if policy != "value_cliff" and os.getenv("AB_LOCAL_REWARD_MODE") != "terminal":
        raise ValueError("random/entropy branching requires AB_LOCAL_REWARD_MODE=terminal")
    return policy


def seed_for_group(group_index: int) -> int:
    if type(group_index) is not int or group_index < 0:
        raise ValueError(f"branching requires nonnegative group_index, got {group_index!r}")
    raw = os.getenv("AB_BRANCH_SEED", "42")
    if not raw.isdecimal():
        raise ValueError(f"AB_BRANCH_SEED must be nonnegative, got {raw!r}")
    return int.from_bytes(hashlib.sha256(f"{int(raw)}:{group_index}".encode()).digest()[:8], "big")


def entropy_score(meta: dict, vocab_size: int) -> dict:
    """ARPO-code truncated entropy: first 20 tokens, top10 union sampled token."""
    if not isinstance(meta, dict) or type(vocab_size) is not int or vocab_size <= 1:
        raise BranchDataError("entropy requires meta_info and vocabulary size > 1")
    actual, top = meta.get("output_token_logprobs"), meta.get("output_top_logprobs")
    if not isinstance(actual, list) or not actual or not isinstance(top, list) or len(top) != len(actual):
        raise BranchDataError("missing or misaligned output_token_logprobs/output_top_logprobs")
    values = []
    for offset, (sampled, candidates) in enumerate(zip(actual[:TOKEN_WINDOW], top[:TOKEN_WINDOW], strict=True)):
        if not isinstance(candidates, list) or len(candidates) != TOP_LOGPROBS:
            raise BranchDataError(f"token {offset}: expected {TOP_LOGPROBS} top logprobs")
        probabilities = {}
        for row in [*candidates, sampled]:
            if not isinstance(row, (list, tuple)) or len(row) < 2:
                raise BranchDataError(f"token {offset}: invalid logprob row")
            logprob, token = row[:2]
            if type(token) is not int or not 0 <= token < vocab_size:
                raise BranchDataError(f"token {offset}: invalid token ID {token!r}")
            if (
                isinstance(logprob, bool)
                or not isinstance(logprob, (int, float))
                or not math.isfinite(logprob)
                or logprob > 0
            ):
                raise BranchDataError(f"token {offset}: invalid logprob {logprob!r}")
            if token in probabilities and not math.isclose(probabilities[token], logprob, abs_tol=2e-4):
                raise BranchDataError(f"token {offset}: inconsistent sampled/top probability")
            probabilities[token] = logprob
        if len({row[1] for row in candidates}) != TOP_LOGPROBS:
            raise BranchDataError(f"token {offset}: duplicate top token")
        mass = sum(math.exp(lp) for lp in probabilities.values())
        if not 0 < mass <= 1.0001:
            raise BranchDataError(f"token {offset}: invalid probability mass {mass}")
        values.append(-sum(math.exp(lp) * lp for lp in probabilities.values()))
    return {
        "entropy": sum(values) / math.log(vocab_size),
        "tokens": len(values),
        "weight_version": meta.get("weight_version"),
    }


def candidate_positions(messages: list[dict]) -> list[tuple[int, int]]:
    """Return (message index, 1-based assistant turn), after tools, before a non-final action."""
    if not isinstance(messages, list) or any(not isinstance(m, dict) for m in messages):
        raise ValueError("branch history must be a list of message objects")
    positions = []
    turn = 0
    for index, message in enumerate(messages):
        if message.get("role") != "assistant":
            continue
        turn += 1
        if index and messages[index - 1].get("role") == "tool" and message.get("tool_calls"):
            positions.append((index, turn))
    return positions


def choose_position(
    candidates: list[tuple[int, int]], policy: str, rng: random.Random, scores: dict
) -> tuple[int, int]:
    if not candidates or policy not in {"random", "entropy_delta_max"}:
        raise ValueError("selection requires candidates and a terminal branch policy")
    if policy == "random":
        return rng.choice(candidates)
    if not isinstance(scores, dict):
        raise BranchDataError("entropy scores must be a dictionary")
    initial = scores.get("1")
    if not _valid_entropy(initial):
        raise BranchDataError("missing initial entropy")
    ranked = []
    for index, turn in candidates:
        row = scores.get(str(turn))
        if not _valid_entropy(row):
            raise BranchDataError(f"missing/invalid entropy at turn {turn}")
        ranked.append((row["entropy"] - initial["entropy"], -turn, index, turn))
    _, _, index, turn = max(ranked)
    return index, turn


def _valid_entropy(row: Any) -> bool:
    if not isinstance(row, dict):
        return False
    value = row.get("entropy")
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def generation_cost(records: list[Any]) -> dict[str, int]:
    """Observed API usage, including retries; unknown usage is never reported as measured zero."""
    if not isinstance(records, list):
        raise ValueError("generation records must be a list")
    cost = {"calls": len(records), "prompt_tokens": 0, "completion_tokens": 0, "unknown_usage_calls": 0}
    for index, record in enumerate(records):
        if not isinstance(record.response, dict):
            raise ValueError(f"generation record {index} response must be an object")
        usage = record.response.get("usage")
        if usage is None:
            cost["unknown_usage_calls"] += 1
            continue
        if not isinstance(usage, dict):
            raise ValueError(f"generation record {index} usage must be an object")
        for key in ("prompt_tokens", "completion_tokens"):
            value = usage.get(key)
            if type(value) is not int or value < 0:
                raise ValueError(f"generation record {index} invalid {key}: {value!r}")
            cost[key] += value
    return cost


def annotate_terminal_branch(samples: list[Any], entries: list[tuple], policy: str) -> None:
    if not samples or policy not in {"random", "entropy_delta_max"}:
        raise ValueError("terminal annotation requires a nonempty full group and valid policy")
    ids = {s.group_index for s in samples}
    if len(ids) != 1:
        raise ValueError(f"mixed source group IDs: {ids}")
    seed = seed_for_group(samples[0].group_index)
    rng = random.Random(seed)
    eligible = []
    for _, sample, outcome, _ in entries:
        if sample.metadata.get("agent_excluded_from_training") or sample.remove_sample:
            continue
        history = sample.metadata.get("ab_branch_history")
        if not isinstance(history, list):
            raise BranchDataError(f"missing captured branch history for sample {sample.index}")
        candidates = candidate_positions(history)
        if candidates:
            eligible.append((sample, outcome, history, candidates))
    info = {"policy": policy, "seed": seed, "eligible_parents": len(eligible), "branch_count": 0}
    if eligible:
        eligible.sort(key=lambda item: item[0].index)
        parent, outcome, history, candidates = rng.choice(eligible)
        scores = parent.metadata.get("ab_branch_entropies", {})
        position, turn = choose_position(candidates, policy, rng, scores)
        prefix = copy.deepcopy(history[:position])
        if parent.metadata.get("agent_history_mode") == "keep5":
            keep = parent.metadata.get("agent_keep_tool_results")
            if type(keep) is not int or keep <= 0:
                raise ValueError("keep5 branch requires a positive agent_keep_tool_results")
            tools = [i for i, m in enumerate(prefix) if m.get("role") == "tool"]
            for i in tools[:-keep]:
                prefix[i]["content"] = "Tool result is omitted to save tokens."
        info.update(
            parent_sample_index=parent.index,
            parent_acc=outcome.get("acc"),
            event_turn=turn,
            candidate_count=len(candidates),
            branch_count=1,
            parent_weight_versions=list(parent.weight_versions),
        )
        if policy == "entropy_delta_max":
            info.update(
                initial_entropy=scores["1"],
                selected_entropy=scores[str(turn)],
                entropy_delta=scores[str(turn)]["entropy"] - scores["1"]["entropy"],
            )
        parent.metadata["ab_branch_spec"] = {
            "branch_type": BRANCH_TYPE,
            "prefix_messages": prefix,
            "reward_mode": "terminal",
            "event_turn": turn,
            "selection": copy.deepcopy(info),
            "branch_id": f"{policy}:{parent.group_index}:{parent.index}:{turn}",
        }
    else:
        info["skip_reason"] = "no_eligible_parent"
    for sample in samples:
        sample.metadata["ab_branch_selection"] = copy.deepcopy(info)


def validate_branch_supply(misses: int, target: int) -> None:
    if type(misses) is not int or misses < 0 or type(target) is not int or target <= 0:
        raise ValueError("branch supply counters must be nonnegative with a positive target")
    if misses >= target:
        raise RuntimeError(f"{misses} consecutive full groups have no eligible branch; cannot fill full/local quota")
