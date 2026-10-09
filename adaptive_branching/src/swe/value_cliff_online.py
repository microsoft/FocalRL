"""Attach SWE replay branches to mixed full-outcome groups."""

from __future__ import annotations

import asyncio
import copy
from typing import Any

from adaptive_branching.src.deep_research.judge_client import JudgeClient
from adaptive_branching.src.deep_research.judge_config import judge_settings, positive_int
from adaptive_branching.src.deep_research.value_cliff_locator import build_value_cliff_prefixes
from adaptive_branching.src.deep_research.value_cliff_online import (
    group_is_local,
    local_group_budget,
    local_horizon_max_turns,
    local_rollout_enabled,
)
from adaptive_branching.src.deep_research.value_cliff_rubric import V6_PRM_REWARD_MODE
from adaptive_branching.src.swe.harbor_client import create_trial_replay, get_trial_messages
from adaptive_branching.src.swe.value_cliff_locator import (
    SWE_VALUE_CLIFF_LOCATOR_VERSION,
    SWE_VALUE_CLIFF_REQUIRED_KEYS,
    build_swe_value_cliff_locator_prompt,
    swe_value_cliff_locator_system,
    validate_swe_value_cliff_verdict,
)

_LOCATOR_SEMAPHORES: dict[asyncio.AbstractEventLoop, tuple[int, asyncio.Semaphore]] = {}


async def annotate_swe_value_cliff_branches(args: Any, samples: list[Any], results: list[dict[str, Any]]) -> None:
    del args
    if not samples or group_is_local(samples) or not local_rollout_enabled():
        return
    if len(samples) != len(results):
        raise ValueError("SWE locator requires one full-outcome result per sample")

    for sample in samples:
        metadata = _metadata(sample)
        metadata.pop("ab_branch_spec", None)
        metadata.setdefault("ab_rollout_kind", "full")
        metadata.setdefault("ab_original_question", str(sample.prompt))

    eligible = []
    for sample in samples:
        metadata = _metadata(sample)
        if (
            metadata.get("agent_excluded_from_training")
            or metadata.get("agent_function_failed")
            or getattr(sample, "remove_sample", False)
            or getattr(getattr(sample, "status", None), "name", None) == "ABORTED"
        ):
            continue
        # Locator outcomes are the raw verifier labels, independent of terminal
        # status and training shaping (including single-call truncation penalties).
        raw = metadata.get("reward")
        if type(raw) not in (int, float) or raw not in (0, 1):
            raise ValueError(f"SWE locator sample_index={sample.index}: missing binary verifier reward: {raw!r}")
        eligible.append((sample, raw))
    positives = [entry for entry in eligible if entry[1] == 1]
    negatives = [entry for entry in eligible if entry[1] == 0]
    budget = local_group_budget()
    selected = sorted(negatives, key=lambda entry: entry[0].index)[:budget]
    info = {
        "ab_selection_policy": "swe_raw_verifier_mixed_v2",
        "ab_local_budget": budget,
        "ab_eligible_positive_count": len(positives),
        "ab_eligible_negative_count": len(negatives),
        "ab_locator_attempted_count": len(selected) if positives else 0,
        "ab_locator_event_count": 0,
        "ab_locator_error_count": 0,
        "ab_locator_init_error_count": 0,
        "ab_event_branch_count": 0,
    }
    if not positives or not selected:
        _store_selection_info(samples, info)
        return

    positive = min(positives, key=lambda entry: entry[0].index)[0]
    issue = str(positive.prompt).strip()
    golden_patch = _metadata(positive).get("ab_swe_golden_patch")
    if not isinstance(golden_patch, str) or not golden_patch.strip():
        raise ValueError("mixed SWE group requires metadata.ab_swe_golden_patch")
    positive_messages = await get_trial_messages(_trial_dir(positive))

    settings = judge_settings("locator")
    trace_max_chars = positive_int(settings, "trace_max_chars")
    semaphore = _shared_semaphore(positive_int(settings, "max_concurrency"))
    client = JudgeClient(config_section="locator")
    horizon = local_horizon_max_turns()

    async def localize(sample: Any) -> None:
        failed_messages = await get_trial_messages(_trial_dir(sample))
        prompt, prompt_metadata = build_swe_value_cliff_locator_prompt(
            issue=issue,
            golden_patch=golden_patch,
            failed_messages=failed_messages,
            successful_messages=positive_messages,
            trace_max_chars=trace_max_chars,
        )
        async with semaphore:
            verdict = await client.complete_json(
                swe_value_cliff_locator_system(horizon),
                prompt,
                required_keys=SWE_VALUE_CLIFF_REQUIRED_KEYS,
                tag=f"swe_value_cliff_g{sample.group_index}_s{sample.index}",
                validate=lambda raw: validate_swe_value_cliff_verdict(
                    raw,
                    assistant_turns=prompt_metadata["failed_assistant_turns"],
                ),
            )
        replay = await create_trial_replay(_trial_dir(sample), verdict["selected_turn"])
        before, _ = build_value_cliff_prefixes(failed_messages, verdict["selected_turn"])
        if replay["prefix_messages"] != before or replay["n_calls"] != verdict["selected_turn"] - 1:
            raise RuntimeError("Harbor replay prefix does not match the locator-selected pre-action state")

        metadata = _metadata(sample)
        metadata["swe_replay_path"] = replay["replay_path"]
        metadata["swe_replay_n_calls"] = replay["n_calls"]
        metadata["ab_branch_spec"] = {
            "branch_type": "value_cliff_local",
            "prefix_messages": copy.deepcopy(before),
            "event": copy.deepcopy(verdict),
            "event_turn": verdict["selected_turn"],
            "locator_version": SWE_VALUE_CLIFF_LOCATOR_VERSION,
            "reward_mode": V6_PRM_REWARD_MODE,
            "recovery_rubric": copy.deepcopy(verdict["recovery_rubric"]),
            "value_drop_reason": verdict["value_drop_reason"],
        }

    await asyncio.gather(*(localize(sample) for sample, _ in selected))
    info["ab_locator_event_count"] = len(selected)
    info["ab_event_branch_count"] = len(selected)
    _store_selection_info(samples, info)


def _metadata(sample: Any) -> dict[str, Any]:
    metadata = getattr(sample, "metadata", None)
    if not isinstance(metadata, dict):
        raise TypeError("SWE sample metadata must be a dict")
    return metadata


def _trial_dir(sample: Any) -> str:
    trial_dir = _metadata(sample).get("trial_dir")
    if not isinstance(trial_dir, str) or not trial_dir:
        raise ValueError(f"SWE locator sample_index={sample.index}: missing trial_dir")
    return trial_dir


def _store_selection_info(samples: list[Any], info: dict[str, Any]) -> None:
    for sample in samples:
        _metadata(sample)["ab_event_selection_info"] = copy.deepcopy(info)


def _shared_semaphore(limit: int) -> asyncio.Semaphore:
    loop = asyncio.get_running_loop()
    current = _LOCATOR_SEMAPHORES.get(loop)
    if current is None or current[0] != limit:
        current = (limit, asyncio.Semaphore(limit))
        _LOCATOR_SEMAPHORES[loop] = current
    return current[1]
