"""Full verifier and local-rubric rewards for Harbor-managed SWE rollouts."""

import asyncio
import json
import os
from typing import Any

from adaptive_branching.src.deep_research.judge_client import JudgeClient
from adaptive_branching.src.deep_research.judge_config import judge_settings, positive_int
from adaptive_branching.src.deep_research.value_cliff_online import group_is_local
from adaptive_branching.src.deep_research.value_cliff_reward import build_value_cliff_rubric_judge_prompt
from adaptive_branching.src.deep_research.value_cliff_rubric import (
    VALUE_CLIFF_JUDGE_REQUIRED_KEYS,
    validate_recovery_rubric,
    validate_value_cliff_judge_verdict,
)
from adaptive_branching.src.swe.behavior_guard import hard_failure_reason
from adaptive_branching.src.swe.termination import context_reserve_hit, max_turns_hit, normalize_force_exclude
from adaptive_branching.src.swe.value_cliff_locator import (
    SWE_PRM_REQUIRED_KEYS,
    SWE_PRM_VERSION,
    SWE_TWO_CRITERION_PRM_VERSION,
    prm_behavior_veto_enabled,
    swe_two_criterion_prm_system,
    swe_value_cliff_prm_system,
    validate_swe_prm_verdict,
)
from adaptive_branching.src.swe.value_cliff_online import annotate_swe_value_cliff_branches
from miles.utils.types import Sample

_PRM_SEMAPHORES: dict[asyncio.AbstractEventLoop, tuple[int, asyncio.Semaphore]] = {}


def _sample_reward(sample: Sample) -> float:
    if not isinstance(sample.metadata, dict):
        raise TypeError("sample.metadata must be a dict")
    reward = sample.metadata.get("reward")
    if isinstance(reward, bool) or not isinstance(reward, (int, float)):
        raise TypeError(f"sample.metadata.reward must be numeric, got {reward!r}")
    reward = float(reward)
    if reward not in (0.0, 1.0):
        raise ValueError(f"SWE reward must be binary, got {reward}")
    return reward


def _terminal_result(sample: Sample, *, local: bool = False) -> dict[str, Any] | None:
    """Resolve unusable samples and terminal model caps before asking any judge."""
    metadata = sample.metadata
    if not isinstance(metadata, dict):
        raise TypeError(f"SWE sample {sample.index}: metadata must be a dict")
    mode = normalize_force_exclude(os.environ.get("FORCE_EXCLUDE", "ALL"))
    reserve_hit = context_reserve_hit(metadata)
    turns_hit = max_turns_hit(metadata)
    metadata["agent_context_reserve_hit"] = reserve_hit
    metadata["agent_max_turns_hit"] = turns_hit
    length = metadata.get("agent_last_finish_reason") == "length"
    excluded = (
        sample.remove_sample
        or metadata.get("agent_excluded_from_training")
        or metadata.get("agent_function_failed")
        or (not local and metadata.get("exit_status") == "LimitsExceeded")
        or sample.status == Sample.Status.ABORTED
        or (sample.status == Sample.Status.TRUNCATED and not length)
        or (length and (reserve_hit or turns_hit or metadata.get("agent_forced_final_answer")))
    )
    forced = reserve_hit or turns_hit or bool(metadata.get("agent_forced_final_answer"))
    if forced and not excluded:
        if local or mode == "all":
            excluded = True
        elif mode == "wrong":
            excluded = _sample_reward(sample) == 0.0
    if not excluded and not length:
        return None
    if length and not excluded:
        assert (
            sample.status == Sample.Status.TRUNCATED
        ), f"SWE sample {sample.index}: terminal length requires TRUNCATED, got {sample.status}"
    sample.remove_sample = bool(excluded)
    metadata["agent_excluded_from_training"] = bool(excluded)
    metadata["agent_single_call_length_penalized"] = not excluded
    result = {
        "score": 0.0,
        "acc": False,
        "pred": str(metadata.get("exit_status") or ""),
        "judge_raw": "excluded_from_training" if excluded else "single_call_length",
    }
    if local:
        metadata["ab_local_trainable"] = not excluded
        result["event_rubric_state"] = "excluded" if excluded else "avoid_failed"
    return result


def _full_result(sample: Sample) -> dict[str, Any]:
    terminal = _terminal_result(sample)
    if terminal is not None:
        return terminal
    reward = _sample_reward(sample)
    return {
        "score": reward,
        "acc": reward == 1.0,
        "pred": str(sample.metadata.get("exit_status") or ""),
        "judge_raw": "harbor_verifier",
    }


async def reward_func(args, samples: Sample | list[Sample], **kwargs) -> dict[str, Any] | list[dict[str, Any]]:
    del kwargs
    is_batch = isinstance(samples, list)
    batch = samples if is_batch else [samples]
    if not batch:
        raise ValueError("samples must not be empty")

    if group_is_local(batch):
        results = await _score_local_group(batch)
    else:
        results = [_full_result(sample) for sample in batch]
        await annotate_swe_value_cliff_branches(args, batch, results)
    return results if is_batch else results[0]


async def _score_local_group(samples: list[Sample], *, unpenalized: bool = False) -> list[dict[str, Any]]:
    if not samples:
        raise ValueError("local SWE group must not be empty")
    if type(unpenalized) is not bool:
        raise TypeError("unpenalized must be bool")
    behavior_veto = prm_behavior_veto_enabled()
    validate_verdict = validate_swe_prm_verdict if behavior_veto else validate_value_cliff_judge_verdict
    prompt_system = swe_value_cliff_prm_system if behavior_veto else swe_two_criterion_prm_system
    required_keys = SWE_PRM_REQUIRED_KEYS if behavior_veto else VALUE_CLIFF_JUDGE_REQUIRED_KEYS
    judge_version = SWE_PRM_VERSION if behavior_veto else SWE_TWO_CRITERION_PRM_VERSION
    results = [
        _unpenalized_local_terminal(sample) if unpenalized else _terminal_result(sample, local=True)
        for sample in samples
    ]
    judge_indices = [index for index, result in enumerate(results) if result is None]
    if not judge_indices:
        for sample in samples:
            sample.metadata["ab_event_local_group_informative"] = False
        return results

    horizons = {_local_horizon(samples[index]) for index in judge_indices}
    if len(horizons) != 1:
        raise ValueError(f"local SWE group mixes horizons: {sorted(horizons)}")
    horizon = horizons.pop()
    settings = judge_settings("local_prm")
    client = JudgeClient(config_section="local_prm")
    semaphore = _shared_prm_semaphore(positive_int(settings, "max_concurrency"))
    trace_max_chars = positive_int(settings, "trace_max_chars")

    async def score_one(sample: Sample) -> dict[str, Any]:
        metadata = sample.metadata
        messages = metadata.get("messages")
        prefix_count = metadata.get("ab_local_prefix_message_count")
        if not isinstance(messages, list) or not messages:
            raise ValueError("local SWE sample has no trajectory messages")
        if isinstance(prefix_count, bool) or not isinstance(prefix_count, int) or not 0 < prefix_count < len(messages):
            raise ValueError("local SWE sample has an invalid prefix boundary")
        prefix = messages[:prefix_count]
        continuation = messages[prefix_count:]
        turns = sum(message.get("role") == "assistant" for message in continuation)
        if not 1 <= turns <= horizon:
            raise ValueError(f"local SWE continuation must contain 1-{horizon} assistant turns, got {turns}")
        rubric = validate_recovery_rubric(metadata.get("ab_event_recovery_rubric"))
        prompt, prefix_truncated, continuation_truncated = build_value_cliff_rubric_judge_prompt(
            prefix_messages=prefix,
            local_messages=continuation,
            recovery_rubric=rubric,
            trace_max_chars=trace_max_chars,
        )
        async with semaphore:
            verdict = await client.complete_json(
                prompt_system(local_turns=horizon),
                prompt,
                required_keys=required_keys,
                tag=f"swe_local_prm_g{sample.group_index}_s{sample.index}",
                validate=lambda raw: validate_verdict(raw, local_turns=turns),
            )
        verdict = validate_verdict(verdict, local_turns=turns)
        score = float(verdict["score"])
        state = "redirected" if score == 1.0 else ("avoid_only" if score == 0.5 else "avoid_failed")
        metadata.update(
            ab_event_rubric_judge=verdict,
            ab_event_rubric_judge_version=judge_version,
            agent_behavior_anomaly_penalized=verdict["behavior_anomaly"] if behavior_veto else False,
            ab_event_rubric_prefix_truncated=prefix_truncated,
            ab_event_rubric_continuation_truncated=continuation_truncated,
            ab_local_trainable=True,
            agent_excluded_from_training=False,
        )
        return {
            "score": score,
            "acc": score == 1.0,
            "pred": verdict["reason"],
            "judge_raw": json.dumps(verdict, ensure_ascii=False),
            "event_rubric_state": state,
        }

    judged = await asyncio.gather(*(score_one(samples[index]) for index in judge_indices))
    for index, result in zip(judge_indices, judged, strict=True):
        results[index] = result
    informative = (
        len({result["score"] for sample, result in zip(samples, results, strict=True) if not sample.remove_sample}) > 1
    )
    for sample in samples:
        sample.metadata["ab_event_local_group_informative"] = informative
    return results


def _unpenalized_local_terminal(sample):
    metadata = sample.metadata
    if not isinstance(metadata, dict):
        raise TypeError("local PRM metadata must be an object")
    excluded = bool(
        sample.remove_sample
        or metadata.get("agent_excluded_from_training")
        or metadata.get("agent_function_failed")
        or sample.status == Sample.Status.ABORTED
    )
    if excluded:
        sample.remove_sample = True
        metadata["ab_local_trainable"] = False
        return {
            "score": 0.0,
            "acc": False,
            "pred": str(metadata.get("exit_status", "")),
            "judge_raw": "excluded_from_training",
            "event_rubric_state": "excluded",
        }
    if metadata.get("swe_local_prm_only") is not True:
        raise ValueError("unpenalized local PRM requires audited continuation")
    failure = hard_failure_reason(sample)
    if failure is not None:
        # A model format failure is a trainable zero, not an infrastructure
        # exclusion. Never let semantic progress override an unusable action.
        metadata.update(
            ab_local_trainable=True,
            agent_excluded_from_training=False,
            agent_format_failure_penalized=failure != "single_call_length",
            agent_single_call_length_penalized=failure == "single_call_length",
        )
        return {
            "score": 0.0,
            "acc": False,
            "pred": str(metadata.get("exit_status", "")),
            "judge_raw": failure,
            "event_rubric_state": "avoid_failed",
        }
    # Ordinary H completion/context exhaustion still go to PRM; malformed
    # generated content and single-call truncation are trainable zeroes.
    return None


def _local_horizon(sample: Sample) -> int:
    if not isinstance(sample.metadata, dict):
        raise TypeError("sample.metadata must be a dict")
    horizon = sample.metadata.get("agent_max_turns")
    if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon <= 0:
        raise ValueError("local SWE sample requires positive agent_max_turns")
    return horizon


def _shared_prm_semaphore(limit: int) -> asyncio.Semaphore:
    loop = asyncio.get_running_loop()
    current = _PRM_SEMAPHORES.get(loop)
    if current is None or current[0] != limit:
        current = (limit, asyncio.Semaphore(limit))
        _PRM_SEMAPHORES[loop] = current
    return current[1]
