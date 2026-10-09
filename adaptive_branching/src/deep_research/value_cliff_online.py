"""Create value-cliff local branches from matched full-rollout groups."""

from __future__ import annotations

import asyncio
import copy
import logging
import os
from typing import Any

from adaptive_branching.src.deep_research.value_cliff_locator import (
    ValueCliffLocalizationResult,
    build_value_cliff_prefixes,
    locate_value_cliff,
    locator_policy_version,
    value_cliff_local_horizon,
)
from adaptive_branching.src.deep_research.judge_client import JudgeClient
from adaptive_branching.src.deep_research.judge_config import judge_settings, positive_int
from adaptive_branching.src.deep_research.value_cliff_rubric import configured_local_reward_mode, validate_recovery_rubric

logger = logging.getLogger(__name__)

_DEFAULT_LOCAL_GROUP_BUDGET = 8
_SELECTION_POLICY = "matched_positive_value_cliff_v1"
_SYNTHETIC_FINAL_PROMPT_PREFIX = (
    "Summarize the above conversation, and output the FINAL ANSWER to the original question."
)
_INVALID_SOURCE_FINISH_REASONS = {
    "abort",
    "length",
    "forced_answer_fail",
    "forced_answer_empty",
    "forced_answer_skipped_no_budget",
}
_LOCATOR_SEMAPHORES: dict[asyncio.AbstractEventLoop, tuple[int, asyncio.Semaphore]] = {}


def local_rollout_enabled() -> bool:
    return _truthy(os.getenv("AB_LOCAL_ROLLOUT_ENABLE", "0"))


def local_group_budget() -> int:
    return _nonnegative_int("AB_LOCAL_MAX_GROUPS_PER_FULL_GROUP", default=_DEFAULT_LOCAL_GROUP_BUDGET)


def local_horizon_max_turns() -> int:
    """Compatibility-free accessor for the retained horizon environment variable."""

    return value_cliff_local_horizon()


def full_rollout_max_turns() -> int:
    return _positive_int("AGENT_MAX_TURNS", default=30)


def is_local_sample(sample: Any) -> bool:
    metadata = getattr(sample, "metadata", None)
    if metadata is None:
        return False
    if not isinstance(metadata, dict):
        raise TypeError(
            f"sample metadata must be a dict or None, got {type(metadata).__name__} "
            f"for sample_index={_sample_index(sample)!r}"
        )
    return bool(metadata.get("ab_local_rollout")) or metadata.get("ab_rollout_kind") == "local"


def group_is_local(samples: list[Any]) -> bool:
    if not isinstance(samples, list):
        raise TypeError("samples must be a list")
    return any(is_local_sample(sample) for sample in samples)


async def annotate_value_cliff_branches(
    args: Any,
    samples: list[Any],
    results: list[dict[str, Any]],
) -> None:
    """Attach one current-schema value-cliff branch to every selected failure."""

    del args
    if not samples or group_is_local(samples) or not local_rollout_enabled():
        return

    _clear_stale_annotations(samples)
    _mark_full_group(samples)
    from adaptive_branching.src.deep_research.branch_sampling import annotate_terminal_branch, selection_policy

    policy = selection_policy()
    if policy != "value_cliff":
        annotate_terminal_branch(samples, _reference_entries(samples, results), policy)
        return
    budget = local_group_budget()
    reward_mode = configured_local_reward_mode()
    entries = _reference_entries(samples, results)
    orm_positives = [entry for entry in entries if bool(entry[2].get("acc"))]
    positives = [entry for entry in orm_positives if not _sample_has_synthetic_final(entry[1], entry[3])]
    negatives = sorted(
        (
            entry
            for entry in entries
            if not bool(entry[2].get("acc")) and _locator_negative_eligible(entry[1])
        ),
        key=lambda entry: _entry_sort_key(entry[0], entry[1]),
    )
    info: dict[str, Any] = {
        "ab_selection_policy": _SELECTION_POLICY,
        "ab_event_locator_version": locator_policy_version(),
        "ab_local_reward_mode": reward_mode,
        "ab_local_budget": budget,
        "ab_locator_concurrency": _locator_concurrency(),
        "ab_positive_count": len(positives),
        "ab_orm_positive_count": len(orm_positives),
        "ab_positive_reference_index": None,
        "ab_negative_count": len(negatives),
        "ab_negative_attempt_order": [],
        "ab_event_negative_indices": [],
        "ab_failed_negative_indices": [],
        "ab_locator_attempted_count": 0,
        "ab_locator_event_count": 0,
        "ab_locator_error_count": 0,
        "ab_locator_init_error_count": 0,
        "ab_event_branch_count": 0,
    }
    if budget == 0:
        info["ab_skipped_reason"] = "zero_local_budget"
        _store_selection_info(samples, info)
        return
    if not positives or not negatives:
        info["ab_skipped_reason"] = "missing_positive_or_negative"
        _store_selection_info(samples, info)
        return

    selected_negatives = negatives[:budget]
    info["ab_negative_attempt_order"] = [_sample_index(entry[1]) for entry in selected_negatives]
    info["ab_locator_attempted_count"] = len(selected_negatives)
    positive_reference = min(positives, key=lambda entry: _entry_sort_key(entry[0], entry[1]))
    info["ab_positive_reference_index"] = _sample_index(positive_reference[1])
    question = _original_question_for_sample(positive_reference[1])
    ground_truth = str(getattr(positive_reference[1], "label", "") or "").strip()
    if not ground_truth:
        raise ValueError("matched positive sample must have a non-empty label")
    positive_messages = positive_reference[3]

    try:
        client = _make_locator_client()
    except Exception as exc:  # noqa: BLE001 - full-outcome training remains valid
        logger.warning("value-cliff locator initialization failed: %s", exc, exc_info=True)
        info["ab_locator_init_error"] = _short_error(exc)
        info["ab_locator_init_error_count"] = 1
        info["ab_failed_negative_indices"] = [_sample_index(entry[1]) for entry in selected_negatives]
        _finalize_selection_counts(info)
        _store_selection_info(samples, info)
        return

    semaphore = _shared_locator_semaphore(_locator_concurrency())

    async def locate_one(entry: tuple[int, Any, dict[str, Any], list[dict[str, Any]]]):
        _, sample, _, negative_messages = entry
        sample_index = _sample_index(sample)
        try:
            async with semaphore:
                localization = await locate_value_cliff(
                    client,
                    question=question,
                    ground_truth=ground_truth,
                    positive_messages=positive_messages,
                    negative_messages=negative_messages,
                    tag=f"value_cliff_g{_group_index(sample)}_s{sample_index}",
                    trace_max_chars=_locator_trace_max_chars(),
                )
            return entry, localization, None
        except Exception as exc:  # noqa: BLE001 - isolate one failed trajectory
            logger.warning("value-cliff localization failed for sample %s: %s", sample_index, exc, exc_info=True)
            return entry, None, exc

    localized = await asyncio.gather(*(locate_one(entry) for entry in selected_negatives))
    for entry, localization, error in localized:
        _, sample, _, negative_messages = entry
        sample_index = _sample_index(sample)
        if error is not None or not isinstance(localization, ValueCliffLocalizationResult):
            info["ab_failed_negative_indices"].append(sample_index)
            continue
        event = localization.event
        diagnostics = localization.diagnostics
        try:
            event_turn = int(event["event_turn"])
            before_event, _ = build_value_cliff_prefixes(negative_messages, event_turn)
            rubric = validate_recovery_rubric(event.get("recovery_rubric"), name="event.recovery_rubric")
            reason = str(event.get("value_drop_reason") or "").strip()
            if not reason:
                raise ValueError("locator result is missing value_drop_reason")
        except Exception as exc:  # noqa: BLE001 - malformed output affects one failure only
            logger.warning("failed to build value-cliff branch for sample %s: %s", sample_index, exc, exc_info=True)
            info["ab_failed_negative_indices"].append(sample_index)
            continue
        _sample_metadata(sample)["ab_branch_spec"] = {
            "branch_type": "value_cliff_local",
            "prefix_messages": before_event,
            "event": copy.deepcopy(event),
            "event_turn": event_turn,
            "locator_version": str(diagnostics.get("policy") or locator_policy_version()),
            "reward_mode": reward_mode,
            "recovery_rubric": rubric,
            "value_drop_reason": reason,
        }
        info["ab_event_negative_indices"].append(sample_index)
        info["ab_locator_event_count"] += 1

    _finalize_selection_counts(info)
    _store_selection_info(samples, info)


def _make_locator_client() -> JudgeClient:
    return JudgeClient(config_section="locator")


def _locator_trace_max_chars() -> int:
    return positive_int(judge_settings("locator"), "trace_max_chars")


def _locator_concurrency() -> int:
    return positive_int(judge_settings("locator"), "max_concurrency")


def _reference_entries(
    samples: list[Any],
    results: list[dict[str, Any]],
) -> list[tuple[int, Any, dict[str, Any], list[dict[str, Any]]]]:
    if len(samples) != len(results):
        raise ValueError(
            f"value-cliff annotation requires one result per sample, got samples={len(samples)} results={len(results)}"
        )
    entries = []
    for position, (sample, result) in enumerate(zip(samples, results, strict=True)):
        metadata = getattr(sample, "metadata", None)
        if not isinstance(metadata, dict):
            raise TypeError(
                f"sample metadata must be a dict during value-cliff annotation, got {type(metadata).__name__} "
                f"for sample_index={_sample_index(sample)!r}"
            )
        if not isinstance(result, dict) or result.get("judge_error") or result.get("judge_raw") == "missing_response":
            continue
        if metadata.get("outcome_judge_failed") or metadata.get("agent_function_failed"):
            continue
        status = getattr(getattr(sample, "status", ""), "value", getattr(sample, "status", ""))
        if str(status).strip().lower() in {"aborted", "truncated", "failed", "error"}:
            continue
        finish_reason = str(metadata.get("agent_last_finish_reason") or "").strip().lower()
        if finish_reason in _INVALID_SOURCE_FINISH_REASONS:
            continue
        messages = metadata.get("messages")
        if isinstance(messages, list) and messages:
            entries.append((position, sample, result, messages))
    return entries


def _locator_negative_eligible(sample: Any) -> bool:
    return not _sample_training_excluded(sample) or _forced_context_negative_eligible(sample)


def _sample_training_excluded(sample: Any) -> bool:
    metadata = getattr(sample, "metadata", None) or {}
    return bool(metadata.get("agent_excluded_from_training")) or bool(getattr(sample, "remove_sample", False))


def _forced_context_negative_eligible(sample: Any) -> bool:
    metadata = getattr(sample, "metadata", None) or {}
    if not (
        metadata.get("agent_forced_final_answer")
        and metadata.get("agent_context_reserve_hit")
        and str(metadata.get("agent_forced_final_answer_reason") or "").strip().lower() == "context_reserve"
        and str(metadata.get("agent_last_finish_reason") or "").strip().lower() == "stop"
    ):
        return False
    if any(
        metadata.get(key)
        for key in (
            "agent_forced_final_answer_failed",
            "agent_forced_final_answer_skipped_no_budget",
            "agent_function_failed",
            "outcome_judge_failed",
        )
    ):
        return False
    return _has_trailing_synthetic_final_exchange(metadata.get("messages"))


def _has_trailing_synthetic_final_exchange(messages: Any) -> bool:
    if not isinstance(messages, list) or len(messages) < 2:
        return False
    synthetic_user, forced_assistant = messages[-2:]
    return bool(
        isinstance(synthetic_user, dict)
        and synthetic_user.get("role") == "user"
        and str(synthetic_user.get("content") or "").startswith(_SYNTHETIC_FINAL_PROMPT_PREFIX)
        and isinstance(forced_assistant, dict)
        and forced_assistant.get("role") == "assistant"
        and str(forced_assistant.get("content") or "").strip()
    )


def _sample_has_synthetic_final(sample: Any, messages: list[dict[str, Any]]) -> bool:
    metadata = getattr(sample, "metadata", None) or {}
    return bool(metadata.get("agent_forced_final_answer")) or _has_trailing_synthetic_final_exchange(messages)


def _clear_stale_annotations(samples: list[Any]) -> None:
    for sample in samples:
        metadata = _sample_metadata(sample)
        metadata.pop("ab_branch_spec", None)
        metadata.pop("ab_event_selection_info", None)


def _shared_locator_semaphore(limit: int) -> asyncio.Semaphore:
    loop = asyncio.get_running_loop()
    existing = _LOCATOR_SEMAPHORES.get(loop)
    if existing is None or existing[0] != limit:
        existing = (limit, asyncio.Semaphore(limit))
        _LOCATOR_SEMAPHORES[loop] = existing
    return existing[1]


def _original_question_for_sample(sample: Any) -> str:
    metadata = getattr(sample, "metadata", None) or {}
    if metadata.get("ab_original_question"):
        return str(metadata["ab_original_question"])
    prompt = getattr(sample, "prompt", "")
    if isinstance(prompt, list):
        for message in prompt:
            if isinstance(message, dict) and message.get("role") == "user":
                return str(message.get("content") or "")
    return str(prompt)


def _mark_full_group(samples: list[Any]) -> None:
    for sample in samples:
        metadata = _sample_metadata(sample)
        metadata.setdefault("ab_rollout_kind", "full")
        metadata.setdefault("ab_original_question", _original_question_for_sample(sample))


def _store_selection_info(samples: list[Any], info: dict[str, Any]) -> None:
    for sample in samples:
        _sample_metadata(sample)["ab_event_selection_info"] = copy.deepcopy(info)


def _finalize_selection_counts(info: dict[str, Any]) -> None:
    info["ab_locator_error_count"] = len(info["ab_failed_negative_indices"])
    info["ab_event_branch_count"] = len(info["ab_event_negative_indices"])
    info["ab_branch_count"] = info["ab_event_branch_count"]


def _sample_metadata(sample: Any) -> dict[str, Any]:
    metadata = getattr(sample, "metadata", None)
    if metadata is None:
        metadata = {}
        sample.metadata = metadata
    elif not isinstance(metadata, dict):
        raise TypeError(
            f"sample metadata must be a dict or None, got {type(metadata).__name__} "
            f"for sample_index={_sample_index(sample)!r}"
        )
    return metadata


def _entry_sort_key(position: int, sample: Any) -> tuple[bool, int, int]:
    index = _sample_index(sample)
    return index is None, index if index is not None else 0, position


def _sample_index(sample: Any) -> int | None:
    try:
        value = getattr(sample, "index", None)
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _group_index(sample: Any) -> Any:
    return getattr(sample, "group_index", "group")


def _positive_int(name: str, *, default: int) -> int:
    raw = os.getenv(name, str(default))
    try:
        value = int(str(raw).strip())
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")
    return value


def _nonnegative_int(name: str, *, default: int) -> int:
    raw = os.getenv(name, str(default))
    try:
        value = int(str(raw).strip())
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc
    if value < 0:
        raise ValueError(f"{name} must be nonnegative, got {value}")
    return value


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").strip().lower() in {"1", "true", "yes", "y", "on"}


def _short_error(exc: Exception) -> str:
    return f"{type(exc).__name__}: {exc}"[:500]
