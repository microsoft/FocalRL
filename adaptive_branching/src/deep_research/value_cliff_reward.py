"""Score value-cliff local groups with V6, V6 Plus, V7, or terminal rewards."""

from __future__ import annotations

import asyncio
import json
from typing import Any

from adaptive_branching.src.deep_research.value_cliff_locator import balanced_trace_budgets, render_message_turns
from adaptive_branching.src.deep_research.judge_client import JudgeClient
from adaptive_branching.src.deep_research.judge_config import judge_settings, positive_int
from adaptive_branching.src.deep_research.value_cliff_rubric import (
    VALUE_CLIFF_JUDGE_REQUIRED_KEYS,
    VALUE_CLIFF_JUDGE_VERSION,
    VALUE_CLIFF_OUTCOME4_REWARD_VERSION,
    VALUE_CLIFF_V6P_REWARD_VERSION,
    TERMINAL_REWARD_MODE,
    V6P_REWARD_MODE,
    V6_PRM_REWARD_MODE,
    V7_PRM_REWARD_MODE,
    validate_recovery_rubric,
    validate_value_cliff_judge_verdict,
    value_cliff_judge_system,
)

_INVALID_FINISH_REASONS = {
    "abort",
    "length",
    "forced_answer_fail",
    "forced_answer_empty",
    "forced_answer_skipped_no_budget",
    "local_context_reserve",
}
_JUDGE_SEMAPHORES: dict[asyncio.AbstractEventLoop, tuple[int, asyncio.Semaphore]] = {}


async def score_value_cliff_group(samples: list[Any]) -> list[dict[str, Any]]:
    reward_mode = _group_reward_mode(samples)
    if reward_mode == V6_PRM_REWARD_MODE:
        return await _score_value_cliff_rubric_group(samples)
    if reward_mode == V6P_REWARD_MODE:
        return await _score_value_cliff_hybrid_group(samples, reward_mode=reward_mode)
    if reward_mode == V7_PRM_REWARD_MODE:
        return await _score_value_cliff_hybrid_group(samples, reward_mode=reward_mode)
    if reward_mode == TERMINAL_REWARD_MODE:
        return await _score_full_outcome_samples(samples)
    raise AssertionError(f"unhandled local reward mode: {reward_mode}")


async def _score_value_cliff_rubric_group(samples: list[Any]) -> list[dict[str, Any]]:
    try:
        horizon = _group_local_horizon(samples)
        prepared = [
            _prepare_value_cliff_judge_input(
                sample,
                horizon=horizon,
            )
            for sample in samples
        ]
        length_indices = [index for index, sample in enumerate(samples) if is_single_call_length_truncation(sample)]
        length_index_set = set(length_indices)
        for index, sample in enumerate(samples):
            _metadata(sample)["agent_single_call_length_penalized"] = index in length_index_set
            if index in length_index_set:
                continue
            reason = _structural_invalid_reason(sample)
            if reason is not None:
                raise ValueError(reason)
    except Exception as exc:  # noqa: BLE001 - one malformed row invalidates its GRPO group
        return _exclude_group(samples, f"event_rubric_input_error: {type(exc).__name__}: {exc}")

    judge_indices = [index for index in range(len(samples)) if index not in length_index_set]
    verdicts = await _judge_value_cliff_rubric_replays(
        [samples[index] for index in judge_indices],
        [prepared[index] for index in judge_indices],
        horizon=horizon,
    )
    failures = [verdict for verdict in verdicts if isinstance(verdict, BaseException)]
    if failures:
        summary = "; ".join(f"{type(exc).__name__}: {exc}" for exc in failures[:3])
        return _exclude_group(samples, f"event_rubric_judge_error: {summary}")

    decisions: list[dict[str, Any] | None] = [None] * len(samples)
    state_by_score = {0.0: "avoid_failed", 0.5: "avoid_only", 1.0: "redirected"}
    for index in length_indices:
        decisions[index] = _single_call_length_decision()
    for index, verdict in zip(judge_indices, verdicts, strict=True):
        assert isinstance(verdict, dict)
        score = float(verdict["score"])
        decisions[index] = {
            "score": score,
            "acc": score == 1.0,
            "pred": "",
            "judge_raw": json.dumps(verdict, ensure_ascii=False),
            "state": state_by_score[score],
            "judge_metadata": {**verdict, "score": score},
        }
    if any(decision is None for decision in decisions):
        raise RuntimeError("value-cliff rubric scorer left a sample without a decision")
    return _finalize_value_cliff_group(
        samples,
        prepared,
        [decision for decision in decisions if decision is not None],
        judge_version=VALUE_CLIFF_JUDGE_VERSION,
        allowed_scores={0.0, 0.5, 1.0},
    )


async def _score_value_cliff_hybrid_group(
    samples: list[Any],
    *,
    reward_mode: str,
) -> list[dict[str, Any]]:
    if reward_mode not in {V6P_REWARD_MODE, V7_PRM_REWARD_MODE}:
        raise ValueError(f"hybrid local scorer requires v6p or v7_prm mode, got {reward_mode!r}")
    if _group_reward_mode(samples) != reward_mode:
        raise ValueError("hybrid local scorer reward_mode does not match sample metadata")
    try:
        horizon = _group_local_horizon(samples)
        prepared = [
            _prepare_value_cliff_judge_input(
                sample,
                horizon=horizon,
            )
            for sample in samples
        ]
        length_indices = [index for index, sample in enumerate(samples) if is_single_call_length_truncation(sample)]
        length_index_set = set(length_indices)
        for index, sample in enumerate(samples):
            _metadata(sample)["agent_single_call_length_penalized"] = index in length_index_set
            if index in length_index_set:
                continue
            reason = _structural_invalid_reason(sample)
            if reason is not None:
                raise ValueError(reason)
    except Exception as exc:  # noqa: BLE001 - one malformed row invalidates its GRPO group
        return _exclude_group(samples, f"event_rubric_input_error: {type(exc).__name__}: {exc}")

    natural_indices = [index for index, sample in enumerate(samples) if _is_natural_local_finish(sample)]
    natural_index_set = set(natural_indices)
    unfinished_indices = [
        index for index in range(len(samples)) if index not in natural_index_set and index not in length_index_set
    ]
    decisions: list[dict[str, Any] | None] = [None] * len(samples)
    for index in length_indices:
        decisions[index] = _single_call_length_decision()

    if natural_indices:
        try:
            outcome_results = await _score_full_outcome_samples([samples[index] for index in natural_indices])
        except Exception as exc:  # noqa: BLE001 - one failed judge invalidates its GRPO group
            return _exclude_group(
                samples,
                f"event_rubric_outcome_judge_error: {type(exc).__name__}: {exc}",
            )
        if len(outcome_results) != len(natural_indices):
            return _exclude_group(samples, "event_rubric_outcome_judge_error: result count mismatch")
        try:
            for index, result in zip(natural_indices, outcome_results, strict=True):
                outcome_correct = _validated_full_outcome_correct(result)
                score, state = map_value_cliff_outcome4_reward(
                    natural_finished=True,
                    outcome_correct=outcome_correct,
                )
                decisions[index] = {
                    "score": score,
                    "acc": outcome_correct,
                    "pred": str(result.get("pred") or ""),
                    "judge_raw": str(result.get("judge_raw") or ""),
                    "state": state,
                    "natural_finish": True,
                    "outcome_result": dict(result),
                    "judge_metadata": {
                        "reward_source": "full_outcome",
                        "outcome_correct": outcome_correct,
                        "score": score,
                    },
                }
        except (TypeError, ValueError) as exc:
            return _exclude_group(
                samples,
                f"event_rubric_outcome_judge_error: {type(exc).__name__}: {exc}",
            )

    if unfinished_indices:
        rubric_verdicts = await _judge_value_cliff_rubric_replays(
            [samples[index] for index in unfinished_indices],
            [prepared[index] for index in unfinished_indices],
            horizon=horizon,
        )
        failures = [verdict for verdict in rubric_verdicts if isinstance(verdict, BaseException)]
        if failures:
            summary = "; ".join(f"{type(exc).__name__}: {exc}" for exc in failures[:3])
            return _exclude_group(samples, f"event_rubric_judge_error: {summary}")
        for index, verdict in zip(unfinished_indices, rubric_verdicts, strict=True):
            assert isinstance(verdict, dict)
            if reward_mode == V6P_REWARD_MODE:
                score = float(verdict["score"])
                state = {0.0: "avoid_failed", 0.5: "avoid_only", 1.0: "redirected"}[score]
            else:
                score, state = map_value_cliff_outcome4_reward(
                    natural_finished=False,
                    avoid_error_met=verdict["avoid_error_met"],
                    redirect_met=verdict["redirect_met"],
                )
            decisions[index] = {
                "score": score,
                "acc": reward_mode == V6P_REWARD_MODE and score == 1.0,
                "pred": "",
                "judge_raw": json.dumps(verdict, ensure_ascii=False),
                "state": state,
                "natural_finish": _is_natural_local_finish(samples[index]),
                "judge_metadata": {
                    **verdict,
                    "reward_source": "local_rubric",
                    "rubric_score": float(verdict["score"]),
                    "score": score,
                },
            }

    if any(decision is None for decision in decisions):
        raise RuntimeError("value-cliff outcome4 scorer left a sample without a decision")
    return _finalize_value_cliff_group(
        samples,
        prepared,
        [decision for decision in decisions if decision is not None],
        judge_version=(
            VALUE_CLIFF_V6P_REWARD_VERSION if reward_mode == V6P_REWARD_MODE else VALUE_CLIFF_OUTCOME4_REWARD_VERSION
        ),
        allowed_scores=({0.0, 0.5, 1.0} if reward_mode == V6P_REWARD_MODE else {0.0, 0.25, 0.5, 1.0}),
    )


async def _judge_value_cliff_rubric_replays(
    samples: list[Any],
    prepared: list[tuple[str, int, bool, bool]],
    *,
    horizon: int,
) -> list[dict[str, Any] | BaseException]:
    if len(samples) != len(prepared):
        raise ValueError(f"rubric judge input mismatch: samples={len(samples)} prepared={len(prepared)}")
    if not samples:
        return []
    client = JudgeClient(config_section="local_prm")
    semaphore = _shared_judge_semaphore(positive_int(judge_settings("local_prm"), "max_concurrency"))
    system_prompt = value_cliff_judge_system(horizon)

    async def judge_one(index: int, sample: Any, prepared_input: tuple[str, int, bool, bool]):
        prompt, num_turns, _, _ = prepared_input
        async with semaphore:
            return await client.complete_json(
                system_prompt,
                prompt,
                required_keys=VALUE_CLIFF_JUDGE_REQUIRED_KEYS,
                tag=f"event_rubric_g{getattr(sample, 'group_index', 'x')}_s{getattr(sample, 'index', index)}",
                validate=lambda raw, turns=num_turns: validate_value_cliff_judge_verdict(raw, local_turns=turns),
            )

    return await asyncio.gather(
        *(
            judge_one(index, sample, prepared_input)
            for index, (sample, prepared_input) in enumerate(zip(samples, prepared, strict=True))
        ),
        return_exceptions=True,
    )


def _finalize_value_cliff_group(
    samples: list[Any],
    prepared: list[tuple[str, int, bool, bool]],
    decisions: list[dict[str, Any]],
    *,
    judge_version: str,
    allowed_scores: set[float],
) -> list[dict[str, Any]]:
    if not (len(samples) == len(prepared) == len(decisions)):
        raise ValueError(
            f"value-cliff result mismatch: samples={len(samples)} prepared={len(prepared)} decisions={len(decisions)}"
        )
    if not judge_version.strip():
        raise ValueError("judge_version must be non-empty")
    scores = [float(decision["score"]) for decision in decisions]
    if any(score not in allowed_scores for score in scores):
        raise RuntimeError(f"value-cliff judge returned scores outside {sorted(allowed_scores)}: {scores}")
    informative = len(set(scores)) >= 2
    results: list[dict[str, Any]] = []
    for sample, decision, prepared_input in zip(samples, decisions, prepared, strict=True):
        score = float(decision["score"])
        _, _, prefix_truncated, continuation_truncated = prepared_input
        metadata = _metadata(sample)
        metadata["ab_event_rubric_judge"] = dict(decision["judge_metadata"])
        metadata["ab_event_rubric_judge_version"] = judge_version
        metadata["ab_event_rubric_prefix_truncated"] = prefix_truncated
        metadata["ab_event_rubric_continuation_truncated"] = continuation_truncated
        metadata["ab_event_local_group_informative"] = informative
        metadata["ab_local_trainable"] = True
        metadata.pop("ab_event_rubric_judge_error", None)
        metadata["agent_excluded_from_training"] = False
        sample.remove_sample = False
        if "natural_finish" in decision:
            metadata["ab_event_local_natural_finish"] = decision["natural_finish"]
        if "outcome_result" in decision:
            metadata["ab_event_outcome_judge"] = dict(decision["outcome_result"])
        results.append(
            {
                "score": score,
                "acc": bool(decision["acc"]),
                "pred": str(decision["pred"]),
                "judge_raw": str(decision["judge_raw"]),
                "event_rubric_state": str(decision["state"]),
            }
        )
    return results


def map_value_cliff_outcome4_reward(
    *,
    natural_finished: bool,
    outcome_correct: bool | None = None,
    avoid_error_met: bool | None = None,
    redirect_met: bool | None = None,
) -> tuple[float, str]:
    if not isinstance(natural_finished, bool):
        raise TypeError("natural_finished must be boolean")
    if natural_finished:
        if not isinstance(outcome_correct, bool):
            raise TypeError("a naturally finished replay requires boolean outcome_correct")
        if avoid_error_met is not None or redirect_met is not None:
            raise ValueError("a naturally finished replay must not include rubric verdicts")
        return (1.0, "outcome_correct") if outcome_correct else (0.0, "outcome_wrong")
    if outcome_correct is not None:
        raise ValueError("an unfinished replay must not include an outcome verdict")
    if not isinstance(avoid_error_met, bool) or not isinstance(redirect_met, bool):
        raise TypeError("an unfinished replay requires boolean avoid_error_met and redirect_met")
    if not avoid_error_met:
        return 0.0, "avoid_failed"
    if redirect_met:
        return 0.5, "redirected"
    return 0.25, "avoid_only"


def _is_natural_local_finish(sample: Any) -> bool:
    metadata = _metadata(sample)
    return (
        str(metadata.get("agent_last_finish_reason") or "").strip().lower() == "stop"
        and not metadata.get("agent_forced_final_answer")
        and not metadata.get("agent_context_reserve_hit")
        and not metadata.get("agent_local_horizon_hit")
        and not metadata.get("agent_max_turns_hit")
    )


async def _score_full_outcome_samples(samples: list[Any]) -> list[dict[str, Any]]:
    if not isinstance(samples, list) or not samples:
        raise ValueError("FullOutcome local scoring requires a non-empty sample list")
    from adaptive_branching.src.deep_research.reward_function import score_full_outcome_group

    results = await score_full_outcome_group(None, samples)
    if not isinstance(results, list):
        raise TypeError("FullOutcome group scorer must return a list")
    return results


def _validated_full_outcome_correct(result: Any) -> bool:
    if not isinstance(result, dict):
        raise TypeError("FullOutcome verdict must be an object")
    if result.get("judge_error"):
        raise ValueError("FullOutcome verdict contains judge_error")
    correct = result.get("acc")
    if not isinstance(correct, bool):
        raise TypeError("FullOutcome verdict acc must be boolean")
    return correct


def _group_reward_mode(samples: list[Any]) -> str:
    if not isinstance(samples, list) or not samples:
        raise ValueError("value-cliff local reward requires a non-empty sample group")
    modes = {str(_metadata(sample).get("ab_local_reward_mode") or "").strip().lower() for sample in samples}
    if len(modes) != 1:
        raise ValueError(f"value-cliff local group mixes reward modes: {sorted(modes)}")
    mode = next(iter(modes))
    if mode not in {V6_PRM_REWARD_MODE, V6P_REWARD_MODE, V7_PRM_REWARD_MODE, TERMINAL_REWARD_MODE}:
        raise ValueError(f"unsupported value-cliff local reward mode: {mode!r}")
    return mode


def _group_local_horizon(samples: list[Any]) -> int:
    horizons: set[int] = set()
    for sample in samples:
        raw = _metadata(sample).get("agent_max_turns")
        if isinstance(raw, bool) or not isinstance(raw, int) or raw <= 0:
            raise ValueError(f"value-cliff local sample has invalid agent_max_turns={raw!r}")
        horizons.add(raw)
    if len(horizons) != 1:
        raise ValueError(f"value-cliff local group mixes horizons: {sorted(horizons)}")
    return next(iter(horizons))


def _prepare_value_cliff_judge_input(
    sample: Any,
    *,
    horizon: int,
) -> tuple[str, int, bool, bool]:
    if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon <= 0:
        raise ValueError("horizon must be a positive integer")
    metadata = _metadata(sample)
    reward_mode = str(metadata.get("ab_local_reward_mode") or "").strip().lower()
    if reward_mode not in {V6_PRM_REWARD_MODE, V6P_REWARD_MODE, V7_PRM_REWARD_MODE}:
        raise ValueError(f"PRM reward requires v6_prm, v6p, or v7_prm mode, got {reward_mode!r}")
    rubric = validate_recovery_rubric(metadata.get("ab_event_recovery_rubric"), name="ab_event_recovery_rubric")
    messages = metadata.get("messages")
    if not isinstance(messages, list) or not messages or any(not isinstance(message, dict) for message in messages):
        raise ValueError("missing or invalid generated messages")
    prefix_count = metadata.get("ab_local_prefix_message_count")
    if isinstance(prefix_count, bool) or not isinstance(prefix_count, int) or not 0 < prefix_count < len(messages):
        raise ValueError(f"invalid local prefix message count: {prefix_count!r}")
    prefix = messages[:prefix_count]
    continuation = messages[prefix_count:]
    num_turns = sum(message.get("role") == "assistant" for message in continuation)
    if not 1 <= num_turns <= horizon:
        raise ValueError(f"local continuation must contain 1-{horizon} assistant turns, got {num_turns}")
    prompt, prefix_truncated, continuation_truncated = build_value_cliff_rubric_judge_prompt(
        prefix_messages=prefix,
        local_messages=continuation,
        recovery_rubric=rubric,
        trace_max_chars=positive_int(judge_settings("local_prm"), "trace_max_chars"),
    )
    return prompt, num_turns, prefix_truncated, continuation_truncated


def build_value_cliff_rubric_judge_prompt(
    *,
    prefix_messages: list[dict[str, Any]],
    local_messages: list[dict[str, Any]],
    recovery_rubric: dict[str, str],
    trace_max_chars: int,
) -> tuple[str, bool, bool]:
    if not isinstance(prefix_messages, list) or not prefix_messages:
        raise ValueError("prefix_messages must be a non-empty list")
    if any(not isinstance(message, dict) for message in prefix_messages):
        raise TypeError("prefix_messages must contain only objects")
    if not isinstance(local_messages, list) or not local_messages:
        raise ValueError("local_messages must be a non-empty list")
    if any(not isinstance(message, dict) for message in local_messages):
        raise TypeError("local_messages must contain only objects")
    if not any(message.get("role") == "assistant" for message in local_messages):
        raise ValueError("local_messages must contain at least one assistant turn")
    if isinstance(trace_max_chars, bool) or not isinstance(trace_max_chars, int) or trace_max_chars <= 0:
        raise ValueError("trace_max_chars must be a positive integer")
    rubric = validate_recovery_rubric(recovery_rubric)

    prefix_full = render_value_cliff_prefix_context(prefix_messages)
    continuation_full = render_message_turns(local_messages, heading="Local Turn")
    if not continuation_full:
        raise ValueError("local continuation rendered as empty")
    prefix_budget, continuation_budget = balanced_trace_budgets(
        len(prefix_full), len(continuation_full), trace_max_chars
    )
    prefix = render_value_cliff_prefix_context(prefix_messages, max_chars=max(80, prefix_budget))
    continuation = _truncate_middle(continuation_full, limit=max(80, continuation_budget))
    prefix_truncated = len(prefix_full) > prefix_budget
    continuation_truncated = len(continuation_full) > continuation_budget
    rubric_json = json.dumps(rubric, ensure_ascii=False, sort_keys=True)
    prompt = (
        f"<recovery_rubric>\n{rubric_json}\n</recovery_rubric>\n\n"
        f'<prefix_context truncated="{str(prefix_truncated).lower()}">\n'
        f"{prefix}\n</prefix_context>\n\n"
        f'<local_continuation truncated="{str(continuation_truncated).lower()}">\n'
        f"{continuation}\n</local_continuation>"
    )
    return prompt, prefix_truncated, continuation_truncated


def render_value_cliff_prefix_context(messages: list[dict[str, Any]], *, max_chars: int | None = None) -> str:
    if not isinstance(messages, list) or not messages or any(not isinstance(message, dict) for message in messages):
        raise ValueError("prefix messages must be a non-empty list of objects")
    if any(not isinstance(message.get("role"), str) or not message["role"].strip() for message in messages):
        raise ValueError("every prefix message must have a non-empty role")
    if max_chars is not None and (isinstance(max_chars, bool) or not isinstance(max_chars, int) or max_chars <= 0):
        raise ValueError("max_chars must be a positive integer or None")
    first_assistant = next(
        (index for index, message in enumerate(messages) if message.get("role") == "assistant"),
        len(messages),
    )
    initial_blocks = []
    for message in messages[:first_assistant]:
        role = message["role"].strip()
        content = str(message.get("content") or "").strip()
        initial_blocks.append(f"[{role}] {content}")
    rendered_turns = render_message_turns(messages[first_assistant:], heading="Prefix Turn")
    parts = [part for part in ("\n".join(initial_blocks), rendered_turns) if part]
    rendered = "\n\n".join(parts)
    if not rendered:
        raise ValueError("prefix context rendered as empty")
    return rendered if max_chars is None else _truncate_middle(rendered, limit=max_chars)


def _truncate_middle(text: str, *, limit: int) -> str:
    if not isinstance(text, str) or not text:
        raise ValueError("text must be a non-empty string")
    if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
        raise ValueError("limit must be a positive integer")
    if len(text) <= limit:
        return text
    marker = f"\n... [{len(text) - limit} chars omitted from prefix context] ...\n"
    if len(marker) >= limit:
        return text[:limit]
    remaining = limit - len(marker)
    return text[: (remaining + 1) // 2] + marker + text[-(remaining // 2) :]


def _structural_invalid_reason(sample: Any) -> str | None:
    metadata = _metadata(sample)
    finish_reason = str(metadata.get("agent_last_finish_reason") or "").strip().lower()
    status = getattr(getattr(sample, "status", ""), "value", getattr(sample, "status", ""))
    if bool(getattr(sample, "remove_sample", False)):
        return "sample_already_removed"
    if metadata.get("agent_function_failed"):
        return "agent_function_failed"
    if metadata.get("agent_forced_final_answer"):
        return "agent_forced_final_answer"
    if metadata.get("agent_context_reserve_hit"):
        return "agent_context_reserve_hit"
    if finish_reason in _INVALID_FINISH_REASONS:
        return f"agent_last_finish_reason={finish_reason}"
    if str(status).strip().lower() in {"aborted", "truncated"}:
        return f"sample_status={status}"
    return None


def is_single_call_length_truncation(sample: Any) -> bool:
    """Return whether only the model call output cap ended this trajectory."""

    metadata = _metadata(sample)
    finish_reason = str(metadata.get("agent_last_finish_reason") or "").strip().lower()
    if finish_reason != "length":
        return False
    status = getattr(getattr(sample, "status", ""), "value", getattr(sample, "status", ""))
    return not (
        bool(getattr(sample, "remove_sample", False))
        or bool(metadata.get("agent_excluded_from_training"))
        or bool(metadata.get("agent_function_failed"))
        or bool(metadata.get("agent_forced_final_answer"))
        or bool(metadata.get("agent_context_reserve_hit"))
        or str(status).strip().lower() in {"aborted", "truncated"}
    )


def _single_call_length_decision() -> dict[str, Any]:
    return {
        "score": 0.0,
        "acc": False,
        "pred": "",
        "judge_raw": "single_call_length",
        "state": "avoid_failed",
        "judge_metadata": {
            "reward_source": "single_call_length",
            "finish_reason": "length",
            "score": 0.0,
        },
    }


def _exclude_group(samples: list[Any], reason: str) -> list[dict[str, Any]]:
    reason = reason[:1000]
    results: list[dict[str, Any]] = []
    for sample in samples:
        metadata = _metadata(sample)
        metadata["ab_event_rubric_judge_error"] = reason
        metadata["ab_local_trainable"] = False
        metadata["agent_excluded_from_training"] = True
        sample.remove_sample = True
        results.append(
            {
                "score": 0.0,
                "acc": False,
                "pred": "",
                "judge_raw": reason,
                "judge_error": True,
                "event_rubric_state": "judge_error",
            }
        )
    return results


def _shared_judge_semaphore(limit: int) -> asyncio.Semaphore:
    loop = asyncio.get_running_loop()
    existing = _JUDGE_SEMAPHORES.get(loop)
    if existing is None or existing[0] != limit:
        existing = (limit, asyncio.Semaphore(limit))
        _JUDGE_SEMAPHORES[loop] = existing
    return existing[1]


def _metadata(sample: Any) -> dict[str, Any]:
    metadata = getattr(sample, "metadata", None)
    if not isinstance(metadata, dict):
        metadata = {}
        sample.metadata = metadata
    return metadata
