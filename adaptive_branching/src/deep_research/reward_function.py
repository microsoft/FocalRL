from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from typing import Any

from adaptive_branching.src.deep_research.value_cliff_online import annotate_value_cliff_branches, group_is_local
from adaptive_branching.src.deep_research.value_cliff_reward import (
    is_single_call_length_truncation,
    score_value_cliff_group,
)
from adaptive_branching.src.deep_research.judge_config import (
    judge_settings,
    positive_float,
    positive_int,
    required_str,
)

# A single model call that reaches its output-token cap is a trainable behavior
# failure: keep that sibling in GRPO with reward 0 and skip its judge request.
# Context-reserve/window exhaustion is an infrastructure limit and remains
# excluded from training, as do aborts and other structurally invalid samples.

logger = logging.getLogger(__name__)

_DEFAULT_POSITIVE_REPEAT_SEARCH_MIN_RATE = 0.2
_DEFAULT_POSITIVE_REPEAT_FETCH_URL_MIN_RATE = 0.25
_FORCED_FINAL_EXCLUDE_REASONS = {"context_reserve", "max_turns"}
_FORCE_EXCLUDE_MODES = {"none", "wrong", "all"}
_EXCLUDE_FINISH_REASONS = {
    "abort",
    "length",
    "forced_answer_fail",
    "forced_answer_empty",
    "forced_answer_skipped_no_budget",
}
_OUTCOME_JUDGE_SEMAPHORES: dict[asyncio.AbstractEventLoop, tuple[int, asyncio.Semaphore]] = {}

LLM_JUDGE_PROMPT_TEMPLATE = (
    "Judge whether the following [response] to [question] is correct or not based on the precise and "
    "unambiguous [correct_answer] below.\n\n"
    "[question]: {question}\n\n"
    "[response]: {response}\n\n"
    "Your judgement must be in the format and criteria specified below:\n\n"
    "extracted_final_answer: The final exact answer extracted from the [response]. Put the extracted answer "
    "as 'None' if there is no exact, final answer to extract from the response.\n\n"
    "[correct_answer]: {correct_answer}\n\n"
    "reasoning: Explain why the extracted_final_answer is correct or incorrect based on [correct_answer], "
    "focusing only on if there are meaningful differences between [correct_answer] and the "
    "extracted_final_answer. Do not comment on any background to the problem, do not attempt to solve the "
    "problem, do not argue for any answer different than [correct_answer], focus only on whether the answers "
    "match.\n\n"
    "correct: Answer 'yes' if extracted_final_answer matches the [correct_answer] given above, or is within "
    "a small margin of error for numerical problems. Answer 'no' otherwise, i.e. if there if there is any "
    "inconsistency, ambiguity, non-equivalency, or if the extracted answer is incorrect.\n\n"
    "confidence: The extracted confidence score between 0% and 100% from [response]. Put 100 if there is no "
    "confidence score available."
)


async def reward_func(args, samples, **kwargs):
    """LLM-as-a-judge reward for Miles custom RM.

    The DeepResearch run uses --input-key question and --label-key answer, so
    Sample.prompt is the question, Sample.response is the generated trajectory,
    and Sample.label is the ground-truth answer.
    """
    is_batched = isinstance(samples, list)
    batch = samples if is_batched else [samples]
    if not batch:
        return []

    if is_batched and group_is_local(batch):
        return await score_value_cliff_group(batch)

    return await score_full_outcome_group(args, samples)


async def score_full_outcome_group(args, samples):
    """Run the canonical FullOutcome judge without Value-Cliff local dispatch."""
    from openai import AsyncOpenAI

    is_batched = isinstance(samples, list)
    batch = samples if is_batched else [samples]
    if not batch:
        return []

    outcome_config = judge_settings("full_outcome")
    base_url = required_str(outcome_config, "base_url")
    api_key = required_str(outcome_config, "api_key")
    model = required_str(outcome_config, "model")

    # Reasoning effort is explicit in judge.yaml. Leave it empty there for
    # non-reasoning models that reject the parameter.
    reasoning_effort = str(outcome_config.get("reasoning_effort") or "").strip()
    if reasoning_effort:
        if reasoning_effort not in ("low", "medium", "high", "max"):
            raise ValueError(f"outcome reasoning_effort must be one of low/medium/high/max, got {reasoning_effort!r}")
    use_responses_api = model.lower().startswith("gpt-")

    timeout = positive_float(outcome_config, "timeout")
    client = AsyncOpenAI(api_key=api_key, base_url=base_url, timeout=timeout)
    judge_max_tokens = positive_int(outcome_config, "max_tokens")
    judge_max_retries = positive_int(outcome_config, "max_retries")
    judge_concurrency = positive_int(outcome_config, "max_concurrency")
    judge_semaphore = _shared_outcome_judge_semaphore(judge_concurrency)
    exclude_positive_repeated_search = _truthy(os.environ.get("AGENT_EXCLUDE_POSITIVE_REPEATED_SEARCH"))
    exclude_positive_repeated_fetch_url = _truthy(os.environ.get("AGENT_EXCLUDE_POSITIVE_REPEATED_FETCH_URL"))
    force_exclude_mode = _normalize_force_exclude_mode(os.environ.get("FORCE_EXCLUDE", "Wrong"))
    soft_overlong_start_tokens = _optional_int(os.environ.get("AGENT_SOFT_OVERLONG_START_TOKENS"))
    soft_overlong_hard_tokens = _optional_int(os.environ.get("AGENT_SOFT_OVERLONG_HARD_TOKENS"))
    soft_overlong_scale_raw = os.environ.get("AGENT_SOFT_OVERLONG_PENALTY_SCALE")
    soft_overlong_scale = _optional_float(soft_overlong_scale_raw)
    if soft_overlong_scale is None:
        if soft_overlong_scale_raw not in (None, ""):
            raise ValueError(
                "AGENT_SOFT_OVERLONG_PENALTY_SCALE must be a non-negative float, " f"got {soft_overlong_scale_raw!r}"
            )
        soft_overlong_scale = 0.0
    soft_overlong_positive_only = _truthy(os.environ.get("AGENT_SOFT_OVERLONG_POSITIVE_ONLY", "1"))
    soft_overlong_enabled = _truthy(os.environ.get("AGENT_SOFT_OVERLONG_ENABLE"))
    if soft_overlong_enabled:
        if soft_overlong_start_tokens is None or soft_overlong_hard_tokens is None:
            raise ValueError(
                "AGENT_SOFT_OVERLONG_START_TOKENS and AGENT_SOFT_OVERLONG_HARD_TOKENS must be set together"
            )
        if soft_overlong_hard_tokens <= soft_overlong_start_tokens:
            raise ValueError(
                "AGENT_SOFT_OVERLONG_HARD_TOKENS must be greater than " "AGENT_SOFT_OVERLONG_START_TOKENS"
            )
        if soft_overlong_scale < 0:
            raise ValueError("AGENT_SOFT_OVERLONG_PENALTY_SCALE must be non-negative, " f"got {soft_overlong_scale!r}")
    repeat_min_rate_raw = os.environ.get("AGENT_REPEAT_SEARCH_MIN_RATE")
    positive_repeat_min_rate = _optional_float(repeat_min_rate_raw)
    if positive_repeat_min_rate is None:
        if exclude_positive_repeated_search and repeat_min_rate_raw not in (None, ""):
            raise ValueError(f"AGENT_REPEAT_SEARCH_MIN_RATE must be a float, got {repeat_min_rate_raw!r}")
        positive_repeat_min_rate = _DEFAULT_POSITIVE_REPEAT_SEARCH_MIN_RATE
    if exclude_positive_repeated_search and not 0 <= positive_repeat_min_rate <= 1:
        raise ValueError("AGENT_REPEAT_SEARCH_MIN_RATE must be between 0 and 1, " f"got {positive_repeat_min_rate!r}")
    repeat_fetch_url_min_rate_raw = os.environ.get("AGENT_REPEAT_FETCH_URL_MIN_RATE")
    positive_repeat_fetch_url_min_rate = _optional_float(repeat_fetch_url_min_rate_raw)
    if positive_repeat_fetch_url_min_rate is None:
        if exclude_positive_repeated_fetch_url and repeat_fetch_url_min_rate_raw not in (None, ""):
            raise ValueError(f"AGENT_REPEAT_FETCH_URL_MIN_RATE must be a float, got {repeat_fetch_url_min_rate_raw!r}")
        positive_repeat_fetch_url_min_rate = _DEFAULT_POSITIVE_REPEAT_FETCH_URL_MIN_RATE
    if exclude_positive_repeated_fetch_url and not 0 <= positive_repeat_fetch_url_min_rate <= 1:
        raise ValueError(
            "AGENT_REPEAT_FETCH_URL_MIN_RATE must be between 0 and 1, " f"got {positive_repeat_fetch_url_min_rate!r}"
        )

    def predicted_response(sample) -> str:
        metadata = getattr(sample, "metadata", None) or {}
        text = metadata.get("agent_final_answer") or metadata.get("agent_final_content") or sample.response or ""
        return _strip_thinking(str(text))

    def apply_soft_overlong_penalty(sample, result: dict[str, Any], metadata: dict[str, Any]) -> dict[str, Any]:
        if not soft_overlong_enabled:
            return result

        length_tokens = _sample_length_tokens(sample, metadata)
        base_score = float(result.get("score") or 0.0)
        metadata["agent_soft_overlong_tokens"] = length_tokens or 0.0
        metadata["agent_soft_overlong_base_score"] = base_score
        metadata["agent_soft_overlong_penalty"] = 0.0
        metadata["agent_soft_overlong_fraction"] = 0.0
        metadata["agent_soft_overlong_applied"] = False
        metadata["agent_soft_overlong_score"] = base_score

        if length_tokens is None or (soft_overlong_positive_only and not bool(result.get("acc"))):
            return result

        span = soft_overlong_hard_tokens - soft_overlong_start_tokens
        fraction = (length_tokens - soft_overlong_start_tokens) / span
        fraction = min(1.0, max(0.0, fraction))
        penalty = -soft_overlong_scale * fraction
        shaped_score = max(0.0, base_score + penalty)

        metadata["agent_soft_overlong_penalty"] = penalty
        metadata["agent_soft_overlong_fraction"] = fraction
        metadata["agent_soft_overlong_applied"] = penalty < 0.0
        metadata["agent_soft_overlong_score"] = shaped_score
        return {**result, "score": shaped_score}

    def mark_training_exclusion(sample, result: dict[str, Any]) -> dict[str, Any]:
        metadata = getattr(sample, "metadata", None)
        if not isinstance(metadata, dict):
            metadata = {}
            sample.metadata = metadata

        result = apply_soft_overlong_penalty(sample, result, metadata)

        forced_reason = str(metadata.get("agent_forced_final_answer_reason") or "")
        finish_reason = str(metadata.get("agent_last_finish_reason") or "")
        sample_status = getattr(getattr(sample, "status", ""), "value", getattr(sample, "status", ""))
        search_query_count = _optional_float(metadata.get("agent_search_query_count")) or 0.0
        search_query_repeat_count = _optional_float(metadata.get("agent_search_query_repeat_count")) or 0.0
        search_query_repeat_rate = search_query_repeat_count / search_query_count if search_query_count > 0 else 0.0
        fetch_url_count = _optional_float(metadata.get("agent_fetch_url_count")) or 0.0
        fetch_url_repeat_count = _optional_float(metadata.get("agent_fetch_url_repeat_count")) or 0.0
        fetch_url_repeat_rate = fetch_url_repeat_count / fetch_url_count if fetch_url_count > 0 else 0.0
        metadata["agent_search_query_repeat_rate"] = search_query_repeat_rate
        metadata["agent_fetch_url_repeat_rate"] = fetch_url_repeat_rate
        positive_repeated_search = (
            exclude_positive_repeated_search
            and bool(result.get("acc"))
            and search_query_count > 0
            and search_query_repeat_rate >= positive_repeat_min_rate
        )
        positive_repeated_fetch_url = (
            exclude_positive_repeated_fetch_url
            and bool(result.get("acc"))
            and fetch_url_count > 0
            and fetch_url_repeat_rate >= positive_repeat_fetch_url_min_rate
        )
        metadata["agent_positive_repeated_search_excluded"] = positive_repeated_search
        metadata["agent_positive_repeated_fetch_url_excluded"] = positive_repeated_fetch_url
        forced_final = bool(metadata.get("agent_forced_final_answer"))
        forced_wrong = forced_final and forced_reason in _FORCED_FINAL_EXCLUDE_REASONS and not bool(result.get("acc"))
        forced_excluded = force_exclude_mode == "all" and forced_final
        if force_exclude_mode == "wrong":
            forced_excluded = forced_wrong
        abnormal = (
            bool(metadata.get("agent_excluded_from_training"))
            or bool(metadata.get("outcome_judge_failed"))
            or bool(metadata.get("agent_function_failed"))
            or bool(metadata.get("agent_forced_final_answer_failed"))
            or bool(metadata.get("agent_forced_final_answer_skipped_no_budget"))
            or (finish_reason in _EXCLUDE_FINISH_REASONS and not is_single_call_length_truncation(sample))
            or str(sample_status) in {"aborted", "truncated"}
        )

        exclude = forced_excluded or abnormal or positive_repeated_search or positive_repeated_fetch_url
        metadata["agent_excluded_from_training"] = exclude
        if exclude:
            sample.remove_sample = True
        return result

    def parse_verdict(text: str) -> bool | None:
        stripped = text.strip()
        json_candidates = [stripped]
        fenced = re.search(r"```(?:json)?\s*(.*?)```", stripped, re.IGNORECASE | re.DOTALL)
        if fenced:
            json_candidates.insert(0, fenced.group(1).strip())
        for candidate in json_candidates:
            start = candidate.find("{")
            if start < 0:
                continue
            try:
                parsed, _ = json.JSONDecoder().raw_decode(candidate[start:])
            except json.JSONDecodeError:
                continue
            if not isinstance(parsed, dict) or "correct" not in parsed:
                continue
            value = parsed["correct"]
            if isinstance(value, bool):
                return value
            normalized = str(value).strip().lower()
            if normalized in {"yes", "true"}:
                return True
            if normalized in {"no", "false"}:
                return False

        for line in text.splitlines():
            normalized = line.strip().lower()
            if normalized.startswith("correct:"):
                value = normalized.split(":", 1)[1].strip()
                if value.startswith("yes"):
                    return True
                if value.startswith("no"):
                    return False
        lowered = text.lower()
        if "correct: yes" in lowered:
            return True
        if "correct: no" in lowered:
            return False
        return None

    def mark_outcome_judge_failure(sample, raw_text: str, error: str) -> dict[str, Any]:
        metadata = getattr(sample, "metadata", None)
        if not isinstance(metadata, dict):
            metadata = {}
            sample.metadata = metadata
        detail = error or "unparseable_verdict"
        metadata["outcome_judge_failed"] = True
        metadata["outcome_judge_error"] = detail[:1000]
        return mark_training_exclusion(
            sample,
            {
                "score": 0.0,
                "acc": False,
                "pred": predicted_response(sample),
                "judge_raw": raw_text or detail,
                "judge_error": True,
            },
        )

    async def score_one(sample) -> dict[str, Any]:
        metadata = getattr(sample, "metadata", None)
        if isinstance(metadata, dict):
            # A recycled sample must not inherit a failure from an older judge
            # attempt if the current attempt succeeds.
            metadata.pop("outcome_judge_failed", None)
            metadata.pop("outcome_judge_error", None)
            metadata["agent_single_call_length_penalized"] = False
        response_text = predicted_response(sample)
        if is_single_call_length_truncation(sample):
            metadata = getattr(sample, "metadata", None)
            if not isinstance(metadata, dict):
                raise TypeError("single-call length truncation requires sample.metadata to be a dict")
            metadata["agent_single_call_length_penalized"] = True
            return mark_training_exclusion(
                sample,
                {
                    "score": 0.0,
                    "acc": False,
                    "pred": response_text,
                    "judge_raw": "single_call_length",
                },
            )
        if not response_text:
            return mark_training_exclusion(
                sample,
                {"score": 0.0, "acc": False, "pred": "", "judge_raw": "missing_response"},
            )

        ground_truth = sample.label
        sample_metadata = getattr(sample, "metadata", None) or {}
        question = str(sample_metadata.get("ab_original_question") or sample.prompt)
        prompt = LLM_JUDGE_PROMPT_TEMPLATE.format(
            question=question,
            correct_answer=str(ground_truth or ""),
            response=response_text,
        )

        last_text = ""
        last_error = ""
        async with judge_semaphore:
            for attempt in range(judge_max_retries):
                try:
                    if use_responses_api:
                        response_kwargs: dict[str, Any] = {}
                        if reasoning_effort:
                            response_kwargs["reasoning"] = {"effort": reasoning_effort}
                        response = await client.responses.create(
                            model=model,
                            input=prompt,
                            max_output_tokens=judge_max_tokens,
                            **response_kwargs,
                        )
                        status = getattr(response, "status", "") or ""
                        details = getattr(response, "incomplete_details", None)
                        incomplete_reason = getattr(details, "reason", "") if details is not None else ""
                        if status == "incomplete":
                            last_text = ""
                            last_error = f"incomplete_response: {incomplete_reason or status}"
                        else:
                            last_text = getattr(response, "output_text", "") or ""
                            last_error = "empty_output_text (model may have exhausted max_output_tokens on reasoning)"
                    else:
                        chat_kwargs: dict[str, Any] = {}
                        if reasoning_effort:
                            chat_kwargs["reasoning_effort"] = reasoning_effort
                        response = await client.chat.completions.create(
                            model=model,
                            messages=[{"role": "user", "content": prompt}],
                            max_tokens=judge_max_tokens,
                            **chat_kwargs,
                        )
                        if response.choices:
                            last_text = response.choices[0].message.content or ""
                            last_error = "empty_content"
                        else:
                            last_text = ""
                            last_error = "empty_choices (model may have exhausted max_tokens on reasoning)"
                    if not last_text:
                        last_text = ""
                        logger.warning(
                            "LLM judge returned empty output (attempt %d/%d): model=%s effort=%r error=%s",
                            attempt + 1,
                            judge_max_retries,
                            model,
                            reasoning_effort or None,
                            last_error,
                        )
                        if attempt + 1 < judge_max_retries:
                            await asyncio.sleep(0.5 * (2**attempt))
                        continue
                    verdict = parse_verdict(last_text)
                    if verdict is False:
                        return mark_training_exclusion(
                            sample,
                            {"score": 0.0, "acc": False, "pred": response_text, "judge_raw": last_text},
                        )
                    if verdict is True:
                        return mark_training_exclusion(
                            sample,
                            {"score": 1.0, "acc": True, "pred": response_text, "judge_raw": last_text},
                        )
                    last_error = "unparseable_verdict"
                except Exception as exc:
                    last_error = f"{type(exc).__name__}: {exc}"
                    logger.warning("LLM judge call failed (attempt %d/%d): %s", attempt + 1, judge_max_retries, exc)

                if attempt + 1 < judge_max_retries:
                    await asyncio.sleep(0.5 * (2**attempt))

        logger.warning("LLM judge returned no parseable verdict: raw=%r error=%s", last_text[:200], last_error)
        return mark_outcome_judge_failure(sample, last_text, last_error)

    results = await asyncio.gather(*(score_one(sample) for sample in batch))
    if is_batched:
        if not group_is_local(batch):
            await annotate_value_cliff_branches(args, batch, results)
    return results if is_batched else results[0]


def _shared_outcome_judge_semaphore(limit: int) -> asyncio.Semaphore:
    """Cap outcome-judge calls across concurrent groups in this worker loop."""
    loop = asyncio.get_running_loop()
    existing = _OUTCOME_JUDGE_SEMAPHORES.get(loop)
    if existing is None or existing[0] != limit:
        existing = (limit, asyncio.Semaphore(limit))
        _OUTCOME_JUDGE_SEMAPHORES[loop] = existing
    return existing[1]


def _strip_thinking(text: str) -> str:
    text = text or ""
    if "</think>" in text:
        return text.rsplit("</think>", 1)[-1].strip()
    return text.strip()


def _optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _optional_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _sample_length_tokens(sample, metadata: dict[str, Any]) -> float | None:
    for value in (
        metadata.get("agent_session_tokens"),
        getattr(sample, "response_length", None),
    ):
        parsed = _optional_float(value)
        if parsed is not None:
            return parsed
    tokens = getattr(sample, "tokens", None)
    if tokens is not None:
        try:
            return float(len(tokens))
        except TypeError:
            return None
    return None


def _normalize_force_exclude_mode(value: Any) -> str:
    mode = str(value or "Wrong").strip().lower()
    if mode not in _FORCE_EXCLUDE_MODES:
        allowed = ", ".join(sorted(_FORCE_EXCLUDE_MODES))
        raise ValueError(f"FORCE_EXCLUDE must be one of {allowed}, got {value!r}")
    return mode


def _truthy(value: Any) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y", "on"}
