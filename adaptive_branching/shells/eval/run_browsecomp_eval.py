#!/usr/bin/env python3
"""Evaluate Deep Research checkpoints on BrowseComp or GAIA-text.

Run the shared search/browse agent, save its interaction trace, and judge its
final answer against the reference answer. Supports JSONL and Parquet inputs.
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import hashlib
import importlib
import json
import math
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from urllib.parse import urlsplit

import httpx

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from adaptive_branching.src.deep_research import reward_function  # noqa: E402
from adaptive_branching.src.deep_research.eval_continuation import (  # noqa: E402
    load_continuation_samples,
)
from adaptive_branching.src.deep_research.judge_config import (  # noqa: E402
    judge_settings,
    positive_int,
    required_str,
)
from adaptive_branching.src.deep_research.value_cliff_prefixes import (  # noqa: E402
    PREFIX_INPUT_MODE,
    PREFIX_PROVENANCE_FIELDS,
    assert_returned_messages_preserve_prefix,
    prefix_provenance,
    validate_prefix_row,
)

INPUT_MODES = ("question", PREFIX_INPUT_MODE)


def _load_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".parquet":
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("reading parquet eval data requires pyarrow") from exc
        rows = pq.read_table(path).to_pylist()
    else:
        rows = []
        with path.open(encoding="utf-8") as f:
            for row_idx, line in enumerate(f):
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid JSON on row {row_idx} in {path}: {exc}") from exc
                rows.append(row)
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"all rows in {path} must be objects")
    return rows


def _load_samples(
    path: Path,
    *,
    limit: int | None,
    offset: int,
    input_mode: str = "question",
    max_turns: int | None = None,
) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"BrowseComp data not found: {path}")
    if offset < 0:
        raise ValueError("--offset must be >= 0")
    if limit is not None and limit <= 0:
        raise ValueError("--limit must be positive")
    if input_mode not in INPUT_MODES:
        raise ValueError(f"input_mode={input_mode!r}; expected one of {INPUT_MODES}")
    if input_mode == PREFIX_INPUT_MODE and (
        isinstance(max_turns, bool) or not isinstance(max_turns, int) or max_turns <= 0
    ):
        raise ValueError("prefix_messages input requires a positive max_turns contract")

    samples: list[dict[str, Any]] = []
    for row_idx, row in enumerate(_load_rows(path)):
        if row_idx < offset:
            continue
        if limit is not None and len(samples) >= limit:
            break
        if input_mode == PREFIX_INPUT_MODE:
            samples.append(
                validate_prefix_row(
                    row,
                    name=f"{path}:{row_idx + 1}",
                    expected_source_max_turns=max_turns,
                )
            )
        else:
            question = next((row[key] for key in ("problem", "task_question", "question", "Question")
                             if row.get(key) is not None and row[key] != ""), None)
            answer = next((row[key] for key in ("answer", "ground_truth", "Final answer")
                           if row.get(key) is not None and row[key] != ""), None)
            if not isinstance(question, str) or not question.strip():
                raise ValueError(f"{path}:{row_idx + 1}: missing nonempty question")
            if isinstance(answer, bool) or not isinstance(answer, (str, int, float)) or not str(answer).strip():
                raise ValueError(f"{path}:{row_idx + 1}: missing nonempty reference answer")
            if isinstance(answer, float) and not math.isfinite(answer):
                raise ValueError(f"{path}:{row_idx + 1}: reference answer must be finite")
            if row.get("file_name"):
                raise ValueError(f"{path}:{row_idx + 1}: GAIA-text input must exclude tasks with file attachments")
            task_id = row.get("task_id")
            if task_id is None:
                task_id = row.get("id")
            if task_id is None:
                task_id = f"row_{row_idx}"
            if isinstance(task_id, bool) or not isinstance(task_id, (str, int)) or not str(task_id).strip():
                raise ValueError(f"{path}:{row_idx + 1}: task_id must be a nonempty string or integer")
            samples.append(
                {
                    "task_id": str(task_id),
                    "question": str(question),
                    "ground_truth": str(answer),
                }
            )
    if not samples:
        raise ValueError(f"no samples loaded from {path}")
    task_ids = [sample["task_id"] for sample in samples]
    if len(set(task_ids)) != len(task_ids):
        raise ValueError(f"input data contains duplicate task_id values: {path}")
    if input_mode == PREFIX_INPUT_MODE:
        prefix_indices = [sample["prefix_index"] for sample in samples]
        if len(set(prefix_indices)) != len(prefix_indices):
            raise ValueError(f"prefix input contains duplicate prefix_index values: {path}")
    return samples


def _sample_prompt(sample: dict[str, Any], *, input_mode: str) -> str | list[dict[str, Any]]:
    if input_mode == "question":
        if "continuation" in sample:
            return copy.deepcopy(sample["continuation_messages"])
        return str(sample["question"])
    if input_mode == PREFIX_INPUT_MODE:
        return copy.deepcopy(sample["prefix_messages"])
    raise ValueError(f"input_mode={input_mode!r}; expected one of {INPUT_MODES}")


def _history_modes(mode: str) -> list[str]:
    if mode not in ("react", "keep5", "both"):
        raise ValueError(f"invalid history mode: {mode!r}")
    if mode == "both":
        return ["react", "keep5"]
    return [mode]


def _base_url(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("base URL must be nonempty")
    parsed = urlsplit(value)
    if (parsed.scheme not in {"http", "https"} or not parsed.hostname
            or parsed.username or parsed.password or parsed.query or parsed.fragment):
        raise ValueError("base URL must be HTTP(S), without credentials, query, or fragment")
    url = value.rstrip("/")
    return url.removesuffix("/v1")


def _request_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    if not isinstance(args.model, str) or not args.model.strip():
        raise ValueError("model must be nonempty")
    if type(args.max_tokens) is not int or args.max_tokens <= 0:
        raise ValueError("max_tokens must be a positive integer")
    if not math.isfinite(args.temperature) or args.temperature < 0:
        raise ValueError("temperature must be finite and nonnegative")
    if not math.isfinite(args.top_p) or not 0 < args.top_p <= 1:
        raise ValueError("top_p must be finite and in (0, 1]")
    if args.top_k is not None and (type(args.top_k) is not int or args.top_k < -1):
        raise ValueError("top_k must be an integer >= -1")
    if args.repetition_penalty is not None and (
        not math.isfinite(args.repetition_penalty) or args.repetition_penalty <= 0
    ):
        raise ValueError("repetition_penalty must be finite and positive")
    # Qwen3.5's forced-final request appends a new user message.  Preserve all
    # earlier assistant reasoning when the server renders that request.
    chat_template_kwargs: dict[str, Any] = {"clear_thinking": False}
    kwargs: dict[str, Any] = {
        "model": args.model,
        "temperature": args.temperature,
        "top_p": args.top_p,
    }
    kwargs["max_tokens"] = args.max_tokens
    if args.top_k is not None:
        kwargs["top_k"] = args.top_k
    if args.repetition_penalty is not None:
        kwargs["repetition_penalty"] = args.repetition_penalty
    if args.reasoning_effort:
        kwargs["reasoning_effort"] = args.reasoning_effort
    if args.thinking_type is not None:
        kwargs["thinking"] = {"type": args.thinking_type}
    if args.extra_request_json:
        extra = json.loads(args.extra_request_json)
        if not isinstance(extra, dict):
            raise ValueError("--extra-request-json must decode to a JSON object")
        extra_chat_template_kwargs = extra.pop("chat_template_kwargs", {})
        if not isinstance(extra_chat_template_kwargs, dict):
            raise ValueError("--extra-request-json chat_template_kwargs must be a JSON object")
        chat_template_kwargs.update(extra_chat_template_kwargs)
        kwargs.update(extra)
    thinking_budget = getattr(args, "thinking_budget", None)
    if thinking_budget is not None:
        if args.agent_api != "chat":
            raise ValueError("--thinking-budget currently requires --agent-api=chat")
        if isinstance(thinking_budget, bool) or not isinstance(thinking_budget, int) or thinking_budget < 0:
            raise ValueError("--thinking-budget must be a non-negative integer")
        conflicting_fields = {"custom_logit_processor", "custom_params"}.intersection(kwargs)
        if conflicting_fields:
            raise ValueError(
                "--thinking-budget cannot be combined with these --extra-request-json fields: "
                + ", ".join(sorted(conflicting_fields))
            )
        try:
            from adaptive_branching.src.deep_research.sglang_thinking_budget import (
                Glm5ThinkingBudgetLogitProcessor,
            )
        except ImportError as exc:
            raise RuntimeError("--thinking-budget requires the dill package used by SGLang") from exc
        kwargs["custom_logit_processor"] = Glm5ThinkingBudgetLogitProcessor.to_str()
        kwargs["custom_params"] = {"thinking_budget": thinking_budget}
    if args.agent_api == "chat":
        if chat_template_kwargs.get("clear_thinking") is not False:
            raise ValueError("BrowseComp eval requires chat_template_kwargs.clear_thinking=false")
        kwargs["chat_template_kwargs"] = chat_template_kwargs
    return kwargs


def _metadata(args: argparse.Namespace, history_mode: str) -> dict[str, Any]:
    system_prompt = "" if getattr(args, "no_system_prompt", False) else getattr(args, "system_prompt", None)
    return {
        "agent_max_turns": args.max_turns,
        "max_seq_len": args.max_seq_len,
        "agent_history_mode": history_mode,
        "agent_keep_tool_results": args.keep_tool_results,
        "agent_return_messages": True,
        "agent_logprobs": False,
        "agent_api_mode": args.agent_api,
        "agent_system_prompt": system_prompt,
        "agent_tool_call_text_retries": getattr(args, "tool_call_text_retries", 0),
        "agent_capture_raw_responses": bool(getattr(args, "save_raw_responses", False)),
    }


def _without_duplicate_responses_output(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not isinstance(messages, list):
        raise TypeError(f"messages must be a list, got {type(messages).__name__}")
    cleaned: list[dict[str, Any]] = []
    for message_index, message in enumerate(messages):
        if not isinstance(message, dict):
            raise TypeError(f"messages[{message_index}] must be an object, got {type(message).__name__}")
        cleaned.append({key: value for key, value in message.items() if key != "_responses_output"})
    return cleaned


async def _score(question: str, ground_truth: str, answer: str) -> dict[str, Any]:
    if not all(isinstance(value, str) for value in (question, ground_truth, answer)):
        raise TypeError("question, reference answer, and generated answer must be strings")
    if not question.strip() or not ground_truth.strip():
        raise ValueError("question and reference answer must be nonempty")
    sample = SimpleNamespace(
        prompt=question,
        response=answer,
        label=ground_truth,
        metadata={"agent_final_answer": answer},
    )
    return await reward_function.score_full_outcome_group(None, sample)


async def _check_agent_endpoint(
    base_url: str, model: str, timeout: float, request_kwargs: dict[str, Any], agent_api: str
) -> None:
    chat_payload = {
        **request_kwargs,
        "model": model,
        "messages": [{"role": "user", "content": "ping"}],
        "max_tokens": 16 if agent_api == "responses" else 1,
        "stream": False,
    }
    if agent_api == "responses":
        url = f"{base_url}/v1/responses"
        payload = {
            "model": model,
            "input": [{"role": "user", "content": "ping"}],
            "max_output_tokens": 16,
        }
        if request_kwargs.get("reasoning_effort"):
            payload["reasoning"] = {"effort": request_kwargs["reasoning_effort"]}
    else:
        url = f"{base_url}/v1/chat/completions"
        payload = chat_payload
    api_key = os.getenv("AGENT_CHAT_API_KEY")
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else None
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(url, json=payload, headers=headers)
    except (httpx.TimeoutException, httpx.TransportError) as exc:
        raise RuntimeError(f"{agent_api} endpoint preflight failed: {type(exc).__name__}: {exc}") from exc
    if response.status_code != 200:
        raise RuntimeError(f"{agent_api} endpoint preflight failed ({response.status_code}): {response.text[:500]}")


def _metric(record: dict[str, Any], key: str) -> float | None:
    value = (record.get("metrics") or {}).get(key)
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, int | float):
        return float(value)
    return None


def _mean_metric(records: list[dict[str, Any]], key: str) -> float | None:
    values = [_metric(record, key) for record in records]
    values = [value for value in values if value is not None]
    return sum(values) / len(values) if values else None


def _aggregate(
    records: list[dict[str, Any]],
    *,
    model: str,
    history_mode: str,
    elapsed: float,
    input_mode: str = "question",
    input_sample_count: int | None = None,
    expected_valid_records: int | None = None,
    require_judge: bool = True,
) -> dict[str, Any]:
    if input_mode not in INPUT_MODES:
        raise ValueError(f"input_mode={input_mode!r}; expected one of {INPUT_MODES}")
    if input_sample_count is not None and input_sample_count <= 0:
        raise ValueError("input_sample_count must be positive when provided")
    if expected_valid_records is not None and expected_valid_records <= 0:
        raise ValueError("expected_valid_records must be positive when provided")

    valid_records = [
        record for record in records if _resume_record_counts_as_attempt(record, require_judge=require_judge)
    ]
    n = len(valid_records)
    correct = sum(1 for record in valid_records if record.get("acc") is True)
    task_ids = {str(record.get("task_id") or "") for record in valid_records if record.get("task_id")}
    solved_task_ids = {
        str(record.get("task_id") or "")
        for record in valid_records
        if record.get("task_id") and record.get("acc") is True
    }
    judge_errors = sum(1 for record in records if record.get("judge_error"))
    gen_errors = sum(1 for record in records if record.get("gen_error"))
    answered = sum(1 for record in valid_records if (record.get("metrics") or {}).get("agent_final_answer_present"))
    return {
        "model": model,
        "history_mode": history_mode,
        "input_mode": input_mode,
        "input_records": input_sample_count,
        "input_prefixes": input_sample_count if input_mode == PREFIX_INPUT_MODE else None,
        "attempt_records": len(records),
        "expected_valid_records": expected_valid_records,
        "actual_valid_records": n,
        "valid_record_target_met": expected_valid_records is None or n == expected_valid_records,
        "n": n,
        "unique_tasks": len(task_ids),
        "solved_tasks": len(solved_task_ids),
        "sample_success_rate": len(solved_task_ids) / len(task_ids) if task_ids else 0.0,
        "correct": correct,
        "accuracy": correct / n if n else 0.0,
        "answered": answered,
        "answer_rate": answered / n if n else 0.0,
        "gen_errors": gen_errors,
        "judge_errors": judge_errors,
        "duration_seconds": round(elapsed, 2),
        "agent_turns_mean": _mean_metric(valid_records, "agent_turns"),
        "agent_session_tokens_mean": _mean_metric(valid_records, "agent_session_tokens"),
        "agent_forced_final_answer_rate": _mean_metric(valid_records, "agent_forced_final_answer"),
        "agent_context_reserve_hit_rate": _mean_metric(valid_records, "agent_context_reserve_hit"),
        "agent_final_answer_tool_call_like_rate": _mean_metric(valid_records, "agent_final_answer_tool_call_like"),
        "agent_forced_final_answer_tool_call_like_rate": _mean_metric(
            valid_records,
            "agent_forced_final_answer_tool_call_like",
        ),
        "agent_tool_unit_success_rate_mean": _mean_metric(valid_records, "agent_tool_unit_success_rate"),
        "agent_keep5_omitted_tool_results_mean": _mean_metric(
            valid_records, "agent_history_keep5_omitted_tool_results"
        ),
    }


def _resume_record_counts_as_attempt(record: dict[str, Any], *, require_judge: bool) -> bool:
    if record.get("gen_error") or record.get("judge_error"):
        return False
    metrics = record.get("metrics")
    if not isinstance(metrics, dict):
        return False
    # A length/abort termination can still contain partial text and can even be
    # judged, but it is not a completed agent trajectory.  Do not consume an
    # attempt slot or treat it as resumable evaluation data.
    if metrics.get("agent_finished") is False:
        return False
    if require_judge and "acc" not in record:
        return False
    return True


def _record_attempt_number(record: dict[str, Any], fallback_attempt: int) -> int:
    attempt = record.get("attempt")
    if isinstance(attempt, int) and attempt > 0:
        return attempt
    if isinstance(attempt, str) and attempt.isdigit() and int(attempt) > 0:
        return int(attempt)
    return fallback_attempt


def _mark_valid_attempt(state: dict[str, Any], attempt: int) -> None:
    valid_attempts = state.setdefault("valid_attempts", set())
    valid_attempts.add(attempt)
    contiguous_attempts = 0
    while contiguous_attempts + 1 in valid_attempts:
        contiguous_attempts += 1
    state["attempts"] = contiguous_attempts


def _is_task_finished(state: dict[str, Any] | None, max_attempts: int, *, stop_on_success: bool) -> bool:
    if not state:
        return False
    return (stop_on_success and bool(state.get("success"))) or int(state.get("attempts") or 0) >= max_attempts


def _count_finished_tasks(
    samples: list[dict[str, Any]],
    state_by_id: dict[str, dict[str, Any]],
    max_attempts: int,
    *,
    stop_on_success: bool,
) -> int:
    return sum(
        1
        for sample in samples
        if _is_task_finished(state_by_id.get(sample["task_id"]), max_attempts, stop_on_success=stop_on_success)
    )


def _load_resume_records(
    out_path: Path,
    samples: list[dict[str, Any]],
    *,
    model: str,
    history_mode: str,
    require_judge: bool,
    max_attempts: int,
    samples_per_task: int,
    stop_on_success: bool,
    input_mode: str = "question",
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    if input_mode not in INPUT_MODES:
        raise ValueError(f"input_mode={input_mode!r}; expected one of {INPUT_MODES}")
    sample_by_id = {sample["task_id"]: sample for sample in samples}
    if len(sample_by_id) != len(samples):
        raise ValueError("resume requires unique task_id values in the input data")

    records: list[dict[str, Any]] = []
    state_by_id: dict[str, dict[str, Any]] = {}
    with out_path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"cannot resume from malformed JSONL line {line_no} in {out_path}: {exc}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"cannot resume from non-object JSONL line {line_no} in {out_path}")
            task_id = str(record.get("task_id") or "")
            sample = sample_by_id.get(task_id)
            if sample is None:
                continue
            mismatches = _resume_record_mismatches(
                record,
                sample,
                model=model,
                history_mode=history_mode,
                max_attempts=max_attempts,
                samples_per_task=samples_per_task,
                input_mode=input_mode,
            )
            if mismatches:
                raise ValueError(
                    f"cannot resume incompatible record for task {task_id!r} on line {line_no} in {out_path}: "
                    + "; ".join(mismatches)
                )
            records.append(record)
            state = state_by_id.setdefault(
                task_id,
                {"attempts": 0, "valid_attempts": set(), "success": False, "complete": False},
            )
            if _resume_record_counts_as_attempt(record, require_judge=require_judge):
                fallback_attempt = int(state.get("attempts") or 0) + 1
                valid_attempt = _record_attempt_number(record, fallback_attempt)
                if valid_attempt > max_attempts:
                    raise ValueError(
                        f"cannot resume from attempt {valid_attempt} beyond max_attempts={max_attempts} "
                        f"for task {task_id!r} on line {line_no} in {out_path}"
                    )
                if valid_attempt in state["valid_attempts"]:
                    raise ValueError(
                        f"cannot resume from duplicate valid attempt {valid_attempt} "
                        f"for task {task_id!r} on line {line_no} in {out_path}"
                    )
                _mark_valid_attempt(state, valid_attempt)
                if record.get("acc") is True:
                    state["success"] = True
            if _is_complete_resume_record(
                record,
                sample,
                model=model,
                history_mode=history_mode,
                require_judge=require_judge,
                max_attempts=max_attempts,
                samples_per_task=samples_per_task,
                input_mode=input_mode,
            ):
                state["complete"] = True

    for state in state_by_id.values():
        state["complete"] = _is_task_finished(state, max_attempts, stop_on_success=stop_on_success)

    return records, state_by_id


def _resume_record_mismatches(
    record: dict[str, Any],
    sample: dict[str, Any],
    *,
    model: str,
    history_mode: str,
    max_attempts: int,
    samples_per_task: int,
    input_mode: str,
) -> list[str]:
    if input_mode not in INPUT_MODES:
        raise ValueError(f"input_mode={input_mode!r}; expected one of {INPUT_MODES}")
    mismatches: list[str] = []
    if record.get("model") != model or record.get("history_mode") != history_mode:
        mismatches.append(
            f"model/history_mode={(record.get('model'), record.get('history_mode'))!r}, "
            f"expected {(model, history_mode)!r}"
        )
    if str(record.get("question") or "") != sample["question"]:
        mismatches.append("question differs from the current input")
    if str(record.get("ground_truth") or "") != sample["ground_truth"]:
        mismatches.append("ground_truth differs from the current input")
    if record.get("max_attempts", 1) != max_attempts:
        mismatches.append(f"max_attempts={record.get('max_attempts', 1)!r}, expected {max_attempts}")
    if record.get("samples_per_task", 1) != samples_per_task:
        mismatches.append(f"samples_per_task={record.get('samples_per_task', 1)!r}, expected {samples_per_task}")

    # Old question-mode records predate this explicit field and remain resumable.
    record_input_mode = record.get("input_mode", "question")
    if record_input_mode != input_mode:
        mismatches.append(f"input_mode={record_input_mode!r}, expected {input_mode!r}")
    if record.get("continuation") != sample.get("continuation"):
        mismatches.append("continuation source or total turn budget differs")
    if input_mode == PREFIX_INPUT_MODE:
        for field in PREFIX_PROVENANCE_FIELDS:
            if field not in record:
                mismatches.append(f"missing prefix provenance field {field!r}")
            elif record[field] != sample[field]:
                mismatches.append(f"{field}={record[field]!r}, expected {sample[field]!r}")
        if not record.get("gen_error"):
            try:
                assert_returned_messages_preserve_prefix(
                    record.get("messages"),
                    sample["prefix_messages"],
                    name="resumed record",
                )
            except (TypeError, ValueError) as exc:
                mismatches.append(str(exc))
    return mismatches


def _is_resume_record_for_sample(
    record: dict[str, Any],
    sample: dict[str, Any],
    *,
    model: str,
    history_mode: str,
    max_attempts: int,
    samples_per_task: int,
    input_mode: str = "question",
) -> bool:
    return not _resume_record_mismatches(
        record,
        sample,
        model=model,
        history_mode=history_mode,
        max_attempts=max_attempts,
        samples_per_task=samples_per_task,
        input_mode=input_mode,
    )


def _is_complete_resume_record(
    record: dict[str, Any],
    sample: dict[str, Any],
    *,
    model: str,
    history_mode: str,
    require_judge: bool,
    max_attempts: int,
    samples_per_task: int,
    input_mode: str = "question",
) -> bool:
    if not _is_resume_record_for_sample(
        record,
        sample,
        model=model,
        history_mode=history_mode,
        max_attempts=max_attempts,
        samples_per_task=samples_per_task,
        input_mode=input_mode,
    ):
        return False
    if record.get("gen_error") or record.get("judge_error"):
        return False
    metrics = record.get("metrics")
    if not isinstance(metrics, dict):
        return False
    if metrics.get("agent_finished") is False:
        return False
    if require_judge and ("score" not in record or "acc" not in record):
        return False
    return True


async def _run_mode(args: argparse.Namespace, samples: list[dict[str, Any]], history_mode: str) -> dict[str, Any]:
    if not samples or len({sample["task_id"] for sample in samples}) != len(samples):
        raise ValueError("evaluation requires nonempty samples with unique task IDs")
    agent_runner = importlib.import_module(args.agent_module)
    if not callable(getattr(agent_runner, "run", None)):
        raise ValueError(f"agent module has no callable run(): {args.agent_module}")
    request_kwargs = _request_kwargs(args)
    metadata = _metadata(args, history_mode)
    run_config = {
        "agent_module": args.agent_module,
        "request_kwargs": request_kwargs,
        "metadata": metadata,
        "context_reserve_tokens": args.context_reserve_tokens,
        "search_provider": os.getenv("AGENT_SEARCH_PROVIDER", "microsoft"),
        "model_endpoint": _base_url(args.base_url),
        "service_settings": {name: os.getenv(name) for name in (
            "SERPER_ENDPOINT", "JINA_BASE_URL", "BROWSER_LLM_URL", "BROWSER_LLM_MODEL", "LLM_JUDGE_URL"
        )},
        "config_hashes": {name: hashlib.sha256(Path(os.environ[name]).read_bytes()).hexdigest()
                          for name in ("AGENT_TOOLS_CONFIG", "AGENT_JUDGE_CONFIG") if name in os.environ},
    }

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    # An OpenAI model identifier may be an absolute local checkpoint path.
    # Joining that path directly would silently discard out_dir.
    model_output_name = Path(args.model.rstrip("/")).name or "model"
    out_path = out_dir / f"{model_output_name}.{history_mode}.jsonl"
    fixed_sampling = args.samples_per_task > 1
    max_attempts = args.samples_per_task if fixed_sampling else 1 + args.retry_failed_attempts
    stop_on_success = not fixed_sampling
    if args.resume and out_path.exists():
        existing_records, state_by_id = _load_resume_records(
            out_path,
            samples,
            model=args.model,
            history_mode=history_mode,
            require_judge=not args.skip_judge,
            max_attempts=max_attempts,
            samples_per_task=args.samples_per_task,
            stop_on_success=stop_on_success,
            input_mode=args.input_mode,
        )
        for record in existing_records:
            if record.get("evaluation_config") != run_config:
                raise ValueError(f"cannot resume task={record['task_id']}: evaluation configuration differs")
        completed = _count_finished_tasks(samples, state_by_id, max_attempts, stop_on_success=stop_on_success)
        print(
            f"[{history_mode}] resume: loaded {len(existing_records)} attempt records from {out_path}; "
            f"completed={completed}/{len(samples)} remaining={len(samples) - completed} "
            f"max_attempts={max_attempts}",
            flush=True,
        )
    else:
        if out_path.exists() and out_path.stat().st_size:
            raise FileExistsError(f"output already contains records; use --resume or a new output directory: {out_path}")
        out_path.write_text("", encoding="utf-8")
        existing_records = []
        state_by_id = {}

    base_url = _base_url(args.base_url)
    gen_sema = asyncio.Semaphore(args.concurrency)
    judge_sema = asyncio.Semaphore(args.judge_concurrency)
    write_lock = asyncio.Lock()
    state_lock = asyncio.Lock()
    started = time.time()

    async def append_record(record: dict[str, Any]) -> None:
        line = json.dumps(record, ensure_ascii=False, default=str) + "\n"
        async with write_lock:
            with out_path.open("a", encoding="utf-8") as f:
                f.write(line)
                f.flush()

    async def one_attempt(sample: dict[str, Any], attempt: int) -> dict[str, Any]:
        state = state_by_id.setdefault(
            sample["task_id"],
            {"attempts": 0, "valid_attempts": set(), "success": False, "complete": False},
        )
        sample_started = time.time()
        record: dict[str, Any] = {
            "task_id": sample["task_id"],
            "question": sample["question"],
            "ground_truth": sample["ground_truth"],
            "model": args.model,
            "history_mode": history_mode,
            "attempt": attempt,
            "max_attempts": max_attempts,
            "retry_failed_attempts": args.retry_failed_attempts,
            "samples_per_task": args.samples_per_task,
            "sample_index": attempt - 1,
            "sampling_mode": "fixed" if fixed_sampling else "retry_until_success",
            "input_mode": args.input_mode,
            "evaluation_config": run_config,
        }
        if args.input_mode == PREFIX_INPUT_MODE:
            record.update(prefix_provenance(sample))
        prompt = _sample_prompt(sample, input_mode=args.input_mode)
        attempt_metadata = dict(metadata)
        if "continuation" in sample:
            record["continuation"] = sample["continuation"]
            record["source_metrics"] = sample["source_metrics"]
            attempt_metadata["agent_turn_offset"] = sample["continuation"]["prefix_turns"]
            attempt_metadata["agent_initial_session_tokens"] = sample["initial_session_tokens"]
        if args.input_mode == PREFIX_INPUT_MODE:
            attempt_metadata["agent_max_turns"] = sample["remaining_turns"]
        try:
            async with gen_sema:
                result = await agent_runner.run(
                    base_url=base_url,
                    prompt=prompt,
                    request_kwargs=request_kwargs,
                    metadata=attempt_metadata,
                )
            if not isinstance(result, dict):
                raise RuntimeError(f"agent.run returned {type(result).__name__}")
            if result.get("agent_finished") is False:
                raise RuntimeError(f"task={sample['task_id']} attempt={attempt}: agent trajectory is incomplete")
        except Exception as exc:
            record["gen_error"] = f"{type(exc).__name__}: {exc}"
            if not args.keep_going:
                await append_record(record)
                raise
        else:
            messages = result.pop("messages", [])
            raw_responses = result.pop("agent_raw_responses", None)
            if bool(getattr(args, "save_raw_responses", False)):
                if not isinstance(raw_responses, list) or not raw_responses:
                    raise RuntimeError(
                        "--save-raw-responses requires the Responses agent to return a non-empty raw response trace"
                    )
                record["raw_responses"] = raw_responses
                messages = _without_duplicate_responses_output(messages)
            if args.input_mode == PREFIX_INPUT_MODE:
                assert_returned_messages_preserve_prefix(
                    messages,
                    sample["prefix_messages"],
                    name=f"task={sample['task_id']} attempt={attempt}",
                )
            answer = str(result.get("agent_final_answer") or "")
            record.update({"messages": messages, "answer": answer, "metrics": result})

            if not args.skip_judge:
                try:
                    async with judge_sema:
                        verdict = await _score(sample["question"], sample["ground_truth"], answer)
                        if verdict.get("judge_error"):
                            raise RuntimeError(f"task={sample['task_id']} attempt={attempt}: answer judge failed")
                except Exception as exc:
                    record["judge_error"] = f"{type(exc).__name__}: {exc}"
                    if not args.keep_going:
                        await append_record(record)
                        raise
                else:
                    record.update(
                        {
                            "score": float(verdict.get("score", 0.0)),
                            "acc": bool(verdict.get("acc", False)),
                            "judge_raw": verdict.get("judge_raw"),
                            "judge_error": verdict.get("judge_error"),
                        }
                    )

        record["elapsed_seconds"] = round(time.time() - sample_started, 2)
        await append_record(record)
        metrics = record.get("metrics") or {}
        print(
            f"[{history_mode}] attempt task={record['task_id']} attempt={attempt}/{max_attempts} "
            f"acc={record.get('acc')} turns={metrics.get('agent_turns')} "
            f"tokens={metrics.get('agent_session_tokens')} forced={metrics.get('agent_forced_final_answer')} "
            f"elapsed={record['elapsed_seconds']:.1f}s",
            flush=True,
        )
        async with state_lock:
            valid_attempt = _resume_record_counts_as_attempt(record, require_judge=not args.skip_judge)
            if valid_attempt:
                _mark_valid_attempt(state, attempt)
                if record.get("acc") is True:
                    state["success"] = True
            if _is_task_finished(state, max_attempts, stop_on_success=stop_on_success):
                state["complete"] = True
            attempts = int(state.get("attempts") or 0)
            success = bool(state.get("success"))
            completed = _count_finished_tasks(samples, state_by_id, max_attempts, stop_on_success=stop_on_success)
        print(
            f"[{history_mode}] progress completed={completed}/{len(samples)} task={sample['task_id']} "
            f"attempts={attempts}/{max_attempts} success={success} last_acc={record.get('acc')} "
            f"turns={metrics.get('agent_turns')} tokens={metrics.get('agent_session_tokens')} "
            f"forced={metrics.get('agent_forced_final_answer')}",
            flush=True,
        )
        return record

    new_records: list[dict[str, Any]] = []
    for attempt in range(1, max_attempts + 1):
        retry_round_limit = 1
        if fixed_sampling and args.retry_incomplete_fixed_samples:
            retry_round_limit += args.max_incomplete_fixed_retries
        for retry_round in range(1, retry_round_limit + 1):
            round_samples: list[dict[str, Any]] = []
            for sample in samples:
                state = state_by_id.get(sample["task_id"])
                attempts = int(state.get("attempts") or 0) if state else 0
                if attempts == attempt - 1 and not (stop_on_success and state and state.get("success")):
                    round_samples.append(sample)
            if not round_samples:
                break
            completed = _count_finished_tasks(samples, state_by_id, max_attempts, stop_on_success=stop_on_success)
            print(
                f"[{history_mode}] round attempt={attempt}/{max_attempts} retry_round={retry_round} "
                f"scheduled={len(round_samples)} completed={completed}/{len(samples)}",
                flush=True,
            )
            round_records = await asyncio.gather(*(one_attempt(sample, attempt) for sample in round_samples))
            new_records.extend(round_records)
        if fixed_sampling and args.retry_incomplete_fixed_samples:
            unresolved = []
            for sample in samples:
                state = state_by_id.get(sample["task_id"])
                attempts = int(state.get("attempts") or 0) if state else 0
                if attempts == attempt - 1:
                    unresolved.append(sample["task_id"])
            if unresolved:
                print(
                    f"[{history_mode}] WARNING: attempt={attempt}/{max_attempts} exhausted "
                    f"{args.max_incomplete_fixed_retries} additional retries for {len(unresolved)} task(s); "
                    "continuing other tasks and leaving these slots for --resume",
                    flush=True,
                )
    records = [*existing_records, *new_records]
    expected_valid_records = len(samples) * args.samples_per_task if fixed_sampling else None
    summary = _aggregate(
        records,
        model=args.model,
        history_mode=history_mode,
        elapsed=time.time() - started,
        input_mode=args.input_mode,
        input_sample_count=len(samples),
        expected_valid_records=expected_valid_records,
        require_judge=not args.skip_judge,
    )
    print(
        f"[{history_mode}] accuracy={summary['accuracy']:.4f} "
        f"correct={summary['correct']}/{summary['n']} "
        f"valid={summary['actual_valid_records']}/{summary['expected_valid_records']} out={out_path}",
        flush=True,
    )
    incomplete_task_ids = [
        sample["task_id"]
        for sample in samples
        if not _is_task_finished(state_by_id.get(sample["task_id"]), max_attempts, stop_on_success=stop_on_success)
    ]
    if fixed_sampling and args.retry_incomplete_fixed_samples and incomplete_task_ids:
        preview = ", ".join(incomplete_task_ids[:10])
        suffix = " ..." if len(incomplete_task_ids) > 10 else ""
        raise RuntimeError(
            f"fixed sampling remains incomplete for {len(incomplete_task_ids)}/{len(samples)} task(s) after "
            f"{args.max_incomplete_fixed_retries} additional retries per attempt; resume the same output to retry: "
            f"{preview}{suffix}"
        )
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--continue-from", help="old keep5 output JSONL; continue all max-turn stops regardless of score"
    )
    parser.add_argument("--data", help="JSONL or Parquet file containing questions and reference answers")
    parser.add_argument("--out", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url", required=True, help="OpenAI-compatible base URL, with or without /v1")
    parser.add_argument(
        "--input-mode",
        choices=INPUT_MODES,
        default="question",
        help="read a plain question or replay the validated prefix_messages list from each JSON row",
    )
    parser.add_argument("--agent-module", default="adaptive_branching.src.deep_research.agent")
    parser.add_argument("--agent-api", choices=("chat", "responses"), default="chat")
    parser.add_argument(
        "--save-raw-responses",
        action="store_true",
        help="save every complete raw Responses API object, including heterogeneous output items and usage",
    )
    system_prompt_group = parser.add_mutually_exclusive_group()
    system_prompt_group.add_argument(
        "--system-prompt",
        default=None,
        help="Optional system prompt override for this evaluation process; the shared agent default is unchanged",
    )
    system_prompt_group.add_argument(
        "--no-system-prompt",
        action="store_true",
        help="Send no system-role message; tool schemas are still passed separately",
    )
    parser.add_argument("--history-mode", choices=("react", "keep5", "both"), default="keep5")
    parser.add_argument("--concurrency", type=int, default=128)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--max-turns", type=int, default=200)
    parser.add_argument("--max-tokens", type=int, default=22000)
    parser.add_argument("--max-seq-len", type=int, default=262144)
    parser.add_argument("--context-reserve-tokens", type=int, default=32768)
    parser.add_argument("--keep-tool-results", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=-1)
    parser.add_argument("--repetition-penalty", type=float, default=None)
    parser.add_argument("--reasoning-effort", default=None)
    parser.add_argument(
        "--thinking-budget",
        type=int,
        default=None,
        help="hard per-turn GLM-5 reasoning-token budget enforced by an SGLang custom logit processor",
    )
    parser.add_argument(
        "--tool-call-text-retries",
        type=int,
        default=0,
        help=(
            "resample the same model turn when its content looks like a textual tool call but structured "
            "tool_calls are absent (default: 0, disabled)"
        ),
    )
    parser.add_argument("--thinking-type", default=None)
    parser.add_argument(
        "--extra-request-json",
        default='{"chat_template_kwargs":{"clear_thinking":false}}',
        help=(
            "JSON object merged into chat completion request kwargs after standard generation options; "
            "chat_template_kwargs.clear_thinking must remain false"
        ),
    )
    parser.add_argument("--chat-timeout", type=float, default=1800.0)
    parser.add_argument("--endpoint-check-timeout", type=float, default=60.0)
    parser.add_argument("--skip-judge", action="store_true")
    parser.add_argument("--skip-endpoint-check", action="store_true")
    parser.add_argument("--keep-going", action="store_true", help="record per-sample errors instead of failing fast")
    parser.add_argument(
        "--resume", action="store_true", help="skip completed records already present in the output JSONL"
    )
    parser.add_argument(
        "--retry-failed-attempts",
        type=int,
        default=0,
        help="additional attempts to append for each sample until one attempt is judged correct",
    )
    parser.add_argument(
        "--samples-per-task",
        type=int,
        default=1,
        help="fixed independent samples per task; values >1 disable success early-stop",
    )
    parser.add_argument(
        "--retry-incomplete-fixed-samples",
        action="store_true",
        help=(
            "with --samples-per-task >1 and --keep-going, retry generation/judge errors in the current process "
            "up to --max-incomplete-fixed-retries times per slot; fail after preserving partial output if any "
            "fixed slot remains incomplete"
        ),
    )
    parser.add_argument(
        "--max-incomplete-fixed-retries",
        type=int,
        default=3,
        help="maximum additional in-process retries per fixed attempt slot (default: 3)",
    )
    return parser.parse_args()


def _validate_input_configuration(args: argparse.Namespace) -> None:
    if args.input_mode not in INPUT_MODES:
        raise ValueError(f"--input-mode={args.input_mode!r}; expected one of {INPUT_MODES}")
    if args.input_mode != PREFIX_INPUT_MODE:
        return
    if args.history_mode != "react":
        raise ValueError(
            "--input-mode prefix_messages requires --history-mode react; keep5/both mutate replay history"
        )
    if args.system_prompt is not None or args.no_system_prompt:
        raise ValueError(
            "--input-mode prefix_messages forbids --system-prompt/--no-system-prompt; "
            "the source prefix's system message is immutable"
        )


async def _amain() -> None:
    args = _parse_args()
    os.environ.setdefault("AGENT_TOOLS_CONFIG", str(REPO_ROOT / "adaptive_branching/config/eval_tools.yaml"))
    os.environ.setdefault("AGENT_JUDGE_CONFIG", str(REPO_ROOT / "adaptive_branching/config/judge.yaml"))
    os.environ.setdefault("AGENT_SEARCH_PROVIDER", "serper")
    _validate_input_configuration(args)
    if args.concurrency <= 0:
        raise ValueError("--concurrency must be positive")
    if args.retry_failed_attempts < 0:
        raise ValueError("--retry-failed-attempts must be >= 0")
    if args.samples_per_task <= 0:
        raise ValueError("--samples-per-task must be positive")
    if args.samples_per_task > 1 and args.retry_failed_attempts:
        raise ValueError("--samples-per-task >1 cannot be combined with --retry-failed-attempts")
    if args.retry_incomplete_fixed_samples and args.samples_per_task <= 1:
        raise ValueError("--retry-incomplete-fixed-samples requires --samples-per-task >1")
    if args.retry_incomplete_fixed_samples and not args.keep_going:
        raise ValueError("--retry-incomplete-fixed-samples requires --keep-going")
    if args.max_incomplete_fixed_retries < 0:
        raise ValueError("--max-incomplete-fixed-retries must be >= 0")
    if args.tool_call_text_retries < 0:
        raise ValueError("--tool-call-text-retries must be >= 0")
    if args.save_raw_responses and args.agent_api != "responses":
        raise ValueError("--save-raw-responses requires --agent-api=responses")
    if args.max_turns <= 0 or args.max_tokens <= 0 or args.max_seq_len <= 0:
        raise ValueError("max-turns/max-tokens/max-seq-len must be positive")
    if (
        args.context_reserve_tokens <= 0
        or args.keep_tool_results <= 0
        or args.chat_timeout <= 0
        or args.endpoint_check_timeout <= 0
    ):
        raise ValueError(
            "context-reserve-tokens/keep-tool-results/chat-timeout/endpoint-check-timeout must be positive"
        )
    if not args.skip_judge:
        outcome_config = judge_settings("full_outcome")
        required_str(outcome_config, "base_url")
        required_str(outcome_config, "api_key")
        required_str(outcome_config, "model")
        args.judge_concurrency = positive_int(outcome_config, "max_concurrency")
    else:
        args.judge_concurrency = 1
    if not os.getenv("LLM_API_KEY"):
        raise ValueError("LLM_API_KEY is required by the agent tools config")
    os.environ["AGENT_CHAT_TIMEOUT"] = str(args.chat_timeout)
    os.environ["AGENT_CONTEXT_RESERVE_TOKENS"] = str(args.context_reserve_tokens)

    request_kwargs = _request_kwargs(args)
    _base_url(args.base_url)
    if args.context_reserve_tokens >= args.max_seq_len:
        raise ValueError("context-reserve-tokens must be smaller than max-seq-len")
    if not args.continue_from and not args.data:
        raise ValueError("--data is required unless --continue-from is provided")
    if args.continue_from:
        if (
            args.input_mode != "question"
            or args.history_mode != "keep5"
            or args.agent_api != "chat"
            or args.samples_per_task != 1
            or args.retry_failed_attempts
            or args.limit is not None
            or args.offset
            or args.system_prompt is not None
            or args.no_system_prompt
        ):
            raise ValueError(
                "--continue-from requires question/keep5/chat, one attempt, no subset or prompt overrides"
            )
        source_path = Path(args.continue_from).resolve()
        output_path = Path(args.out) / f"{Path(args.model.rstrip('/')).name}.keep5.jsonl"
        if source_path == output_path.resolve():
            raise ValueError("--continue-from must write to a separate output path")
        samples = load_continuation_samples(
            source_path,
            model=args.model,
            total_turns=args.max_turns,
            max_seq_len=args.max_seq_len,
            keep_tool_results=args.keep_tool_results,
            final_prompt=importlib.import_module(args.agent_module)._final_answer_prompt,
        )
    else:
        samples = _load_samples(
            Path(args.data),
            limit=args.limit,
            offset=args.offset,
            input_mode=args.input_mode,
            max_turns=args.max_turns,
        )
    if not args.skip_endpoint_check:
        await _check_agent_endpoint(
            _base_url(args.base_url), args.model, args.endpoint_check_timeout, request_kwargs, args.agent_api
        )
    print(
        f"model={args.model} input_mode={args.input_mode} samples={len(samples)} "
        f"expected_valid_records={len(samples) * args.samples_per_task} "
        f"modes={_history_modes(args.history_mode)} "
        f"concurrency={args.concurrency} "
        f"agent_module={args.agent_module} agent_api={args.agent_api} "
        f"thinking_budget={args.thinking_budget} "
        f"chat_template_kwargs={request_kwargs.get('chat_template_kwargs')} out={args.out}",
        flush=True,
    )
    summaries = {}
    for mode in _history_modes(args.history_mode):
        summaries[mode] = await _run_mode(args, samples, mode)

    summary_path = Path(args.out) / "summary.json"
    summary_path.write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"summary -> {summary_path}", flush=True)


if __name__ == "__main__":
    asyncio.run(_amain())
