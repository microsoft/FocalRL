#!/usr/bin/env python3
"""Locate one value-drop action and generate its local recovery rubric."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from adaptive_branching.src.deep_research.judge_client import JudgeClient
from adaptive_branching.src.deep_research.judge_config import judge_settings, positive_int
from adaptive_branching.src.deep_research.value_cliff_locator import (
    assistant_turn_count,
    build_value_cliff_locator_prompt as build_locator_prompt,
)
from adaptive_branching.src.deep_research.value_cliff_rubric import (
    VALUE_CLIFF_LOCATOR_REQUIRED_KEYS as REQUIRED_KEYS,
    VALUE_CLIFF_LOCATOR_SYSTEM as RUBRIC_LOCATOR_SYSTEM,
    VALUE_CLIFF_LOCATOR_VERSION as SCORER_VERSION,
    validate_value_cliff_locator_verdict as validate_rubric_locator_verdict,
)

INPUT_MODE = "failed_trajectory_plus_matched_success_and_ground_truth"


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows: list[dict[str, Any]] = []
    with path.open() as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                raise ValueError(f"blank line at {path}:{line_number}")
            row = json.loads(line)
            if not isinstance(row, dict):
                raise TypeError(f"row at {path}:{line_number} must be an object")
            rows.append(row)
    if not rows:
        raise ValueError(f"JSONL is empty: {path}")
    return rows


def _index_source_rows(rows: list[dict[str, Any]], *, expected_role: str) -> dict[str, dict[str, Any]]:
    if expected_role not in {"failure", "matched_success"}:
        raise ValueError(f"unsupported expected_role: {expected_role!r}")
    indexed: dict[str, dict[str, Any]] = {}
    for position, row in enumerate(rows):
        task_id = row.get("task_id")
        if not isinstance(task_id, str) or not task_id or task_id in indexed:
            raise ValueError(f"invalid or duplicate task_id at {expected_role}[{position}]: {task_id!r}")
        selection = row.get("_value_cliff_selection")
        if not isinstance(selection, dict) or selection.get("role") != expected_role:
            raise ValueError(f"{expected_role}[{position}] has invalid selection role")
        if selection.get("task_id") != task_id:
            raise ValueError(f"{expected_role}[{position}] selection task_id mismatch")
        expected_acc = expected_role == "matched_success"
        if row.get("acc") is not expected_acc:
            raise ValueError(f"{expected_role}[{position}].acc must be {expected_acc}")
        indexed[task_id] = row
    return indexed


def pair_source_rows(
    failures: list[dict[str, Any]], successes: list[dict[str, Any]]
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    failure_by_id = _index_source_rows(failures, expected_role="failure")
    success_by_id = _index_source_rows(successes, expected_role="matched_success")
    if set(failure_by_id) != set(success_by_id):
        raise ValueError(
            f"failure/success task IDs differ: failure_only={len(set(failure_by_id) - set(success_by_id))}, "
            f"success_only={len(set(success_by_id) - set(failure_by_id))}"
        )
    pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for failure in failures:
        success = success_by_id[failure["task_id"]]
        for key in ("question", "ground_truth"):
            if (
                failure.get(key) != success.get(key)
                or not isinstance(failure.get(key), str)
                or not failure[key].strip()
            ):
                raise ValueError(f"paired task {failure['task_id']} has invalid or different {key}")
        failure_selection = failure["_value_cliff_selection"]
        success_selection = success["_value_cliff_selection"]
        for key in ("cohort_index", "failure_sample_index", "matched_success_sample_index"):
            if failure_selection.get(key) != success_selection.get(key):
                raise ValueError(f"paired task {failure['task_id']} has different selection.{key}")
        if failure.get("sample_index") != failure_selection.get("failure_sample_index"):
            raise ValueError(f"failure sample index mismatch for task {failure['task_id']}")
        if success.get("sample_index") != success_selection.get("matched_success_sample_index"):
            raise ValueError(f"success sample index mismatch for task {failure['task_id']}")
        pairs.append((failure, success))
    return pairs


def _load_completed(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    completed: dict[str, dict[str, Any]] = {}
    for line_number, row in enumerate(_load_jsonl(path), 1):
        if row.get("schema_version") != SCORER_VERSION:
            raise ValueError(f"unexpected schema_version at {path}:{line_number}")
        task_id = row.get("task_id")
        if not isinstance(task_id, str) or not task_id or task_id in completed:
            raise ValueError(f"invalid or duplicate task_id at {path}:{line_number}")
        completed[task_id] = row
    return completed


def validate_resume_rows(
    completed: dict[str, dict[str, Any]],
    pairs: list[tuple[dict[str, Any], dict[str, Any]]],
    *,
    judge_model: str,
    judge_reasoning_effort: str,
    judge_max_tokens: int,
    judge_sampling_params: dict[str, int | float],
    trace_max_chars: int,
    judge_enable_thinking: bool | None = None,
) -> None:
    if judge_enable_thinking is not None and not isinstance(judge_enable_thinking, bool):
        raise ValueError("judge_enable_thinking must be a boolean or None")
    pair_by_id = {failure["task_id"]: (failure, success) for failure, success in pairs}
    if len(pair_by_id) != len(pairs):
        raise ValueError("pairs contain duplicate failure task IDs")
    unknown = set(completed) - set(pair_by_id)
    if unknown:
        raise ValueError(f"resume output contains task IDs outside this run: {sorted(unknown)[:5]}")
    for task_id, row in completed.items():
        if row.get("judge_enable_thinking") is not judge_enable_thinking:
            raise ValueError(f"resume row {task_id} has different judge_enable_thinking")
        failure, success = pair_by_id[task_id]
        expected = {
            "input_mode": INPUT_MODE,
            "failure_sample_index": failure.get("sample_index"),
            "matched_success_sample_index": success.get("sample_index"),
            "failed_assistant_turns": assistant_turn_count(failure.get("messages")),
            "successful_assistant_turns": assistant_turn_count(success.get("messages")),
            "judge_model": judge_model,
            "judge_reasoning_effort": judge_reasoning_effort,
            "judge_max_tokens": judge_max_tokens,
            "judge_sampling_params": judge_sampling_params,
            "trace_max_chars": trace_max_chars,
        }
        for key, value in expected.items():
            if row.get(key) != value:
                raise ValueError(f"resume row {task_id} has {key}={row.get(key)!r}; expected {value!r}")


async def _score_pair(
    client: JudgeClient,
    semaphore: asyncio.Semaphore,
    *,
    failure: dict[str, Any],
    success: dict[str, Any],
    trace_max_chars: int,
) -> dict[str, Any]:
    task_id = failure["task_id"]
    prompt, prompt_metadata = build_locator_prompt(
        question=failure["question"],
        ground_truth=failure["ground_truth"],
        failed_messages=failure.get("messages"),
        successful_messages=success.get("messages"),
        trace_max_chars=trace_max_chars,
    )
    async with semaphore:
        verdict = await client.complete_json(
            RUBRIC_LOCATOR_SYSTEM,
            prompt,
            required_keys=REQUIRED_KEYS,
            tag=f"value_cliff_rubric_locator_{task_id}",
            validate=lambda raw: validate_rubric_locator_verdict(
                raw, assistant_turns=prompt_metadata["failed_assistant_turns"]
            ),
        )
    return {
        "schema_version": SCORER_VERSION,
        "input_mode": INPUT_MODE,
        "task_id": task_id,
        "source_cohort_index": failure["_value_cliff_selection"].get("cohort_index"),
        "failure_sample_index": failure.get("sample_index"),
        "matched_success_sample_index": success.get("sample_index"),
        "judge_model": client.model,
        "judge_reasoning_effort": client.reasoning_effort,
        "judge_enable_thinking": client.enable_thinking,
        "judge_max_tokens": client.max_tokens,
        "judge_sampling_params": client.sampling_params,
        "trace_max_chars": trace_max_chars,
        **prompt_metadata,
        "verdict": verdict,
    }


async def _run(args: argparse.Namespace) -> None:
    failures = _load_jsonl(args.failures)
    successes = _load_jsonl(args.successes)
    pairs = pair_source_rows(failures, successes)
    if args.limit is not None:
        pairs = pairs[: args.limit]
    if not pairs:
        raise ValueError("no source pairs selected")

    settings = judge_settings("locator")
    trace_max_chars = positive_int(settings, "trace_max_chars")
    concurrency = args.concurrency or positive_int(settings, "max_concurrency")
    client = JudgeClient(config_section="locator")
    semaphore = asyncio.Semaphore(concurrency)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    completed = _load_completed(args.output)
    expected_ids = {failure["task_id"] for failure, _ in pairs}
    validate_resume_rows(
        completed,
        pairs,
        judge_model=client.model,
        judge_reasoning_effort=client.reasoning_effort,
        judge_enable_thinking=client.enable_thinking,
        judge_max_tokens=client.max_tokens,
        judge_sampling_params=client.sampling_params,
        trace_max_chars=trace_max_chars,
    )
    pending = [(failure, success) for failure, success in pairs if failure["task_id"] not in completed]
    print(
        f"START scorer={SCORER_VERSION} model={client.model} expected={len(pairs)} "
        f"resumed={len(completed)} pending={len(pending)} concurrency={concurrency}",
        flush=True,
    )

    async def tagged_score(failure: dict[str, Any], success: dict[str, Any]) -> dict[str, Any]:
        try:
            return await _score_pair(
                client,
                semaphore,
                failure=failure,
                success=success,
                trace_max_chars=trace_max_chars,
            )
        except Exception as exc:
            raise RuntimeError(f"rubric locator failed for task_id={failure['task_id']}: {exc}") from exc

    tasks = [asyncio.create_task(tagged_score(failure, success)) for failure, success in pending]
    written = len(completed)
    try:
        with args.output.open("a", buffering=1) as output:
            for future in asyncio.as_completed(tasks):
                row = await future
                output.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")
                output.flush()
                os.fsync(output.fileno())
                written += 1
                print(f"PROGRESS valid={written}/{len(pairs)} task_id={row['task_id']}", flush=True)
    except Exception:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise

    final = _load_completed(args.output)
    if set(final) != expected_ids:
        raise RuntimeError(f"incomplete rubric locator output: valid={len(final)}/{len(expected_ids)}")
    print(f"SUCCESS valid={len(final)}/{len(expected_ids)} output={args.output}", flush=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--failures", type=Path, required=True)
    parser.add_argument("--successes", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--concurrency", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    if args.concurrency is not None and args.concurrency <= 0:
        parser.error("--concurrency must be positive")
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    return args


def main() -> None:
    asyncio.run(_run(_parse_args()))


if __name__ == "__main__":
    main()
