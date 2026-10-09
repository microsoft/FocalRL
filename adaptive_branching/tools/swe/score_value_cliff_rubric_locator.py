#!/usr/bin/env python3
"""Run the GPT SWE value-cliff locator over prepared mixed pairs."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
from typing import Any

from adaptive_branching.src.deep_research.judge_client import JudgeClient
from adaptive_branching.src.deep_research.judge_config import judge_settings, positive_int
from adaptive_branching.src.swe.value_cliff_locator import (
    DEFAULT_SWE_LOCAL_HORIZON_ASSISTANT_TURNS,
    SWE_VALUE_CLIFF_LOCATOR_VERSION,
    SWE_VALUE_CLIFF_REQUIRED_KEYS,
    build_swe_value_cliff_locator_prompt,
    swe_value_cliff_locator_system,
    validate_swe_value_cliff_verdict,
)
from adaptive_branching.tools.swe.build_value_cliff_locator_inputs import INPUT_SCHEMA_VERSION


def _read_inputs(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    with path.open() as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                raise ValueError(f"blank line at {path}:{line_number}")
            row = json.loads(line)
            if not isinstance(row, dict):
                raise TypeError(f"row at {path}:{line_number} must be an object")
            if row.get("schema_version") != INPUT_SCHEMA_VERSION:
                raise ValueError(f"unexpected schema_version at {path}:{line_number}")
            task_id = row.get("task_id")
            if not isinstance(task_id, str) or not task_id or task_id in seen:
                raise ValueError(f"invalid or duplicate task_id at {path}:{line_number}")
            for key in ("issue", "golden_patch", "failure_trial", "matched_success_trial"):
                if not isinstance(row.get(key), str) or not row[key].strip():
                    raise ValueError(f"{key} must be non-empty at {path}:{line_number}")
            for key in ("failed_messages", "successful_messages"):
                messages = row.get(key)
                if (
                    not isinstance(messages, list)
                    or not messages
                    or any(not isinstance(item, dict) for item in messages)
                ):
                    raise ValueError(f"{key} must be a non-empty list of objects at {path}:{line_number}")
            seen.add(task_id)
            rows.append(row)
    if not rows:
        raise ValueError(f"locator input is empty: {path}")
    return rows


def _load_completed(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    completed: dict[str, dict[str, Any]] = {}
    with path.open() as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                raise ValueError(f"blank line at {path}:{line_number}")
            row = json.loads(line)
            task_id = row.get("task_id") if isinstance(row, dict) else None
            if not isinstance(row, dict) or row.get("schema_version") != SWE_VALUE_CLIFF_LOCATOR_VERSION:
                raise ValueError(f"invalid locator output at {path}:{line_number}")
            if not isinstance(task_id, str) or not task_id or task_id in completed:
                raise ValueError(f"invalid or duplicate output task_id at {path}:{line_number}")
            completed[task_id] = row
    return completed


def _validate_resume(
    completed: dict[str, dict[str, Any]],
    inputs: list[dict[str, Any]],
    *,
    client: JudgeClient,
    trace_max_chars: int,
    local_horizon_turns: int,
) -> None:
    input_by_id = {row["task_id"]: row for row in inputs}
    unknown = sorted(set(completed) - set(input_by_id))
    if unknown:
        raise ValueError(f"resume output contains unknown task IDs: {unknown[:5]}")
    expected_config = {
        "judge_model": client.model,
        "judge_reasoning_effort": client.reasoning_effort,
        "judge_max_tokens": client.max_tokens,
        "judge_sampling_params": client.sampling_params,
        "trace_max_chars": trace_max_chars,
        "local_horizon_turns": local_horizon_turns,
    }
    for task_id, output in completed.items():
        source = input_by_id[task_id]
        expected = {
            **expected_config,
            "failure_sample_index": source["failure_sample_index"],
            "matched_success_sample_index": source["matched_success_sample_index"],
            "failure_trial": source["failure_trial"],
            "matched_success_trial": source["matched_success_trial"],
        }
        for key, value in expected.items():
            if output.get(key) != value:
                raise ValueError(f"resume row {task_id} has {key}={output.get(key)!r}; expected {value!r}")


async def _score_one(
    client: JudgeClient,
    semaphore: asyncio.Semaphore,
    row: dict[str, Any],
    *,
    trace_max_chars: int,
    local_horizon_turns: int,
) -> dict[str, Any]:
    prompt, prompt_metadata = build_swe_value_cliff_locator_prompt(
        issue=row["issue"],
        golden_patch=row["golden_patch"],
        failed_messages=row["failed_messages"],
        successful_messages=row["successful_messages"],
        trace_max_chars=trace_max_chars,
    )
    system = swe_value_cliff_locator_system(local_horizon_turns)
    async with semaphore:
        verdict = await client.complete_json(
            system,
            prompt,
            required_keys=SWE_VALUE_CLIFF_REQUIRED_KEYS,
            tag=f"swe_value_cliff_locator_{row['task_id']}",
            validate=lambda raw: validate_swe_value_cliff_verdict(
                raw,
                assistant_turns=prompt_metadata["failed_assistant_turns"],
            ),
        )
    return {
        "schema_version": SWE_VALUE_CLIFF_LOCATOR_VERSION,
        "task_id": row["task_id"],
        "failure_sample_index": row["failure_sample_index"],
        "matched_success_sample_index": row["matched_success_sample_index"],
        "failure_trial": row["failure_trial"],
        "matched_success_trial": row["matched_success_trial"],
        "golden_patch_files": row["golden_patch_files"],
        "judge_model": client.model,
        "judge_reasoning_effort": client.reasoning_effort,
        "judge_max_tokens": client.max_tokens,
        "judge_sampling_params": client.sampling_params,
        "trace_max_chars": trace_max_chars,
        "local_horizon_turns": local_horizon_turns,
        **prompt_metadata,
        "verdict": verdict,
    }


async def _run(args: argparse.Namespace) -> None:
    inputs = _read_inputs(args.inputs)
    if args.limit is not None:
        inputs = inputs[: args.limit]
    if not inputs:
        raise ValueError("no locator inputs selected")

    settings = judge_settings("locator")
    trace_max_chars = positive_int(settings, "trace_max_chars")
    concurrency = args.concurrency or positive_int(settings, "max_concurrency")
    client = JudgeClient(config_section="locator")
    semaphore = asyncio.Semaphore(concurrency)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    completed = _load_completed(args.output)
    _validate_resume(
        completed,
        inputs,
        client=client,
        trace_max_chars=trace_max_chars,
        local_horizon_turns=args.local_horizon_turns,
    )
    expected_ids = {row["task_id"] for row in inputs}
    pending = [row for row in inputs if row["task_id"] not in completed]
    print(
        f"START scorer={SWE_VALUE_CLIFF_LOCATOR_VERSION} model={client.model} expected={len(inputs)} "
        f"resumed={len(completed)} pending={len(pending)} concurrency={concurrency}",
        flush=True,
    )

    async def tagged_score(row: dict[str, Any]) -> dict[str, Any]:
        try:
            return await _score_one(
                client,
                semaphore,
                row,
                trace_max_chars=trace_max_chars,
                local_horizon_turns=args.local_horizon_turns,
            )
        except Exception as exc:
            raise RuntimeError(f"SWE locator failed for task_id={row['task_id']}: {exc}") from exc

    tasks = [asyncio.create_task(tagged_score(row)) for row in pending]
    written = len(completed)
    try:
        with args.output.open("a", buffering=1) as output:
            for future in asyncio.as_completed(tasks):
                row = await future
                output.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")
                output.flush()
                os.fsync(output.fileno())
                written += 1
                print(f"PROGRESS valid={written}/{len(inputs)} task_id={row['task_id']}", flush=True)
    except Exception:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise

    final = _load_completed(args.output)
    if set(final) != expected_ids:
        raise RuntimeError(f"incomplete SWE locator output: valid={len(final)}/{len(expected_ids)}")
    print(f"SUCCESS valid={len(final)}/{len(expected_ids)} output={args.output}", flush=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--concurrency", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--local-horizon-turns",
        type=int,
        default=DEFAULT_SWE_LOCAL_HORIZON_ASSISTANT_TURNS,
    )
    args = parser.parse_args()
    for name in ("concurrency", "limit", "local_horizon_turns"):
        value = getattr(args, name)
        if value is not None and value <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    return args


def main() -> None:
    asyncio.run(_run(_parse_args()))


if __name__ == "__main__":
    main()
