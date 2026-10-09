#!/usr/bin/env python3
"""Score bounded value-cliff replays with their hidden recovery rubrics."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from adaptive_branching.src.deep_research.value_cliff_reward import (
    build_value_cliff_rubric_judge_prompt as _build_shared_rubric_judge_prompt,
)
from adaptive_branching.src.deep_research.judge_client import JudgeClient
from adaptive_branching.src.deep_research.judge_config import judge_settings, positive_int
from adaptive_branching.src.deep_research.value_cliff_rubric import (
    VALUE_CLIFF_JUDGE_REQUIRED_KEYS as REQUIRED_KEYS,
    VALUE_CLIFF_JUDGE_SYSTEM as RUBRIC_REPLAY_JUDGE_SYSTEM,
    VALUE_CLIFF_JUDGE_VERSION as SCORER_VERSION,
    validate_value_cliff_judge_verdict as validate_rubric_verdict,
)
from adaptive_branching.tools.deep_research.build_value_cliff_rubric_replays import REPLAY_SCHEMA_VERSION

DEFAULT_TRACE_MAX_CHARS = 240_000


def _read_jsonl(path: Path) -> Iterable[tuple[int, dict[str, Any]]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    saw_row = False
    with path.open() as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                raise ValueError(f"blank line at {path}:{line_number}")
            row = json.loads(line)
            if not isinstance(row, dict):
                raise TypeError(f"row at {path}:{line_number} must be an object")
            saw_row = True
            yield line_number, row
    if not saw_row:
        raise ValueError(f"JSONL file is empty: {path}")


def validate_replay_row(row: dict[str, Any], *, name: str) -> dict[str, Any]:
    if not isinstance(row, dict) or row.get("schema_version") != REPLAY_SCHEMA_VERSION:
        raise ValueError(f"{name} has an unsupported replay schema")
    replay_id = row.get("replay_id")
    prefix_task_id = row.get("prefix_task_id")
    source_task_id = row.get("source_task_id")
    if not isinstance(replay_id, str) or not replay_id:
        raise ValueError(f"{name}.replay_id must be a non-empty string")
    if not isinstance(prefix_task_id, str) or not prefix_task_id:
        raise ValueError(f"{name}.prefix_task_id must be a non-empty string")
    if not isinstance(source_task_id, str) or not source_task_id:
        raise ValueError(f"{name}.source_task_id must be a non-empty string")
    attempt = row.get("attempt")
    if isinstance(attempt, bool) or not isinstance(attempt, int) or not 1 <= attempt <= 8:
        raise ValueError(f"{name}.attempt must be an integer in [1, 8]")
    if replay_id != f"{prefix_task_id}::attempt_{attempt:02d}":
        raise ValueError(f"{name}.replay_id does not match prefix_task_id/attempt")
    if not isinstance(row.get("outcome_acc"), bool):
        raise TypeError(f"{name}.outcome_acc must be boolean")
    local_turns = row.get("local_assistant_turns")
    if isinstance(local_turns, bool) or not isinstance(local_turns, int) or not 1 <= local_turns <= 5:
        raise ValueError(f"{name}.local_assistant_turns must be an integer in [1, 5]")
    prefix_messages = row.get("prefix_messages")
    local_messages = row.get("local_messages")
    for key, messages in (("prefix_messages", prefix_messages), ("local_messages", local_messages)):
        if (
            not isinstance(messages, list)
            or not messages
            or any(not isinstance(message, dict) for message in messages)
        ):
            raise ValueError(f"{name}.{key} must be a non-empty list of objects")
    if sum(message.get("role") == "assistant" for message in local_messages) != local_turns:
        raise ValueError(f"{name}.local_messages does not match local_assistant_turns")
    rubric = row.get("recovery_rubric")
    if not isinstance(rubric, dict) or set(rubric) != {"avoid_error", "redirect"}:
        raise ValueError(f"{name}.recovery_rubric must contain exactly avoid_error and redirect")
    if any(not isinstance(rubric[key], str) or not rubric[key].strip() for key in rubric):
        raise ValueError(f"{name}.recovery_rubric criteria must be non-empty strings")
    return row


def build_rubric_judge_prompt(row: dict[str, Any], *, trace_max_chars: int) -> tuple[str, bool, bool]:
    validate_replay_row(row, name="replay")
    if isinstance(trace_max_chars, bool) or not isinstance(trace_max_chars, int) or trace_max_chars <= 0:
        raise ValueError("trace_max_chars must be a positive integer")
    return _build_shared_rubric_judge_prompt(
        prefix_messages=row["prefix_messages"],
        local_messages=row["local_messages"],
        recovery_rubric=row["recovery_rubric"],
        trace_max_chars=trace_max_chars,
    )


def _replay_digest(row: dict[str, Any]) -> str:
    payload = json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _load_replays(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for path in paths:
        for line_number, raw in _read_jsonl(path):
            row = validate_replay_row(raw, name=f"{path}:{line_number}")
            if row["replay_id"] in seen_ids:
                raise ValueError(f"duplicate replay_id across inputs: {row['replay_id']}")
            seen_ids.add(row["replay_id"])
            rows.append(row)
    if not rows:
        raise ValueError("no replay rows loaded")
    attempts_by_prefix: dict[str, set[int]] = {}
    for row in rows:
        attempts_by_prefix.setdefault(row["prefix_task_id"], set()).add(row["attempt"])
    malformed = {
        task_id: sorted(values) for task_id, values in attempts_by_prefix.items() if values != set(range(1, 9))
    }
    if malformed:
        raise ValueError(f"replay inputs contain incomplete prefix groups: {list(malformed.items())[:3]}")
    return sorted(rows, key=lambda row: (row["source_cohort_index"], row["attempt"]))


def _load_completed(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    completed: dict[str, dict[str, Any]] = {}
    for line_number, row in _read_jsonl(path):
        if row.get("schema_version") != SCORER_VERSION:
            raise ValueError(f"unexpected scorer schema at {path}:{line_number}")
        replay_id = row.get("replay_id")
        if not isinstance(replay_id, str) or not replay_id or replay_id in completed:
            raise ValueError(f"invalid or duplicate replay_id at {path}:{line_number}")
        completed[replay_id] = row
    return completed


def validate_resume_rows(
    completed: dict[str, dict[str, Any]],
    replays: list[dict[str, Any]],
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
    replay_by_id = {row["replay_id"]: row for row in replays}
    unknown = set(completed) - set(replay_by_id)
    if unknown:
        raise ValueError(f"resume output contains unknown replay IDs: {sorted(unknown)[:3]}")
    for replay_id, output in completed.items():
        if output.get("judge_enable_thinking") is not judge_enable_thinking:
            raise ValueError(f"resume row {replay_id} has different judge_enable_thinking")
        replay = replay_by_id[replay_id]
        expected = {
            "input_sha256": _replay_digest(replay),
            "judge_model": judge_model,
            "judge_reasoning_effort": judge_reasoning_effort,
            "judge_max_tokens": judge_max_tokens,
            "judge_sampling_params": judge_sampling_params,
            "trace_max_chars": trace_max_chars,
            "outcome_acc": replay["outcome_acc"],
        }
        for key, value in expected.items():
            if output.get(key) != value:
                raise ValueError(f"resume row {replay_id} has {key}={output.get(key)!r}; expected {value!r}")


async def _score_one(
    client: JudgeClient,
    semaphore: asyncio.Semaphore,
    row: dict[str, Any],
    *,
    trace_max_chars: int,
) -> dict[str, Any]:
    prompt, prefix_truncated, continuation_truncated = build_rubric_judge_prompt(row, trace_max_chars=trace_max_chars)
    async with semaphore:
        verdict = await client.complete_json(
            RUBRIC_REPLAY_JUDGE_SYSTEM,
            prompt,
            required_keys=REQUIRED_KEYS,
            tag=f"value_cliff_rubric_replay_{row['replay_id']}",
            validate=lambda raw: validate_rubric_verdict(raw, local_turns=row["local_assistant_turns"]),
        )
    return {
        "schema_version": SCORER_VERSION,
        "replay_id": row["replay_id"],
        "prefix_task_id": row["prefix_task_id"],
        "source_task_id": row["source_task_id"],
        "source_cohort_index": row["source_cohort_index"],
        "selected_turn": row["selected_turn"],
        "prefix_turn": row["prefix_turn"],
        "attempt": row["attempt"],
        "outcome_acc": row["outcome_acc"],
        "full_suffix_assistant_turns": row["full_suffix_assistant_turns"],
        "local_assistant_turns": row["local_assistant_turns"],
        "natural_end_within_horizon": row["natural_end_within_horizon"],
        "prefix_trace_truncated": prefix_truncated,
        "continuation_trace_truncated": continuation_truncated,
        "input_sha256": _replay_digest(row),
        "judge_model": client.model,
        "judge_reasoning_effort": client.reasoning_effort,
        "judge_enable_thinking": client.enable_thinking,
        "judge_max_tokens": client.max_tokens,
        "judge_sampling_params": client.sampling_params,
        "trace_max_chars": trace_max_chars,
        "verdict": verdict,
    }


async def _run(args: argparse.Namespace) -> None:
    replays = _load_replays(args.input)
    client = JudgeClient(config_section="local_prm")
    concurrency = args.concurrency or positive_int(judge_settings("local_prm"), "max_concurrency")
    semaphore = asyncio.Semaphore(concurrency)
    completed = _load_completed(args.output)
    validate_resume_rows(
        completed,
        replays,
        judge_model=client.model,
        judge_reasoning_effort=client.reasoning_effort,
        judge_enable_thinking=client.enable_thinking,
        judge_max_tokens=client.max_tokens,
        judge_sampling_params=client.sampling_params,
        trace_max_chars=args.trace_max_chars,
    )
    pending = [row for row in replays if row["replay_id"] not in completed]
    print(
        f"START scorer={SCORER_VERSION} model={client.model} expected={len(replays)} "
        f"resumed={len(completed)} pending={len(pending)} concurrency={concurrency}",
        flush=True,
    )

    async def tagged(row: dict[str, Any]) -> dict[str, Any]:
        try:
            return await _score_one(client, semaphore, row, trace_max_chars=args.trace_max_chars)
        except Exception as exc:
            raise RuntimeError(f"rubric replay judge failed for replay_id={row['replay_id']}: {exc}") from exc

    tasks = [asyncio.create_task(tagged(row)) for row in pending]
    written = len(completed)
    try:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("a", buffering=1) as handle:
            for future in asyncio.as_completed(tasks):
                result = await future
                handle.write(json.dumps(result, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
                written += 1
                print(f"PROGRESS valid={written}/{len(replays)} replay_id={result['replay_id']}", flush=True)
    except Exception:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise

    final = _load_completed(args.output)
    if {row["replay_id"] for row in replays} != set(final):
        raise RuntimeError(f"incomplete rubric replay output: valid={len(final)}/{len(replays)}")
    print(f"SUCCESS valid={len(final)}/{len(replays)} output={args.output}", flush=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--concurrency", type=int, default=None)
    parser.add_argument("--trace-max-chars", type=int, default=DEFAULT_TRACE_MAX_CHARS)
    args = parser.parse_args()
    if args.concurrency is not None and args.concurrency <= 0:
        parser.error("--concurrency must be positive")
    if args.trace_max_chars <= 0:
        parser.error("--trace-max-chars must be positive")
    return args


def main() -> None:
    asyncio.run(_run(_parse_args()))


if __name__ == "__main__":
    main()
