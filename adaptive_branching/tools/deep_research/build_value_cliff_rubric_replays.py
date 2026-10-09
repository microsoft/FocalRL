#!/usr/bin/env python3
"""Extract bounded local replays at rubric-locator turns from one rollout shard."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from adaptive_branching.src.deep_research.value_cliff_prefixes import validate_prefix_row
from adaptive_branching.src.deep_research.value_cliff_rubric import (
    DEFAULT_LOCAL_HORIZON_ASSISTANT_TURNS,
    VALUE_CLIFF_FIXED_TURN_RUBRIC_VERSION,
)
from adaptive_branching.tools.deep_research.aggregate_value_cliff_results import (
    COMBINED_ARTIFACT_TYPE,
    SCHEMA_VERSION as AGGREGATE_SCHEMA_VERSION,
)
from adaptive_branching.tools.deep_research.score_value_cliff_rubric_locator import (
    SCORER_VERSION as LOCATOR_VERSION,
    validate_rubric_locator_verdict,
)


REPLAY_SCHEMA_VERSION = "value_cliff_rubric_replay_v1_20260821"


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise TypeError(f"JSON root must be an object: {path}")
    return value


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


def _message_digest(messages: list[dict[str, Any]]) -> str:
    if not isinstance(messages, list) or not messages or any(not isinstance(row, dict) for row in messages):
        raise ValueError("messages must be a non-empty list of objects")
    payload = json.dumps(messages, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def local_continuation(messages: list[dict[str, Any]], *, max_assistant_turns: int) -> tuple[list[dict[str, Any]], int]:
    if not isinstance(messages, list) or not messages or any(not isinstance(row, dict) for row in messages):
        raise ValueError("continuation messages must be a non-empty list of objects")
    if isinstance(max_assistant_turns, bool) or not isinstance(max_assistant_turns, int) or max_assistant_turns <= 0:
        raise ValueError("max_assistant_turns must be a positive integer")
    assistant_positions = [index for index, message in enumerate(messages) if message.get("role") == "assistant"]
    if not assistant_positions:
        raise ValueError("continuation contains no assistant turns")
    stop = assistant_positions[max_assistant_turns] if len(assistant_positions) > max_assistant_turns else len(messages)
    bounded = messages[:stop]
    bounded_turns = sum(message.get("role") == "assistant" for message in bounded)
    if bounded_turns != min(len(assistant_positions), max_assistant_turns):
        raise AssertionError("bounded continuation has an unexpected assistant-turn count")
    return bounded, len(assistant_positions)


def _selected_prefixes(combined: dict[str, Any], locator_rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    if combined.get("schema_version") != AGGREGATE_SCHEMA_VERSION:
        raise ValueError("unsupported combined-results schema")
    if combined.get("artifact_type") != COMBINED_ARTIFACT_TYPE:
        raise ValueError("combined input is not a value-cliff combined-results artifact")
    prefixes = combined.get("prefixes")
    if not isinstance(prefixes, list) or not prefixes:
        raise ValueError("combined.prefixes must be a non-empty list")
    prefix_index: dict[tuple[str, int], dict[str, Any]] = {}
    for position, prefix in enumerate(prefixes):
        if not isinstance(prefix, dict):
            raise TypeError(f"combined.prefixes[{position}] must be an object")
        key = (prefix.get("source_task_id"), prefix.get("prefix_turn"))
        if not isinstance(key[0], str) or not key[0] or isinstance(key[1], bool) or not isinstance(key[1], int):
            raise ValueError(f"invalid combined prefix identity at index {position}")
        if key in prefix_index:
            raise ValueError(f"duplicate combined prefix identity: {key}")
        prefix_index[key] = prefix

    selected: dict[str, dict[str, Any]] = {}
    seen_source_ids: set[str] = set()
    for position, row in enumerate(locator_rows):
        row_schema = row.get("schema_version")
        if row_schema not in {LOCATOR_VERSION, VALUE_CLIFF_FIXED_TURN_RUBRIC_VERSION}:
            raise ValueError(f"unexpected locator schema at row {position + 1}")
        source_task_id = row.get("task_id")
        assistant_turns = row.get("failed_assistant_turns")
        if not isinstance(source_task_id, str) or not source_task_id or source_task_id in seen_source_ids:
            raise ValueError(f"invalid or duplicate locator task_id at row {position + 1}")
        verdict = validate_rubric_locator_verdict(row.get("verdict"), assistant_turns=assistant_turns)
        selected_turn = verdict["selected_turn"]
        if row_schema == VALUE_CLIFF_FIXED_TURN_RUBRIC_VERSION and row.get("selected_turn") != selected_turn:
            raise ValueError(f"fixed-turn rubric row {source_task_id} has inconsistent selected_turn")
        aggregate = prefix_index.get((source_task_id, selected_turn - 1))
        if aggregate is None:
            raise ValueError(f"selected prefix is absent for task={source_task_id} turn={selected_turn}")
        prefix_task_id = aggregate.get("task_id")
        if not isinstance(prefix_task_id, str) or not prefix_task_id or prefix_task_id in selected:
            raise ValueError(f"invalid or duplicate selected prefix task_id: {prefix_task_id!r}")
        attempts = aggregate.get("attempts")
        if not isinstance(attempts, list) or len(attempts) != 8:
            raise ValueError(f"selected prefix {prefix_task_id} must contain exactly eight attempts")
        attempt_numbers = {attempt.get("attempt") for attempt in attempts if isinstance(attempt, dict)}
        if attempt_numbers != set(range(1, 9)):
            raise ValueError(f"selected prefix {prefix_task_id} has invalid attempt numbers: {attempt_numbers}")
        for attempt in attempts:
            if attempt.get("imputed_failure") is not False or not isinstance(attempt.get("rollout_line"), int):
                raise ValueError(f"selected prefix {prefix_task_id} contains an imputed or missing attempt")
            if not isinstance(attempt.get("acc"), bool):
                raise TypeError(f"selected prefix {prefix_task_id} attempt acc must be boolean")
        selected[prefix_task_id] = {
            "aggregate": aggregate,
            "selected_turn": selected_turn,
            "recovery_rubric": verdict["recovery_rubric"],
        }
        seen_source_ids.add(source_task_id)
    if not selected:
        raise ValueError("no locator rows were selected")
    return selected


def extract_shard_replays(
    *,
    combined: dict[str, Any],
    locator_rows: list[dict[str, Any]],
    prefix_rows: list[dict[str, Any]],
    rollout_rows: Iterable[tuple[int, dict[str, Any]]],
    rollout_source: str,
    max_assistant_turns: int = DEFAULT_LOCAL_HORIZON_ASSISTANT_TURNS,
) -> list[dict[str, Any]]:
    selected = _selected_prefixes(combined, locator_rows)
    shard_prefixes: dict[str, dict[str, Any]] = {}
    for position, raw_prefix in enumerate(prefix_rows):
        prefix = validate_prefix_row(raw_prefix, name=f"prefixes[{position}]")
        task_id = prefix["task_id"]
        if task_id in selected:
            if task_id in shard_prefixes:
                raise ValueError(f"duplicate selected prefix in shard: {task_id}")
            shard_prefixes[task_id] = prefix
    if not shard_prefixes:
        raise ValueError("this shard contains none of the locator-selected prefixes")

    expected_by_line: dict[int, tuple[dict[str, Any], dict[str, Any], dict[str, Any]]] = {}
    for task_id, prefix in shard_prefixes.items():
        selection = selected[task_id]
        for attempt in selection["aggregate"]["attempts"]:
            line_number = attempt["rollout_line"]
            if line_number in expected_by_line:
                raise ValueError(f"duplicate rollout line across selected attempts: {line_number}")
            expected_by_line[line_number] = (prefix, selection, attempt)

    extracted: list[dict[str, Any]] = []
    seen_lines: set[int] = set()
    for line_number, record in rollout_rows:
        expected = expected_by_line.get(line_number)
        if expected is None:
            continue
        prefix, selection, attempt = expected
        task_id = prefix["task_id"]
        if record.get("task_id") != task_id or record.get("attempt") != attempt["attempt"]:
            raise ValueError(f"rollout identity mismatch at {rollout_source}:{line_number}")
        if record.get("acc") is not attempt["acc"]:
            raise ValueError(f"rollout outcome mismatch at {rollout_source}:{line_number}")
        if record.get("gen_error") or record.get("judge_error"):
            raise ValueError(f"selected rollout contains an error at {rollout_source}:{line_number}")
        metrics = record.get("metrics")
        if not isinstance(metrics, dict) or metrics.get("agent_finished") is not True:
            raise ValueError(f"selected rollout is unfinished at {rollout_source}:{line_number}")
        messages = record.get("messages")
        prefix_messages = prefix["prefix_messages"]
        if not isinstance(messages, list) or messages[: len(prefix_messages)] != prefix_messages:
            raise ValueError(f"rollout does not preserve its prefix at {rollout_source}:{line_number}")
        suffix_messages = messages[len(prefix_messages) :]
        bounded, full_turns = local_continuation(suffix_messages, max_assistant_turns=max_assistant_turns)
        if metrics.get("agent_turns") != full_turns:
            raise ValueError(f"agent_turns mismatch at {rollout_source}:{line_number}")
        local_turns = sum(message.get("role") == "assistant" for message in bounded)
        replay_id = f"{task_id}::attempt_{attempt['attempt']:02d}"
        extracted.append(
            {
                "schema_version": REPLAY_SCHEMA_VERSION,
                "replay_id": replay_id,
                "prefix_task_id": task_id,
                "source_task_id": prefix["source_task_id"],
                "source_cohort_index": prefix["source_cohort_index"],
                "selected_turn": selection["selected_turn"],
                "prefix_turn": prefix["prefix_turn"],
                "attempt": attempt["attempt"],
                "outcome_acc": attempt["acc"],
                "full_suffix_assistant_turns": full_turns,
                "local_assistant_turns": local_turns,
                "natural_end_within_horizon": full_turns <= max_assistant_turns,
                "rollout_source": rollout_source,
                "rollout_line": line_number,
                "prefix_sha256": _message_digest(prefix_messages),
                "prefix_messages": prefix_messages,
                "local_messages": bounded,
                "recovery_rubric": selection["recovery_rubric"],
            }
        )
        seen_lines.add(line_number)
    missing = sorted(set(expected_by_line) - seen_lines)
    if missing:
        raise ValueError(f"rollout shard is missing {len(missing)} selected lines; first={missing[:5]}")
    expected_count = len(shard_prefixes) * 8
    if len(extracted) != expected_count:
        raise AssertionError(f"expected {expected_count} extracted rows, got {len(extracted)}")
    return sorted(extracted, key=lambda row: (row["source_cohort_index"], row["attempt"]))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--combined", type=Path, required=True)
    parser.add_argument("--locator-scores", type=Path, required=True)
    parser.add_argument("--prefixes", type=Path, required=True)
    parser.add_argument("--rollouts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--local-turns", type=int, default=DEFAULT_LOCAL_HORIZON_ASSISTANT_TURNS)
    args = parser.parse_args()
    if args.local_turns <= 0:
        parser.error("--local-turns must be positive")
    return args


def main() -> None:
    args = _parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {args.output}")
    locator_rows = [row for _line, row in _read_jsonl(args.locator_scores)]
    prefix_rows = [row for _line, row in _read_jsonl(args.prefixes)]
    rows = extract_shard_replays(
        combined=_read_json(args.combined),
        locator_rows=locator_rows,
        prefix_rows=prefix_rows,
        rollout_rows=_read_jsonl(args.rollouts),
        rollout_source=str(args.rollouts),
        max_assistant_turns=args.local_turns,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")
    print(f"SUCCESS prefixes={len({row['prefix_task_id'] for row in rows})} replays={len(rows)} positive={sum(row['outcome_acc'] for row in rows)} output={args.output}")


if __name__ == "__main__":
    main()
