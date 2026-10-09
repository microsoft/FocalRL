#!/usr/bin/env python3
"""Aggregate value-cliff prefix rollouts without embedding full message histories."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from adaptive_branching.src.deep_research.value_cliff_prefixes import (  # noqa: E402
    PREFIX_INPUT_MODE,
    PREFIX_PROVENANCE_FIELDS,
    validate_prefix_row,
)


SCHEMA_VERSION = 1
SHARD_ARTIFACT_TYPE = "value_cliff_shard_aggregate"
COMBINED_ARTIFACT_TYPE = "value_cliff_combined_results"


def _positive_int(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return value


def _nonempty_string(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def _read_jsonl(path: Path) -> Iterable[tuple[int, dict[str, Any]]]:
    if not path.is_file():
        raise FileNotFoundError(f"JSONL file does not exist: {path}")
    saw_record = False
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise ValueError(f"blank line at {path}:{line_number}")
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}: {exc}") from exc
            if not isinstance(record, dict):
                raise TypeError(f"record at {path}:{line_number} must be an object")
            saw_record = True
            yield line_number, record
    if not saw_record:
        raise ValueError(f"JSONL file is empty: {path}")


def _invalid_reason(record: Mapping[str, Any]) -> str | None:
    if record.get("gen_error"):
        return "gen_error"
    if record.get("judge_error"):
        return "judge_error"
    metrics = record.get("metrics")
    if not isinstance(metrics, dict):
        return "missing_metrics"
    if metrics.get("agent_finished") is False:
        finish_reason = str(metrics.get("agent_last_finish_reason") or "unknown")
        return f"unfinished:{finish_reason}"
    return None


def _validate_valid_rollout(
    record: Mapping[str, Any],
    prefix: Mapping[str, Any],
    *,
    line_number: int,
    pass_k: int,
) -> tuple[int, bool]:
    name = f"rollouts[{line_number}]"
    attempt = _positive_int(record.get("attempt"), name=f"{name}.attempt")
    if attempt > pass_k:
        raise ValueError(f"{name}.attempt={attempt} exceeds pass_k={pass_k}")
    if record.get("samples_per_task") != pass_k or record.get("max_attempts") != pass_k:
        raise ValueError(
            f"{name} has samples_per_task/max_attempts="
            f"{(record.get('samples_per_task'), record.get('max_attempts'))!r}; expected {(pass_k, pass_k)!r}"
        )
    if record.get("input_mode") != PREFIX_INPUT_MODE:
        raise ValueError(f"{name}.input_mode={record.get('input_mode')!r}; expected {PREFIX_INPUT_MODE!r}")
    if record.get("question") != prefix["question"] or record.get("ground_truth") != prefix["ground_truth"]:
        raise ValueError(f"{name} question/ground_truth differs from its prefix input")
    for field in PREFIX_PROVENANCE_FIELDS:
        if record.get(field) != prefix[field]:
            raise ValueError(
                f"{name}.{field}={record.get(field)!r}; expected prefix value {prefix[field]!r}"
            )
    acc = record.get("acc")
    if not isinstance(acc, bool):
        raise TypeError(f"{name}.acc must be boolean, got {acc!r}")
    return attempt, acc


def aggregate_shard(
    prefix_path: Path,
    rollout_path: Path,
    *,
    pass_k: int,
    zero_prefix_task_ids: Iterable[str] = (),
) -> dict[str, Any]:
    """Aggregate one shard, imputing only explicitly approved all-zero prefixes."""

    pass_k = _positive_int(pass_k, name="pass_k")
    zero_prefix_ids = {_nonempty_string(value, name="zero_prefix_task_id") for value in zero_prefix_task_ids}

    prefixes: dict[str, dict[str, Any]] = {}
    for position, (_line_number, raw_prefix) in enumerate(_read_jsonl(prefix_path)):
        prefix = validate_prefix_row(raw_prefix, name=f"prefixes[{position}]")
        task_id = prefix["task_id"]
        if task_id in prefixes:
            raise ValueError(f"duplicate prefix task_id in {prefix_path}: {task_id!r}")
        prefixes[task_id] = prefix
    unknown_zero_ids = sorted(zero_prefix_ids - set(prefixes))
    if unknown_zero_ids:
        raise ValueError(f"zero-prefix overrides are absent from this shard: {unknown_zero_ids}")

    valid_by_task: dict[str, dict[int, tuple[bool, int]]] = defaultdict(dict)
    invalid_by_reason: Counter[str] = Counter()
    invalid_by_task: Counter[str] = Counter()
    models: set[str] = set()
    history_modes: set[str] = set()
    physical_records = 0
    for line_number, record in _read_jsonl(rollout_path):
        physical_records += 1
        task_id = _nonempty_string(record.get("task_id"), name=f"rollouts[{line_number}].task_id")
        if task_id not in prefixes:
            raise ValueError(f"rollout task {task_id!r} at {rollout_path}:{line_number} is absent from prefix input")
        invalid_reason = _invalid_reason(record)
        if invalid_reason is not None:
            invalid_by_reason[invalid_reason] += 1
            invalid_by_task[task_id] += 1
            continue
        attempt, acc = _validate_valid_rollout(record, prefixes[task_id], line_number=line_number, pass_k=pass_k)
        if attempt in valid_by_task[task_id]:
            previous_line = valid_by_task[task_id][attempt][1]
            raise ValueError(
                f"duplicate valid attempt for task={task_id!r} attempt={attempt}: "
                f"lines {previous_line} and {line_number}"
            )
        valid_by_task[task_id][attempt] = (acc, line_number)
        models.add(_nonempty_string(record.get("model"), name=f"rollouts[{line_number}].model"))
        history_modes.add(
            _nonempty_string(record.get("history_mode"), name=f"rollouts[{line_number}].history_mode")
        )
    if len(models) != 1 or len(history_modes) != 1:
        raise ValueError(f"expected one model/history_mode, got models={sorted(models)} modes={sorted(history_modes)}")

    aggregate_prefixes: list[dict[str, Any]] = []
    observed_valid_suffixes = 0
    imputed_failure_suffixes = 0
    correct_suffixes = 0
    solved_prefixes = 0
    correct_count_histogram: Counter[int] = Counter()
    sorted_prefixes = sorted(
        prefixes.items(), key=lambda item: (item[1]["source_cohort_index"], item[1]["prefix_turn"])
    )
    for task_id, prefix in sorted_prefixes:
        observed = valid_by_task.get(task_id, {})
        if task_id in zero_prefix_ids and any(acc for acc, _line in observed.values()):
            raise ValueError(f"refusing to force prefix {task_id!r} to zero because it has an observed correct suffix")
        missing_attempts = sorted(set(range(1, pass_k + 1)) - set(observed))
        if missing_attempts and task_id not in zero_prefix_ids:
            raise ValueError(
                f"prefix {task_id!r} is missing attempts {missing_attempts}; "
                "add an explicit zero-prefix override only after auditing the missing outcomes"
            )

        outcomes: list[dict[str, Any]] = []
        for attempt in range(1, pass_k + 1):
            if attempt in observed:
                acc, line_number = observed[attempt]
                outcomes.append(
                    {"attempt": attempt, "acc": acc, "imputed_failure": False, "rollout_line": line_number}
                )
            else:
                outcomes.append({"attempt": attempt, "acc": False, "imputed_failure": True, "rollout_line": None})
        correct_count = sum(int(outcome["acc"]) for outcome in outcomes)
        observed_valid = len(observed)
        imputed = pass_k - observed_valid
        observed_valid_suffixes += observed_valid
        imputed_failure_suffixes += imputed
        correct_suffixes += correct_count
        solved_prefixes += int(correct_count > 0)
        correct_count_histogram[correct_count] += 1
        total_turns = prefix["source_trajectory_assistant_turns"]
        progress = prefix["prefix_turn"] / (total_turns - 1) if total_turns > 1 else 0.0
        aggregate_prefixes.append(
            {
                "task_id": task_id,
                "source_task_id": prefix["source_task_id"],
                "source_cohort_index": prefix["source_cohort_index"],
                "source_failure_sample_index": prefix["source_failure_sample_index"],
                "source_failure_line_number": prefix["source_failure_line_number"],
                "source_trajectory_assistant_turns": total_turns,
                "prefix_turn": prefix["prefix_turn"],
                "prefix_kind": prefix["prefix_kind"],
                "normalized_progress": progress,
                "k4_success_count": prefix["k4_success_count"],
                "k4_failure_count": prefix["k4_failure_count"],
                "question": prefix["question"],
                "ground_truth": prefix["ground_truth"],
                "observed_valid_suffixes": observed_valid,
                "imputed_failure_suffixes": imputed,
                "invalid_physical_records": invalid_by_task[task_id],
                "correct_count": correct_count,
                "sample_count": pass_k,
                "value": correct_count / pass_k,
                "pass_at_k": correct_count > 0,
                "attempts": outcomes,
            }
        )

    effective_suffixes = len(aggregate_prefixes) * pass_k
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": SHARD_ARTIFACT_TYPE,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "contains_full_rollout_messages": False,
        "policy": {
            "pass_k": pass_k,
            "valid_record": "no gen/judge error, metrics object, and agent_finished is not false",
            "invalid_physical_records": "excluded from logical attempts",
            "zero_prefix_task_ids": sorted(zero_prefix_ids),
            "zero_prefix_semantics": (
                "observed correct suffixes must be zero; missing logical attempts are imputed acc=false"
            ),
        },
        "sources": {"prefixes": str(prefix_path), "rollouts": str(rollout_path)},
        "model": next(iter(models)),
        "history_mode": next(iter(history_modes)),
        "summary": {
            "prefixes": len(aggregate_prefixes),
            "source_tasks": len({prefix["source_task_id"] for prefix in aggregate_prefixes}),
            "physical_rollout_records": physical_records,
            "invalid_physical_records": sum(invalid_by_reason.values()),
            "invalid_physical_records_by_reason": dict(sorted(invalid_by_reason.items())),
            "observed_valid_suffixes": observed_valid_suffixes,
            "imputed_failure_suffixes": imputed_failure_suffixes,
            "effective_suffixes": effective_suffixes,
            "correct_suffixes": correct_suffixes,
            "accuracy": correct_suffixes / effective_suffixes,
            "solved_prefixes": solved_prefixes,
            "pass_at_k_rate": solved_prefixes / len(aggregate_prefixes),
            "correct_count_histogram": {str(key): correct_count_histogram.get(key, 0) for key in range(pass_k + 1)},
        },
        "prefixes": aggregate_prefixes,
    }


def _trajectory_records(prefixes: list[dict[str, Any]], *, pass_k: int) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for prefix in prefixes:
        grouped[prefix["source_task_id"]].append(prefix)
    trajectories: list[dict[str, Any]] = []
    for source_task_id, curve in grouped.items():
        curve.sort(key=lambda prefix: prefix["prefix_turn"])
        first = curve[0]
        last = curve[-1]
        if first["prefix_turn"] != 0:
            raise ValueError(f"trajectory {source_task_id!r} does not start at prefix_turn=0")
        if last["prefix_turn"] != last["source_trajectory_assistant_turns"] - 1:
            raise ValueError(f"trajectory {source_task_id!r} does not end at its pre-final state")
        identity_fields = (
            "source_cohort_index",
            "source_failure_sample_index",
            "source_failure_line_number",
            "source_trajectory_assistant_turns",
            "question",
            "ground_truth",
        )
        for field in identity_fields:
            if len({json.dumps(prefix[field], sort_keys=True) for prefix in curve}) != 1:
                raise ValueError(f"trajectory {source_task_id!r} has inconsistent {field}")
        compact_curve = [
            {
                "task_id": prefix["task_id"],
                "prefix_turn": prefix["prefix_turn"],
                "normalized_progress": prefix["normalized_progress"],
                "correct_count": prefix["correct_count"],
                "sample_count": pass_k,
                "value": prefix["value"],
                "pass_at_k": prefix["pass_at_k"],
                "imputed_failure_suffixes": prefix["imputed_failure_suffixes"],
            }
            for prefix in curve
        ]
        adjacent_drops = [curve[index]["value"] - curve[index + 1]["value"] for index in range(len(curve) - 1)]
        trajectories.append(
            {
                "source_task_id": source_task_id,
                "source_cohort_index": first["source_cohort_index"],
                "source_failure_sample_index": first["source_failure_sample_index"],
                "source_failure_line_number": first["source_failure_line_number"],
                "source_trajectory_assistant_turns": first["source_trajectory_assistant_turns"],
                "question": first["question"],
                "ground_truth": first["ground_truth"],
                "prefixes": len(curve),
                "turn0_value": first["value"],
                "pre_final_value": last["value"],
                "endpoint_drop": first["value"] - last["value"],
                "largest_adjacent_drop": max(adjacent_drops, default=0.0),
                "curve": compact_curve,
            }
        )
    return sorted(trajectories, key=lambda trajectory: trajectory["source_cohort_index"])


def merge_shard_aggregates(
    shards: list[Mapping[str, Any]],
    *,
    expected_prefixes: int | None = None,
    expected_source_tasks: int | None = None,
) -> dict[str, Any]:
    """Merge validated shard aggregates and derive trajectory/global summaries."""

    if not isinstance(shards, list) or not shards:
        raise ValueError("shards must be a non-empty list")
    pass_values = {shard.get("policy", {}).get("pass_k") for shard in shards}
    models = {shard.get("model") for shard in shards}
    history_modes = {shard.get("history_mode") for shard in shards}
    if len(pass_values) != 1 or None in pass_values:
        raise ValueError(f"shards have inconsistent pass_k values: {pass_values}")
    if len(models) != 1 or None in models or len(history_modes) != 1 or None in history_modes:
        raise ValueError(f"shards have inconsistent model/history mode: {models}, {history_modes}")
    pass_k = _positive_int(next(iter(pass_values)), name="pass_k")

    prefixes: list[dict[str, Any]] = []
    seen_task_ids: set[str] = set()
    source_shards: dict[str, int] = {}
    for shard_index, shard in enumerate(shards):
        if shard.get("schema_version") != SCHEMA_VERSION or shard.get("artifact_type") != SHARD_ARTIFACT_TYPE:
            raise ValueError(f"shards[{shard_index}] is not a supported shard aggregate")
        shard_prefixes = shard.get("prefixes")
        if not isinstance(shard_prefixes, list) or not shard_prefixes:
            raise ValueError(f"shards[{shard_index}].prefixes must be a non-empty list")
        for prefix in shard_prefixes:
            if not isinstance(prefix, dict):
                raise TypeError(f"shards[{shard_index}] contains a non-object prefix")
            task_id = _nonempty_string(prefix.get("task_id"), name="prefix.task_id")
            if task_id in seen_task_ids:
                raise ValueError(f"duplicate prefix task_id across shards: {task_id!r}")
            seen_task_ids.add(task_id)
            source_task_id = _nonempty_string(prefix.get("source_task_id"), name="prefix.source_task_id")
            previous_shard = source_shards.setdefault(source_task_id, shard_index)
            if previous_shard != shard_index:
                raise ValueError(f"source trajectory {source_task_id!r} is split across shards")
            attempts = prefix.get("attempts")
            if not isinstance(attempts, list) or len(attempts) != pass_k:
                raise ValueError(f"prefix {task_id!r} must contain exactly {pass_k} attempts")
            prefixes.append(dict(prefix))
    prefixes.sort(key=lambda prefix: (prefix["source_cohort_index"], prefix["prefix_turn"]))
    trajectories = _trajectory_records(prefixes, pass_k=pass_k)

    if expected_prefixes is not None:
        expected_prefixes = _positive_int(expected_prefixes, name="expected_prefixes")
        if len(prefixes) != expected_prefixes:
            raise ValueError(f"merged {len(prefixes)} prefixes; expected {expected_prefixes}")
    if expected_source_tasks is not None:
        expected_source_tasks = _positive_int(expected_source_tasks, name="expected_source_tasks")
        if len(trajectories) != expected_source_tasks:
            raise ValueError(f"merged {len(trajectories)} source tasks; expected {expected_source_tasks}")

    effective_suffixes = len(prefixes) * pass_k
    correct_suffixes = sum(prefix["correct_count"] for prefix in prefixes)
    observed_valid_suffixes = sum(prefix["observed_valid_suffixes"] for prefix in prefixes)
    imputed_failure_suffixes = sum(prefix["imputed_failure_suffixes"] for prefix in prefixes)
    if observed_valid_suffixes + imputed_failure_suffixes != effective_suffixes:
        raise AssertionError("observed plus imputed suffixes do not equal the effective suffix budget")
    correct_count_histogram = Counter(prefix["correct_count"] for prefix in prefixes)
    solved_prefixes = sum(int(prefix["pass_at_k"]) for prefix in prefixes)

    relative_deciles: list[dict[str, Any]] = []
    for decile in range(10):
        selected = [
            prefix
            for prefix in prefixes
            if min(9, int(math.floor(prefix["normalized_progress"] * 10))) == decile
        ]
        if not selected:
            raise ValueError(f"relative progress decile {decile} is empty")
        decile_correct = sum(prefix["correct_count"] for prefix in selected)
        decile_suffixes = len(selected) * pass_k
        relative_deciles.append(
            {
                "decile": decile,
                "progress_start": decile / 10,
                "progress_end": (decile + 1) / 10,
                "prefixes": len(selected),
                "correct_suffixes": decile_correct,
                "suffixes": decile_suffixes,
                "mean_value": decile_correct / decile_suffixes,
            }
        )

    turn0 = [prefix for prefix in prefixes if prefix["prefix_turn"] == 0]
    pre_final = [
        prefix
        for prefix in prefixes
        if prefix["prefix_turn"] == prefix["source_trajectory_assistant_turns"] - 1
    ]
    if len(turn0) != len(trajectories) or len(pre_final) != len(trajectories):
        raise AssertionError("every trajectory must contribute exactly one turn-0 and pre-final prefix")
    turn0_correct = sum(prefix["correct_count"] for prefix in turn0)
    pre_final_correct = sum(prefix["correct_count"] for prefix in pre_final)
    endpoint_suffixes = len(trajectories) * pass_k

    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": COMBINED_ARTIFACT_TYPE,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "contains_full_rollout_messages": False,
        "model": next(iter(models)),
        "history_mode": next(iter(history_modes)),
        "policy": {
            "pass_k": pass_k,
            "value": "correct_count / pass_k",
            "pass_at_k": "correct_count > 0",
            "imputed_failures_are_explicit": True,
        },
        "sources": [dict(shard["sources"]) for shard in shards],
        "summary": {
            "source_tasks": len(trajectories),
            "prefixes": len(prefixes),
            "observed_valid_suffixes": observed_valid_suffixes,
            "imputed_failure_suffixes": imputed_failure_suffixes,
            "effective_suffixes": effective_suffixes,
            "correct_suffixes": correct_suffixes,
            "accuracy": correct_suffixes / effective_suffixes,
            "solved_prefixes": solved_prefixes,
            "pass_at_k_rate": solved_prefixes / len(prefixes),
            "correct_count_histogram": {str(key): correct_count_histogram.get(key, 0) for key in range(pass_k + 1)},
            "turn0": {
                "prefixes": len(turn0),
                "correct_suffixes": turn0_correct,
                "suffixes": endpoint_suffixes,
                "mean_value": turn0_correct / endpoint_suffixes,
            },
            "pre_final": {
                "prefixes": len(pre_final),
                "correct_suffixes": pre_final_correct,
                "suffixes": endpoint_suffixes,
                "mean_value": pre_final_correct / endpoint_suffixes,
            },
            "endpoint_mean_value_drop": (turn0_correct - pre_final_correct) / endpoint_suffixes,
            "relative_progress_deciles": relative_deciles,
        },
        "trajectories": trajectories,
        "prefixes": prefixes,
    }


def _load_json_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"JSON file does not exist: {path}")
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"JSON file must contain an object: {path}")
    return value


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    """Write a new JSON artifact atomically without overwriting prior results."""

    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {path}")
    if not path.name:
        raise ValueError("output path must have a final component")
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.with_name(f".{path.name}.tmp")
    if staging.exists():
        raise FileExistsError(f"staging path already exists: {staging}")
    try:
        with staging.open("w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
        staging.replace(path)
    except BaseException:
        staging.unlink(missing_ok=True)
        raise


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    shard = subparsers.add_parser("shard", help="aggregate one prefix/rollout shard")
    shard.add_argument("--prefixes", type=Path, required=True)
    shard.add_argument("--rollouts", type=Path, required=True)
    shard.add_argument("--output", type=Path, required=True)
    shard.add_argument("--pass-k", type=int, default=8)
    shard.add_argument("--zero-prefix-task-id", action="append", default=[])

    merge = subparsers.add_parser("merge", help="merge shard aggregate JSON files")
    merge.add_argument("--input", type=Path, action="append", required=True)
    merge.add_argument("--output", type=Path, required=True)
    merge.add_argument("--expected-prefixes", type=int)
    merge.add_argument("--expected-source-tasks", type=int)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.command == "shard":
        result = aggregate_shard(
            args.prefixes,
            args.rollouts,
            pass_k=args.pass_k,
            zero_prefix_task_ids=args.zero_prefix_task_id,
        )
    elif args.command == "merge":
        result = merge_shard_aggregates(
            [_load_json_object(path) for path in args.input],
            expected_prefixes=args.expected_prefixes,
            expected_source_tasks=args.expected_source_tasks,
        )
    else:  # pragma: no cover - argparse enforces the choices
        raise AssertionError(f"unexpected command: {args.command}")
    write_json(args.output, result)
    print(json.dumps(result["summary"], ensure_ascii=False, sort_keys=True))
    print(f"output -> {args.output}")


if __name__ == "__main__":
    main()
