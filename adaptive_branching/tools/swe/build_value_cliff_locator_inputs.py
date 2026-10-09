#!/usr/bin/env python3
"""Build deterministic SWE value-cliff locator inputs from mixed rollouts."""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

INPUT_SCHEMA_VERSION = "swe_value_cliff_locator_input_v1_20260902"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
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


def _task_id(repo_name: Any, commit_hash: Any) -> str:
    if not isinstance(repo_name, str) or not repo_name.strip():
        raise ValueError("repo_name must be a non-empty string")
    if not isinstance(commit_hash, str) or len(commit_hash.strip()) != 40:
        raise ValueError(f"commit_hash must be a 40-character string, got {commit_hash!r}")
    return f"r2e-{repo_name.strip()}-{commit_hash.strip()}"


def _format_range(raw: Any, *, name: str) -> str:
    if not isinstance(raw, dict):
        raise TypeError(f"{name} must be an object")
    start = raw.get("start")
    length = raw.get("length")
    if isinstance(start, bool) or not isinstance(start, int) or start < 0:
        raise ValueError(f"{name}.start must be a non-negative integer")
    if isinstance(length, bool) or not isinstance(length, int) or length < 0:
        raise ValueError(f"{name}.length must be a non-negative integer")
    return f"{start},{length}"


def extract_golden_source_patch(parsed_commit_content: str, relevant_files: list[str]) -> str:
    if not isinstance(parsed_commit_content, str) or not parsed_commit_content.strip():
        raise ValueError("parsed_commit_content must be a non-empty string")
    if not isinstance(relevant_files, list) or not relevant_files:
        raise ValueError("relevant_files must be a non-empty list")
    if any(not isinstance(path, str) or not path.strip() for path in relevant_files):
        raise ValueError("relevant_files must contain non-empty strings")
    expected = set(relevant_files)
    if len(expected) != len(relevant_files):
        raise ValueError("relevant_files must not contain duplicates")

    parsed = json.loads(parsed_commit_content)
    if not isinstance(parsed, dict) or not isinstance(parsed.get("file_diffs"), list):
        raise ValueError("parsed_commit_content must contain a file_diffs list")

    sections: list[str] = []
    found: set[str] = set()
    for diff_index, file_diff in enumerate(parsed["file_diffs"]):
        if not isinstance(file_diff, dict):
            raise TypeError(f"file_diffs[{diff_index}] must be an object")
        header = file_diff.get("header")
        file_header = header.get("file") if isinstance(header, dict) else None
        path = file_header.get("path") if isinstance(file_header, dict) else None
        if path not in expected:
            continue
        if path in found:
            raise ValueError(f"duplicate file diff for relevant file {path!r}")
        if file_diff.get("is_binary_file"):
            raise ValueError(f"binary relevant file is unsupported: {path}")
        hunks = file_diff.get("hunks")
        if not isinstance(hunks, list) or not hunks:
            raise ValueError(f"relevant file has no hunks: {path}")

        lines = [f"diff --git a/{path} b/{path}", f"--- a/{path}", f"+++ b/{path}"]
        for hunk_index, hunk in enumerate(hunks):
            if not isinstance(hunk, dict):
                raise TypeError(f"{path}.hunks[{hunk_index}] must be an object")
            descriptor = hunk.get("descriptor")
            line_group = hunk.get("line_group")
            if not isinstance(descriptor, dict) or not isinstance(line_group, dict):
                raise ValueError(f"{path}.hunks[{hunk_index}] is missing descriptor or line_group")
            old_range = _format_range(descriptor.get("old_range"), name=f"{path}.old_range")
            new_range = _format_range(descriptor.get("new_range"), name=f"{path}.new_range")
            section = str(descriptor.get("section") or "").strip()
            lines.append(f"@@ -{old_range} +{new_range} @@{f' {section}' if section else ''}")
            all_lines = line_group.get("all_lines")
            if not isinstance(all_lines, list) or not all_lines:
                raise ValueError(f"{path}.hunks[{hunk_index}] has no lines")
            prefixes = {"context": " ", "added": "+", "deleted": "-"}
            for line_index, line in enumerate(all_lines):
                if isinstance(line, dict) and line.get("type") == "note":
                    if (
                        line.get("content") != "No newline at end of file"
                        or line_index == 0
                        or all_lines[line_index - 1].get("type") not in prefixes
                    ):
                        raise ValueError(f"invalid diff note at {path}.hunks[{hunk_index}].lines[{line_index}]")
                    lines.append("\\ No newline at end of file")
                    continue
                if not isinstance(line, dict) or line.get("type") not in prefixes:
                    raise ValueError(f"invalid line at {path}.hunks[{hunk_index}].lines[{line_index}]")
                content = line.get("content")
                if not isinstance(content, str):
                    raise TypeError(f"line content at {path}.hunks[{hunk_index}].lines[{line_index}] must be a string")
                lines.append(prefixes[line["type"]] + content)
        sections.append("\n".join(lines))
        found.add(path)

    missing = sorted(expected - found)
    if missing:
        raise ValueError(f"relevant files are missing from parsed commit diff: {missing}")
    return "\n\n".join(sections)


def _load_dataset_rows(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    import pyarrow.parquet as pq

    required = {"repo_name", "commit_hash", "parsed_commit_content", "relevant_files"}
    table = pq.read_table(path)
    missing = sorted(required - set(table.column_names))
    if missing:
        raise ValueError(f"task parquet is missing columns: {missing}")
    indexed: dict[str, dict[str, Any]] = {}
    for position, row in enumerate(table.select(sorted(required)).to_pylist()):
        task_id = _task_id(row["repo_name"], row["commit_hash"])
        if task_id in indexed:
            raise ValueError(f"duplicate task at parquet row {position}: {task_id}")
        indexed[task_id] = row
    if not indexed:
        raise ValueError(f"task parquet is empty: {path}")
    return indexed


def _load_prompts(path: Path) -> dict[str, str]:
    indexed: dict[str, str] = {}
    for position, row in enumerate(_read_jsonl(path)):
        metadata = row.get("metadata")
        instance_id = metadata.get("instance_id") if isinstance(metadata, dict) else None
        prompt = row.get("prompt")
        if not isinstance(instance_id, str) or not instance_id or instance_id in indexed:
            raise ValueError(f"invalid or duplicate prompt instance_id at row {position}")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(f"prompt is empty for {instance_id}")
        indexed[instance_id] = prompt.strip()
    return indexed


def _normalize_locator_messages(messages: Any, *, path: Path) -> list[dict[str, Any]]:
    if not isinstance(messages, list) or not messages:
        raise ValueError(f"trajectory has no messages: {path}")
    normalized: list[dict[str, Any]] = []
    for message_index, message in enumerate(messages):
        if not isinstance(message, dict):
            raise TypeError(f"message {message_index} is not an object: {path}")
        role = message.get("role")
        if not isinstance(role, str) or not role:
            raise ValueError(f"message {message_index} has invalid role: {path}")
        selected: dict[str, Any] = {"role": role}
        for key in ("content", "reasoning_content", "name", "tool_call_id"):
            value = message.get(key)
            if value is not None:
                if not isinstance(value, str):
                    raise TypeError(f"message {message_index}.{key} must be a string: {path}")
                selected[key] = value
        if role == "assistant" and message.get("tool_calls") is not None:
            calls = message["tool_calls"]
            if not isinstance(calls, list):
                raise TypeError(f"message {message_index}.tool_calls must be a list: {path}")
            selected_calls: list[dict[str, Any]] = []
            for call_index, call in enumerate(calls):
                function = call.get("function") if isinstance(call, dict) else None
                if not isinstance(function, dict):
                    raise TypeError(f"message {message_index}.tool_calls[{call_index}] has no function: {path}")
                name = function.get("name")
                arguments = function.get("arguments")
                if not isinstance(name, str) or not name or not isinstance(arguments, str):
                    raise ValueError(f"message {message_index}.tool_calls[{call_index}] is invalid: {path}")
                selected_calls.append({"function": {"name": name, "arguments": arguments}})
            selected["tool_calls"] = selected_calls
        normalized.append(selected)
    return normalized


def _trajectory_messages(trials_root: Path, rollout: dict[str, Any]) -> tuple[str, list[dict[str, Any]]]:
    response = rollout.get("response")
    trial_dir = response.get("trial_dir") if isinstance(response, dict) else None
    if not isinstance(trial_dir, str) or not trial_dir.strip():
        raise ValueError(f"rollout {rollout.get('job_id')!r} is missing response.trial_dir")
    trial_name = Path(trial_dir).name
    if not trial_name or trial_name in {".", ".."}:
        raise ValueError(f"invalid trial directory for {rollout.get('job_id')!r}: {trial_dir!r}")
    path = trials_root / trial_name / "agent" / "mini-swe-agent.trajectory.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    trajectory = json.loads(path.read_text())
    messages = trajectory.get("messages") if isinstance(trajectory, dict) else None
    return trial_name, _normalize_locator_messages(messages, path=path)


def build_locator_inputs(
    *,
    mixed_rollouts: list[dict[str, Any]],
    prompts: dict[str, str],
    dataset_rows: dict[str, dict[str, Any]],
    trials_root: Path,
) -> list[dict[str, Any]]:
    if not mixed_rollouts:
        raise ValueError("mixed_rollouts must be non-empty")
    if not prompts or not dataset_rows:
        raise ValueError("prompts and dataset_rows must be non-empty")
    if not trials_root.is_dir():
        raise NotADirectoryError(trials_root)

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for position, rollout in enumerate(mixed_rollouts):
        instance_id = rollout.get("instance_id")
        sample_index = rollout.get("sample_index")
        response = rollout.get("response")
        if not isinstance(instance_id, str) or not instance_id:
            raise ValueError(f"invalid instance_id at mixed rollout {position}")
        if isinstance(sample_index, bool) or not isinstance(sample_index, int) or sample_index < 0:
            raise ValueError(f"invalid sample_index at mixed rollout {position}")
        if not isinstance(response, dict):
            raise TypeError(f"response must be an object at mixed rollout {position}")
        reward = response.get("reward")
        if isinstance(reward, bool) or not isinstance(reward, (int, float)) or float(reward) not in (0.0, 1.0):
            raise ValueError(f"reward must be binary at mixed rollout {position}, got {reward!r}")
        grouped[instance_id].append(rollout)

    records: list[dict[str, Any]] = []
    for instance_id in sorted(grouped):
        submitted = [row for row in grouped[instance_id] if row["response"].get("exit_status") == "Submitted"]
        positives = sorted(
            (row for row in submitted if row["response"]["reward"] == 1), key=lambda row: row["sample_index"]
        )
        negatives = sorted(
            (row for row in submitted if row["response"]["reward"] == 0), key=lambda row: row["sample_index"]
        )
        if not positives or not negatives:
            continue
        if instance_id not in prompts:
            raise KeyError(f"missing prompt for {instance_id}")
        if instance_id not in dataset_rows:
            raise KeyError(f"missing parquet row for {instance_id}")

        positive = positives[0]
        negative = negatives[0]
        positive_trial, positive_messages = _trajectory_messages(trials_root, positive)
        negative_trial, negative_messages = _trajectory_messages(trials_root, negative)
        dataset_row = dataset_rows[instance_id]
        golden_patch = extract_golden_source_patch(
            dataset_row["parsed_commit_content"],
            dataset_row["relevant_files"],
        )
        records.append(
            {
                "schema_version": INPUT_SCHEMA_VERSION,
                "task_id": instance_id,
                "issue": prompts[instance_id],
                "golden_patch": golden_patch,
                "golden_patch_files": list(dataset_row["relevant_files"]),
                "failure_sample_index": negative["sample_index"],
                "matched_success_sample_index": positive["sample_index"],
                "failure_trial": negative_trial,
                "matched_success_trial": positive_trial,
                "failed_messages": negative_messages,
                "successful_messages": positive_messages,
            }
        )
    if not records:
        raise ValueError("no Submitted mixed positive/negative pairs were found")
    return records


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {path}")
    if not rows:
        raise ValueError("refusing to write empty locator input")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        with temporary.open("x") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mixed-rollouts", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--tasks-parquet", type=Path, required=True)
    parser.add_argument("--trials-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    records = build_locator_inputs(
        mixed_rollouts=_read_jsonl(args.mixed_rollouts),
        prompts=_load_prompts(args.prompts),
        dataset_rows=_load_dataset_rows(args.tasks_parquet),
        trials_root=args.trials_root,
    )
    _write_jsonl(args.output, records)
    print(f"SUCCESS tasks={len(records)} output={args.output}")


if __name__ == "__main__":
    main()
