"""Validated data contract for value-cliff prefix replay."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


PREFIX_SCHEMA_VERSION = 1
PREFIX_INPUT_MODE = "prefix_messages"
PREFIX_KINDS = frozenset({"coarse_grid", "pre_final", "coarse_grid_and_pre_final"})
PREFIX_PROVENANCE_FIELDS = (
    "schema_version",
    "source_task_id",
    "source_cohort_index",
    "source_failure_sample_index",
    "source_failure_line_number",
    "source_trajectory_assistant_turns",
    "source_max_turns",
    "prefix_index",
    "prefix_turn",
    "prefix_kind",
    "remaining_turns",
    "k4_success_count",
    "k4_failure_count",
)


def _required_nonempty_string(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def _required_nonnegative_int(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _required_positive_int(value: Any, *, name: str) -> int:
    value = _required_nonnegative_int(value, name=name)
    if value == 0:
        raise ValueError(f"{name} must be positive")
    return value


def _validate_message_objects(messages: Any, *, name: str) -> list[dict[str, Any]]:
    if not isinstance(messages, list) or not messages:
        raise ValueError(f"{name} must be a non-empty list")
    validated: list[dict[str, Any]] = []
    for message_index, message in enumerate(messages):
        if not isinstance(message, dict):
            raise TypeError(f"{name}[{message_index}] must be an object")
        _required_nonempty_string(message.get("role"), name=f"{name}[{message_index}].role")
        validated.append(message)
    return validated


def _tool_call_ids(assistant: Mapping[str, Any], *, name: str) -> list[str]:
    calls = assistant.get("tool_calls")
    if not isinstance(calls, list) or not calls:
        raise ValueError(f"{name}.tool_calls must be a non-empty list")
    ids: list[str] = []
    for call_index, call in enumerate(calls):
        if not isinstance(call, Mapping):
            raise TypeError(f"{name}.tool_calls[{call_index}] must be an object")
        ids.append(_required_nonempty_string(call.get("id"), name=f"{name}.tool_calls[{call_index}].id"))
    if len(set(ids)) != len(ids):
        raise ValueError(f"{name}.tool_calls contains duplicate ids: {ids}")
    return ids


def validate_replay_prefix_messages(
    messages: Any,
    *,
    question: str,
    prefix_turn: int,
    name: str,
) -> list[dict[str, Any]]:
    """Validate a state immediately before an assistant action."""

    question = _required_nonempty_string(question, name=f"{name}.question")
    prefix_turn = _required_nonnegative_int(prefix_turn, name=f"{name}.prefix_turn")
    messages = _validate_message_objects(messages, name=f"{name}.prefix_messages")
    if len(messages) < 2:
        raise ValueError(f"{name}.prefix_messages must contain at least system and user messages")
    roles = [message["role"] for message in messages]
    if roles[0] != "system":
        raise ValueError(f"{name}.prefix_messages[0] must have role='system', got {roles[0]!r}")
    if roles[1] != "user":
        raise ValueError(f"{name}.prefix_messages[1] must have role='user', got {roles[1]!r}")
    if sum(role == "system" for role in roles) != 1:
        raise ValueError(f"{name}.prefix_messages must contain exactly one system message")
    if sum(role == "user" for role in roles) != 1:
        raise ValueError(f"{name}.prefix_messages must contain exactly one user message")
    if messages[1].get("content") != question:
        raise ValueError(f"{name}.prefix_messages user content does not equal question")

    cursor = 2
    observed_assistant_turns = 0
    while cursor < len(messages):
        assistant = messages[cursor]
        if assistant.get("role") != "assistant":
            raise ValueError(
                f"{name}.prefix_messages[{cursor}] must be an assistant action, got {assistant.get('role')!r}"
            )
        expected_tool_ids = _tool_call_ids(assistant, name=f"{name}.prefix_messages[{cursor}]")
        cursor += 1
        observed_tool_ids: list[str] = []
        while cursor < len(messages) and messages[cursor].get("role") == "tool":
            observed_tool_ids.append(
                _required_nonempty_string(
                    messages[cursor].get("tool_call_id"),
                    name=f"{name}.prefix_messages[{cursor}].tool_call_id",
                )
            )
            cursor += 1
        if observed_tool_ids != expected_tool_ids:
            raise ValueError(
                f"{name} assistant turn {observed_assistant_turns} tool observation ids "
                f"{observed_tool_ids} do not match tool-call ids {expected_tool_ids}"
            )
        observed_assistant_turns += 1

    if observed_assistant_turns != prefix_turn:
        raise ValueError(
            f"{name}.prefix_turn={prefix_turn} but prefix contains {observed_assistant_turns} assistant turns"
        )
    expected_last_role = "user" if prefix_turn == 0 else "tool"
    if roles[-1] != expected_last_role:
        raise ValueError(
            f"{name}.prefix_messages must end with role={expected_last_role!r} before the next assistant action, "
            f"got {roles[-1]!r}"
        )
    return messages


def validate_prefix_row(
    row: Mapping[str, Any],
    *,
    name: str,
    expected_source_max_turns: int | None = None,
) -> dict[str, Any]:
    """Validate and normalize one prefix-replay input row."""

    if not isinstance(row, Mapping):
        raise TypeError(f"{name} must be an object")
    schema_version = _required_positive_int(row.get("schema_version"), name=f"{name}.schema_version")
    if schema_version != PREFIX_SCHEMA_VERSION:
        raise ValueError(f"{name}.schema_version={schema_version}; expected supported version {PREFIX_SCHEMA_VERSION}")
    task_id = _required_nonempty_string(row.get("task_id"), name=f"{name}.task_id")
    source_task_id = _required_nonempty_string(row.get("source_task_id"), name=f"{name}.source_task_id")
    source_failure_sample_index = _required_nonnegative_int(
        row.get("source_failure_sample_index"), name=f"{name}.source_failure_sample_index"
    )
    prefix_turn = _required_nonnegative_int(row.get("prefix_turn"), name=f"{name}.prefix_turn")
    expected_task_id = f"{source_task_id}::failure_s{source_failure_sample_index:02d}::turn_{prefix_turn:03d}"
    if task_id != expected_task_id:
        raise ValueError(f"{name}.task_id={task_id!r}; expected deterministic id {expected_task_id!r}")

    prefix_index = _required_nonnegative_int(row.get("prefix_index"), name=f"{name}.prefix_index")
    source_max_turns = _required_positive_int(row.get("source_max_turns"), name=f"{name}.source_max_turns")
    remaining_turns = _required_positive_int(row.get("remaining_turns"), name=f"{name}.remaining_turns")
    source_trajectory_assistant_turns = _required_positive_int(
        row.get("source_trajectory_assistant_turns"), name=f"{name}.source_trajectory_assistant_turns"
    )
    if expected_source_max_turns is not None:
        expected_source_max_turns = _required_positive_int(expected_source_max_turns, name="expected_source_max_turns")
        if source_max_turns != expected_source_max_turns:
            raise ValueError(
                f"{name}.source_max_turns={source_max_turns}; evaluator --max-turns={expected_source_max_turns}"
            )
    if remaining_turns != source_max_turns - prefix_turn:
        raise ValueError(
            f"{name}.remaining_turns={remaining_turns}; expected source_max_turns - prefix_turn = "
            f"{source_max_turns - prefix_turn}"
        )
    if prefix_turn >= source_trajectory_assistant_turns:
        raise ValueError(
            f"{name}.prefix_turn={prefix_turn} must precede one of the source trajectory's "
            f"{source_trajectory_assistant_turns} assistant turns"
        )

    prefix_kind = _required_nonempty_string(row.get("prefix_kind"), name=f"{name}.prefix_kind")
    if prefix_kind not in PREFIX_KINDS:
        raise ValueError(f"{name}.prefix_kind={prefix_kind!r}; expected one of {sorted(PREFIX_KINDS)}")
    question = _required_nonempty_string(row.get("question"), name=f"{name}.question")
    ground_truth = _required_nonempty_string(row.get("ground_truth"), name=f"{name}.ground_truth")
    messages = validate_replay_prefix_messages(
        row.get("prefix_messages"),
        question=question,
        prefix_turn=prefix_turn,
        name=name,
    )
    normalized = dict(row)
    normalized.update(
        {
            "schema_version": schema_version,
            "task_id": task_id,
            "source_task_id": source_task_id,
            "source_failure_sample_index": source_failure_sample_index,
            "prefix_index": prefix_index,
            "prefix_turn": prefix_turn,
            "source_max_turns": source_max_turns,
            "remaining_turns": remaining_turns,
            "source_trajectory_assistant_turns": source_trajectory_assistant_turns,
            "prefix_kind": prefix_kind,
            "question": question,
            "ground_truth": ground_truth,
            "prefix_messages": messages,
        }
    )
    for field in PREFIX_PROVENANCE_FIELDS:
        if field not in normalized:
            raise ValueError(f"{name} is missing required provenance field {field!r}")
    normalized["source_cohort_index"] = _required_nonnegative_int(
        normalized["source_cohort_index"], name=f"{name}.source_cohort_index"
    )
    normalized["source_failure_line_number"] = _required_positive_int(
        normalized["source_failure_line_number"], name=f"{name}.source_failure_line_number"
    )
    for field in ("k4_success_count", "k4_failure_count"):
        normalized[field] = _required_positive_int(normalized[field], name=f"{name}.{field}")
    if normalized["k4_success_count"] + normalized["k4_failure_count"] != 4:
        raise ValueError(
            f"{name} must describe a strict K=4 mixed group, got "
            f"success={normalized['k4_success_count']} failure={normalized['k4_failure_count']}"
        )
    return normalized


def prefix_provenance(row: Mapping[str, Any]) -> dict[str, Any]:
    """Extract the immutable prefix identity stored in every output attempt."""

    missing = [field for field in PREFIX_PROVENANCE_FIELDS if field not in row]
    if missing:
        raise ValueError(f"prefix row is missing provenance fields: {missing}")
    return {field: row[field] for field in PREFIX_PROVENANCE_FIELDS}


def assert_returned_messages_preserve_prefix(
    returned_messages: Any,
    prefix_messages: Sequence[Mapping[str, Any]],
    *,
    name: str,
) -> None:
    """Fail if an agent drops or mutates any replayed prefix message."""

    if not isinstance(returned_messages, list):
        raise TypeError(f"{name} returned messages must be a list")
    if len(returned_messages) <= len(prefix_messages):
        raise ValueError(
            f"{name} returned {len(returned_messages)} messages for a {len(prefix_messages)}-message prefix; "
            "at least one suffix message is required"
        )
    if returned_messages[: len(prefix_messages)] != list(prefix_messages):
        raise ValueError(f"{name} returned messages do not preserve the replay prefix byte-for-byte semantically")
