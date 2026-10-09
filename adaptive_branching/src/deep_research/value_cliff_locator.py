"""Locate the largest value-drop action in one failed trajectory."""

from __future__ import annotations

import copy
import os
from dataclasses import dataclass
from typing import Any

from adaptive_branching.src.deep_research.judge_client import JudgeClient
from adaptive_branching.src.deep_research.value_cliff_rubric import (
    DEFAULT_LOCAL_HORIZON_ASSISTANT_TURNS,
    VALUE_CLIFF_LOCATOR_REQUIRED_KEYS,
    VALUE_CLIFF_LOCATOR_VERSION,
    validate_value_cliff_locator_verdict,
    value_cliff_locator_system,
)


@dataclass(frozen=True)
class ValueCliffLocalizationResult:
    """Validated value-cliff locator result used by online and offline paths."""

    event: dict[str, Any]
    diagnostics: dict[str, Any]


async def locate_value_cliff(
    client: JudgeClient,
    *,
    question: str,
    ground_truth: str,
    positive_messages: list[dict[str, Any]],
    negative_messages: list[dict[str, Any]],
    tag: str,
    trace_max_chars: int,
) -> ValueCliffLocalizationResult:
    """Select exactly one non-terminal failed-trajectory action."""

    if not callable(getattr(client, "complete_json", None)):
        raise TypeError("client must provide an async complete_json method")
    if not isinstance(tag, str) or not tag.strip():
        raise ValueError("tag must be a non-empty string")
    prompt, metadata = build_value_cliff_locator_prompt(
        question=question,
        ground_truth=ground_truth,
        failed_messages=negative_messages,
        successful_messages=positive_messages,
        trace_max_chars=trace_max_chars,
    )
    horizon = value_cliff_local_horizon()
    verdict = await client.complete_json(
        value_cliff_locator_system(horizon),
        prompt,
        required_keys=VALUE_CLIFF_LOCATOR_REQUIRED_KEYS,
        tag=tag.strip(),
        validate=lambda raw: validate_value_cliff_locator_verdict(raw, assistant_turns=metadata["failed_assistant_turns"]),
    )
    event = {
        "event_type": "value_cliff",
        "event_turn": verdict["selected_turn"],
        "event_summary": verdict["value_drop_reason"],
        "value_drop_reason": verdict["value_drop_reason"],
        "recovery_rubric": copy.deepcopy(verdict["recovery_rubric"]),
    }
    return ValueCliffLocalizationResult(
        event=event,
        diagnostics={"policy": VALUE_CLIFF_LOCATOR_VERSION, "stage_status": "selected"},
    )


def build_value_cliff_locator_prompt(
    *,
    question: str,
    ground_truth: str,
    failed_messages: list[dict[str, Any]],
    successful_messages: list[dict[str, Any]],
    trace_max_chars: int,
) -> tuple[str, dict[str, Any]]:
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be a non-empty string")
    if not isinstance(ground_truth, str) or not ground_truth.strip():
        raise ValueError("ground_truth must be a non-empty string")
    if isinstance(trace_max_chars, bool) or not isinstance(trace_max_chars, int) or trace_max_chars <= 0:
        raise ValueError("trace_max_chars must be a positive integer")
    failed_turns = assistant_turn_count(failed_messages)
    successful_turns = assistant_turn_count(successful_messages)
    if failed_turns < 2:
        raise ValueError("failed trajectory must contain at least two assistant turns")
    if successful_turns < 1:
        raise ValueError("successful trajectory must contain at least one assistant turn")

    failed, successful, failed_truncated, successful_truncated = render_trace_pair(
        failed_messages,
        successful_messages,
        first_heading="Failed Turn",
        second_heading="Successful Turn",
        max_chars=trace_max_chars,
    )
    prompt = (
        f"## Original question\n{question.strip()}\n\n"
        f"## Candidate range\nSelect exactly one Failed Turn in [1, {failed_turns - 1}].\n\n"
        "## Failed trajectory\n"
        f"[failed_trace_truncated={str(failed_truncated).lower()}]\n{failed}\n\n"
        f"## Reference answer\n{ground_truth.strip()}\n\n"
        "## Matched successful trajectory\n"
        f"[successful_trace_truncated={str(successful_truncated).lower()}]\n{successful}"
    )
    return prompt, {
        "failed_assistant_turns": failed_turns,
        "successful_assistant_turns": successful_turns,
        "failed_trace_truncated": failed_truncated,
        "successful_trace_truncated": successful_truncated,
    }


def build_fixed_turn_rubric_prompt(
    *,
    question: str,
    ground_truth: str,
    failed_messages: list[dict[str, Any]],
    successful_messages: list[dict[str, Any]],
    selected_turn: int,
    trace_max_chars: int,
) -> tuple[str, dict[str, Any]]:
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be a non-empty string")
    if not isinstance(ground_truth, str) or not ground_truth.strip():
        raise ValueError("ground_truth must be a non-empty string")
    if isinstance(trace_max_chars, bool) or not isinstance(trace_max_chars, int) or trace_max_chars <= 0:
        raise ValueError("trace_max_chars must be a positive integer")
    failed_turns = assistant_turn_count(failed_messages)
    successful_turns = assistant_turn_count(successful_messages)
    if failed_turns < 2:
        raise ValueError("failed trajectory must contain at least two assistant turns")
    if successful_turns < 1:
        raise ValueError("successful trajectory must contain at least one assistant turn")
    if isinstance(selected_turn, bool) or not isinstance(selected_turn, int):
        raise TypeError("selected_turn must be an integer")
    if not 1 <= selected_turn < failed_turns:
        raise ValueError(f"selected_turn must be in [1, {failed_turns - 1}], got {selected_turn!r}")

    failed, successful, failed_truncated, successful_truncated = render_trace_pair(
        failed_messages,
        successful_messages,
        first_heading="Failed Turn",
        second_heading="Successful Turn",
        max_chars=trace_max_chars,
    )
    prompt = (
        f"## Original question\n{question.strip()}\n\n"
        f"## Externally fixed value-drop action\nFailed Turn {selected_turn}\n"
        "Do not choose or move this turn. Explain its causal error and write only its recovery rubric.\n\n"
        "## Failed trajectory\n"
        f"[failed_trace_truncated={str(failed_truncated).lower()}]\n{failed}\n\n"
        f"## Reference answer\n{ground_truth.strip()}\n\n"
        "## Matched successful trajectory\n"
        f"[successful_trace_truncated={str(successful_truncated).lower()}]\n{successful}"
    )
    return prompt, {
        "selected_turn": selected_turn,
        "failed_assistant_turns": failed_turns,
        "successful_assistant_turns": successful_turns,
        "failed_trace_truncated": failed_truncated,
        "successful_trace_truncated": successful_truncated,
    }


def locator_policy_version() -> str:
    return VALUE_CLIFF_LOCATOR_VERSION


def value_cliff_local_horizon() -> int:
    raw = os.getenv("AB_EVENT_HIDDEN_MAX_TURNS", str(DEFAULT_LOCAL_HORIZON_ASSISTANT_TURNS)).strip()
    if not raw or any(character not in "0123456789" for character in raw):
        raise ValueError(f"AB_EVENT_HIDDEN_MAX_TURNS must be a positive integer, got {raw!r}")
    value = int(raw)
    if value <= 0:
        raise ValueError(f"AB_EVENT_HIDDEN_MAX_TURNS must be a positive integer, got {raw!r}")
    return value


def build_value_cliff_prefixes(messages: list[dict[str, Any]], selected_turn: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return prefixes immediately before and after the selected action."""

    if not isinstance(messages, list) or not messages:
        raise ValueError("messages must be a non-empty list")
    if any(not isinstance(message, dict) for message in messages):
        raise TypeError("messages must contain only objects")
    assistant_indices = [index for index, message in enumerate(messages) if message.get("role") == "assistant"]
    if isinstance(selected_turn, bool) or not isinstance(selected_turn, int):
        raise ValueError("selected_turn must be an integer")
    if not 1 <= selected_turn < len(assistant_indices):
        raise ValueError(f"selected_turn must be within [1, {max(0, len(assistant_indices) - 1)}]")
    selected_index = assistant_indices[selected_turn - 1]
    next_assistant_index = assistant_indices[selected_turn]
    before = copy.deepcopy(messages[:selected_index])
    after = copy.deepcopy(messages[:next_assistant_index])
    if not before:
        raise ValueError("value-cliff prefix before the selected action is empty")
    return before, after


def assistant_turn_count(messages: list[dict[str, Any]]) -> int:
    if not isinstance(messages, list) or any(not isinstance(message, dict) for message in messages):
        raise TypeError("messages must be a list of objects")
    return sum(message.get("role") == "assistant" for message in messages)


def render_trace_pair(
    first_messages: list[dict[str, Any]],
    second_messages: list[dict[str, Any]],
    *,
    first_heading: str,
    second_heading: str,
    max_chars: int,
) -> tuple[str, str, bool, bool]:
    if isinstance(max_chars, bool) or not isinstance(max_chars, int) or max_chars <= 0:
        raise ValueError("max_chars must be a positive integer")
    first_full = render_message_turns(first_messages, heading=first_heading)
    second_full = render_message_turns(second_messages, heading=second_heading)
    first_budget, second_budget = balanced_trace_budgets(len(first_full), len(second_full), max_chars)
    first = render_message_turns(first_messages, heading=first_heading, max_chars=max(80, first_budget))
    second = render_message_turns(second_messages, heading=second_heading, max_chars=max(80, second_budget))
    return first, second, len(first_full) > first_budget, len(second_full) > second_budget


def balanced_trace_budgets(first_chars: int, second_chars: int, max_chars: int) -> tuple[int, int]:
    for name, value in (("first_chars", first_chars), ("second_chars", second_chars), ("max_chars", max_chars)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a nonnegative integer")
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")
    if first_chars + second_chars <= max_chars:
        return first_chars, second_chars
    half = max_chars // 2
    if first_chars <= half:
        return first_chars, max_chars - first_chars
    if second_chars <= half:
        return max_chars - second_chars, second_chars
    return half, max_chars - half


def render_message_turns(messages: list[dict[str, Any]], *, heading: str, max_chars: int | None = None) -> str:
    if not isinstance(messages, list) or any(not isinstance(message, dict) for message in messages):
        raise TypeError("messages must be a list of objects")
    if not isinstance(heading, str) or not heading.strip():
        raise ValueError("heading must be a non-empty string")
    if max_chars is not None and (isinstance(max_chars, bool) or not isinstance(max_chars, int) or max_chars <= 0):
        raise ValueError("max_chars must be a positive integer or None")
    blocks: list[str] = []
    current: list[str] | None = None
    turn = 0
    for message in messages:
        role = str(message.get("role") or "")
        if role == "assistant":
            if current is not None:
                blocks.append("\n".join(current))
            turn += 1
            current = [f"### {heading.strip()} {turn}"]
            reasoning = str(message.get("reasoning_content") or "").strip()
            content = str(message.get("content") or "").strip()
            if reasoning:
                current.append(f"[reasoning] {reasoning}")
            if content:
                current.append(f"[assistant] {content}")
            for tool_call in message.get("tool_calls") or []:
                function = (tool_call or {}).get("function") or {}
                current.append(f"[tool_call:{str(function.get('name') or 'tool')}] {str(function.get('arguments') or '')}")
            continue
        if current is None:
            continue
        content = str(message.get("content") or "").strip()
        if role == "tool":
            current.append(f"[following_observation:{str(message.get('name') or 'tool')}] {content}")
        elif role:
            current.append(f"[{role}] {content}")
    if current is not None:
        blocks.append("\n".join(current))
    if not blocks:
        return "(no assistant turns)"
    rendered = "\n\n".join(blocks)
    if max_chars is None or len(rendered) <= max_chars:
        return rendered
    separator_chars = 2 * max(0, len(blocks) - 1)
    block_budget = max(80, (max_chars - separator_chars) // len(blocks))
    return "\n\n".join(_truncate_middle(block, block_budget) for block in blocks)


def _truncate_middle(text: str, limit: int) -> str:
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
        raise ValueError("limit must be a positive integer")
    if len(text) <= limit:
        return text
    marker = f"\n... [{len(text) - limit} chars omitted from this turn] ...\n"
    if len(marker) >= limit:
        return text[:limit]
    remaining = limit - len(marker)
    return text[: (remaining + 1) // 2] + marker + text[-(remaining // 2) :]
