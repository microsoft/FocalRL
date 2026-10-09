"""Select turn-capped keep5 trajectories without consulting their judge score."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from collections.abc import Callable

from adaptive_branching.src.deep_research.value_cliff_prefixes import validate_replay_prefix_messages


def load_continuation_samples(
    path: Path,
    *,
    model: str,
    total_turns: int,
    max_seq_len: int,
    keep_tool_results: int,
    final_prompt: Callable[[str], str],
) -> list[dict]:
    """Read a stable snapshot; reject duplicate trials and malformed continuation boundaries."""
    if not path.is_file():
        raise FileNotFoundError(path)
    for name, value in (
        ("total_turns", total_turns),
        ("max_seq_len", max_seq_len),
        ("keep_tool_results", keep_tool_results),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if not isinstance(model, str) or not model.strip() or not callable(final_prompt):
        raise ValueError("model and final_prompt are required")
    samples, seen = [], set()
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            label = f"{path}:{line_number}"
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"{label}: expected an object")
            task_id = record.get("task_id")
            if not isinstance(task_id, str) or not task_id or task_id in seen:
                raise ValueError(f"{label}: missing or duplicate task_id={task_id!r}")
            seen.add(task_id)
            if record.get("model") != model or record.get("history_mode") != "keep5":
                raise ValueError(f"{label}: model/history_mode mismatch")
            if record.get("samples_per_task", 1) != 1 or record.get("attempt", 1) != 1:
                raise ValueError(f"{label}: only single-attempt outputs are supported")
            metrics = record.get("metrics")
            if not isinstance(metrics, dict):
                raise ValueError(f"{label}: missing metrics")
            if not metrics.get("agent_max_turns_hit"):
                continue
            if metrics.get("agent_context_reserve_hit"):
                raise ValueError(f"{label}: contradictory max-turn/context-reserve flags")
            turns = metrics.get("agent_turns")
            if isinstance(turns, bool) or not isinstance(turns, int) or not 0 < turns < total_turns:
                raise ValueError(f"{label}: source turns={turns!r} must be below total_turns={total_turns}")
            if (
                metrics.get("agent_max_seq_len") != max_seq_len
                or metrics.get("agent_keep_tool_results") != keep_tool_results
            ):
                raise ValueError(f"{label}: context length or keep_tool_results differs")
            question, answer = record.get("question"), record.get("ground_truth")
            if not isinstance(question, str) or not question.strip() or not isinstance(answer, str):
                raise ValueError(f"{label}: invalid question/ground_truth")
            messages = record.get("messages")
            if not isinstance(messages, list) or len(messages) < 4:
                raise ValueError(f"{label}: missing messages")
            if not all(isinstance(message, dict) for message in messages):
                raise ValueError(f"{label}: message must be an object")
            if (
                not metrics.get("agent_forced_final_answer")
                or metrics.get("agent_forced_final_answer_reason") != "max_turns"
                or messages[-2] != {"role": "user", "content": final_prompt(question)}
                or messages[-1].get("role") != "assistant"
            ):
                raise ValueError(f"{label}: cannot identify the forced-final prompt/answer pair")
            prefix = messages[:-2]
            validate_replay_prefix_messages(prefix, question=question, prefix_turn=turns, name=label)
            initial_tokens = metrics.get("agent_pre_forced_session_tokens", 0)
            if isinstance(initial_tokens, bool) or not isinstance(initial_tokens, int) or initial_tokens < 0:
                raise ValueError(f"{label}: invalid pre-forced token count")
            samples.append(
                {
                    "task_id": task_id,
                    "question": question,
                    "ground_truth": answer,
                    "continuation_messages": prefix,
                    "source_metrics": metrics,
                    "initial_session_tokens": initial_tokens,
                    "continuation": {
                        "source_path": str(path.resolve()),
                        "source_line": line_number,
                        "source_record_sha256": hashlib.sha256(line.encode()).hexdigest(),
                        "prefix_turns": turns,
                        "total_max_turns": total_turns,
                        "remaining_turns": total_turns - turns,
                        "initial_token_count_known": "agent_pre_forced_session_tokens" in metrics,
                        "metrics_scope": "agent_turns is total; tool counters and elapsed_seconds are suffix-only",
                    },
                }
            )
    if not samples:
        raise ValueError(f"{path}: no max-turn trajectories eligible for continuation")
    return samples
