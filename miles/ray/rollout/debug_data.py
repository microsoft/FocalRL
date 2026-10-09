import json
import logging
import os
from pathlib import Path
from typing import Any

import torch

from miles.utils.types import Sample

logger = logging.getLogger(__name__)

_HEAVY_METADATA_KEYS = {
    "ab_branch_history",
    "ab_branch_entropies",
    "ab_branch_spec",
    "ab_event",
    "ab_event_selection_info",
    "ab_event_rubric_judge",
    "ab_event_rubric_judge_error",
    "messages",
}
_SENSITIVE_METADATA_PREFIXES = ("agent_chat_",)
_JSONL_FILES_INITIALIZED: set[str] = set()


def load_debug_rollout_data(args, rollout_id: int):
    data = torch.load(
        args.load_debug_rollout_data.format(rollout_id=rollout_id),
        weights_only=False,
    )["samples"]
    data = [Sample.from_dict(sample) for sample in data]
    if (ratio := args.load_debug_rollout_data_subsample) is not None:
        original_num_rows = len(data)
        rough_subsample_num_rows = int(original_num_rows * ratio)
        data = data[: rough_subsample_num_rows // 2] + data[-rough_subsample_num_rows // 2 :]
        logger.info(
            "Subsample loaded debug rollout data using %s and change num rows %s -> %s",
            f"{ratio=}",
            original_num_rows,
            len(data),
        )
    return data


def save_debug_rollout_data(args, data, rollout_id, evaluation: bool):
    # TODO to be refactored (originally Buffer._set_data)
    dump_rollout_messages_jsonl(args, data, rollout_id=rollout_id, evaluation=evaluation)
    strip_rollout_heavy_metadata(data)

    if (path_template := args.save_debug_rollout_data) is not None:
        path = Path(path_template.format(rollout_id=("eval_" if evaluation else "") + str(rollout_id)))
        logger.info(f"Save debug rollout data to {path}")
        path.parent.mkdir(parents=True, exist_ok=True)

        # TODO may improve the format
        if evaluation:
            dump_data = dict(
                samples=[sample.to_dict() for dataset_name, info in data.items() for sample in info["samples"]]
            )
        else:
            dump_data = dict(
                samples=[sample.to_dict() for sample in data],
            )

        torch.save(dict(rollout_id=rollout_id, **dump_data), path)


def dump_rollout_messages_jsonl(args, data, rollout_id, evaluation: bool = False) -> None:
    path = rollout_messages_jsonl_path(args, rollout_id=rollout_id, evaluation=evaluation)
    if path is None:
        return

    rows = [_debug_jsonl_row(rollout_id, sample, evaluation=evaluation) for sample in _iter_samples(data)]
    rows = [row for row in rows if row.get("messages") is not None]
    if not rows:
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    path_key = str(path)
    if path_key not in _JSONL_FILES_INITIALIZED:
        if path.exists():
            path.unlink()
        _JSONL_FILES_INITIALIZED.add(path_key)
    with path.open("a", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def strip_rollout_heavy_metadata(data) -> None:
    for sample in _iter_samples(data):
        metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
        for key in _HEAVY_METADATA_KEYS:
            metadata.pop(key, None)
        for key in list(metadata):
            if isinstance(key, str) and key.startswith(_SENSITIVE_METADATA_PREFIXES):
                metadata.pop(key, None)


def rollout_messages_jsonl_path(args, rollout_id, evaluation: bool = False) -> Path | None:
    rollout_id_text = ("eval_" if evaluation else "") + str(rollout_id)
    if getattr(args, "save_debug_rollout_data", None):
        try:
            pt_path = Path(str(args.save_debug_rollout_data).format(rollout_id=rollout_id_text))
        except Exception:  # noqa: BLE001
            pt_path = Path(str(args.save_debug_rollout_data))
        return pt_path.parent / "trajectory_jsonl" / f"rollout_{rollout_id_text}.jsonl"
    if os.getenv("DEBUG_ROLLOUT_DIR"):
        return Path(os.environ["DEBUG_ROLLOUT_DIR"]) / "trajectory_jsonl" / f"rollout_{rollout_id_text}.jsonl"
    return None


def _iter_samples(data):
    if isinstance(data, dict):
        for info in data.values():
            if isinstance(info, dict):
                yield from info.get("samples", [])
        return
    for item in data or []:
        if isinstance(item, list):
            yield from item
        else:
            yield item


def _debug_jsonl_row(rollout_id, sample: Sample, evaluation: bool = False) -> dict[str, Any]:
    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    reward = sample.reward
    reward_dict = reward if isinstance(reward, dict) else {}
    kind = metadata.get("ab_rollout_kind") or ("local" if metadata.get("ab_local_rollout") else "full")
    branch_spec = metadata.get("ab_branch_spec") if isinstance(metadata.get("ab_branch_spec"), dict) else {}
    row = {
        "rollout_id": rollout_id,
        "evaluation": evaluation,
        "kind": kind,
        "group_index": sample.group_index,
        "sample_index": sample.index,
        "parent_group_index": metadata.get("ab_parent_group_index"),
        "parent_sample_index": metadata.get("ab_parent_sample_index"),
        "original_question": metadata.get("ab_original_question") or _original_question_from_prompt(sample.prompt),
        "label": sample.label,
        "response": sample.response,
        "reward": reward,
        "acc": reward_dict.get("acc"),
        "branch_type": branch_spec.get("branch_type"),
        "local_reward_mode": metadata.get("ab_local_reward_mode") or branch_spec.get("reward_mode"),
        "event": metadata.get("ab_event") or branch_spec.get("event"),
        "event_turn": metadata.get("ab_event_turn") or branch_spec.get("event_turn"),
        "local_prefix_assistant_turns": metadata.get("ab_local_prefix_assistant_turns"),
        "generated_turns": metadata.get("agent_turns"),
        "recovery_rubric": metadata.get("ab_event_recovery_rubric")
        or branch_spec.get("recovery_rubric"),
        "value_drop_reason": metadata.get("ab_event_value_drop_reason")
        or branch_spec.get("value_drop_reason"),
        "event_locator_version": metadata.get("ab_event_locator_version")
        or branch_spec.get("locator_version"),
        "event_selection_info": metadata.get("ab_event_selection_info"),
        "branch_selection": metadata.get("ab_branch_selection"),
        "branch_id": metadata.get("ab_branch_id") or branch_spec.get("branch_id"),
        "branch_entropies": metadata.get("ab_branch_entropies"),
        "branch_replay_history": metadata.get("ab_branch_history"),
        "branch_generation_cost": metadata.get("ab_generation_cost"),
        "event_rubric_judge": metadata.get("ab_event_rubric_judge"),
        "event_rubric_judge_version": metadata.get("ab_event_rubric_judge_version"),
        "event_rubric_judge_error": metadata.get("ab_event_rubric_judge_error"),
        "local_stop_reason": metadata.get("agent_last_finish_reason"),
        "local_horizon_hit": metadata.get("agent_local_horizon_hit"),
        "messages": metadata.get("messages"),
    }
    return row


def _original_question_from_prompt(prompt: Any) -> str:
    if isinstance(prompt, list):
        for message in prompt:
            if isinstance(message, dict) and message.get("role") == "user":
                return str(message.get("content") or "")
    return str(prompt)
