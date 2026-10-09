from typing import Any

import ray
import torch

from miles.utils.metric_utils import SWE_AGENT_METRIC_KEYS
from miles.utils.ray_utils import Box
from miles.utils.seqlen_balancing import get_seqlen_balanced_partitions
from miles.utils.types import Sample

_AGENT_TRAIN_METRIC_KEYS = {
    "agent_turns",
    "agent_finished",
    "agent_tool_call_count",
    "agent_tool_call_failure_rate",
    "agent_tool_unit_count",
    "agent_tool_unit_success_count",
    "agent_tool_unit_success_rate",
    "agent_search_count",
    "agent_search_query_count",
    "agent_search_query_repeat_count",
    "agent_search_query_repeat_rate",
    "agent_search_cache_hit_count",
    "agent_search_retry_429_count",
    "agent_search_non_200_attempt_count",
    "agent_fetch_url_count",
    "agent_fetch_url_repeat_count",
    "agent_fetch_url_repeat_rate",
    "agent_positive_repeated_search_excluded",
    "agent_positive_repeated_fetch_url_excluded",
    "agent_soft_overlong_applied",
    "agent_soft_overlong_base_score",
    "agent_soft_overlong_fraction",
    "agent_soft_overlong_penalty",
    "agent_soft_overlong_score",
    "agent_soft_overlong_tokens",
    "agent_context_reserve_hit",
    "agent_context_limit_hit",
    "agent_forced_final_answer",
    "agent_max_turns_hit",
    "agent_local_horizon_hit",
    "agent_forced_final_answer_skipped_no_budget",
    "agent_forced_final_answer_failed",
    "agent_final_answer_present",
    "agent_final_answer_tool_call_like",
    "agent_forced_final_answer_tool_call_like",
    "agent_excluded_from_training",
    "agent_single_call_length_penalized",
    "outcome_judge_failed",
    "process_advantage_valid_mask",
    "process_all_steps_correct",
    "process_boundary_reward_sum",
    "process_boxed_present",
    "process_correct_prefix_fraction",
    "process_correct_prefix_steps",
    "process_first_error_fraction",
    "process_first_error_index",
    "process_final_label",
    "process_judge_parse_failed",
    "process_judge_request_failed",
    "process_num_steps",
    "process_positive_return_fraction",
    "process_target_return_mean",
    "process_trajectory_split_failed",
    "process_trajectory_truncated",
}


def convert_samples_to_train_data(
    args,
    samples: list[Sample] | list[list[Sample]],
    metadata: dict[str, Any],
    custom_convert_samples_to_train_data_func,
    custom_reward_post_process_func,
):
    """
    Convert inference generated samples to training data.
    """
    if (f := custom_convert_samples_to_train_data_func) is not None:
        return f(args, samples)

    raw_rewards, rewards = _post_process_rewards(
        args, samples, custom_reward_post_process_func=custom_reward_post_process_func
    )

    assert len(raw_rewards) == len(samples)
    assert len(rewards) == len(samples)

    train_data = {
        "tokens": [sample.tokens for sample in samples],
        "response_lengths": [sample.response_length for sample in samples],
        # some reward model, e.g. remote rm, may return multiple rewards,
        # we could use key to select the reward.
        "rewards": rewards,
        "raw_reward": raw_rewards,
        "truncated": [1 if sample.status == Sample.Status.TRUNCATED else 0 for sample in samples],
        "sample_indices": [sample.index for sample in samples],
    }

    _apply_process_validity_masks(
        samples,
        require_two_valid_per_group=args.advantage_estimator == "process_rloo",
    )
    process_boundary_rewards = _extract_process_boundary_rewards(samples)
    if process_boundary_rewards is not None:
        train_data["process_boundary_rewards"] = process_boundary_rewards
        train_data["group_indices"] = [sample.group_index for sample in samples]

    # loss mask
    # TODO: compress the loss mask
    loss_masks = []
    for sample in samples:
        # always instantiate loss_mask if not provided
        if sample.loss_mask is None:
            sample.loss_mask = [1] * sample.response_length

        assert (
            len(sample.loss_mask) == sample.response_length
        ), f"loss mask length {len(sample.loss_mask)} != response length {sample.response_length}"
        if sample.remove_sample:
            sample.loss_mask = [0] * sample.response_length
        loss_masks.append(sample.loss_mask)
    train_data["loss_masks"] = loss_masks

    # overwriting the raw reward
    if samples[0].metadata and "raw_reward" in samples[0].metadata:
        train_data["raw_reward"] = [sample.metadata["raw_reward"] for sample in samples]

    train_data.update(_extract_numeric_agent_metrics(samples))

    # For rollout buffer
    if samples[0].metadata and "round_number" in samples[0].metadata:
        train_data["round_number"] = [sample.metadata["round_number"] for sample in samples]

    # Add rollout log probabilities for off-policy correction
    if samples[0].rollout_log_probs is not None:
        train_data["rollout_log_probs"] = [sample.rollout_log_probs for sample in samples]

    if samples[0].rollout_routed_experts is not None:
        train_data["rollout_routed_experts"] = [sample.rollout_routed_experts for sample in samples]

    if samples[0].rollout_indexer_topk is not None:
        train_data["rollout_indexer_topk"] = [sample.rollout_indexer_topk for sample in samples]

    if samples[0].train_metadata is not None:
        train_data["metadata"] = [sample.train_metadata for sample in samples]

    if any(sample.multimodal_train_inputs is not None for sample in samples):
        train_data["multimodal_train_inputs"] = [sample.multimodal_train_inputs for sample in samples]

    if any(sample.weight_versions for sample in samples):
        train_data["weight_versions"] = [sample.weight_versions for sample in samples]

    if samples[0].teacher_log_probs is not None:
        train_data["teacher_log_probs"] = [sample.teacher_log_probs for sample in samples]

    x = metadata.get("dynamic_global_batch_size")
    assert args.use_dynamic_global_batch_size == (x is not None)
    if x is not None:
        train_data["dynamic_global_batch_size"] = x

    return train_data


def _extract_process_boundary_rewards(samples: list[Sample]) -> list[list[float]] | None:
    """Extract an all-or-none per-response process-reward payload."""

    payloads = [
        sample.reward.get("process_boundary_rewards") if isinstance(sample.reward, dict) else None
        for sample in samples
    ]
    if not any(payload is not None for payload in payloads):
        return None
    if not all(payload is not None for payload in payloads):
        raise ValueError("process_boundary_rewards must be present for every training sample or none")

    normalized: list[list[float]] = []
    for index, (sample, payload) in enumerate(zip(samples, payloads, strict=True)):
        if not isinstance(payload, (list, tuple)):
            raise TypeError(f"sample {index} process_boundary_rewards must be a list or tuple")
        if len(payload) != sample.response_length:
            raise ValueError(
                f"sample {index} process reward length {len(payload)} != response length {sample.response_length}"
            )
        values: list[float] = []
        for token_index, value in enumerate(payload):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"sample {index} process_boundary_rewards[{token_index}] must be a finite number")
            normalized_value = float(value)
            if not torch.isfinite(torch.tensor(normalized_value)):
                raise ValueError(f"sample {index} process_boundary_rewards[{token_index}] must be finite")
            values.append(normalized_value)
        normalized.append([0.0] * sample.response_length if sample.remove_sample else values)
    return normalized


def _apply_process_validity_masks(samples: list[Sample], *, require_two_valid_per_group: bool = True) -> None:
    """Mask invalid rows and, for RLOO, groups with fewer than two valid rows."""

    process_rows = [
        isinstance(sample.reward, dict) and sample.reward.get("process_boundary_rewards") is not None
        for sample in samples
    ]
    if not any(process_rows):
        return
    if not all(process_rows):
        raise ValueError("process reward payload must be present for every training sample or none")

    groups: dict[int, list[Sample]] = {}
    for index, sample in enumerate(samples):
        if isinstance(sample.group_index, bool) or not isinstance(sample.group_index, int):
            raise TypeError(f"process sample {index} group_index must be an integer")
        assert isinstance(sample.reward, dict)
        valid = sample.reward.get("advantage_valid_mask")
        if valid not in (0, 0.0, 1, 1.0, False, True):
            raise ValueError(f"process sample {index} advantage_valid_mask must be 0 or 1")
        if not bool(valid):
            sample.remove_sample = True
        groups.setdefault(sample.group_index, []).append(sample)

    if not require_two_valid_per_group:
        return

    for group_samples in groups.values():
        if sum(not sample.remove_sample for sample in group_samples) >= 2:
            continue
        for sample in group_samples:
            sample.remove_sample = True
            sample.metadata["process_advantage_valid_mask"] = 0.0


def _extract_numeric_agent_metrics(samples: list[Sample]) -> dict[str, list[float]]:
    metric_keys = sorted(
        {
            key
            for sample in samples
            for key, value in sample.metadata.items()
            if key in (_AGENT_TRAIN_METRIC_KEYS | SWE_AGENT_METRIC_KEYS) and isinstance(value, (int, float, bool))
        }
    )
    metrics = {
        key: [float(value) if (value := sample.metadata.get(key)) is not None else 0.0 for sample in samples]
        for key in metric_keys
    }
    if "agent_tool_unit_count" in metrics and "agent_tool_unit_success_count" not in metrics:
        rates = metrics.get("agent_tool_unit_success_rate", [1.0] * len(samples))
        metrics["agent_tool_unit_success_count"] = [
            count * rate for count, rate in zip(metrics["agent_tool_unit_count"], rates, strict=True)
        ]
    return metrics


def _post_process_rewards(args, samples: list[Sample] | list[list[Sample]], custom_reward_post_process_func):
    if (f := custom_reward_post_process_func) is not None:
        return f(args, samples)

    raw_rewards = [sample.get_reward_value(args) for sample in samples]
    if args.advantage_estimator in ["grpo", "gspo", "reinforce_plus_plus_baseline"] and args.rewards_normalization:
        # group norm
        rewards = torch.tensor(raw_rewards, dtype=torch.float)
        normalization_mask = torch.tensor(
            [_include_sample_in_reward_normalization(sample) for sample in samples],
            dtype=torch.bool,
        )
        local_only_std = bool(getattr(args, "grpo_std_normalization_local_only", False))
        if local_only_std and not args.grpo_std_normalization:
            raise ValueError("local-only GRPO std normalization requires grpo_std_normalization=True")
        local_flags = None
        if local_only_std:
            local_flags = torch.tensor([_is_adaptive_local_sample(sample) for sample in samples], dtype=torch.bool)
        if rewards.shape[-1] == args.n_samples_per_prompt * args.rollout_batch_size:
            rewards = rewards.reshape(-1, args.n_samples_per_prompt)
            normalization_mask = normalization_mask.reshape(-1, args.n_samples_per_prompt)
            if local_flags is not None:
                local_flags = local_flags.reshape(-1, args.n_samples_per_prompt)
        else:
            # when samples count are not equal in each group
            rewards = rewards.view(-1, rewards.shape[-1])
            normalization_mask = normalization_mask.view(-1, normalization_mask.shape[-1])
            if local_flags is not None:
                local_flags = local_flags.view(-1, local_flags.shape[-1])
        mask = normalization_mask.to(dtype=rewards.dtype)
        counts = mask.sum(dim=-1, keepdim=True)
        mean = (rewards * mask).sum(dim=-1, keepdim=True) / counts.clamp_min(1.0)
        rewards = torch.where(normalization_mask, rewards - mean, torch.zeros_like(rewards))

        if args.advantage_estimator in ["grpo", "gspo"] and args.grpo_std_normalization:
            variance = (rewards.square() * mask).sum(dim=-1, keepdim=True) / (counts - 1.0).clamp_min(1.0)
            std = variance.sqrt()
            if local_flags is not None:
                homogeneous = local_flags.all(dim=-1) | (~local_flags).all(dim=-1)
                if not bool(homogeneous.all().item()):
                    mixed_groups = (~homogeneous).nonzero(as_tuple=False).flatten().tolist()
                    raise ValueError(f"local-only GRPO std normalization found mixed groups at rows {mixed_groups}")
                normalize_group = local_flags.all(dim=-1, keepdim=True)
                denominator = torch.where(normalize_group, std + 1e-6, torch.ones_like(std))
            else:
                denominator = std + 1e-6
            rewards = torch.where(normalization_mask, rewards / denominator, torch.zeros_like(rewards))

        return raw_rewards, rewards.flatten().tolist()

    return raw_rewards, raw_rewards


def _is_adaptive_local_sample(sample: Sample) -> bool:
    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    return bool(metadata.get("ab_local_rollout")) or metadata.get("ab_rollout_kind") == "local"


def _include_sample_in_reward_normalization(sample: Sample) -> bool:
    """Keep masked or failed samples out of every GRPO baseline."""
    reward = sample.reward
    state = reward.get("process_state") if isinstance(reward, dict) else None
    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    if bool(sample.remove_sample) or bool(metadata.get("outcome_judge_failed")):
        return False
    if _is_adaptive_local_sample(sample):
        return True
    is_process = state is not None
    if not is_process:
        return True
    if state in {"uncommitted", "unresolved", "judge_error"}:
        return False
    return True


def split_train_data_by_dp(args, data, dp_size):
    """Split the train data by data parallel size."""
    rollout_data = {}

    if "prompt" in data:
        rollout_data["prompt"] = data["prompt"]

    total_lengths = [len(t) for t in data["tokens"]]
    data["total_lengths"] = total_lengths

    if args.advantage_estimator == "process_rloo":
        if "group_indices" not in data:
            raise ValueError("process_rloo requires group_indices")
        partitions = _get_group_preserving_partitions(total_lengths, data["group_indices"], dp_size)
    elif args.balance_data:
        partitions = get_seqlen_balanced_partitions(total_lengths, dp_size, equal_size=True)
    else:
        partitions = [range(i, len(total_lengths), dp_size) for i in range(dp_size)]

    rollout_data_refs = []

    for i in range(dp_size):
        rollout_data = {}
        partition = partitions[i]
        rollout_data["partition"] = partition
        split_keys = [
            "tokens",
            "multimodal_train_inputs",
            "response_lengths",
            "rewards",
            "process_boundary_rewards",
            "group_indices",
            "truncated",
            "loss_masks",
            "round_number",
            "sample_indices",
            "rollout_log_probs",
            "rollout_routed_experts",
            "rollout_indexer_topk",
            "prompt",
            "teacher_log_probs",
            "weight_versions",
        ]
        split_keys.extend(sorted(key for key in data if key in _AGENT_TRAIN_METRIC_KEYS))
        for key in split_keys:
            if key not in data:
                continue
            val = [data[key][j] for j in partition]
            rollout_data[key] = val
        # keys that need to be splited at train side
        for key in [
            "raw_reward",
            "total_lengths",
            "dynamic_global_batch_size",
        ]:
            if key not in data:
                continue
            rollout_data[key] = data[key]
        rollout_data_refs.append(Box(ray.put(rollout_data)))
    return rollout_data_refs


def _get_group_preserving_partitions(
    total_lengths: list[int], group_indices: list[int], dp_size: int
) -> list[list[int]]:
    """Balance complete prompt groups across DP ranks without splitting them."""

    if len(total_lengths) != len(group_indices):
        raise ValueError("total_lengths and group_indices must have the same length")
    if dp_size <= 0:
        raise ValueError("dp_size must be positive")

    groups: dict[int, list[int]] = {}
    for sample_index, group_index in enumerate(group_indices):
        if isinstance(group_index, bool) or not isinstance(group_index, int):
            raise TypeError(f"group_indices[{sample_index}] must be an integer")
        groups.setdefault(group_index, []).append(sample_index)
    if len(groups) % dp_size != 0:
        raise ValueError(
            f"group-preserving partition requires the number of prompt groups ({len(groups)}) "
            f"to be divisible by dp_size ({dp_size})"
        )

    group_sizes = {len(rows) for rows in groups.values()}
    if len(group_sizes) != 1:
        raise ValueError(f"group-preserving partition requires equal prompt-group sizes, got {sorted(group_sizes)}")

    groups_per_rank = len(groups) // dp_size
    partitions: list[list[int]] = [[] for _ in range(dp_size)]
    assigned_group_counts = [0] * dp_size
    assigned_token_counts = [0] * dp_size
    ranked_groups = sorted(
        groups.values(),
        key=lambda rows: sum(total_lengths[index] for index in rows),
        reverse=True,
    )
    for rows in ranked_groups:
        eligible_ranks = [rank for rank in range(dp_size) if assigned_group_counts[rank] < groups_per_rank]
        rank = min(eligible_ranks, key=lambda value: (assigned_token_counts[value], value))
        partitions[rank].extend(rows)
        assigned_group_counts[rank] += 1
        assigned_token_counts[rank] += sum(total_lengths[index] for index in rows)
    return [sorted(partition) for partition in partitions]
