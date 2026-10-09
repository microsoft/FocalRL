import math
from argparse import Namespace
from pathlib import Path

import torch
import torch.distributed as dist

from miles.backends.training_utils.cp_utils import slice_log_prob_with_cp
from miles.utils.types import RolloutBatch

_POLICY_LOSS_DUMP_COUNTER = 0


def maybe_dump_policy_loss_debug(
    *,
    args: Namespace,
    batch: RolloutBatch,
    train_log_probs: list[torch.Tensor],
    old_log_probs: list[torch.Tensor],
    rollout_log_probs: list[torch.Tensor] | None,
    advantages: list[torch.Tensor],
    local_loss_masks: list[torch.Tensor],
    ppo_kl: torch.Tensor,
    pg_loss: torch.Tensor,
) -> None:
    clip_dump_dir = getattr(args, "dump_clip_tokens", None)
    details_dump_dir = getattr(args, "dump_details", None)
    if clip_dump_dir is None and details_dump_dir is None:
        return

    global _POLICY_LOSS_DUMP_COUNTER
    counter = _POLICY_LOSS_DUMP_COUNTER
    _POLICY_LOSS_DUMP_COUNTER += 1

    dump_every = int(getattr(args, "dump_clip_tokens_every", 1))
    if dump_every <= 0:
        raise ValueError("--dump-clip-tokens-every must be positive")
    if clip_dump_dir is not None and counter % dump_every != 0:
        return

    rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
    if clip_dump_dir is not None:
        path = Path(clip_dump_dir) / f"rank_{rank}_call_{counter}.pt"
    else:
        path = Path(details_dump_dir) / "policy_loss_debug" / f"rank_{rank}_call_{counter}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)

    def to_cpu_float(tensor: torch.Tensor) -> torch.Tensor:
        return tensor.detach().float().cpu()

    def to_cpu(tensor: torch.Tensor) -> torch.Tensor:
        return tensor.detach().cpu()

    eps_clip = float(args.eps_clip)
    eps_clip_high = float(args.eps_clip_high if args.eps_clip_high is not None else args.eps_clip)
    lower_ratio = 1.0 - eps_clip
    upper_ratio = 1.0 + eps_clip_high
    if lower_ratio <= 0:
        raise ValueError("--eps-clip must be smaller than 1 for PPO token clipping dumps")
    lower_log_ratio = math.log(lower_ratio)
    upper_log_ratio = math.log(upper_ratio)

    split_sizes = [train_lp.numel() for train_lp in train_log_probs]
    ppo_kl_by_sample = ppo_kl.reshape(-1).split(split_sizes)
    pg_loss_by_sample = pg_loss.reshape(-1).split(split_sizes)
    max_seq_lens = batch.get("max_seq_lens")
    sample_indices = batch.get("sample_indices")

    samples = []
    lower_clipped_tokens = 0
    upper_clipped_tokens = 0
    active_token_count = 0
    for index, train_lp in enumerate(train_log_probs):
        total_length = int(batch["total_lengths"][index])
        response_length = int(batch["response_lengths"][index])
        token_stream = batch["unconcat_tokens"][index]
        full_response_token_ids = token_stream[-response_length:] if response_length else token_stream[:0]
        full_response_positions = torch.arange(
            response_length,
            dtype=torch.long,
            device=full_response_token_ids.device,
        )
        max_seq_len = int(max_seq_lens[index]) if max_seq_lens is not None else None
        local_token_ids = slice_log_prob_with_cp(
            full_response_token_ids,
            total_length,
            response_length,
            args.qkv_format,
            max_seq_len,
        )
        local_response_positions = slice_log_prob_with_cp(
            full_response_positions,
            total_length,
            response_length,
            args.qkv_format,
            max_seq_len,
        )

        advantage = advantages[index].reshape(-1)
        loss_mask = local_loss_masks[index].reshape(-1)
        sample_ppo_kl = ppo_kl_by_sample[index].reshape(-1)
        sample_pg_loss = pg_loss_by_sample[index].reshape(-1)
        expected_length = train_lp.numel()
        aligned_tensors = {
            "old_log_probs": old_log_probs[index],
            "advantages": advantage,
            "local_loss_mask": loss_mask,
            "ppo_kl": sample_ppo_kl,
            "pg_loss": sample_pg_loss,
            "token_ids": local_token_ids,
            "response_positions": local_response_positions,
        }
        bad_shapes = {
            name: tuple(tensor.shape) for name, tensor in aligned_tensors.items() if tensor.numel() != expected_length
        }
        if bad_shapes:
            raise RuntimeError(
                f"clip-token dump alignment failed for sample {index}: "
                f"train_log_probs={tuple(train_lp.shape)} mismatches={bad_shapes}"
            )

        log_ratio = -sample_ppo_kl
        active_tokens = loss_mask.bool()
        lower_clipped = active_tokens & (advantage < 0) & (log_ratio < lower_log_ratio)
        upper_clipped = active_tokens & (advantage > 0) & (log_ratio > upper_log_ratio)
        clip_side = torch.zeros_like(log_ratio, dtype=torch.int8)
        clip_side = torch.where(lower_clipped, clip_side.new_full((), -1), clip_side)
        clip_side = torch.where(upper_clipped, clip_side.new_full((), 1), clip_side)

        # A bounded exponential is sufficient for inspection and avoids making
        # a debug dump fail on an otherwise diagnosable extreme log-ratio.
        ratio = torch.exp(torch.clamp(log_ratio, min=-20.0, max=20.0))
        unclipped_pg_loss = -ratio * advantage
        clip_loss_delta = sample_pg_loss - unclipped_pg_loss

        lower_clipped_tokens += int(lower_clipped.sum().item())
        upper_clipped_tokens += int(upper_clipped.sum().item())
        active_token_count += int(active_tokens.sum().item())

        sample_index = index
        if sample_indices is not None:
            value = sample_indices[index]
            sample_index = int(value.item()) if isinstance(value, torch.Tensor) else int(value)
        sample = {
            "index": index,
            "sample_index": sample_index,
            "total_length": total_length,
            "response_length": response_length,
            "token_ids": to_cpu(local_token_ids),
            "response_positions": to_cpu(local_response_positions),
            "train_log_probs": to_cpu_float(train_lp),
            "old_log_probs": to_cpu_float(old_log_probs[index]),
            "advantages": to_cpu_float(advantage),
            "local_loss_mask": to_cpu(loss_mask),
            "log_ratio": to_cpu_float(log_ratio),
            "ratio": to_cpu_float(ratio),
            "effective_clip_side": to_cpu(clip_side),
            "pg_loss": to_cpu_float(sample_pg_loss),
            "unclipped_pg_loss": to_cpu_float(unclipped_pg_loss),
            "clip_loss_delta": to_cpu_float(clip_loss_delta),
        }
        if rollout_log_probs is not None:
            sample["rollout_log_probs"] = to_cpu_float(rollout_log_probs[index])
            if train_lp.shape == rollout_log_probs[index].shape:
                sample["train_rollout_abs_diff"] = to_cpu_float((train_lp - rollout_log_probs[index]).abs())
        samples.append(sample)

    torch.save(
        {
            "rank": rank,
            "call": counter,
            "clip": {
                "eps_clip": eps_clip,
                "eps_clip_high": eps_clip_high,
                "lower_ratio": lower_ratio,
                "upper_ratio": upper_ratio,
                "side_encoding": {"lower": -1, "none": 0, "upper": 1},
                "active_token_count": active_token_count,
                "lower_clipped_token_count": lower_clipped_tokens,
                "upper_clipped_token_count": upper_clipped_tokens,
            },
            "samples": samples,
            "ppo_kl": to_cpu_float(ppo_kl),
            "pg_loss": to_cpu_float(pg_loss),
            "finite": {
                "ppo_kl": torch.isfinite(ppo_kl).all().item(),
                "pg_loss": torch.isfinite(pg_loss).all().item(),
                "train_log_probs": all(torch.isfinite(t).all().item() for t in train_log_probs),
                "old_log_probs": all(torch.isfinite(t).all().item() for t in old_log_probs),
                "advantages": all(torch.isfinite(t).all().item() for t in advantages),
            },
        },
        path,
    )
    print(
        f"[MILES_CLIP_TOKEN_DUMP] wrote {path} active={active_token_count} "
        f"lower={lower_clipped_tokens} upper={upper_clipped_tokens}",
        flush=True,
    )
