import os
from argparse import Namespace
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any

import torch
from torch.utils.checkpoint import checkpoint

from miles.backends.training_utils.cp_utils import (
    allgather_cp_redistribute,
    get_logits_and_tokens_offset_with_cp,
)
from miles.backends.training_utils.loss_hub.math_utils import calculate_log_probs_and_entropy
from miles.backends.training_utils.parallel import get_parallel_state


@dataclass(frozen=True)
class ChunkedLMHeadContext:
    """Inputs required to turn decoder rows directly into target log-probs."""

    args: Namespace
    unconcat_tokens: list[torch.Tensor]
    total_lengths: list[int]
    response_lengths: list[int]
    max_seq_lens: list[int] | None
    checkpoint_chunks: bool


def _binary_env_enabled(name: str) -> bool:
    """Read a strict binary feature flag from the environment."""

    value = os.getenv(name, "0")
    if value not in {"0", "1"}:
        raise ValueError(f"{name} must be '0' or '1', got {value!r}")
    return value == "1"


def bf16_logprob_forward_enabled() -> bool:
    """Whether forward-only scoring should use the response-only chunked LM head."""

    return _binary_env_enabled("MILES_BF16_LOGPROB_FORWARD")


def bf16_logprob_backward_enabled() -> bool:
    """Whether training should use the checkpointed response-only chunked LM head."""

    return _binary_env_enabled("MILES_BF16_LOGPROB_BACKWARD")


def bf16_logprob_any_enabled() -> bool:
    """Whether either supported caller may hand this module bf16 logits."""

    # Read both flags before combining them so an invalid BACKWARD value cannot
    # hide behind a true FORWARD value (or vice versa).
    forward_enabled = bf16_logprob_forward_enabled()
    backward_enabled = bf16_logprob_backward_enabled()
    return forward_enabled or backward_enabled


def build_chunked_lm_head_context(
    *,
    args: Namespace,
    unconcat_tokens: list[torch.Tensor],
    total_lengths: list[int],
    response_lengths: list[int],
    max_seq_lens: list[int] | None,
    checkpoint_chunks: bool,
) -> ChunkedLMHeadContext:
    """Validate and package metadata consumed inside Megatron's LM-head hook."""

    num_samples = len(unconcat_tokens)
    if num_samples == 0:
        raise ValueError("chunked LM head requires at least one sample")
    if len(total_lengths) != num_samples or len(response_lengths) != num_samples:
        raise ValueError(
            "chunked LM-head metadata length mismatch: "
            f"tokens={num_samples}, total_lengths={len(total_lengths)}, response_lengths={len(response_lengths)}"
        )
    if max_seq_lens is not None and len(max_seq_lens) != num_samples:
        raise ValueError(f"max_seq_lens length mismatch: expected {num_samples}, got {len(max_seq_lens)}")
    chunk_size = getattr(args, "log_probs_chunk_size", None)
    if not isinstance(chunk_size, int) or chunk_size <= 0:
        raise ValueError(f"chunked LM head requires log_probs_chunk_size > 0, got {chunk_size!r}")
    if getattr(args, "tensor_model_parallel_size", None) != 1:
        raise ValueError(
            "chunked LM head currently requires tensor-model-parallel-size=1; "
            f"got {getattr(args, 'tensor_model_parallel_size', None)!r}"
        )
    if getattr(args, "true_on_policy_mode", False):
        raise ValueError("chunked LM head does not support true-on-policy mode")

    for sample_idx, (tokens, total_length, response_length) in enumerate(
        zip(unconcat_tokens, total_lengths, response_lengths, strict=True)
    ):
        if tokens.ndim != 1:
            raise ValueError(f"sample {sample_idx}: tokens must be 1D, got shape={tuple(tokens.shape)}")
        if not isinstance(total_length, int) or total_length <= 0:
            raise ValueError(f"sample {sample_idx}: total_length must be a positive int, got {total_length!r}")
        if tokens.numel() != total_length:
            raise ValueError(
                f"sample {sample_idx}: token count {tokens.numel()} does not equal total_length {total_length}"
            )
        if not isinstance(response_length, int) or not 0 < response_length < total_length:
            raise ValueError(
                f"sample {sample_idx}: response_length must satisfy 0 < response_length < total_length; "
                f"got response_length={response_length!r}, total_length={total_length}"
            )

    return ChunkedLMHeadContext(
        args=args,
        unconcat_tokens=unconcat_tokens,
        total_lengths=total_lengths,
        response_lengths=response_lengths,
        max_seq_lens=max_seq_lens,
        checkpoint_chunks=checkpoint_chunks,
    )


def get_responses(
    logits: torch.Tensor,
    *,
    args: Namespace,
    unconcat_tokens: list[torch.Tensor],
    total_lengths: list[int],
    response_lengths: list[int],
    max_seq_lens: list[int] | None = None,
    allow_bf16_logits: bool = False,
) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    """Yield response-aligned `(logits_chunk, tokens_chunk)` pairs per sample.

    After squeezing batch dimension and applying temperature scaling, this
    function extracts the logits and tokens corresponding to response segments
    for each sample. When context parallelism is disabled, it slices directly
    from the concatenated sequence. With context parallelism enabled, it
    handles split sequences across ranks.

    Args:
        logits: Model outputs with shape `[1, T, V]` (policy) or `[1, T, 1]`
            (value). Must be float32 unless `allow_bf16_logits=True`.
        args: Configuration containing `rollout_temperature` for scaling.
        unconcat_tokens: List of token tensors (prompt+response) per sample.
        total_lengths: Total sequence lengths (prompt+response) per sample.
        response_lengths: Response segment lengths per sample.

    Yields:
        Tuple of `(logits_chunk, tokens_chunk)` where `logits_chunk` is shape
        `[R, V]` (policy) or `[R, 1]` (value) and `tokens_chunk` is shape `[R]`
        (1D int64), both aligned to response tokens for one sample.
    """
    qkv_format = args.qkv_format

    if not args.true_on_policy_mode:
        if allow_bf16_logits:
            assert logits.dtype in (torch.bfloat16, torch.float32), f"{logits.dtype}"
        else:
            assert logits.dtype == torch.float32, f"{logits.dtype}"
    assert len(logits.shape) == 3, f"{logits.shape}"

    if qkv_format == "thd":
        assert logits.size(0) == 1, f"{logits.shape}"
        logits = logits.squeeze(0)
    else:
        assert max_seq_lens is not None
        logits = logits.view(-1, logits.size(-1))

    if logits.size(-1) > 1 and args.rollout_temperature > 0 and args.rollout_temperature != 1.0:
        logits = logits.div(args.rollout_temperature)
    if args.true_on_policy_mode:
        if getattr(args, "bf16", False):
            logits = logits.to(torch.bfloat16)
        elif getattr(args, "fp16", False):
            logits = logits.to(torch.float16)

    parallel_state = get_parallel_state()
    cp_size = parallel_state.cp.size
    end = 0
    seq_start = 0
    for i, (tokens, total_length, response_length) in enumerate(
        zip(unconcat_tokens, total_lengths, response_lengths, strict=False)
    ):
        max_seq_len = max_seq_lens[i] if max_seq_lens is not None else None

        if cp_size == 1:
            if qkv_format == "bshd":
                end = max_seq_len * i + total_length
                start = end - response_length
                logits_chunk = logits[start - 1 : end - 1]
            else:
                end += total_length
                start = end - response_length
                logits_chunk = logits[start - 1 : end - 1]
            tokens_chunk = tokens[-response_length:]
        elif args.allgather_cp:
            # DSA: global concat then contiguous CP split. Each rank owns logits for
            # global positions [chunk_start, chunk_end).
            logits_local_len = logits.size(0)
            cp_rank = parallel_state.cp.rank
            chunk_start = cp_rank * logits_local_len
            chunk_end = chunk_start + logits_local_len

            prompt_length = total_length - response_length
            resp_token_start = seq_start + prompt_length
            resp_token_end = seq_start + total_length
            logit_global_start = resp_token_start - 1
            logit_global_end = resp_token_end - 1

            s = max(logit_global_start, chunk_start)
            e = min(logit_global_end, chunk_end)
            if e <= s:
                logits_chunk = logits[0:0]
                tokens_chunk = tokens[0:0]
            else:
                logits_chunk = logits[s - chunk_start : e - chunk_start]
                tokens_chunk = tokens[(s + 1) - seq_start : (e + 1) - seq_start]
            assert logits_chunk.size(0) == tokens_chunk.size(0), f"{logits_chunk.size(0)} vs {tokens_chunk.size(0)}"
        else:
            # TODO: this is super ugly... do better abstraction.
            chunk_size, chunks_offset, logits_offset, tokens_offset = get_logits_and_tokens_offset_with_cp(
                total_length, response_length, qkv_format, max_seq_len
            )

            logits_0, logits_1 = logits[end : end + chunk_size], logits[end + chunk_size : end + 2 * chunk_size]
            end += 2 * chunk_size

            logits_0 = logits_0[logits_offset[0][0] - chunks_offset[0][0] : logits_offset[0][1] - chunks_offset[0][0]]
            tokens_0 = tokens[tokens_offset[0][0] : tokens_offset[0][1]]

            logits_1 = logits_1[logits_offset[1][0] - chunks_offset[1][0] : logits_offset[1][1] - chunks_offset[1][0]]
            tokens_1 = tokens[tokens_offset[1][0] : tokens_offset[1][1]]

            assert logits_0.size(0) == tokens_0.size(0), f"{logits_0.size(0)} vs {tokens_0.size(0)}"
            assert logits_1.size(0) == tokens_1.size(0), f"{logits_1.size(0)} vs {tokens_1.size(0)}"

            logits_chunk = torch.cat([logits_0, logits_1], dim=0)
            tokens_chunk = torch.cat([tokens_0, tokens_1], dim=0)

        seq_start += total_length

        yield logits_chunk, tokens_chunk


def _response_row_indices_and_tokens(
    hidden_states_bsh: torch.Tensor,
    context: ChunkedLMHeadContext,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return flattened local hidden-row indices and their next-token targets."""

    if hidden_states_bsh.ndim != 3:
        raise ValueError(f"hidden_states_bsh must be 3D [B,S,H], got {tuple(hidden_states_bsh.shape)}")
    batch_size, seq_length, hidden_size = hidden_states_bsh.shape
    if batch_size <= 0 or seq_length <= 0 or hidden_size <= 0:
        raise ValueError(f"hidden_states_bsh dimensions must be positive, got {tuple(hidden_states_bsh.shape)}")

    flat_rows = batch_size * seq_length
    row_ids = torch.arange(flat_rows, device=hidden_states_bsh.device, dtype=torch.float32).view(
        batch_size, seq_length, 1
    )
    response_parts = list(
        get_responses(
            row_ids,
            args=context.args,
            unconcat_tokens=context.unconcat_tokens,
            total_lengths=context.total_lengths,
            response_lengths=context.response_lengths,
            max_seq_lens=context.max_seq_lens,
            allow_bf16_logits=False,
        )
    )
    if len(response_parts) != len(context.total_lengths):
        raise RuntimeError(
            f"response alignment returned {len(response_parts)} samples, expected {len(context.total_lengths)}"
        )

    row_indices = torch.cat([rows.reshape(-1).to(torch.long) for rows, _ in response_parts], dim=0)
    target_tokens = torch.cat([tokens.reshape(-1).to(torch.long) for _, tokens in response_parts], dim=0)
    if row_indices.numel() != target_tokens.numel():
        raise RuntimeError(f"response row/token mismatch: rows={row_indices.numel()}, tokens={target_tokens.numel()}")
    if row_indices.numel() > 0:
        min_row = int(row_indices.min().item())
        max_row = int(row_indices.max().item())
        if min_row < 0 or max_row >= flat_rows:
            raise RuntimeError(f"response row index outside [0,{flat_rows}): min={min_row}, max={max_row}")
        if torch.unique(row_indices).numel() != row_indices.numel():
            raise RuntimeError("response alignment produced duplicate hidden-row indices")
    return row_indices, target_tokens


def _call_lm_head_for_target_log_probs(
    hidden_rows: torch.Tensor,
    target_tokens: torch.Tensor,
    *,
    output_layer: Callable[..., Any],
    output_weight: torch.Tensor | None,
    runtime_gather_output: bool | None,
    temperature: float,
) -> torch.Tensor:
    """Materialize one small vocab-logit tile and immediately reduce it to log-probs."""

    if hidden_rows.ndim != 2:
        raise ValueError(f"hidden_rows must be 2D [R,H], got {tuple(hidden_rows.shape)}")
    if target_tokens.ndim != 1 or target_tokens.numel() != hidden_rows.size(0):
        raise ValueError(
            "target_tokens must be 1D and match hidden rows; "
            f"got tokens={tuple(target_tokens.shape)}, hidden_rows={tuple(hidden_rows.shape)}"
        )
    output = output_layer(
        hidden_rows.unsqueeze(1),
        weight=output_weight,
        runtime_gather_output=runtime_gather_output,
    )
    if not isinstance(output, tuple) or len(output) < 1 or not isinstance(output[0], torch.Tensor):
        raise TypeError(f"output_layer must return a tuple whose first item is a tensor, got {type(output)!r}")
    logits = output[0]
    if logits.ndim != 3 or logits.size(0) != hidden_rows.size(0) or logits.size(1) != 1:
        raise RuntimeError(
            "output_layer returned an unexpected shape; expected [R,1,V], "
            f"got {tuple(logits.shape)} for hidden_rows={tuple(hidden_rows.shape)}"
        )
    logits = logits.squeeze(1)
    if logits.size(1) <= 0:
        raise RuntimeError(f"output_layer returned an empty vocabulary dimension: {tuple(logits.shape)}")
    if target_tokens.numel() > 0:
        min_token = int(target_tokens.min().item())
        max_token = int(target_tokens.max().item())
        if min_token < 0 or max_token >= logits.size(1):
            raise ValueError(
                f"target token outside output vocabulary [0,{logits.size(1)}): min={min_token}, max={max_token}"
            )
    if temperature > 0 and temperature != 1.0:
        logits = logits / temperature
    logits_fp32 = logits.float()
    targets = logits_fp32.gather(dim=-1, index=target_tokens.unsqueeze(-1)).squeeze(-1)
    return targets - torch.logsumexp(logits_fp32, dim=-1)


def chunked_lm_head_logprob_processor(
    *,
    hidden_states: torch.Tensor,
    output_layer: Callable[..., Any],
    output_weight: torch.Tensor | None,
    runtime_gather_output: bool | None,
    context: ChunkedLMHeadContext,
    labels: torch.Tensor | None = None,
    **_: Any,
) -> torch.Tensor:
    """Megatron output processor that never materializes the full ``[T,V]`` logits."""

    if not isinstance(context, ChunkedLMHeadContext):
        raise TypeError(f"context must be ChunkedLMHeadContext, got {type(context)!r}")
    if labels is not None:
        raise ValueError("chunked LM-head log-prob processor requires labels=None")
    if hidden_states.ndim != 3:
        raise ValueError(f"hidden_states must be [S,B,H], got {tuple(hidden_states.shape)}")

    hidden_states_bsh = hidden_states.transpose(0, 1)
    flat_hidden = hidden_states_bsh.reshape(-1, hidden_states_bsh.size(-1))
    row_indices, target_tokens = _response_row_indices_and_tokens(hidden_states_bsh, context)
    selected_hidden = flat_hidden.index_select(0, row_indices)

    chunk_size = context.args.log_probs_chunk_size
    temperature = float(getattr(context.args, "rollout_temperature", 1.0))
    log_prob_chunks = []
    for start in range(0, selected_hidden.size(0), chunk_size):
        end = min(start + chunk_size, selected_hidden.size(0))
        hidden_chunk = selected_hidden[start:end]
        token_chunk = target_tokens[start:end]

        def compute(chunk_hidden: torch.Tensor, chunk_tokens: torch.Tensor) -> torch.Tensor:
            return _call_lm_head_for_target_log_probs(
                chunk_hidden,
                chunk_tokens,
                output_layer=output_layer,
                output_weight=output_weight,
                runtime_gather_output=runtime_gather_output,
                temperature=temperature,
            )

        if context.checkpoint_chunks and torch.is_grad_enabled():
            log_prob_chunk = checkpoint(compute, hidden_chunk, token_chunk, use_reentrant=False)
        else:
            log_prob_chunk = compute(hidden_chunk, token_chunk)
        if log_prob_chunk.shape != (end - start,):
            raise RuntimeError(
                f"LM-head chunk returned shape {tuple(log_prob_chunk.shape)}, expected {(end - start,)}"
            )
        log_prob_chunks.append(log_prob_chunk)

    if log_prob_chunks:
        selected_log_probs = torch.cat(log_prob_chunks, dim=0)
        dense_flat = selected_log_probs.new_zeros((flat_hidden.size(0),))
        dense_flat = dense_flat.index_copy(0, row_indices, selected_log_probs)
    else:
        # Preserve an autograd edge on CP ranks that own no response rows.
        dense_flat = flat_hidden[:, 0].float() * 0.0
    return dense_flat.view(hidden_states_bsh.size(0), hidden_states_bsh.size(1), 1)


def get_precomputed_log_probs(
    log_probs: torch.Tensor,
    *,
    args: Namespace,
    unconcat_tokens: list[torch.Tensor],
    total_lengths: list[int],
    response_lengths: list[int],
    with_entropy: bool = False,
    non_loss_data: bool = True,
    max_seq_lens: list[int] | None = None,
) -> dict[str, list[torch.Tensor]]:
    """Slice dense response-aligned scalar log-probs produced inside the LM head."""

    if not non_loss_data:
        raise ValueError("get_precomputed_log_probs requires non_loss_data=True")
    if with_entropy:
        raise ValueError("chunked LM-head precomputed log-probs do not support entropy")
    if log_probs.ndim != 3 or log_probs.size(-1) != 1:
        raise ValueError(f"precomputed log_probs must have shape [B,S,1], got {tuple(log_probs.shape)}")
    if log_probs.dtype != torch.float32:
        raise ValueError(f"precomputed log_probs must be float32, got {log_probs.dtype}")

    values = [
        rows.reshape(-1)
        for rows, _ in get_responses(
            log_probs,
            args=args,
            unconcat_tokens=unconcat_tokens,
            total_lengths=total_lengths,
            response_lengths=response_lengths,
            max_seq_lens=max_seq_lens,
            allow_bf16_logits=False,
        )
    ]
    result = {"log_probs": values}
    if args.allgather_cp:
        allgather_cp_redistribute(
            result,
            logits=log_probs,
            args=args,
            total_lengths=total_lengths,
            response_lengths=response_lengths,
            max_seq_lens=max_seq_lens,
        )
    return result


def get_log_probs_and_entropy(
    logits: torch.Tensor,
    *,
    args: Namespace,
    unconcat_tokens: list[torch.Tensor],
    total_lengths: list[int],
    response_lengths: list[int],
    with_entropy: bool = False,
    non_loss_data: bool = True,
    max_seq_lens: list[int] | None = None,
) -> dict[str, list[torch.Tensor]]:
    """Compute per-token log-probabilities (and optionally entropy) on responses.

    For each sample, extracts response-aligned logits and tokens, then computes
    log-probabilities via softmax across the tensor-parallel group. Log-probs
    are normalized to `[R]`, including the single-token case. Entropy values are always appended
    (even when `with_entropy=False`), but only included in the result dict
    when requested.

    Args:
        logits: Policy logits with shape `[1, T, V]`.
        args: Configuration (temperature applied in `get_responses`).
        unconcat_tokens: List of token tensors per sample.
        total_lengths: Total sequence lengths per sample.
        response_lengths: Response segment lengths per sample.
        with_entropy: If True, include "entropy" key in result.
        non_loss_data: Unused; kept for API compatibility.

    Returns:
        Dict with key "log_probs" mapping to a list of `[R]` tensors per
        sample. If `with_entropy` is True, also includes "entropy" key with
        a list of `[R]` tensors.
    """
    assert non_loss_data
    parallel_state = get_parallel_state()
    log_probs_list = []
    entropy_list = []
    allow_bf16_logits = bf16_logprob_any_enabled()
    for logits_chunk, tokens_chunk in get_responses(
        logits,
        args=args,
        unconcat_tokens=unconcat_tokens,
        total_lengths=total_lengths,
        response_lengths=response_lengths,
        max_seq_lens=max_seq_lens,
        allow_bf16_logits=allow_bf16_logits,
    ):
        log_prob, entropy = calculate_log_probs_and_entropy(
            logits_chunk,
            tokens_chunk,
            parallel_state.tp.group,
            with_entropy=with_entropy,
            chunk_size=args.log_probs_chunk_size,
            true_on_policy=args.true_on_policy_mode,
            vocab_size=getattr(args, "vocab_size", None),
            bf16_rowwise=allow_bf16_logits,
        )

        # `calculate_log_probs_and_entropy` normally returns `[R]`. Using
        # squeeze(-1) here turns the valid single-token shape `[1]` into a
        # scalar, which later breaks torch.cat over per-sample log-probs.
        log_probs_list.append(log_prob.reshape(-1))
        entropy_list.append(entropy)

    res = {
        "log_probs": log_probs_list,
    }
    if with_entropy:
        res["entropy"] = entropy_list

    # we need to turn the all gather kv into zigzag ring attn kv
    if args.allgather_cp:
        allgather_cp_redistribute(
            res,
            logits=logits,
            args=args,
            total_lengths=total_lengths,
            response_lengths=response_lengths,
            max_seq_lens=max_seq_lens,
        )

    return res


def get_values(
    logits: torch.Tensor,
    *,
    args: Namespace,
    unconcat_tokens: list[torch.Tensor],
    total_lengths: list[int],
    response_lengths: list[int],
    with_entropy: bool = False,
    non_loss_data: bool = True,
    max_seq_lens: list[int] | None = None,
) -> dict[str, list[torch.Tensor]]:
    """Extract per-token value predictions over response tokens.

    For each sample, extracts response-aligned chunks from the value head
    output and squeezes the final dimension from `[R, 1]` to `[R]`.

    Args:
        logits: Value head output with shape `[1, T, 1]`.
        args: Configuration (passed to `get_responses` which uses
            `rollout_temperature` even though values don't need temperature).
        unconcat_tokens: List of token tensors per sample.
        total_lengths: Total sequence lengths per sample.
        response_lengths: Response segment lengths per sample.
        with_entropy: Unused; kept for signature compatibility.
        non_loss_data: Unused; kept for signature compatibility.

    Returns:
        Dict with key "values" mapping to a list of `[R]` value tensors
        per sample.
    """
    value_list = []
    for logits_chunk, _ in get_responses(
        logits,
        args=args,
        unconcat_tokens=unconcat_tokens,
        total_lengths=total_lengths,
        response_lengths=response_lengths,
        max_seq_lens=max_seq_lens,
    ):
        assert logits_chunk.size(-1) == 1, f"{logits_chunk.shape}"
        value_list.append(logits_chunk.squeeze(-1))

    res = {
        "values": value_list,
    }

    if args.allgather_cp:
        allgather_cp_redistribute(
            res,
            logits=logits,
            args=args,
            total_lengths=total_lengths,
            response_lengths=response_lengths,
            max_seq_lens=max_seq_lens,
        )

    return res
