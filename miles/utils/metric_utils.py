import math
from typing import Any, Literal

import numpy as np

SWE_AGENT_METRIC_KEYS = {
    "agent_repeated_function_close",
    "agent_format_failure_penalized",
    "agent_single_call_length_penalized",
    "agent_behavior_anomaly_penalized",
    "agent_swe_tool_metrics_available",
    "agent_swe_tool_result_count",
    "agent_swe_tool_execution_exception_count",
    "agent_swe_tool_timeout_count",
    "agent_swe_tool_returncode_zero_count",
    "agent_swe_tool_returncode_nonzero_count",
    "agent_swe_tool_returncode_zero_rate",
    "agent_swe_tool_submit_count",
    "agent_swe_tool_unobserved_action_count",
    "agent_swe_harbor_request_count",
    "agent_swe_harbor_request_success_count",
    "agent_swe_harbor_connect_error_count",
    "agent_swe_harbor_http_error_count",
    "agent_swe_harbor_timeout_count",
    "agent_swe_harbor_transport_error_count",
}


def finalize_swe_tool_rates(metrics: dict[str, float], *, prefix: str = "") -> None:
    """Recompute pooled ratios after sample/rank aggregation; omit undefined ratios."""
    if not isinstance(metrics, dict) or not isinstance(prefix, str):
        raise TypeError("metrics must be a dict and prefix a string")
    if f"{prefix}agent_swe_tool_metrics_available" not in metrics:
        return  # Preserve other agents' established empty-batch semantics.
    counts = {}
    for key in (
        "agent_swe_tool_returncode_zero_count",
        "agent_swe_tool_returncode_nonzero_count",
        "agent_swe_tool_result_count",
    ):
        value = metrics.get(prefix + key, 0.0)
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(f"invalid SWE aggregated count {key}={value!r}")
        counts[key] = value
    returned = counts["agent_swe_tool_returncode_zero_count"] + counts["agent_swe_tool_returncode_nonzero_count"]
    observed = counts["agent_swe_tool_result_count"]
    if returned > observed and not math.isclose(returned, observed):
        raise ValueError("SWE returned command count exceeds observed tool results")
    for key, numerator, denominator in (
        ("agent_tool_unit_success_rate", returned, observed),
        ("agent_swe_tool_returncode_zero_rate", counts["agent_swe_tool_returncode_zero_count"], returned),
    ):
        if denominator:
            metrics[prefix + key] = numerator / denominator
        else:
            metrics.pop(prefix + key, None)


def dict_add_prefix(d: dict[str, Any], prefix: str) -> dict[str, Any]:
    return {f"{prefix}{k}": v for k, v in d.items()}


def compute_tool_unit_success_rate(
    unit_counts: list[float],
    *,
    unit_success_counts: list[float] | None = None,
    unit_success_rates: list[float] | None = None,
) -> float:
    """Compute a micro-averaged tool-unit success rate.

    Samples with no tool units contribute neither successes nor attempts. If
    the complete collection has no attempted units, success is vacuously 1.
    """
    counts = [float(value) for value in unit_counts]
    if unit_success_counts is None:
        if unit_success_rates is None:
            raise ValueError("unit_success_counts or unit_success_rates is required")
        rates = [float(value) for value in unit_success_rates]
        if len(rates) != len(counts):
            raise ValueError("unit_counts and unit_success_rates must have the same length")
        successes = [count * rate for count, rate in zip(counts, rates, strict=True)]
    else:
        successes = [float(value) for value in unit_success_counts]
        if len(successes) != len(counts):
            raise ValueError("unit_counts and unit_success_counts must have the same length")

    total_units = sum(counts)
    return sum(successes) / total_units if total_units > 0 else 1.0


def compute_pass_rate(
    flat_rewards: list[float],
    group_size: int,
    num_groups: int | None = None,
):
    if group_size == 1:
        return {}

    if num_groups is None:
        num_groups = len(flat_rewards) // group_size

    pass_rate_name_list = [2**i for i in range(int(math.log2(group_size)) + 1)]

    assert len(flat_rewards) == num_groups * group_size, f"{len(flat_rewards)=} {num_groups=} {group_size=}"
    rewards_of_group = np.array(flat_rewards).reshape(num_groups, group_size)

    log_dict = {}
    for k in pass_rate_name_list:
        num_correct = np.sum(rewards_of_group == 1, axis=1)
        num_samples = np.full(num_groups, group_size)

        pass_k_estimates = _estimate_pass_at_k(num_samples, num_correct, k)

        pass_k = np.mean(pass_k_estimates)
        log_dict[f"pass@{k}"] = pass_k

    return log_dict


def _estimate_pass_at_k(num_samples, num_correct, k):
    """
    Estimates pass@k of each problem and returns them in an array.
    """

    def estimator(n, c, k):
        """
        Calculates 1 - comb(n - c, k) / comb(n, k).
        """
        if n - c < k:
            return 1.0
        return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))

    return np.array([estimator(int(n), int(c), k) for n, c in zip(num_samples, num_correct, strict=False)])


def compute_statistics(values: list[float]) -> dict[str, float]:
    values = np.array(values)
    return {
        "mean": np.mean(values).item(),
        "median": np.median(values).item(),
        "max": np.max(values).item(),
        "min": np.min(values).item(),
    }


def compression_ratio(
    data: str | bytes,
    *,
    encoding: str = "utf-8",
    algorithm: Literal["zlib", "gzip", "bz2", "lzma"] = "zlib",
    level: int = 9,
) -> tuple[float, float]:
    if isinstance(data, str):
        raw = data.encode(encoding)
    else:
        raw = data

    original = len(raw)
    if original == 0:
        return float("inf"), 0.0

    if algorithm == "zlib":
        import zlib

        compressed = zlib.compress(raw, level)
    elif algorithm == "gzip":
        import gzip

        compressed = gzip.compress(raw, compresslevel=level)
    elif algorithm == "bz2":
        import bz2

        compressed = bz2.compress(raw, compresslevel=level)
    elif algorithm == "lzma":
        import lzma

        compressed = lzma.compress(raw, preset=level)
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")

    comp_len = len(compressed)
    if comp_len == 0:
        return float("inf"), 100.0

    ratio = original / comp_len
    savings_pct = 100.0 * (1.0 - comp_len / original)
    return ratio, savings_pct


def has_repetition(text: str):
    if len(text) > 10000 and compression_ratio(text[-10000:])[0] > 10:
        return True
    else:
        return False


def compute_rollout_step(args, rollout_id):
    if args.wandb_always_use_train_step:
        return rollout_id * args.rollout_batch_size * args.n_samples_per_prompt // args.global_batch_size
    return rollout_id
