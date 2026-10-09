import torch

from miles.rollout.filter_hub.base_types import DynamicFilterOutput
from miles.utils.types import Sample

__all__ = ["check_reward_nonzero_std", "check_no_aborted"]


def check_reward_nonzero_std(args, samples: list[Sample], **kwargs):
    rewards = [sample.get_reward_value(args) for sample in samples]
    keep = torch.tensor(rewards, dtype=torch.float64).std() > 1e-8
    return DynamicFilterOutput(
        keep=keep,
        reason=None if keep else f"zero_std_{round(rewards[0], 1)}",
    )


def _flatten_samples(samples):
    """Flatten nested sample lists produced by ``--generate-multi-samples``."""
    if not isinstance(samples, list):
        raise TypeError(f"samples must be a list, got {type(samples).__name__}")
    for sample in samples:
        if isinstance(sample, list):
            yield from _flatten_samples(sample)
        elif isinstance(sample, Sample):
            yield sample
        else:
            raise TypeError(f"sample must be Sample or list[Sample], got {type(sample).__name__}")


def check_no_aborted(args, samples: list[Sample], **kwargs):
    """Reject a group containing an aborted or explicitly excluded sample."""
    flat_samples = list(_flatten_samples(samples))
    if any(sample.status == Sample.Status.ABORTED for sample in flat_samples):
        return DynamicFilterOutput(keep=False, reason="group_has_aborted")
    for sample in flat_samples:
        if not isinstance(sample.metadata, dict):
            raise TypeError(f"sample.metadata must be a dict, got {type(sample.metadata).__name__}")
        excluded = sample.metadata.get("agent_excluded_from_training", False)
        if not isinstance(excluded, bool):
            raise TypeError("sample.metadata['agent_excluded_from_training'] must be a bool")
        if excluded:
            return DynamicFilterOutput(keep=False, reason="group_has_agent_excluded")
    return DynamicFilterOutput(keep=True)
