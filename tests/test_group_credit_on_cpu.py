"""Exercise the real Miles reward centering with transport-only Ray stubs."""
import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest

from miles.utils.types import Sample


@pytest.fixture
def conversion(monkeypatch):
    fake_ray = ModuleType("ray")
    fake_utils = ModuleType("miles.utils.ray_utils")
    fake_utils.Box = object
    monkeypatch.setitem(sys.modules, "ray", fake_ray)
    monkeypatch.setitem(sys.modules, "miles.utils.ray_utils", fake_utils)
    path = Path(__file__).resolve().parents[1] / "miles/ray/rollout/train_data_conversion.py"
    spec = importlib.util.spec_from_file_location("credit_conversion", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def config(groups, size):
    assert groups > 0 and size > 0
    return SimpleNamespace(advantage_estimator="grpo", rewards_normalization=True,
                           grpo_std_normalization=False, grpo_std_normalization_local_only=False,
                           n_samples_per_prompt=size, rollout_batch_size=groups, reward_key="score")


def test_full_and_local_rewards_center_independently(conversion):
    samples = [Sample(reward={"score": score}, group_index=i // 2, loss_mask=[1],
                      metadata={"ab_local_rollout": i >= 2}) for i, score in enumerate([0, 1, 0.5, 1])]
    raw, advantages = conversion._post_process_rewards(config(2, 2), samples, None)
    assert raw == [0, 1, 0.5, 1]
    assert advantages == [-0.5, 0.5, -0.25, 0.25]


@pytest.mark.parametrize("rewards", [[0], [1], [0.5, 0.5], [1, 1, 1]])
def test_constant_or_singleton_group_has_zero_credit(conversion, rewards):
    samples = [Sample(reward={"score": score}, loss_mask=[1]) for score in rewards]
    assert conversion._post_process_rewards(config(1, len(rewards)), samples, None)[1] == [0] * len(rewards)


def test_excluded_sibling_is_not_in_baseline(conversion):
    samples = [Sample(reward={"score": score}, loss_mask=[1], remove_sample=(i == 2))
               for i, score in enumerate([0, 1, 1])]
    assert conversion._post_process_rewards(config(1, 3), samples, None)[1] == [-0.5, 0.5, 0]


def test_inconsistent_normalization_mode_fails(conversion):
    args = config(1, 1)
    args.grpo_std_normalization_local_only = True
    with pytest.raises(ValueError, match="requires"):
        conversion._post_process_rewards(args, [Sample(reward={"score": 1}, loss_mask=[1])], None)
