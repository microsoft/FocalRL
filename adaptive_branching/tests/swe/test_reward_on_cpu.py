import asyncio
import importlib.util
import statistics
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from adaptive_branching.src.swe import reward
from miles.utils.types import Sample


@pytest.fixture(autouse=True)
def clean_force_exclude(monkeypatch):
    monkeypatch.delenv("FORCE_EXCLUDE", raising=False)


def test_reward_accepts_single_and_batched_samples():
    single = asyncio.run(reward.reward_func(None, Sample(metadata={"reward": 1})))
    batch = asyncio.run(reward.reward_func(None, [Sample(metadata={"reward": 0.0}), Sample(metadata={"reward": 1.0})]))

    assert single["score"] == 1.0
    assert [item["score"] for item in batch] == [0.0, 1.0]


@pytest.mark.parametrize("reward_value", [None, True, "1", 0.5])
def test_reward_rejects_missing_or_non_binary_values(reward_value):
    with pytest.raises((TypeError, ValueError), match="reward"):
        asyncio.run(reward.reward_func(None, Sample(metadata={"reward": reward_value})))


def test_reward_rejects_empty_batch():
    with pytest.raises(ValueError, match="must not be empty"):
        asyncio.run(reward.reward_func(None, []))


def _sample(index=0, *, local=False, status=Sample.Status.COMPLETED, **metadata):
    metadata = {"swe_generated_format_audit": {"version": 2, "spans": 1, "repeated_spans": 0, "invalid_spans": 0, "format_failures": {}}, **metadata}
    if local:
        metadata = {
            "ab_local_rollout": True,
            "agent_max_turns": 10,
            "ab_local_prefix_message_count": 2,
            "ab_event_recovery_rubric": {"avoid_error": "stop X", "redirect": "test Y"},
            "messages": [
                {"role": "system", "content": "system"},
                {"role": "user", "content": "issue"},
                {"role": "assistant", "content": "test Y"},
                {"role": "tool", "content": "evidence"},
            ],
            **metadata,
        }
    return Sample(
        group_index=0,
        index=index,
        prompt="q",
        response="generated",
        tokens=[1, 2, 3],
        response_length=2,
        loss_mask=[1, 1],
        status=status,
        metadata=metadata,
    )


@pytest.mark.parametrize("local", [False, True])
def test_terminal_length_is_trainable_zero_without_judge(monkeypatch, local):
    def unexpected(*args, **kwargs):
        pytest.fail("terminal length must not initialize or call the local judge")

    monkeypatch.setattr(reward, "JudgeClient", unexpected)
    monkeypatch.setattr(reward, "judge_settings", unexpected)
    sample = _sample(
        local=local,
        status=Sample.Status.TRUNCATED,
        agent_last_finish_reason="length",
        exit_status="LengthTruncated",
        reward=1,
    )
    result = asyncio.run(reward.reward_func(None, [sample]))[0]
    assert result["score"] == 0
    assert result["judge_raw"] == "single_call_length"
    assert sample.status == Sample.Status.TRUNCATED
    assert sample.tokens == [1, 2, 3] and sample.loss_mask == [1, 1]
    assert sample.remove_sample is False
    assert sample.metadata["agent_excluded_from_training"] is False


@pytest.mark.parametrize("local", [False, True])
@pytest.mark.parametrize(
    "flags",
    [
        {"agent_excluded_from_training": True},
        {"agent_function_failed": True},
        {"agent_context_reserve_hit": True},
        {"agent_forced_final_answer": True},
    ],
)
def test_infra_failure_takes_priority_over_length_and_never_calls_prm(monkeypatch, local, flags):
    def unexpected(*args, **kwargs):
        pytest.fail("excluded samples must not be judged")

    monkeypatch.setattr(reward, "judge_settings", unexpected)
    sample = _sample(local=local, status=Sample.Status.TRUNCATED, agent_last_finish_reason="length", **flags)
    sample.metadata.pop("messages", None)
    result = asyncio.run(reward.reward_func(None, [sample]))[0]
    assert result["score"] == 0
    assert sample.remove_sample is True
    assert sample.metadata["agent_excluded_from_training"] is True
    assert sample.metadata["agent_single_call_length_penalized"] is False


@pytest.mark.parametrize("status", [Sample.Status.ABORTED, Sample.Status.TRUNCATED])
def test_other_incomplete_samples_are_excluded(status):
    sample = _sample(status=status)
    result = asyncio.run(reward.reward_func(None, sample))
    assert result["score"] == 0 and sample.remove_sample
    assert sample.status == status


def test_length_requires_actual_truncated_status():
    sample = _sample(agent_last_finish_reason="length")
    with pytest.raises(AssertionError, match="requires TRUNCATED"):
        asyncio.run(reward.reward_func(None, sample))


@pytest.mark.parametrize("local", [False, True])
@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("exit_status", ["Submitted", "LimitsExceeded", "LengthTruncated"])
@pytest.mark.parametrize("reason", ["context_reserve", "max_turns"])
@pytest.mark.parametrize("verifier_reward", [0, 1])
def test_forced_limit_excludes_any_verifier_result_and_skips_judges(monkeypatch, local, nested, exit_status, reason, verifier_reward):
    def unexpected(*args, **kwargs):
        pytest.fail("limit-excluded samples must not call PRM")

    monkeypatch.setattr(reward, "judge_settings", unexpected)
    flags = {"agent_forced_final_answer_reason": reason}
    sample = _sample(
        local=local,
        reward=verifier_reward,
        exit_status=exit_status,
        status=Sample.Status.TRUNCATED if exit_status == "LengthTruncated" else Sample.Status.COMPLETED,
        agent_last_finish_reason="length" if exit_status == "LengthTruncated" else "stop",
        **({"agent_metrics": flags} if nested else flags),
    )
    result = asyncio.run(reward.reward_func(None, sample))
    assert result["judge_raw"] == "excluded_from_training"
    assert sample.remove_sample is True
    assert sample.metadata["agent_context_reserve_hit"] is (reason == "context_reserve")
    assert sample.metadata["agent_max_turns_hit"] is (reason == "max_turns")
    assert sample.metadata["reward"] == verifier_reward
    if local:
        assert sample.metadata["ab_local_trainable"] is False


@pytest.mark.parametrize("valid_rewards", [[], [1], [1, 0]])
@pytest.mark.parametrize("reason", ["context_reserve", "max_turns"])
@pytest.mark.parametrize("verifier_reward", [0, 1])
@pytest.mark.parametrize("normalize_std", [False, True])
def test_limit_exclusion_reaches_real_grpo_baseline_and_whole_loss_mask(monkeypatch, valid_rewards, reason, verifier_reward, normalize_std):
    monkeypatch.setitem(sys.modules, "ray", ModuleType("ray"))
    ray_utils = ModuleType("miles.utils.ray_utils")
    ray_utils.Box = object
    monkeypatch.setitem(sys.modules, "miles.utils.ray_utils", ray_utils)
    path = Path(__file__).resolve().parents[3] / "miles/ray/rollout/train_data_conversion.py"
    spec = importlib.util.spec_from_file_location("swe_context_conversion_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    samples = [_sample(index, reward=value) for index, value in enumerate(valid_rewards)]
    samples.append(
        _sample(
            len(samples),
            reward=verifier_reward,
            exit_status="Submitted",
            agent_forced_final_answer_reason=reason,
        )
    )
    results = asyncio.run(reward.reward_func(None, samples))
    for sample, result in zip(samples, results, strict=True):
        sample.reward = result
    args = SimpleNamespace(
        advantage_estimator="grpo",
        rewards_normalization=True,
        grpo_std_normalization=normalize_std,
        n_samples_per_prompt=len(samples),
        rollout_batch_size=1,
        reward_key="score",
        use_dynamic_global_batch_size=False,
    )
    data = module.convert_samples_to_train_data(args, samples, {}, None, None)
    expected = [value - sum(valid_rewards) / len(valid_rewards) for value in valid_rewards]
    if normalize_std:
        std = statistics.stdev(valid_rewards) if len(valid_rewards) > 1 else 0
        expected = [value / (std + 1e-6) for value in expected]
    assert samples[-1].remove_sample is True
    assert data["rewards"] == pytest.approx(expected + [0])
    assert data["loss_masks"] == [[1, 1] for _ in valid_rewards] + [[0, 0]]


@pytest.mark.parametrize("verifier_reward", [0, 1])
def test_unmarked_full_limit_is_excluded_but_last_turn_submission_and_local_horizon_are_kept(verifier_reward):
    metadata = {"reward": verifier_reward, "agent_turns": 250, "agent_max_turns": 250}
    full = _sample(exit_status="LimitsExceeded", **metadata)
    result = asyncio.run(reward.reward_func(None, full))
    assert full.remove_sample and result["judge_raw"] == "excluded_from_training"

    submitted = _sample(exit_status="Submitted", **metadata)
    result = asyncio.run(reward.reward_func(None, submitted))
    assert not submitted.remove_sample and result["score"] == verifier_reward

    local = _sample(local=True, exit_status="LimitsExceeded", **metadata)
    assert reward._terminal_result(local, local=True) is None
    assert not local.remove_sample


def test_infra_exclusion_and_length_reach_real_grpo_normalization_and_loss_masks(monkeypatch):
    # The conversion itself is CPU-only; stub only unused Ray transport imports.
    monkeypatch.setitem(sys.modules, "ray", ModuleType("ray"))
    ray_utils = ModuleType("miles.utils.ray_utils")
    ray_utils.Box = object
    monkeypatch.setitem(sys.modules, "miles.utils.ray_utils", ray_utils)
    path = Path(__file__).resolve().parents[3] / "miles/ray/rollout/train_data_conversion.py"
    spec = importlib.util.spec_from_file_location("swe_train_conversion_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    samples = [
        _sample(0, reward=1),
        _sample(1, status=Sample.Status.TRUNCATED, agent_last_finish_reason="length", reward=1),
        _sample(2, agent_excluded_from_training=True),
    ]
    results = asyncio.run(reward.reward_func(None, samples))
    for sample, result in zip(samples, results, strict=True):
        sample.reward = result
    args = SimpleNamespace(
        advantage_estimator="grpo",
        rewards_normalization=True,
        grpo_std_normalization=False,
        n_samples_per_prompt=3,
        rollout_batch_size=1,
        reward_key="score",
        use_dynamic_global_batch_size=False,
    )
    data = module.convert_samples_to_train_data(args, samples, {}, None, None)
    assert data["rewards"] == [0.5, -0.5, 0]
    assert data["loss_masks"] == [[1, 1], [1, 1], [0, 0]]
    assert data["truncated"] == [0, 1, 0]
    assert data["sample_indices"] == [0, 1, 2]


@pytest.mark.parametrize("include_length", [False, True])
def test_local_prm_only_judges_valid_samples_and_counts_trainable_variance(monkeypatch, include_length):
    calls = []

    class FakeClient:
        def __init__(self, **kwargs):
            pass

        async def complete_json(self, *args, **kwargs):
            calls.append(kwargs["tag"])
            return kwargs["validate"](
                {
                    "avoid_error_met": True,
                    "behavior_anomaly": False,
                    "redirect_met": True,
                    "score": 1,
                    "evidence_turns": [1],
                    "reason": "tested Y",
                }
            )

    monkeypatch.setattr(reward, "JudgeClient", FakeClient)
    monkeypatch.setattr(reward, "judge_settings", lambda section: {"max_concurrency": 1, "trace_max_chars": 10000})
    samples = [_sample(0, local=True), _sample(1, local=True, agent_function_failed=True)]
    samples[1].remove_sample = True
    if include_length:
        samples.append(_sample(2, local=True, status=Sample.Status.TRUNCATED, agent_last_finish_reason="length"))
    results = asyncio.run(reward.reward_func(None, samples))
    assert len(calls) == 1
    assert [item["score"] for item in results] == ([1, 0, 0] if include_length else [1, 0])
    assert samples[1].remove_sample and samples[1].metadata["agent_excluded_from_training"]
    assert all(sample.metadata["ab_event_local_group_informative"] == include_length for sample in samples)


def test_local_group_uses_zero_half_one_rubric_reward(monkeypatch):
    verdicts = iter(
        [
            {
                "avoid_error_met": False,
                "behavior_anomaly": False,
                "redirect_met": False,
                "score": 0,
                "evidence_turns": [],
                "reason": "continued the wrong assumption",
            },
            {
                "avoid_error_met": True,
                "behavior_anomaly": False,
                "redirect_met": False,
                "score": 0.5,
                "evidence_turns": [1],
                "reason": "abandoned the mistake",
            },
        ]
    )

    class FakeClient:
        def __init__(self, *, config_section):
            assert config_section == "local_prm"

        async def complete_json(self, _system, _prompt, *, required_keys, tag, validate):
            assert required_keys
            assert tag.startswith("swe_local_prm_")
            return validate(next(verdicts))

    monkeypatch.setattr(reward, "JudgeClient", FakeClient)
    monkeypatch.setattr(reward, "judge_settings", lambda section: {"max_concurrency": 8, "trace_max_chars": 10000})
    prefix = [{"role": "system", "content": "system"}, {"role": "user", "content": "issue"}]
    samples = []
    for index in range(2):
        samples.append(
            Sample(
                group_index=1,
                index=index,
                metadata={
                    "ab_local_rollout": True,
                    "agent_max_turns": 10,
                    "ab_local_prefix_message_count": len(prefix),
                    "ab_event_recovery_rubric": {"avoid_error": "stop X", "redirect": "test Y"},
                    "messages": prefix
                    + [
                        {"role": "assistant", "content": "new direction"},
                        {"role": "tool", "content": "evidence"},
                    ],
                },
            )
        )

    results = asyncio.run(reward.reward_func(None, samples))

    assert [result["score"] for result in results] == [0.0, 0.5]
    assert [result["event_rubric_state"] for result in results] == ["avoid_failed", "avoid_only"]
    assert all(sample.metadata["ab_event_local_group_informative"] for sample in samples)


@pytest.mark.parametrize("mode", ["ALL", "Wrong", "None"])
@pytest.mark.parametrize("reason", ["context_reserve", "max_turns"])
@pytest.mark.parametrize("score", [0, 1])
@pytest.mark.parametrize("failure", [None, "missing_verifier", "verifier_error", "LimitsExceeded", "LengthTruncated"])
def test_force_exclude_adapter_reward_and_grpo_masks(monkeypatch, mode, reason, score, failure):
    from adaptive_branching.src.swe import harbor_client

    monkeypatch.setenv("FORCE_EXCLUDE", mode)
    report = {"reward": score}
    if failure == "missing_verifier":
        report = {}
    elif failure == "verifier_error":
        report["infrastructure_error"] = "test verifier failure"
    status = failure if failure in {"LimitsExceeded", "LengthTruncated"} else "Submitted"

    async def fake_post(url, payload):
        return {
            "exit_status": status,
            "reward": score,
            "eval_report": report,
            "agent_metrics": {"agent_forced_final_answer_reason": reason},
        }

    monkeypatch.setattr(harbor_client, "_post_trial", fake_post)
    metadata = asyncio.run(harbor_client.run("http://node:30000/sessions/one", "ignored", metadata={"instance_id": "r2e-pandas-deadbeef"}))
    excluded = failure is not None or mode == "ALL" or (mode == "Wrong" and score == 0)
    assert metadata["agent_excluded_from_training"] is excluded
    assert metadata["reward"] == score
    forced = _sample(2, status=Sample.Status.TRUNCATED if status == "LengthTruncated" else Sample.Status.COMPLETED, **metadata)
    samples = [_sample(0, reward=0), _sample(1, reward=1), forced]
    for sample in samples:
        sample.reward = asyncio.run(reward.reward_func(None, sample))
    assert forced.remove_sample is excluded
    assert forced.reward["score"] == (0 if excluded else score)

    monkeypatch.setitem(sys.modules, "ray", ModuleType("ray"))
    ray_utils = ModuleType("miles.utils.ray_utils")
    ray_utils.Box = object
    monkeypatch.setitem(sys.modules, "miles.utils.ray_utils", ray_utils)
    path = Path(__file__).resolve().parents[3] / "miles/ray/rollout/train_data_conversion.py"
    spec = importlib.util.spec_from_file_location("swe_force_policy_conversion_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    args = SimpleNamespace(
        advantage_estimator="grpo",
        rewards_normalization=True,
        grpo_std_normalization=False,
        n_samples_per_prompt=3,
        rollout_batch_size=1,
        reward_key="score",
        use_dynamic_global_batch_size=False,
    )
    data = module.convert_samples_to_train_data(args, samples, {}, None, None)
    baseline = 0.5 if excluded else (1 + score) / 3
    assert data["rewards"] == pytest.approx([-baseline, 1 - baseline, 0 if excluded else score - baseline])
    assert data["loss_masks"] == [[1, 1], [1, 1], [0, 0] if excluded else [1, 1]]


@pytest.mark.parametrize("mode", ["ALL", "Wrong", "None"])
@pytest.mark.parametrize("reason", ["context_reserve", "max_turns"])
def test_force_exclude_does_not_reenable_local_limits(monkeypatch, mode, reason):
    monkeypatch.setenv("FORCE_EXCLUDE", mode)
    sample = _sample(local=True, agent_forced_final_answer_reason=reason)
    assert reward._terminal_result(sample, local=True)["judge_raw"] == "excluded_from_training"
    assert sample.remove_sample


@pytest.mark.parametrize("mode", ["ALL", "Wrong", "None"])
def test_force_exclude_preserves_normal_full_results(monkeypatch, mode):
    monkeypatch.setenv("FORCE_EXCLUDE", mode)
    for score in [0, 1]:
        sample = _sample(reward=score)
        assert asyncio.run(reward.reward_func(None, sample))["score"] == score
        assert not sample.remove_sample


def test_force_exclude_invalid_runtime_mode_fails_before_trial(monkeypatch):
    from adaptive_branching.src.swe import harbor_client

    async def unexpected(*args, **kwargs):
        pytest.fail("invalid mode must fail before network calls")

    monkeypatch.setenv("FORCE_EXCLUDE", "typo")
    monkeypatch.setattr(harbor_client, "_post_trial", unexpected)
    with pytest.raises(ValueError, match="FORCE_EXCLUDE"):
        asyncio.run(harbor_client.run("http://node:30000/sessions/one", "ignored", metadata={"instance_id": "q"}))
    with pytest.raises(ValueError, match="FORCE_EXCLUDE"):
        asyncio.run(reward.reward_func(None, _sample(reward=1)))
