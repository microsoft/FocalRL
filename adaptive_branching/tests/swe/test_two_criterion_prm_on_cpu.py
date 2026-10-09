import asyncio
import hashlib

import pytest

from adaptive_branching.src.swe import lightning_local, reward
from adaptive_branching.src.swe.value_cliff_locator import (
    SWE_TWO_CRITERION_PRM_VERSION,
    prm_behavior_veto_enabled,
    swe_two_criterion_prm_system,
)
from adaptive_branching.tests.swe.test_behavior_guard_on_cpu import GOOD, make_sample
from miles.utils.types import Sample


@pytest.mark.parametrize("value,expected", [("0", False), ("1", True), (None, True)])
def test_behavior_toggle(monkeypatch, value, expected):
    if value is None:
        monkeypatch.delenv("SWE_PRM_BEHAVIOR_VETO", raising=False)
    else:
        monkeypatch.setenv("SWE_PRM_BEHAVIOR_VETO", value)
    assert prm_behavior_veto_enabled() is expected


@pytest.mark.parametrize("value", ["", "false", "true", "2", " 0"])
def test_behavior_toggle_rejects_invalid(monkeypatch, value):
    monkeypatch.setenv("SWE_PRM_BEHAVIOR_VETO", value)
    with pytest.raises(ValueError, match="exactly 0 or 1"):
        prm_behavior_veto_enabled()


def test_original_h20_prompt_matches_archived_experiment():
    prompt = swe_two_criterion_prm_system(local_turns=20)
    assert (
        hashlib.sha256(prompt.encode()).hexdigest()
        == "7d95791a96d853e12f59d4539662966b1992251166cb2c99f3f9562e32ef3774"
    )
    assert "behavior_anomaly" not in prompt
    assert "most 1 local assistant turns" in swe_two_criterion_prm_system(local_turns=1)


@pytest.mark.parametrize("turns", [0, -1, True, None, 1.5])
def test_original_prompt_rejects_invalid_horizon(turns):
    with pytest.raises(ValueError):
        swe_two_criterion_prm_system(local_turns=turns)


@pytest.mark.parametrize(
    "avoid,redirect,score", [(False, False, 0), (False, True, 0), (True, False, 0.5), (True, True, 1)]
)
def test_original_prm_schema_and_reward(monkeypatch, avoid, redirect, score):
    monkeypatch.setenv("SWE_PRM_BEHAVIOR_VETO", "0")

    class Judge:
        def __init__(self, **kwargs):
            assert kwargs["config_section"] == "local_prm"

        async def complete_json(self, system, prompt, **kwargs):
            assert "behavior_anomaly" not in system
            assert "behavior_anomaly" not in kwargs["required_keys"]
            raw = dict(
                avoid_error_met=avoid, redirect_met=redirect, score=score, evidence_turns=[1], reason="evidence"
            )
            with pytest.raises(ValueError):
                kwargs["validate"]({**raw, "behavior_anomaly": True})
            return kwargs["validate"](raw)

    monkeypatch.setattr(reward, "JudgeClient", Judge)
    monkeypatch.setattr(reward, "judge_settings", lambda _: {"max_concurrency": 1, "trace_max_chars": 10000})
    sample = make_sample([(GOOD, 1)])
    sample.metadata["agent_behavior_anomaly_penalized"] = True
    result = asyncio.run(lightning_local.reward_func(None, sample))
    assert result["score"] == score
    assert sample.metadata["ab_event_rubric_judge_version"] == SWE_TWO_CRITERION_PRM_VERSION
    assert sample.metadata["agent_behavior_anomaly_penalized"] is False
    assert not sample.remove_sample and all(sample.loss_mask)


@pytest.mark.parametrize("failure", ["strict_format", "single_call_length"])
def test_format_and_length_remain_trainable_zero_without_prm(monkeypatch, failure):
    monkeypatch.setenv("SWE_PRM_BEHAVIOR_VETO", "0")
    monkeypatch.setattr(reward, "JudgeClient", lambda **_: pytest.fail("terminal veto must not call PRM"))
    sample = make_sample([("broken format" if failure == "strict_format" else GOOD, 1)])
    if failure == "single_call_length":
        sample.status = Sample.Status.TRUNCATED
        sample.metadata.update(exit_status="LengthTruncated", agent_last_finish_reason="length")
    mask = sample.loss_mask[:]
    result = asyncio.run(lightning_local.reward_func(None, sample))
    assert result["score"] == 0 and result["judge_raw"] == failure
    assert not sample.remove_sample and sample.loss_mask == mask
    assert sample.metadata["ab_local_trainable"] is True
