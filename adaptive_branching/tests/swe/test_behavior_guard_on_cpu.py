import asyncio
from types import SimpleNamespace

import pytest

from adaptive_branching.src.swe import behavior_guard as guard, lightning_local, lightning_reward, reward
from adaptive_branching.src.swe.value_cliff_locator import swe_value_cliff_prm_system, validate_swe_prm_verdict
from adaptive_branching.tests.swe.test_reward_on_cpu import _sample
from miles.utils.types import Sample


GOOD = "</think><tool_call><function=bash><parameter=command>echo ok</parameter></function></tool_call><|im_end|>"


@pytest.fixture(autouse=True)
def strict_policy(monkeypatch):
    monkeypatch.delenv("SWE_REWARD_POLICY", raising=False)


class CharTokenizer:
    def batch_decode(self, spans, **kwargs):
        assert kwargs == {"skip_special_tokens": False, "clean_up_tokenization_spaces": False}
        return ["".join(map(chr, ids)) for ids in spans]


def make_sample(parts, *, prefix="", local=True):
    sample = _sample(local=local, swe_local_prm_only=True, exit_status="LocalHorizon" if local else "Submitted")
    text = "".join(text for text, active in parts)
    sample.tokens = list(map(ord, prefix + text))
    sample.response_length = len(text)
    sample.loss_mask = [active for text, active in parts for _ in text]
    sample.rollout_log_probs = [-0.1] * len(text)
    sample.response = text
    sample.metadata.update(
        reward=1,
        swe_lightning_evaluation=False,
        agent_metrics={"turns": 1, "agent_lightning_max_prompt_tokens": 10},
    )
    guard.audit_generated_format(sample, CharTokenizer())
    return sample


@pytest.mark.parametrize(
    "text", ["</function></function>", "</function>\n</function>", "</function> \t\n</function>" * 100]
)
@pytest.mark.parametrize("local", [False, True])
def test_repeated_raw_generated_closers_are_trainable_zero_without_judge(monkeypatch, text, local):
    def unexpected(*args, **kwargs):
        pytest.fail("hard veto must not initialize the PRM")

    monkeypatch.setattr(reward, "JudgeClient", unexpected)
    monkeypatch.setattr(reward, "judge_settings", unexpected)
    sample = make_sample([(text, 1)], local=local)
    masks = sample.loss_mask[:]
    # Parsed assistant text need not preserve malformed raw closers.
    sample.response = "canonical parsed content without duplicate tags"
    result = (
        asyncio.run(lightning_local.reward_func(None, [sample]))[0] if local else lightning_reward.score_sample(sample)
    )
    assert result["score"] == 0 and result["judge_raw"] == "strict_format"
    assert sample.loss_mask == masks and not sample.remove_sample


@pytest.mark.parametrize(
    "parts,prefix",
    [
        ([("ok", 1)], "</function>\n</function>"),
        ([("ok", 1), ("</function>\n</function>", 0), ("done", 1)], ""),
        ([("</function>", 1), ("observation", 0), ("</function>", 1)], ""),
        ([("</function>\n</tool_call>\n<tool_call><function=bash></function>", 1)], ""),
        ([("x", 0)], ""),
        ([], "prompt"),
        ([("x", 1)], ""),
    ],
)
def test_audit_ignores_prefix_tool_output_and_separate_calls(parts, prefix):
    sample = make_sample(parts, prefix=prefix)
    # Only the diagnostic counter is under test: malformed spans still fail
    # the strict grammar, and empty generations are rejected separately.
    assert guard.hard_failure_reason(sample) in {"strict_format", "empty_generated_span"}
    assert not sample.metadata["agent_repeated_function_close"]


@pytest.mark.parametrize("bad", [None, [1], [1, 2], [1, -1]])
def test_audit_rejects_invalid_masks(bad):
    sample = _sample()
    sample.loss_mask = bad
    with pytest.raises(ValueError, match="loss_mask"):
        guard.audit_generated_format(sample, CharTokenizer())


def test_audit_rejects_invalid_lengths_decoder_and_missing_audit():
    sample = _sample()
    sample.response_length = 4
    with pytest.raises(ValueError, match="response_length"):
        guard.audit_generated_format(sample, CharTokenizer())
    sample.response_length = 2
    with pytest.raises(TypeError, match="tokenizer"):
        guard.audit_generated_format(sample, None)
    with pytest.raises(ValueError, match="decoder"):
        guard.audit_generated_format(sample, SimpleNamespace(batch_decode=lambda *a, **kw: []))
    sample.metadata.pop(guard.AUDIT_KEY)
    with pytest.raises(ValueError, match="audit required"):
        guard.hard_failure_reason(sample)
    sample.metadata[guard.AUDIT_KEY] = {"version": guard.AUDIT_VERSION, "spans": 0, "repeated_spans": 1}
    with pytest.raises(ValueError, match="invalid generated-format audit"):
        guard.hard_failure_reason(sample)


def test_new_continuation_clears_inherited_penalty_telemetry():
    sample = make_sample([("ok", 1)])
    flags = (
        "agent_format_failure_penalized",
        "agent_single_call_length_penalized",
        "agent_behavior_anomaly_penalized",
    )
    sample.metadata.update({key: True for key in flags})
    guard.audit_generated_format(sample, CharTokenizer())
    assert all(sample.metadata[key] is False for key in flags)


@pytest.mark.parametrize("size", [1, 8])
@pytest.mark.parametrize("status", ["LengthTruncated", "LocalHorizon"])
def test_local_length_is_trainable_zero_without_prm(monkeypatch, size, status):
    def unexpected(*args, **kwargs):
        pytest.fail("length veto must not call PRM")

    monkeypatch.setattr(reward, "JudgeClient", unexpected)
    monkeypatch.setattr(reward, "judge_settings", unexpected)
    samples = [
        _sample(
            i,
            local=True,
            swe_local_prm_only=True,
            exit_status=status,
            agent_last_finish_reason="length",
            status=Sample.Status.TRUNCATED,
        )
        for i in range(size)
    ]
    results = asyncio.run(lightning_local.reward_func(None, samples))
    for sample, result in zip(samples, results, strict=True):
        assert result["score"] == 0 and result["judge_raw"] == "single_call_length"
        assert not sample.remove_sample and sample.loss_mask == [1, 1]
        assert sample.metadata["ab_local_trainable"] and sample.metadata["agent_single_call_length_penalized"]


@pytest.mark.parametrize("avoid", [False, True])
@pytest.mark.parametrize("redirect", [False, True])
@pytest.mark.parametrize("anomaly", [False, True])
def test_prm_behavior_veto_is_independent_of_rubric(avoid, redirect, anomaly):
    score = 0 if anomaly or not avoid else (1 if redirect else 0.5)
    raw = dict(
        avoid_error_met=avoid,
        redirect_met=redirect,
        behavior_anomaly=anomaly,
        score=score,
        evidence_turns=[1],
        reason="local evidence",
    )
    out = validate_swe_prm_verdict(raw, local_turns=1)
    assert out == raw
    if anomaly:
        with pytest.raises(ValueError, match="requires score=0"):
            validate_swe_prm_verdict({**raw, "score": 1}, local_turns=1)


@pytest.mark.parametrize(
    "patch",
    [
        {"behavior_anomaly": "true"},
        {"score": True},
        {"evidence_turns": []},
        {"evidence_turns": [2]},
        {"reason": ""},
        {"avoid_error_met": 1},
    ],
)
def test_prm_rejects_invalid_behavior_verdict(patch):
    raw = dict(
        avoid_error_met=False,
        redirect_met=False,
        behavior_anomaly=True,
        score=0,
        evidence_turns=[1],
        reason="repeated calls",
    )
    with pytest.raises((TypeError, ValueError)):
        validate_swe_prm_verdict({**raw, **patch}, local_turns=1)
    raw.pop("behavior_anomaly")
    with pytest.raises(ValueError, match="exactly"):
        validate_swe_prm_verdict(raw, local_turns=1)


def test_semantic_veto_reaches_actual_local_reward_and_preserves_mask(monkeypatch):
    class Judge:
        def __init__(self, **kwargs):
            pass

        async def complete_json(self, system, prompt, **kwargs):
            assert "behavior_anomaly" in kwargs["required_keys"]
            assert "normal test reruns after edits" in system
            return kwargs["validate"](
                dict(
                    avoid_error_met=True,
                    redirect_met=True,
                    behavior_anomaly=True,
                    score=0,
                    evidence_turns=[1],
                    reason="mass-duplicated calls",
                )
            )

    monkeypatch.setattr(reward, "JudgeClient", Judge)
    monkeypatch.setattr(reward, "judge_settings", lambda _: {"max_concurrency": 1, "trace_max_chars": 10000})
    sample = make_sample([(GOOD, 1)])
    result = asyncio.run(lightning_local.reward_func(None, sample))
    assert result["score"] == 0 and sample.metadata["agent_behavior_anomaly_penalized"]
    assert not sample.remove_sample and all(sample.loss_mask)
    assert sample.metadata["ab_event_rubric_judge"]["redirect_met"] is True


def test_prompt_does_not_punish_prefix_retries_or_real_tool_output():
    prompt = swe_value_cliff_prm_system(local_turns=20)
    for text in [
        "NEW LOCAL CONTINUATION ONLY",
        "normal test reruns after edits",
        "justified retries",
        "real tool output",
        "mass-duplicated tool",
        "assistant-fabricated",
        "overrides",
    ]:
        assert text in prompt or (text == "overrides" and "override" in prompt)
    with pytest.raises(ValueError):
        swe_value_cliff_prm_system(local_turns=0)
