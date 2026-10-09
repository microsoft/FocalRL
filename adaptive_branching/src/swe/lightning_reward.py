"""Pinned Lightning reward shaping, separate from legacy SWE remove semantics."""

from adaptive_branching.src.swe import lightning_reference as reference
from adaptive_branching.src.swe.agent import training_config
from adaptive_branching.src.swe.behavior_guard import hard_failure_reason


def score_sample(sample):
    metadata = sample.metadata
    if not isinstance(metadata, dict):
        raise TypeError("sample.metadata must be an object")
    excluded = bool(
        sample.remove_sample
        or metadata.get("agent_excluded_from_training")
        or metadata.get("agent_function_failed")
        or sample.status.name == "ABORTED"
    )
    if excluded:
        sample.remove_sample = True
        return {
            "score": 0.0,
            "acc": False,
            "pred": metadata.get("exit_status", ""),
            "judge_raw": "excluded_from_training",
        }
    raw = metadata.get("reward")
    metrics = metadata.get("agent_metrics", {})
    turns = metrics.get("turns")
    prompt_tokens = metrics.get("agent_lightning_max_prompt_tokens")
    if type(raw) not in (int, float) or raw not in (0, 1):
        raise ValueError(f"missing binary verifier reward: {raw!r}")
    if (
        type(turns) is not int
        or not 1 <= turns <= training_config().max_turns
        or type(prompt_tokens) is not int
        or prompt_tokens <= 0
    ):
        raise ValueError(f"invalid Lightning turns/prompt tokens: {turns!r}/{prompt_tokens!r}")
    evaluation = metadata.get("swe_lightning_evaluation")
    if type(evaluation) is not bool:
        raise ValueError("Lightning generate wrapper must set swe_lightning_evaluation")
    failure = hard_failure_reason(sample)
    if failure is not None:
        metadata["agent_lightning_raw_reward"] = raw
        metadata["agent_lightning_shaping_penalty"] = 0.0 if failure == "single_call_length" else float(raw)
        metadata["agent_format_failure_penalized"] = failure != "single_call_length"
        metadata["agent_single_call_length_penalized"] = failure == "single_call_length"
        return {"score": 0.0, "acc": False, "pred": metadata.get("exit_status", ""), "judge_raw": failure}
    # 100 is the penalty saturation point, NOT the runtime turn limit.
    # Keep the original 80–100 ramp even when the agent can run for 200 turns.
    score = reference.length_penalized_reward(raw, turns, 100, t0=80, lam=0.1, is_train=not evaluation)
    score = reference.prompt_length_penalty(
        score,
        prompt_tokens,
        soft_start=50000,
        hard_cap=64000,
        max_pen=0.1,
        is_train=not evaluation,
        solved=raw == 1,
    )
    metadata["agent_lightning_raw_reward"] = raw
    metadata["agent_lightning_shaping_penalty"] = raw - score
    return {
        "score": score,
        "acc": raw == 1,
        "pred": metadata.get("exit_status", ""),
        "judge_raw": "harbor_verifier_lightning_shaping",
    }


async def reward_func(args, samples, **kwargs):
    batch = samples if isinstance(samples, list) else [samples]
    if not batch:
        raise ValueError("samples must be nonempty")
    results = [score_sample(sample) for sample in batch]
    return results if isinstance(samples, list) else results[0]
