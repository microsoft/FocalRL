"""Full Lightning reward plus unpenalized local PRM; transport remains Harbor."""

import asyncio
import os

import httpx

from adaptive_branching.src.swe import agent, harbor_client, lightning_reward

HARBOR_AGENT = "adaptive_branching.src.swe.lightning_local_harbor_agent:LocalLightningSweAgent"
LOCAL_STOPS = {"Submitted", "LocalHorizon", "ContextLimit", "LengthTruncated", "FormatLimit"}


def is_local(metadata):
    if not isinstance(metadata, dict):
        raise TypeError("sample metadata must be an object")
    return bool(metadata.get("ab_local_rollout")) or metadata.get("ab_rollout_kind") == "local"


def build_request(base_url, request_kwargs, metadata):
    local = is_local(metadata)
    config = agent.training_config()
    if local:
        request = harbor_client.build_request(base_url=base_url, request_kwargs=request_kwargs, metadata=metadata)
        horizon = metadata.get("agent_max_turns")
        if type(horizon) is not int or horizon not in (10, 20, 30):
            raise ValueError("local training requires integer H=10/20/30")
        prefix = metadata.get("swe_replay_n_calls")
        if type(prefix) is not int or not 0 <= prefix < config.max_turns:
            raise ValueError("invalid replay prefix call count")
        request.update(
            max_seq_len=config.model_context,
            context_reserve_tokens=None,
            sampling_params={
                "temperature": 1.0,
                "max_tokens": config.max_tokens,
                "chat_template_kwargs": {"enable_thinking": True},
            },
        )
    else:
        request = agent.build_request(base_url, request_kwargs, metadata)
    request["agent_name"] = HARBOR_AGENT
    if config.model_context == 262144:
        request["agent_name"] = "adaptive_branching.src.swe.lightning_local_harbor_agent:LargeLocalLightningSweAgent"
    return request


async def decode_local(response, metadata, prompt):
    if not isinstance(response, dict) or not is_local(metadata):
        raise ValueError("local response requires local request metadata")
    metrics = response.get("agent_metrics")
    status = response.get("exit_status")
    if not isinstance(metrics, dict) or not isinstance(status, str):
        raise ValueError("local response missing metrics/status")
    total = metrics.get("turns", 0)
    prefix = metadata["swe_replay_n_calls"]
    horizon = metadata.get("agent_max_turns")
    if type(horizon) is not int or horizon not in (10, 20, 30):
        raise ValueError("local response requires integer H=10/20/30")
    if (
        type(total) is not int
        or total < 0
        or type(prefix) is not int
        or not 0 <= prefix < agent.training_config().max_turns
    ):
        raise ValueError("invalid local turn telemetry")
    # Infrastructure failure may happen before restoration: do not subtract a
    # historical count from an uninitialized episode or manufacture a sample.
    if status not in LOCAL_STOPS:
        return {**harbor_client._failed_trial(status), "trial_dir": response.get("trial_dir")}
    generated = total - prefix
    if not 0 <= generated <= horizon:
        raise ValueError(f"local generated turns outside H={horizon}: total={total}, prefix={prefix}")
    trial = response.get("trial_dir")
    if not isinstance(trial, str) or not trial:
        raise ValueError("local result missing trial_dir")
    messages = await harbor_client.get_trial_messages(trial)
    if not isinstance(prompt, list) or messages[: len(prompt)] != prompt:
        raise ValueError("restored local conversation differs from collector prefix")
    continuation = messages[len(prompt) :]
    if sum(m.get("role") == "assistant" for m in continuation) != generated:
        raise ValueError("local continuation turn telemetry differs from messages")
    if status == "LocalHorizon" and generated != horizon:
        raise ValueError(f"LocalHorizon without {horizon} generated turns")
    return {
        **harbor_client._promoted_tool_metrics(metrics),
        **harbor_client._harbor_request_metrics(),
        "reward": 0.0,
        "exit_status": status,
        "agent_metrics": metrics,
        "agent_turns": generated,
        "agent_excluded_from_training": generated == 0,
        "agent_function_failed": False,
        "agent_function_error": None,
        "agent_max_turns_hit": False,
        "agent_context_limit_hit": status == "ContextLimit",
        "agent_context_reserve_hit": False,
        "agent_last_finish_reason": "length" if status == "LengthTruncated" else None,
        "trial_dir": trial,
        "messages": messages,
        "swe_local_prm_only": True,
    }


async def run(base_url, prompt, request_kwargs=None, metadata=None, **_):
    metadata = metadata or {}
    request = build_request(base_url, request_kwargs or {}, metadata)
    try:
        response = await harbor_client._run_trial(request)
    except (httpx.HTTPError, asyncio.TimeoutError) as exc:
        return {**harbor_client._failed_trial(type(exc).__name__), **harbor_client._harbor_request_metrics(exc)}
    return await decode_local(response, metadata, prompt) if is_local(metadata) else agent.decode_result(response)


async def reward_func(args, samples, **_):
    from adaptive_branching.src.swe.reward import _score_local_group
    from adaptive_branching.src.swe.value_cliff_online import annotate_swe_value_cliff_branches

    batch = samples if isinstance(samples, list) else [samples]
    if not batch or len({is_local(s.metadata) for s in batch}) != 1:
        raise ValueError("reward requires a nonempty homogeneous Full/Local group")
    if is_local(batch[0].metadata):
        for sample in batch:
            if sample.metadata.get("swe_local_prm_only") is not True and not sample.remove_sample:
                if not sample.metadata.get("agent_excluded_from_training") and sample.status.name != "ABORTED":
                    raise ValueError("local sample missing audited continuation marker")
        results = await _score_local_group(batch, unpenalized=True)
    else:
        results = [lightning_reward.score_sample(s) for s in batch]
        await annotate_swe_value_cliff_branches(args, batch, results)
    return results if isinstance(samples, list) else results[0]
