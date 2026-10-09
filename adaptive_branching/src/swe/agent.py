"""Native-tool ReAct loop derived from the pinned Agent Lightning SWE-smith agent.

Harbor owns sandbox lifetime and isolated grading; Miles owns token tracing.
This module is CPU-only and the loop accepts async model/shell callables.
"""

import asyncio
import copy
import hashlib
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any

import httpx

from adaptive_branching.src.swe import lightning_reference as reference
from adaptive_branching.src.swe import thinking_protocol as protocol
from adaptive_branching.src.swe.tool_metrics import extract_swe_tool_metrics

HARBOR_AGENT = "adaptive_branching.src.swe.lightning_harbor_agent:LightningSweAgent"
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AgentConfig:
    max_turns: int = 100
    max_tokens: int = 12288
    observation_chars: int = 6000
    command_timeout: int = 120
    max_format_errors: int = 3
    model_context: int = 81920

    def __post_init__(self):
        for name, value in vars(self).items():
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer, got {value!r}")
        if self.max_tokens >= self.model_context:
            raise ValueError("max_tokens must be smaller than model_context")
        if self.observation_chars < 2:
            raise ValueError("observation_chars must be at least 2")


class ContextOverflow(Exception):
    """A confirmed model context budget error, not an infrastructure failure."""


def training_config(budget: str | None = None) -> AgentConfig:
    """Select an explicit RL runtime budget, independently of reward shaping."""
    budget = os.environ.get("SWE_LIGHTNING_BUDGET", "standard") if budget is None else budget
    if budget == "standard":
        return AgentConfig()
    if budget == "large":
        return AgentConfig(max_turns=200, model_context=262144)
    raise ValueError(f"unknown SWE_LIGHTNING_BUDGET: {budget!r}")


@dataclass
class Episode:
    messages: list[dict[str, Any]] = field(default_factory=list)
    tool_events: list[dict[str, Any]] = field(default_factory=list)
    turns: int = 0
    max_prompt_tokens: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    format_errors: int = 0
    blocked_actions: int = 0
    stop_reason: str = "Running"
    command_results: list[dict[str, Any]] = field(default_factory=list)

    def metrics(self) -> dict[str, Any]:
        if self.turns < 0 or self.max_prompt_tokens < 0:
            raise ValueError("invalid episode counters")
        return {
            **extract_swe_tool_metrics(self.tool_events),
            "turns": self.turns,
            "agent_exit_status": self.stop_reason,
            "agent_lightning_max_prompt_tokens": self.max_prompt_tokens,
            "agent_lightning_format_errors": self.format_errors,
            "agent_lightning_blocked_actions": self.blocked_actions,
            "agent_lightning_submitted": self.stop_reason == "Submitted",
            "agent_max_turns_hit": self.stop_reason == "TurnLimit",
            "agent_context_limit_hit": self.stop_reason == "ContextLimit",
            "agent_context_reserve_hit": False,
            "agent_last_finish_reason": "length" if self.stop_reason == "LengthTruncated" else None,
        }


def validate_completion(body: dict[str, Any]) -> tuple[str, str, str, int, int]:
    if not isinstance(body, dict) or not isinstance(body.get("choices"), list) or len(body["choices"]) != 1:
        raise ValueError("model response must contain exactly one choice")
    choice = body["choices"][0]
    if not isinstance(choice, dict) or not isinstance(choice.get("message"), dict):
        raise ValueError("model choice must contain a message object")
    content = choice["message"].get("content")
    if content is None:
        content = ""
    finish = choice.get("finish_reason")
    usage = body.get("usage", {})
    if not isinstance(usage, dict):
        raise ValueError("model usage must be an object")
    prompt_tokens, output_tokens = usage.get("prompt_tokens"), usage.get("completion_tokens")
    if not isinstance(content, str) or finish not in {"stop", "length", "tool_calls"}:
        raise ValueError(f"invalid text completion/finish_reason: {finish!r}")
    reasoning = choice["message"].get("reasoning_content")
    if reasoning is None:
        reasoning = ""
    if not isinstance(reasoning, str):
        raise ValueError("reasoning_content must be text or null")
    calls = choice["message"].get("tool_calls")
    if calls is not None and finish != "length":
        if not isinstance(calls, list):
            raise ValueError("tool_calls must be a list or null")
        ids = []
        for call in calls:
            if not isinstance(call, dict) or not isinstance(call.get("id"), str) or not call["id"].strip():
                raise ValueError("serving tool calls require nonempty IDs for matching tool results")
            ids.append(call["id"])
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate tool call IDs in completion")
    if content.lstrip().startswith("<think>"):
        raise ValueError("unparsed thinking in content; configure the qwen3 reasoning parser")
    if type(prompt_tokens) is not int or prompt_tokens <= 0 or type(output_tokens) is not int or output_tokens < 0:
        raise ValueError(f"missing/invalid model usage: {usage!r}")
    return content, reasoning, finish, prompt_tokens, output_tokens


async def query_model(client, url: str, model: str, messages: list, config: AgentConfig) -> dict:
    if not isinstance(url, str) or not url.startswith(("https://", "http://")) or not model or not messages:
        raise ValueError("model query requires an absolute URL, model and nonempty messages")
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": config.max_tokens,
        "temperature": 1.0,
        "tools": protocol.TOOLS,
        "tool_choice": "auto",
        "parallel_tool_calls": True,
        "chat_template_kwargs": {"enable_thinking": True},
    }
    # Retry the same model request, not the episode or any shell action. Keep
    # weight-sync pauses independent from the three transient-502 retries.
    gateway_pauses = 0
    retries_502 = 0
    for _ in range(124):
        response = await client.post(url, json=payload)
        if response.status_code == 429 and "gateway paused" in response.text.lower() and gateway_pauses < 120:
            gateway_pauses += 1
            await asyncio.sleep(5)
            continue
        if response.status_code == 502 and retries_502 < 3:
            delay = 0.5 * (2**retries_502)
            retries_502 += 1
            logger.warning("SWE model HTTP 502; retry %d/3 in %.1fs: %s", retries_502, delay, response.text[:200])
            await asyncio.sleep(delay)
            continue
        if response.status_code == 400 and (
            any(marker in response.text.lower() for marker in ("maximum context length", "'max_tokens' is too large"))
            or re.search(
                r"the input \([\d,]+ tokens\) is longer than the model's context length \([\d,]+ tokens\)",
                response.text,
                re.IGNORECASE,
            )
        ):
            raise ContextOverflow("model rejected prompt for exceeding its context budget")
        response.raise_for_status()
        body = response.json()
        validate_completion(body)
        return body
    raise AssertionError("unreachable gateway retry state")


async def run_episode(
    problem: str,
    query,
    execute,
    *,
    config: AgentConfig | None = None,
    episode=None,
    resume: bool = False,
    local_horizon: int | None = None,
    before_turn=None,
) -> Episode:
    if not isinstance(problem, str) or not problem.strip() or not callable(query) or not callable(execute):
        raise ValueError("run_episode requires a nonempty problem and model/shell callables")
    config = config or AgentConfig()
    episode = episode or Episode()
    if type(resume) is not bool or (before_turn is not None and not callable(before_turn)):
        raise TypeError("resume must be bool and before_turn must be callable or None")
    if local_horizon is not None and (type(local_horizon) is not int or not 1 <= local_horizon <= config.max_turns):
        raise ValueError("local_horizon must be within the full turn budget")
    if resume != (local_horizon is not None):
        raise ValueError("resume requires an explicit local_horizon, and vice versa")
    if not resume and (episode.messages or episode.turns):
        raise ValueError("resume/local branching is not supported by this aligned agent")
    if resume:
        from adaptive_branching.src.swe.lightning_replay import validate_prefix

        validate_prefix(episode.messages, episode.turns, problem=problem, max_turns=config.max_turns)
        if (
            episode.stop_reason != "Running"
            or episode.tool_events
            or episode.command_results
            or any(
                type(value) is not int or value != 0
                for value in (
                    episode.max_prompt_tokens,
                    episode.input_tokens,
                    episode.output_tokens,
                    episode.format_errors,
                    episode.blocked_actions,
                )
            )
        ):
            raise ValueError("resumed episode must have fresh continuation telemetry")
        episode.messages = copy.deepcopy(episode.messages)
    else:
        episode.messages = [
            {"role": "system", "content": protocol.SYSTEM_PROMPT},
            {"role": "user", "content": protocol.INSTANCE_PROMPT.format(problem_statement=problem)},
        ]
    consecutive_errors = 0
    # Restore consecutive format errors too: a branch cannot erase runtime state.
    if resume:
        for message in reversed(episode.messages[2:]):
            if message.get("role") == "assistant":
                if message.get("tool_calls"):
                    break
                consecutive_errors += 1
        if consecutive_errors >= config.max_format_errors:
            raise ValueError("cannot resume after terminal format errors")
    for _ in range(local_horizon if resume else config.max_turns):
        if before_turn is not None:
            await before_turn(episode)
        try:
            body = await query(episode.messages)
        except ContextOverflow:
            episode.stop_reason = "ContextLimit"
            return episode
        content, reasoning, finish, prompt_tokens, output_tokens = validate_completion(body)
        if prompt_tokens + output_tokens > config.model_context:
            raise ValueError(
                f"turn={episode.turns + 1}: serving returned prompt={prompt_tokens} + output={output_tokens} "
                f"exceeding model_context={config.model_context}"
            )
        episode.turns += 1
        episode.max_prompt_tokens = max(episode.max_prompt_tokens, prompt_tokens)
        episode.input_tokens += prompt_tokens
        episode.output_tokens += output_tokens
        # Preserve original reasoning and tool calls for TITO history matching.
        # Neither response prose nor reasoning is interpreted as shell code.
        assistant_message = {"role": "assistant", "content": content}
        if "reasoning_content" in body["choices"][0]["message"]:
            assistant_message["reasoning_content"] = (
                reasoning if body["choices"][0]["message"]["reasoning_content"] is not None else None
            )
        raw_message = body["choices"][0]["message"]
        if "tool_calls" in raw_message:
            assistant_message["tool_calls"] = raw_message["tool_calls"]
        calls = raw_message.get("tool_calls") or []
        episode.messages.append(assistant_message)
        # A length-truncated output is terminal, even if a partial tool call
        # parsed successfully. Keep its tokens for zero-reward training; never
        # execute its actions or create another turn after this TITO record.
        if finish == "length":
            episode.stop_reason = "LengthTruncated"
            return episode
        if not calls:
            consecutive_errors += 1
            episode.format_errors += 1
            episode.messages.append({"role": "user", "content": protocol.format_error_message(0, finish)})
            if consecutive_errors >= config.max_format_errors:
                episode.stop_reason = "FormatLimit"
                return episode
            continue
        try:
            # Validate the whole batch before performing any side effects.
            actions = [protocol.parse_action([call]) for call in calls]
        except ValueError as exc:
            # Malformed arguments cannot reliably be replayed by the chat template.
            # This is a model format failure, graded normally, not infrastructure.
            episode.format_errors += 1
            feedback = "No calls in this batch were executed. " + str(exc)
            for call in calls:
                episode.messages.append({"role": "tool", "tool_call_id": call["id"], "content": feedback})
            episode.stop_reason = "FormatLimit"
            return episode
        consecutive_errors = 0
        for call_index, (call, action) in enumerate(zip(calls, actions, strict=True)):
            call_id = call["id"]
            reason = reference._forbidden_action(action)
            if reason is not None:
                episode.blocked_actions += 1
                episode.messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call_id,
                        "content": reference.render_observation(1, reason, config.observation_chars),
                    }
                )
                continue
            episode.tool_events.append({"role": "assistant", "extra": {"actions": [{"tool_call_id": call_id}]}})
            # An execution transport exception must escape the loop. A returned
            # nonzero shell status is a legitimate observation, not an infra failure.
            output, code = await execute(action, config.command_timeout)
            if not isinstance(output, str) or type(code) is not int:
                raise TypeError(f"invalid shell result for turn {episode.turns}: {type(output)}, {code!r}")
            episode.command_results.append(
                {
                    "turn": episode.turns,
                    "tool_call_id": call_id,
                    "command": action,
                    "returncode": code,
                    "output_sha256": hashlib.sha256(output.encode()).hexdigest(),
                }
            )
            episode.messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": reference.render_observation(code, output, config.observation_chars),
                }
            )
            if reference.is_submission(output):
                episode.stop_reason = "Submitted"
                episode.tool_events.append({"role": "exit", "extra": {"exit_status": "Submitted"}})
                for skipped in calls[call_index + 1 :]:
                    episode.messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": skipped["id"],
                            "content": "Not executed: the task was already submitted by an earlier tool call.",
                        }
                    )
                return episode
            episode.tool_events.append(
                {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "extra": {
                        "returncode": code,
                        "exception_info": (
                            f"Command timed out after {config.command_timeout} seconds" if code == 124 else ""
                        ),
                    },
                }
            )
    episode.stop_reason = "LocalHorizon" if resume else "TurnLimit"
    return episode


def build_request(base_url: str, request_kwargs: dict, metadata: dict) -> dict:
    from adaptive_branching.src.swe import harbor_client

    if not isinstance(metadata, dict) or not isinstance(request_kwargs, dict):
        raise TypeError("metadata and request_kwargs must be objects")
    if metadata.get("ab_local_rollout") or metadata.get("ab_rollout_kind") == "local":
        raise ValueError("Lightning alignment supports full rollouts only")
    request = harbor_client.build_request(base_url=base_url, request_kwargs=request_kwargs, metadata=metadata)
    config = training_config()
    request.update(
        agent_name=HARBOR_AGENT,
        max_turns=config.max_turns,
        max_seq_len=config.model_context,
        context_reserve_tokens=None,
        force_submit_on_limit=False,
        sampling_params={
            "temperature": 1.0,
            "max_tokens": config.max_tokens,
            "chat_template_kwargs": {"enable_thinking": True},
        },
    )
    if config.model_context == 262144:
        request["agent_name"] = "adaptive_branching.src.swe.lightning_harbor_agent:LargeLightningSweAgent"
    return request


def decode_result(response: dict) -> dict:
    from adaptive_branching.src.swe import harbor_client

    if not isinstance(response, dict):
        raise TypeError("Harbor response must be an object")
    reward = response.get("reward")
    if type(reward) not in (float, int) or reward not in (0, 1):
        raise ValueError(f"expected binary verifier reward, got {reward!r}")
    status = response.get("exit_status")
    metrics = response.get("agent_metrics")
    report = response.get("eval_report")
    if not isinstance(status, str) or not status or not isinstance(metrics, dict) or not isinstance(report, dict):
        raise ValueError("Harbor result missing status, metrics or verifier report")
    trainable = status in {"Submitted", "TurnLimit", "ContextLimit", "FormatLimit", "LengthTruncated"}
    if trainable:
        trainable = harbor_client._has_matching_verifier_reward(report, float(reward))
        if type(metrics.get("turns")) is not int or not 0 < metrics["turns"] <= training_config().max_turns:
            raise ValueError("completed Lightning trial has invalid turns")
        if type(metrics.get("agent_lightning_max_prompt_tokens")) is not int:
            raise ValueError("completed Lightning trial omitted prompt token telemetry")
    return {
        **harbor_client._promoted_tool_metrics(metrics),
        **harbor_client._harbor_request_metrics(),
        "reward": float(reward),
        "exit_status": status,
        "eval_report": report,
        "agent_metrics": metrics,
        "agent_turns": metrics.get("turns", 0),
        "agent_excluded_from_training": not trainable,
        "agent_function_failed": False,
        "agent_function_error": None,
        "agent_max_turns_hit": status == "TurnLimit",
        "agent_context_limit_hit": status == "ContextLimit",
        "agent_context_reserve_hit": False,
        "agent_forced_final_answer_reason": None,
        "agent_last_finish_reason": "length" if status == "LengthTruncated" else None,
        "trial_dir": response.get("trial_dir"),
        "trial_id": response.get("trial_id"),
    }


async def run(base_url: str, prompt: Any, request_kwargs=None, metadata=None, **_) -> dict:
    """Miles custom-agent entry; shell loop runs in the Harbor worker process."""
    from adaptive_branching.src.swe import harbor_client

    request = build_request(base_url, request_kwargs or {}, metadata or {})
    try:
        response = await harbor_client._run_trial(request)
    except (httpx.HTTPError, asyncio.TimeoutError) as exc:
        return {**harbor_client._failed_trial(type(exc).__name__), **harbor_client._harbor_request_metrics(exc)}
    return decode_result(response)
