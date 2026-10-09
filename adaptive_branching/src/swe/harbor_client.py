"""Miles custom-agent adapter for Harbor-managed mini-swe-agent trials."""

import asyncio
import logging
import math
import os
import socket
from typing import Any
from urllib.parse import urlparse, urlsplit, urlunparse

import httpx

from adaptive_branching.src.swe.termination import (
    context_reserve_hit,
    local_context_reserve_hit,
    max_turns_hit,
    normalize_force_exclude,
)
from adaptive_branching.src.swe.tool_metrics import (
    SWE_HARBOR_METRIC_KEYS,
    SWE_PROMOTED_TOOL_METRIC_KEYS,
)

logger = logging.getLogger(__name__)

_DEFAULT_TRIAL_TIMEOUT_SECONDS = 12000
_TRAINABLE_EXIT_STATUSES = {"Submitted", "LimitsExceeded"}
_MAX_HARBOR_CONNECTIONS = 512
_ARTIFACT_REQUEST_ATTEMPTS = 10
_ARTIFACT_REQUEST_TIMEOUT_SECONDS = 60
_client: httpx.AsyncClient | None = None


def _trial_timeout_seconds() -> int:
    value = int(os.environ.get("HARBOR_TRIAL_TIMEOUT_SECONDS", _DEFAULT_TRIAL_TIMEOUT_SECONDS))
    if value <= 0:
        raise ValueError("HARBOR_TRIAL_TIMEOUT_SECONDS must be positive")
    return value


def _optional_positive_env_int(name: str) -> int | None:
    raw = os.environ.get(name)
    if raw is None:
        return None
    value = int(raw)
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _get_client() -> httpx.AsyncClient:
    global _client
    if type(_MAX_HARBOR_CONNECTIONS) is not int or _MAX_HARBOR_CONNECTIONS <= 0:
        raise ValueError("_MAX_HARBOR_CONNECTIONS must be a positive integer")
    if _client is None:
        _client = _new_client(_MAX_HARBOR_CONNECTIONS)
    return _client


def _new_client(connection_limit: int) -> httpx.AsyncClient:
    if type(connection_limit) is not int or connection_limit <= 0:
        raise ValueError("connection_limit must be a positive integer")
    socket_options = [
        (socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1),
        (socket.IPPROTO_TCP, getattr(socket, "TCP_KEEPIDLE", 4), 60),
        (socket.IPPROTO_TCP, getattr(socket, "TCP_KEEPINTVL", 5), 30),
        (socket.IPPROTO_TCP, getattr(socket, "TCP_KEEPCNT", 6), 5),
    ]
    return httpx.AsyncClient(
        transport=httpx.AsyncHTTPTransport(
            proxy=_harbor_proxy_url(),
            socket_options=socket_options,
            # AsyncClient limits do not configure an explicitly supplied transport.
            limits=httpx.Limits(
                max_connections=connection_limit,
                max_keepalive_connections=connection_limit,
            ),
        ),
        timeout=None,
    )


def _harbor_proxy_url() -> str | None:
    proxy_url = os.getenv("HARBOR_PROXY_URL", "").strip()
    if not proxy_url:
        return None
    parsed = urlparse(proxy_url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("HARBOR_PROXY_URL must be an absolute HTTP(S) URL")
    return proxy_url


async def _post_trial(url: str, payload: dict[str, Any]) -> dict[str, Any]:
    response = await _get_client().post(url, json=payload, headers=_admin_headers())
    response.raise_for_status()
    body = response.json()
    if not isinstance(body, dict):
        raise TypeError(f"Harbor {urlsplit(url).path} returned {type(body).__name__}, expected an object")
    return body


async def _run_trial(payload: dict[str, Any]) -> dict[str, Any]:
    """Use reconnectable execution only against an explicitly capable server."""
    if not isinstance(payload, dict) or not payload:
        raise ValueError("Harbor run payload must be a nonempty object")
    enabled = os.getenv("HARBOR_DURABLE_RUNS", "0")
    if enabled not in {"0", "1"}:
        raise ValueError("HARBOR_DURABLE_RUNS must be 0 or 1")
    server = os.getenv("HARBOR_SERVER_URL", "http://localhost:11200").rstrip("/")
    if enabled == "1":
        from adaptive_branching.src.swe.harbor_durable_run import reconnect_run

        return await reconnect_run(
            _get_client(), server, payload, headers=_admin_headers(), timeout=_trial_timeout_seconds()
        )
    return await asyncio.wait_for(_post_trial(server + "/run", payload), timeout=_trial_timeout_seconds())


async def _post_trial_artifact(url: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Retry artifact reads/builds, never a /run that could start a second trial."""
    endpoint = urlsplit(url).path
    if endpoint not in {"/trajectory", "/replay"}:
        raise ValueError(f"Harbor artifact retries are not allowed for {endpoint}")
    for attempt in range(1, _ARTIFACT_REQUEST_ATTEMPTS + 1):
        try:
            return await asyncio.wait_for(_post_trial(url, payload), timeout=_ARTIFACT_REQUEST_TIMEOUT_SECONDS)
        except (httpx.TransportError, httpx.HTTPStatusError, asyncio.TimeoutError) as exc:
            if isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code not in {502, 503, 504}:
                raise
            logger.warning(
                "Harbor %s attempt %d/%d failed: trial_dir=%s selected_turn=%s %s: %s",
                endpoint,
                attempt,
                _ARTIFACT_REQUEST_ATTEMPTS,
                payload.get("trial_dir"),
                payload.get("selected_turn"),
                type(exc).__name__,
                exc,
            )
            if attempt == _ARTIFACT_REQUEST_ATTEMPTS:
                raise
            await asyncio.sleep(1)


def _admin_headers() -> dict[str, str] | None:
    secret = os.getenv("HARBOR_ADMIN_SECRET", "").strip()
    return {"Authorization": f"Bearer {secret}"} if secret else None


async def get_trial_messages(trial_dir: str) -> list[dict[str, Any]]:
    if not isinstance(trial_dir, str) or not trial_dir.strip():
        raise ValueError("trial_dir must be a non-empty string")
    server_url = os.getenv("HARBOR_SERVER_URL", "http://localhost:11200").rstrip("/")
    body = await _post_trial_artifact(f"{server_url}/trajectory", {"trial_dir": trial_dir})
    messages = body.get("messages")
    if not isinstance(messages, list) or not messages or any(not isinstance(message, dict) for message in messages):
        raise ValueError("Harbor trajectory response must contain non-empty object messages")
    return messages


async def create_trial_replay(trial_dir: str, selected_turn: int) -> dict[str, Any]:
    if not isinstance(trial_dir, str) or not trial_dir.strip():
        raise ValueError("trial_dir must be a non-empty string")
    if isinstance(selected_turn, bool) or not isinstance(selected_turn, int) or selected_turn <= 0:
        raise ValueError("selected_turn must be a positive integer")
    server_url = os.getenv("HARBOR_SERVER_URL", "http://localhost:11200").rstrip("/")
    body = await _post_trial_artifact(
        f"{server_url}/replay",
        {"trial_dir": trial_dir, "selected_turn": selected_turn},
    )
    replay_path = body.get("replay_path")
    prefix_messages = body.get("prefix_messages")
    n_calls = body.get("n_calls")
    if not isinstance(replay_path, str) or not replay_path:
        raise ValueError("Harbor replay response has no replay_path")
    if not isinstance(prefix_messages, list) or not prefix_messages:
        raise ValueError("Harbor replay response has no prefix_messages")
    if isinstance(n_calls, bool) or not isinstance(n_calls, int) or n_calls < 0:
        raise ValueError("Harbor replay response has invalid n_calls")
    return body


async def _flush_trials(url: str, payload: dict[str, Any], headers: dict[str, str] | None) -> None:
    for attempt in range(3):
        try:
            response = await _get_client().post(url, json=payload, headers=headers)
            response.raise_for_status()
            return
        except httpx.HTTPError:
            if attempt == 2:
                raise
            await asyncio.sleep(1)


def _failed_trial(exit_status: str) -> dict[str, Any]:
    if not isinstance(exit_status, str) or not exit_status:
        raise ValueError(f"Harbor exit_status must be a non-empty string, got {exit_status!r}")
    return {
        **_promoted_tool_metrics({}),
        "reward": 0.0,
        "exit_status": exit_status,
        "eval_report": {},
        "agent_metrics": {},
        "agent_excluded_from_training": True,
        "agent_context_reserve_hit": False,
        "agent_max_turns_hit": False,
        "agent_forced_final_answer_reason": None,
    }


def _promoted_tool_metrics(agent_metrics: dict[str, Any]) -> dict[str, Any]:
    """Clear retained retry metadata; unavailable observations are explicitly unknown."""
    if not isinstance(agent_metrics, dict):
        raise TypeError("Harbor agent_metrics must be an object")
    available = agent_metrics.get("agent_swe_tool_metrics_available", 0)
    if type(available) is not int or available not in (0, 1):
        raise ValueError(f"invalid SWE tool telemetry availability: {available!r}")
    result = dict.fromkeys(SWE_PROMOTED_TOOL_METRIC_KEYS)
    result["agent_swe_tool_metrics_available"] = available
    if not available:
        return result
    for key in SWE_PROMOTED_TOOL_METRIC_KEYS:
        value = agent_metrics.get(key)
        if value is None:
            if key.endswith("_rate") or key in {"agent_tool_unit_count", "agent_tool_unit_success_count"}:
                continue  # No observations: omit undefined ratios and their empty denominator.
            raise ValueError(f"Harbor SWE telemetry missing {key}")
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(f"Harbor SWE telemetry {key} must be finite and non-negative, got {value!r}")
        if key.endswith("_count") and not isinstance(value, int):
            raise ValueError(f"Harbor SWE telemetry {key} must be an integer")
        if key.endswith("_rate") and value > 1:
            raise ValueError(f"Harbor SWE telemetry {key} must be in [0, 1]")
        result[key] = value
    observed = result["agent_swe_tool_result_count"]
    returned = result["agent_swe_tool_returncode_zero_count"] + result["agent_swe_tool_returncode_nonzero_count"]
    if observed != returned + result["agent_swe_tool_execution_exception_count"]:
        raise ValueError("Harbor SWE tool results do not equal returned commands plus exceptions")
    if result["agent_swe_tool_timeout_count"] > result["agent_swe_tool_execution_exception_count"]:
        raise ValueError("Harbor SWE tool timeouts exceed execution exceptions")
    if (
        result["agent_tool_call_count"]
        != observed + result["agent_swe_tool_submit_count"] + result["agent_swe_tool_unobserved_action_count"]
    ):
        raise ValueError("Harbor SWE requested actions do not match results, submits and unobserved actions")
    if observed and (
        result["agent_tool_unit_count"] != observed or result["agent_tool_unit_success_count"] != returned
    ):
        raise ValueError("Harbor SWE tool success counts do not match observed results")
    for key, numerator, denominator in (
        ("agent_tool_unit_success_rate", returned, observed),
        ("agent_swe_tool_returncode_zero_rate", result["agent_swe_tool_returncode_zero_count"], returned),
    ):
        expected = numerator / denominator if denominator else None
        if result[key] != expected:
            raise ValueError(f"Harbor SWE telemetry inconsistent {key}: {result[key]!r}, expected {expected!r}")
    return result


def _harbor_request_metrics(error: BaseException | None = None) -> dict[str, int]:
    """One /run attempt; these counters do not describe model API or artifact retries."""
    if error is not None and not isinstance(error, (httpx.HTTPError, asyncio.TimeoutError)):
        raise TypeError(f"unsupported Harbor request error: {type(error).__name__}")
    result = dict.fromkeys(SWE_HARBOR_METRIC_KEYS, 0)
    result["agent_swe_harbor_request_count"] = 1
    if error is None:
        key = "request_success"
    elif isinstance(error, (httpx.TimeoutException, asyncio.TimeoutError)):
        key = "timeout"
    elif isinstance(error, httpx.ConnectError):
        key = "connect_error"
    elif isinstance(error, httpx.HTTPStatusError):
        key = "http_error"
    else:
        key = "transport_error"
    result[f"agent_swe_harbor_{key}_count"] = 1
    return result


def _has_matching_verifier_reward(eval_report: dict[str, Any], reward: float) -> bool:
    if not isinstance(eval_report, dict):
        raise TypeError("Harbor eval_report must be an object")
    if not eval_report:
        return False
    report_reward = eval_report.get("reward")
    if isinstance(report_reward, bool) or not isinstance(report_reward, (int, float)):
        raise TypeError(f"Harbor eval_report reward must be numeric, got {report_reward!r}")
    report_reward = float(report_reward)
    if not math.isfinite(report_reward) or report_reward != reward:
        raise ValueError(f"Harbor eval_report reward {report_reward!r} does not match reward {reward!r}")
    if "infrastructure_error" in eval_report:
        error = eval_report["infrastructure_error"]
        if not isinstance(error, str) or not error.strip():
            raise ValueError("Harbor eval_report infrastructure_error must be a non-empty string")
        # A verifier/artifact failure is not evidence that the model's patch is wrong.
        return False
    return True


def _is_trainable_response(*, exit_status: str, eval_report: dict[str, Any], reward: float) -> bool:
    # Full LimitsExceeded is excluded even when an older runtime omits the reason.
    # Local horizon completion is handled separately and remains trainable.
    if not isinstance(exit_status, str) or not exit_status:
        raise ValueError(f"Harbor exit_status must be a non-empty string, got {exit_status!r}")
    return exit_status == "Submitted" and _has_matching_verifier_reward(eval_report, reward)


def _generated_agent_turns(agent_metrics: dict[str, Any], *, replay_prefix_calls: int = 0) -> int | None:
    turns = agent_metrics.get("turns")
    if turns is None:
        return None
    if isinstance(turns, bool) or not isinstance(turns, int) or turns < 0:
        raise ValueError(f"Harbor agent_metrics.turns must be a non-negative integer, got {turns!r}")
    if isinstance(replay_prefix_calls, bool) or not isinstance(replay_prefix_calls, int) or replay_prefix_calls < 0:
        raise ValueError("replay_prefix_calls must be a non-negative integer")
    if replay_prefix_calls > turns:
        raise ValueError(f"replay prefix has {replay_prefix_calls} calls but Harbor reports only {turns} total turns")
    return turns - replay_prefix_calls


def _replace_url_host(url: str, host: str, port: int | None = None) -> str:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError(f"base_url must be an absolute HTTP(S) URL, got {url!r}")
    if not host or "://" in host or "/" in host:
        raise ValueError("MILES_ROUTER_EXTERNAL_HOST must be a hostname or IP address")
    target_port = parsed.port if port is None else port
    if target_port is not None and not 1 <= target_port <= 65535:
        raise ValueError("MILES_ROUTER_EXTERNAL_PORT must be between 1 and 65535")
    netloc = f"{host}:{target_port}" if target_port is not None else host
    return urlunparse(parsed._replace(netloc=netloc))


def build_request(
    *,
    base_url: str,
    request_kwargs: dict[str, Any],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    """Build the request consumed by Harbor's ``miles_agent_server.py``."""
    if not isinstance(metadata, dict):
        raise TypeError("metadata must be a dict")
    instance_id = metadata.get("instance_id")
    if not isinstance(instance_id, str) or not instance_id.strip():
        raise ValueError("SWE sample metadata.instance_id must be a non-empty string")
    if not isinstance(request_kwargs, dict):
        raise TypeError("request_kwargs must be a dict")

    parsed_base_url = urlparse(base_url)
    if parsed_base_url.scheme not in {"http", "https"} or not parsed_base_url.hostname:
        raise ValueError(f"base_url must be an absolute HTTP(S) URL, got {base_url!r}")
    session_url = f"{base_url.rstrip('/')}/v1"
    external_host = os.getenv("MILES_ROUTER_EXTERNAL_HOST", "").strip()
    external_port = _optional_positive_env_int("MILES_ROUTER_EXTERNAL_PORT")
    if external_port is not None and external_port > 65535:
        raise ValueError("MILES_ROUTER_EXTERNAL_PORT must be at most 65535")
    if external_port is not None and not external_host:
        raise ValueError("MILES_ROUTER_EXTERNAL_PORT requires MILES_ROUTER_EXTERNAL_HOST")
    if external_host:
        session_url = _replace_url_host(session_url, external_host, external_port)

    model_name = os.getenv("AGENT_MODEL_NAME", "model").strip()
    if not model_name:
        raise ValueError("AGENT_MODEL_NAME must be non-empty")

    sampling_params = dict(request_kwargs)
    max_tokens_per_turn = _optional_positive_env_int("SWE_MAX_TOKENS_PER_TURN")
    if max_tokens_per_turn is not None:
        sampling_params["max_tokens"] = max_tokens_per_turn

    request = dict(
        instance_id=instance_id,
        base_url=session_url,
        model=f"openai/{model_name}",
        agent_name="mini-swe-agent",
        sampling_params=sampling_params,
    )

    max_seq_len = metadata.get("max_seq_len")
    if max_seq_len is not None:
        request["max_seq_len"] = int(max_seq_len)
    is_local = bool(metadata.get("ab_local_rollout")) or metadata.get("ab_rollout_kind") == "local"
    if is_local:
        replay_path = metadata.get("swe_replay_path")
        prefix_calls = metadata.get("swe_replay_n_calls")
        local_horizon = metadata.get("agent_max_turns")
        if not isinstance(replay_path, str) or not replay_path:
            raise ValueError("local SWE rollout requires swe_replay_path")
        if isinstance(prefix_calls, bool) or not isinstance(prefix_calls, int) or prefix_calls < 0:
            raise ValueError("local SWE rollout requires non-negative swe_replay_n_calls")
        if isinstance(local_horizon, bool) or not isinstance(local_horizon, int) or local_horizon <= 0:
            raise ValueError("local SWE rollout requires positive agent_max_turns")
        request.update(
            replay_path=replay_path,
            max_turns=prefix_calls + local_horizon,
            run_verifier=False,
            force_submit_on_limit=False,
        )
    else:
        max_turns = _optional_positive_env_int("SWE_AGENT_MAX_TURNS")
        if max_turns is not None:
            request["max_turns"] = max_turns
    context_reserve_tokens = _optional_positive_env_int("SWE_CONTEXT_RESERVE_TOKENS")
    if context_reserve_tokens is not None:
        if max_seq_len is None:
            raise ValueError("SWE_CONTEXT_RESERVE_TOKENS requires metadata.max_seq_len")
        if context_reserve_tokens >= int(max_seq_len):
            raise ValueError("SWE_CONTEXT_RESERVE_TOKENS must be smaller than metadata.max_seq_len")
        request["context_reserve_tokens"] = context_reserve_tokens

    session_server_id = metadata.get("session_server_id")
    if session_server_id is not None and external_host:
        port = urlsplit(f"http://{session_server_id}").port
        port = external_port if external_port is not None else port
        session_server_id = f"{external_host}:{port}" if port else external_host
    if session_server_id is not None:
        request["session_server_id"] = session_server_id
    session_server_instance_id = metadata.get("session_server_instance_id")
    if session_server_instance_id is not None:
        request["session_server_instance_id"] = session_server_instance_id

    return request


async def run(
    base_url: str,
    prompt: Any,
    request_kwargs: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
    **_: Any,
) -> dict[str, Any] | None:
    """Run one official mini-swe-agent trial through the Harbor server."""
    del prompt
    mode = normalize_force_exclude(os.environ.get("FORCE_EXCLUDE", "ALL"))
    request = build_request(
        base_url=base_url,
        request_kwargs={} if request_kwargs is None else request_kwargs,
        metadata={} if metadata is None else metadata,
    )
    try:
        response = await _run_trial(request)
    except asyncio.CancelledError:
        raise
    except asyncio.TimeoutError as exc:
        logger.error("Harbor trial timed out after %ss", _trial_timeout_seconds())
        return {**_failed_trial("HarborTimeout"), **_harbor_request_metrics(exc)}
    except httpx.HTTPError as exc:
        logger.error("Harbor request failed: %s", exc)
        return {**_failed_trial("HarborTransportError"), **_harbor_request_metrics(exc)}

    reward = response.get("reward", 0.0)
    if isinstance(reward, bool) or not isinstance(reward, (int, float)):
        raise TypeError(f"Harbor reward must be numeric, got {reward!r}")
    if not math.isfinite(reward) or reward not in (0, 1):
        raise ValueError(f"Harbor reward must be finite and binary, got {reward!r}")
    exit_status = response.get("exit_status", "")
    if not isinstance(exit_status, str):
        raise TypeError(f"Harbor exit_status must be a string, got {exit_status!r}")

    eval_report = response.get("eval_report", {})
    if not isinstance(eval_report, dict):
        raise TypeError(f"Harbor eval_report must be an object, got {type(eval_report).__name__}")
    agent_metrics = response.get("agent_metrics", {})
    if not isinstance(agent_metrics, dict):
        raise TypeError(f"Harbor agent_metrics must be an object, got {type(agent_metrics).__name__}")
    reserve_hit = context_reserve_hit({"agent_metrics": agent_metrics})
    turns_hit = max_turns_hit({"agent_metrics": agent_metrics})
    is_local = request.get("run_verifier") is False
    messages = None
    if is_local and exit_status in _TRAINABLE_EXIT_STATUSES and not (reserve_hit or turns_hit):
        trial_dir = response.get("trial_dir")
        if not isinstance(trial_dir, str) or not trial_dir:
            raise ValueError("local Harbor response must contain trial_dir")
        messages = await get_trial_messages(trial_dir)
        if exit_status == "LimitsExceeded":
            reserve_hit = local_context_reserve_hit(request, agent_metrics, messages)
            if reserve_hit:
                logger.warning(
                    "Harbor omitted Local context-reserve marker; reconstructed runtime guard: "
                    "trial_dir=%s turns=%s max_turns=%s max_seq_len=%s reserve=%s",
                    trial_dir,
                    agent_metrics.get("turns"),
                    request.get("max_turns"),
                    request.get("max_seq_len"),
                    request.get("context_reserve_tokens"),
                )
    if (reserve_hit or turns_hit) and (is_local or mode == "all" or (mode == "wrong" and reward == 0)):
        trainable = False
    elif exit_status == "LengthTruncated":
        # A model output cap has a known zero reward; no verifier/PRM evidence is needed.
        # Length during a forced context/turn-limit submission is a different termination.
        trainable = not (
            agent_metrics.get("agent_forced_final_answer_reason") or agent_metrics.get("agent_context_reserve_hit")
        )
    else:
        trainable = (
            exit_status in _TRAINABLE_EXIT_STATUSES and bool(messages)
            if is_local
            else _is_trainable_response(exit_status=exit_status, eval_report=eval_report, reward=float(reward))
        )
    replay_prefix_calls = metadata.get("swe_replay_n_calls", 0) if is_local and metadata is not None else 0
    agent_turns = _generated_agent_turns(agent_metrics, replay_prefix_calls=replay_prefix_calls)
    result = {
        **_promoted_tool_metrics(agent_metrics),
        **_harbor_request_metrics(),
        "reward": float(reward),
        "exit_status": exit_status,
        "eval_report": eval_report,
        "agent_metrics": agent_metrics,
        "agent_excluded_from_training": not trainable,
        # These must reach Sample.metadata, and replace any previous retry's markers.
        "agent_context_reserve_hit": reserve_hit,
        "agent_max_turns_hit": turns_hit,
        "agent_forced_final_answer_reason": agent_metrics.get("agent_forced_final_answer_reason"),
        # Group resampling retains metadata; overwrite the previous trial's length marker.
        "agent_last_finish_reason": "length" if exit_status == "LengthTruncated" else None,
    }
    if agent_turns is not None:
        result["agent_turns"] = agent_turns
    if messages is not None:
        result["messages"] = messages
    for key in ("exit_status_detail", "trial_dir", "trial_id"):
        if key in response:
            result[key] = response[key]
    return result


async def abort(args: Any) -> None:
    """Ask Harbor to cancel trials owned by the aborted Miles session."""
    instance_id = getattr(args, "session_server_instance_id", None)
    server_url = os.getenv("HARBOR_SERVER_URL", "").rstrip("/")
    if not instance_id or not server_url:
        return
    headers = None
    if secret := os.getenv("HARBOR_ADMIN_SECRET"):
        headers = {"Authorization": f"Bearer {secret}"}
    await _flush_trials(
        f"{server_url}/flush",
        {"session_server_instance_id": instance_id},
        headers,
    )
