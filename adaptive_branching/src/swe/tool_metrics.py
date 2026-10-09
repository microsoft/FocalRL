"""SWE tool telemetry, vendored unchanged into Harbor's standalone adapter patch."""

from typing import Any

SWE_TOOL_METRIC_KEYS = {
    "agent_swe_tool_metrics_available",
    "agent_swe_tool_result_count",
    "agent_swe_tool_execution_exception_count",
    "agent_swe_tool_timeout_count",
    "agent_swe_tool_returncode_zero_count",
    "agent_swe_tool_returncode_nonzero_count",
    "agent_swe_tool_returncode_zero_rate",
    "agent_swe_tool_submit_count",
    "agent_swe_tool_unobserved_action_count",
}
SWE_HARBOR_METRIC_KEYS = {
    "agent_swe_harbor_request_count",
    "agent_swe_harbor_request_success_count",
    "agent_swe_harbor_connect_error_count",
    "agent_swe_harbor_http_error_count",
    "agent_swe_harbor_timeout_count",
    "agent_swe_harbor_transport_error_count",
}

SWE_PROMOTED_TOOL_METRIC_KEYS = SWE_TOOL_METRIC_KEYS | {
    "agent_tool_call_count",
    "agent_tool_unit_count",
    "agent_tool_unit_success_count",
    "agent_tool_unit_success_rate",
}


def extract_swe_tool_metrics(messages: list[dict[str, Any]], *, prefix_messages: int = 0) -> dict[str, int | float]:
    """Count native tool observations; a nonzero command exit is not a transport failure.

    Only recorded observations enter the execution-result denominator. A successful
    submit has an exit message instead of a tool observation and is counted separately.
    Remaining unobserved actions are coverage gaps, not assumed infrastructure failures.
    Replay history is omitted before matching calls and their observations.
    """
    if not isinstance(messages, list) or any(not isinstance(message, dict) for message in messages):
        raise TypeError("SWE messages must be a list of objects")
    if (
        isinstance(prefix_messages, bool)
        or not isinstance(prefix_messages, int)
        or not 0 <= prefix_messages <= len(messages)
    ):
        raise ValueError(f"invalid SWE prefix_messages={prefix_messages!r} for {len(messages)} messages")
    requested: set[str] = set()
    observed: set[str] = set()
    zero = nonzero = errors = timeouts = submits = 0
    for index, message in enumerate(messages[prefix_messages:], start=prefix_messages):
        extra = message.get("extra", {})
        if not isinstance(extra, dict):
            raise TypeError(f"SWE message {index} extra must be an object")
        role = message.get("role")
        if role == "assistant":
            actions = extra.get("actions", [])
            if not isinstance(actions, list):
                raise TypeError(f"SWE message {index} actions must be a list")
            for action in actions:
                if not isinstance(action, dict):
                    raise TypeError(f"SWE message {index} action must be an object")
                call_id = action.get("tool_call_id")
                if not isinstance(call_id, str) or not call_id or call_id in requested:
                    raise ValueError(f"SWE message {index} invalid or duplicate tool_call_id={call_id!r}")
                requested.add(call_id)
        elif role == "tool":
            call_id = message.get("tool_call_id")
            if not isinstance(call_id, str) or call_id not in requested or call_id in observed:
                raise ValueError(f"SWE message {index} unmatched or duplicate tool result={call_id!r}")
            code, exception = extra.get("returncode"), extra.get("exception_info", "")
            if isinstance(code, bool) or not isinstance(code, int):
                raise TypeError(f"SWE message {index} returncode must be an integer, got {code!r}")
            if exception is not None and not isinstance(exception, str):
                raise TypeError(f"SWE message {index} exception_info must be a string or null")
            observed.add(call_id)
            if exception:
                errors += 1
                # Exact marker emitted by Harbor's command bridge, not arbitrary command output.
                timeouts += int(exception.startswith("Command timed out after "))
            elif code == 0:
                zero += 1
            else:
                nonzero += 1
        elif role == "exit" and extra.get("exit_status") == "Submitted":
            submits += 1
    if submits > 1 or len(observed) + submits > len(requested):
        raise ValueError(f"inconsistent SWE actions/results/submits: {len(requested)}/{len(observed)}/{submits}")
    result: dict[str, int | float] = {
        "agent_swe_tool_metrics_available": 1,
        "agent_tool_call_count": len(requested),
        "agent_swe_tool_result_count": len(observed),
        "agent_swe_tool_execution_exception_count": errors,
        "agent_swe_tool_timeout_count": timeouts,
        "agent_swe_tool_returncode_zero_count": zero,
        "agent_swe_tool_returncode_nonzero_count": nonzero,
        "agent_swe_tool_submit_count": submits,
        "agent_swe_tool_unobserved_action_count": len(requested) - len(observed) - submits,
    }
    if observed:
        result.update(
            agent_tool_unit_count=len(observed),
            agent_tool_unit_success_count=zero + nonzero,
            agent_tool_unit_success_rate=(zero + nonzero) / len(observed),
        )
    if zero + nonzero:
        result["agent_swe_tool_returncode_zero_rate"] = zero / (zero + nonzero)
    return result


def extract_swe_trajectory_tool_metrics(trajectory: dict[str, Any], replay: dict[str, Any] | None = None):
    """Count only newly generated actions, validating the replay prompt prefix."""
    if not isinstance(trajectory, dict) or not isinstance(trajectory.get("messages"), list):
        raise TypeError("SWE trajectory must contain a messages list")
    prefix = []
    if replay is not None:
        if not isinstance(replay, dict) or not isinstance(replay.get("messages"), list):
            raise TypeError("SWE replay must contain a messages list")
        prefix = replay["messages"]
        if not prefix or len(prefix) > len(trajectory["messages"]):
            raise ValueError("SWE replay prefix is empty or longer than trajectory")
        # Ignore diagnostic extras, which Harbor may slim on repeated postprocessing.
        for index, (old, current) in enumerate(zip(prefix, trajectory["messages"], strict=False)):
            if not isinstance(old, dict) or not isinstance(current, dict):
                raise TypeError(f"SWE replay message {index} must be an object")
            for field in ("role", "content", "tool_calls", "tool_call_id"):
                if old.get(field) != current.get(field):
                    raise ValueError(f"SWE replay prefix differs at message {index} field {field}")
    return extract_swe_tool_metrics(trajectory["messages"], prefix_messages=len(prefix))
