"""Training termination markers shared by the Harbor adapter and SWE reward."""

from typing import Any


def normalize_force_exclude(value: str) -> str:
    """Validate the DeepResearch-compatible Full outcome exclusion policy."""
    if not isinstance(value, str) or value.strip().lower() not in {"all", "wrong", "none"}:
        raise ValueError(f"FORCE_EXCLUDE must be ALL, Wrong, or None, got {value!r}")
    return value.strip().lower()


def context_reserve_hit(metadata: dict[str, Any]) -> bool:
    """Recognize a context stop independently of the final status or reward."""
    if not isinstance(metadata, dict):
        raise TypeError("SWE termination metadata must be a dict")
    metrics = metadata.get("agent_metrics", {})
    if not isinstance(metrics, dict):
        raise TypeError("SWE agent_metrics must be a dict")
    hit = False
    for name, source in (("metadata", metadata), ("agent_metrics", metrics)):
        flag = source.get("agent_context_reserve_hit")
        reason = source.get("agent_forced_final_answer_reason")
        if flag is not None and not isinstance(flag, bool):
            raise TypeError(f"SWE {name}.agent_context_reserve_hit must be bool or None, got {flag!r}")
        if reason is not None and reason not in ("context_reserve", "max_turns"):
            raise ValueError(f"SWE {name}.agent_forced_final_answer_reason is invalid: {reason!r}")
        if flag is False and reason == "context_reserve":
            raise ValueError(f"SWE {name} has conflicting context-reserve markers")
        hit = hit or flag is True or reason == "context_reserve"
    return hit


def max_turns_hit(metadata: dict[str, Any]) -> bool:
    """Recognize a forced turn-limit stop, not normal Local horizon completion."""
    if not isinstance(metadata, dict):
        raise TypeError("SWE termination metadata must be a dict")
    metrics = metadata.get("agent_metrics", {})
    if not isinstance(metrics, dict):
        raise TypeError("SWE agent_metrics must be a dict")
    hit = False
    for name, source in (("metadata", metadata), ("agent_metrics", metrics)):
        flag = source.get("agent_max_turns_hit")
        reason = source.get("agent_forced_final_answer_reason")
        if flag is not None and not isinstance(flag, bool):
            raise TypeError(f"SWE {name}.agent_max_turns_hit must be bool or None, got {flag!r}")
        if reason is not None and reason not in ("context_reserve", "max_turns"):
            raise ValueError(f"SWE {name}.agent_forced_final_answer_reason is invalid: {reason!r}")
        if flag is False and reason == "max_turns":
            raise ValueError(f"SWE {name} has conflicting max-turns markers")
        hit = hit or flag is True or reason == "max_turns"
    return hit


def local_context_reserve_hit(
    request: dict[str, Any], metrics: dict[str, Any], messages: list[dict[str, Any]]
) -> bool:
    """Reproduce the legacy runtime guard for an unmarked Local LimitsExceeded.

    The runtime checks the call limit first, then the last recorded response's
    prompt+completion usage. This deliberately does not estimate token counts.
    """
    if not isinstance(request, dict) or request.get("run_verifier") is not False:
        raise ValueError("context-budget reconstruction requires a Local request")
    if not isinstance(metrics, dict) or not isinstance(messages, list):
        raise TypeError("Local context-budget metrics/messages must be dict/list")
    reserve = request.get("context_reserve_tokens")
    if reserve is None:
        return False
    context = request.get("max_seq_len")
    limit = request.get("max_turns")
    turns = metrics.get("turns")
    for name, value in (("context_reserve_tokens", reserve), ("max_seq_len", context), ("max_turns", limit)):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"Local context-budget {name} must be a positive integer, got {value!r}")
    if reserve >= context:
        raise ValueError("Local context reserve must be smaller than max_seq_len")
    if isinstance(turns, bool) or not isinstance(turns, int) or turns < 0:
        raise ValueError(f"Local context-budget turns must be a non-negative integer, got {turns!r}")
    if turns >= limit:
        return False  # Normal Local horizon wins the runtime's guard ordering.
    if not messages:
        raise ValueError("Local context-budget reconstruction requires trajectory messages")
    for index in range(len(messages) - 1, -1, -1):
        message = messages[index]
        if not isinstance(message, dict):
            raise TypeError(f"Local trajectory message {index} must be a dict")
        value = message
        for key in ("extra", "response", "usage"):
            nested = value.get(key)
            value = {} if nested is None else nested
            if not isinstance(value, dict):
                raise TypeError(f"Local trajectory message {index} {key} must be a dict")
        counts = [0 if value.get(key) is None else value[key] for key in ("prompt_tokens", "completion_tokens")]
        if any(isinstance(n, bool) or not isinstance(n, int) or n < 0 for n in counts):
            raise ValueError(f"Local trajectory message {index} has invalid token usage: {counts!r}")
        if any(counts):
            return sum(counts) >= context - reserve
    return False
