"""Hard SWE reward vetoes, audited on newly generated tokens before scoring."""

import os
import re
from collections import Counter

from adaptive_branching.src.swe.strict_format import format_failure

AUDIT_KEY = "swe_generated_format_audit"
AUDIT_VERSION = 2
_REPEATED_CLOSE = re.compile(r"</function>\s*</function>")
STRICT_POLICY = "strict_format_20260922"


def reward_policy() -> str:
    """Identify the sole supported policy; reject stale environment overrides."""
    policy = os.environ.get("SWE_REWARD_POLICY", STRICT_POLICY)
    if policy != STRICT_POLICY:
        raise ValueError(f"unsupported SWE_REWARD_POLICY={policy!r}; only {STRICT_POLICY!r} is supported")
    return policy


def audit_generated_format(sample, tokenizer) -> None:
    """Never inspect inherited prompt tokens or masked tool observations."""
    if not isinstance(sample.metadata, dict):
        raise TypeError(f"sample={sample.index}: metadata must be an object")
    n = sample.response_length
    if type(n) is not int or not 0 <= n <= len(sample.tokens):
        raise ValueError(f"sample={sample.index}: invalid response_length={n!r}")
    mask = sample.loss_mask
    if mask is None and n == 0:
        mask = []
    if mask is None or len(mask) != n or any(value not in (0, 1) for value in mask):
        raise ValueError(f"sample={sample.index}: expected binary response-aligned loss_mask")
    ids = sample.tokens[-n:] if n else []
    spans = []
    start = None
    for index, active in enumerate([*mask, 0]):
        if active and start is None:
            start = index
        elif not active and start is not None:
            spans.append(ids[start:index])
            start = None
    if spans and not callable(getattr(tokenizer, "batch_decode", None)):
        raise TypeError(f"sample={sample.index}: tokenizer.batch_decode is required")
    texts = tokenizer.batch_decode(spans, skip_special_tokens=False, clean_up_tokenization_spaces=False) if spans else []
    if len(texts) != len(spans) or any(not isinstance(text, str) for text in texts):
        raise ValueError(f"sample={sample.index}: decoder returned invalid spans")
    repeated = sum(bool(_REPEATED_CLOSE.search(text)) for text in texts)
    failures = Counter(reason for text in texts if (reason := format_failure(text)) is not None)
    sample.metadata[AUDIT_KEY] = {
        "version": AUDIT_VERSION,
        "spans": len(spans),
        "repeated_spans": repeated,
        "invalid_spans": sum(failures.values()),
        "format_failures": dict(failures),
    }
    sample.metadata["swe_reward_policy"] = reward_policy()
    sample.metadata["agent_repeated_function_close"] = bool(repeated)
    # A Local branch can inherit metadata from a failed Full parent. Penalty
    # telemetry must describe this continuation, not that parent's verdict.
    sample.metadata.update(
        agent_format_failure_penalized=False,
        agent_single_call_length_penalized=False,
        agent_behavior_anomaly_penalized=False,
    )


def hard_failure_reason(sample) -> str | None:
    """Return a trainable-zero reason; missing audits must never pass silently."""
    reward_policy()
    metadata = sample.metadata
    if not isinstance(metadata, dict):
        raise TypeError("SWE behavior guard requires metadata")
    status = metadata.get("exit_status")
    if status == "FormatLimit":
        return "format_limit"
    if status == "LengthTruncated" or metadata.get("agent_last_finish_reason") == "length":
        if status == "LengthTruncated" and metadata.get("agent_last_finish_reason") != "length":
            raise ValueError("LengthTruncated requires agent_last_finish_reason=length")
        return "single_call_length"
    audit = metadata.get(AUDIT_KEY)
    if not isinstance(audit, dict) or type(audit.get("version")) is not int or audit["version"] != AUDIT_VERSION:
        raise ValueError(f"sample={getattr(sample, 'index', None)}: missing/current generated-format audit required")
    spans, repeated = audit.get("spans"), audit.get("repeated_spans")
    if type(spans) is not int or type(repeated) is not int or not 0 <= repeated <= spans:
        raise ValueError(f"invalid generated-format audit: {audit!r}")
    invalid, failures = audit.get("invalid_spans"), audit.get("format_failures")
    if (
        type(invalid) is not int
        or not repeated <= invalid <= spans
        or not isinstance(failures, dict)
        or any(not isinstance(k, str) or not k or type(v) is not int or v <= 0 for k, v in failures.items())
        or sum(failures.values()) != invalid
    ):
        raise ValueError(f"invalid strict-format audit: {audit!r}")
    # Repeated closers are covered by the grammar; their count is telemetry only.
    if spans == 0:
        return "empty_generated_span"
    if invalid:
        return "strict_format"
    return None
