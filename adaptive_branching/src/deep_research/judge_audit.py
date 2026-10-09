"""Opt-in, per-attempt raw judge responses for offline diagnostics/distillation."""

from __future__ import annotations

import json
import os
import time
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path

_CURRENT: ContextVar[dict | None] = ContextVar("judge_audit_attempt", default=None)


@contextmanager
def audit_attempt(directory: str | None, *, tag: str, attempt: int, request: dict):
    if directory is None:
        yield None
        return
    if not isinstance(directory, str) or not directory.strip():
        raise ValueError("audit directory must be a nonempty path")
    if not isinstance(tag, str) or not tag or type(attempt) is not int or attempt < 1:
        raise ValueError("audit requires a nonempty tag and positive integer attempt")
    if not isinstance(request, dict) or not request:
        raise ValueError("audit request must be a nonempty dict")
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    record = dict(tag=tag, attempt=attempt, request=request, started_at=time.time(), response=None, error=None)
    token = _CURRENT.set(record)
    try:
        yield record
    except BaseException as exc:
        record["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        _CURRENT.reset(token)
        record["ended_at"] = time.time()
        path = root / f"{time.time_ns()}-{uuid.uuid4().hex}.json"
        # Fail closed on storage/serialization errors; never silently lose telemetry.
        payload = json.dumps(record, ensure_ascii=False, allow_nan=False)
        with path.open("x") as handle:
            os.chmod(path, 0o600)
            handle.write(payload + "\n")
            handle.flush()
            os.fsync(handle.fileno())


def record_response(response) -> None:
    record = _CURRENT.get()
    if record is None:
        return
    if not callable(getattr(response, "model_dump", None)):
        raise TypeError("audit response must support model_dump")
    value = response.model_dump(mode="json")
    if not isinstance(value, dict) or not value:
        raise ValueError("audit response must serialize to a nonempty dict")
    record["response"] = value
