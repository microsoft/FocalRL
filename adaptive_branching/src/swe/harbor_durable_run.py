"""Idempotent Harbor runs: reconnect to one execution and persist its result.

The journal deliberately refuses to replay an unfinished run after a server
restart. Its remote process may still exist; silently starting it again would
duplicate shell actions. Completed responses survive a server restart.
"""

import asyncio
import hashlib
import json
import logging
import re
import sqlite3
from pathlib import Path

import httpx
from fastapi import HTTPException

logger = logging.getLogger(__name__)
HEADER = "X-Harbor-Run-Key"
PROTOCOL = "durable_run_v1"


def _identity(key, payload):
    if not isinstance(key, str) or re.fullmatch(r"[0-9a-f]{64}", key) is None:
        raise ValueError("run key must be 64 lowercase hexadecimal characters")
    if not isinstance(payload, dict) or not payload:
        raise ValueError("run payload must be a nonempty object")
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode()).hexdigest()


class DurableRuns:
    def __init__(self, path, *, capacity=512):
        path = Path(path)
        if not path.is_absolute() or not path.parent.is_dir():
            raise ValueError("journal needs an absolute path with an existing parent")
        if type(capacity) is not int or capacity <= 0:
            raise ValueError("capacity must be a positive integer")
        self.path, self.capacity = path, capacity
        self.active = {}
        self.admissions = set()
        self.lock = asyncio.Lock()
        with sqlite3.connect(path) as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.execute("CREATE TABLE IF NOT EXISTS runs (key TEXT PRIMARY KEY, digest TEXT NOT NULL, result TEXT)")
        path.chmod(0o600)

    def _read(self, key):
        if re.fullmatch(r"[0-9a-f]{64}", key) is None:
            raise ValueError("invalid journal key")
        with sqlite3.connect(self.path) as db:
            return db.execute("SELECT digest, result FROM runs WHERE key=?", (key,)).fetchone()

    def _write(self, key, digest, result=None):
        if re.fullmatch(r"[0-9a-f]{64}", key) is None or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise ValueError("invalid journal identity")
        if result is not None and not isinstance(result, dict):
            raise TypeError("terminal result must be an object")
        with sqlite3.connect(self.path) as db:
            if result is None:
                db.execute("INSERT INTO runs VALUES (?, ?, NULL)", (key, digest))
            else:
                encoded = json.dumps(result, allow_nan=False)
                cursor = db.execute(
                    "UPDATE runs SET result=? WHERE key=? AND digest=? AND result IS NULL", (encoded, key, digest)
                )
                if cursor.rowcount != 1:
                    raise RuntimeError(f"journal terminal transition failed: run={key}")

    async def _execute(self, key, digest, execute):
        try:
            result = await execute()
            if hasattr(result, "model_dump"):
                result = result.model_dump()
            if not isinstance(result, dict) or not result:
                raise TypeError(f"invalid Harbor terminal response: run={key}")
            await asyncio.to_thread(self._write, key, digest, result)
            return result
        except BaseException:
            # Leave a pending tombstone: even an ambiguous exception must never
            # permit duplicate execution. Retries fail explicitly for this key.
            logger.exception("Harbor run unresolved; preserving journal identity: run=%s", key)
            raise
        finally:
            self.active.pop(key, None)

    async def _admit(self, key, digest, execute):
        if not callable(execute) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise ValueError("invalid run admission")
        async with self.lock:
            row = await asyncio.to_thread(self._read, key)
            if row is not None:
                if row[0] != digest:
                    raise HTTPException(409, "run key reused with a different payload")
                if row[1] is not None:
                    result = json.loads(row[1])
                    if not isinstance(result, dict) or not result:
                        raise RuntimeError(f"invalid persisted result: run={key}")
                    return result
                if key not in self.active:
                    raise HTTPException(409, "run outcome unresolved; operator recovery required; do not resubmit")
            else:
                if len(self.active) >= self.capacity:
                    raise HTTPException(503, "durable run capacity reached")
                await asyncio.to_thread(self._write, key, digest)
                task = asyncio.create_task(self._execute(key, digest, execute))
                self.active[key] = task
                task.add_done_callback(lambda done: None if done.cancelled() else done.exception())
            return self.active[key]

    def _admitted(self, task):
        self.admissions.discard(task)
        if not task.cancelled():
            task.exception()  # Observe errors even when the HTTP waiter disconnected.

    async def run(self, key, payload, execute):
        digest = _identity(key, payload)
        if not callable(execute):
            raise TypeError("execute must be callable")
        # Admission includes disk IO. Cancellation between INSERT and task
        # creation must not strand a run that was accepted but never executed.
        admission = asyncio.create_task(self._admit(key, digest, execute))
        self.admissions.add(admission)
        admission.add_done_callback(self._admitted)
        task = await asyncio.shield(admission)
        if isinstance(task, dict):
            return task
        # The remote trial belongs to the key, not to this particular HTTP
        # connection. Client disconnect/cancellation must not cancel it.
        return await asyncio.shield(task)


async def reconnect_run(client, server, payload, *, headers=None, timeout=12000, retry_delay=1):
    from miles.utils.remote_trial import RemoteTrialUnresolved

    if not isinstance(server, str) or not server.startswith(("http://", "https://")):
        raise ValueError("server must be an absolute HTTP URL")
    if not isinstance(payload, dict) or not isinstance(payload.get("base_url"), str) or not payload["base_url"]:
        raise ValueError("run requires a nonempty model session URL")
    if type(timeout) not in (int, float) or not 0 < timeout < float("inf"):
        raise ValueError("timeout must be finite and positive")
    if type(retry_delay) not in (int, float) or not 0 <= retry_delay < timeout:
        raise ValueError("retry delay must be nonnegative and below timeout")
    key = hashlib.sha256(payload["base_url"].encode()).hexdigest()
    _identity(key, payload)
    headers = {**(headers or {}), HEADER: key}
    deadline = asyncio.get_running_loop().time() + timeout
    capability_checked = False
    attempts = 0
    try:
        while True:
            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                raise TimeoutError(f"Harbor run recovery deadline reached: run={key}")
            try:
                if not capability_checked:
                    response = await asyncio.wait_for(client.get(server + "/health", headers=headers), min(15, remaining))
                    response.raise_for_status()
                    if response.json().get(PROTOCOL) is not True:
                        raise ValueError("Harbor does not advertise durable runs; refusing unsafe retries")
                    capability_checked = True
                    continue
                attempts += 1
                response = await asyncio.wait_for(client.post(server + "/run", json=payload, headers=headers), remaining)
                response.raise_for_status()
                result = response.json()
                if not isinstance(result, dict) or not result:
                    raise ValueError(f"invalid Harbor response: run={key}")
                return result
            except (httpx.TransportError, httpx.HTTPStatusError, asyncio.TimeoutError) as exc:
                if isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code not in {502, 503, 504}:
                    raise
                logger.warning("Reconnecting to original Harbor run=%s attempt=%s error=%s", key, attempts, type(exc).__name__)
                await asyncio.sleep(min(retry_delay, max(0, deadline - asyncio.get_running_loop().time())))
    except Exception as exc:
        # Propagate past the ordinary sample-exclusion path: its finally block
        # would otherwise delete a session still used by the remote agent.
        raise RemoteTrialUnresolved(f"Preserve model session; Harbor run={key} unresolved: {type(exc).__name__}: {exc}") from exc
