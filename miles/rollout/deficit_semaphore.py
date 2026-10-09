"""Optional Full/Local arbitration at the existing per-trajectory generation gate."""

import asyncio
import os
import threading
from collections import deque
from contextlib import asynccontextmanager


def _integer(name, value, minimum=1):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}, got {value!r}")
    return value


class DeficitSemaphore:
    """One capacity, FIFO per kind, work-conserving and non-preemptive.

    Futures and active leases belong to one event loop. Only ready-stock updates
    and metrics are called from the collector thread; a lock protects the snapshot.
    Group creation, completion and scoring are deliberately outside this class.
    """

    def __init__(self, capacity, full_target, local_target, samples_per_group):
        self.capacity = _integer("capacity", capacity)
        self.targets = (_integer("full_target", full_target), _integer("local_target", local_target))
        self.samples_per_group = _integer("samples_per_group", samples_per_group)
        self._ready = (0, 0)
        self._active = [0, 0]
        self._waiting = (deque(), deque())
        self._loop = None
        self._lock = threading.RLock()
        self._tie_next = 0

    def update_ready(self, full, local):
        counts = (_integer("full ready groups", full, 0), _integer("local ready groups", local, 0))
        with self._lock:
            self._ready = counts

    def _choose(self):
        # Caller holds the lock and has pruned cancelled queue heads.
        available = [index for index in (0, 1) if self._waiting[index]]
        if not available:
            raise RuntimeError("cannot choose from empty generation queues")
        if len(available) == 1:
            return available[0]
        missing = [
            max(0, (self.targets[i] - self._ready[i]) * self.samples_per_group - self._active[i]) for i in (0, 1)
        ]
        # Compare normalized deficits without floating-point rounding.
        scores = (missing[0] * self.targets[1], missing[1] * self.targets[0])
        if scores[0] != scores[1]:
            return 0 if scores[0] > scores[1] else 1
        if not any(missing):
            # Both stocks are covered: keep prefetch work-conserving, using the
            # configured ratio to balance active trajectories instead of idling.
            occupancy = (self._active[0] * self.targets[1], self._active[1] * self.targets[0])
            if occupancy[0] != occupancy[1]:
                return 0 if occupancy[0] < occupancy[1] else 1
        chosen = self._tie_next
        self._tie_next = 1 - chosen
        return chosen

    def _wake(self):
        # Called only on the owning event loop, with the lock held.
        assert 0 <= sum(self._active) <= self.capacity, "generation capacity accounting is corrupt"
        while sum(self._active) < self.capacity:
            for waiting in self._waiting:
                while waiting and waiting[0].cancelled():
                    waiting.popleft()
            if not any(self._waiting):
                break
            kind = self._choose()
            future = self._waiting[kind].popleft()
            if future.done():
                raise RuntimeError("generation waiter was granted more than once")
            self._active[kind] += 1
            future.set_result(None)

    async def _acquire(self, kind):
        _integer("kind", kind, 0)
        if kind > 1:
            raise ValueError(f"kind must be 0 (full) or 1 (local), got {kind}")
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        with self._lock:
            if self._loop is not None and self._loop is not loop:
                raise RuntimeError("generation gate cannot be shared across event loops")
            self._loop = loop
            self._waiting[kind].append(future)
            self._wake()
        try:
            await future
        except BaseException:
            with self._lock:
                if future.done() and not future.cancelled():
                    # Cancellation raced with the grant, before the context body.
                    self._release(kind)
                else:
                    if future in self._waiting[kind]:
                        self._waiting[kind].remove(future)
                    self._wake()
            raise

    def _release(self, kind):
        _integer("kind", kind, 0)
        if kind > 1:
            raise ValueError(f"kind must be 0 (full) or 1 (local), got {kind}")
        with self._lock:
            if self._loop is not asyncio.get_running_loop() or self._active[kind] <= 0:
                raise RuntimeError("generation lease released without a matching acquisition")
            self._active[kind] -= 1
            self._wake()

    @asynccontextmanager
    async def slot(self, sample):
        metadata = getattr(sample, "metadata", None)
        if metadata is not None and not isinstance(metadata, dict):
            raise TypeError(f"sample {getattr(sample, 'index', None)} metadata must be a dict or None")
        metadata = metadata or {}
        kind = int(bool(metadata.get("ab_local_rollout")) or metadata.get("ab_rollout_kind") == "local")
        await self._acquire(kind)
        try:
            yield
        finally:
            self._release(kind)

    def metrics(self):
        with self._lock:
            values = {"capacity": self.capacity}
            for i, kind in enumerate(("full", "local")):
                values[f"{kind}_active"] = self._active[i]
                values[f"{kind}_waiting"] = sum(not f.cancelled() for f in self._waiting[i])
                values[f"{kind}_ready_groups"] = self._ready[i]
            return {f"rollout/generation_gate/{key}": float(value) for key, value in values.items()}


def make_generation_semaphore(args):
    enabled = os.getenv("AB_BALANCED_GENERATION", "0").strip()
    if enabled not in {"0", "1"}:
        raise ValueError(f"AB_BALANCED_GENERATION must be 0 or 1, got {enabled!r}")
    if enabled == "1":
        for name in ("sglang_server_concurrency", "rollout_num_gpus", "rollout_num_gpus_per_engine"):
            _integer(name, getattr(args, name, None))
    capacity = args.sglang_server_concurrency * args.rollout_num_gpus // args.rollout_num_gpus_per_engine
    if enabled == "0":
        return asyncio.Semaphore(capacity)
    if os.getenv("AB_LOCAL_ROLLOUT_ENABLE", "0").strip().lower() not in {"1", "true", "yes", "y", "on"}:
        raise ValueError("balanced generation requires local rollout enabled")
    targets = []
    for name in ("AB_FULL_TRAIN_GROUPS_PER_BATCH", "AB_LOCAL_TRAIN_GROUPS_PER_BATCH"):
        raw = os.getenv(name, "").strip()
        if not raw.isascii() or not raw.isdecimal():
            raise ValueError(f"{name} must be a positive integer, got {raw!r}")
        targets.append(_integer(name, int(raw)))
    return DeficitSemaphore(capacity, *targets, args.n_samples_per_prompt)


def generation_slot(semaphore, sample):
    """Leave the standard semaphore path unchanged when the feature is off."""
    if not isinstance(semaphore, (asyncio.Semaphore, DeficitSemaphore)):
        raise TypeError(f"unsupported generation semaphore: {type(semaphore).__name__}")
    return semaphore.slot(sample) if isinstance(semaphore, DeficitSemaphore) else semaphore
