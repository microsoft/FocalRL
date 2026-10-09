from __future__ import annotations

import asyncio
import importlib
import sys
import threading
import types
from collections import Counter
from types import SimpleNamespace

import pytest

from miles.utils.types import Sample


def _import_collector(monkeypatch):
    fake_fully_async = types.ModuleType("examples.fully_async.fully_async_rollout")
    fake_fully_async._cached_version = object()
    fake_fully_async.generate_rollout_fully_async = lambda *args, **kwargs: ("base", args, kwargs)
    fake_fully_async.get_global_worker = None
    fake_fully_async.group_oldest_weight_version = lambda group: None
    monkeypatch.setitem(sys.modules, "examples.fully_async.fully_async_rollout", fake_fully_async)
    fake_data_source = types.ModuleType("miles.rollout.data_source")
    fake_data_source.DataSource = object
    monkeypatch.setitem(sys.modules, "miles.rollout.data_source", fake_data_source)
    sys.modules.pop("adaptive_branching.src.deep_research.fully_async_value_cliff_collector", None)
    return importlib.import_module("adaptive_branching.src.deep_research.fully_async_value_cliff_collector")


def _local_group(group_index: int, scores: list[float], *, mode: str = "v6_prm") -> list[Sample]:
    state_by_score = {0.0: "avoid_failed", 0.25: "avoid_only", 0.5: "avoid_only", 1.0: "redirected"}
    return [
        Sample(
            prompt="prefix context",
            response=f"answer-{score}",
            label="GOLD",
            reward={"score": score, "acc": bool(score), "event_rubric_state": state_by_score[score]},
            group_index=group_index,
            index=group_index * 10 + offset,
            status=Sample.Status.COMPLETED,
            metadata={
                "ab_local_rollout": True,
                "ab_rollout_kind": "local",
                "ab_local_reward_mode": mode,
                "agent_turns": 2,
                "ab_local_prefix_assistant_turns": 3,
            },
        )
        for offset, score in enumerate(scores)
    ]


def _full_group(group_index: int, scores: list[float]) -> list[Sample]:
    return [
        Sample(
            prompt="original question",
            response=f"full-answer-{score}",
            label="GOLD",
            reward={"score": score, "acc": bool(score)},
            group_index=group_index,
            index=group_index * 10 + offset,
            status=Sample.Status.COMPLETED,
            metadata={"ab_rollout_kind": "full", "agent_turns": 10},
        )
        for offset, score in enumerate(scores)
    ]


def test_disabled_local_rollout_delegates_to_base_collector(monkeypatch):
    collector = _import_collector(monkeypatch)
    monkeypatch.setenv("AB_LOCAL_ROLLOUT_ENABLE", "0")
    args = object()
    data_buffer = object()
    assert collector.generate_rollout_fully_async(args, 7, data_buffer) == (
        "base",
        (args, 7, data_buffer),
        {"evaluation": False},
    )


@pytest.mark.parametrize("balanced_generation", [False, True])
def test_balanced_queues_wait_for_both_kinds_and_preserve_excess(monkeypatch, balanced_generation):
    collector = _import_collector(monkeypatch)
    monkeypatch.setenv("AB_LOCAL_ROLLOUT_ENABLE", "1")
    monkeypatch.setenv("AB_FULL_TRAIN_GROUPS_PER_BATCH", "2")
    monkeypatch.setenv("AB_LOCAL_TRAIN_GROUPS_PER_BATCH", "2")
    batches = [
        [(index, _full_group(index, [0.0, 1.0])) for index in range(3)],
        [(index, _local_group(index, [0.0, 1.0])) for index in range(3, 6)],
        [(6, _full_group(6, [0.0, 1.0])), (7, _local_group(7, [0.0, 1.0]))],
    ]

    class BatchedWorker:
        def get_completed_groups(self):
            return batches.pop(0) if batches else []

        def get_queue_size(self):
            return sum(len(batch) for batch in batches)

    worker = BatchedWorker()
    if balanced_generation:
        from miles.rollout.deficit_semaphore import DeficitSemaphore

        worker.state = SimpleNamespace(semaphore=DeficitSemaphore(4, 2, 2, 2))
    monkeypatch.setattr(collector, "get_global_worker", lambda args, data_buffer: worker)
    args = SimpleNamespace(
        rollout_global_dataset=True,
        rollout_batch_size=4,
        n_samples_per_prompt=2,
        global_batch_size=8,
        max_weight_staleness=None,
        save_debug_rollout_data=None,
    )
    first = asyncio.run(collector.generate_rollout_async(args, 0, SimpleNamespace()))
    if balanced_generation:
        assert worker.state.semaphore.metrics()["rollout/generation_gate/full_ready_groups"] == 1
        assert worker.state.semaphore.metrics()["rollout/generation_gate/local_ready_groups"] == 1
    second = asyncio.run(collector.generate_rollout_async(args, 1, SimpleNamespace()))
    if balanced_generation:
        assert worker.state.semaphore.metrics()["rollout/generation_gate/full_ready_groups"] == 0
        assert worker.state.semaphore.metrics()["rollout/generation_gate/local_ready_groups"] == 0
    assert [group[0].group_index for group in first.samples] == [0, 3, 1, 4]
    assert [group[0].group_index for group in second.samples] == [2, 5, 6, 7]
    assert first.metrics["rollout/ab_event/train_queue/full_depth_after_take"] == 1.0
    assert first.metrics["rollout/ab_event/train_queue/local_depth_after_take"] == 1.0


@pytest.mark.parametrize(
    ("full", "local", "message"),
    [
        ("0", "2", "positive integer"),
        ("-1", "2", "positive integer"),
        ("two", "2", "positive integer"),
        ("2", "", "must be set together"),
    ],
)
def test_balanced_queue_targets_fail_fast(monkeypatch, full, local, message):
    collector = _import_collector(monkeypatch)
    monkeypatch.setenv("AB_LOCAL_ROLLOUT_ENABLE", "1")
    monkeypatch.setenv("AB_FULL_TRAIN_GROUPS_PER_BATCH", full)
    monkeypatch.setenv("AB_LOCAL_TRAIN_GROUPS_PER_BATCH", local)
    args = SimpleNamespace(rollout_batch_size=4, n_samples_per_prompt=2, global_batch_size=8)
    with pytest.raises(ValueError, match=message):
        collector._train_group_batch_targets(args)


def test_queue_rejects_mixed_groups_and_evicts_oldest_local(monkeypatch):
    collector = _import_collector(monkeypatch)
    queues = collector._FullLocalTrainingQueues()
    targets = collector._TrainGroupBatchTargets(full=1, local=1)
    full = _full_group(0, [1.0])
    locals_ = [_local_group(index, [0.0, 1.0]) for index in (1, 2, 3)]
    collector._enqueue_training_group(queues, full, local_capacity=targets.local_queue_capacity)
    assert collector._enqueue_training_group(queues, locals_[0], local_capacity=2) is None
    assert collector._enqueue_training_group(queues, locals_[1], local_capacity=2) is None
    assert collector._enqueue_training_group(queues, locals_[2], local_capacity=2) is locals_[0]
    with pytest.raises(ValueError, match="cannot mix"):
        collector._enqueue_training_group(queues, [full[0], locals_[0][0]], local_capacity=2)


@pytest.mark.parametrize(
    ("mode", "expected_turns", "force_final"),
    [("v6_prm", 5, False), ("v6p", 5, False), ("v7_prm", 5, False), ("terminal", 30, True)],
)
def test_local_child_metadata_supports_only_retained_modes(monkeypatch, mode, expected_turns, force_final):
    collector = _import_collector(monkeypatch)
    monkeypatch.setenv("AB_EVENT_HIDDEN_MAX_TURNS", "5")
    monkeypatch.setenv("AGENT_MAX_TURNS", "30")
    parent = Sample(
        prompt="question",
        label="GOLD",
        group_index=1,
        index=2,
        metadata={"ab_original_question": "question", "agent_history_mode": "all"},
    )
    prefix = [{"role": "user", "content": "question"}, {"role": "assistant", "content": "background"}]
    spec = {
        "branch_type": "value_cliff_local",
        "prefix_messages": prefix,
        "event": {"event_turn": 2},
        "event_turn": 2,
        "locator_version": "v6",
        "reward_mode": mode,
        "recovery_rubric": {"avoid_error": "stop X", "redirect": "test Y"},
        "value_drop_reason": "X caused a value drop",
    }
    metadata = collector._local_child_metadata(parent, prefix, branch_spec=spec)
    assert metadata["ab_local_reward_mode"] == mode
    assert metadata["agent_max_turns"] == expected_turns
    assert metadata["agent_force_final_on_max_turns"] is force_final
    assert metadata["ab_event_recovery_rubric"] == {"avoid_error": "stop X", "redirect": "test Y"}


def test_local_groups_preserve_original_label(monkeypatch):
    collector = _import_collector(monkeypatch)
    monkeypatch.setenv("AB_LOCAL_ROLLOUT_ENABLE", "1")
    parent = Sample(
        prompt="question",
        label="ORIGINAL GOLD",
        group_index=1,
        index=2,
        generate_function_path="agent",
        metadata={
            "ab_branch_spec": {
                "branch_type": "value_cliff_local",
                "prefix_messages": [{"role": "user", "content": "question"}],
                "event": {"event_turn": 1},
                "event_turn": 1,
                "locator_version": "v6",
                "reward_mode": "v6_prm",
                "recovery_rubric": {"avoid_error": "stop X", "redirect": "test Y"},
                "value_drop_reason": "X caused a value drop",
            }
        },
    )
    data_source = SimpleNamespace(sample_group_index=10, sample_index=20, identity_lock=threading.Lock())
    groups = collector._build_local_groups(SimpleNamespace(n_samples_per_prompt=2), data_source, [parent])
    assert len(groups) == 1
    assert [child.label for child in groups[0]] == ["ORIGINAL GOLD", "ORIGINAL GOLD"]


def test_zero_variance_group_is_measured_but_not_filtered(monkeypatch):
    collector = _import_collector(monkeypatch)
    group = _local_group(1, [0.0, 0.0])
    stats = Counter()
    states = collector._event_local_states(group)
    collector._accumulate_event_local_stats(stats, group, "v6_prm", states)
    metrics = collector._build_event_rollout_metrics([group], processed_full_groups=0, event_stats=stats)
    assert metrics["rollout/ab_event/v6_prm/informative_group_rate"] == 0.0
    assert not any("dropped" in key for key in metrics)


def test_v6p_metrics_include_local_and_outcome_states(monkeypatch):
    collector = _import_collector(monkeypatch)
    group = _local_group(1, [0.0, 0.5, 1.0], mode="v6p")
    group[0].reward["event_rubric_state"] = "outcome_wrong"
    group[2].reward["event_rubric_state"] = "outcome_correct"
    stats = Counter()
    states = collector._event_local_states(group)
    collector._accumulate_event_local_stats(stats, group, "v6p", states)

    metrics = collector._build_event_rollout_metrics([group], processed_full_groups=0, event_stats=stats)

    assert metrics["rollout/ab_event/group_ratio/v6p"] == 1.0
    assert metrics["rollout/ab_event/v6p/state/outcome_wrong_rate"] == pytest.approx(1 / 3)
    assert metrics["rollout/ab_event/v6p/state/avoid_only_rate"] == pytest.approx(1 / 3)
    assert metrics["rollout/ab_event/v6p/state/outcome_correct_rate"] == pytest.approx(1 / 3)


@pytest.mark.parametrize(
    "statuses, expected",
    [
        (["Submitted"], 1.0),
        (["LocalHorizon"], 0.0),
        (["Submitted", "Submitted"], 1.0),
        (["Submitted", "LocalHorizon", "ContextLimit", "HTTPStatusError"], 0.25),
    ],
)
def test_swe_local_natural_finish_rate(monkeypatch, statuses, expected):
    collector = _import_collector(monkeypatch)
    # All-zero PRM rewards and training exclusions must not change finish telemetry.
    local = _local_group(1, [0.0] * len(statuses))
    for sample, status in zip(local, statuses, strict=True):
        sample.metadata.update(swe_replay_n_calls=3, exit_status=status, agent_excluded_from_training=True)
    full = _full_group(2, [1.0])
    full[0].metadata.update(swe_replay_n_calls=3, exit_status="Submitted")
    research = _local_group(3, [1.0])
    stats = Counter()
    for group, is_local in [(local, True), (full, False), (research, True)]:
        collector._accumulate_full_local_rollout_stats(stats, group, is_local=is_local)
    metrics = collector._build_event_rollout_metrics(
        [local, full, research], processed_full_groups=1, event_stats=stats
    )
    assert metrics["rollout/swe/local_sample_count"] == len(statuses)
    assert metrics["rollout/swe/local_natural_finish_count"] == statuses.count("Submitted")
    assert metrics["rollout/swe/local_natural_finish_rate"] == expected


def test_swe_local_finish_missing_data_is_not_a_zero_rate(monkeypatch):
    collector = _import_collector(monkeypatch)
    metrics = collector._build_event_rollout_metrics([], processed_full_groups=0, event_stats=None)
    assert not any(key.startswith("rollout/swe/local_") for key in metrics)
    group = _local_group(1, [0.0])
    group[0].metadata["swe_replay_n_calls"] = 3
    for status in (None, "", " ", 1):
        group[0].metadata["exit_status"] = status
        with pytest.raises(ValueError, match="missing exit_status"):
            collector._accumulate_full_local_rollout_stats(Counter(), group, is_local=True)


@pytest.mark.parametrize("total, submitted", [(0, 1), (-1, 0), (1, -1), (True, 0), (1, 0.5)])
def test_swe_local_finish_invalid_counts_fail(monkeypatch, total, submitted):
    collector = _import_collector(monkeypatch)
    with pytest.raises(ValueError, match="invalid SWE local finish counts"):
        collector._build_event_rollout_metrics(
            [],
            processed_full_groups=0,
            event_stats={"swe_local_sample_count": total, "swe_local_natural_finish_count": submitted},
        )


class _RecordingLock:
    def __init__(self):
        self._lock = threading.Lock()
        self.acquisitions = 0

    def __enter__(self):
        self._lock.acquire()
        self.acquisitions += 1
        return self

    def __exit__(self, exc_type, exc, tb):
        self._lock.release()
        return False


def test_local_and_source_groups_share_identity_lock(monkeypatch):
    collector = _import_collector(monkeypatch)
    lock = _RecordingLock()
    data_source = SimpleNamespace(sample_group_index=0, sample_index=0, identity_lock=lock)
    first = [Sample() for _ in range(4)]
    second = [Sample() for _ in range(4)]
    collector._assign_new_group_identity(data_source, first)
    collector._assign_new_group_identity(data_source, second)
    assert lock.acquisitions == 2
    assert [sample.index for sample in [*first, *second]] == list(range(8))
    assert {sample.group_index for sample in first} == {0}
    assert {sample.group_index for sample in second} == {1}


def test_generation_gate_target_mismatch_fails_before_collection(monkeypatch):
    from miles.rollout.deficit_semaphore import DeficitSemaphore

    collector = _import_collector(monkeypatch)
    monkeypatch.setenv("AB_LOCAL_ROLLOUT_ENABLE", "1")
    monkeypatch.setenv("AB_FULL_TRAIN_GROUPS_PER_BATCH", "1")
    monkeypatch.setenv("AB_LOCAL_TRAIN_GROUPS_PER_BATCH", "1")
    worker = SimpleNamespace(state=SimpleNamespace(semaphore=DeficitSemaphore(2, 1, 2, 2)))
    monkeypatch.setattr(collector, "get_global_worker", lambda *args: worker)
    args = SimpleNamespace(
        rollout_global_dataset=True, rollout_batch_size=2, n_samples_per_prompt=2, global_batch_size=4
    )
    with pytest.raises(ValueError, match="targets must match"):
        asyncio.run(collector.generate_rollout_async(args, 0, SimpleNamespace()))


def _gate_collector_fixture(monkeypatch, *, gate_n=2):
    from miles.rollout.deficit_semaphore import DeficitSemaphore

    collector = _import_collector(monkeypatch)
    monkeypatch.setenv("AB_LOCAL_ROLLOUT_ENABLE", "1")
    monkeypatch.setenv("AB_FULL_TRAIN_GROUPS_PER_BATCH", "1")
    monkeypatch.setenv("AB_LOCAL_TRAIN_GROUPS_PER_BATCH", "1")
    worker = SimpleNamespace(
        state=SimpleNamespace(semaphore=DeficitSemaphore(2, 1, 1, gate_n)), get_queue_size=lambda: 0
    )
    monkeypatch.setattr(collector, "get_global_worker", lambda *args: worker)
    args = SimpleNamespace(
        rollout_global_dataset=True,
        rollout_batch_size=2,
        n_samples_per_prompt=2,
        global_batch_size=4,
        max_weight_staleness=None,
        save_debug_rollout_data=None,
    )
    return collector, worker, args


def test_generation_gate_sample_count_mismatch_fails_before_collection(monkeypatch):
    collector, worker, args = _gate_collector_fixture(monkeypatch, gate_n=8)
    with pytest.raises(ValueError, match="samples_per_group"):
        asyncio.run(collector.generate_rollout_async(args, 0, SimpleNamespace()))


def test_exception_after_partial_enqueue_publishes_stock_and_releases_lock(monkeypatch):
    collector, worker, args = _gate_collector_fixture(monkeypatch)
    # A valid completed group followed by a malformed one: collection must fail,
    # but the persistent queue and the scheduler cannot disagree about inventory.
    worker.get_completed_groups = lambda: [(0, _full_group(0, [0.0, 1.0])), (1, [])]
    with pytest.raises(ValueError, match="non-empty list"):
        asyncio.run(collector.generate_rollout_async(args, 0, SimpleNamespace()))
    queues = collector._worker_training_queues(worker)
    assert len(queues.full) == 1
    assert worker.state.semaphore.metrics()["rollout/generation_gate/full_ready_groups"] == 1
    queues.acquire_collector()
    queues.release_collector()


def test_restored_inventory_concurrent_collector_and_cancellation(monkeypatch):
    collector, worker, args = _gate_collector_fixture(monkeypatch)
    queues = collector._worker_training_queues(worker)
    group = _full_group(0, [0.0, 1.0])
    queues.full.append(group)
    worker.get_completed_groups = lambda: []
    args.max_weight_staleness = 1

    async def scenario():
        entered = asyncio.Event()

        async def version(_args):
            entered.set()
            await asyncio.Event().wait()

        monkeypatch.setattr(collector, "_cached_version", SimpleNamespace(get=version))
        first = asyncio.create_task(collector.generate_rollout_async(args, 0, SimpleNamespace()))
        await asyncio.wait_for(entered.wait(), 1)
        assert worker.state.semaphore.metrics()["rollout/generation_gate/full_ready_groups"] == 1
        snapshot = worker.state.semaphore.metrics()
        with pytest.raises(RuntimeError, match="concurrent collectors"):
            await collector.generate_rollout_async(args, 1, SimpleNamespace())
        assert worker.state.semaphore.metrics() == snapshot
        assert queues._collector_lock.locked()
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first
        assert not queues._collector_lock.locked()
        assert list(queues.full) == [group]
        args.max_weight_staleness = None
        local = _local_group(1, [0.0, 1.0])
        worker.get_completed_groups = lambda: [(1, local)]
        result = await collector.generate_rollout_async(args, 2, SimpleNamespace())
        assert result.samples[0] is group and result.samples[1] is local
        assert worker.state.semaphore.metrics()["rollout/generation_gate/full_ready_groups"] == 0
        assert worker.state.semaphore.metrics()["rollout/generation_gate/local_ready_groups"] == 0

    asyncio.run(asyncio.wait_for(scenario(), 5))


def test_inventory_counts_retained_groups_after_overflow_and_batch_take(monkeypatch):
    collector, worker, args = _gate_collector_fixture(monkeypatch)
    groups = [_full_group(0, [0.0, 1.0]), *[_local_group(i, [0.0, 1.0]) for i in range(1, 5)]]
    worker.get_completed_groups = lambda: list(enumerate(groups))
    snapshots = []
    gate = worker.state.semaphore
    original = gate.update_ready

    def record(full, local):
        snapshots.append((full, local))
        original(full, local)

    monkeypatch.setattr(gate, "update_ready", record)
    result = asyncio.run(collector.generate_rollout_async(args, 0, SimpleNamespace()))
    assert result.samples[0] is groups[0] and result.samples[1] is groups[3]
    assert (1, 2) in snapshots and snapshots[-1] == (0, 1)
    assert result.metrics["rollout/ab_event/train_queue/local_overflow_evicted_count"] == 2
    assert list(collector._worker_training_queues(worker).local) == [groups[4]]


def test_snapshot_failure_does_not_leave_collector_lock_held(monkeypatch):
    collector, worker, args = _gate_collector_fixture(monkeypatch)

    def fail(*_args):
        raise RuntimeError("controlled snapshot failure")

    monkeypatch.setattr(worker.state.semaphore, "update_ready", fail)
    with pytest.raises(RuntimeError, match="controlled snapshot failure"):
        asyncio.run(collector.generate_rollout_async(args, 0, SimpleNamespace()))
    queues = collector._worker_training_queues(worker)
    queues.acquire_collector()
    queues.release_collector()


def test_recycled_aborted_and_stale_groups_do_not_inflate_ready_stock(monkeypatch):
    collector, worker, args = _gate_collector_fixture(monkeypatch)
    aborted = _full_group(0, [0.0, 1.0])
    aborted[0].status = Sample.Status.ABORTED
    stale = _local_group(1, [0.0, 1.0])
    full, local = _full_group(2, [0.0, 1.0]), _local_group(3, [0.0, 1.0])
    worker.get_completed_groups = lambda: list(enumerate([aborted, stale, full, local]))
    args.max_weight_staleness = 1
    recycled = []
    buffer = SimpleNamespace(add_samples=lambda groups: recycled.extend(groups))

    async def version(_args):
        return 10

    monkeypatch.setattr(collector, "_cached_version", SimpleNamespace(get=version))
    monkeypatch.setattr(collector, "group_oldest_weight_version", lambda group: 1 if group is stale else 10)
    result = asyncio.run(collector.generate_rollout_async(args, 0, buffer))
    assert recycled == [aborted, stale]
    assert result.samples == [full, local]
    assert result.metrics["rollout/ab_event/train_queue/full_depth_before_take"] == 1
    assert result.metrics["rollout/ab_event/train_queue/local_depth_before_take"] == 1
    assert worker.state.semaphore.metrics()["rollout/generation_gate/full_ready_groups"] == 0
    assert worker.state.semaphore.metrics()["rollout/generation_gate/local_ready_groups"] == 0
