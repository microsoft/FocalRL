"""Fully-async collector that adds Value-Cliff local groups."""

from __future__ import annotations

import asyncio
import copy
import logging
import math
import os
import threading
import time
from collections import Counter, deque
from dataclasses import dataclass
from typing import Any

from adaptive_branching.src.deep_research.branch_sampling import BRANCH_TYPE, selection_policy, validate_branch_supply
from adaptive_branching.src.deep_research.value_cliff_online import (
    full_rollout_max_turns,
    local_horizon_max_turns,
    local_rollout_enabled,
)
from adaptive_branching.src.deep_research.value_cliff_rubric import (
    TERMINAL_REWARD_MODE,
    V6_PRM_REWARD_MODE,
    V6P_REWARD_MODE,
    V7_PRM_REWARD_MODE,
    validate_recovery_rubric,
)
from examples.fully_async.fully_async_rollout import (
    _cached_version,
    get_global_worker,
    group_oldest_weight_version,
)
from examples.fully_async.fully_async_rollout import (
    generate_rollout_fully_async as generate_base_rollout_fully_async,
)
from miles.ray.rollout.debug_data import (
    dump_rollout_messages_jsonl,
    rollout_messages_jsonl_path,
    strip_rollout_heavy_metadata,
)
from miles.rollout.base_types import RolloutFnTrainOutput
from miles.rollout.data_source import DataSource
from miles.rollout.deficit_semaphore import DeficitSemaphore
from miles.utils.async_utils import run
from miles.utils.types import Sample

logger = logging.getLogger(__name__)
_DEBUG_FILES_INITIALIZED: set[tuple[str, int]] = set()
_LOCAL_AGENT_METADATA_CARRY_KEYS = {
    "agent_history_mode",
    "agent_keep_tool_results",
    "agent_logprobs",
    "agent_max_turns",
    "agent_return_messages",
}
_LOCAL_AGENT_METADATA_CARRY_PREFIXES = ("agent_chat_",)
_LOCAL_METADATA_DROP_KEYS = {
    "ab_branch_spec",
    "accumulated_token_ids",
    "max_trim_tokens",
    "messages",
    "session_server_id",
    "session_server_instance_id",
    "start_rollout_id",
}
_FULL_TRAIN_GROUPS_PER_BATCH_ENV = "AB_FULL_TRAIN_GROUPS_PER_BATCH"
_LOCAL_TRAIN_GROUPS_PER_BATCH_ENV = "AB_LOCAL_TRAIN_GROUPS_PER_BATCH"
_LOCAL_TRAIN_QUEUE_CAPACITY_BATCHES = 2
_WORKER_TRAINING_QUEUES_ATTR = "_ab_full_local_training_queues"
_WORKER_TRAINING_QUEUES_INIT_LOCK = threading.Lock()


@dataclass(frozen=True)
class _TrainGroupBatchTargets:
    full: int
    local: int

    def __post_init__(self) -> None:
        if isinstance(self.full, bool) or not isinstance(self.full, int) or self.full <= 0:
            raise ValueError(f"full training-group target must be a positive integer, got {self.full!r}")
        if isinstance(self.local, bool) or not isinstance(self.local, int) or self.local <= 0:
            raise ValueError(f"local training-group target must be a positive integer, got {self.local!r}")

    @property
    def total(self) -> int:
        return self.full + self.local

    @property
    def local_queue_capacity(self) -> int:
        return _LOCAL_TRAIN_QUEUE_CAPACITY_BATCHES * self.local


class _FullLocalTrainingQueues:
    """Persistent completed-group queues owned by one async rollout worker."""

    def __init__(self) -> None:
        self.full: deque[list[Sample]] = deque()
        self.local: deque[list[Sample]] = deque()
        self._collector_lock = threading.Lock()

    def acquire_collector(self) -> None:
        if not self._collector_lock.acquire(blocking=False):
            raise RuntimeError("concurrent collectors cannot consume the same full/local training queues")

    def release_collector(self) -> None:
        if not self._collector_lock.locked():
            raise RuntimeError("full/local training queue collector lock is not held")
        self._collector_lock.release()


async def generate_rollout_async(args, rollout_id: int, data_buffer: DataSource) -> RolloutFnTrainOutput:
    if not bool(args.rollout_global_dataset):
        raise ValueError("Value-Cliff async rollout requires rollout_global_dataset")

    worker = get_global_worker(args, data_buffer)
    targets = _train_group_batch_targets(args)
    target_data_size = args.rollout_batch_size
    training_queues = _worker_training_queues(worker) if targets is not None else None
    gate = getattr(getattr(worker, "state", None), "semaphore", None)
    gate = gate if isinstance(gate, DeficitSemaphore) else None
    if gate is not None:
        if targets is None or gate.targets != (targets.full, targets.local):
            raise ValueError("generation gate targets must match collector full/local targets")
        if gate.samples_per_group != args.n_samples_per_prompt:
            raise ValueError("generation gate samples_per_group must match collector n_samples_per_prompt")
    if training_queues is not None:
        training_queues.acquire_collector()

    try:
        if gate is not None:
            gate.update_ready(len(training_queues.full), len(training_queues.local))
        data: list[list[Sample]] = []
        completed_groups: dict[int, list[Sample]] = {}
        do_print = True
        stale_groups_recycled = 0
        staleness_values: list[int] = []
        use_staleness_filter = getattr(args, "max_weight_staleness", None) is not None
        processed_full_groups = 0
        branch_supply_misses = 0
        event_stats: Counter[str] = Counter()

        _initialize_debug_jsonl(args, rollout_id)

        if targets is None:
            print(f"Starting adaptive async rollout generation for {target_data_size} groups")
        else:
            print(
                "Starting balanced adaptive async rollout generation for "
                f"full={targets.full}, local={targets.local}, total={targets.total} groups"
            )
        print(f"Global worker queue size: {worker.get_queue_size()}")
        if use_staleness_filter:
            print(f"Staleness filter enabled: max_weight_staleness={args.max_weight_staleness}")

        start_time = time.time()
        last_progress_time = start_time
        no_progress_timeout = 30.0

        while not _training_batch_ready(data, training_queues, targets, target_data_size):
            completed = worker.get_completed_groups()
            made_progress = False
            for group_id, group in completed:
                if group_id in completed_groups:
                    raise RuntimeError(f"duplicate completed rollout group id {group_id}")
                completed_groups[group_id] = group
                made_progress = True
            if made_progress:
                last_progress_time = time.time()

            current_engine_version = None
            if use_staleness_filter:
                current_engine_version = await _cached_version.get(args)

            processed_any = False
            for group_id in list(completed_groups.keys()):
                if targets is None and len(data) >= target_data_size:
                    break
                group = completed_groups.pop(group_id)
                if not isinstance(group, list) or not group:
                    raise ValueError(f"completed rollout group {group_id} must be a non-empty list")

                try:
                    any_aborted = any(sample.status == Sample.Status.ABORTED for sample in group)
                except Exception:
                    any_aborted = False
                if any_aborted:
                    try:
                        for sample in group:
                            sample.reset_for_retry()
                        data_buffer.add_samples([group])
                        print(f"Returned aborted group {group_id} to data buffer", flush=True)
                    except Exception as exc:  # noqa: BLE001
                        print(f"Failed to return aborted group {group_id} to buffer: {exc}", flush=True)
                    continue

                oldest = group_oldest_weight_version(group)
                if oldest is not None and current_engine_version is not None:
                    staleness = current_engine_version - oldest
                    staleness_values.append(staleness)
                    if staleness > args.max_weight_staleness:
                        try:
                            for sample in group:
                                sample.reset_for_retry()
                            data_buffer.add_samples([group])
                        except Exception as exc:  # noqa: BLE001
                            logger.warning("Failed to recycle stale group %s: %s", group_id, exc)
                        stale_groups_recycled += 1
                        logger.info(
                            "Recycled stale group %s (oldest_version=%s, current=%s, staleness=%s > max=%s)",
                            group_id,
                            oldest,
                            current_engine_version,
                            staleness,
                            args.max_weight_staleness,
                        )
                        continue

                local_groups = _build_local_groups(args, data_buffer, group)
                if selection_policy() != "value_cliff" and not _group_is_local(group):
                    branch_supply_misses = 0 if local_groups else branch_supply_misses + 1
                    validate_branch_supply(branch_supply_misses, targets.full if targets else target_data_size)
                if local_groups:
                    try:
                        data_buffer.add_samples(local_groups)
                        logger.info("Enqueued %d adaptive local group(s)", len(local_groups))
                    except Exception as exc:  # noqa: BLE001
                        if selection_policy() != "value_cliff":
                            raise RuntimeError("failed to enqueue terminal branch group") from exc
                        logger.warning("Failed to enqueue local groups: %s", exc, exc_info=True)

                if training_queues is None:
                    processed_full_groups += int(
                        _prepare_selected_training_group(args, rollout_id, group, event_stats)
                    )
                    if do_print:
                        _print_first_rollout_group(group)
                        do_print = False
                    data.append(group)
                else:
                    assert targets is not None
                    evicted = _enqueue_training_group(
                        training_queues,
                        group,
                        local_capacity=targets.local_queue_capacity,
                    )
                    if evicted is not None:
                        event_stats["local_queue_overflow_evicted_groups"] += 1
                        logger.warning(
                            "Evicted oldest completed local group %s to keep queue at capacity=%d",
                            evicted[0].group_index,
                            targets.local_queue_capacity,
                        )
                processed_any = True

            if gate is not None:
                gate.update_ready(len(training_queues.full), len(training_queues.local))
            current_time = time.time()
            if current_time - last_progress_time > no_progress_timeout:
                print(
                    f"Warning: No progress for {no_progress_timeout}s. "
                    f"Queue size: {worker.get_queue_size()}, "
                    f"Collected: {_training_collection_progress(data, training_queues, targets, target_data_size)}"
                )
                last_progress_time = current_time

            if not processed_any:
                await asyncio.sleep(0.01)

        queue_metrics: dict[str, float] = {}
        if training_queues is not None:
            assert targets is not None
            full_depth_before = len(training_queues.full)
            local_depth_before = len(training_queues.local)
            data = _take_balanced_training_batch(training_queues, targets)
            if gate is not None:
                gate.update_ready(len(training_queues.full), len(training_queues.local))
            for group in data:
                processed_full_groups += int(_prepare_selected_training_group(args, rollout_id, group, event_stats))
            if data:
                _print_first_rollout_group(data[0])
            queue_metrics = {
                "rollout/ab_event/train_queue/full_target": float(targets.full),
                "rollout/ab_event/train_queue/local_target": float(targets.local),
                "rollout/ab_event/train_queue/full_selected": float(targets.full),
                "rollout/ab_event/train_queue/local_selected": float(targets.local),
                "rollout/ab_event/train_queue/local_capacity": float(targets.local_queue_capacity),
                "rollout/ab_event/train_queue/local_overflow_evicted_count": float(
                    event_stats.get("local_queue_overflow_evicted_groups", 0)
                ),
                "rollout/ab_event/train_queue/full_depth_before_take": float(full_depth_before),
                "rollout/ab_event/train_queue/local_depth_before_take": float(local_depth_before),
                "rollout/ab_event/train_queue/full_depth_after_take": float(len(training_queues.full)),
                "rollout/ab_event/train_queue/local_depth_after_take": float(len(training_queues.local)),
            }

        if gate is not None:
            queue_metrics.update(gate.metrics())
        duration = time.time() - start_time
        print(f"Rollout completed in {duration:.2f}s! Global worker queue size: {worker.get_queue_size()}")
        if stale_groups_recycled > 0 or staleness_values:
            avg_staleness = sum(staleness_values) / len(staleness_values) if staleness_values else 0
            print(
                f"Staleness stats: recycled={stale_groups_recycled}, "
                f"avg_staleness={avg_staleness:.1f}, "
                f"max_staleness={max(staleness_values) if staleness_values else 0}"
            )

        if data:
            print(
                f"Finish rollout: {[str(data[-1][0].prompt) + data[-1][0].response]}, "
                f"label: {data[-1][0].label}, reward: {data[-1][0].reward}",
                flush=True,
            )

        if targets is None:
            data = sorted(data, key=lambda group: group[0].index)
        if len(data) != target_data_size:
            raise RuntimeError(f"collector returned {len(data)} groups, expected {target_data_size}")
        metrics = _build_rollout_metrics(
            data,
            processed_full_groups=processed_full_groups,
            event_stats=event_stats,
        )
        metrics.update(queue_metrics)
        return RolloutFnTrainOutput(samples=data, metrics=metrics)
    finally:
        if training_queues is not None:
            try:
                if gate is not None:
                    gate.update_ready(len(training_queues.full), len(training_queues.local))
            finally:
                training_queues.release_collector()


def _train_group_batch_targets(args) -> _TrainGroupBatchTargets | None:
    raw_full = os.getenv(_FULL_TRAIN_GROUPS_PER_BATCH_ENV)
    raw_local = os.getenv(_LOCAL_TRAIN_GROUPS_PER_BATCH_ENV)
    full_is_set = raw_full is not None and raw_full.strip() != ""
    local_is_set = raw_local is not None and raw_local.strip() != ""
    if not full_is_set and not local_is_set:
        return None
    if full_is_set != local_is_set:
        raise ValueError(
            f"{_FULL_TRAIN_GROUPS_PER_BATCH_ENV} and {_LOCAL_TRAIN_GROUPS_PER_BATCH_ENV} must be set together"
        )
    if not local_rollout_enabled():
        raise ValueError("balanced full/local training queues require AB_LOCAL_ROLLOUT_ENABLE=1")

    assert raw_full is not None and raw_local is not None
    targets = _TrainGroupBatchTargets(
        full=_parse_positive_env_int(_FULL_TRAIN_GROUPS_PER_BATCH_ENV, raw_full),
        local=_parse_positive_env_int(_LOCAL_TRAIN_GROUPS_PER_BATCH_ENV, raw_local),
    )
    rollout_batch_size = getattr(args, "rollout_batch_size", None)
    if isinstance(rollout_batch_size, bool) or not isinstance(rollout_batch_size, int) or rollout_batch_size <= 0:
        raise ValueError(f"rollout_batch_size must be a positive integer, got {rollout_batch_size!r}")
    if rollout_batch_size != targets.total:
        raise ValueError(
            f"rollout_batch_size={rollout_batch_size} must equal full+local training-group targets "
            f"({targets.full}+{targets.local}={targets.total})"
        )
    samples_per_prompt = getattr(args, "n_samples_per_prompt", None)
    if isinstance(samples_per_prompt, bool) or not isinstance(samples_per_prompt, int) or samples_per_prompt <= 0:
        raise ValueError(f"n_samples_per_prompt must be a positive integer, got {samples_per_prompt!r}")
    expected_global_batch_size = targets.total * samples_per_prompt
    global_batch_size = getattr(args, "global_batch_size", None)
    if global_batch_size != expected_global_batch_size:
        raise ValueError(
            f"global_batch_size={global_batch_size!r} must equal "
            f"(full+local groups)*n_samples_per_prompt={expected_global_batch_size} "
            "so the balanced collector batch is trained in one optimizer step"
        )
    return targets


def _parse_positive_env_int(name: str, raw_value: str) -> int:
    value = raw_value.strip()
    if not value or any(character not in "0123456789" for character in value):
        raise ValueError(f"{name} must be a positive integer, got {raw_value!r}")
    parsed = int(value)
    if parsed <= 0:
        raise ValueError(f"{name} must be a positive integer, got {raw_value!r}")
    return parsed


def _worker_training_queues(worker: Any) -> _FullLocalTrainingQueues:
    with _WORKER_TRAINING_QUEUES_INIT_LOCK:
        queues = getattr(worker, _WORKER_TRAINING_QUEUES_ATTR, None)
        if queues is None:
            queues = _FullLocalTrainingQueues()
            try:
                setattr(worker, _WORKER_TRAINING_QUEUES_ATTR, queues)
            except Exception as exc:  # noqa: BLE001
                raise TypeError("async rollout worker must allow persistent full/local training queue state") from exc
        if not isinstance(queues, _FullLocalTrainingQueues):
            raise TypeError(
                f"worker attribute {_WORKER_TRAINING_QUEUES_ATTR} has unexpected type {type(queues).__name__}"
            )
        return queues


def _training_batch_ready(
    data: list[list[Sample]],
    queues: _FullLocalTrainingQueues | None,
    targets: _TrainGroupBatchTargets | None,
    target_data_size: int,
) -> bool:
    if (queues is None) != (targets is None):
        raise ValueError("training queues and targets must either both be set or both be absent")
    if queues is None:
        return len(data) >= target_data_size
    if data:
        raise ValueError("balanced collection must not populate the legacy training list before selection")
    assert targets is not None
    return len(queues.full) >= targets.full and len(queues.local) >= targets.local


def _training_collection_progress(
    data: list[list[Sample]],
    queues: _FullLocalTrainingQueues | None,
    targets: _TrainGroupBatchTargets | None,
    target_data_size: int,
) -> str:
    if queues is None:
        return f"{len(data)}/{target_data_size}"
    if targets is None:
        raise ValueError("balanced queue progress requires training-group targets")
    return f"full={len(queues.full)}/{targets.full}, local={len(queues.local)}/{targets.local}"


def _enqueue_training_group(
    queues: _FullLocalTrainingQueues,
    group: list[Sample],
    *,
    local_capacity: int,
) -> list[Sample] | None:
    if not isinstance(group, list) or not group:
        raise ValueError("training group must be a non-empty list")
    if isinstance(local_capacity, bool) or not isinstance(local_capacity, int) or local_capacity <= 0:
        raise ValueError(f"local training queue capacity must be a positive integer, got {local_capacity!r}")
    local_flags = {_sample_is_local(sample) for sample in group}
    if len(local_flags) != 1:
        raise ValueError("training group cannot mix full and local samples")
    if local_flags == {False}:
        queues.full.append(group)
        return None

    queues.local.append(group)
    if len(queues.local) <= local_capacity:
        return None
    evicted = queues.local.popleft()
    if evicted is group:
        raise RuntimeError("local queue overflow evicted the newly appended group")
    return evicted


def _take_balanced_training_batch(
    queues: _FullLocalTrainingQueues,
    targets: _TrainGroupBatchTargets,
) -> list[list[Sample]]:
    if len(queues.full) < targets.full or len(queues.local) < targets.local:
        raise RuntimeError(
            "cannot take balanced training batch: "
            f"full={len(queues.full)}/{targets.full}, local={len(queues.local)}/{targets.local}"
        )
    selected_full = [queues.full.popleft() for _ in range(targets.full)]
    selected_local = [queues.local.popleft() for _ in range(targets.local)]
    batch: list[list[Sample]] = []
    for index in range(max(targets.full, targets.local)):
        if index < targets.full:
            batch.append(selected_full[index])
        if index < targets.local:
            batch.append(selected_local[index])
    if len(batch) != targets.total:
        raise RuntimeError(f"balanced batch contains {len(batch)} groups, expected {targets.total}")
    return batch


def _prepare_selected_training_group(
    args,
    rollout_id: int,
    group: list[Sample],
    event_stats: Counter[str],
) -> bool:
    if not group:
        raise ValueError("selected training group must not be empty")
    is_local_group = _group_is_local(group)
    _dump_group_jsonl(args, rollout_id, group)
    _accumulate_full_local_rollout_stats(event_stats, group, is_local=is_local_group)
    if not is_local_group:
        _accumulate_event_locator_stats(event_stats, group)
    local_mode = _event_local_group_mode(group)
    local_states = _event_local_states(group)
    if local_mode is not None:
        _accumulate_event_local_stats(event_stats, group, local_mode, local_states)
    _cleanup_training_metadata(group)
    return not is_local_group


def _print_first_rollout_group(group: list[Sample]) -> None:
    if not group:
        raise ValueError("cannot print an empty rollout group")
    sample = group[0]
    print(
        f"First rollout sample: {[str(sample.prompt) + str(sample.response)]}, "
        f"label: {sample.label}, reward: {sample.reward}",
        flush=True,
    )


def generate_rollout_fully_async(args, rollout_id, data_buffer: DataSource, evaluation=False):
    if evaluation:
        raise ValueError("Evaluation mode not supported in Value-Cliff async rollout")
    if not local_rollout_enabled():
        # 不开 Local：使用通用 collector，当前 SWE 配置下收集 Full
        return generate_base_rollout_fully_async(
            args,
            rollout_id,
            data_buffer,
            evaluation=False,
        )
    # 开启 Local：使用增强 collector
    # 收集 Full，创建 Local 任务，再收集完成的 Local，组成混合 batch
    return run(generate_rollout_async(args, rollout_id, data_buffer))


def _build_local_groups(args, data_buffer: DataSource, group: list[Sample]) -> list[list[Sample]]:
    if not local_rollout_enabled():
        return []

    local_groups: list[list[Sample]] = []
    for parent in group:
        metadata = parent.metadata if isinstance(parent.metadata, dict) else {}
        if metadata.get("ab_local_rollout") or metadata.get("ab_rollout_kind") == "local":
            continue
        branch_spec = metadata.get("ab_branch_spec")
        if not isinstance(branch_spec, dict) or branch_spec.get("branch_type") not in {"value_cliff_local", BRANCH_TYPE}:
            continue
        sampled = branch_spec["branch_type"] == BRANCH_TYPE
        if sampled:
            if type(args.n_samples_per_prompt) is not int or args.n_samples_per_prompt <= 0:
                raise ValueError("terminal branch group size must be a positive integer")
            emitted = getattr(data_buffer, "_ab_emitted_terminal_parents", set())
            source_id = parent.group_index
            if source_id in emitted:
                continue
        prefix = branch_spec.get("prefix_messages")
        if not isinstance(prefix, list) or not prefix:
            raise ValueError("value-cliff local branch is missing prefix_messages")
        reward_mode = str(branch_spec.get("reward_mode") or "").strip().lower()
        if reward_mode not in {V6_PRM_REWARD_MODE, V6P_REWARD_MODE, V7_PRM_REWARD_MODE, TERMINAL_REWARD_MODE}:
            raise ValueError(f"unsupported value-cliff local reward mode: {reward_mode!r}")
        if not sampled:
            validate_recovery_rubric(branch_spec.get("recovery_rubric"), name="branch_spec.recovery_rubric")
        child_label = copy.deepcopy(parent.label)

        local_group: list[Sample] = []
        for _ in range(args.n_samples_per_prompt):
            child = Sample()
            child.prompt = copy.deepcopy(prefix)
            child.label = copy.deepcopy(child_label)
            child.generate_function_path = parent.generate_function_path
            child.metadata = _local_child_metadata(parent, prefix, branch_spec=branch_spec)
            local_group.append(child)
        _assign_new_group_identity(data_buffer, local_group)
        local_groups.append(local_group)
        if sampled:
            emitted.add(source_id)
            data_buffer._ab_emitted_terminal_parents = emitted
    return local_groups


def _group_is_local(group: list[Sample]) -> bool:
    return bool(group) and any(_sample_is_local(sample) for sample in group)


def _sample_is_local(sample: Sample) -> bool:
    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    return bool(metadata.get("ab_local_rollout")) or metadata.get("ab_rollout_kind") == "local"


def _event_local_group_mode(group: list[Sample]) -> str | None:
    modes = {
        str((sample.metadata if isinstance(sample.metadata, dict) else {}).get("ab_local_reward_mode") or "").strip()
        for sample in group
    }
    modes.discard("")
    if len(modes) > 1:
        raise ValueError(f"group mixes value-cliff local reward modes: {sorted(modes)!r}")
    return next(iter(modes), None)


def _outcome_judge_failed(sample: Sample) -> bool:
    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    reward = sample.reward if isinstance(sample.reward, dict) else {}
    return bool(metadata.get("outcome_judge_failed")) or reward.get("judge_error") is True


def _sample_training_excluded(sample: Sample) -> bool:
    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    return bool(sample.remove_sample) or bool(metadata.get("agent_excluded_from_training"))


def _event_local_states(group: list[Sample]) -> list[str] | None:
    if _event_local_group_mode(group) is None:
        return None
    states: list[str] = []
    for sample in group:
        reward = sample.reward
        if not isinstance(reward, dict):
            return None
        state = reward.get("event_rubric_state")
        if state is None:
            return None
        states.append(str(state).strip().lower())
    return states


def _build_rollout_metrics(
    data: list[list[Sample]],
    *,
    processed_full_groups: int,
    event_stats: dict[str, int] | None = None,
) -> dict[str, float]:
    return _build_event_rollout_metrics(
        data,
        processed_full_groups=processed_full_groups,
        event_stats=event_stats,
    )


def _accumulate_event_locator_stats(
    stats: Counter[str],
    group: list[Sample],
) -> None:
    if not group:
        raise ValueError("cannot accumulate Value-Cliff locator stats for an empty group")
    metadata = group[0].metadata if isinstance(group[0].metadata, dict) else {}
    info = metadata.get("ab_event_selection_info")
    if not isinstance(info, dict):
        stats["selection_info_missing_groups"] += 1
        return
    field_aliases = {
        "locator_attempts": ("ab_locator_attempted_count",),
        "locator_events": ("ab_locator_event_count",),
        "locator_failures": ("ab_locator_error_count",),
        "locator_init_failures": ("ab_locator_init_error_count",),
        "event_branches": ("ab_event_branch_count",),
    }
    for target, aliases in field_aliases.items():
        for key in aliases:
            if key in info:
                stats[target] += _safe_nonnegative_int(info.get(key))
                break


def _accumulate_full_local_rollout_stats(
    stats: Counter[str],
    group: list[Sample],
    *,
    is_local: bool,
) -> None:
    kind = "local" if is_local else "full"
    for sample in group:
        metadata = sample.metadata or {}
        if is_local and "swe_replay_n_calls" in metadata:
            status = metadata.get("exit_status")
            if not isinstance(status, str) or not status.strip():
                raise ValueError(f"SWE local sample index={sample.index} missing exit_status: {status!r}")
            stats["swe_local_sample_count"] += 1
            stats["swe_local_natural_finish_count"] += int(status == "Submitted")
        turns = _sample_agent_turns(sample)
        if turns is not None:
            stats[f"{kind}_turns_count"] += 1
            stats[f"{kind}_turns_sum"] += turns
            stats[f"{kind}_turns_max"] = max(stats[f"{kind}_turns_max"], turns)

        if not _outcome_judge_failed(sample):
            reward = _sample_reward_score(sample)
            if reward is not None:
                stats[f"{kind}_reward_count"] += 1
                stats[f"{kind}_reward_sum"] += reward

        if not is_local:
            continue
        start_turn = _sample_local_start_turn(sample)
        if start_turn is None:
            continue
        stats["local_start_turn_count"] += 1
        stats["local_start_turn_sum"] += start_turn
        stats["local_start_turn_max"] = max(stats["local_start_turn_max"], start_turn)
        if "local_start_turn_min" not in stats:
            stats["local_start_turn_min"] = start_turn
        else:
            stats["local_start_turn_min"] = min(stats["local_start_turn_min"], start_turn)


def _accumulate_event_local_stats(
    stats: Counter[str],
    group: list[Sample],
    mode: str,
    states: list[str] | None,
) -> None:
    prefix = f"local_{mode}"
    stats[f"{prefix}_groups_seen"] += 1
    if any(_outcome_judge_failed(sample) for sample in group):
        stats[f"{prefix}_judge_failure_groups"] += 1
        return
    outcomes = [_sample_reward_acc(sample) for sample in group if not _sample_training_excluded(sample)]
    valid_outcomes = [outcome for outcome in outcomes if outcome is not None]
    stats[f"{prefix}_valid_samples"] += len(valid_outcomes)
    stats[f"{prefix}_correct_samples"] += sum(valid_outcomes)
    scores = [_sample_reward_score(sample) for sample in group if not _sample_training_excluded(sample)]
    valid_scores = [score for score in scores if score is not None]
    if len(set(valid_scores)) >= 2:
        stats[f"{prefix}_informative_groups"] += 1
    if states is not None:
        for state in states:
            stats[f"{prefix}_state_{state}"] += 1
    if _event_local_group_mode(group) == V7_PRM_REWARD_MODE:
        score_names = {0.0: "zero", 0.25: "quarter", 0.5: "half", 1.0: "one"}
        for state in states or []:
            stats[f"value_cliff_outcome4_state_{state}"] += 1
        for score in valid_scores:
            if score not in score_names:
                raise RuntimeError(f"outcome4 local reward must be one of {sorted(score_names)}, got {score}")
            stats["value_cliff_outcome4_reward_samples"] += 1
            stats[f"value_cliff_outcome4_reward_{score_names[score]}"] += 1


def _build_event_rollout_metrics(
    data: list[list[Sample]],
    *,
    processed_full_groups: int,
    event_stats: dict[str, int] | None,
) -> dict[str, float]:
    stats = event_stats or {}
    local_count = stats.get("swe_local_sample_count", 0)
    natural_count = stats.get("swe_local_natural_finish_count", 0)
    if type(local_count) is not int or type(natural_count) is not int or not 0 <= natural_count <= local_count:
        raise ValueError(f"invalid SWE local finish counts: submitted={natural_count}, total={local_count}")
    attempts = stats.get("locator_attempts", 0)
    technical_failures = stats.get("locator_failures", 0)
    accepted_events = stats.get("event_branches", 0)
    metrics: dict[str, float] = {
        "rollout/ab_event/locator_attempt_count": float(attempts),
        "rollout/ab_event/locator_event_count": float(stats.get("locator_events", 0)),
        "rollout/ab_event/locator_failure_count": float(stats.get("locator_failures", 0)),
        "rollout/ab_event/locator_init_failure_count": float(stats.get("locator_init_failures", 0)),
        "rollout/ab_event/locator_event_rate": (stats.get("locator_events", 0) / attempts if attempts else 0.0),
        "rollout/ab_event/locator_failure_rate": technical_failures / attempts if attempts else 0.0,
        "rollout/ab_event/event_branch_rate": accepted_events / attempts if attempts else 0.0,
        "rollout/ab_event/event_branch_yield_per_full": (
            stats.get("event_branches", 0) / processed_full_groups if processed_full_groups else 0.0
        ),
        "rollout/ab_event/selection_info_missing_group_count": float(stats.get("selection_info_missing_groups", 0)),
    }
    if local_count:
        metrics["rollout/swe/local_sample_count"] = float(local_count)
        metrics["rollout/swe/local_natural_finish_count"] = float(natural_count)
        metrics["rollout/swe/local_natural_finish_rate"] = natural_count / local_count

    for kind in ("local", "full"):
        turns_count = stats.get(f"{kind}_turns_count", 0)
        reward_count = stats.get(f"{kind}_reward_count", 0)
        metrics[f"rollout/ab_event/{kind}_turns/mean"] = (
            stats.get(f"{kind}_turns_sum", 0) / turns_count if turns_count else 0.0
        )
        metrics[f"rollout/ab_event/{kind}_turns/max"] = float(stats.get(f"{kind}_turns_max", 0))
        metrics[f"rollout/ab_event/{kind}_reward/mean"] = (
            stats.get(f"{kind}_reward_sum", 0) / reward_count if reward_count else 0.0
        )

    start_turn_count = stats.get("local_start_turn_count", 0)
    metrics["rollout/ab_event/local_start_turn/mean"] = (
        stats.get("local_start_turn_sum", 0) / start_turn_count if start_turn_count else 0.0
    )
    metrics["rollout/ab_event/local_start_turn/min"] = float(stats.get("local_start_turn_min", 0))
    metrics["rollout/ab_event/local_start_turn/max"] = float(stats.get("local_start_turn_max", 0))

    group_counts = Counter(_group_metric_kind(group) for group in data)
    group_total = len(data)
    modes = (V6_PRM_REWARD_MODE, V6P_REWARD_MODE, V7_PRM_REWARD_MODE, TERMINAL_REWARD_MODE)
    for kind in ("full", *modes):
        metrics[f"rollout/ab_event/group_ratio/{kind}"] = group_counts[kind] / group_total if group_total else 0.0
    for mode in modes:
        prefix = f"local_{mode}"
        seen = stats.get(f"{prefix}_groups_seen", 0)
        valid = stats.get(f"{prefix}_valid_samples", 0)
        metrics[f"rollout/ab_event/{mode}/pass_rate"] = (
            stats.get(f"{prefix}_correct_samples", 0) / valid if valid else 0.0
        )
        metrics[f"rollout/ab_event/{mode}/informative_group_rate"] = (
            stats.get(f"{prefix}_informative_groups", 0) / seen if seen else 0.0
        )
        metrics[f"rollout/ab_event/{mode}/judge_failure_group_rate"] = (
            stats.get(f"{prefix}_judge_failure_groups", 0) / seen if seen else 0.0
        )
        state_names = (
            ("avoid_failed", "avoid_only", "redirected")
            if mode == V6_PRM_REWARD_MODE
            else ("avoid_failed", "avoid_only", "redirected", "outcome_wrong", "outcome_correct")
        )
        state_total = sum(stats.get(f"{prefix}_state_{state}", 0) for state in state_names)
        for state in state_names:
            metrics[f"rollout/ab_event/{mode}/state/{state}_rate"] = (
                stats.get(f"{prefix}_state_{state}", 0) / state_total if state_total else 0.0
            )
    outcome4_states = ("avoid_failed", "avoid_only", "redirected", "outcome_wrong", "outcome_correct")
    outcome4_state_total = sum(stats.get(f"value_cliff_outcome4_state_{state}", 0) for state in outcome4_states)
    for state in outcome4_states:
        metrics[f"rollout/ab_event/value_cliff_outcome4/state/{state}_rate"] = (
            stats.get(f"value_cliff_outcome4_state_{state}", 0) / outcome4_state_total if outcome4_state_total else 0.0
        )
    natural_finish_count = sum(
        stats.get(f"value_cliff_outcome4_state_{state}", 0) for state in ("outcome_wrong", "outcome_correct")
    )
    metrics["rollout/ab_event/value_cliff_outcome4/natural_finish_rate"] = (
        natural_finish_count / outcome4_state_total if outcome4_state_total else 0.0
    )
    metrics["rollout/ab_event/value_cliff_outcome4/natural_finish_correct_rate"] = (
        stats.get("value_cliff_outcome4_state_outcome_correct", 0) / natural_finish_count
        if natural_finish_count
        else 0.0
    )
    outcome4_reward_total = stats.get("value_cliff_outcome4_reward_samples", 0)
    for score_name in ("zero", "quarter", "half", "one"):
        metrics[f"rollout/ab_event/value_cliff_outcome4/reward/{score_name}_rate"] = (
            stats.get(f"value_cliff_outcome4_reward_{score_name}", 0) / outcome4_reward_total
            if outcome4_reward_total
            else 0.0
        )

    token_counts: Counter[str] = Counter()
    for group in data:
        kind = _group_metric_kind(group)
        token_counts[kind] += sum(_trainable_tokens(sample) for sample in group)
    total_tokens = sum(token_counts.values())
    for kind in ("full", *modes):
        metrics[f"rollout/ab_event/trainable_token_ratio/{kind}"] = (
            token_counts[kind] / total_tokens if total_tokens else 0.0
        )
    return metrics


def _group_metric_kind(group: list[Sample]) -> str:
    return _event_local_group_mode(group) or "full"


def _trainable_tokens(sample: Sample) -> int:
    if bool(sample.remove_sample):
        return 0
    if sample.loss_mask is not None:
        return sum(int(value) for value in sample.loss_mask)
    try:
        return max(0, int(sample.response_length))
    except (TypeError, ValueError):
        return 0


def _safe_nonnegative_int(value: Any) -> int:
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0


def _sample_reward_score(sample: Sample) -> float | None:
    reward = sample.reward
    if isinstance(reward, dict):
        reward = reward.get("score")
    if reward is None:
        return None
    try:
        score = float(reward)
    except (TypeError, ValueError):
        return None
    return score if math.isfinite(score) else None


def _sample_reward_acc(sample: Sample) -> bool | None:
    reward = sample.reward
    if not isinstance(reward, dict) or not isinstance(reward.get("acc"), bool):
        return None
    return reward["acc"]


def _sample_agent_turns(sample: Sample) -> int | None:
    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    turns = metadata.get("agent_turns")
    if isinstance(turns, bool) or not isinstance(turns, int) or turns < 0:
        return None
    return turns


def _sample_local_start_turn(sample: Sample) -> int | None:
    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    prefix_turns = metadata.get("ab_local_prefix_assistant_turns")
    if isinstance(prefix_turns, bool) or not isinstance(prefix_turns, int) or prefix_turns < 0:
        return None
    return prefix_turns + 1


def _local_child_metadata(
    parent: Sample,
    prefix: list[dict[str, Any]] | None = None,
    *,
    branch_spec: dict[str, Any] | None = None,
) -> dict[str, Any]:
    parent_meta = parent.metadata if isinstance(parent.metadata, dict) else {}
    if not isinstance(branch_spec, dict) or branch_spec.get("branch_type") not in {"value_cliff_local", BRANCH_TYPE}:
        raise ValueError("local child requires a value_cliff_local branch specification")
    if prefix is None:
        candidate = branch_spec.get("prefix_messages")
        prefix = candidate if isinstance(candidate, list) else []
    reward_mode = str(branch_spec.get("reward_mode") or "").strip().lower()
    if reward_mode not in {V6_PRM_REWARD_MODE, V6P_REWARD_MODE, V7_PRM_REWARD_MODE, TERMINAL_REWARD_MODE}:
        raise ValueError(f"unsupported value-cliff local reward mode: {reward_mode!r}")
    terminal = reward_mode == TERMINAL_REWARD_MODE
    prefix_assistant_turns = sum(
        1 for message in prefix if isinstance(message, dict) and message.get("role") == "assistant"
    )
    metadata = {key: copy.deepcopy(value) for key, value in parent_meta.items() if _carry_local_metadata_key(key)}
    metadata.update(
        {
            "ab_local_rollout": True,
            "ab_rollout_kind": "local",
            "ab_original_question": parent_meta.get("ab_original_question") or str(parent.prompt),
            "ab_parent_group_index": parent.group_index,
            "ab_parent_sample_index": parent.index,
            "ab_local_prefix_message_count": len(prefix),
            "ab_local_prefix_assistant_turns": prefix_assistant_turns,
            "ab_local_reward_mode": reward_mode,
            "ab_event": copy.deepcopy(branch_spec.get("event")),
            "ab_event_turn": branch_spec.get("event_turn"),
            "ab_event_locator_version": branch_spec.get("locator_version"),
            "agent_force_final_on_max_turns": terminal,
            "agent_force_final_on_context_reserve": terminal,
        }
    )
    if branch_spec["branch_type"] == BRANCH_TYPE:
        if not terminal or not isinstance(branch_spec.get("selection"), dict) or not branch_spec.get("branch_id"):
            raise ValueError("sampled branch requires terminal reward, selection and branch_id")
        metadata.update(
            {
                "ab_branch_selection": copy.deepcopy(branch_spec["selection"]),
                "ab_branch_id": branch_spec["branch_id"],
                "agent_max_turns": full_rollout_max_turns(),
            }
        )
        return metadata
    rubric = validate_recovery_rubric(branch_spec.get("recovery_rubric"), name="branch_spec.recovery_rubric")
    reason = str(branch_spec.get("value_drop_reason") or "").strip()
    if not reason:
        raise ValueError("value-cliff local branch requires value_drop_reason")
    metadata.update(
        {
            "ab_event_recovery_rubric": rubric,
            "ab_event_value_drop_reason": reason,
            "agent_max_turns": full_rollout_max_turns() if terminal else local_horizon_max_turns(),
        }
    )
    return metadata


def _carry_local_metadata_key(key: str) -> bool:
    if key in _LOCAL_METADATA_DROP_KEYS:
        return False
    if key.startswith("ab_"):
        return False
    if key.startswith("session_"):
        return False
    if key.startswith("agent_"):
        return key in _LOCAL_AGENT_METADATA_CARRY_KEYS or key.startswith(_LOCAL_AGENT_METADATA_CARRY_PREFIXES)
    return True


def _assign_new_group_identity(data_buffer: DataSource, group: list[Sample]) -> None:
    lock = getattr(data_buffer, "identity_lock", None)
    if lock is not None:
        with lock:
            if _assign_new_group_identity_unlocked(data_buffer, group):
                return
    elif _assign_new_group_identity_unlocked(data_buffer, group):
        return

    # Fallback for custom data sources; still keep the group internally unique.
    base = int(time.time() * 1_000_000)
    for offset, sample in enumerate(group):
        sample.group_index = base
        sample.index = base + offset


def _assign_new_group_identity_unlocked(data_buffer: DataSource, group: list[Sample]) -> bool:
    group_index = getattr(data_buffer, "sample_group_index", None)
    sample_index = getattr(data_buffer, "sample_index", None)
    if isinstance(group_index, int) and isinstance(sample_index, int):
        for offset, sample in enumerate(group):
            sample.group_index = group_index
            sample.index = sample_index + offset
        data_buffer.sample_group_index = group_index + 1
        data_buffer.sample_index = sample_index + len(group)
        return True
    return False


def _dump_group_jsonl(args, rollout_id: int, group: list[Sample]) -> None:
    dump_rollout_messages_jsonl(args, group, rollout_id=rollout_id, evaluation=False)


def _cleanup_training_metadata(group: list[Sample]) -> None:
    strip_rollout_heavy_metadata(group)


def _initialize_debug_jsonl(args, rollout_id: int) -> None:
    path = rollout_messages_jsonl_path(args, rollout_id=rollout_id, evaluation=False)
    if path is None:
        return
    key = (str(path), rollout_id)
    if key in _DEBUG_FILES_INITIALIZED:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    _DEBUG_FILES_INITIALIZED.add(key)
