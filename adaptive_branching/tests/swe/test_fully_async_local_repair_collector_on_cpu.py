import importlib
import sys
import types

from miles.utils.types import Sample


def _import_adapter(monkeypatch):
    fake_fully_async = types.ModuleType("examples.fully_async.fully_async_rollout")
    fake_fully_async._cached_version = object()
    fake_fully_async.generate_rollout_fully_async = lambda *args, **kwargs: None
    fake_fully_async.get_global_worker = None
    fake_fully_async.group_oldest_weight_version = lambda group: None
    monkeypatch.setitem(sys.modules, "examples.fully_async.fully_async_rollout", fake_fully_async)
    fake_data_source = types.ModuleType("miles.rollout.data_source")
    fake_data_source.DataSource = object
    monkeypatch.setitem(sys.modules, "miles.rollout.data_source", fake_data_source)
    module_names = (
        "adaptive_branching.src.deep_research.fully_async_value_cliff_collector",
        "adaptive_branching.src.swe.fully_async_local_repair_collector",
    )
    for name in module_names:
        sys.modules.pop(name, None)
    adapter = importlib.import_module(module_names[1])
    shared = importlib.import_module(module_names[0])
    return adapter, shared


def test_adapter_reuses_shared_collector_and_preserves_swe_replay_metadata(monkeypatch):
    adapter, shared = _import_adapter(monkeypatch)
    monkeypatch.setenv("AB_EVENT_HIDDEN_MAX_TURNS", "10")
    prefix = [{"role": "user", "content": "issue"}]
    parent = Sample(
        prompt="issue",
        group_index=7,
        index=2,
        metadata={
            "instance_id": "r2e-pandas-deadbeef",
            "swe_replay_path": "/trials/negative/agent/local-replay-turn-1.json",
            "swe_replay_n_calls": 0,
            "ab_swe_golden_patch": "diff --git a/private.py b/private.py",
        },
    )
    branch_spec = {
        "branch_type": "value_cliff_local",
        "prefix_messages": prefix,
        "event": {"selected_turn": 1},
        "event_turn": 1,
        "locator_version": "swe-v3",
        "reward_mode": "v6_prm",
        "recovery_rubric": {"avoid_error": "stop X", "redirect": "test Y"},
        "value_drop_reason": "X caused the largest value drop",
    }

    metadata = shared._local_child_metadata(parent, prefix, branch_spec=branch_spec)

    assert adapter.generate_rollout_fully_async is shared.generate_rollout_fully_async
    assert metadata["instance_id"] == "r2e-pandas-deadbeef"
    assert metadata["swe_replay_path"] == "/trials/negative/agent/local-replay-turn-1.json"
    assert metadata["swe_replay_n_calls"] == 0
    assert metadata["agent_max_turns"] == 10
    assert "ab_swe_golden_patch" not in metadata
