"""Full/Local replay without filesystem telemetry, including legacy artifacts."""

import asyncio
import copy
import json

import pytest

from adaptive_branching.src.swe import lightning_replay as replay
from adaptive_branching.tests.swe.test_lightning_agent_on_cpu import completion
from adaptive_branching.tests.swe.test_lightning_local_boundaries_on_cpu import load_local_bridge
from adaptive_branching.tests.swe.test_lightning_replay_on_cpu import source


@pytest.mark.parametrize("turns", [1, 2, 100])
def test_full_never_captures_even_at_turn_limit(monkeypatch, tmp_path, turns):
    _, local = load_local_bridge(monkeypatch)
    calls, commands = [], []

    class Environment:
        async def exec(self, *args, **kwargs):
            raise AssertionError("unexpected filesystem collection")

    async def query(messages):
        assert isinstance(messages, list) and messages
        calls.append(1)
        return completion("submit" if len(calls) == turns and turns < 100 else "true")

    async def execute(command, timeout):
        assert command in {"true", "submit"} and timeout == 120
        commands.append(command)
        return ("COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT" if command == "submit" else ""), 0

    bridge = local.LocalLightningSweAgent(tmp_path, "openai/test", extra_env={"OPENAI_API_BASE": "http://unused/v1"})
    asyncio.run(bridge.run_loop("fix", query, execute, Environment()))
    trajectory = bridge.trajectory(bridge.episode.metrics())
    assert len(calls) == len(commands) == turns
    assert trajectory["replay_schema"] == replay.SCHEMA
    assert "boundaries" not in trajectory
    assert len(trajectory["command_results"]) == turns
    if turns == 1:
        with pytest.raises(ValueError, match="non-final"):
            replay.build_replay(trajectory, 1)
    else:
        for selected in {1, turns - 1}:
            state = replay.build_replay(trajectory, selected)
            assert set(state) == {"schema", "n_calls", "messages", "actions"}
            assert state["n_calls"] == selected - 1
            assert len(state["actions"]) == selected - 1
            commands.clear()
            assert asyncio.run(replay.restore_replay(state, execute)) == state["messages"]
            assert len(commands) == selected - 1


@pytest.mark.parametrize("schema", [replay.LEGACY_SCHEMA, replay.SCHEMA])
def test_old_and_new_artifact_retries_preserve_exact_shape(tmp_path, schema):
    trajectory, _ = source(tmp_path)
    trajectory["replay_schema"] = schema
    if schema == replay.SCHEMA:
        del trajectory["boundaries"]
    trial = tmp_path / "trials" / "one"
    (trial / "agent").mkdir(parents=True)
    (trial / "agent" / "lightning-trajectory.json").write_text(json.dumps(trajectory))
    expected = replay.build_replay(trajectory, 2)
    # Reproduce the old endpoint's serialization before invoking the new reader.
    path = trial / "agent" / "lightning-replay-turn-2.json"
    original = json.dumps(expected, ensure_ascii=False, sort_keys=True)
    path.write_text(original)
    assert replay.create_replay(trial.parent, trial, 2) == (path, expected)
    assert path.read_text() == original
    assert replay.load_trial_messages(trial.parent, trial) == trajectory["messages"]
    assert ("initial" in expected) == (schema == replay.LEGACY_SCHEMA)


@pytest.mark.parametrize(
    "mutation", ["extra_field", "missing_ledger", "bool_count", "wrong_command", "bool_code", "missing_action"]
)
def test_v3_rejects_invalid_state_before_shell(tmp_path, mutation):
    trajectory, _ = source(tmp_path)
    trajectory["replay_schema"] = replay.SCHEMA
    del trajectory["boundaries"]
    state = replay.build_replay(trajectory, 2)
    if mutation == "extra_field":
        state["expected"] = {}
    elif mutation == "missing_ledger":
        del state["actions"]
    elif mutation == "bool_count":
        state["n_calls"] = True
    elif mutation == "wrong_command":
        state["actions"][0]["command"] = "another command"
    elif mutation == "bool_code":
        state["actions"][0]["returncode"] = False
    else:
        state["actions"] = []

    async def unexpected(*args):
        raise AssertionError("invalid replay reached shell")

    with pytest.raises(ValueError):
        asyncio.run(replay.restore_replay(state, unexpected))


@pytest.mark.parametrize("mutation", ["empty", "legacy_boundaries", "missing_ledger", "schema"])
def test_v3_source_boundaries_are_explicit(tmp_path, mutation):
    trajectory, _ = source(tmp_path)
    trajectory["replay_schema"] = replay.SCHEMA
    if mutation != "legacy_boundaries":
        del trajectory["boundaries"]
    if mutation == "empty":
        trajectory["messages"] = []
    elif mutation == "missing_ledger":
        trajectory["command_results"] = None
    elif mutation == "schema":
        trajectory["replay_schema"] = "unknown"
    with pytest.raises(ValueError):
        replay.build_replay(trajectory, 2)


def test_v3_returncode_mismatch_still_aborts_before_continuation(tmp_path):
    trajectory, _ = source(tmp_path)
    trajectory["replay_schema"] = replay.SCHEMA
    del trajectory["boundaries"]
    state = replay.build_replay(trajectory, 2)
    original = copy.deepcopy(state)

    async def execute(command, timeout):
        assert command == "printf one > a" and timeout == 120
        return "different output is allowed", 1

    with pytest.raises(RuntimeError, match="returncode diverged"):
        asyncio.run(replay.restore_replay(state, execute))
    assert state == original
