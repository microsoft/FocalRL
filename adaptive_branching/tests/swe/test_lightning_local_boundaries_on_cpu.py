# Anonymous release: run the real process-group timeout check on Linux.
"""Boundary checks for native replay, including real CPU shell processes."""

import asyncio
import copy
import importlib
import json
import os
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from adaptive_branching.src.swe import agent, lightning_replay as replay, thinking_protocol as protocol
from adaptive_branching.tests.swe.test_lightning_agent_on_cpu import completion
from adaptive_branching.tests.swe.test_lightning_integration_on_cpu import load_bridge
from adaptive_branching.tests.swe.test_lightning_replay_on_cpu import source


def test_fingerprint_distinguishes_hardlinks_from_independent_copies(tmp_path):
    linked, independent = tmp_path / "linked", tmp_path / "independent"
    linked.mkdir()
    independent.mkdir()
    (linked / "a").write_text("same")
    os.link(linked / "a", linked / "b")
    for name in ("a", "b"):
        (independent / name).write_text("same")
    # Identical bytes/modes are insufficient: writing a changes b only in one tree.
    assert replay.fingerprint(linked) != replay.fingerprint(independent)
    second = tmp_path / "second"
    second.mkdir()
    (second / "a").write_text("same")
    os.link(second / "a", second / "b")
    assert replay.fingerprint(linked) == replay.fingerprint(second)


@pytest.mark.parametrize("change", ["root_mode", "earlier_file"])
def test_fingerprint_rejects_changes_after_an_entry_was_read(monkeypatch, tmp_path, change):
    root = tmp_path / "workspace"
    root.mkdir()
    (root / "a").write_text("old")
    (root / "b").write_text("second")
    original = Path.open

    def mutate(path, *args, **kwargs):
        if path == root / "b":
            if change == "root_mode":
                root.chmod(0o700 if root.stat().st_mode & 0o777 != 0o700 else 0o755)
            else:
                with original(root / "a", "w") as stream:
                    stream.write("new")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", mutate)
    with pytest.raises(RuntimeError, match="changed"):
        replay.fingerprint(root)


@pytest.mark.parametrize(
    "field", ["input_tokens", "output_tokens", "max_prompt_tokens", "format_errors", "blocked_actions"]
)
def test_resume_refuses_stale_continuation_counters(tmp_path, field):
    trajectory, _ = source(tmp_path)
    state = replay.build_replay(trajectory, 2)
    episode = agent.Episode(messages=state["messages"], turns=1)
    setattr(episode, field, 1)

    async def unexpected(*args):
        raise AssertionError("invalid recovery must fail before model/shell")

    with pytest.raises(ValueError, match="telemetry"):
        asyncio.run(agent.run_episode("fix", unexpected, unexpected, episode=episode, resume=True, local_horizon=10))


def load_local_bridge(monkeypatch):
    base = load_bridge(monkeypatch)
    name = "adaptive_branching.src.swe.lightning_local_harbor_agent"
    monkeypatch.delitem(sys.modules, name, raising=False)
    module = importlib.import_module(name)
    monkeypatch.setitem(sys.modules, name, module)
    return base, module


@pytest.mark.parametrize("horizon", [10, 20, 30])
def test_latest_possible_prefix_gets_requested_turns_and_preserves_reasoning(horizon):
    prefix = [
        {"role": "system", "content": protocol.SYSTEM_PROMPT},
        {"role": "user", "content": protocol.INSTANCE_PROMPT.format(problem_statement="fix")},
    ]
    for turn in range(99):
        message = completion()["choices"][0]["message"]
        prefix.extend(
            [
                {"role": "assistant", **message, "reasoning_content": f"reasoning-{turn}"},
                {"role": "tool", "tool_call_id": message["tool_calls"][0]["id"], "content": "observed"},
            ]
        )
    original = copy.deepcopy(prefix)
    queries = []

    async def query(messages):
        queries.append(copy.deepcopy(messages))
        return completion()

    async def execute(command, timeout):
        return "", 0

    episode = agent.Episode(messages=prefix, turns=99)
    asyncio.run(agent.run_episode("fix", query, execute, episode=episode, resume=True, local_horizon=horizon))
    assert episode.turns == 99 + horizon and episode.stop_reason == "LocalHorizon"
    assert queries[0] == original and prefix == original and len(queries) == horizon
    assert [entry["turn"] for entry in episode.command_results] == list(range(100, 100 + horizon))


@pytest.mark.skipif(sys.platform != "linux", reason="Harbor process-group timeout contract requires Linux")
def test_real_timeout_kills_descendants_and_preserves_partial_workspace(monkeypatch, tmp_path):
    timeout = shutil.which("timeout")
    assert timeout is not None, "GNU timeout is required for the real CPU timeout boundary test"
    base = load_bridge(monkeypatch)
    results = []
    fingerprints = []

    class Environment:
        network_policy = SimpleNamespace(network_mode="no-network")

        def __init__(self, workspace):
            self.workspace = workspace

        def _egress_controlled_service_names(self):
            return ["main"]

        async def exec(self, command, *, cwd, timeout_sec, env=None):
            if command in (base._HIDE_GIT, base._RESTORE_GIT):
                return SimpleNamespace(return_code=0, stdout="", stderr="")
            assert command.startswith("exec timeout --signal=TERM --kill-after=5s 1s ")
            proc = await asyncio.create_subprocess_exec(
                "bash",
                "-c",
                command,
                cwd=self.workspace,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout_sec)
            return SimpleNamespace(return_code=proc.returncode, stdout=stdout.decode(), stderr=stderr.decode())

    async def loop(instruction, query, execute, environment):
        results.append(await execute("printf partial > a; (sleep 2; printf leaked > late) & wait", 1))
        fingerprints.append(replay.fingerprint(environment.workspace))

    async def check():
        roots = []
        for name in ("source", "replay"):
            root = tmp_path / name
            root.mkdir()
            roots.append(root)
            bridge = base.LightningSweAgent(
                tmp_path / (name + "-logs"), "openai/test", extra_env={"OPENAI_API_BASE": "http://unused/v1"}
            )
            monkeypatch.setattr(bridge, "run_loop", loop)
            await bridge.run("fix", Environment(root), SimpleNamespace())
        await asyncio.sleep(1.2)
        for root in roots:
            assert (root / "a").read_text() == "partial"
            assert not (root / "late").exists(), "timed-out descendant continued changing workspace"

    asyncio.run(check())
    assert results == [("[timed out after 1s]", 124)] * 2
    assert fingerprints[0] == fingerprints[1]


@pytest.mark.parametrize("target", ["issue", "assistant", "tool", "call", "function"])
def test_replay_refuses_opaque_judge_fields_inside_conversation(tmp_path, target):
    trajectory, _ = source(tmp_path)
    state = replay.build_replay(trajectory, 2)
    fields = {
        "issue": state["messages"][1],
        "assistant": state["messages"][2],
        "tool": state["messages"][3],
        "call": state["messages"][2]["tool_calls"][0],
        "function": state["messages"][2]["tool_calls"][0]["function"],
    }
    fields[target]["golden_patch"] = "PRIVATE_ORACLE"
    with pytest.raises(ValueError):
        replay.validate_replay(state)


@pytest.mark.parametrize("value", [False, 0, "", {}])
def test_prefix_rejects_falsy_non_list_tool_calls(value):
    messages = [
        {"role": "system", "content": protocol.SYSTEM_PROMPT},
        {"role": "user", "content": protocol.INSTANCE_PROMPT.format(problem_statement="fix")},
        {"role": "assistant", "content": "no calls", "tool_calls": value},
        {"role": "user", "content": protocol.format_error_message(0, "stop")},
    ]
    with pytest.raises(ValueError, match="tool calls"):
        replay.validate_prefix(messages, 1, problem="fix")


@pytest.mark.parametrize("failure", ["issue", "horizon", "missing", "state_type", "returncode"])
def test_local_bridge_recovery_failure_never_calls_model(monkeypatch, tmp_path, failure):
    _, local = load_local_bridge(monkeypatch)
    trajectory, _ = source(tmp_path)
    state = replay.build_replay(trajectory, 2)
    if failure == "issue":
        state["messages"][1]["content"] = "other issue"
    commands = []
    logs = tmp_path / "logs"
    logs.mkdir()
    if failure != "missing":
        (logs / "replay.json").write_text(json.dumps([] if failure == "state_type" else state))

    class Environment:
        async def exec(self, command, **kwargs):
            raise AssertionError("replay must not capture filesystem fingerprints")

    async def execute(command, timeout):
        commands.append(command)
        return "", 1 if failure == "returncode" else 0

    async def unexpected(*args):
        raise AssertionError("recovery failure must not reach the policy model")

    bridge = local.LocalLightningSweAgent(
        logs,
        "openai/test",
        replay=True,
        max_turns=10 if failure == "horizon" else 11,
        extra_env={"OPENAI_API_BASE": "http://unused/v1"},
    )
    with pytest.raises((ValueError, RuntimeError, FileNotFoundError)):
        asyncio.run(bridge.run_loop("fix", unexpected, execute, Environment()))
    assert commands == (["printf one > a"] if failure == "returncode" else [])


@pytest.mark.parametrize("horizon", [10, 20, 30])
def test_local_bridge_full_source_then_fresh_replay_uses_exact_model_boundary(monkeypatch, tmp_path, horizon):
    _, local = load_local_bridge(monkeypatch)
    events = []

    class Environment:
        def __init__(self):
            self.changed = False

        async def exec(self, command, **kwargs):
            raise AssertionError("Full and Local must not launch filesystem capture")

        async def execute(self, command, timeout):
            events.append(command)
            self.changed = True
            return ("COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT" if command == "submit" else ""), 0

    replies = iter([completion("change"), completion("inspect"), completion("submit")])

    async def query(messages):
        events.append("query")
        body = next(replies)
        body["choices"][0]["message"]["reasoning_content"] = "preserved reasoning"
        return body

    source_env = Environment()
    full = local.LocalLightningSweAgent(
        tmp_path / "full", "openai/test", extra_env={"OPENAI_API_BASE": "http://unused/v1"}
    )
    asyncio.run(full.run_loop("fix", query, source_env.execute, source_env))
    assert events == ["query", "change", "query", "inspect", "query", "submit"]
    trajectory = full.trajectory(full.episode.metrics())
    assert "boundaries" not in trajectory
    state = replay.build_replay(trajectory, 2)
    assert state["schema"] == replay.SCHEMA and "expected" not in state
    logs = tmp_path / "local"
    logs.mkdir()
    (logs / "replay.json").write_text(json.dumps(state))
    fresh = Environment()
    seen = []

    async def continuation(messages):
        seen.append(copy.deepcopy(messages))
        return completion("change")

    child = local.LocalLightningSweAgent(
        logs, "openai/test", replay=True, max_turns=1 + horizon, extra_env={"OPENAI_API_BASE": "http://unused/v1"}
    )
    events.clear()
    asyncio.run(child.run_loop("fix", continuation, fresh.execute, fresh))
    assert "capture" not in events
    assert seen[0] == state["messages"] and len(seen) == horizon
    assert child.prefix_calls == 1 and child.episode.stop_reason == "LocalHorizon"
    assert (
        len(child.episode.command_results) == horizon
    ), "replayed commands must not become continuation training data"
    assert child.episode.messages[2]["reasoning_content"] == "preserved reasoning"


@pytest.mark.parametrize("zero", ["no_response", "all_masked"])
def test_no_trainable_tokens_are_excluded_before_local_prm(monkeypatch, zero):
    from adaptive_branching.src.swe import lightning_local, reward
    from adaptive_branching.src.swe.lightning_generate import constrain_sample
    from adaptive_branching.tests.swe.test_reward_on_cpu import _sample

    def unexpected(*args, **kwargs):
        raise AssertionError("no trainable tokens must not initialize the PRM")

    monkeypatch.setattr(reward, "JudgeClient", unexpected)
    sample = _sample(local=True, swe_local_prm_only=True, exit_status="ContextLimit")
    if zero == "no_response":
        sample.response_length = 0
        sample.loss_mask = []
    else:
        sample.loss_mask = [0, 0]
    constrain_sample(sample, evaluation=False)
    result = asyncio.run(lightning_local.reward_func(None, sample))
    assert result["event_rubric_state"] == "excluded" and sample.remove_sample
    assert sample.metadata["ab_local_trainable"] is False
