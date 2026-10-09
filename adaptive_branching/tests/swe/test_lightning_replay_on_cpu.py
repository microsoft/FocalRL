import asyncio
import copy
import json
import subprocess

import pytest

from adaptive_branching.src.swe import agent, lightning_replay as replay, thinking_protocol as protocol
from adaptive_branching.tests.swe.test_lightning_agent_on_cpu import completion


def source(tmp_path, commands=None):
    workspace = tmp_path / "source"
    workspace.mkdir()
    commands = commands or ["printf one > a", "printf two >> a", "cat a"]
    episode = agent.Episode()
    boundaries = {}
    replies = iter([completion(c) for c in commands])

    async def query(messages):
        return next(replies)

    async def execute(command, timeout):
        p = subprocess.run(["bash", "-c", command], cwd=workspace, capture_output=True, text=True, timeout=timeout)
        return p.stdout + p.stderr, p.returncode

    async def boundary(ep):
        boundaries[str(ep.turns + 1)] = {"/testbed": replay.fingerprint(workspace), "/tmp": "0" * 64}

    asyncio.run(
        agent.run_episode(
            "fix",
            query,
            execute,
            episode=episode,
            config=agent.AgentConfig(max_turns=len(commands)),
            before_turn=boundary,
        )
    )
    trajectory = {
        "replay_schema": replay.LEGACY_SCHEMA,
        "messages": episode.messages,
        "command_results": episode.command_results,
        "boundaries": boundaries,
    }
    return trajectory, workspace


def restore(state, workspace, *, broken_output=False):
    workspace.mkdir()
    executed = []

    async def execute(command, timeout):
        executed.append(command)
        p = subprocess.run(["bash", "-c", command], cwd=workspace, capture_output=True, text=True, timeout=timeout)
        return p.stdout + p.stderr + ("wrong" if broken_output else ""), p.returncode

    result = asyncio.run(replay.restore_replay(state, execute))
    return result, executed


@pytest.mark.parametrize("turn", [1, 2])
def test_replay_only_prefix_and_exact_conversation(tmp_path, turn):
    trajectory, _ = source(tmp_path)
    state = replay.build_replay(trajectory, turn)
    messages, commands = restore(state, tmp_path / "dest")
    assert messages == state["messages"] and messages is not state["messages"]
    assert len(commands) == turn - 1
    if turn == 1:
        assert not (tmp_path / "dest/a").exists()
    else:
        assert (tmp_path / "dest/a").read_text() == "one"
    assert all("gold" not in key and "rubric" not in key for key in state)


@pytest.mark.parametrize(
    "commands",
    [
        ["mkdir d; printf abc > d/a; chmod 755 d/a", "true", "true"],
        ["printf abc > a; mv a b; ln -s b link", "true", "true"],
        ["printf abc > a; rm a", "true", "true"],
        ["printf error; exit 2", "true", "true"],
        ["git log", "true", "true"],
        ["python -c 'print(\"x\" * 9000)'", "true", "true"],
    ],
)
def test_replay_files_permissions_symlinks_nonzero_blocked_long_output(tmp_path, commands):
    trajectory, workspace = source(tmp_path, commands)
    state = replay.build_replay(trajectory, 2)
    restore(state, tmp_path / "dest")
    assert replay.fingerprint(tmp_path / "dest") == state["expected"]["/testbed"]


@pytest.mark.parametrize("mutation", ["missing_action", "wrong_command", "returncode"])
def test_invalid_ledger_or_returncode_mismatch_rejected(tmp_path, mutation):
    trajectory, _ = source(tmp_path)
    state = replay.build_replay(trajectory, 2)
    if mutation == "missing_action":
        state["actions"] = []
    elif mutation == "wrong_command":
        state["actions"][0]["command"] = "true"
    elif mutation == "returncode":
        state["actions"][0]["returncode"] = 1
    with pytest.raises((ValueError, RuntimeError)):
        restore(state, tmp_path / "dest")


@pytest.mark.parametrize("turn", [1, 2])
def test_output_and_workspace_differences_are_accepted(tmp_path, turn):
    trajectory, _ = source(tmp_path)
    state = replay.build_replay(trajectory, turn)
    state["initial"] = {root: "f" * 64 for root in state["initial"]}
    state["expected"] = {root: "e" * 64 for root in state["expected"]}
    messages, commands = restore(state, tmp_path / "dest", broken_output=True)
    assert messages == state["messages"] and len(commands) == turn - 1


@pytest.mark.parametrize("code", [0, 1, 2, 124])
def test_only_returncode_controls_replay_acceptance(tmp_path, code):
    trajectory, _ = source(tmp_path, [f"exit {code}", "true", "true"])
    state = replay.build_replay(trajectory, 2)

    async def execute(command, timeout):
        assert command == f"exit {code}" and timeout == 120
        return "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\nchanged output", code

    assert asyncio.run(replay.restore_replay(state, execute)) == state["messages"]


@pytest.mark.parametrize("result", [("", True), ("", None), (None, 0)])
def test_invalid_shell_result_rejected(tmp_path, result):
    trajectory, _ = source(tmp_path)
    state = replay.build_replay(trajectory, 2)

    async def execute(command, timeout):
        return result

    with pytest.raises(TypeError, match="invalid replay shell result at turn 1"):
        asyncio.run(replay.restore_replay(state, execute))


def test_infrastructure_exception_propagates_and_invalid_execute_rejected(tmp_path):
    trajectory, _ = source(tmp_path)
    state = replay.build_replay(trajectory, 2)

    async def execute(command, timeout):
        raise ConnectionError("sandbox unavailable")

    with pytest.raises(ConnectionError, match="sandbox unavailable"):
        asyncio.run(replay.restore_replay(state, execute))
    with pytest.raises(TypeError, match="execute callable"):
        asyncio.run(replay.restore_replay(state, None))


@pytest.mark.parametrize("turn", [0, -1, True, 3, 100, "2"])
def test_invalid_branch_turn(tmp_path, turn):
    trajectory, _ = source(tmp_path)
    with pytest.raises(ValueError):
        replay.build_replay(trajectory, turn)


@pytest.mark.parametrize("mutation", ["system", "issue", "missing_tool", "tool_id", "count", "unknown_user"])
def test_invalid_prefix(tmp_path, mutation):
    trajectory, _ = source(tmp_path)
    state = replay.build_replay(trajectory, 2)
    messages, count = state["messages"], state["n_calls"]
    if mutation == "system":
        messages[0]["content"] = "bad"
    elif mutation == "issue":
        messages[1]["content"] = "wrong issue"
    elif mutation == "missing_tool":
        messages.pop()
    elif mutation == "tool_id":
        messages[-1]["tool_call_id"] = "bad"
    elif mutation == "count":
        count = 2
    elif mutation == "unknown_user":
        messages.append({"role": "user", "content": "golden hint"})
    with pytest.raises(ValueError):
        replay.validate_prefix(messages, count, problem="fix")


def test_continue_ten_turns_preserves_prefix_and_horizon(tmp_path):
    trajectory, _ = source(tmp_path)
    state = replay.build_replay(trajectory, 2)
    original = copy.deepcopy(state["messages"])
    seen = []

    async def query(messages):
        seen.append(copy.deepcopy(messages))
        return completion("true")

    async def execute(command, timeout):
        return "", 0

    episode = agent.Episode(messages=state["messages"], turns=state["n_calls"])
    asyncio.run(agent.run_episode("fix", query, execute, episode=episode, resume=True, local_horizon=10))
    assert episode.stop_reason == "LocalHorizon" and episode.turns == 11
    assert not episode.metrics()["agent_max_turns_hit"]
    assert seen[0] == original and state["messages"] == original
    assert len(seen) == 10 and len(episode.command_results) == 10


@pytest.mark.parametrize("terminal", ["context", "length", "submit", "format"])
def test_local_early_termination_does_not_force_extra_model_call(tmp_path, terminal):
    trajectory, _ = source(tmp_path)
    state = replay.build_replay(trajectory, 2)
    calls = []

    async def query(messages):
        calls.append(1)
        if terminal == "context":
            raise agent.ContextOverflow()
        if terminal == "length":
            return completion("should not run", finish="length")
        if terminal == "format":
            return completion(content="no tools")
        return completion("echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT")

    async def execute(command, timeout):
        assert terminal == "submit"
        return "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT", 0

    episode = agent.Episode(messages=state["messages"], turns=state["n_calls"])
    asyncio.run(agent.run_episode("fix", query, execute, episode=episode, resume=True, local_horizon=10))
    assert (
        episode.stop_reason
        == {"context": "ContextLimit", "length": "LengthTruncated", "submit": "Submitted", "format": "FormatLimit"}[
            terminal
        ]
    )
    assert len(calls) == (3 if terminal == "format" else 1)


def test_fingerprint_detects_bytes_mode_link_and_special_files(tmp_path):
    root = tmp_path / "files"
    root.mkdir()
    empty = replay.fingerprint(root)
    file = root / "a"
    file.write_text("1")
    first = replay.fingerprint(root)
    assert first != empty
    file.write_text("2")
    assert replay.fingerprint(root) != first
    first = replay.fingerprint(root)
    file.chmod(0o755)
    assert replay.fingerprint(root) != first
    (root / "link").symlink_to("missing")
    first = replay.fingerprint(root)
    (root / "link").unlink()
    assert replay.fingerprint(root) != first
    import os

    os.mkfifo(root / "fifo")
    with pytest.raises(ValueError, match="special file"):
        replay.fingerprint(root)


def test_artifact_scope_idempotence_and_conflicts(tmp_path):
    trajectory, _ = source(tmp_path)
    trial = tmp_path / "trials/t1"
    (trial / "agent").mkdir(parents=True)
    (trial / "agent/lightning-trajectory.json").write_text(json.dumps(trajectory))
    path, state = replay.create_replay(trial.parent, str(trial), 2)
    assert replay.create_replay(trial.parent, str(trial), 2) == (path, state)
    assert replay.load_trial_messages(trial.parent, str(trial)) == trajectory["messages"]
    path.write_text("changed")
    with pytest.raises(RuntimeError, match="conflicting"):
        replay.create_replay(trial.parent, str(trial), 2)
    with pytest.raises(ValueError, match="direct child"):
        replay.load_trial_messages(tmp_path, str(trial))


def test_multi_call_turn_replays_all_in_order_and_no_selected_turn(tmp_path):
    trajectory, _ = source(tmp_path)
    # Combine first two executed commands into one assistant turn, retaining
    # each tool result and checking the corresponding ordered ledger.
    m = trajectory["messages"]
    m[2]["tool_calls"].extend(m[4]["tool_calls"])
    del m[4]
    m.extend([copy.deepcopy(m[-2]), copy.deepcopy(m[-1])])
    trajectory["command_results"][1]["turn"] = 1
    trajectory["command_results"][2]["turn"] = 2
    trajectory["boundaries"]["2"] = trajectory["boundaries"]["3"]
    state = replay.build_replay(trajectory, 2)
    _, actions = restore(state, tmp_path / "dest")
    assert actions == ["printf one > a", "printf two >> a"]
    assert (tmp_path / "dest/a").read_text() == "onetwo"


def test_format_error_counter_restored(tmp_path):
    episode = agent.Episode(
        messages=[
            {"role": "system", "content": protocol.SYSTEM_PROMPT},
            {"role": "user", "content": protocol.INSTANCE_PROMPT.format(problem_statement="fix")},
        ],
        turns=2,
    )
    for _ in range(2):
        episode.messages.extend(
            [
                {"role": "assistant", "content": "oops", "reasoning_content": "keep thinking"},
                {"role": "user", "content": protocol.format_error_message(0, "stop")},
            ]
        )
    count = []

    async def query(messages):
        count.append(1)
        return completion(content="oops")

    async def execute(*args):
        raise AssertionError("no shell call")

    asyncio.run(agent.run_episode("fix", query, execute, episode=episode, resume=True, local_horizon=10))
    assert len(count) == 1 and episode.stop_reason == "FormatLimit"


@pytest.mark.parametrize("change", ["returncode", "ledger_order", "extra_metadata", "schema", "roots"])
def test_corrupt_replay_fails_closed(tmp_path, change):
    trajectory, _ = source(tmp_path)
    state = replay.build_replay(trajectory, 2)
    if change == "returncode":
        state["actions"][0]["returncode"] = True
    elif change == "ledger_order":
        state["actions"][0]["turn"] = 2
    elif change == "extra_metadata":
        state["golden_patch"] = "private"
    elif change == "schema":
        state["schema"] = "old-mini"
    elif change == "roots":
        del state["initial"]["/tmp"]
    with pytest.raises(ValueError):
        replay.validate_replay(state)


def test_timeout_partial_file_changes_are_replayed_and_checked(tmp_path):
    trajectory, _ = source(tmp_path, ["printf partial > a; exit 124", "true", "true"])
    state = replay.build_replay(trajectory, 2)
    restore(state, tmp_path / "dest")
    assert (tmp_path / "dest/a").read_text() == "partial"
    state["actions"][0]["returncode"] = 0
    with pytest.raises(RuntimeError, match="diverged"):
        restore(state, tmp_path / "dest2")
