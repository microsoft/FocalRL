"""Regression cases from the 2026-09-11 Verified cleanup incident."""

import asyncio
import json
import shlex
import subprocess
from types import SimpleNamespace

import httpx
import pytest

from adaptive_branching.src.swe import lightning_replay as replay
from adaptive_branching.tests.swe.test_lightning_agent_on_cpu import completion
from adaptive_branching.tests.swe.test_lightning_integration_on_cpu import load_bridge
from adaptive_branching.tests.swe.test_lightning_replay_on_cpu import source


@pytest.mark.parametrize("failure", ["stdout_only", "transport", "bool_code", "none_code"])
@pytest.mark.parametrize("primary", ["FormatLimit", "Submitted", "model_error"])
def test_cleanup_error_retains_original_stop_and_actual_cause(monkeypatch, tmp_path, failure, primary):
    bridge = load_bridge(monkeypatch)

    class Environment:
        network_policy = SimpleNamespace(network_mode="no-network")

        def _egress_controlled_service_names(self):
            return ["main"]

        async def exec(self, command, **kwargs):
            if command != bridge._RESTORE_GIT:
                return SimpleNamespace(return_code=0, stdout=None, stderr=None)
            if failure == "transport":
                raise RuntimeError("Docker exec transport failed")
            code = {"stdout_only": 1, "bool_code": False, "none_code": None}[failure]
            return SimpleNamespace(
                return_code=code, stdout="Error response from daemon: container unavailable", stderr=None
            )

    async def loop(*args, episode, **kwargs):
        if primary == "model_error":
            raise httpx.ConnectError("original model transport failure")
        episode.stop_reason = primary

    monkeypatch.setattr(bridge, "run_episode", loop)
    instance = bridge.LightningSweAgent(tmp_path, "openai/test", extra_env={"OPENAI_API_BASE": "http://unused/v1"})
    context = SimpleNamespace()
    with pytest.raises((RuntimeError, TypeError)) as raised:
        asyncio.run(instance.run("fix", Environment(), context))
    assert context.metadata["agent_exit_status"] == "AgentInfrastructureFailure"
    assert context.metadata["agent_exit_status_before_cleanup"] == (
        "AgentInfrastructureFailure" if primary == "model_error" else primary
    )
    if primary == "model_error":
        assert "original model transport failure" in context.metadata["agent_primary_error"]
    if failure == "stdout_only":
        assert "container unavailable" in str(raised.value) and "return_code=1" in str(raised.value)
        assert context.metadata["agent_git_restore_result"]["stderr"] is None
    elif failure == "transport":
        assert "Docker exec transport failed" in context.metadata["agent_cleanup_error"]
    saved = json.loads((tmp_path / "lightning-trajectory.json").read_text())
    assert saved["metrics"] == context.metadata


def test_actual_empty_tool_arguments_end_as_format_limit_if_restore_succeeds(monkeypatch, tmp_path):
    bridge = load_bridge(monkeypatch)
    executions, queries = [], []

    class Environment:
        network_policy = SimpleNamespace(network_mode="no-network")

        def _egress_controlled_service_names(self):
            return ["main"]

        async def exec(self, command, **kwargs):
            executions.append(command)
            return SimpleNamespace(return_code=0, stdout=None, stderr=None)

    async def query(*args):
        queries.append(1)
        body = completion()
        call = body["choices"][0]["message"]["tool_calls"][0]
        call["index"] = 0  # Present in the actual SGLang response.
        call["function"]["arguments"] = "{}"
        return body

    monkeypatch.setattr(bridge, "query_model", query)
    instance = bridge.LightningSweAgent(tmp_path, "openai/test", extra_env={"OPENAI_API_BASE": "http://unused/v1"})
    context = SimpleNamespace()
    asyncio.run(instance.run("fix", Environment(), context))
    assert context.metadata["agent_exit_status"] == "FormatLimit"
    assert executions == [bridge._HIDE_GIT, bridge._RESTORE_GIT] and len(queries) == 1


@pytest.mark.parametrize("index", [0, False, -1, 1, "0", None])
def test_replay_handles_real_sglang_index_without_accepting_corruption(tmp_path, index):
    trajectory, _ = source(tmp_path)
    call = trajectory["messages"][2]["tool_calls"][0]
    call["index"] = index
    if type(index) is int and index == 0:
        state = replay.build_replay(trajectory, 2)
        assert state["messages"][2]["tool_calls"][0] == call
    else:
        with pytest.raises(ValueError, match="tool call index"):
            replay.build_replay(trajectory, 2)


@pytest.mark.parametrize("state", ["missing_backup", "existing_git", "clean"])
def test_restore_shell_reports_failed_invariant_and_never_overwrites(monkeypatch, tmp_path, state):
    bridge = load_bridge(monkeypatch)
    backup = tmp_path / "saved metadata"
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    if state != "missing_backup":
        backup.mkdir()
        (backup / "HEAD").write_text("original baseline")
    if state == "existing_git":
        (workspace / ".git").mkdir()
        (workspace / ".git/HEAD").write_text("unexpected existing state")
    command = bridge._RESTORE_GIT.replace("/opt/agl_tmp", shlex.quote(str(backup)))
    result = subprocess.run(["bash", "-c", command], cwd=workspace, capture_output=True, text=True, timeout=5)
    if state == "clean":
        assert result.returncode == 0 and not backup.exists()
        assert (workspace / ".git/HEAD").read_text() == "original baseline"
    else:
        assert result.returncode != 0
        assert ("missing saved" if state == "missing_backup" else "refusing to overwrite") in result.stderr
        if state == "existing_git":
            assert (workspace / ".git/HEAD").read_text() == "unexpected existing state"
            assert (backup / "HEAD").read_text() == "original baseline"
