import asyncio
import ast
import hashlib
import logging
import os
import threading
import time
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from fastapi import FastAPI, Request

from adaptive_branching.src.swe.harbor_durable_run import HEADER, PROTOCOL, DurableRuns, reconnect_run
from adaptive_branching.tools.swe.prepare_harbor_durable_runs import patch_source
from miles.utils.remote_trial import RemoteTrialUnresolved

PAYLOAD = {"base_url": "http://model/sessions/one/v1", "instance_id": "task"}
KEY = hashlib.sha256(PAYLOAD["base_url"].encode()).hexdigest()


@pytest.mark.parametrize("loss", ["before_accept", "during_execution", "after_commit"])
def test_disconnect_reconnect_runs_once_and_returns_original_verifier(tmp_path, loss):
    async def scenario():
        store = DurableRuns(tmp_path / "runs.sqlite")
        started, release = asyncio.Event(), asyncio.Event()
        executions = 0
        app = FastAPI()

        async def execute():
            nonlocal executions
            executions += 1
            started.set()
            await release.wait()
            return {"reward": 1, "exit_status": "Submitted", "trial_id": "original"}

        @app.get("/health")
        async def health():
            return {PROTOCOL: True}

        @app.post("/run")
        async def run(request: Request):
            return await store.run(request.headers[HEADER], await request.json(), execute)

        attempts = 0
        transport = httpx.ASGITransport(app)

        async def wire(request):
            nonlocal attempts
            if request.url.path != "/run":
                return await transport.handle_async_request(request)
            attempts += 1
            if attempts == 1:
                if loss == "before_accept":
                    release.set()
                    return httpx.Response(502)
                if loss == "during_execution":
                    connection = asyncio.create_task(transport.handle_async_request(request))
                    await started.wait()
                    connection.cancel()
                    with pytest.raises(asyncio.CancelledError):
                        await connection
                    assert executions == 1 and len(store.active) == 1
                    release.set()
                    raise httpx.ReadError("injected Serve link change", request=request)
                release.set()
                await transport.handle_async_request(request)
                raise httpx.ReadError("terminal reply lost", request=request)
            return await transport.handle_async_request(request)

        async with httpx.AsyncClient(transport=httpx.MockTransport(wire)) as client:
            result = await reconnect_run(client, "http://harbor", PAYLOAD, timeout=5, retry_delay=0)
        assert result == {"reward": 1, "exit_status": "Submitted", "trial_id": "original"}
        assert executions == 1 and attempts == 2 and not store.active
        # Completion survives server restart; the same request cannot re-execute.
        restarted = DurableRuns(tmp_path / "runs.sqlite")
        assert await restarted.run(KEY, PAYLOAD, execute) == result
        assert executions == 1

    asyncio.run(scenario())


def test_concurrent_retries_share_one_execution_and_bound_active_memory(tmp_path):
    async def scenario():
        store = DurableRuns(tmp_path / "runs.sqlite", capacity=1)
        gate = asyncio.Event()
        count = 0

        async def execute():
            nonlocal count
            count += 1
            await gate.wait()
            return {"reward": 0}

        calls = [asyncio.create_task(store.run(KEY, PAYLOAD, execute)) for _ in range(16)]
        while count == 0:
            await asyncio.sleep(0.001)
        other = "f" * 64
        with pytest.raises(Exception) as error:
            await store.run(other, PAYLOAD, execute)
        assert error.value.status_code == 503
        gate.set()
        assert await asyncio.gather(*calls) == [{"reward": 0}] * 16
        assert count == 1 and not store.active
        # A completed entry doesn't use active capacity.
        assert await store.run(other, PAYLOAD, execute) == {"reward": 0}

    asyncio.run(scenario())


def test_conflict_unfinished_restart_and_execution_failure_never_reexecute(tmp_path):
    async def scenario():
        store = DurableRuns(tmp_path / "runs.sqlite")

        async def fail():
            raise RuntimeError("remote state uncertain")

        with pytest.raises(RuntimeError, match="uncertain"):
            await store.run(KEY, PAYLOAD, fail)
        restarted = DurableRuns(tmp_path / "runs.sqlite")
        for instance in (store, restarted):
            with pytest.raises(Exception) as error:
                await instance.run(KEY, PAYLOAD, fail)
            assert error.value.status_code == 409
        with pytest.raises(Exception) as error:
            await restarted.run(KEY, {**PAYLOAD, "instance_id": "different"}, fail)
        assert error.value.status_code == 409

    asyncio.run(scenario())


def test_disconnect_during_journal_admission_does_not_strand_task(tmp_path, monkeypatch):
    async def scenario():
        store = DurableRuns(tmp_path / "runs.sqlite")
        entered, release = threading.Event(), threading.Event()
        original = store._write
        executions = 0

        def delayed(*args):
            if len(args) == 2:
                entered.set()
                assert release.wait(5)
            return original(*args)

        monkeypatch.setattr(store, "_write", delayed)

        async def execute():
            nonlocal executions
            executions += 1
            return {"reward": 1}

        first = asyncio.create_task(store.run(KEY, PAYLOAD, execute))
        try:
            assert await asyncio.to_thread(entered.wait, 3)
            first.cancel()
            with pytest.raises(asyncio.CancelledError):
                await first
        finally:
            release.set()
        assert await store.run(KEY, PAYLOAD, execute) == {"reward": 1}
        assert executions == 1
        await asyncio.sleep(0)
        assert not store.active and not store.admissions

    asyncio.run(scenario())


def test_128_simultaneous_disconnects_preserve_128_original_results(tmp_path):
    async def scenario():
        store = DurableRuns(tmp_path / "runs.sqlite", capacity=128)
        entered, release = asyncio.Event(), asyncio.Event()
        executions = 0

        async def execute(index):
            nonlocal executions
            executions += 1
            if executions == 128:
                entered.set()
            await release.wait()
            return {"reward": index % 2, "trial_id": str(index)}

        keys = [hashlib.sha256(str(i).encode()).hexdigest() for i in range(128)]
        first = [asyncio.create_task(store.run(key, PAYLOAD, lambda i=i: execute(i))) for i, key in enumerate(keys)]
        await asyncio.wait_for(entered.wait(), 15)
        for task in first:
            task.cancel()
        errors = await asyncio.gather(*first, return_exceptions=True)
        assert all(isinstance(e, asyncio.CancelledError) for e in errors)
        assert len(store.active) == 128
        retries = [asyncio.create_task(store.run(key, PAYLOAD, lambda i=i: execute(i))) for i, key in enumerate(keys)]
        release.set()
        results = await asyncio.wait_for(asyncio.gather(*retries), 15)
        assert results == [{"reward": i % 2, "trial_id": str(i)} for i in range(128)]
        assert executions == 128 and not store.active

    asyncio.run(scenario())


@pytest.mark.parametrize("mode", ["old_server", "conflict", "deadline", "bad_result"])
def test_unrecoverable_run_raises_session_preservation_error(mode):
    calls = []

    async def wire(request):
        calls.append(request.method)
        if request.url.path == "/health":
            return httpx.Response(200, json={PROTOCOL: mode != "old_server"})
        if mode == "conflict":
            return httpx.Response(409)
        if mode == "deadline":
            return httpx.Response(502)
        return httpx.Response(200, json=[])

    async def scenario():
        async with httpx.AsyncClient(transport=httpx.MockTransport(wire)) as client:
            with pytest.raises(RemoteTrialUnresolved):
                await reconnect_run(client, "http://harbor", PAYLOAD, timeout=.05, retry_delay=.001)
        if mode == "old_server":
            assert calls == ["GET"]

    asyncio.run(scenario())


@pytest.mark.parametrize("key,payload", [("", PAYLOAD), ("../escape", PAYLOAD), (KEY, {}), (KEY, [])])
def test_invalid_request_has_no_execution_or_journal_row(tmp_path, key, payload):
    async def scenario():
        store = DurableRuns(tmp_path / "runs.sqlite")
        with pytest.raises(ValueError):
            await store.run(key, payload, lambda: None)
        assert not store.active

    asyncio.run(scenario())


@pytest.mark.parametrize("capacity", [0, -1, True])
def test_capacity_validation(tmp_path, capacity):
    with pytest.raises(ValueError):
        DurableRuns(tmp_path / "runs.sqlite", capacity=capacity)


@pytest.mark.parametrize("failure", [RemoteTrialUnresolved("unknown"), asyncio.CancelledError(), RuntimeError("ordinary"), None])
def test_real_agent_finally_preserves_only_unresolved_or_cancelled_sessions(failure):
    # Execute the real production function with tiny CPU doubles; importing the
    # SGLang GPU protocol stack is unnecessary for testing finally semantics.
    path = Path("miles/rollout/generate_hub/agentic_tool_call.py")
    node = next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.AsyncFunctionDef) and n.name == "generate")
    node.returns = None
    node.args.args[0].annotation = None
    collected = []

    class Tracer:
        session_id = "one"
        session_server_instance_id = None
        base_url = "http://model/sessions/one"

        @staticmethod
        async def create(args):
            return Tracer()

        async def collect_records(self):
            collected.append(True)
            return [], {}

    async def agent(**kwargs):
        if failure is not None:
            raise failure
        return {"reward": 1}

    namespace = dict(
        asyncio=asyncio, time=time, os=os, logger=logging.getLogger(__name__), RemoteTrialUnresolved=RemoteTrialUnresolved,
        OpenAIEndpointTracer=Tracer, load_function=lambda _: agent, build_chat_request_kwargs=lambda _: {},
        deepcopy=deepcopy, GenerateFnOutput=lambda **kw: kw, Sample=SimpleNamespace(Status=SimpleNamespace(ABORTED="aborted")),
    )
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    args = SimpleNamespace(session_server_ip="model", session_server_port=80, custom_agent_function_path="agent")
    sample = SimpleNamespace(metadata={}, prompt="question")
    inp = SimpleNamespace(args=args, sample=sample, sampling_params={})
    if isinstance(failure, (RemoteTrialUnresolved, asyncio.CancelledError)):
        with pytest.raises(type(failure)):
            asyncio.run(namespace["generate"](inp))
        assert collected == []
    else:
        asyncio.run(namespace["generate"](inp))
        assert collected == [True]


def test_server_patch_fail_fast_and_wraps_original_function():
    source = '''from agent_server.state import _state
async def startup():
    _state.semaphore = asyncio.Semaphore(max_concurrent)
@app.post("/run")
async def run_instance(request: RunRequest, raw_request: Request) -> RunResponse:
    return await execute(request)
@app.get("/health")
async def health():
    return {"status": "ok"}
'''
    patched = patch_source(source)
    assert patched.count('@app.post("/run")') == 1
    assert 'async def _run_instance_once' in patched
    assert 'lambda: _run_instance_once(request, raw_request)' in patched
    for bad in ("", patched, source.replace('async def run_instance(', 'async def other(')):
        with pytest.raises(ValueError):
            patch_source(bad)
