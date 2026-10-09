"""Run the real adapter function with CPU session/agent doubles, without SGLang imports."""

import ast
import asyncio
import json
import logging
import os
import time
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from adaptive_branching.src.deep_research.branch_sampling import BranchDataError
from miles.utils.remote_trial import RemoteTrialUnresolved
from miles.utils.types import Sample


@pytest.mark.parametrize("scenario", ["normal", "missing_entropy", "no_records", "truncated"])
def test_adapter_cost_and_fatal_entropy(monkeypatch, caplog, scenario):
    monkeypatch.setenv("AB_BRANCH_SELECTION", "entropy_delta_max")
    monkeypatch.setenv("AB_LOCAL_REWARD_MODE", "terminal")
    caplog.set_level(logging.INFO)
    path = Path(__file__).resolve().parents[3] / "miles/rollout/generate_hub/agentic_tool_call.py"
    tree = ast.parse(path.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.AsyncFunctionDef) and n.name == "generate")
    node.returns = None
    node.args.args[0].annotation = None
    collected = []

    class Tracer:
        session_id = "attempt-1"
        session_server_instance_id = None
        base_url = "http://unused/session"

        @staticmethod
        async def create(args):
            return Tracer()

        async def collect_records(self):
            collected.append(True)
            records = (
                []
                if scenario == "no_records"
                else [SimpleNamespace(response={"usage": {"prompt_tokens": 100, "completion_tokens": 25}})]
            )
            return records, {}

    async def agent(**kwargs):
        assert kwargs["metadata"]["ab_entropy_vocab_size"] == 100
        if scenario == "missing_entropy":
            raise BranchDataError("missing top logprobs")
        return {"agent_turns": 1}

    namespace = dict(
        asyncio=asyncio,
        time=time,
        os=os,
        json=json,
        logger=logging.getLogger(__name__),
        RemoteTrialUnresolved=RemoteTrialUnresolved,
        OpenAIEndpointTracer=Tracer,
        load_function=lambda _: agent,
        build_chat_request_kwargs=lambda _: {},
        deepcopy=deepcopy,
        GenerateFnOutput=lambda **kwargs: SimpleNamespace(**kwargs),
        Sample=Sample,
        compute_samples_from_openai_records=lambda args, sample, *a, **kw: [deepcopy(sample)],
        truncate_samples_by_total_tokens=lambda samples, *a: [] if scenario == "truncated" else samples,
        merge_samples=lambda samples, _: samples[0],
    )
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    args = SimpleNamespace(
        session_server_ip="unused",
        session_server_port=80,
        custom_agent_function_path="agent",
        max_seq_len=1024,
        generate_multi_samples=False,
    )
    inp = SimpleNamespace(
        args=args,
        sample=Sample(group_index=1, index=8, prompt="q"),
        sampling_params={},
        state=SimpleNamespace(tokenizer=list(range(100))),
    )
    if scenario == "missing_entropy":
        with pytest.raises(BranchDataError, match="missing top"):
            asyncio.run(namespace["generate"](inp))
    else:
        result = asyncio.run(namespace["generate"](inp)).samples
        cost = result.metadata["ab_generation_cost"]
        assert cost["completion_tokens"] == (0 if scenario == "no_records" else 25)
        assert cost["session_id"] == "attempt-1"
        if scenario != "normal":
            assert result.status == Sample.Status.ABORTED
    assert collected == [True]
    assert "BRANCH_GENERATION_COST" in caplog.text
