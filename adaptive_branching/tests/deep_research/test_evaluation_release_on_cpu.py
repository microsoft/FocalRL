"""Public evaluation inputs, failure policies, and resumable agent execution."""

import asyncio
import copy
import json
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import httpx
import pytest

from adaptive_branching.tests.deep_research.test_browsecomp_eval_on_cpu import (
    EVAL,
    _fixed_sampling_args,
    _request_args,
)


@pytest.mark.parametrize("row,answer", [
    ({"task_id": "gaia", "Question": "question", "Final answer": "answer", "file_name": ""}, "answer"),
    ({"id": 0, "question": "question", "answer": 0}, "0"),
    ({"problem": "question", "ground_truth": "answer"}, "answer"),
])
def test_benchmark_columns_and_zero_reference(tmp_path, row, answer):
    path = tmp_path / "data.jsonl"
    path.write_text(json.dumps(row) + "\n")
    samples = EVAL._load_samples(path, limit=None, offset=0)
    assert len(samples) == 1 and samples[0]["ground_truth"] == answer


@pytest.mark.parametrize("row", [
    {"question": " ", "answer": "a"}, {"question": [], "answer": "a"},
    {"question": "q", "answer": " "}, {"question": "q", "answer": True},
    {"question": "q", "answer": float("nan")},
    {"question": "q", "answer": "a", "task_id": ""},
    {"question": "q", "answer": "a", "file_name": "attachment.pdf"},
])
def test_invalid_labeled_rows_fail_with_context(tmp_path, row):
    path = tmp_path / "data.jsonl"
    path.write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="data.jsonl:1"):
        EVAL._load_samples(path, limit=None, offset=0)


@pytest.mark.parametrize("text,kwargs", [
    ("", {}), ("not-json\n", {}), ("[]\n", {}),
    ('{"question":"q","answer":"a"}\n', {"offset": 1}),
    ('{"question":"q","answer":"a"}\n', {"limit": 0}),
])
def test_empty_malformed_or_empty_selection_fails(tmp_path, text, kwargs):
    path = tmp_path / "data.jsonl"
    path.write_text(text)
    with pytest.raises(ValueError):
        EVAL._load_samples(path, limit=kwargs.get("limit"), offset=kwargs.get("offset", 0))


@pytest.mark.parametrize("url", ["", "ftp://localhost", "https://user:pass@example.test", "https://x?q=1", "https://x#f"])
def test_invalid_endpoint_fails(url):
    with pytest.raises(ValueError):
        EVAL._base_url(url)


@pytest.mark.parametrize("field,value", [
    ("model", ""), ("max_tokens", 0), ("temperature", float("nan")),
    ("temperature", -1), ("top_p", 0), ("top_p", 1.1),
    ("top_k", -2), ("repetition_penalty", 0),
])
def test_invalid_sampling_bounds_fail(field, value):
    args = _request_args()
    setattr(args, field, value)
    with pytest.raises(ValueError):
        EVAL._request_kwargs(args)


def test_endpoint_and_history_boundaries():
    assert EVAL._base_url("http://localhost:30000/v1/") == "http://localhost:30000"
    assert EVAL._history_modes("both") == ["react", "keep5"]
    with pytest.raises(ValueError):
        EVAL._history_modes("unknown")


def install_agent(monkeypatch, run):
    module = ModuleType("focalrl_test_eval_agent")
    module.run = run
    monkeypatch.setitem(sys.modules, module.__name__, module)
    return module.__name__


def test_evaluation_resume_reuses_results_and_rejects_changed_settings(tmp_path, monkeypatch):
    calls = []

    async def run(**kwargs):
        calls.append(kwargs)
        assert "ground_truth" not in kwargs["metadata"]
        return {"messages": [{"role": "assistant", "content": kwargs["prompt"]}],
                "agent_final_answer": kwargs["prompt"], "agent_finished": True}

    async def score(question, ground_truth, answer):
        correct = ground_truth == answer
        return {"score": float(correct), "acc": correct, "judge_raw": "yes" if correct else "no"}

    args = _fixed_sampling_args(tmp_path, skip_judge=False, samples_per_task=1)
    args.agent_module = install_agent(monkeypatch, run)
    args.retry_incomplete_fixed_samples = False
    args.keep_going = False
    monkeypatch.setattr(EVAL, "_score", score)
    samples = [{"task_id": "one", "question": "a", "ground_truth": "a"},
               {"task_id": "two", "question": "b", "ground_truth": "different"}]
    first = asyncio.run(EVAL._run_mode(args, samples, "keep5"))
    assert first["n"] == 2 and first["accuracy"] == 0.5 and first["valid_record_target_met"]
    args.resume = True
    assert asyncio.run(EVAL._run_mode(args, samples, "keep5"))["accuracy"] == 0.5
    assert len(calls) == 2
    args.max_turns += 1
    with pytest.raises(ValueError, match="configuration differs"):
        asyncio.run(EVAL._run_mode(args, samples, "keep5"))
    assert len(calls) == 2


@pytest.mark.parametrize("failure", ["generation", "incomplete", "judge"])
def test_runner_failures_are_saved_and_raise(tmp_path, monkeypatch, failure):
    async def run(**_kwargs):
        if failure == "generation":
            raise RuntimeError("generation failed")
        return {"messages": [], "agent_final_answer": "a", "agent_finished": failure != "incomplete"}

    async def score(*_args):
        return {"judge_error": True}

    args = _fixed_sampling_args(tmp_path, skip_judge=False, samples_per_task=1)
    args.agent_module = install_agent(monkeypatch, run)
    args.retry_incomplete_fixed_samples = False
    args.keep_going = False
    monkeypatch.setattr(EVAL, "_score", score)
    with pytest.raises(RuntimeError):
        asyncio.run(EVAL._run_mode(args, [{"task_id": "task", "question": "q", "ground_truth": "a"}], "keep5"))
    record = json.loads((tmp_path / "test-model.keep5.jsonl").read_text())
    assert record.get("gen_error") or record.get("judge_error")


def test_existing_output_is_not_overwritten(tmp_path, monkeypatch):
    async def run(**_kwargs):
        raise AssertionError("must not generate")

    args = _fixed_sampling_args(tmp_path, skip_judge=True, samples_per_task=1)
    args.agent_module = install_agent(monkeypatch, run)
    path = tmp_path / "test-model.keep5.jsonl"
    path.write_text("saved output\n")
    with pytest.raises(FileExistsError, match="use --resume"):
        asyncio.run(EVAL._run_mode(args, [{"task_id": "task", "question": "q", "ground_truth": "a"}], "keep5"))
    assert path.read_text() == "saved output\n"


@pytest.mark.asyncio
async def test_shared_agent_search_fetch_and_answer_flow(tmp_path, monkeypatch):
    from adaptive_branching.src.deep_research import agent
    from adaptive_branching.tools.deep_research.microsoft_browse_tool import (
        ToolExecution,
    )

    requests = []

    async def post(_client, _url, payload, **_kwargs):
        requests.append(copy.deepcopy(payload))
        index = len(requests)
        if index <= 2:
            name = "search" if index == 1 else "fetch_url"
            arguments = {"query": ["city clue"]} if index == 1 else {"url": ["https://example.test"], "purpose": "city"}
            message = {"role": "assistant", "content": "", "tool_calls": [
                {"id": f"call-{index}", "type": "function", "function": {"name": name, "arguments": json.dumps(arguments)}}
            ]}
        else:
            message = {"role": "assistant", "content": "Reference city"}
        return httpx.Response(200, json={"choices": [{"message": message, "finish_reason": "stop"}],
                                        "usage": {"total_tokens": index * 10}})

    async def tool(_parameters):
        return ToolExecution("Reference city", {"success": True})

    async def score(_question, truth, answer):
        return {"score": float(truth == answer), "acc": truth == answer}

    monkeypatch.setenv("AGENT_CONTEXT_RESERVE_TOKENS", "16")
    monkeypatch.setenv("AB_BRANCH_SELECTION", "value_cliff")
    monkeypatch.setattr(agent, "_post_chat_completion", post)
    monkeypatch.setattr(agent.SEARCH_TOOL, "execute", tool)
    monkeypatch.setattr(agent.BROWSE_TOOL, "execute", tool)
    monkeypatch.setattr(EVAL, "_score", score)
    args = _fixed_sampling_args(tmp_path, skip_judge=False, samples_per_task=1)
    args.max_turns = 4
    args.max_seq_len = 4096
    args.retry_incomplete_fixed_samples = False
    result = await EVAL._run_mode(args, [{"task_id": "task", "question": "Find the city", "ground_truth": "Reference city"}], "keep5")
    assert result["accuracy"] == 1 and len(requests) == 3
    assert all("ground_truth" not in request for request in requests)
    record = json.loads((tmp_path / "test-model.keep5.jsonl").read_text())
    assert record["metrics"]["agent_search_count"] == record["metrics"]["agent_fetch_url_count"] == 1
    assert sum(message["role"] == "tool" for message in record["messages"]) == 2


def browse_config():
    from adaptive_branching.src.deep_research import agent
    config = copy.deepcopy(agent.TOOL_CONFIG["fetch_url"])
    config.update(ms_api_keys="", ms_fallback_enabled=False)
    return config


@pytest.mark.parametrize("raw,sentinel", [("page", None), ("", "not found"), ("", None)])
@pytest.mark.asyncio
async def test_jina_only_fetch_never_calls_microsoft(tmp_path, monkeypatch, raw, sentinel):
    from adaptive_branching.tools.deep_research.microsoft_browse_tool import (
        MicrosoftBrowseTool,
    )
    tool = MicrosoftBrowseTool(config=browse_config(), cache_path=str(tmp_path / "cache"), raw_cache_path=str(tmp_path / "raw"))

    async def fetch(_url):
        return raw, sentinel

    async def forbidden(_url):
        raise AssertionError("Microsoft fallback is disabled")

    monkeypatch.setattr(tool, "_jina_fetch", fetch)
    monkeypatch.setattr(tool, "_ms_browse", forbidden)
    result = await tool._fetch_raw_uncached("https://example.test")
    assert result == ((raw, None) if raw else ("", sentinel or "(Jina could not fetch URL: https://example.test)"))


@pytest.mark.parametrize("change,error", [
    ({"ms_fallback_enabled": "false"}, TypeError),
    ({"ms_fallback_enabled": False, "jina_primary": False}, ValueError),
    ({"ms_fallback_enabled": True}, RuntimeError),
])
def test_browse_backend_configuration_fails_fast(tmp_path, change, error):
    from adaptive_branching.tools.deep_research.microsoft_browse_tool import (
        MicrosoftBrowseTool,
    )
    with pytest.raises(error):
        MicrosoftBrowseTool(config={**browse_config(), **change}, cache_path=str(tmp_path / "cache"), raw_cache_path=str(tmp_path / "raw"))


def test_evaluation_config_imports_without_microsoft_keys(tmp_path):
    root = Path(__file__).resolve().parents[3]
    env = {**os.environ, "PYTHONPATH": str(root), "AGENT_TOOLS_CONFIG": str(root / "adaptive_branching/config/eval_tools.yaml"),
           "AGENT_SEARCH_PROVIDER": "serper", "SERPER_API_KEY": "test-only", "BROWSER_LLM_URL": "http://localhost:8003/v1"}
    env.pop("MS_API_KEYS", None)
    result = subprocess.run([sys.executable, "-c", "from adaptive_branching.src.deep_research import agent; assert not agent.BROWSE_TOOL.ms_fallback_enabled; assert not agent.BROWSE_TOOL.api_keys"],
                            cwd=tmp_path, env=env, text=True, capture_output=True, timeout=30)
    assert result.returncode == 0, result.stderr
