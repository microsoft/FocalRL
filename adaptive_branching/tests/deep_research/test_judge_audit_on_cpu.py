import asyncio
import json
from types import SimpleNamespace

import pytest
from openai.types.chat import ChatCompletion

from adaptive_branching.src.deep_research.judge_audit import audit_attempt, record_response
from adaptive_branching.src.deep_research.judge_client import JudgeClient


def response(content='{"ok":true}', finish="stop", reasoning="thinking"):
    return ChatCompletion.model_validate(
        {
            "id": "test",
            "object": "chat.completion",
            "created": 1,
            "model": "test",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": finish,
                    "message": {"role": "assistant", "content": content, "reasoning_content": reasoning},
                }
            ],
        }
    )


@pytest.mark.parametrize(
    "directory,tag,attempt,payload",
    [
        ("", "x", 1, {"x": 1}),
        ("x", "", 1, {"x": 1}),
        ("x", "x", 0, {"x": 1}),
        ("x", "x", True, {"x": 1}),
        ("x", "x", 1, {}),
    ],
)
def test_invalid_boundary(directory, tag, attempt, payload):
    with pytest.raises(ValueError):
        with audit_attempt(directory, tag=tag, attempt=attempt, request=payload):
            pass


def test_disabled_and_bad_response(tmp_path):
    with audit_attempt(None, tag="", attempt=0, request={}):
        record_response(object())
    with pytest.raises(TypeError):
        with audit_attempt(str(tmp_path), tag="x", attempt=1, request={"x": 1}):
            record_response(object())
    row = json.loads(next(tmp_path.glob("*.json")).read_text())
    assert "TypeError" in row["error"] and row["response"] is None


@pytest.mark.asyncio
async def test_concurrent_attribution_and_missing_usage(tmp_path):
    async def task(tag):
        with audit_attempt(str(tmp_path), tag=tag, attempt=1, request={"user": tag}):
            await asyncio.sleep(0)
            record_response(response(reasoning=tag))

    await asyncio.gather(task("a"), task("b"))
    rows = [json.loads(p.read_text()) for p in tmp_path.glob("*.json")]
    assert len(rows) == 2
    for row in rows:
        assert row["response"]["choices"][0]["message"]["reasoning_content"] == row["tag"]
        assert row["response"]["usage"] is None


@pytest.mark.asyncio
async def test_preserves_length_json_failures_and_success(tmp_path, monkeypatch):
    monkeypatch.setenv("JUDGE_AUDIT_DIR", str(tmp_path))
    answers = iter([response("partial", "length"), response("broken"), response()])

    async def create(**kwargs):
        return next(answers)

    client = object.__new__(JudgeClient)
    client.model = "test"
    client.enable_thinking = True
    client.reasoning_effort = ""
    client.max_tokens = 32768
    client.max_retries = 3
    client._responses_api = False
    client.sampling_params = dict(
        temperature=0.6, top_p=0.95, presence_penalty=0, top_k=20, min_p=0, repetition_penalty=1.1
    )
    client._client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    assert await client.complete_json("sys", "user", ("ok",), "task1", lambda x: x) == {"ok": True}
    rows = sorted([json.loads(p.read_text()) for p in tmp_path.glob("*.json")], key=lambda r: r["attempt"])
    assert len(rows) == 3
    assert "max_tokens" in rows[0]["error"] and rows[0]["response"]["choices"][0]["finish_reason"] == "length"
    assert "JSON" in rows[1]["error"] and not rows[1]["validated"]
    assert rows[2]["validated"] and rows[2]["error"] is None
    assert "Required correction" in rows[2]["request"]["user"]


def test_storage_failure_is_fatal(tmp_path):
    path = tmp_path / "not_a_dir"
    path.write_text("occupied")
    with pytest.raises(FileExistsError):
        with audit_attempt(str(path), tag="x", attempt=1, request={"user": "x"}):
            pass
