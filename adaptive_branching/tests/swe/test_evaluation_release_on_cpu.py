"""Evaluation service launch and default infrastructure failure handling."""

import asyncio
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from adaptive_branching.tools.swe import run_lightning_verified, start_eval_session


def checkpoint(tmp_path):
    directory = tmp_path / "checkpoint"
    directory.mkdir()
    for name in ("config.json", "tokenizer_config.json", "tokenizer.json"):
        (directory / name).write_text("{}")
    return directory


def test_session_configuration_uses_checkpoint_and_fixed_template(tmp_path):
    path = checkpoint(tmp_path)
    args = start_eval_session.build_session_args(path, "127.0.0.1", 30001, 1800)
    assert args.hf_checkpoint == str(path.resolve())
    assert args.tito_allowed_append_roles == ["tool", "user"]
    assert args.apply_chat_template_kwargs == {"clear_thinking": False}
    assert args.tito_model == "qwen35" and args.session_server_port == 30001
    assert len(args.session_server_instance_id) == 32
    assert Path(args.chat_template_path).is_file()
    assert Path(args.chat_template_path).name == "qwen3.5_fixed.jinja"


@pytest.mark.parametrize("host,port,timeout", [("", 1, 1), ("localhost", 0, 1), ("localhost", 65536, 1), ("localhost", 1, 0), ("localhost", 1, float("nan"))])
def test_session_invalid_bounds_fail(tmp_path, host, port, timeout):
    with pytest.raises(ValueError):
        start_eval_session.build_session_args(checkpoint(tmp_path), host, port, timeout)


def test_session_missing_checkpoint_and_tokenizer_fail(tmp_path):
    with pytest.raises(ValueError, match="existing directory"):
        start_eval_session.build_session_args(tmp_path / "absent", "localhost", 1, 1)
    path = checkpoint(tmp_path)
    (path / "tokenizer.json").unlink()
    with pytest.raises(FileNotFoundError):
        start_eval_session.build_session_args(path, "localhost", 1, 1)


@pytest.mark.parametrize("url", ["ftp://localhost", "https://user:pass@example.test", "http://localhost/v1", "http://localhost?q=1"])
def test_session_cli_rejects_invalid_backend_before_loading_model(tmp_path, monkeypatch, url):
    monkeypatch.setattr(sys, "argv", ["start_eval_session", "--checkpoint", str(tmp_path), "--backend-url", url])
    with pytest.raises(ValueError, match="backend-url"):
        start_eval_session.main()


def test_default_swe_failure_preserves_result_and_stops_dispatch(tmp_path):
    prompts = tmp_path / "prompts.jsonl"
    prompts.write_text("".join(json.dumps({"prompt": "fix", "metadata": {"instance_id": f"task-{i}"}}) + "\n" for i in range(500)))
    args = SimpleNamespace(prompts=prompts, output=tmp_path / "output", budget="large", model="test",
                           harbor="http://harbor", session="http://session", concurrency=1, timeout=20)
    attempts = []

    def handler(request):
        if request.url.path == "/health":
            return httpx.Response(200, json={"session_server_instance_id": "server"})
        if request.url.path == "/sessions":
            return httpx.Response(200, json={"session_id": "session"})
        if request.method == "DELETE":
            return httpx.Response(200, json={})
        attempts.append(json.loads(request.content)["instance_id"])
        raise httpx.ReadError("simulated Harbor failure")

    with pytest.raises(RuntimeError, match="task-0"):
        asyncio.run(run_lightning_verified.evaluate(args, transport=httpx.MockTransport(handler)))
    rows = (args.output / "results.jsonl").read_text().splitlines()
    assert len(rows) == 1 and json.loads(rows[0])["error"]
    assert attempts == ["task-0"] and not (args.output / "DONE.json").exists()


@pytest.mark.parametrize("url", ["ftp://localhost", "https://user:pass@example.test", "http://localhost#f"])
def test_swe_invalid_services_fail_before_dispatch(tmp_path, url):
    args = SimpleNamespace(budget="large", concurrency=1, timeout=20, model="test", harbor=url, session="http://session")
    with pytest.raises(ValueError, match="harbor"):
        asyncio.run(run_lightning_verified.evaluate(args))
