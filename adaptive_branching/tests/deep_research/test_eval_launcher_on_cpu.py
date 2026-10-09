"""Shell launcher portability and credential handling without external calls."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
LAUNCHER = ROOT / "adaptive_branching/shells/eval/run_browsecomp_eval.sh"


def environment(tmp_path):
    data = tmp_path / "data.jsonl"
    data.write_text('{"question":"q","answer":"a"}\n')
    return {**os.environ, "MODEL": "test-model", "BASE_URL": "http://localhost:30000/v1", "DATA": str(data),
            "OUT": str(tmp_path / "output"), "LLM_API_KEY": "browser-test-key", "LLM_JUDGE_KEY": "judge-test-key",
            "BROWSER_LLM_URL": "http://localhost:8003/v1", "LLM_JUDGE_URL": "http://localhost:8001/v1",
            "SERPER_API_KEY": "serper-test-key", "AGENT_SEARCH_PROVIDER": "serper"}


def test_shell_help_requires_no_service_credentials(tmp_path):
    env = {**os.environ, "PYTHON_BIN": sys.executable}
    for name in ("MODEL", "DATA", "OUT", "LLM_API_KEY", "LLM_JUDGE_KEY", "SERPER_API_KEY"):
        env.pop(name, None)
    result = subprocess.run(["bash", str(LAUNCHER), "--help"], cwd=tmp_path, env=env, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert "--data" in result.stdout and "--resume" in result.stdout


@pytest.mark.parametrize("missing", ["MODEL", "BASE_URL", "DATA", "OUT", "LLM_API_KEY", "LLM_JUDGE_KEY", "SERPER_API_KEY"])
def test_missing_launch_configuration_fails_without_printing_keys(tmp_path, missing):
    env = environment(tmp_path)
    env.pop(missing)
    result = subprocess.run(["bash", str(LAUNCHER)], cwd=tmp_path, env=env, capture_output=True, text=True, timeout=10)
    assert result.returncode != 0 and missing in result.stderr
    assert "browser-test-key" not in result.stdout + result.stderr
    assert "judge-test-key" not in result.stdout + result.stderr
    assert "serper-test-key" not in result.stdout + result.stderr


def test_launcher_preserves_arguments_and_uses_public_config(tmp_path):
    env = environment(tmp_path)
    stub = tmp_path / "python-stub"
    stub.write_text(f"#!{sys.executable}\nimport json, os, sys\nprint(json.dumps({{'argv':sys.argv[1:], 'tools':os.environ['AGENT_TOOLS_CONFIG']}}))\n")
    stub.chmod(0o755)
    env["PYTHON_BIN"] = str(stub)
    result = subprocess.run(["bash", str(LAUNCHER), "--limit", "1", "--resume"], cwd=tmp_path, env=env, capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    data = json.loads(result.stdout)
    assert data["argv"][-3:] == ["--limit", "1", "--resume"]
    assert data["tools"] == str(ROOT / "adaptive_branching/config/eval_tools.yaml")
    assert data["argv"][data["argv"].index("--max-turns") + 1] == "200"
    assert all(key not in result.stdout for key in ("browser-test-key", "judge-test-key", "serper-test-key"))
