import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    assert name in {"launch", "audit_release"}
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


launch = load("launch")
scanner = load("audit_release")


@pytest.fixture
def args(tmp_path):
    hf, ref = tmp_path / "hf", tmp_path / "ref"
    hf.mkdir()
    ref.mkdir()
    (hf / "config.json").write_text("{}")
    data = tmp_path / "data.jsonl"
    data.write_text(json.dumps({"question": "Find the answer", "answer": "Example"}) + "\n")
    return SimpleNamespace(domain="deep_research", actor_gpus=4, rollout_gpus=12, rollout_tp=4,
                           context_parallel=4, gpus_per_node=8, hf_checkpoint=hf, ref_load=ref,
                           prompt_data=data, save=tmp_path / "output", model_size="4B")


@pytest.mark.parametrize("domain,local,horizon,steps", [("deep_research", 48, 5, 80), ("swe", 32, 20, 40)])
def test_paper_recipes(args, domain, local, horizon, steps):
    args.domain = domain
    if domain == "swe":
        args.prompt_data.write_text(json.dumps({"prompt": "Fix bug", "metadata": {
            "instance_id": "example", "ab_swe_golden_patch": "synthetic patch"}}) + "\n")
    command, env = launch.build(args)
    assert command[command.index("--global-batch-size") + 1] == str((16 + local) * 8)
    assert command[command.index("--num-rollout") + 1] == str(steps)
    assert env["AB_EVENT_HIDDEN_MAX_TURNS"] == str(horizon)
    assert env["AB_LOCAL_REWARD_MODE"] == "v6_prm"
    assert "--disable-grpo-std-normalization" in command and "--calculate-per-token-loss" in command
    assert "LLM_JUDGE_KEY" not in env
    assert env["SWE_PRM_BEHAVIOR_VETO"] == "0"
    assert ("--label-key" in command) == (domain == "deep_research")


@pytest.mark.parametrize("name,value", [("actor_gpus", 0), ("rollout_gpus", 1), ("rollout_tp", True),
                                        ("context_parallel", 3), ("gpus_per_node", -1), ("actor_gpus", 16)])
def test_invalid_topology(args, name, value):
    setattr(args, name, value)
    with pytest.raises(ValueError):
        launch.build(args)


@pytest.mark.parametrize("row", [None, {}, {"question": "x"}, {"question": "", "answer": "a"},
                                 {"question": "x", "answer": 1}])
def test_malformed_data(args, row):
    args.prompt_data.write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError):
        launch.build(args)


def test_empty_and_missing_paths(args):
    args.prompt_data.write_text("")
    with pytest.raises(ValueError, match="nonempty"):
        launch.build(args)
    args.prompt_data.write_text('{"question":"x","answer":"y"}\n')
    args.save.mkdir()
    with pytest.raises(ValueError, match="new output"):
        launch.build(args)
    args.ref_load = None
    with pytest.raises(ValueError, match="ref_load"):
        launch.build(args)


def test_missing_hf_config(args):
    (args.hf_checkpoint / "config.json").unlink()
    with pytest.raises(FileNotFoundError):
        launch.build(args)


@pytest.mark.parametrize("size", ["4B", "9B"])
def test_model_spec(size):
    command = launch.model_args(size)
    assert command[command.index("--spec") + 1] == "miles_plugins.models.qwen3_5"
    assert "--num-layers" in command


def test_unknown_recipe():
    with pytest.raises(ValueError):
        launch.recipe("")
    with pytest.raises(ValueError):
        launch.model_args("other")


def test_required_environment(monkeypatch):
    monkeypatch.delenv("TEST_SERVICE", raising=False)
    with pytest.raises(ValueError):
        launch.required_env("TEST_SERVICE")
    monkeypatch.setenv("TEST_SERVICE", "http://localhost:8001/v1")
    assert launch.required_env("TEST_SERVICE", url=True).endswith("/v1")
    for value in ["file:///tmp/data", "http://", "http://user:password@localhost/"]:
        monkeypatch.setenv("TEST_SERVICE", value)
        with pytest.raises(ValueError):
            launch.required_env("TEST_SERVICE", url=True)


def test_scanner_synthetic_patterns():
    synthetic = ["sk-" + "A" * 30, "wandb_v1_" + "B" * 30,
                 "ghp_" + "C" * 30, "-----BEGIN " + "PRIVATE KEY-----",
                 "https://example.org/?sig=" + "D" * 20,
                 "/" + "home/" + "person/project", "10." + "20.30.40", "100." + "65.1.2"]
    assert all(scanner.scan_text(value) for value in synthetic)
    assert scanner.scan_text("private nickname", ("nickname",)) == [(1, "private-identifier")]
    assert scanner.scan_text("http://127.0.0.1:8000 ${oc.env:API_KEY}") == []
    assert scanner.scan_text("") == []
    assert scanner.scan_text("adapter", ("ada",)) == []
    assert scanner.scan_text("ADA", ("ada",)) == [(1, "private-identifier")]
    with pytest.raises(ValueError):
        scanner.scan_text("", ("",))


def test_scanner_files_and_symlinks(tmp_path):
    (tmp_path / "ok.py").write_text("print('example')\n")
    assert scanner.audit(tmp_path) == []
    (tmp_path / "binary").write_bytes(b"\xff")
    (tmp_path / "link").symlink_to(tmp_path / "ok.py")
    (tmp_path / ".env").write_text("example")
    result = scanner.audit(tmp_path)
    assert any("unreviewed-binary" in item for item in result)
    assert any("symlink" in item for item in result)
    assert any("private-artifact" in item for item in result)
    with pytest.raises(NotADirectoryError):
        scanner.audit(tmp_path / "missing")


def test_malformed_json_has_row_context(args):
    args.prompt_data.write_text("{broken\n")
    with pytest.raises(ValueError, match=r":1: invalid JSON"):
        launch.build(args)


def test_cli_describe(monkeypatch, capsys):
    monkeypatch.setattr(launch.sys, "argv", ["launch.py", "swe", "--describe"])
    launch.main()
    assert json.loads(capsys.readouterr().out)["local_horizon"] == 20


@pytest.mark.parametrize("execute", [False, True])
def test_cli_preview_and_launch_propagate_credentials_only_at_runtime(args, monkeypatch, capsys, execute):
    import sys
    from types import ModuleType

    argv = ["launch.py", args.domain]
    for name in ("hf_checkpoint", "ref_load", "prompt_data", "save"):
        argv += ["--" + name.replace("_", "-"), str(getattr(args, name))]
    if execute:
        argv.append("--launch")
    monkeypatch.setattr(launch.sys, "argv", argv)
    monkeypatch.setattr(launch.os, "environ", dict(launch.os.environ))
    for name in ("LLM_JUDGE_KEY", "LLM_API_KEY", "MS_API_KEYS"):
        monkeypatch.setenv(name, "synthetic-runtime-only")
    for name in ("LLM_JUDGE_URL", "SEARCH_ENDPOINT", "BROWSE_ENDPOINT", "BROWSER_LLM_URL"):
        monkeypatch.setenv(name, "http://localhost:8000/v1")
    calls = []
    fake = ModuleType("ray")
    fake.init = lambda **kwargs: calls.append(kwargs)
    monkeypatch.setitem(sys.modules, "ray", fake)
    monkeypatch.setattr(launch.runpy, "run_path", lambda path, **kwargs: calls.append((path, kwargs)))
    launch.main()
    output = capsys.readouterr().out
    assert "synthetic-runtime-only" not in output
    if execute:
        assert len(calls) == 2
        assert calls[0]["address"] == "auto"
        assert calls[0]["runtime_env"]["env_vars"]["LLM_JUDGE_KEY"] == "synthetic-runtime-only"
        assert calls[1][1] == {"run_name": "__main__"}
        assert "synthetic-runtime-only" not in " ".join(launch.sys.argv)
    else:
        assert calls == []
        assert json.loads(output)["command"][1].endswith("train_async.py")
