# Anonymous release: site-specific launcher test omitted.
import asyncio
import importlib
import json
import shlex
import sys
import types
from types import SimpleNamespace

import httpx
import pytest
import yaml

from adaptive_branching.tools.swe import prepare_lightning_tasks as prep

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib


TOML = """schema_version = "1.3"
[agent]
timeout_sec = 12000
network_mode = "public"
[verifier]
environment_mode = "separate"
network_mode = "public"
[verifier.environment]
network_mode = "public"
[environment]
workdir = "/testbed"
"""
COMPOSE = """services:
  main:
    tmpfs: [/r2e_tests]
    networks: [miles-swe-public]
networks:
  miles-swe-public:
    external: true
"""


def test_prepare_offline_task_copy_preserves_original_and_verifier(tmp_path):
    source = tmp_path / "source"
    task = source / "r2e-test"
    (task / "environment").mkdir(parents=True)
    (task / "tests").mkdir()
    (task / "task.toml").write_text(TOML)
    (task / "instruction.md").write_text("Fix it")
    (task / "environment/docker-compose.yaml").write_text(COMPOSE)
    (task / "tests/test.sh").write_text("grade")
    destination = tmp_path / "new"
    assert prep.prepare(source, destination) == 1
    config = tomllib.loads((destination / "r2e-test/task.toml").read_text())
    assert config["agent"]["network_mode"] == config["environment"]["network_mode"] == "no-network"
    assert config["verifier"] == tomllib.loads(TOML)["verifier"]
    assert (task / "task.toml").read_text() == TOML
    assert (task / "environment/docker-compose.yaml").read_text() == COMPOSE
    new_compose = yaml.safe_load((destination / "r2e-test/environment/docker-compose.yaml").read_text())
    assert new_compose == {"services": {"main": {"tmpfs": ["/r2e_tests"]}}}
    assert (destination / "r2e-test/tests/test.sh").read_text() == "grade"
    assert json.loads((destination / "lightning-manifest.json").read_text())["tasks"] == 1
    with pytest.raises(ValueError):
        prep.prepare(source, destination)
    with pytest.raises(ValueError):
        prep.prepare(source, source / "nested")


@pytest.mark.parametrize(
    "value",
    [
        "",
        "[agent]\n",
        TOML.replace('"separate"', '"shared"'),
        TOML.replace("[agent]", '[agent]\nallowed_hosts = ["example.com"]'),
    ],
)
def test_bad_task_config_fails(value):
    with pytest.raises(ValueError):
        prep.transform_config(value)


@pytest.mark.parametrize("value", ["[]", "services: {}", "services: {main: {}, db: {}}", "services: {main: null}"])
def test_bad_compose_fails(value):
    with pytest.raises(ValueError):
        prep.transform_compose(value)


def test_empty_task_collection_fails(tmp_path):
    with pytest.raises(ValueError):
        prep.prepare(tmp_path, tmp_path.parent / "not-created")




def load_bridge(monkeypatch):
    class BaseAgent:
        def __init__(self, logs_dir, model_name, extra_env=None, **kwargs):
            self.logs_dir = logs_dir
            self.model_name = model_name
            self.extra_env = extra_env or {}

    for name in ("harbor", "harbor.agents", "harbor.agents.base"):
        module = types.ModuleType(name)
        module.BaseAgent = BaseAgent
        monkeypatch.setitem(sys.modules, name, module)
    name = "adaptive_branching.src.swe.lightning_harbor_agent"
    monkeypatch.delitem(sys.modules, name, raising=False)
    module = importlib.import_module(name)
    # Ensure this stub-derived class is not reused outside this test.
    monkeypatch.setitem(sys.modules, name, module)
    return module


@pytest.mark.parametrize("outcome", ["Submitted", "TurnLimit", "LengthTruncated", "error", "cancel"])
def test_harbor_restores_git_before_verifier_and_saves_metadata(monkeypatch, tmp_path, outcome):
    module = load_bridge(monkeypatch)
    seen = []

    class Environment:
        network_policy = SimpleNamespace(network_mode="no-network")

        def _egress_controlled_service_names(self):
            return ["main"]

        async def exec(self, command, **kwargs):
            seen.append((command, kwargs))
            return SimpleNamespace(return_code=0, stdout="", stderr="")

    async def loop(instruction, query, execute, *, config, episode):
        assert instruction == "fix" and config.max_tokens == 12288
        await execute("printf hi", 120)
        episode.turns = 1
        episode.max_prompt_tokens = 100
        episode.messages = [{"role": "user", "content": "fix"}]
        if outcome == "error":
            raise httpx.ConnectError("broken model")
        if outcome == "cancel":
            raise asyncio.CancelledError()
        episode.stop_reason = outcome

    monkeypatch.setattr(module, "run_episode", loop)
    context = SimpleNamespace()
    bridge = module.LightningSweAgent(tmp_path, "openai/Qwen3.5-4B", extra_env={"OPENAI_API_BASE": "http://model/v1"})
    if outcome in {"error", "cancel"}:
        with pytest.raises(httpx.ConnectError if outcome == "error" else asyncio.CancelledError):
            asyncio.run(bridge.run("fix", Environment(), context))
        assert context.metadata["agent_exit_status"] == "AgentInfrastructureFailure"
    else:
        asyncio.run(bridge.run("fix", Environment(), context))
        assert context.metadata["agent_exit_status"] == outcome
    assert seen[0][0] == module._HIDE_GIT and seen[-1][0] == module._RESTORE_GIT
    assert "timeout --signal=TERM" in seen[1][0]
    assert seen[1][1]["cwd"] == "/testbed" and seen[1][1]["timeout_sec"] == 130
    assert (tmp_path / "lightning-trajectory.json").is_file()


def test_bridge_refuses_public_network_and_wrong_context(monkeypatch, tmp_path):
    module = load_bridge(monkeypatch)
    with pytest.raises(ValueError):
        module.LightningSweAgent(tmp_path, "openai/Qwen3.5-4B", max_seq_len=262144)
    bridge = module.LightningSweAgent(tmp_path, "openai/Qwen3.5-4B", extra_env={"OPENAI_API_BASE": "http://model/v1"})
    with pytest.raises(RuntimeError, match="no-network"):
        asyncio.run(
            bridge.run(
                "fix", SimpleNamespace(network_policy=SimpleNamespace(network_mode="public")), SimpleNamespace()
            )
        )


def test_failed_restore_cannot_be_reported_as_submitted(monkeypatch, tmp_path):
    module = load_bridge(monkeypatch)

    class Environment:
        network_policy = SimpleNamespace(network_mode="no-network")

        def _egress_controlled_service_names(self):
            return ["main"]

        async def exec(self, command, **kwargs):
            return SimpleNamespace(return_code=int(command == module._RESTORE_GIT), stdout="", stderr="failed")

    async def loop(*args, episode, **kwargs):
        episode.stop_reason = "Submitted"

    monkeypatch.setattr(module, "run_episode", loop)
    context = SimpleNamespace()
    bridge = module.LightningSweAgent(tmp_path, "openai/Qwen3.5-4B", extra_env={"OPENAI_API_BASE": "http://model/v1"})
    with pytest.raises(RuntimeError, match="restore"):
        asyncio.run(bridge.run("fix", Environment(), context))
    assert context.metadata["agent_exit_status"] == "AgentInfrastructureFailure"


def load_generate_types(monkeypatch):
    # base_types imports DataSource for annotations; isolate its Ray dependency.
    dependency = types.ModuleType("miles.rollout.data_source")
    dependency.DataSource = object
    monkeypatch.setitem(sys.modules, dependency.__name__, dependency)
    name = "miles.rollout.base_types"
    monkeypatch.delitem(sys.modules, name, raising=False)
    module = importlib.import_module(name)
    monkeypatch.setitem(sys.modules, name, module)
    return module.GenerateFnInput, module.GenerateFnOutput


def test_trace_preserves_long_row_without_mutating_shared_state(monkeypatch):
    GenerateFnInput, GenerateFnOutput = load_generate_types(monkeypatch)
    from miles.utils.types import Sample
    from adaptive_branching.src.swe import lightning_generate

    args = SimpleNamespace(generate_multi_samples=False, max_seq_len=81920)
    state = SimpleNamespace(args=args, tokenizer=SimpleNamespace(batch_decode=lambda spans, **kw: ["ok"] * len(spans)))
    row = Sample(
        tokens=[1] * 70000, response_length=69999, loss_mask=[1] * 69999, rollout_log_probs=[-0.1] * 69999, reward=1.0
    )
    input = GenerateFnInput(state=state, sample=Sample(), sampling_params={}, evaluation=False)
    calls = []

    async def trace(traced_input):
        assert traced_input is not input and traced_input.state is not state
        assert traced_input.args is not args and traced_input.args.max_seq_len is None
        assert traced_input.state.tokenizer is state.tokenizer
        assert args.max_seq_len == 81920
        calls.append(traced_input)
        return GenerateFnOutput(samples=row)

    module = types.ModuleType("miles.rollout.generate_hub.agentic_tool_call")
    module.generate = trace
    monkeypatch.setitem(sys.modules, module.__name__, module)
    result = asyncio.run(lightning_generate.generate(input))
    assert result.samples is row and len(row.tokens) == 70000
    assert row.response_length == len(row.loss_mask) == len(row.rollout_log_probs) == 69999
    assert row.reward == 1.0 and not row.remove_sample
    assert args.max_seq_len == 81920 and state.args is args and len(calls) == 1
    args.max_seq_len = 131072
    with pytest.raises(ValueError, match="must equal model_context"):
        asyncio.run(lightning_generate.generate(input))
    assert len(calls) == 1
    args.max_seq_len = 81920
    args.generate_multi_samples = True
    with pytest.raises(ValueError, match="merged trajectories"):
        asyncio.run(lightning_generate.generate(input))


@pytest.mark.parametrize("empty_list", [False, True])
def test_trace_empty_records_and_empty_list(monkeypatch, empty_list):
    GenerateFnInput, GenerateFnOutput = load_generate_types(monkeypatch)
    from miles.utils.types import Sample
    from adaptive_branching.src.swe import lightning_generate

    row = Sample(status=Sample.Status.ABORTED)
    input = GenerateFnInput(
        state=SimpleNamespace(args=SimpleNamespace(generate_multi_samples=False, max_seq_len=81920)),
        sample=Sample(),
        sampling_params={},
        evaluation=False,
    )

    async def trace(_):
        return GenerateFnOutput(samples=[] if empty_list else row)

    module = types.ModuleType("miles.rollout.generate_hub.agentic_tool_call")
    module.generate = trace
    monkeypatch.setitem(sys.modules, module.__name__, module)
    if empty_list:
        with pytest.raises(ValueError, match="empty sample list"):
            asyncio.run(lightning_generate.generate(input))
    else:
        result = asyncio.run(lightning_generate.generate(input))
        assert result.samples.status == Sample.Status.ABORTED
        assert result.samples.remove_sample and result.samples.metadata["agent_excluded_from_training"]


@pytest.mark.parametrize("prior_turn", [False, True])
@pytest.mark.parametrize("verifier_reward", [0, 1])
def test_length_episode_merges_to_trainable_zero(monkeypatch, prior_turn, verifier_reward):
    from miles.utils.types import Sample
    from miles.rollout.generate_utils.sample_utils import merge_samples
    from adaptive_branching.src.swe import agent, lightning_generate, lightning_reward
    from adaptive_branching.tests.swe.test_lightning_agent_on_cpu import completion, run_loop

    GenerateFnInput, GenerateFnOutput = load_generate_types(monkeypatch)
    bodies = ([completion("true", prompt=1, output=1)] if prior_turn else []) + [
        completion("echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT", finish="length", prompt=3, output=1)
    ]
    episode, calls = run_loop(bodies, [("ok", 0)] if prior_turn else [])
    assert episode.stop_reason == "LengthTruncated" and len(calls) == len(bodies)
    decoded = agent.decode_result(
        {
            "exit_status": episode.stop_reason,
            "agent_metrics": episode.metrics(),
            "reward": verifier_reward,
            "eval_report": {"reward": verifier_reward},
        }
    )
    tokenizer = SimpleNamespace(
        decode=lambda ids: " ".join(map(str, ids)),
        batch_decode=lambda spans, **kw: [" ".join(map(str, ids)) for ids in spans],
    )
    rows = []
    if prior_turn:
        rows.append(
            Sample(
                prompt="fix",
                tokens=[1, 2],
                response="2",
                response_length=1,
                loss_mask=[1],
                rollout_log_probs=[-0.1],
                metadata=dict(decoded),
                status=Sample.Status.COMPLETED,
            )
        )
    rows.append(
        Sample(
            prompt="fix",
            tokens=[1, 2, 3, 4],
            response="4",
            response_length=1,
            loss_mask=[1],
            rollout_log_probs=[-0.2],
            metadata=dict(decoded),
            status=Sample.Status.TRUNCATED,
        )
    )
    merged = merge_samples(rows, tokenizer)

    async def trace(_):
        return GenerateFnOutput(samples=merged)

    module = types.ModuleType("miles.rollout.generate_hub.agentic_tool_call")
    module.generate = trace
    monkeypatch.setitem(sys.modules, module.__name__, module)
    input = GenerateFnInput(
        state=SimpleNamespace(
            args=SimpleNamespace(generate_multi_samples=False, max_seq_len=81920), tokenizer=tokenizer
        ),
        sample=Sample(),
        sampling_params={},
        evaluation=False,
    )
    row = asyncio.run(lightning_generate.generate(input)).samples
    reward = lightning_reward.score_sample(row)
    assert reward["score"] == 0 and not reward["acc"] and not row.remove_sample
    assert not row.metadata["agent_excluded_from_training"]
    assert row.tokens == [1, 2, 3, 4] and row.status == Sample.Status.TRUNCATED
    assert row.loss_mask == ([1, 0, 1] if prior_turn else [1])
    assert row.rollout_log_probs == ([-0.1, 0.0, -0.2] if prior_turn else [-0.2])
    row.validate()
