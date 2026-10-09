"""Real Docker replay smoke with scripted completions and optional lifecycle gates.

No GPU/model is used. Creates two isolated containers, checks source-to-replay
state, then executes a ten-turn local continuation. Does not start training.
"""

import argparse
import asyncio
import json
import sys
import time
import uuid
from pathlib import Path
from types import SimpleNamespace


def validate_args(args):
    if not isinstance(args.image, str) or not args.image.startswith("sha256:") or len(args.image) != 71:
        raise ValueError("smoke requires a pinned local image digest")
    if any(c not in "0123456789abcdef" for c in args.image[7:]):
        raise ValueError("smoke image digest must be hexadecimal")
    if args.gate_module_dir is not None and not Path(args.gate_module_dir).is_dir():
        raise ValueError("explicit lifecycle gate directory must exist")
    if not Path(args.socket).is_socket():
        raise ValueError("smoke requires an existing Docker socket")
    if Path(args.output).exists():
        raise FileExistsError(args.output)


async def check(args):
    validate_args(args)
    import docker
    from docker.api.container import ContainerApiMixin
    from adaptive_branching.src.swe import lightning_harbor_agent as bridge
    from adaptive_branching.src.swe.lightning_local_harbor_agent import LocalLightningSweAgent
    from adaptive_branching.src.swe.lightning_replay import build_replay

    if args.gate_module_dir is not None:
        sys.path.insert(0, str(Path(args.gate_module_dir).resolve()))
        from verifier_lifecycle import install_sdk_gates

        install_sdk_gates(ContainerApiMixin, docker.__version__)
    client = docker.DockerClient(base_url="unix://" + str(args.socket), timeout=180)
    client.images.get(args.image)
    output = Path(args.output)
    output.mkdir(parents=True)
    containers = []
    started = time.time()

    class Environment:
        network_policy = SimpleNamespace(network_mode="no-network")

        def __init__(self, container):
            self.container = container

        def _egress_controlled_service_names(self):
            return ["main"]

        async def exec(self, command, *, cwd, timeout_sec, env=None):
            # Docker is truly network=none; the egress-sidecar method above is
            # only the bridge protocol adapter for this isolated smoke.
            if not isinstance(command, str) or not command or type(timeout_sec) is not int or timeout_sec <= 0:
                raise ValueError("smoke exec requires a command and positive timeout")
            if "def fingerprint(root)" in command:
                raise AssertionError("Full/Local smoke must not launch filesystem capture")
            result = await asyncio.wait_for(
                asyncio.to_thread(
                    self.container.exec_run, ["bash", "-c", command], workdir=cwd, environment=env, demux=True
                ),
                timeout=timeout_sec,
            )
            stdout, stderr = result.output
            return SimpleNamespace(
                return_code=result.exit_code, stdout=(stdout or b"").decode(), stderr=(stderr or b"").decode()
            )

    calls = []
    commands = [
        "mkdir -p /testbed/.replay-smoke; printf abc > /testbed/.replay-smoke/file; "
        "chmod 755 /testbed/.replay-smoke/file; ln -s file /testbed/.replay-smoke/link; "
        "printf temp > /tmp/replay-smoke-state",
        "cat /testbed/.replay-smoke/file /tmp/replay-smoke-state",
        "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT",
    ]

    async def query(client, url, model, messages, config):
        index = len(calls)
        calls.append(json.loads(json.dumps(messages)))
        command = commands[index] if index < 3 else "cat /testbed/.replay-smoke/file /tmp/replay-smoke-state"
        return {
            "choices": [
                {
                    "message": {
                        "content": "",
                        "reasoning_content": "inspect state",
                        "tool_calls": [
                            {
                                "id": f"call-{index}",
                                "type": "function",
                                "function": {"name": "bash", "arguments": json.dumps({"command": command})},
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {"prompt_tokens": 100, "completion_tokens": 20},
        }

    old_query = bridge.query_model
    bridge.query_model = query
    try:
        for kind in ("source", "local"):
            container = await asyncio.to_thread(
                client.containers.create,
                args.image,
                command=["sleep", "infinity"],
                name="lightning-local-smoke-" + kind + "-" + uuid.uuid4().hex[:10],
                network_mode="none",
                labels={"miles.local-replay-smoke": "20260911"},
            )
            containers.append(container)
            await asyncio.to_thread(container.start)
        source_env, local_env = map(Environment, containers)
        for environment in (source_env, local_env):
            # Use the actual task image. Existing synthetic baseline must pass
            # the same setup checks as production; no replacement or git reset.
            result = await environment.exec(
                "test -d /testbed/.git && test ! -e /testbed/.replay-smoke && " "test ! -e /tmp/replay-smoke-state",
                cwd="/testbed",
                timeout_sec=30,
            )
            if result.return_code != 0:
                raise RuntimeError("smoke image does not have a clean task baseline")
        source_agent = LocalLightningSweAgent(
            output / "source", "openai/smoke", extra_env={"OPENAI_API_BASE": "http://unused/v1"}
        )
        await source_agent.setup(source_env)
        source_context = SimpleNamespace()
        await source_agent.run("smoke issue", source_env, source_context)
        if source_context.metadata["agent_exit_status"] != "Submitted":
            raise RuntimeError("source smoke did not submit")
        trajectory = json.loads((output / "source/lightning-trajectory.json").read_text())
        state = build_replay(trajectory, 2)
        (output / "local").mkdir()
        (output / "local/replay.json").write_text(json.dumps(state))
        local_agent = LocalLightningSweAgent(
            output / "local",
            "openai/smoke",
            replay=True,
            max_turns=11,
            extra_env={"OPENAI_API_BASE": "http://unused/v1"},
        )
        await local_agent.setup(local_env)
        context = SimpleNamespace()
        await local_agent.run("smoke issue", local_env, context)
        if context.metadata["agent_exit_status"] != "LocalHorizon" or context.n_steps != 11 or len(calls) != 13:
            raise RuntimeError(f"local horizon/turn count mismatch: {context.metadata}, calls={len(calls)}")
        if calls[3] != state["messages"]:
            raise RuntimeError("first continuation model call differs from restored prefix")
        result = {
            "status": "PASS",
            "model": "scripted",
            "image": args.image,
            "model_calls": len(calls),
            "prefix_calls": 1,
            "new_local_calls": 10,
            "replay_schema": state["schema"],
            "lifecycle_gates": args.gate_module_dir is not None,
            "workspace_roots": ["/testbed", "/tmp"],
            "elapsed_seconds": time.time() - started,
        }
        (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        return result
    finally:
        bridge.query_model = old_query
        for container in reversed(containers):
            await asyncio.to_thread(container.remove, force=True, v=True)
        client.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True)
    parser.add_argument("--gate-module-dir", type=Path, help="Opt into legacy SDK lifecycle gates")
    parser.add_argument("--socket", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(asyncio.run(check(args))))


if __name__ == "__main__":
    main()
