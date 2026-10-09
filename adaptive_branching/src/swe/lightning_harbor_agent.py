"""Harbor host-process bridge for the standalone SWE Lightning ReAct loop."""

import asyncio
import json
import shlex
from pathlib import Path

import httpx
from harbor.agents.base import BaseAgent

from adaptive_branching.src.swe.agent import AgentConfig, Episode, query_model, run_episode, training_config
from adaptive_branching.src.swe.lightning_reference import _CMD_ENV

# R2E's isolated verifier expects .git restored before its collect hook runs.
_HIDE_GIT = "test -d .git && test ! -e /opt/agl_tmp && mv .git /opt/agl_tmp"
_RESTORE_GIT = """if ! test -d /opt/agl_tmp; then
    echo 'missing saved synthetic Git directory: /opt/agl_tmp' >&2
    exit 1
fi
if test -e .git; then
    echo 'refusing to overwrite existing .git during synthetic Git restoration' >&2
    exit 1
fi
mv /opt/agl_tmp .git"""


class LightningSweAgent(BaseAgent):
    CONFIG = AgentConfig()

    def __init__(
        self, logs_dir: Path, model_name=None, *, max_seq_len=AgentConfig.model_context, max_turns=None, **kwargs
    ):
        if not isinstance(self.CONFIG, AgentConfig):
            raise TypeError("CONFIG must be an AgentConfig")
        if type(max_seq_len) is not int or max_seq_len != self.CONFIG.model_context:
            raise ValueError(f"Lightning model context must be {self.CONFIG.model_context}, got {max_seq_len}")
        if max_turns is not None and (type(max_turns) is not int or max_turns != self.CONFIG.max_turns):
            raise ValueError(f"Lightning max_turns must be {self.CONFIG.max_turns}, got {max_turns}")
        if not isinstance(model_name, str) or not model_name.startswith("openai/"):
            raise ValueError("LightningSweAgent requires openai/<model> model_name")
        super().__init__(logs_dir=logs_dir, model_name=model_name, **kwargs)
        self.config = self.CONFIG
        self.episode = Episode()
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.base_url = self.extra_env.get("OPENAI_API_BASE", "").rstrip("/")
        if not self.base_url.startswith(("http://", "https://")):
            raise ValueError("missing absolute OPENAI_API_BASE for the Miles session")

    @staticmethod
    def name():
        return "lightning-swe-react"

    def version(self):
        return "agl-218f1f7c-miles-4-length-terminal"

    async def setup(self, environment):
        # No package installs. These are prepared R2E images with synthetic git.
        result = await environment.exec(
            "test -d /testbed/.git && command -v bash && command -v timeout && "
            'test "$(git rev-list --all --count)" -eq 1 && test -z "$(git remote)"',
            cwd="/testbed",
            timeout_sec=30,
        )
        if result.return_code != 0:
            raise RuntimeError(f"Lightning R2E sandbox preflight failed: {result.stderr}")

    async def run(self, instruction, environment, context):
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError("Harbor instruction must be nonempty")
        if environment.network_policy.network_mode != "no-network":
            raise RuntimeError("Lightning agent requires an enforced no-network sandbox policy")
        controlled_services = getattr(environment, "_egress_controlled_service_names", None)
        if not callable(controlled_services) or "main" not in controlled_services():
            raise RuntimeError("Lightning requires Harbor Docker egress control on the main service")
        moved = await environment.exec(_HIDE_GIT, cwd="/testbed", timeout_sec=30)
        if moved.return_code != 0:
            raise RuntimeError(f"failed to relocate synthetic git: {moved.stderr}")

        async def execute(action, timeout):
            if not isinstance(action, str) or type(timeout) is not int or timeout <= 0:
                raise ValueError("shell command must be text and timeout positive")
            result = await environment.exec(
                f"exec timeout --signal=TERM --kill-after=5s {timeout}s bash -c {shlex.quote(action)}",
                cwd="/testbed",
                env=dict(_CMD_ENV),
                timeout_sec=timeout + 10,
            )
            if type(result.return_code) is not int:
                raise TypeError("shell bridge returned no integer exit status")
            # Match upstream TimeoutExpired observation, while ensuring the
            # container-side command is killed, not merely its HTTP request.
            if result.return_code == 124:
                return f"[timed out after {timeout}s]", 124
            return (result.stdout or "") + (result.stderr or ""), result.return_code

        diagnostics = {}
        try:
            async with httpx.AsyncClient(
                timeout=12000, headers={"Authorization": f"Bearer {self.extra_env.get('OPENAI_API_KEY', 'dummy')}"}
            ) as client:

                async def query(messages):
                    return await query_model(
                        client,
                        self.base_url + "/chat/completions",
                        self.model_name.removeprefix("openai/"),
                        messages,
                        self.config,
                    )

                await self.run_loop(instruction, query, execute, environment)
        except BaseException as exc:
            diagnostics["agent_primary_error"] = f"{type(exc).__name__}: {exc}"
            self.episode.stop_reason = "AgentInfrastructureFailure"
            raise
        finally:
            status_before_cleanup = self.episode.stop_reason
            try:
                restored = await asyncio.shield(environment.exec(_RESTORE_GIT, cwd="/testbed", timeout_sec=30))
                if type(restored.return_code) is not int:
                    raise TypeError(f"git restore returned invalid exit status: {restored.return_code!r}")
                if restored.return_code != 0:
                    diagnostics["agent_git_restore_result"] = {
                        "return_code": restored.return_code,
                        "stdout": restored.stdout,
                        "stderr": restored.stderr,
                    }
                    raise RuntimeError(
                        "failed to restore synthetic git before verifier: "
                        f"return_code={restored.return_code}, stdout={restored.stdout!r}, stderr={restored.stderr!r}"
                    )
            except BaseException as exc:
                self.episode.stop_reason = "AgentInfrastructureFailure"
                diagnostics["agent_exit_status_before_cleanup"] = status_before_cleanup
                diagnostics["agent_cleanup_error"] = f"{type(exc).__name__}: {exc}"
                raise
            finally:
                context.n_steps = self.episode.turns
                context.n_input_tokens = self.episode.input_tokens
                context.n_output_tokens = self.episode.output_tokens
                context.metadata = {**self.episode.metrics(), **diagnostics}
                (self.logs_dir / "lightning-trajectory.json").write_text(
                    json.dumps(self.trajectory(context.metadata), ensure_ascii=False) + "\n",
                    encoding="utf-8",
                )
        # Returning normally invokes Harbor's isolated verifier even at the cap.
        # There is deliberately no model-generated patch.txt and no forced call.

    async def run_loop(self, instruction, query, execute, environment):
        if not callable(query) or not callable(execute):
            raise TypeError("run_loop requires model and shell callables")
        await run_episode(instruction, query, execute, config=self.config, episode=self.episode)

    def trajectory(self, metrics):
        if not isinstance(metrics, dict):
            raise TypeError("trajectory metrics must be an object")
        return {"messages": self.episode.messages, "metrics": metrics}


class LargeLightningSweAgent(LightningSweAgent):
    """200-turn/256K RL runtime; retain the original per-call output budget."""

    CONFIG = training_config("large")
