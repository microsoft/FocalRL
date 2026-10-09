"""Build the pinned host runtime used by Harbor's external mini-swe controller."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_MINI_SWE_AGENT_VERSION = "2.4.6"
DEFAULT_LITELLM_VERSION = "1.98.0"
MANIFEST_NAME = "runtime-manifest.json"


def _replace_once(path: Path, old: str, new: str, *, marker: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"patch target does not exist: {path}")
    source = path.read_text(encoding="utf-8")
    if marker in source:
        raise ValueError(f"patch target is already modified ({marker!r}): {path}")
    count = source.count(old)
    if count != 1:
        raise ValueError(f"expected one patch anchor in {path}, found {count}: {old!r}")
    path.write_text(source.replace(old, new, 1), encoding="utf-8")


def patch_runtime(purelib: Path) -> dict[str, str]:
    """Add replay, terminal-length, forced-submit, and timing hooks to mini-swe-agent 2.4.6."""
    if not purelib.is_dir():
        raise NotADirectoryError(f"site-packages directory does not exist: {purelib}")

    default_path = purelib / "minisweagent/agents/default.py"
    helper_anchor = "\n\nclass AgentConfig(BaseModel):"
    helpers = r'''

_HARBOR_FORCE_SUBMIT_PROMPT = """You have reached the configured trajectory limit. Do not perform more analysis or tests. Immediately submit the current repository state by calling exactly `echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`."""


def _harbor_last_seq_len(messages: list[dict]) -> int:
    for message in reversed(messages):
        usage = ((message.get("extra") or {}).get("response") or {}).get("usage") or {}
        prompt_tokens = usage.get("prompt_tokens") or 0
        completion_tokens = usage.get("completion_tokens") or 0
        if prompt_tokens or completion_tokens:
            return int(prompt_tokens) + int(completion_tokens)
    return 0


def _harbor_force_submit_reason(agent) -> str | None:
    if 0 < agent.config.step_limit <= agent.n_calls:
        return "max_turns"
    if not agent.config.max_seq_len and not agent.config.context_reserve_tokens:
        return None
    if not 0 < agent.config.context_reserve_tokens < agent.config.max_seq_len:
        raise ValueError("context_reserve_tokens must be positive and smaller than max_seq_len")
    if _harbor_last_seq_len(agent.messages) >= agent.config.max_seq_len - agent.config.context_reserve_tokens:
        return "context_reserve"
    return None


def _harbor_restore_replay(agent) -> bool:
    if agent.config.replay_path is None:
        return False
    replay = json.loads(agent.config.replay_path.read_text(encoding="utf-8"))
    agent.messages = replay["messages"]
    agent.n_calls = replay["n_calls"]
    for item in replay["actions"]:
        output = agent.env.execute(item["action"])
        if output["returncode"] != item["expected_returncode"]:
            raise RuntimeError(
                f"replay diverged at source turn {item['turn']}: "
                f"expected returncode {item['expected_returncode']}, got {output['returncode']}"
            )
    return True
'''
    _replace_once(default_path, helper_anchor, helpers + helper_anchor, marker="_HARBOR_FORCE_SUBMIT_PROMPT")

    config_anchor = '''    output_path: Path | None = None
    """Save the trajectory to this path."""
'''
    config_replacement = config_anchor + """    max_seq_len: int = 0
    context_reserve_tokens: int = 0
    replay_path: Path | None = None
    force_submit_on_limit: bool = True
"""
    _replace_once(default_path, config_anchor, config_replacement, marker="context_reserve_tokens: int = 0")

    replay_anchor = """        self.messages = []
        self.add_messages(
            self.model.format_message(role="system", content=self._render_template(self.config.system_template)),
            self.model.format_message(role="user", content=self._render_template(self.config.instance_template)),
        )"""
    replay_replacement = """        self.messages = []
        if not _harbor_restore_replay(self):
            self.add_messages(
                self.model.format_message(role="system", content=self._render_template(self.config.system_template)),
                self.model.format_message(role="user", content=self._render_template(self.config.instance_template)),
            )"""
    _replace_once(default_path, replay_anchor, replay_replacement, marker="_harbor_restore_replay(self)")

    query_anchor = '''    def query(self) -> dict:
        """Query the model and return model messages. Override to add hooks."""
        if 0 < self.config.step_limit <= self.n_calls or 0 < self.config.cost_limit <= self.cost:
            raise LimitsExceeded('''
    query_replacement = '''    def query(self) -> dict:
        """Query the model and return model messages. Override to add hooks."""
        forced_reason = getattr(self, "_harbor_forced_submit_reason", None)
        if forced_reason is not None:
            raise LimitsExceeded(
                {
                    "role": "exit",
                    "content": "LimitsExceeded",
                    "extra": {"exit_status": "LimitsExceeded", "submission": ""},
                }
            )
        forced_reason = _harbor_force_submit_reason(self)
        if forced_reason is not None and not self.config.force_submit_on_limit:
            terminal_extra = {"exit_status": "LimitsExceeded", "submission": ""}
            if forced_reason == "context_reserve":
                # Preserve the context stop even when Local skips a forced model call.
                terminal_extra["harbor_forced_submit_reason"] = forced_reason
            raise LimitsExceeded(
                {
                    "role": "exit",
                    "content": "LimitsExceeded",
                    "extra": terminal_extra,
                }
            )
        if forced_reason is not None:
            self.add_messages(
                self.model.format_message(
                    role="user",
                    content=_HARBOR_FORCE_SUBMIT_PROMPT,
                    extra={"harbor_forced_submit_reason": forced_reason},
                )
            )
            self._harbor_forced_submit_reason = forced_reason
        if 0 < self.config.cost_limit <= self.cost:
            raise LimitsExceeded('''
    _replace_once(default_path, query_anchor, query_replacement, marker='_harbor_forced_submit_reason", None)')

    query_call = "        message = self.model.query(self.messages)"
    _replace_once(
        default_path,
        query_call,
        """        _harbor_t0 = time.time()
        message = self.model.query(self.messages)
        message.setdefault("extra", {})["harbor_llm_wait_msec"] = (time.time() - _harbor_t0) * 1000""",
        marker="harbor_llm_wait_msec",
    )
    execute_anchor = '        outputs = [self.env.execute(action) for action in message.get("extra", {}).get("actions", [])]\n        return self.add_messages(*self.model.format_observation_messages(message, outputs, self.get_template_vars()))'
    execute_replacement = """        if message.get("extra", {}).get("harbor_finish_reason") == "length":
            # query() already saved the assistant output and charged its cost.
            raise InterruptAgentFlow(
                {
                    "role": "exit",
                    "content": "LengthTruncated",
                    "extra": {"exit_status": "LengthTruncated", "submission": ""},
                }
            )
        _harbor_t0 = time.time()
        outputs = [self.env.execute(action) for action in message.get("extra", {}).get("actions", [])]
        elapsed_msec = (time.time() - _harbor_t0) * 1000
        observations = self.model.format_observation_messages(message, outputs, self.get_template_vars())
        if observations:
            observations[0].setdefault("extra", {})["harbor_tool_exec_msec"] = elapsed_msec
        return self.add_messages(*observations)"""
    _replace_once(default_path, execute_anchor, execute_replacement, marker="harbor_tool_exec_msec")

    model_path = purelib / "minisweagent/models/litellm_model.py"
    _replace_once(
        model_path,
        "            actions = self._parse_actions(response)",
        # A truncated/missing tool call must not enter the FormatError recovery loop.
        '            actions = [] if response.choices[0].finish_reason == "length" else self._parse_actions(response)',
        marker='actions = [] if response.choices[0].finish_reason == "length"',
    )
    _replace_once(
        model_path,
        '            "actions": actions,',
        '            "actions": actions,\n            "harbor_finish_reason": response.choices[0].finish_reason,',
        marker='"harbor_finish_reason":',
    )

    litellm_path = purelib / "litellm/types/utils.py"
    _replace_once(
        litellm_path,
        "    ChatCompletionReasoningItem,\n    ChatCompletionRedactedThinkingBlock,",
        "    ChatCompletionReasoningItem,\n    ChatCompletionReasoningSummaryTextBlock,\n    ChatCompletionRedactedThinkingBlock,",
        marker="ChatCompletionReasoningSummaryTextBlock,\n    ChatCompletionRedactedThinkingBlock",
    )

    required = [
        default_path,
        model_path,
        litellm_path,
        purelib / "minisweagent/config/benchmarks/swebench.yaml",
    ]
    return {str(path.relative_to(purelib)): _sha256(path) for path in required}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run(command: list[str], *, env: dict[str, str] | None = None) -> None:
    subprocess.run(command, check=True, env=env)


def _runtime_purelib(python: Path) -> Path:
    output = subprocess.check_output(
        [str(python), "-c", "import sysconfig; print(sysconfig.get_path('purelib'))"],
        text=True,
    ).strip()
    purelib = Path(output)
    if not purelib.is_dir():
        raise NotADirectoryError(f"runtime reported a missing site-packages directory: {purelib}")
    return purelib


def build_runtime(
    *,
    output: Path,
    python_runtime: Path,
    uv: Path,
    mini_swe_agent_version: str,
    litellm_version: str,
    index_url: str | None,
    cache_dir: Path | None,
) -> None:
    output = output.expanduser().resolve()
    python_runtime = python_runtime.expanduser().resolve()
    uv = uv.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to replace existing runtime: {output}")
    python = python_runtime / "bin/python"
    for name, path in (("python", python), ("uv", uv)):
        if not path.is_file() or not os.access(path, os.X_OK):
            raise FileNotFoundError(f"{name} executable does not exist or is not executable: {path}")
    if mini_swe_agent_version != DEFAULT_MINI_SWE_AGENT_VERSION:
        raise ValueError(f"mini-swe-agent version must be {DEFAULT_MINI_SWE_AGENT_VERSION}")
    if litellm_version != DEFAULT_LITELLM_VERSION:
        raise ValueError(f"LiteLLM version must be {DEFAULT_LITELLM_VERSION}")

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.with_name(f".{output.name}.building-{os.getpid()}")
    if staging.exists():
        raise FileExistsError(f"staging directory already exists: {staging}")
    env = os.environ.copy()
    if cache_dir is not None:
        cache_dir = cache_dir.expanduser().resolve()
        if not cache_dir.is_dir():
            raise NotADirectoryError(f"cache_dir does not exist: {cache_dir}")
        env["UV_CACHE_DIR"] = str(cache_dir)
    if index_url:
        env["UV_DEFAULT_INDEX"] = index_url

    try:
        _run(
            [
                str(uv),
                "venv",
                "--relocatable",
                "--python",
                str(python),
                "--no-managed-python",
                str(staging),
            ],
            env=env,
        )
        runtime_python = staging / "bin/python"
        _run(
            [
                str(uv),
                "pip",
                "install",
                "--link-mode",
                "copy",
                "--python",
                str(runtime_python),
                f"mini-swe-agent=={mini_swe_agent_version}",
                f"litellm=={litellm_version}",
            ],
            env=env,
        )
        purelib = _runtime_purelib(runtime_python)
        required_files = patch_runtime(purelib)
        site_packages = purelib.relative_to(staging)
        manifest = {
            "schema_version": 2,
            "built_at_utc": datetime.now(timezone.utc).isoformat(),
            "mini_swe_agent_version": mini_swe_agent_version,
            "litellm_version": litellm_version,
            "python_version": subprocess.check_output(
                [str(runtime_python), "-c", "import platform; print(platform.python_version())"],
                text=True,
            ).strip(),
            "site_packages": str(site_packages),
            "required_files_sha256": required_files,
        }
        (staging / MANIFEST_NAME).write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        staging.rename(output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--python-runtime", type=Path, required=True)
    parser.add_argument("--uv", type=Path, required=True)
    parser.add_argument("--mini-swe-agent-version", default=DEFAULT_MINI_SWE_AGENT_VERSION)
    parser.add_argument("--litellm-version", default=DEFAULT_LITELLM_VERSION)
    parser.add_argument("--index-url")
    parser.add_argument("--cache-dir", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_runtime(
        output=args.output,
        python_runtime=args.python_runtime,
        uv=args.uv,
        mini_swe_agent_version=args.mini_swe_agent_version,
        litellm_version=args.litellm_version,
        index_url=args.index_url,
        cache_dir=args.cache_dir,
    )


if __name__ == "__main__":
    main()
