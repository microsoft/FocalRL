"""Build an isolated Harbor source tree with native replay endpoints/kwargs."""

import argparse
import ast
import shutil
from pathlib import Path

NATIVE_AGENT = "adaptive_branching.src.swe.lightning_local_harbor_agent:LocalLightningSweAgent"
LARGE_NATIVE_AGENT = "adaptive_branching.src.swe.lightning_local_harbor_agent:LargeLocalLightningSweAgent"


def patch_source(name, source):
    if not isinstance(source, str) or not source:
        raise ValueError("Harbor source must be nonempty")
    if name == "miles_agent_server.py":
        old = "from agent_server.replay import create_replay, load_trial_messages"
        new = "from adaptive_branching.src.swe.lightning_replay import create_replay, load_trial_messages"
    elif name == "agent_server/trial_runner.py":
        old = "    if request.agent_name in _HOST_PROCESS_AGENTS:\n"
        new = f"""    if request.agent_name in ({NATIVE_AGENT!r}, {LARGE_NATIVE_AGENT!r}):
        expected_context = 262144 if request.agent_name == {LARGE_NATIVE_AGENT!r} else 81920
        if request.max_seq_len != expected_context or request.context_reserve_tokens is not None:
            raise ValueError(f"Local Lightning requires the {{expected_context}} model context without a reserve guard")
        if type(request.max_turns) is not int or request.max_turns <= 0:
            raise ValueError("Local Lightning requires explicit max_turns")
        replay = request.replay_path is not None
        if replay != (request.run_verifier is False):
            raise ValueError("Only local replay may disable the verifier")
        if request.force_submit_on_limit:
            raise ValueError("Local Lightning must not force a final submission")
        return {{"max_seq_len": request.max_seq_len, "max_turns": request.max_turns, "replay": replay}}, {{
            "OPENAI_API_BASE": request.base_url, "OPENAI_API_KEY": request.api_key or "dummy",
        }}

""" + old
    else:
        raise ValueError(f"unsupported Harbor file: {name}")
    if source.count(old) != 1 or new in source:
        raise ValueError(f"unexpected or already patched Harbor source: {name}")
    return source.replace(old, new, 1)


def disable_lifecycle_limit(source):
    """Remove lock acquisition from both host and nested-verifier gate contexts."""
    if not isinstance(source, str) or not source.strip():
        raise ValueError("lifecycle gate source must be nonempty")
    marker = "LIFECYCLE_LIMIT_ENABLED = False"
    if marker in source:
        raise ValueError("lifecycle limit is already disabled")
    tree = ast.parse(source)
    lines = source.splitlines(keepends=True)
    replacements = []
    for name, node_type, decorator, prefix in (
        ("environment_gate", ast.AsyncFunctionDef, "asynccontextmanager", "async "),
        ("sync_environment_gate", ast.FunctionDef, "contextmanager", ""),
    ):
        matches = [
            node
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name
        ]
        if len(matches) != 1 or not isinstance(matches[0], node_type):
            raise ValueError(f"expected exactly one {name} context")
        node = matches[0]
        expected = ast.parse("def f(root=LOCK_ROOT, slots=DEFAULT_SLOTS): pass").body[0].args
        if ast.dump(node.args) != ast.dump(expected) or [ast.unparse(d) for d in node.decorator_list] != [decorator]:
            raise ValueError(f"unexpected lifecycle context signature: {name}")
        if not any(
            isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "try_acquire"
            for n in ast.walk(node)
        ):
            raise ValueError(f"missing expected lock acquisition in {name}")
        replacement = (
            f"@{decorator}\n{prefix}def {name}(root=LOCK_ROOT, slots=DEFAULT_SLOTS):\n"
            '    """Local RL: validate arguments without acquiring lifecycle slots."""\n'
            "    validate_gate(root, slots)\n    yield\n"
        )
        replacements.append((node.decorator_list[0].lineno - 1, node.end_lineno, replacement))
    for start, end, replacement in sorted(replacements, reverse=True):
        lines[start:end] = [replacement]
    result = "".join(lines) + "\n# Explicit Local RL override; evaluation source is unchanged.\n" + marker + "\n"
    compile(result, "environment_gate.py", "exec")
    return result


def prepare(source, destination, *, disable_lifecycle_slots=False):
    if type(disable_lifecycle_slots) is not bool:
        raise ValueError("disable_lifecycle_slots must be a boolean")
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if not source.is_dir() or destination.exists() or source == destination or source in destination.parents:
        raise ValueError("prepare requires an existing source and a new external destination")
    names = ("miles_agent_server.py", "agent_server/trial_runner.py")
    patched = {name: patch_source(name, (source / name).read_text()) for name in names}
    if disable_lifecycle_slots:
        gate = "src/harbor/trial/environment_gate.py"
        if not (source / gate).is_file():
            raise FileNotFoundError(source / gate)
        patched[gate] = disable_lifecycle_limit((source / gate).read_text())
    shutil.copytree(source, destination, ignore=shutil.ignore_patterns(".git", "__pycache__", "*.pyc"))
    for name, content in patched.items():
        (destination / name).write_text(content)
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--disable-lifecycle-slots", action="store_true")
    args = parser.parse_args()
    print(prepare(args.source, args.destination, disable_lifecycle_slots=args.disable_lifecycle_slots))


if __name__ == "__main__":
    main()
