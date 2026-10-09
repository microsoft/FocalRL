"""Native-tool replay accepted by matching command exit codes.

New v3 artifacts omit filesystem fingerprints; v2 artifacts remain readable.
Restoration does not capture or compare filesystem state or command output.
No judge or golden data enters state.
"""

import copy
import hashlib
import json
import os
import stat
from pathlib import Path

from adaptive_branching.src.swe import lightning_reference as reference
from adaptive_branching.src.swe import thinking_protocol as protocol

SCHEMA = "lightning-native-replay-v3"
LEGACY_SCHEMA = "lightning-native-replay-v2"


def _trial(root, trial_dir):
    root, trial = Path(root).resolve(), Path(trial_dir).resolve()
    if not root.is_dir() or not trial.is_dir() or trial.parent != root:
        raise ValueError("trial must be an existing direct child of the controller's trial root")
    return trial


def load_trial_messages(trials_dir, trial_dir):
    path = _trial(trials_dir, trial_dir) / "agent" / "lightning-trajectory.json"
    data = json.loads(path.read_text())
    if not isinstance(data, dict) or data.get("replay_schema") not in (SCHEMA, LEGACY_SCHEMA):
        raise ValueError(f"missing native replay schema: {path}")
    messages = data.get("messages")
    if not isinstance(messages, list) or not messages or any(not isinstance(m, dict) for m in messages):
        raise ValueError(f"invalid messages: {path}")
    return messages


def create_replay(trials_dir, trial_dir, selected_turn):
    trial = _trial(trials_dir, trial_dir)
    state = build_replay(json.loads((trial / "agent" / "lightning-trajectory.json").read_text()), selected_turn)
    path = trial / "agent" / f"lightning-replay-turn-{selected_turn}.json"
    # Artifact endpoints may retry. Identical existing artifacts are reusable;
    # a changed source at the same location is not silently overwritten.
    serialized = json.dumps(state, ensure_ascii=False, sort_keys=True)
    if path.exists():
        if path.read_text() != serialized:
            raise RuntimeError(f"conflicting replay artifact: {path}")
    else:
        with path.open("x") as stream:
            stream.write(serialized)
    return path, state


def _digest(value):
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError(f"invalid SHA256: {value!r}")
    return value


def validate_prefix(messages, n_calls, *, problem=None, max_turns=100):
    if type(max_turns) is not int or max_turns <= 0:
        raise ValueError("replay max_turns must be a positive integer")
    if type(n_calls) is not int or not 0 <= n_calls < max_turns:
        raise ValueError(f"replay prefix requires 0 <= n_calls < {max_turns}")
    if not isinstance(messages, list) or len(messages) < 2 or any(not isinstance(m, dict) for m in messages):
        raise ValueError("replay prefix must contain system and user messages")
    if messages[0] != {"role": "system", "content": protocol.SYSTEM_PROMPT}:
        raise ValueError("replay system prompt differs from the active prompt")
    if (
        set(messages[1]) != {"role", "content"}
        or messages[1].get("role") != "user"
        or not isinstance(messages[1].get("content"), str)
        or not messages[1]["content"].strip()
    ):
        raise ValueError("replay issue must be a user message")
    if problem is not None and messages[1] != {
        "role": "user",
        "content": protocol.INSTANCE_PROMPT.format(problem_statement=problem),
    }:
        raise ValueError("replay belongs to a different issue")
    calls, turns = [], 0
    for index, message in enumerate(messages[2:], start=2):
        role = message.get("role")
        fields = {
            "assistant": {"role", "content", "reasoning_content", "tool_calls"},
            "tool": {"role", "content", "tool_call_id"},
            "user": {"role", "content"},
        }.get(role)
        if fields is None or set(message) - fields:
            raise ValueError(f"unsupported prefix message fields/role at {index}")
        if role == "assistant":
            if calls:
                raise ValueError(f"unanswered tool calls before message {index}")
            turns += 1
            if not isinstance(message.get("content"), str):
                raise ValueError(f"invalid assistant content at {index}")
            if message.get("reasoning_content") is not None and not isinstance(message["reasoning_content"], str):
                raise ValueError(f"invalid reasoning at {index}")
            raw = message.get("tool_calls")
            if raw is None:
                raw = []
            if not isinstance(raw, list):
                raise ValueError(f"invalid tool calls at {index}")
            for call_index, call in enumerate(raw):
                protocol.parse_action([call])
                required = {"id", "type", "function"}
                if (
                    not required <= set(call)
                    or set(call) - required - {"index"}
                    or set(call["function"]) != {"name", "arguments"}
                ):
                    raise ValueError(f"unsupported tool call fields at {index}")
                if "index" in call and (type(call["index"]) is not int or call["index"] != call_index):
                    raise ValueError(f"invalid native tool call index at message {index}")
                cid = call.get("id")
                if not isinstance(cid, str) or not cid.strip() or cid in calls:
                    raise ValueError(f"invalid/duplicate tool ID at {index}")
                calls.append(cid)
        elif role == "tool":
            if not calls or message.get("tool_call_id") != calls.pop(0):
                raise ValueError(f"unmatched/out-of-order tool result at {index}")
            if not isinstance(message.get("content"), str):
                raise ValueError(f"invalid observation at {index}")
        elif role == "user":
            if calls or index == 2 or messages[index - 1].get("role") != "assistant":
                raise ValueError(f"unexpected user feedback at {index}")
            if messages[index - 1].get("tool_calls"):
                raise ValueError(f"unexpected tool-call feedback at {index}")
            if message.get("content") not in [protocol.format_error_message(0, f) for f in ("stop", "tool_calls")]:
                raise ValueError(f"unknown format feedback at {index}")
        else:
            raise ValueError(f"unsupported prefix role {role!r} at {index}")
    if calls or turns != n_calls or (n_calls and messages[-1].get("role") not in {"tool", "user"}):
        raise ValueError("incomplete prefix or inconsistent assistant-turn count")


def fingerprint(root):
    """Hash portable filesystem state, including owners and hardlink relationships."""
    root = Path(root)
    if not root.is_dir() or root.is_symlink():
        raise ValueError(f"fingerprint requires a real directory: {root}")
    root_before = root.stat()
    records = [[".", root_before.st_mode, root_before.st_uid, root_before.st_gid, "", ""]]
    snapshots = [(root, root_before)]
    hardlinks = {}
    paths = sorted(root.rglob("*"))
    for path in paths:
        relative = path.relative_to(root).as_posix()
        if relative == ".git" or relative.startswith(".git/"):
            continue
        before = path.lstat()
        mode = before.st_mode
        if stat.S_ISLNK(mode):
            payload = os.readlink(path)
        elif stat.S_ISREG(mode):
            h = hashlib.sha256()
            with path.open("rb") as stream:
                opened = os.fstat(stream.fileno())
                if (opened.st_dev, opened.st_ino, opened.st_mode) != (before.st_dev, before.st_ino, mode):
                    raise RuntimeError(f"file changed before replay capture read: {path}")
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    h.update(chunk)
            payload = h.hexdigest()
        elif stat.S_ISDIR(mode):
            payload = ""
        else:
            raise ValueError(f"unsupported special file during replay capture: {path}")
        after = path.lstat()
        if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ):
            raise RuntimeError(f"file changed during replay capture: {path}")
        # Inodes differ between containers. Store the first path sharing an
        # inode, rather than its machine-local number, to compare link topology.
        link = ""
        if stat.S_ISREG(mode):
            link = hardlinks.setdefault((before.st_dev, before.st_ino), relative)
        records.append([relative, mode, before.st_uid, before.st_gid, link, payload])
        snapshots.append((path, after))
    if sorted(root.rglob("*")) != paths:
        raise RuntimeError(f"directory entries changed during replay capture: {root}")
    # Also catch changes to already-read entries while later files were read.
    for path, before in snapshots:
        after = path.lstat()
        for field in (
            "st_dev",
            "st_ino",
            "st_mode",
            "st_uid",
            "st_gid",
            "st_nlink",
            "st_size",
            "st_mtime_ns",
            "st_ctime_ns",
        ):
            if getattr(before, field) != getattr(after, field):
                raise RuntimeError(f"file changed during replay capture: {path}, field={field}")
    return hashlib.sha256(json.dumps(records, ensure_ascii=True, separators=(",", ":")).encode()).hexdigest()


def build_replay(trajectory, selected_turn):
    if not isinstance(trajectory, dict) or trajectory.get("replay_schema") not in (SCHEMA, LEGACY_SCHEMA):
        raise ValueError("trajectory lacks native replay telemetry")
    messages = trajectory.get("messages")
    if not isinstance(messages, list) or any(not isinstance(m, dict) for m in messages):
        raise ValueError("invalid trajectory messages")
    positions = [i for i, m in enumerate(messages) if m.get("role") == "assistant"]
    if type(selected_turn) is not int or not 1 <= selected_turn < len(positions):
        raise ValueError("selected turn must be a non-final assistant turn")
    prefix = copy.deepcopy(messages[: positions[selected_turn - 1]])
    source_max_turns = trajectory.get("source_max_turns", 100)
    if type(source_max_turns) is not int or source_max_turns not in (100, 200, 250):
        raise ValueError("replay source budget must be 100, 200 or 250 turns")
    validate_prefix(prefix, selected_turn - 1, max_turns=source_max_turns)
    ledger = trajectory.get("command_results")
    if not isinstance(ledger, list):
        raise ValueError("missing command ledger")
    telemetry = {}
    if trajectory["replay_schema"] == LEGACY_SCHEMA:
        boundaries = trajectory.get("boundaries")
        if not isinstance(boundaries, dict):
            raise ValueError("missing workspace boundaries")
        telemetry = {"initial": boundaries.get("1"), "expected": boundaries.get(str(selected_turn))}
    elif "boundaries" in trajectory:
        raise ValueError("v3 source must not contain filesystem boundaries")
    actions = []
    for item in ledger:
        if not isinstance(item, dict) or type(item.get("turn")) is not int or item["turn"] < 1:
            raise ValueError("invalid command ledger entry")
        if item["turn"] < selected_turn:
            actions.append(copy.deepcopy(item))
    state = {
        "schema": trajectory["replay_schema"],
        "n_calls": selected_turn - 1,
        "messages": prefix,
        "actions": actions,
        **telemetry,
    }
    if source_max_turns != 100:
        state["source_max_turns"] = source_max_turns
    validate_replay(state)
    return copy.deepcopy(state)


def validate_replay(state):
    if not isinstance(state, dict) or state.get("schema") not in (SCHEMA, LEGACY_SCHEMA):
        raise ValueError("unknown replay schema")
    fields = {"schema", "n_calls", "messages", "actions"}
    if "source_max_turns" in state:
        fields.add("source_max_turns")
    source_max_turns = state.get("source_max_turns", 100)
    if type(source_max_turns) is not int or source_max_turns not in (100, 200, 250):
        raise ValueError("replay source budget must be 100, 200 or 250 turns")
    if state["schema"] == LEGACY_SCHEMA:
        fields |= {"initial", "expected"}
    if set(state) != fields:
        raise ValueError("invalid replay fields; refusing opaque metadata")
    validate_prefix(state["messages"], state["n_calls"], max_turns=source_max_turns)
    if not isinstance(state["actions"], list):
        raise ValueError("replay actions must be a list")
    if state["schema"] == LEGACY_SCHEMA:
        for key in ("initial", "expected"):
            if not isinstance(state[key], dict) or set(state[key]) != {"/testbed", "/tmp"}:
                raise ValueError("legacy replay must fingerprint /testbed and /tmp")
            for value in state[key].values():
                _digest(value)
    expected_actions, turn = [], 0
    for message in state["messages"]:
        if message.get("role") == "assistant":
            turn += 1
            for call in message.get("tool_calls") or []:
                command = protocol.parse_action([call])
                if reference._forbidden_action(command) is None:
                    expected_actions.append((turn, call["id"], command))
    actual = []
    for item in state["actions"]:
        if not isinstance(item, dict) or set(item) != {
            "turn",
            "tool_call_id",
            "command",
            "returncode",
            "output_sha256",
        }:
            raise ValueError("invalid replay action fields")
        if type(item["turn"]) is not int or type(item["returncode"]) is not int:
            raise ValueError("invalid replay action counters")
        _digest(item["output_sha256"])
        actual.append((item["turn"], item["tool_call_id"], item["command"]))
    if actual != expected_actions:
        raise ValueError("command ledger does not match native tool-call prefix")


async def restore_replay(state, execute):
    validate_replay(state)
    if not callable(execute):
        raise TypeError("replay requires an execute callable")
    for item in state["actions"]:
        output, code = await execute(item["command"], 120)
        if not isinstance(output, str) or type(code) is not int:
            raise TypeError(f"invalid replay shell result at turn {item['turn']}")
        if code != item["returncode"]:
            raise RuntimeError(
                f"replay returncode diverged at source turn {item['turn']}, tool {item['tool_call_id']}: "
                f"expected={item['returncode']}, actual={code}"
            )
    return copy.deepcopy(state["messages"])
