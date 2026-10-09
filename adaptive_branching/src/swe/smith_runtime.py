"""Dependency-free SWE-smith task bootstrap, reference extraction and verifier.

Copied into task images; never imports Miles, a model, or network clients.
Original images are shared. Task checkout and Git sanitization happen on the
container's writable layer before Docker's healthcheck admits the agent.
"""

import argparse
import base64
import hashlib
import json
import os
import re
import signal
import shutil
import subprocess
import sys
from pathlib import Path, PurePosixPath

ROOT = Path("/testbed")
ASSETS = Path("/opt/miles-swesmith")
MAX_PATCH_BYTES = 16 * 1024 * 1024


def run(argv, *, root=ROOT, timeout=120):
    if not argv or any(not isinstance(s, str) or not s for s in argv):
        raise ValueError("command must contain nonempty strings")
    if not Path(root).is_dir() or type(timeout) is not int or timeout <= 0:
        raise ValueError("command needs an existing cwd and positive timeout")
    return subprocess.run(argv, cwd=root, check=True, capture_output=True, timeout=timeout).stdout


def safe_path(value):
    if not isinstance(value, str) or not value or "\x00" in value or "\\" in value:
        raise ValueError(f"invalid relative path: {value!r}")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or ".git" in path.parts or str(path) != value:
        raise ValueError(f"unsafe relative path: {value!r}")
    return path


def validate_row(row):
    if not isinstance(row, dict):
        raise TypeError("SWE-smith row must be an object")
    for key in ("instance_id", "image_name", "repo", "problem_statement"):
        if not isinstance(row.get(key), str) or not row[key].strip():
            raise ValueError(f"missing {key}")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", row["instance_id"]):
        raise ValueError("unsafe instance_id")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_./:@-]*", row["image_name"]):
        raise ValueError("unsafe image_name")
    for key in ("FAIL_TO_PASS", "PASS_TO_PASS"):
        nodes = row.get(key)
        if not isinstance(nodes, list) or (key == "FAIL_TO_PASS" and not nodes):
            raise ValueError(f"invalid {key}")
        for node in nodes:
            if not isinstance(node, str) or not node or "\n" in node or "\x00" in node:
                raise ValueError(f"invalid test node: {node!r}")
            safe_path(node.split("::", 1)[0])
        if len(set(nodes)) != len(nodes):
            raise ValueError(f"duplicate tests in {key}")
    if set(row["FAIL_TO_PASS"]) & set(row["PASS_TO_PASS"]):
        raise ValueError("F2P and P2P overlap")
    return row


def selected_tests(row, policy):
    validate_row(row)
    if policy not in ("f2p-files", "all"):
        raise ValueError(f"unknown test policy: {policy!r}")
    f2p = row["FAIL_TO_PASS"]
    files = {n.split("::", 1)[0] for n in f2p}
    p2p = [n for n in row["PASS_TO_PASS"] if policy == "all" or n.split("::", 1)[0] in files]
    return list(f2p), p2p


def extract_reference(row, *, root=ROOT):
    """Read pinned branch objects; reverse precisely the bug commit, not tests."""
    validate_row(row)
    refs = run(["git", "for-each-ref", "--format=%(refname)", "refs/heads", "refs/remotes"], root=root)
    candidates = [f"refs/heads/{row['instance_id']}", f"refs/remotes/origin/{row['instance_id']}"]
    matches = [r for r in candidates if r in refs.decode().splitlines()]
    if not matches:
        raise ValueError(f"missing problem branch: {row['instance_id']}")
    heads = {run(["git", "rev-parse", ref], root=root).decode().strip() for ref in matches}
    if len(heads) != 1:
        raise ValueError(f"local/remote branch mismatch: {row['instance_id']}")
    head = heads.pop()
    subject = run(["git", "show", "-s", "--format=%s", head], root=root).decode().strip()
    if subject != "Remove F2P Tests":
        raise ValueError(f"unsupported branch layout {row['instance_id']}: HEAD={subject!r}")
    bug = run(["git", "rev-parse", head + "^"], root=root).decode().strip()
    if run(["git", "show", "-s", "--format=%s", bug], root=root).decode().strip() != "Bug Patch":
        raise ValueError(f"missing Bug Patch parent: {row['instance_id']}")
    for commit in (head, bug):
        if len(run(["git", "rev-list", "--parents", "-n", "1", commit], root=root).split()) != 2:
            raise ValueError(f"nonlinear task history: {row['instance_id']}")
    base = run(["git", "rev-parse", bug + "^"], root=root).decode().strip()
    removed = run(["git", "diff", "--name-status", "--no-renames", bug, head], root=root).decode().splitlines()
    if not removed or any(not line.startswith("D\t") for line in removed):
        raise ValueError(f"test-removal commit has non-deletions: {row['instance_id']}")
    deleted = [line[2:] for line in removed]
    for name in deleted:
        safe_path(name)
    f2p_files = {n.split("::", 1)[0] for n in row["FAIL_TO_PASS"]}
    if not f2p_files.issubset(deleted):
        raise ValueError(f"F2P files not hidden at task HEAD: {row['instance_id']}")
    files = run(["git", "diff", "--name-only", "--no-renames", bug, base], root=root).decode().splitlines()
    for name in files:
        safe_path(name)
    if not files or set(files) & set(deleted):
        raise ValueError(f"empty/source-test-overlapping golden patch: {row['instance_id']}")
    patch = run(["git", "diff", "--binary", "--no-ext-diff", "--no-renames", bug, base], root=root)
    if not patch or len(patch) > MAX_PATCH_BYTES:
        raise ValueError(f"invalid golden patch size: {row['instance_id']}")
    return {
        "head": head,
        "bug": bug,
        "base": base,
        "head_tree": run(["git", "rev-parse", head + "^{tree}"], root=root).decode().strip(),
        "golden_patch": patch.decode("utf-8"),
        "golden_files": files,
        "deleted_test_files": deleted,
        "golden_sha256": hashlib.sha256(patch).hexdigest(),
        "python": sys.executable,
    }


def bootstrap(config, *, root=ROOT, assets=ASSETS):
    """Initialize the same source baseline for agent and separate verifier."""
    row = validate_row(config["row"])
    if config.get("role") not in ("agent", "verifier") or (assets / "READY").exists():
        raise ValueError("invalid role or attempted second bootstrap")
    pins = config["pins"]
    for key in ("head", "bug", "head_tree"):
        if not re.fullmatch(r"[a-f0-9]{40,64}", pins.get(key, "")):
            raise ValueError(f"invalid pinned {key}")
    if not (root / ".git").is_dir():
        raise FileNotFoundError(root / ".git")
    run(["git", "checkout", "--detach", pins["head"]], root=root)
    run(["git", "reset", "--hard", pins["head"]], root=root)
    if run(["git", "rev-parse", "HEAD^{tree}"], root=root).decode().strip() != pins["head_tree"]:
        raise ValueError("image tree differs from prepared manifest")
    f2p, p2p = selected_tests(row, config["test_policy"])
    if config["role"] == "verifier":
        for name in sorted({node.split("::", 1)[0] for node in f2p + p2p}):
            safe_path(name)
            target = assets / "pristine" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(run(["git", "show", f"{pins['bug']}:{name}"], root=root))
    tracked = run(["git", "ls-files", "-z"], root=root)
    if not tracked:
        raise ValueError("task baseline has no tracked files")
    # No reference history remains reachable by the model. Only the current
    # bug worktree is committed; original lower image layers stay shared.
    shutil.rmtree(root / ".git")
    for directory, dirs, _ in os.walk(root):
        if "__pycache__" in dirs:
            shutil.rmtree(Path(directory) / "__pycache__")
            dirs.remove("__pycache__")
    run(["git", "init", "--quiet"], root=root)
    run(["git", "config", "user.name", "SWE-smith baseline"], root=root)
    run(["git", "config", "user.email", "baseline@invalid"], root=root)
    subprocess.run(
        ["git", "add", "-f", "--pathspec-from-file=-", "--pathspec-file-nul"],
        input=tracked,
        cwd=root,
        check=True,
        capture_output=True,
        timeout=120,
    )
    run(["git", "commit", "--quiet", "-m", "Synthetic baseline"], root=root)
    if run(["git", "rev-parse", "HEAD^{tree}"], root=root).decode().strip() != pins["head_tree"]:
        raise ValueError("synthetic baseline changed the pinned source tree")
    if run(["git", "rev-list", "--all", "--count"], root=root).strip() != b"1":
        raise ValueError("history sanitization failed")
    if run(["git", "remote"], root=root).strip():
        raise ValueError("unexpected Git remote after sanitization")
    (assets / "READY").write_text(json.dumps({"instance_id": row["instance_id"], "head": pins["head"]}))


def grade(log, f2p, p2p, returncode, *, timed_out=False):
    if not isinstance(log, str) or not f2p or not isinstance(p2p, list):
        raise ValueError("grader needs log, nonempty F2P and a P2P list")
    if type(returncode) is not int or type(timed_out) is not bool:
        raise TypeError("invalid test process status")
    nodes = f2p + p2p
    if any(not isinstance(n, str) or not n for n in nodes) or len(set(nodes)) != len(nodes):
        raise ValueError("invalid or duplicate graded nodes")
    statuses = {}
    log = re.sub(r"\x1b\[[0-9;]*m", "", log)
    for line in log.splitlines():
        match = re.match(r"^(PASSED|FAILED|ERROR|SKIPPED|XFAIL|XPASS) (.+?)(?: - .*|$)", line)
        if match:
            status, node = match.groups()
            if node in statuses and statuses[node] != status:
                raise ValueError(f"conflicting test statuses: {node}")
            statuses[node] = status
    accepted = {"PASSED", "XFAIL"}  # Match the pinned Lightning evaluator.
    reward = float(not timed_out and returncode in (0, 1) and all(statuses.get(n) in accepted for n in nodes))
    return {
        "reward": reward,
        "actual": statuses,
        "expected": {n: "PASSED_OR_XFAIL" for n in nodes},
        "missing": [n for n in nodes if n not in statuses],
        "test_returncode": returncode,
        "timed_out": timed_out,
        "f2p_pass": sum(statuses.get(n) in accepted for n in f2p),
        "p2p_pass": sum(statuses.get(n) in accepted for n in p2p),
    }


def verify(config, *, root=ROOT, assets=ASSETS, patch=Path("/tmp/r2e-submission.patch")):
    if config.get("role") != "verifier" or not (assets / "READY").is_file():
        raise ValueError("verifier baseline is not ready")
    f2p, p2p = selected_tests(config["row"], config["test_policy"])
    if not patch.is_file() or patch.stat().st_size > MAX_PATCH_BYTES:
        raise ValueError("submission patch missing or oversized")
    if patch.stat().st_size:
        # A valid source edit can target any source file, not only golden_files.
        run(["git", "apply", "--check", "--binary", str(patch)], root=root)
        run(["git", "apply", "--binary", str(patch)], root=root)
    for name in sorted({n.split("::", 1)[0] for n in f2p + p2p}):
        destination = root / name
        if destination.is_symlink() or root.resolve() not in destination.resolve().parents:
            raise ValueError(f"test path escaped worktree: {name}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(assets / "pristine" / name, destination)
    probe = subprocess.run(
        [sys.executable, "-c", "import importlib.util; exit(importlib.util.find_spec('xdist') is None)"],
        capture_output=True,
        timeout=30,
    )
    xdist = ["-n4"] if probe.returncode == 0 else ["-p", "no:xdist"]
    timeout = config["test_timeout"]
    if type(timeout) is not int or timeout <= 0:
        raise ValueError("test_timeout must be positive")
    command = [sys.executable, "-m", "pytest", "-rA", "-p", "no:cacheprovider", *xdist, *f2p, *p2p]
    process = subprocess.Popen(
        command, cwd=root, stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True
    )
    timed_out = False
    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        os.killpg(process.pid, signal.SIGKILL)
        stdout, stderr = process.communicate()
    output = (stdout + stderr).decode("utf-8", errors="replace")
    report = grade(output, f2p, p2p, 124 if timed_out else process.returncode, timed_out=timed_out)
    return report, output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("extract", "bootstrap", "verify"))
    args = parser.parse_args()
    if args.action == "extract":
        rows = json.load(sys.stdin)
        if not isinstance(rows, list) or not rows:
            raise ValueError("extract requires a nonempty row list")
        for row in rows:
            print(json.dumps({"instance_id": row["instance_id"], **extract_reference(row)}), flush=True)
        return
    encoded = os.environ.get("MILES_SMITH_CONFIG", "")
    if not encoded or len(encoded) > 96000:
        raise ValueError("missing or oversized task configuration")
    config = json.loads(base64.b64decode(encoded, validate=True))
    ASSETS.mkdir(parents=True, exist_ok=True)
    if args.action == "bootstrap":
        bootstrap(config)
        os.execl("/bin/sleep", "sleep", "infinity")
    logs = Path("/logs/verifier")
    logs.mkdir(parents=True, exist_ok=True)
    try:
        report, output = verify(config)
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        # An explicit marker excludes harness errors from training; never turn
        # missing artifacts or broken infrastructure into ordinary negatives.
        report = {"reward": 0.0, "infrastructure_error": f"{type(exc).__name__}: {exc}"}
        output = repr(exc)
    (logs / "test-output.log").write_text(output)
    (logs / "report.json").write_text(json.dumps(report, indent=2))
    (logs / "reward.txt").write_text(str(report["reward"]))
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
