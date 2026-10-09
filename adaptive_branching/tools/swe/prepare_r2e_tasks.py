#!/usr/bin/env python3
"""Convert R2E-Gym rows into Harbor tasks and a Miles prompt JSONL."""

import argparse
import json
import re
import shutil
import subprocess
import sys
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from textwrap import dedent
from typing import Any

from adaptive_branching.src.swe import r2e_grader
from adaptive_branching.tools.swe.build_value_cliff_locator_inputs import extract_golden_source_patch
from adaptive_branching.tools.swe.task_common import public_network_compose

_SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_SAFE_IMAGE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/@-]*$")
_ISSUE = re.compile(r"\[ISSUE\](.*?)\[/ISSUE\]", re.DOTALL)

_MAX_PATCH_BYTES = 16 * 1024 * 1024

_COLLECT_PATCH_SH = """set -euo pipefail
artifact_dir=/tmp
patch_path=${artifact_dir}/r2e-submission.patch
rm -f "${patch_path}"
cd /testbed
test -d .git
test "$(git rev-list --max-parents=0 --all --count)" -eq 1
baseline="$(git rev-list --max-parents=0 --all)"
git add -A
temporary_patch="$(mktemp "${artifact_dir}/.r2e-submission.patch.XXXXXX")"
trap 'rm -f "${temporary_patch}"' EXIT
git diff --cached --binary "${baseline}" -- . > "${temporary_patch}"
test "$(wc -c < "${temporary_patch}")" -le __MAX_PATCH_BYTES__
chmod 0444 "${temporary_patch}"
mv -f "${temporary_patch}" "${patch_path}"
trap - EXIT
""".replace("__MAX_PATCH_BYTES__", str(_MAX_PATCH_BYTES))

_TEST_SH = """#!/usr/bin/env bash
set -uo pipefail

mkdir -p /logs/verifier
test_log=/logs/verifier/test-output.log
hidden_tests=/tests/r2e_tests
run_tests=/tests/run_tests.sh
submission_patch=/tmp/r2e-submission.patch
max_patch_bytes=__MAX_PATCH_BYTES__
grader_python=/usr/bin/python3

if [[ ! -x "${grader_python}" ]]; then
  echo "R2E grader Python is not executable: ${grader_python}" >&2
  exit 2
fi

record_infra_failure() {
  local reason="$1"
  printf 'R2E verifier rejected submission: %s\n' "${reason}" >&2
  "${grader_python}" /tests/r2e_grader.py \
    --infra-error "${reason}" \
    --expected /tests/expected.json \
    --reward /logs/verifier/reward.txt \
    --report /logs/verifier/report.json
}

if [[ ! -d "${hidden_tests}" ]]; then
  echo "Verifier upload is missing ${hidden_tests}" >&2
  exit 2
fi
if [[ ! -s "${run_tests}" ]]; then
  echo "Verifier upload is missing ${run_tests}" >&2
  exit 2
fi
if [[ ! -f "${submission_patch}" ]]; then
  record_infra_failure "submission patch is missing"
  exit 0
fi
patch_bytes="$(wc -c < "${submission_patch}")"
if (( patch_bytes > max_patch_bytes )); then
  record_infra_failure "submission patch exceeds ${max_patch_bytes} bytes"
  exit 0
fi
if (( patch_bytes > 0 )); then
  if ! (cd /testbed && git apply --check --binary "${submission_patch}"); then
    record_infra_failure "submission patch does not apply to the clean baseline"
    exit 0
  fi
  (cd /testbed && git apply --binary "${submission_patch}")
fi
rm -rf /testbed/r2e_tests
ln -s "${hidden_tests}" /testbed/r2e_tests
(cd /testbed && bash "${run_tests}") >"${test_log}" 2>&1
test_exit=$?
cat "${test_log}"
"${grader_python}" /tests/r2e_grader.py \
  --log "${test_log}" \
  --expected /tests/expected.json \
  --reward /logs/verifier/reward.txt \
  --report /logs/verifier/report.json
grader_exit=$?
printf 'R2E test command exit code: %s\n' "${test_exit}"
printf 'R2E grader exit code: %s\n' "${grader_exit}"
exit 0
""".replace("__MAX_PATCH_BYTES__", str(_MAX_PATCH_BYTES))


def _required_string(row: dict[str, Any], key: str) -> str:
    value = row.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} must be a non-empty string")
    return value.strip()


def _expected_statuses(value: Any) -> dict[str, str]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError("expected_output_json is not valid JSON") from exc
    if not isinstance(value, dict) or not value:
        raise ValueError("expected_output_json must contain a non-empty object")
    if any(not isinstance(key, str) or not isinstance(status, str) for key, status in value.items()):
        raise TypeError("expected_output_json keys and values must be strings")
    return value


@dataclass(frozen=True)
class R2ETask:
    task_id: str
    repo_name: str
    commit_hash: str
    docker_image: str
    instruction: str
    expected_statuses: dict[str, str]
    golden_patch: str

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> "R2ETask":
        if not isinstance(row, dict):
            raise TypeError("each R2E row must be a dict")
        repo_name = _required_string(row, "repo_name")
        commit_hash = _required_string(row, "commit_hash")
        docker_image = _required_string(row, "docker_image")
        problem_statement = _required_string(row, "problem_statement")
        if not _SAFE_NAME.fullmatch(repo_name):
            raise ValueError(f"unsafe repo_name: {repo_name!r}")
        if not re.fullmatch(r"[0-9a-fA-F]{7,64}", commit_hash):
            raise ValueError(f"commit_hash must be a 7-64 character hexadecimal hash, got {commit_hash!r}")
        if not _SAFE_IMAGE.fullmatch(docker_image):
            raise ValueError(f"unsafe docker_image: {docker_image!r}")
        issue = _ISSUE.search(problem_statement)
        instruction = dedent(issue.group(1) if issue else problem_statement).strip()
        if not instruction:
            raise ValueError("problem_statement produced an empty instruction")
        return cls(
            task_id=f"r2e-{repo_name}-{commit_hash.lower()}",
            repo_name=repo_name,
            commit_hash=commit_hash.lower(),
            docker_image=docker_image,
            instruction=instruction,
            expected_statuses=_expected_statuses(row.get("expected_output_json")),
            golden_patch=extract_golden_source_patch(
                _required_string(row, "parsed_commit_content"),
                row.get("relevant_files"),
            ),
        )


VerifierAssetsExtractor = Callable[[R2ETask, Path], None]
_CONTAINER_REMOVE_ATTEMPTS = 5


def _run_docker(args: list[str], *, docker_binary: str) -> str:
    if not isinstance(docker_binary, str) or not docker_binary.strip():
        raise ValueError("docker_binary must be a non-empty string")
    if not args or any(not isinstance(arg, str) or not arg for arg in args):
        raise ValueError(f"docker arguments must be non-empty strings, got {args!r}")
    completed = subprocess.run(
        [docker_binary, *args],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "no output").strip()
        raise RuntimeError(f"{docker_binary} {' '.join(args)} failed with code {completed.returncode}: {detail}")
    return completed.stdout


def _remove_temporary_container(container_id: str, *, docker_binary: str) -> None:
    for attempt in range(_CONTAINER_REMOVE_ATTEMPTS):
        try:
            _run_docker(["rm", "-f", container_id], docker_binary=docker_binary)
            return
        except RuntimeError as exc:
            if "dataset is busy" not in str(exc) or attempt + 1 == _CONTAINER_REMOVE_ATTEMPTS:
                raise
            print(
                f"temporary container {container_id} is still busy; retrying removal after {2**attempt}s",
                file=sys.stderr,
            )
            time.sleep(2**attempt)


def extract_verifier_assets(task: R2ETask, destination: Path, *, docker_binary: str = "docker") -> None:
    """Copy pristine hidden tests and their runner without starting the image."""
    if not isinstance(task, R2ETask):
        raise TypeError("task must be an R2ETask")
    if not isinstance(destination, Path):
        raise TypeError("destination must be a pathlib.Path")
    if destination.exists():
        raise FileExistsError(f"verifier-assets destination already exists: {destination}")
    _run_docker(["image", "inspect", task.docker_image], docker_binary=docker_binary)
    destination.mkdir(parents=True)
    hidden_tests = destination / "r2e_tests"
    hidden_tests.mkdir()
    run_tests = destination / "run_tests.sh"
    container_id = _run_docker(["create", task.docker_image], docker_binary=docker_binary).strip()
    if not container_id:
        raise RuntimeError(f"docker create returned an empty container id for {task.docker_image}")
    try:
        _run_docker(["cp", f"{container_id}:/r2e_tests/.", str(hidden_tests)], docker_binary=docker_binary)
        _run_docker(["cp", f"{container_id}:/testbed/run_tests.sh", str(run_tests)], docker_binary=docker_binary)
    finally:
        _remove_temporary_container(container_id, docker_binary=docker_binary)
    if not any(path.is_file() for path in hidden_tests.rglob("*")):
        raise ValueError(f"R2E image contains no hidden test files: {task.docker_image}")
    if not run_tests.is_file() or run_tests.stat().st_size == 0:
        raise ValueError(f"R2E image contains no non-empty /testbed/run_tests.sh: {task.docker_image}")


def _cleanroom_dockerfile(task: R2ETask, *, verifier: bool) -> str:
    if not isinstance(task, R2ETask):
        raise TypeError("task must be an R2ETask")
    if not isinstance(verifier, bool):
        raise TypeError("verifier must be a bool")
    dockerfile = f"""FROM {task.docker_image}
USER root
RUN --network=none set -eux; \\
    cd /testbed; \\
    test -e .git; \\
    rm -rf /r2e_tests /testbed/r2e_tests /testbed/run_tests.sh; \\
    git add -A; \\
    git ls-files -z > /tmp/r2e-baseline-files; \\
    rm -rf .git; \\
    git init --quiet; \\
    git config user.name 'R2E Synthetic Baseline'; \\
    git config user.email 'r2e-baseline@invalid'; \\
    xargs -0 -r git add -f -- < /tmp/r2e-baseline-files; \\
    rm -f /tmp/r2e-baseline-files; \\
    test "$(git ls-files | wc -l)" -gt 0; \\
    git commit --quiet -m 'Synthetic baseline'; \\
    git reflog expire --expire=now --all; \\
    git gc --quiet --prune=now; \\
    rm -rf .git/logs; \\
    test "$(git rev-list --all --count)" -eq 1; \\
    test "$(git rev-list --max-parents=0 --all --count)" -eq 1; \\
    test -z "$(git remote)"; \\
    test -z "$(git status --porcelain)"; \\
    test ! -e /r2e_tests; \\
    test ! -e /testbed/r2e_tests; \\
    test ! -e /testbed/run_tests.sh
WORKDIR /testbed
"""
    if verifier:
        dockerfile += """COPY . /tests/
RUN --network=none set -eux; \\
    test -d /tests/r2e_tests; \\
    test -s /tests/run_tests.sh; \\
    chmod 0555 /tests/test.sh /tests/run_tests.sh; \\
    rm -f /tests/Dockerfile /tests/.dockerignore
"""
    return dockerfile


def _task_toml(task: R2ETask, *, timeout_seconds: int, cpus: int, memory_mb: int) -> str:
    for name, value in (("timeout_seconds", timeout_seconds), ("cpus", cpus), ("memory_mb", memory_mb)):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return f"""schema_version = "1.3"

artifacts = [
  {{ source = "/logs/artifacts", exclude = ["*"] }},
  "/tmp/r2e-submission.patch",
]

[task]
name = "r2e-gym/{task.task_id}"
description = "R2E-Gym executable repository repair task."

[metadata]
repo_name = "{task.repo_name}"
commit_hash = "{task.commit_hash}"

[agent]
timeout_sec = {timeout_seconds}
network_mode = "public"

[verifier]
timeout_sec = {timeout_seconds}
network_mode = "public"
environment_mode = "separate"
user = "root"

[[verifier.collect]]
service = "main"
timeout_sec = 300
user = "root"
command = '''
{_COLLECT_PATCH_SH}'''

[verifier.environment]
network_mode = "public"
build_timeout_sec = 1800
cpus = {cpus}
memory_mb = {memory_mb}
storage_mb = 10240
gpus = 0
workdir = "/testbed"
env = {{ PATH = "/testbed/.venv/bin:/root/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin", VIRTUAL_ENV = "/testbed/.venv" }}

[environment]
build_timeout_sec = 1800
cpus = {cpus}
memory_mb = {memory_mb}
storage_mb = 10240
gpus = 0
workdir = "/testbed"
env = {{ PATH = "/testbed/.venv/bin:/root/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin", VIRTUAL_ENV = "/testbed/.venv", UV_DEFAULT_INDEX = "https://pypi.tuna.tsinghua.edu.cn/simple", PIP_INDEX_URL = "https://pypi.tuna.tsinghua.edu.cn/simple" }}
"""


def _write_task(
    task: R2ETask,
    output_dir: Path,
    *,
    timeout_seconds: int,
    cpus: int,
    memory_mb: int,
    verifier_assets_extractor: VerifierAssetsExtractor,
) -> None:
    task_dir = output_dir / task.task_id
    environment_dir = task_dir / "environment"
    environment_dir.mkdir(parents=True)
    tests_dir = task_dir / "tests"
    verifier_assets_extractor(task, tests_dir)
    (task_dir / "instruction.md").write_text(task.instruction + "\n", encoding="utf-8")
    (task_dir / "task.toml").write_text(
        _task_toml(task, timeout_seconds=timeout_seconds, cpus=cpus, memory_mb=memory_mb),
        encoding="utf-8",
    )
    (tests_dir / "expected.json").write_text(
        json.dumps(task.expected_statuses, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    grader_file = r2e_grader.__file__
    if not grader_file or not Path(grader_file).is_file():
        raise FileNotFoundError("cannot locate adaptive_branching.src.swe.r2e_grader")
    shutil.copyfile(Path(grader_file).resolve(), tests_dir / "r2e_grader.py")
    test_script = tests_dir / "test.sh"
    test_script.write_text(_TEST_SH, encoding="utf-8")
    test_script.chmod(0o755)
    (environment_dir / "Dockerfile").write_text(_cleanroom_dockerfile(task, verifier=False), encoding="utf-8")
    (tests_dir / "Dockerfile").write_text(_cleanroom_dockerfile(task, verifier=True), encoding="utf-8")
    (tests_dir / ".dockerignore").write_text("Dockerfile\ndocker-compose.yaml\n", encoding="utf-8")
    (environment_dir / "docker-compose.yaml").write_text(public_network_compose(hide_r2e_tests=True), encoding="utf-8")
    (tests_dir / "docker-compose.yaml").write_text(public_network_compose(), encoding="utf-8")


def prepare_tasks(
    rows: Iterable[dict[str, Any]],
    *,
    output_dir: Path,
    prompts_path: Path,
    timeout_seconds: int = 1800,
    cpus: int = 2,
    memory_mb: int = 4096,
    overwrite: bool = False,
    verifier_assets_extractor: VerifierAssetsExtractor = extract_verifier_assets,
) -> int:
    tasks = [R2ETask.from_row(row) for row in rows]
    if not tasks:
        raise ValueError("R2E input contains no rows")
    _task_toml(tasks[0], timeout_seconds=timeout_seconds, cpus=cpus, memory_mb=memory_mb)
    task_ids = [task.task_id for task in tasks]
    if len(task_ids) != len(set(task_ids)):
        raise ValueError("R2E input contains duplicate task identifiers")

    existing = [output_dir / task_id for task_id in task_ids if (output_dir / task_id).exists()]
    if existing and not overwrite:
        raise FileExistsError(f"task directory already exists: {existing[0]}")
    if prompts_path.exists() and not overwrite:
        raise FileExistsError(f"prompt file already exists: {prompts_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    prompts_path.parent.mkdir(parents=True, exist_ok=True)
    for task in tasks:
        task_dir = output_dir / task.task_id
        if task_dir.exists():
            shutil.rmtree(task_dir)
        _write_task(
            task,
            output_dir,
            timeout_seconds=timeout_seconds,
            cpus=cpus,
            memory_mb=memory_mb,
            verifier_assets_extractor=verifier_assets_extractor,
        )

    with prompts_path.open("w", encoding="utf-8") as output:
        for task in tasks:
            sample = {
                "prompt": task.instruction,
                "metadata": {
                    "instance_id": task.task_id,
                    "agent_name": "mini-swe-agent",
                    "dataset": "R2E-Gym-Subset",
                    "repo_name": task.repo_name,
                    "commit_hash": task.commit_hash,
                    "docker_image": task.docker_image,
                    "ab_swe_golden_patch": task.golden_patch,
                },
            }
            output.write(json.dumps(sample, ensure_ascii=False) + "\n")
    return len(tasks)


def load_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.suffix == ".parquet":
        import pyarrow.parquet as parquet

        return parquet.read_table(path).to_pylist()
    if path.suffix == ".jsonl":
        rows = []
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}") from exc
            if not isinstance(row, dict):
                raise TypeError(f"{path}:{line_number} must contain an object")
            rows.append(row)
        return rows
    raise ValueError(f"unsupported input format {path.suffix!r}; use .parquet or .jsonl")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    parser.add_argument("--cpus", type=int, default=2)
    parser.add_argument("--memory-mb", type=int, default=4096)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive")
    rows = load_rows(args.input)
    if args.limit is not None:
        rows = rows[: args.limit]
    count = prepare_tasks(
        rows,
        output_dir=args.output_dir,
        prompts_path=args.prompts,
        timeout_seconds=args.timeout_seconds,
        cpus=args.cpus,
        memory_mb=args.memory_mb,
        overwrite=args.overwrite,
    )
    print(f"prepared {count} Harbor R2E tasks in {args.output_dir}")
    print(f"wrote Miles prompts to {args.prompts}")


if __name__ == "__main__":
    main()
