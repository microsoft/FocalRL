#!/usr/bin/env python3
"""Build Miles prompt JSONL from Harbor's pinned SWE-bench Verified or Lite tasks."""

import argparse
import json
import re
import tomllib
from pathlib import Path

_FROM_IMAGE = re.compile(r"^FROM\s+(\S+)", re.MULTILINE)
DATASETS = {"verified": "SWE-bench_Verified", "lite": "SWE-bench_Lite"}
LITE_IDENTITY = {"dataset": DATASETS["lite"], "dataset_repo": "SWE-bench/SWE-bench_Lite", "dataset_split": "test"}


def task_prompt(task_dir: Path, *, dataset: str = "verified") -> dict:
    if dataset not in DATASETS:
        raise ValueError(f"unknown SWE-bench dataset: {dataset!r}")
    if not isinstance(task_dir, Path) or not task_dir.is_dir():
        raise ValueError(f"task directory must exist: {task_dir}")
    instruction = (task_dir / "instruction.md").read_text().strip()
    if not instruction:
        raise ValueError(f"empty instruction: {task_dir}")
    config = tomllib.loads((task_dir / "task.toml").read_text())
    if config["verifier"].get("environment_mode") != "separate":
        raise ValueError(f"SWE-bench verifier must be separate: {task_dir}")
    dockerfile = (task_dir / "environment" / "Dockerfile").read_text()
    match = _FROM_IMAGE.search(dockerfile)
    if match is None:
        raise ValueError(f"missing Docker image: {task_dir}")
    record = json.loads((task_dir / "tests" / "config.json").read_text())
    if not isinstance(record, dict):
        raise ValueError(f"task record must be an object: {task_dir}")
    if dataset == "lite" and any(record.get(key) != value for key, value in LITE_IDENTITY.items()):
        raise ValueError(f"Lite task has wrong dataset/repository/test-split identity: {task_dir}")
    if dataset == "verified" and record.get("dataset", DATASETS[dataset]) != DATASETS[dataset]:
        raise ValueError(f"task dataset does not match requested Verified dataset: {task_dir}")
    instance_id = record["instance_id"]
    if task_dir.name != instance_id:
        raise ValueError(f"task directory {task_dir.name!r} does not match {instance_id!r}")
    golden_patch = record.get("patch")
    if not isinstance(golden_patch, str) or not golden_patch.strip():
        raise ValueError(f"missing golden patch: {task_dir}")
    result = {
        "prompt": instruction,
        "metadata": {
            "instance_id": instance_id,
            "agent_name": "mini-swe-agent",
            "dataset": DATASETS[dataset],
            "repo_name": record["repo"],
            "commit_hash": record["base_commit"],
            "docker_image": match.group(1),
            "ab_swe_golden_patch": golden_patch,
        },
    }
    if dataset == "lite":
        result["metadata"].update(LITE_IDENTITY)
        if "dataset_revision" in record:
            revision = record["dataset_revision"]
            if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision):
                raise ValueError(f"invalid dataset revision: {task_dir}")
            result["metadata"]["dataset_revision"] = revision
    return result


def prepare_prompts(tasks_dir: Path, output: Path, *, overwrite: bool = False, dataset: str = "verified") -> int:
    if dataset not in DATASETS:
        raise ValueError(f"unknown SWE-bench dataset: {dataset!r}")
    if not isinstance(tasks_dir, Path) or not tasks_dir.is_dir() or not isinstance(output, Path):
        raise ValueError("tasks directory must exist and output must be a Path")
    if output.exists() and not overwrite:
        raise FileExistsError(f"prompt file already exists: {output}")
    task_dirs = sorted(path for path in tasks_dir.iterdir() if path.is_dir())
    if not task_dirs:
        raise ValueError(f"no task directories found: {tasks_dir}")
    prompts = [task_prompt(path, dataset=dataset) for path in task_dirs]
    instance_ids = [item["metadata"]["instance_id"] for item in prompts]
    if len(instance_ids) != len(set(instance_ids)):
        raise ValueError("duplicate SWE-bench instance IDs")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for prompt in prompts:
            handle.write(json.dumps(prompt, ensure_ascii=False) + "\n")
    return len(prompts)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dataset", choices=DATASETS, default="verified")
    args = parser.parse_args()
    count = prepare_prompts(args.tasks_dir, args.output, overwrite=args.overwrite, dataset=args.dataset)
    print(f"WROTE_SWEBENCH_PROMPTS count={count} output={args.output}")


if __name__ == "__main__":
    main()
