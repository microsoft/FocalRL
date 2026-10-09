"""Pinned official Multilingual verifier; also copied into isolated task verifiers."""

import json
import re
import uuid
from pathlib import Path

HARNESS_VERSION = "5.0.2"
MAX_PATCH_BYTES = 16 * 1024 * 1024


def validate_instance(row):
    if not isinstance(row, dict):
        raise TypeError("Multilingual instance must be a dict")
    for key in (
        "instance_id",
        "repo",
        "base_commit",
        "version",
        "problem_statement",
        "patch",
        "test_patch",
        "image",
        "eval_script",
        "log_parser",
        "eval_type",
    ):
        if not isinstance(row.get(key), str) or not row[key].strip():
            raise ValueError(f"missing {key}: {row.get('instance_id')}")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", row["instance_id"]):
        raise ValueError("unsafe instance ID")
    if not re.fullmatch(r"[a-f0-9]{40}", row["base_commit"]):
        raise ValueError(f"invalid base commit: {row['instance_id']}")
    if not re.fullmatch(r"swebench/[a-z0-9_.-]+:latest", row["image"]):
        raise ValueError(f"unexpected official image: {row['image']}")
    for key in ("FAIL_TO_PASS", "PASS_TO_PASS"):
        values = row.get(key)
        if not isinstance(values, list) or any(not isinstance(x, str) or not x for x in values):
            raise ValueError(f"{row['instance_id']}: {key} must be a list of test names")
        if len(values) != len(set(values)):
            raise ValueError(f"{row['instance_id']}: duplicate {key} tests")
    if not row["FAIL_TO_PASS"] or set(row["FAIL_TO_PASS"]) & set(row["PASS_TO_PASS"]):
        raise ValueError(f"{row['instance_id']}: invalid positive/negative test sets")
    return row


def checked_report(result, instance_id):
    if not isinstance(instance_id, str) or not instance_id:
        raise ValueError("instance ID required")
    if not isinstance(result, tuple) or len(result) != 2 or result[0] != instance_id:
        raise RuntimeError(f"official verifier returned no valid result: {instance_id}: {result!r}")
    report = result[1]
    if not isinstance(report, dict) or set(report) != {instance_id}:
        raise ValueError(f"official report instance mismatch: {instance_id}")
    item = report[instance_id]
    if not isinstance(item, dict) or type(item.get("resolved")) is not bool:
        raise ValueError(f"missing official resolved flag: {instance_id}")
    return {
        "instance_id": instance_id,
        "reward": int(item["resolved"]),
        "official_harness_version": HARNESS_VERSION,
        "official_report": report,
    }


def run_official(row, patch, *, timeout=1800):
    validate_instance(row)
    if not isinstance(patch, str) or len(patch.encode()) > MAX_PATCH_BYTES:
        raise ValueError(f"invalid/oversized submission patch: {row['instance_id']}")
    if type(timeout) is not int or timeout <= 0:
        raise ValueError("verifier timeout must be a positive integer")
    import swebench
    from swebench.harness.run_evaluation import run_instance
    from swebench.harness.utils import make_test_spec

    import docker

    if swebench.__version__ != HARNESS_VERSION:
        raise RuntimeError(f"expected swebench {HARNESS_VERSION}, got {swebench.__version__}")
    client = docker.from_env(timeout=1800)
    try:
        image = client.images.get(row["image"])
        if image.attrs.get("Architecture") != "amd64" or image.attrs.get("Os") != "linux":
            raise ValueError(f"wrong image platform: {row['image']}")
        prediction = {
            "instance_id": row["instance_id"],
            "model_name_or_path": "miles-multilingual",
            "model_patch": patch,
        }
        result = run_instance(
            make_test_spec(row),
            prediction,
            client,
            "harbor-" + uuid.uuid4().hex,
            timeout=timeout,
            skip_patch=not patch.strip(),
        )
        return checked_report(result, row["instance_id"])
    finally:
        client.close()


def main():
    rows = json.loads(Path("/tests/dataset.json").read_text())
    if not isinstance(rows, list) or len(rows) != 1:
        raise ValueError("expected one isolated verifier instance")
    path = Path("/tmp/swebench-submission.patch")
    if not path.is_file() or path.stat().st_size > MAX_PATCH_BYTES:
        raise ValueError("missing/oversized submission patch")
    report = run_official(rows[0], path.read_text())
    Path("report.json").write_text(json.dumps(report, indent=2) + "\n")
    Path("reward.txt").write_text(str(report["reward"]) + "\n")


if __name__ == "__main__":
    main()
