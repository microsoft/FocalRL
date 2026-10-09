"""Pinned official Lite verifier, also bundled with its validation sibling in tasks."""

import json
import re
import uuid
from pathlib import Path

if __package__:
    from .multilingual_verifier import MAX_PATCH_BYTES
    from .multilingual_verifier import checked_report as _checked_report
    from .multilingual_verifier import validate_instance as _validate_instance
else:
    from multilingual_verifier import MAX_PATCH_BYTES
    from multilingual_verifier import checked_report as _checked_report
    from multilingual_verifier import validate_instance as _validate_instance

HARNESS_VERSION = "5.0.2"
DATASET = "SWE-bench_Lite"


def validate_instance(row):
    """Validate hydrated official data and its frozen Docker config digest."""
    _validate_instance(row)
    if row.get("dataset") != DATASET:
        raise ValueError(f"expected {DATASET}: {row['instance_id']}")
    digest = row.get("image_config_digest")
    if not isinstance(digest, str) or not re.fullmatch(r"sha256:[a-f0-9]{64}", digest):
        raise ValueError(f"invalid image config digest: {row['instance_id']}")
    if not re.fullmatch(r"swebench/sweb\.eval\.x86_64\.[a-z0-9_.-]+:latest", row["image"]):
        raise ValueError(f"unexpected Lite image: {row['image']}")
    return row


def checked_report(result, instance_id):
    """Keep official verdicts, but never turn reported infrastructure faults into zero."""
    report = _checked_report(result, instance_id)
    item = report["official_report"][instance_id]
    if type(item.get("infra_failure")) is not bool:
        raise ValueError(f"missing official infrastructure flag: {instance_id}")
    if item["infra_failure"]:
        raise RuntimeError(f"official infrastructure failure: {instance_id}: {item.get('infra_failure_reason')}")
    return {"dataset": DATASET, **report, "official_harness_version": HARNESS_VERSION}


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
        if image.id != row["image_config_digest"]:
            raise ValueError(f"image config mismatch: {row['instance_id']}: {image.id}")
        spec = make_test_spec(row)
        if spec.instance_id != row["instance_id"] or spec.image != row["image"]:
            raise ValueError(f"official TestSpec identity mismatch: {row['instance_id']}")
        # Pass the immutable local image ID, not a mutable imported tag, to Docker.
        spec.image = image.id
        prediction = {
            "instance_id": row["instance_id"],
            "model_name_or_path": "miles-lite",
            "model_patch": patch,
        }
        result = run_instance(
            spec,
            prediction,
            client,
            "harbor-lite-" + uuid.uuid4().hex,
            timeout=timeout,
            skip_patch=not patch.strip(),
        )
        return checked_report(result, row["instance_id"])
    finally:
        client.close()


def main():
    rows = json.loads(Path("/tests/dataset.json").read_text())
    if not isinstance(rows, list) or len(rows) != 1:
        raise ValueError("expected one isolated Lite verifier instance")
    path = Path("/tmp/swebench-submission.patch")
    if not path.is_file() or path.stat().st_size > MAX_PATCH_BYTES:
        raise ValueError("missing/oversized submission patch")
    # Universal-newline decoding can turn an embedded CR into a new diff line.
    # Preserve the collected patch bytes when passing text to the harness.
    report = run_official(rows[0], path.read_bytes().decode("utf-8"))
    Path("report.json").write_text(json.dumps(report, indent=2) + "\n")
    Path("reward.txt").write_text(str(report["reward"]) + "\n")


if __name__ == "__main__":
    main()
