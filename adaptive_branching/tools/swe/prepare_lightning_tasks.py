"""Copy prepared R2E tasks into an isolated, network-disabled agent recipe.

No images or dataset rows change; the separate verifier keeps its own policy.
The existing tasks must remain untouched for the currently running experiment.
"""

import argparse
import json
import re
import shutil
from pathlib import Path

import yaml

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib


def transform_config(text: str) -> str:
    if not isinstance(text, str) or not text.strip():
        raise ValueError("task TOML must be nonempty")
    original = tomllib.loads(text)
    if original.get("verifier", {}).get("environment_mode") != "separate":
        raise ValueError("Lightning R2E requires a separate verifier environment")
    for section in ("agent", "environment"):
        pattern = rf"(?ms)(^\[{section}\]\s*\n)(.*?)(?=^\[|\Z)"
        match = re.search(pattern, text)
        if match is None:
            raise ValueError(f"missing [{section}] section")
        body = match.group(2)
        if re.search(r"(?m)^allowed_hosts\s*=", body):
            raise ValueError(f"remove explicit [{section}] allowed_hosts before preparing offline tasks")
        body, count = re.subn(r'(?m)^network_mode[ \t]*=[ \t]*"[^"]*"[ \t]*$', 'network_mode = "no-network"', body)
        if count > 1:
            raise ValueError("duplicate network_mode")
        if count == 0:
            body = 'network_mode = "no-network"\n' + body
        text = text[: match.start()] + match.group(1) + body + text[match.end() :]
    updated = tomllib.loads(text)
    if updated["verifier"] != original["verifier"]:
        raise AssertionError("verifier config changed")
    return text


def transform_compose(text: str) -> str:
    config = yaml.safe_load(text)
    if not isinstance(config, dict) or set(config.get("services", {})) != {"main"}:
        raise ValueError("expected a single main service in the R2E agent compose override")
    main = config["services"]["main"]
    if not isinstance(main, dict):
        raise ValueError("main service must be an object")
    # Explicit networks cause Harbor to skip egress sidecar attachment. Remove
    # the old public-network override so Harbor can enforce no-network.
    main.pop("networks", None)
    main.pop("network_mode", None)
    config.pop("networks", None)
    return yaml.safe_dump(config, sort_keys=False)


def prepare(source: Path, destination: Path) -> int:
    source, destination = source.resolve(), destination.resolve()
    if not source.is_dir() or destination.exists() or destination.is_relative_to(source):
        raise ValueError("source must exist and destination must be new and outside source")
    tasks = sorted(source.glob("*/task.toml"))
    if not tasks:
        raise ValueError("no prepared tasks found")
    changes = []
    for config in tasks:
        task = config.parent
        compose = task / "environment/docker-compose.yaml"
        if not compose.is_file() or not (task / "instruction.md").is_file():
            raise FileNotFoundError(f"incomplete prepared task: {task}")
        changes.append((task, transform_config(config.read_text()), transform_compose(compose.read_text())))
    destination.mkdir(parents=True)
    for task, config, compose in changes:
        target = destination / task.name
        shutil.copytree(task, target)
        (target / "task.toml").write_text(config)
        (target / "environment/docker-compose.yaml").write_text(compose)
    (destination / "lightning-manifest.json").write_text(
        json.dumps(
            {
                "source": str(source),
                "tasks": len(tasks),
                "agent_network": "no-network",
                "agent": "adaptive_branching.src.swe.lightning_harbor_agent:LightningSweAgent",
            },
            indent=2,
        )
        + "\n"
    )
    return len(tasks)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--destination", required=True, type=Path)
    args = parser.parse_args()
    print(f"Prepared {prepare(args.source, args.destination)} offline Lightning tasks")


if __name__ == "__main__":
    main()
