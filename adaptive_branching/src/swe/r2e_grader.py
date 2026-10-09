"""R2E-Gym's executable status-map grader, kept dependency-free for task images."""

import argparse
import json
import re
from pathlib import Path
from typing import Any

_ANSI_COLOR = re.compile(r"\x1b\[[0-9;]*m")


def parse_pytest_statuses(log: str) -> dict[str, str]:
    """Match R2E-Gym's parser for pytest's short test summary."""
    if not isinstance(log, str):
        raise TypeError("log must be a string")
    if "short test summary info" not in log:
        return {}

    statuses: dict[str, str] = {}
    for line in log.split("short test summary info", 1)[1].strip().splitlines():
        if "PASSED" in line:
            statuses[".".join(line.split("::")[1:])] = "PASSED"
        elif "FAILED" in line:
            statuses[".".join(line.split("::")[1:]).split(" - ")[0]] = "FAILED"
        elif "ERROR" in line:
            statuses[".".join(line.split("::")[1:]).split(" - ")[0]] = "ERROR"
    return statuses


def normalize_statuses(statuses: dict[str, str]) -> dict[str, str]:
    if not isinstance(statuses, dict):
        raise TypeError("statuses must be a dict")
    normalized: dict[str, str] = {}
    for key in sorted(statuses):
        value = statuses[key]
        if not isinstance(key, str) or not isinstance(value, str):
            raise TypeError("status keys and values must be strings")
        normalized[_ANSI_COLOR.sub("", key).split(" - ")[0]] = value
    return normalized


def exact_status_reward(actual: dict[str, str], expected: dict[str, str]) -> float:
    """Return one only when the normalized test-state mappings are identical."""
    actual = normalize_statuses(actual)
    expected = normalize_statuses(expected)
    if len(actual) != len(expected):
        return 0.0
    return float(all(not key or expected.get(key) == value for key, value in actual.items()))


def grade_log(log: str, expected: dict[str, str]) -> tuple[float, dict[str, Any]]:
    if not expected:
        raise ValueError("expected status map must not be empty")
    actual = parse_pytest_statuses(log)
    reward = exact_status_reward(actual, expected)
    return reward, {"reward": reward, "actual": normalize_statuses(actual), "expected": normalize_statuses(expected)}


def infrastructure_failure(expected: dict[str, str], error: str) -> tuple[float, dict[str, Any]]:
    if not isinstance(expected, dict):
        raise TypeError("expected status map must be a dict")
    if not expected:
        raise ValueError("expected status map must not be empty")
    if not isinstance(error, str) or not error.strip():
        raise ValueError("infrastructure error must be a non-empty string")
    return 0.0, {
        "reward": 0.0,
        "actual": {},
        "expected": normalize_statuses(expected),
        "infrastructure_error": error.strip(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--log", type=Path)
    source.add_argument("--infra-error")
    parser.add_argument("--expected", type=Path, required=True)
    parser.add_argument("--reward", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    if not args.expected.is_file():
        raise FileNotFoundError(args.expected)
    if args.log is not None and not args.log.is_file():
        raise FileNotFoundError(args.log)
    expected = json.loads(args.expected.read_text(encoding="utf-8"))
    if not isinstance(expected, dict):
        raise TypeError("expected JSON must contain an object")
    if args.infra_error is not None:
        reward, report = infrastructure_failure(expected, args.infra_error)
    else:
        reward, report = grade_log(args.log.read_text(encoding="utf-8", errors="replace"), expected)

    args.reward.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.reward.write_text(f"{reward:g}\n", encoding="utf-8")
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    raise SystemExit(0)


if __name__ == "__main__":
    main()
