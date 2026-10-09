"""Load the shared judge configuration used by training and evaluation."""

from __future__ import annotations

import os
import re
from functools import cache
from pathlib import Path
from typing import Any

_OC_ENV_RE = re.compile(r"\$\{oc\.env:([^,}]+)(?:,([^}]*))?\}")
DEFAULT_JUDGE_CONFIG = Path(__file__).resolve().parents[2] / "config" / "judge.yaml"


def judge_config_path() -> Path:
    return Path(os.getenv("AGENT_JUDGE_CONFIG", str(DEFAULT_JUDGE_CONFIG))).expanduser()


def load_judge_config() -> dict[str, Any]:
    path = judge_config_path()
    if not path.is_file():
        raise FileNotFoundError(f"judge config not found: {path}")
    return _resolve_env(_load_raw_config(str(path.resolve())))


@cache
def _load_raw_config(path: str) -> dict[str, Any]:
    import yaml

    loaded = yaml.safe_load(Path(path).read_text())
    if not isinstance(loaded, dict):
        raise ValueError(f"judge config must contain a mapping: {path}")
    return loaded


def judge_settings(section: str) -> dict[str, Any]:
    config = load_judge_config()
    common = config.get("common", {})
    selected = config.get(section, {})
    if not isinstance(common, dict):
        raise ValueError("judge config 'common' must be a mapping")
    if not isinstance(selected, dict):
        raise ValueError(f"judge config {section!r} must be a mapping")
    return {**common, **selected}


def _resolve_env(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _resolve_env(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_resolve_env(item) for item in value]
    if not isinstance(value, str):
        return value

    def replace(match: re.Match[str]) -> str:
        name = match.group(1).strip()
        default = match.group(2)
        if name in os.environ:
            return os.environ[name]
        if default is not None:
            return default.strip()
        raise ValueError(f"required environment variable {name!r} is not set")

    return _OC_ENV_RE.sub(replace, value)


def positive_int(settings: dict[str, Any], key: str) -> int:
    raw = settings.get(key)
    try:
        value = int(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"judge config {key!r} must be an integer, got {raw!r}") from exc
    if value <= 0:
        raise ValueError(f"judge config {key!r} must be positive, got {value}")
    return value


def positive_float(settings: dict[str, Any], key: str) -> float:
    raw = settings.get(key)
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"judge config {key!r} must be a number, got {raw!r}") from exc
    if value <= 0:
        raise ValueError(f"judge config {key!r} must be positive, got {value}")
    return value


def required_str(settings: dict[str, Any], key: str) -> str:
    value = settings.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"judge config {key!r} must be a non-empty string")
    return value.strip()


def optional_enable_thinking(settings: dict[str, Any]) -> bool | None:
    """Missing means server default; explicit values must be YAML booleans."""
    if "enable_thinking" not in settings:
        return None
    value = settings["enable_thinking"]
    if not isinstance(value, bool):
        raise ValueError(f"judge config 'enable_thinking' must be a boolean, got {value!r}")
    if required_str(settings, "model").lower().startswith("gpt-"):
        raise ValueError("enable_thinking is only supported for chat judges, not GPT Responses API judges")
    return value


def validate_judge_config() -> None:
    for section in ("full_outcome", "locator", "local_prm"):
        settings = judge_settings(section)
        required_str(settings, "base_url")
        required_str(settings, "api_key")
        required_str(settings, "model")
        if section in {"locator", "local_prm"}:
            optional_enable_thinking(settings)
        effort = settings.get("reasoning_effort")
        allowed_efforts = {"", "low", "medium", "high", "max"}
        if section != "full_outcome":
            allowed_efforts.add("xhigh")
        if not isinstance(effort, str) or effort.strip() not in allowed_efforts:
            raise ValueError(f"judge config {section!r} has invalid reasoning_effort: {effort!r}")
        positive_float(settings, "timeout")
        positive_int(settings, "max_tokens")

    for section in ("full_outcome", "locator", "local_prm"):
        settings = judge_settings(section)
        positive_int(settings, "max_concurrency")
        positive_int(settings, "max_retries")

    locator = judge_settings("locator")
    positive_int(locator, "max_concurrency")
    positive_int(locator, "trace_max_chars")

    local_prm = judge_settings("local_prm")
    positive_int(local_prm, "trace_max_chars")
