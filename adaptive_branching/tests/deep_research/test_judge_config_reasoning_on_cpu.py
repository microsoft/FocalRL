from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from adaptive_branching.src.deep_research import judge_config


@pytest.fixture
def config(monkeypatch):
    path = Path(judge_config.__file__).resolve().parents[2] / "config/judge_qwen35_4b.yaml"
    value = deepcopy(yaml.safe_load(path.read_text()))
    monkeypatch.setattr(judge_config, "load_judge_config", lambda: value)
    return value


@pytest.mark.parametrize("effort", ["", " ", "low", "medium", "high", "max"])
def test_optional_reasoning_values_are_accepted(config, effort):
    for section in ("full_outcome", "locator", "local_prm"):
        config[section]["reasoning_effort"] = effort
    judge_config.validate_judge_config()


@pytest.mark.parametrize("section", ["full_outcome", "locator", "local_prm"])
@pytest.mark.parametrize("effort", [None, False, 1, [], "none", "typo"])
def test_malformed_reasoning_fails_at_config_boundary(config, section, effort):
    config[section]["reasoning_effort"] = effort
    with pytest.raises(ValueError, match="reasoning_effort"):
        judge_config.validate_judge_config()


@pytest.mark.parametrize("section", ["full_outcome", "locator", "local_prm"])
def test_missing_reasoning_key_is_not_silently_defaulted(config, section):
    del config[section]["reasoning_effort"]
    with pytest.raises(ValueError, match="reasoning_effort"):
        judge_config.validate_judge_config()


def test_xhigh_matches_each_consumer_supported_values(config):
    config["locator"]["reasoning_effort"] = "xhigh"
    config["local_prm"]["reasoning_effort"] = "xhigh"
    judge_config.validate_judge_config()
    config["full_outcome"]["reasoning_effort"] = "xhigh"
    with pytest.raises(ValueError, match="full_outcome"):
        judge_config.validate_judge_config()


def test_mixed_qwen_judges_and_sol_outcome(config):
    config["full_outcome"].update(model="gpt-5.6-sol", reasoning_effort="medium")
    for section in ("locator", "local_prm"):
        config[section]["model"] = "Qwen/Qwen3.5-9B"
    judge_config.validate_judge_config()


@pytest.mark.parametrize("section", ["locator", "local_prm"])
@pytest.mark.parametrize("value", [True, False])
def test_explicit_thinking_config(config, section, value):
    config[section].update(model="Qwen/Qwen3.5-9B", enable_thinking=value)
    judge_config.validate_judge_config()
    assert judge_config.optional_enable_thinking(judge_config.judge_settings(section)) is value


@pytest.mark.parametrize("value", ["true", "false", "", None, 1, 0, [], {}])
def test_invalid_thinking_config(config, value):
    config["locator"]["enable_thinking"] = value
    with pytest.raises(ValueError, match="enable_thinking"):
        judge_config.validate_judge_config()


def test_absent_thinking_keeps_server_default():
    assert judge_config.optional_enable_thinking({}) is None
