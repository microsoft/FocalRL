from __future__ import annotations

import pytest


def test_tool_concurrency_comes_from_tools_yaml():
    from adaptive_branching.src.deep_research import agent

    assert agent.SEARCH_TOOL._sema._value == 16
    assert agent.BROWSE_TOOL._ms_sema._value == 16
    assert agent.BROWSE_TOOL._jina_sema._value == 64
    assert agent.BROWSE_TOOL._browser_llm_sema._value == 16
    assert agent.BROWSE_TOOL.browser_llm_model == "gpt-5.6-luna"


@pytest.mark.asyncio
async def test_judge_concurrency_comes_from_judge_yaml():
    from adaptive_branching.src.deep_research import reward_function, value_cliff_online, value_cliff_reward
    from adaptive_branching.src.deep_research.judge_config import judge_settings, positive_int

    outcome_limit = positive_int(judge_settings("full_outcome"), "max_concurrency")
    local_prm_limit = positive_int(judge_settings("local_prm"), "max_concurrency")
    locator_trace_chars = positive_int(judge_settings("locator"), "trace_max_chars")
    local_prm_trace_chars = positive_int(judge_settings("local_prm"), "trace_max_chars")
    assert outcome_limit == 16
    assert reward_function._shared_outcome_judge_semaphore(outcome_limit)._value == 16
    assert value_cliff_online._locator_concurrency() == 8
    assert value_cliff_reward._shared_judge_semaphore(local_prm_limit)._value == 16
    assert locator_trace_chars == 512000
    assert local_prm_trace_chars == 512000


def test_missing_secret_fails_during_config_load(monkeypatch):
    from adaptive_branching.src.deep_research.judge_config import judge_settings

    monkeypatch.delenv("LLM_JUDGE_KEY")
    with pytest.raises(ValueError, match="LLM_JUDGE_KEY"):
        judge_settings("full_outcome")


def test_judge_yaml_contains_only_live_sections():
    from adaptive_branching.src.deep_research.judge_config import load_judge_config, validate_judge_config

    assert set(load_judge_config()) == {"common", "full_outcome", "locator", "local_prm"}
    validate_judge_config()
