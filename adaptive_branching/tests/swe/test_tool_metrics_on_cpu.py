import pytest

from adaptive_branching.src.swe.tool_metrics import extract_swe_tool_metrics, extract_swe_trajectory_tool_metrics


def action(call_id):
    return {"role": "assistant", "extra": {"actions": [{"tool_call_id": call_id, "command": "pytest"}]}}


def observation(call_id, code=0, exception=""):
    return {"role": "tool", "tool_call_id": call_id, "extra": {"returncode": code, "exception_info": exception}}


def test_separates_command_failures_from_execution_exceptions():
    messages = [
        action("a"),
        observation("a"),
        action("b"),
        observation("b", 1),
        action("c"),
        observation("c", -1, "Command timed out after 120 seconds"),
        action("d"),
        observation("d", -1, "Connection refused"),
    ]
    result = extract_swe_tool_metrics(messages)
    assert result["agent_tool_unit_success_count"] == 2
    assert result["agent_tool_unit_success_rate"] == 0.5
    assert result["agent_swe_tool_returncode_zero_rate"] == 0.5
    assert result["agent_swe_tool_timeout_count"] == 1
    assert result["agent_swe_tool_execution_exception_count"] == 2


def test_submit_and_unobserved_are_not_reported_as_infra_failures():
    messages = [
        action("a"),
        observation("a"),
        action("submit"),
        action("not_executed"),
        {"role": "exit", "extra": {"exit_status": "Submitted"}},
    ]
    result = extract_swe_tool_metrics(messages)
    assert result["agent_tool_call_count"] == 3
    assert result["agent_tool_unit_count"] == 1
    assert result["agent_swe_tool_submit_count"] == 1
    assert result["agent_swe_tool_unobserved_action_count"] == 1
    assert result["agent_tool_unit_success_rate"] == 1


def test_empty_and_no_results_have_no_fabricated_success_rate():
    for messages in ([], [action("pending")]):
        result = extract_swe_tool_metrics(messages)
        assert "agent_tool_unit_success_rate" not in result
        assert "agent_swe_tool_returncode_zero_rate" not in result
        assert result["agent_swe_tool_result_count"] == 0


def test_single_result_and_replay_prefix():
    messages = [action("history"), observation("history", 1), action("new"), observation("new")]
    result = extract_swe_tool_metrics(messages, prefix_messages=2)
    assert result["agent_tool_call_count"] == result["agent_tool_unit_success_count"] == 1
    assert result["agent_swe_tool_returncode_zero_rate"] == 1
    assert extract_swe_tool_metrics(messages, prefix_messages=4)["agent_tool_call_count"] == 0


@pytest.mark.parametrize(
    "messages,prefix,error",
    [
        (None, 0, TypeError),
        ([None], 0, TypeError),
        ([], -1, ValueError),
        ([], True, ValueError),
        ([], 1, ValueError),
        ([observation("orphan")], 0, ValueError),
        ([action("a"), action("a")], 0, ValueError),
        ([action("a"), observation("a"), observation("a")], 0, ValueError),
        ([action("a"), observation("a", True)], 0, TypeError),
        ([action("a"), observation("a", exception={})], 0, TypeError),
        ([{"role": "exit", "extra": {"exit_status": "Submitted"}}], 0, ValueError),
    ],
)
def test_invalid_boundaries_fail_fast(messages, prefix, error):
    with pytest.raises(error):
        extract_swe_tool_metrics(messages, prefix_messages=prefix)


def test_trajectory_replay_does_not_count_history():
    prefix = [action("history"), observation("history", 1)]
    trajectory = {"messages": prefix + [action("new"), observation("new")]}
    result = extract_swe_trajectory_tool_metrics(trajectory, {"messages": prefix})
    assert result["agent_tool_call_count"] == 1
    assert result["agent_swe_tool_returncode_nonzero_count"] == 0
    assert extract_swe_trajectory_tool_metrics({"messages": []})["agent_swe_tool_result_count"] == 0


@pytest.mark.parametrize(
    "trajectory,replay,error",
    [
        ({}, None, TypeError),
        ({"messages": []}, {}, TypeError),
        ({"messages": []}, {"messages": []}, ValueError),
        ({"messages": []}, {"messages": [action("a")]}, ValueError),
        (
            {"messages": [{"role": "user", "content": "changed"}]},
            {"messages": [{"role": "user", "content": "old"}]},
            ValueError,
        ),
    ],
)
def test_trajectory_replay_rejects_bad_prefix(trajectory, replay, error):
    with pytest.raises(error):
        extract_swe_trajectory_tool_metrics(trajectory, replay)


def test_registry_matches_core_logging():
    from adaptive_branching.src.swe.tool_metrics import SWE_TOOL_METRIC_KEYS, SWE_HARBOR_METRIC_KEYS
    from miles.utils.metric_utils import SWE_AGENT_METRIC_KEYS

    assert SWE_AGENT_METRIC_KEYS == SWE_TOOL_METRIC_KEYS | SWE_HARBOR_METRIC_KEYS | {
        "agent_repeated_function_close",
        "agent_format_failure_penalized",
        "agent_single_call_length_penalized",
        "agent_behavior_anomaly_penalized",
    }


def test_harbor_patch_vendors_identical_tool_extractor():
    from pathlib import Path

    root = Path(__file__).resolve().parents[3]
    patch = (root / "adaptive_branching/patches/harbor-miles-v0.20.0.patch").read_text()
    marker = "diff --git a/src/harbor/utils/swe_metrics.py b/src/harbor/utils/swe_metrics.py\n"
    assert patch.count(marker) == 1
    section = patch.split(marker, 1)[1].split("diff --git ", 1)[0]
    actual = "".join(
        line[1:] for line in section.splitlines(keepends=True) if line.startswith("+") and not line.startswith("+++")
    )
    assert actual == (root / "adaptive_branching/src/swe/tool_metrics.py").read_text()
