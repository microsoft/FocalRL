import copy

import pytest

from adaptive_branching.src.swe.termination import context_reserve_hit, local_context_reserve_hit, max_turns_hit


@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize(
    "markers, expected",
    [
        ({}, False),
        ({"agent_context_reserve_hit": None}, False),
        ({"agent_context_reserve_hit": False}, False),
        ({"agent_context_reserve_hit": True}, True),
        ({"agent_forced_final_answer_reason": "context_reserve"}, True),
        ({"agent_forced_final_answer_reason": "max_turns"}, False),
        ({"agent_context_reserve_hit": True, "agent_forced_final_answer_reason": "context_reserve"}, True),
    ],
)
def test_context_marker_is_independent_of_status_reward_and_storage(nested, markers, expected):
    metadata = {"agent_metrics": markers} if nested else markers
    before = copy.deepcopy(metadata)
    assert context_reserve_hit(metadata) is expected
    assert metadata == before


@pytest.mark.parametrize(
    "metadata",
    [
        None,
        [],
        {"agent_metrics": None},
        {"agent_metrics": []},
        {"agent_context_reserve_hit": "false"},
        {"agent_context_reserve_hit": 1},
        {"agent_metrics": {"agent_context_reserve_hit": 0}},
        {"agent_forced_final_answer_reason": ""},
        {"agent_forced_final_answer_reason": "typo"},
        {"agent_forced_final_answer_reason": 1},
        {"agent_context_reserve_hit": False, "agent_forced_final_answer_reason": "context_reserve"},
    ],
)
def test_context_marker_rejects_invalid_or_conflicting_values(metadata):
    with pytest.raises((TypeError, ValueError)):
        context_reserve_hit(metadata)


@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize(
    "markers, expected",
    [
        ({}, False),
        ({"agent_max_turns_hit": None}, False),
        ({"agent_max_turns_hit": False}, False),
        ({"agent_max_turns_hit": True}, True),
        ({"agent_forced_final_answer_reason": "max_turns"}, True),
        ({"agent_forced_final_answer_reason": "context_reserve"}, False),
        ({"agent_max_turns_hit": True, "agent_forced_final_answer_reason": "max_turns"}, True),
        ({"turns": 250, "agent_max_turns": 250}, False),  # Count alone is not evidence of a forced stop.
    ],
)
def test_turn_limit_marker_preserves_unmarked_horizons_and_input(nested, markers, expected):
    metadata = {"agent_metrics": markers} if nested else markers
    before = copy.deepcopy(metadata)
    assert max_turns_hit(metadata) is expected
    assert metadata == before


@pytest.mark.parametrize(
    "metadata",
    [
        None,
        [],
        {"agent_metrics": None},
        {"agent_metrics": []},
        {"agent_max_turns_hit": "false"},
        {"agent_max_turns_hit": 1},
        {"agent_metrics": {"agent_max_turns_hit": 0}},
        {"agent_forced_final_answer_reason": ""},
        {"agent_forced_final_answer_reason": "typo"},
        {"agent_forced_final_answer_reason": 1},
        {"agent_max_turns_hit": False, "agent_forced_final_answer_reason": "max_turns"},
        {"agent_metrics": {"agent_max_turns_hit": False, "agent_forced_final_answer_reason": "max_turns"}},
    ],
)
def test_turn_limit_marker_rejects_invalid_or_conflicting_values(metadata):
    with pytest.raises((TypeError, ValueError)):
        max_turns_hit(metadata)


@pytest.mark.parametrize(
    "turns, prompt, completion, expected",
    [
        (1, 229000, 375, False),
        (1, 229000, 376, True),
        (1, 229000, 377, True),
        (15, 229000, 377, False),
        (0, 0, 0, False),
    ],
)
def test_legacy_local_guard_uses_exact_recorded_budget_and_preserves_horizon(turns, prompt, completion, expected):
    request = {"run_verifier": False, "max_seq_len": 262144, "context_reserve_tokens": 32768, "max_turns": 15}
    messages = [
        {
            "role": "assistant",
            "extra": {
                "response": {
                    "usage": {
                        "prompt_tokens": prompt,
                        "completion_tokens": completion,
                    }
                }
            },
        },
        {"role": "user", "content": "tool observation"},
    ]
    assert local_context_reserve_hit(request, {"turns": turns}, messages) is expected


def test_legacy_local_without_configured_reserve():
    assert local_context_reserve_hit({"run_verifier": False}, {}, []) is False


@pytest.mark.parametrize(
    "failure", ["not_local", "reserve", "context", "turns", "empty", "message", "extra", "usage", "tokens"]
)
def test_legacy_local_guard_rejects_invalid_evidence(failure):
    request = {"run_verifier": False, "max_seq_len": 262144, "context_reserve_tokens": 32768, "max_turns": 15}
    metrics = {"turns": 1}
    messages = [{"extra": {"response": {"usage": {"prompt_tokens": 229376}}}}]
    if failure == "not_local":
        request["run_verifier"] = True
    if failure == "reserve":
        request["context_reserve_tokens"] = 0
    if failure == "context":
        request["max_seq_len"] = 32768
    if failure == "turns":
        metrics["turns"] = None
    if failure == "empty":
        messages = []
    if failure == "message":
        messages = [None]
    if failure == "extra":
        messages = [{"extra": []}]
    if failure == "usage":
        messages = [{"extra": {"response": {"usage": False}}}]
    if failure == "tokens":
        messages = [{"extra": {"response": {"usage": {"prompt_tokens": False}}}}]
    with pytest.raises((TypeError, ValueError)):
        local_context_reserve_hit(request, metrics, messages)


@pytest.mark.parametrize("value,expected", [("ALL", "all"), ("Wrong", "wrong"), ("None", "none"), (" all ", "all")])
def test_force_exclude_normalizes_known_modes(value, expected):
    from adaptive_branching.src.swe.termination import normalize_force_exclude

    assert normalize_force_exclude(value) == expected


@pytest.mark.parametrize("value", ["", "sometimes", None, True, 1, []])
def test_force_exclude_rejects_invalid_modes(value):
    from adaptive_branching.src.swe.termination import normalize_force_exclude

    with pytest.raises(ValueError, match="FORCE_EXCLUDE"):
        normalize_force_exclude(value)
