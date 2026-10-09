import copy
import json

import pytest

from adaptive_branching.src.deep_research.eval_continuation import load_continuation_samples


def _record(task_id="task", acc=False):
    return {
        "task_id": task_id,
        "model": "model",
        "history_mode": "keep5",
        "acc": acc,
        "question": "question",
        "ground_truth": "answer",
        "metrics": {
            "agent_max_turns_hit": True,
            "agent_turns": 1,
            "agent_max_seq_len": 100,
            "agent_keep_tool_results": 5,
            "agent_forced_final_answer": True,
            "agent_forced_final_answer_reason": "max_turns",
        },
        "messages": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "question"},
            {"role": "assistant", "tool_calls": [{"id": "a"}]},
            {"role": "tool", "tool_call_id": "a", "content": "observation"},
            {"role": "user", "content": "final question"},
            {"role": "assistant", "content": "forced answer"},
        ],
    }


def _load(tmp_path, records, **kwargs):
    path = tmp_path / "source.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in records))
    return load_continuation_samples(
        path,
        model="model",
        total_turns=kwargs.get("total", 3),
        max_seq_len=100,
        keep_tool_results=5,
        final_prompt=lambda q: "final " + q,
    )


def test_selects_both_correct_and_wrong_but_not_natural_or_context(tmp_path):
    records = [_record("correct", True), _record("wrong")]
    natural = _record("natural")
    natural["metrics"]["agent_max_turns_hit"] = False
    context = copy.deepcopy(natural)
    context["task_id"] = "context"
    context["metrics"]["agent_context_reserve_hit"] = True
    samples = _load(tmp_path, records + [natural, context])
    assert [s["task_id"] for s in samples] == ["correct", "wrong"]
    assert all(s["continuation_messages"] == records[0]["messages"][:-2] for s in samples)
    assert samples[0]["continuation"]["remaining_turns"] == 2
    assert samples[0]["initial_session_tokens"] == 0
    assert not samples[0]["continuation"]["initial_token_count_known"]


def test_known_initial_budget_and_single_remaining_turn(tmp_path):
    r = _record()
    r["metrics"]["agent_pre_forced_session_tokens"] = 70
    s = _load(tmp_path, [r], total=2)[0]
    assert s["initial_session_tokens"] == 70
    assert s["continuation"]["initial_token_count_known"]
    assert s["continuation"]["remaining_turns"] == 1


@pytest.mark.parametrize(
    "case", ["empty", "duplicate", "bad_tail", "missing_tool", "wrong_turns", "same_cap", "context", "keep", "model"]
)
def test_rejects_ambiguous_or_invalid_sources(tmp_path, case):
    r = _record()
    records = [r]
    total = 3
    if case == "empty":
        records = []
    elif case == "duplicate":
        records = [r, r]
    elif case == "bad_tail":
        r["messages"][-2]["content"] = "some other instruction"
    elif case == "missing_tool":
        r["messages"].pop(3)
    elif case == "wrong_turns":
        r["metrics"]["agent_turns"] = 2
    elif case == "same_cap":
        total = 1
    elif case == "context":
        r["metrics"]["agent_context_reserve_hit"] = True
    elif case == "keep":
        r["metrics"]["agent_keep_tool_results"] = 4
    elif case == "model":
        r["model"] = "other"
    with pytest.raises(ValueError):
        _load(tmp_path, records, total=total)
