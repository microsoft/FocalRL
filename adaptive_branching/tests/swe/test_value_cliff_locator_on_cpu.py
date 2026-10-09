from __future__ import annotations

import asyncio
import json
import subprocess

import pytest

from adaptive_branching.src.swe.value_cliff_locator import (
    SWE_VALUE_CLIFF_LOCATOR_VERSION,
    SWE_VALUE_CLIFF_REQUIRED_KEYS,
    build_swe_value_cliff_locator_prompt,
    swe_value_cliff_locator_system,
    validate_swe_value_cliff_verdict,
)
from adaptive_branching.tools.swe.build_value_cliff_locator_inputs import (
    INPUT_SCHEMA_VERSION,
    build_locator_inputs,
    extract_golden_source_patch,
)
from adaptive_branching.tools.swe.score_value_cliff_rubric_locator import _read_inputs, _score_one


def _messages(*contents: str) -> list[dict]:
    messages: list[dict] = [{"role": "system", "content": "system"}, {"role": "user", "content": "issue"}]
    for index, content in enumerate(contents):
        call_id = f"call-{index}"
        messages.extend(
            [
                {
                    "role": "assistant",
                    "reasoning_content": content,
                    "tool_calls": [
                        {
                            "id": call_id,
                            "type": "function",
                            "function": {"name": "bash", "arguments": json.dumps({"command": f"echo {index}"})},
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": call_id, "name": "bash", "content": "ok"},
            ]
        )
    return messages


def _parsed_commit() -> str:
    return json.dumps(
        {
            "file_diffs": [
                {
                    "header": {"file": {"path": "pkg/core.py"}},
                    "is_binary_file": False,
                    "hunks": [
                        {
                            "descriptor": {
                                "old_range": {"start": 4, "length": 2},
                                "new_range": {"start": 4, "length": 2},
                                "section": "def target():",
                            },
                            "line_group": {
                                "all_lines": [
                                    {"type": "deleted", "content": "    return old"},
                                    {"type": "added", "content": "    return new"},
                                ]
                            },
                        }
                    ],
                },
                {
                    "header": {"file": {"path": "tests/test_core.py"}},
                    "is_binary_file": False,
                    "hunks": [
                        {
                            "descriptor": {
                                "old_range": {"start": 1, "length": 0},
                                "new_range": {"start": 1, "length": 1},
                                "section": "",
                            },
                            "line_group": {"all_lines": [{"type": "added", "content": "assert target() == new"}]},
                        }
                    ],
                },
            ]
        }
    )


def _rollout(instance_id: str, sample_index: int, reward: float, trial_dir: str, *, status: str = "Submitted"):
    return {
        "instance_id": instance_id,
        "sample_index": sample_index,
        "job_id": f"{instance_id}::sample-{sample_index}",
        "response": {
            "reward": reward,
            "exit_status": status,
            "trial_dir": trial_dir,
        },
    }


def test_swe_locator_prompt_matches_deep_research_schema():
    system = swe_value_cliff_locator_system(10)
    normalized_system = " ".join(system.split())
    assert "software-engineering agent" in normalized_system
    assert "at most 10 replacement" in normalized_system
    assert "assistant turns" in normalized_system
    assert "golden source patch" in normalized_system
    assert "Do not turn the golden diff or successful trajectory into a required" in normalized_system
    assert "Select the turn solely by the estimated value drop before writing its rubric" in normalized_system
    assert "first turn where the agent forms and starts relying on the key wrong hypothesis" in normalized_system
    assert "Prefer this upstream reasoning error over later code edits" in normalized_system
    assert "only briefly considered and then tested normally" in normalized_system
    assert "local replay horizon affects how the rubric is written, never which turn is selected" in normalized_system
    assert "easier to grade, or easier to repair" in normalized_system
    assert SWE_VALUE_CLIFF_LOCATOR_VERSION == "swe_value_cliff_rubric_locator_v3_20260904"
    assert SWE_VALUE_CLIFF_REQUIRED_KEYS == ("selected_turn", "value_drop_reason", "recovery_rubric")

    prompt, metadata = build_swe_value_cliff_locator_prompt(
        issue="fix the bug",
        golden_patch="diff --git a/a.py b/a.py",
        failed_messages=_messages("inspect", "wrong edit", "submit"),
        successful_messages=_messages("inspect", "right edit"),
        trace_max_chars=10_000,
    )
    assert "Select exactly one Failed Turn in [1, 2]" in prompt
    assert "## Golden source patch\ndiff --git" in prompt
    assert metadata == {
        "failed_assistant_turns": 3,
        "successful_assistant_turns": 2,
        "failed_trace_truncated": False,
        "successful_trace_truncated": False,
    }


def test_swe_locator_prompt_requires_general_reachable_independent_rubric():
    system = swe_value_cliff_locator_system(10)

    assert "minimum sufficient next diagnostic, inspection, or edit milestone" in system
    assert "Different valid repair strategies must" in system
    assert "A plausible replay that continues the harmful direction must score 0" in system
    assert "visibly abandons the mistake but has not yet made the targeted progress must score 0.5" in system
    assert "both abandons it and performs an equivalent decision-relevant move must score 1" in system
    assert "The two criteria must not restate each other" in system
    assert "Evidence for either criterion must be reachable inside the local horizon" in system
    assert "Do not require final verifier success" in system
    assert "Reject generic criteria" in system
    assert "private golden-patch details" in system


def test_swe_locator_rejects_empty_input_and_invalid_verdict():
    with pytest.raises(ValueError, match="issue"):
        build_swe_value_cliff_locator_prompt(
            issue="",
            golden_patch="patch",
            failed_messages=_messages("a", "b"),
            successful_messages=_messages("a"),
            trace_max_chars=100,
        )
    with pytest.raises(ValueError, match=r"\[1, 2\]"):
        validate_swe_value_cliff_verdict(
            {
                "selected_turn": 3,
                "value_drop_reason": "wrong",
                "recovery_rubric": {"avoid_error": "avoid", "redirect": "redirect"},
            },
            assistant_turns=3,
        )


def test_extract_golden_patch_contains_only_relevant_source_files():
    patch = extract_golden_source_patch(_parsed_commit(), ["pkg/core.py"])
    assert "diff --git a/pkg/core.py b/pkg/core.py" in patch
    assert "@@ -4,2 +4,2 @@ def target():" in patch
    assert "-    return old" in patch
    assert "+    return new" in patch
    assert "tests/test_core.py" not in patch

    with pytest.raises(ValueError, match="missing"):
        extract_golden_source_patch(_parsed_commit(), ["pkg/missing.py"])
    with pytest.raises(ValueError, match="non-empty"):
        extract_golden_source_patch(_parsed_commit(), [])


def test_golden_patch_preserves_missing_final_newline(tmp_path):
    parsed = json.loads(_parsed_commit())
    hunk = parsed["file_diffs"][0]["hunks"][0]
    hunk["descriptor"] = {"old_range": {"start": 1, "length": 1}, "new_range": {"start": 1, "length": 1}}
    hunk["line_group"]["all_lines"] = [
        {"type": "deleted", "content": "old"},
        {"type": "note", "content": "No newline at end of file"},
        {"type": "added", "content": "new"},
        {"type": "note", "content": "No newline at end of file"},
    ]
    patch = extract_golden_source_patch(json.dumps(parsed), ["pkg/core.py"])
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    source = tmp_path / "pkg" / "core.py"
    source.parent.mkdir()
    source.write_bytes(b"old")
    subprocess.run(["git", "apply", "-"], input=patch, text=True, cwd=tmp_path, check=True)
    assert source.read_bytes() == b"new"


@pytest.mark.parametrize("invalid", ["first", "duplicate", "unknown"])
def test_golden_patch_rejects_invalid_diff_notes(invalid):
    parsed = json.loads(_parsed_commit())
    lines = parsed["file_diffs"][0]["hunks"][0]["line_group"]["all_lines"]
    note = {"type": "note", "content": "No newline at end of file"}
    if invalid == "first":
        lines.insert(0, note)
    elif invalid == "duplicate":
        lines.extend([note, note])
    else:
        lines.append({"type": "note", "content": "unexpected"})
    with pytest.raises(ValueError, match="invalid diff note"):
        extract_golden_source_patch(json.dumps(parsed), ["pkg/core.py"])


def test_build_locator_inputs_selects_first_submitted_positive_and_negative(tmp_path):
    instance_id = "r2e-demo-" + "a" * 40
    trials_root = tmp_path / "trials"
    for trial_name, contents in (("positive", ("good",)), ("negative", ("inspect", "bad", "submit"))):
        trajectory = trials_root / trial_name / "agent" / "mini-swe-agent.trajectory.json"
        trajectory.parent.mkdir(parents=True)
        trajectory.write_text(json.dumps({"messages": _messages(*contents)}))

    records = build_locator_inputs(
        mixed_rollouts=[
            _rollout(instance_id, 3, 0.0, "/remote/negative"),
            _rollout(instance_id, 2, 1.0, "/remote/positive"),
            _rollout(instance_id, 0, 0.0, "/remote/ignored", status="LimitsExceeded"),
        ],
        prompts={instance_id: "issue"},
        dataset_rows={
            instance_id: {
                "parsed_commit_content": _parsed_commit(),
                "relevant_files": ["pkg/core.py"],
            }
        },
        trials_root=trials_root,
    )
    assert len(records) == 1
    assert records[0]["schema_version"] == INPUT_SCHEMA_VERSION
    assert records[0]["failure_sample_index"] == 3
    assert records[0]["matched_success_sample_index"] == 2
    assert records[0]["failure_trial"] == "negative"
    assert records[0]["matched_success_trial"] == "positive"


def test_build_locator_inputs_rejects_no_submitted_mixed_pair(tmp_path):
    instance_id = "r2e-demo-" + "b" * 40
    with pytest.raises(ValueError, match="no Submitted mixed"):
        build_locator_inputs(
            mixed_rollouts=[
                _rollout(instance_id, 0, 1.0, "/remote/positive"),
                _rollout(instance_id, 1, 0.0, "/remote/negative", status="LimitsExceeded"),
            ],
            prompts={instance_id: "issue"},
            dataset_rows={instance_id: {"parsed_commit_content": _parsed_commit(), "relevant_files": ["pkg/core.py"]}},
            trials_root=tmp_path,
        )


@pytest.mark.parametrize("invalid_reward", [True, 0.5, "1", None])
def test_build_locator_inputs_rejects_non_binary_reward(tmp_path, invalid_reward):
    instance_id = "r2e-demo-" + "c" * 40
    rollout = _rollout(instance_id, 0, 0.0, "/remote/negative")
    rollout["response"]["reward"] = invalid_reward

    with pytest.raises(ValueError, match="reward must be binary"):
        build_locator_inputs(
            mixed_rollouts=[rollout],
            prompts={instance_id: "issue"},
            dataset_rows={instance_id: {"parsed_commit_content": _parsed_commit(), "relevant_files": ["pkg/core.py"]}},
            trials_root=tmp_path,
        )


def test_score_one_uses_strict_verdict_and_records_provenance():
    class FakeClient:
        model = "gpt-test"
        reasoning_effort = "medium"
        max_tokens = 100
        sampling_params = {}

        async def complete_json(self, system, prompt, *, required_keys, tag, validate):
            assert "software-engineering agent" in system
            assert "## Golden source patch" in prompt
            assert required_keys == SWE_VALUE_CLIFF_REQUIRED_KEYS
            assert tag.startswith("swe_value_cliff_locator_")
            return validate(
                {
                    "selected_turn": 2,
                    "value_drop_reason": "The edit commits to the wrong ownership layer.",
                    "recovery_rubric": {
                        "avoid_error": "Stop modifying the downstream symptom site.",
                        "redirect": "Inspect the upstream owner that determines whether the value reaches this code.",
                    },
                }
            )

    row = {
        "task_id": "r2e-demo-" + "c" * 40,
        "issue": "issue",
        "golden_patch": "diff --git a/a.py b/a.py",
        "golden_patch_files": ["a.py"],
        "failure_sample_index": 1,
        "matched_success_sample_index": 0,
        "failure_trial": "failure",
        "matched_success_trial": "success",
        "failed_messages": _messages("inspect", "bad edit", "submit"),
        "successful_messages": _messages("inspect", "good edit"),
    }
    result = asyncio.run(
        _score_one(
            FakeClient(),
            asyncio.Semaphore(1),
            row,
            trace_max_chars=10_000,
            local_horizon_turns=10,
        )
    )
    assert result["task_id"] == row["task_id"]
    assert result["failure_sample_index"] == 1
    assert result["verdict"]["selected_turn"] == 2
    assert result["local_horizon_turns"] == 10


def test_read_inputs_rejects_duplicate_task_ids(tmp_path):
    path = tmp_path / "inputs.jsonl"
    row = {
        "schema_version": INPUT_SCHEMA_VERSION,
        "task_id": "duplicate",
        "issue": "issue",
        "golden_patch": "patch",
        "failure_trial": "failure",
        "matched_success_trial": "success",
        "failed_messages": _messages("one", "two"),
        "successful_messages": _messages("one"),
    }
    path.write_text(json.dumps(row) + "\n" + json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="duplicate task_id"):
        _read_inputs(path)
