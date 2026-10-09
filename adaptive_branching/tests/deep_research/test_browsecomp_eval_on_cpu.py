import argparse
import asyncio
import copy
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from adaptive_branching.src.deep_research.value_cliff_prefixes import (
    PREFIX_INPUT_MODE,
    PREFIX_PROVENANCE_FIELDS,
    prefix_provenance,
)

MODULE_PATH = Path(__file__).resolve().parents[2] / "shells" / "eval" / "run_browsecomp_eval.py"
SPEC = importlib.util.spec_from_file_location("run_browsecomp_eval", MODULE_PATH)
assert SPEC and SPEC.loader
EVAL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EVAL)


def _prefix_row(*, prefix_turn=1, prefix_index=0):
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "question"},
    ]
    for turn in range(prefix_turn):
        call_id = f"call-{turn}"
        messages.extend(
            [
                {
                    "role": "assistant",
                    "content": "",
                    "reasoning_content": f"think {turn}",
                    "tool_calls": [
                        {
                            "id": call_id,
                            "type": "function",
                            "function": {"name": "search", "arguments": '{"query":"q"}'},
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": call_id, "name": "search", "content": "result"},
            ]
        )
    source_task_id = "source-42"
    failure_sample_index = 3
    return {
        "schema_version": 1,
        "task_id": f"{source_task_id}::failure_s{failure_sample_index:02d}::turn_{prefix_turn:03d}",
        "source_task_id": source_task_id,
        "source_cohort_index": 7,
        "source_failure_sample_index": failure_sample_index,
        "source_failure_line_number": 99,
        "source_trajectory_assistant_turns": 7,
        "source_max_turns": 200,
        "prefix_index": prefix_index,
        "prefix_turn": prefix_turn,
        "prefix_kind": "coarse_grid" if prefix_turn % 5 == 0 else "pre_final",
        "remaining_turns": 200 - prefix_turn,
        "k4_success_count": 1,
        "k4_failure_count": 3,
        "question": "question",
        "ground_truth": "answer",
        "prefix_messages": messages,
    }


def _write_prefix_jsonl(tmp_path, row):
    path = tmp_path / "prefixes.jsonl"
    path.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def test_load_samples_supports_parquet_and_preserves_zero_id(tmp_path):
    path = tmp_path / "eval.parquet"
    pq.write_table(
        pa.table(
            {
                "id": pa.array([0, 1], type=pa.int64()),
                "question": ["q0", "q1"],
                "answer": ["a0", "a1"],
            }
        ),
        path,
    )

    samples = EVAL._load_samples(path, limit=None, offset=0)

    assert samples == [
        {"task_id": "0", "question": "q0", "ground_truth": "a0"},
        {"task_id": "1", "question": "q1", "ground_truth": "a1"},
    ]


def test_load_samples_rejects_duplicate_task_ids(tmp_path):
    path = tmp_path / "eval.jsonl"
    path.write_text(
        '{"id": 1, "question": "q0", "answer": "a0"}\n{"id": 1, "question": "q1", "answer": "a1"}\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate task_id"):
        EVAL._load_samples(path, limit=None, offset=0)


def test_load_samples_accepts_validated_prefix_messages(tmp_path):
    row = _prefix_row(prefix_turn=1)
    path = _write_prefix_jsonl(tmp_path, row)

    samples = EVAL._load_samples(
        path,
        limit=None,
        offset=0,
        input_mode=PREFIX_INPUT_MODE,
        max_turns=200,
    )

    assert samples == [row]
    prompt = EVAL._sample_prompt(samples[0], input_mode=PREFIX_INPUT_MODE)
    assert prompt == row["prefix_messages"]
    assert prompt is not row["prefix_messages"]
    assert prompt[0] is not row["prefix_messages"][0]


@pytest.mark.parametrize("corruption", ["assistant_tail", "turn_count", "max_turns"])
def test_load_samples_rejects_invalid_prefix_contract(tmp_path, corruption):
    row = _prefix_row(prefix_turn=1)
    evaluator_max_turns = 200
    expected_error = ""
    if corruption == "assistant_tail":
        row["prefix_messages"].pop()
        expected_error = "do not match tool-call ids"
    elif corruption == "turn_count":
        row["prefix_turn"] = 2
        row["remaining_turns"] = 198
        row["task_id"] = "source-42::failure_s03::turn_002"
        expected_error = "prefix contains 1 assistant turns"
    else:
        evaluator_max_turns = 199
        expected_error = "evaluator --max-turns=199"
    path = _write_prefix_jsonl(tmp_path, row)

    with pytest.raises(ValueError, match=expected_error):
        EVAL._load_samples(
            path,
            limit=None,
            offset=0,
            input_mode=PREFIX_INPUT_MODE,
            max_turns=evaluator_max_turns,
        )


@pytest.mark.parametrize("history_mode", ["keep5", "both"])
def test_prefix_input_configuration_forbids_history_mutation(history_mode):
    args = argparse.Namespace(
        input_mode=PREFIX_INPUT_MODE,
        history_mode=history_mode,
        system_prompt=None,
        no_system_prompt=False,
    )

    with pytest.raises(ValueError, match="requires --history-mode react"):
        EVAL._validate_input_configuration(args)


def test_prefix_input_configuration_forbids_system_override():
    args = argparse.Namespace(
        input_mode=PREFIX_INPUT_MODE,
        history_mode="react",
        system_prompt="replacement",
        no_system_prompt=False,
    )

    with pytest.raises(ValueError, match="system message is immutable"):
        EVAL._validate_input_configuration(args)


def _request_args(extra_request_json=None, *, agent_api="chat", reasoning_effort=None):
    return argparse.Namespace(
        agent_api=agent_api,
        model="test-model",
        max_tokens=32,
        temperature=0.7,
        top_p=0.95,
        top_k=40,
        repetition_penalty=None,
        reasoning_effort=reasoning_effort,
        thinking_budget=None,
        thinking_type=None,
        extra_request_json=extra_request_json,
    )


def test_request_kwargs_preserve_history_thinking_by_default():
    kwargs = EVAL._request_kwargs(_request_args())

    assert kwargs["chat_template_kwargs"] == {"clear_thinking": False}


def test_request_kwargs_merge_extra_fields_without_losing_clear_thinking():
    kwargs = EVAL._request_kwargs(
        _request_args('{"seed":7,"chat_template_kwargs":{"clear_thinking":false,"custom":"value"}}')
    )

    assert kwargs["seed"] == 7
    assert kwargs["chat_template_kwargs"] == {"clear_thinking": False, "custom": "value"}


def test_request_kwargs_add_glm5_thinking_budget_processor():
    args = _request_args()
    args.thinking_budget = 4096

    kwargs = EVAL._request_kwargs(args)

    assert json.loads(kwargs["custom_logit_processor"])["callable"]
    assert kwargs["custom_params"] == {"thinking_budget": 4096}


@pytest.mark.parametrize("thinking_budget", [-1, True, 1.5])
def test_request_kwargs_reject_invalid_thinking_budget(thinking_budget):
    args = _request_args()
    args.thinking_budget = thinking_budget

    with pytest.raises(ValueError, match="non-negative integer"):
        EVAL._request_kwargs(args)


def test_request_kwargs_reject_thinking_budget_custom_processor_conflict():
    args = _request_args('{"custom_params":{"thinking_budget":8}}')
    args.thinking_budget = 4096

    with pytest.raises(ValueError, match="cannot be combined"):
        EVAL._request_kwargs(args)


def test_request_kwargs_reject_thinking_budget_for_responses_api():
    args = _request_args(agent_api="responses")
    args.thinking_budget = 4096

    with pytest.raises(ValueError, match="requires --agent-api=chat"):
        EVAL._request_kwargs(args)


@pytest.mark.parametrize(
    "extra_request_json",
    [
        '{"chat_template_kwargs":{"clear_thinking":true}}',
        '{"chat_template_kwargs":"not-an-object"}',
    ],
)
def test_request_kwargs_reject_configuration_that_can_clear_history_thinking(extra_request_json):
    with pytest.raises(ValueError):
        EVAL._request_kwargs(_request_args(extra_request_json))


@pytest.mark.asyncio
async def test_endpoint_preflight_sends_clear_thinking(monkeypatch):
    captured = {}

    class FakeResponse:
        status_code = 200
        text = ""

    class FakeClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, traceback):
            return False

        async def post(self, url, json, headers=None):
            captured["url"] = url
            captured["payload"] = json
            captured["headers"] = headers
            return FakeResponse()

    monkeypatch.setattr(EVAL.httpx, "AsyncClient", lambda **kwargs: FakeClient())

    await EVAL._check_agent_endpoint(
        "http://localhost:30000",
        "test-model",
        10,
        EVAL._request_kwargs(_request_args()),
        "chat",
    )

    assert captured["url"] == "http://localhost:30000/v1/chat/completions"
    assert captured["payload"]["chat_template_kwargs"] == {"clear_thinking": False}


@pytest.mark.asyncio
async def test_responses_endpoint_preflight_uses_reasoning_none(monkeypatch):
    captured = {}

    class FakeResponse:
        status_code = 200
        text = ""

    class FakeClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, traceback):
            return False

        async def post(self, url, json, headers=None):
            captured["url"] = url
            captured["payload"] = json
            captured["headers"] = headers
            return FakeResponse()

    monkeypatch.setenv("AGENT_CHAT_API_KEY", "test-key")
    monkeypatch.setattr(EVAL.httpx, "AsyncClient", lambda **kwargs: FakeClient())
    request_kwargs = EVAL._request_kwargs(_request_args("{}", agent_api="responses", reasoning_effort="none"))

    await EVAL._check_agent_endpoint(
        "http://localhost:4141",
        "gpt-5.6-sol",
        10,
        request_kwargs,
        "responses",
    )

    assert captured["url"] == "http://localhost:4141/v1/responses"
    assert captured["payload"]["reasoning"] == {"effort": "none"}
    assert captured["payload"]["input"] == [{"role": "user", "content": "ping"}]
    assert "chat_template_kwargs" not in captured["payload"]
    assert captured["headers"] == {"Authorization": "Bearer test-key"}


@pytest.mark.parametrize(
    ("state", "stop_on_success", "expected"),
    [
        ({"attempts": 1, "success": True}, True, True),
        ({"attempts": 1, "success": True}, False, False),
        ({"attempts": 8, "success": False}, False, True),
    ],
)
def test_fixed_sampling_does_not_early_stop(state, stop_on_success, expected):
    assert EVAL._is_task_finished(state, 8, stop_on_success=stop_on_success) is expected


def test_valid_attempt_tracking_only_advances_through_contiguous_slots():
    state = {"attempts": 0, "valid_attempts": set()}

    EVAL._mark_valid_attempt(state, 2)
    assert state["attempts"] == 0

    EVAL._mark_valid_attempt(state, 1)
    assert state["attempts"] == 2


def test_length_truncated_agent_record_is_not_a_valid_attempt():
    record = {
        "metrics": {"agent_finished": False, "agent_last_finish_reason": "length"},
        "score": 0.0,
        "acc": False,
    }

    assert EVAL._resume_record_counts_as_attempt(record, require_judge=True) is False


def test_metadata_enables_raw_response_capture_only_when_requested():
    args = argparse.Namespace(
        max_turns=5,
        max_seq_len=1024,
        keep_tool_results=5,
        agent_api="responses",
        system_prompt=None,
        no_system_prompt=False,
        tool_call_text_retries=0,
        save_raw_responses=True,
    )

    metadata = EVAL._metadata(args, "react")

    assert metadata["agent_capture_raw_responses"] is True


def test_without_duplicate_responses_output_handles_empty_single_and_invalid_messages():
    assert EVAL._without_duplicate_responses_output([]) == []
    original = [
        {
            "role": "assistant",
            "content": "answer",
            "_responses_output": [{"type": "reasoning", "encrypted_content": "opaque"}],
        }
    ]

    assert EVAL._without_duplicate_responses_output(original) == [{"role": "assistant", "content": "answer"}]
    assert original[0]["_responses_output"][0]["encrypted_content"] == "opaque"

    with pytest.raises(TypeError, match="messages must be a list"):
        EVAL._without_duplicate_responses_output({})
    with pytest.raises(TypeError, match=r"messages\[0\] must be an object"):
        EVAL._without_duplicate_responses_output(["invalid"])


def test_run_mode_saves_raw_response_trace_at_record_top_level(tmp_path, monkeypatch):
    import adaptive_branching.src.deep_research as src_package

    raw_trace = [
        {
            "request_index": 1,
            "agent_turn": 1,
            "request_kind": "agent_turn",
            "retry_index": 0,
            "response": {
                "status": "completed",
                "output": [
                    {"type": "reasoning", "encrypted_content": "opaque"},
                    {
                        "type": "message",
                        "content": [{"type": "output_text", "text": "answer"}],
                    },
                ],
            },
        }
    ]

    async def fake_run(**_kwargs):
        return {
            "messages": [
                {
                    "role": "assistant",
                    "content": "answer",
                    "_responses_output": raw_trace[0]["response"]["output"],
                }
            ],
            "agent_raw_responses": raw_trace,
            "agent_final_answer": "answer",
            "agent_turns": 1,
            "agent_finished": True,
            "agent_final_answer_present": True,
            "agent_session_tokens": 10,
        }

    fake_agent = ModuleType("adaptive_branching.src.deep_research.agent")
    fake_agent.run = fake_run
    monkeypatch.setitem(sys.modules, "adaptive_branching.src.deep_research.agent", fake_agent)
    monkeypatch.setattr(src_package, "agent", fake_agent, raising=False)
    args = _fixed_sampling_args(tmp_path, skip_judge=True, samples_per_task=1)
    args.agent_api = "responses"
    args.save_raw_responses = True
    args.retry_incomplete_fixed_samples = False
    samples = [{"task_id": "task", "question": "question", "ground_truth": "answer"}]

    asyncio.run(EVAL._run_mode(args, samples, "react"))

    record = json.loads((tmp_path / "test-model.react.jsonl").read_text().strip())
    assert record["raw_responses"] == raw_trace
    assert record["raw_responses"][0]["response"]["output"][0]["encrypted_content"] == "opaque"
    assert record["messages"] == [{"role": "assistant", "content": "answer"}]
    assert "agent_raw_responses" not in record["metrics"]


def _fixed_sampling_args(tmp_path, *, skip_judge, samples_per_task=3, max_incomplete_fixed_retries=3):
    return argparse.Namespace(
        agent_module="adaptive_branching.src.deep_research.agent",
        agent_api="chat",
        input_mode="question",
        out=str(tmp_path),
        model="test-model",
        samples_per_task=samples_per_task,
        retry_failed_attempts=0,
        retry_incomplete_fixed_samples=True,
        max_incomplete_fixed_retries=max_incomplete_fixed_retries,
        resume=False,
        skip_judge=skip_judge,
        base_url="http://unused/v1",
        max_tokens=32,
        temperature=0.7,
        top_p=0.95,
        top_k=None,
        repetition_penalty=None,
        reasoning_effort=None,
        thinking_type=None,
        extra_request_json=None,
        max_turns=2,
        max_seq_len=128,
        context_reserve_tokens=16,
        keep_tool_results=1,
        concurrency=1,
        judge_concurrency=1,
        keep_going=True,
    )


def test_prefix_run_passes_full_messages_remaining_horizon_and_writes_provenance(tmp_path, monkeypatch):
    import adaptive_branching.src.deep_research as src_package

    calls = []

    async def fake_run(*, prompt, metadata, **_kwargs):
        calls.append({"prompt": prompt, "metadata": metadata})
        return {
            "messages": [*prompt, {"role": "assistant", "content": "answer"}],
            "agent_final_answer": "answer",
            "agent_turns": 1,
            "agent_finished": True,
            "agent_final_answer_present": True,
            "agent_session_tokens": 40,
        }

    fake_agent = ModuleType("adaptive_branching.src.deep_research.agent")
    fake_agent.run = fake_run
    monkeypatch.setitem(sys.modules, "adaptive_branching.src.deep_research.agent", fake_agent)
    monkeypatch.setattr(src_package, "agent", fake_agent, raising=False)
    args = _fixed_sampling_args(tmp_path, skip_judge=True, samples_per_task=2)
    args.input_mode = PREFIX_INPUT_MODE
    args.max_turns = 200
    sample = _prefix_row(prefix_turn=5)

    summary = asyncio.run(EVAL._run_mode(args, [sample], "react"))

    assert len(calls) == 2
    assert all(call["prompt"] == sample["prefix_messages"] for call in calls)
    assert all(call["prompt"] is not sample["prefix_messages"] for call in calls)
    assert all(call["metadata"]["agent_max_turns"] == 195 for call in calls)
    records = [json.loads(line) for line in (tmp_path / "test-model.react.jsonl").read_text().splitlines()]
    assert len(records) == 2
    assert all(record["input_mode"] == PREFIX_INPUT_MODE for record in records)
    assert all(record["messages"][: len(sample["prefix_messages"])] == sample["prefix_messages"] for record in records)
    for field in PREFIX_PROVENANCE_FIELDS:
        assert all(record[field] == sample[field] for record in records)
    assert summary["input_prefixes"] == 1
    assert summary["expected_valid_records"] == 2
    assert summary["actual_valid_records"] == 2
    assert summary["valid_record_target_met"] is True


def test_prefix_run_fails_if_agent_mutates_replayed_messages(tmp_path, monkeypatch):
    import adaptive_branching.src.deep_research as src_package

    async def fake_run(*, prompt, **_kwargs):
        prompt[1]["content"] = "mutated"
        return {
            "messages": [*prompt, {"role": "assistant", "content": "answer"}],
            "agent_final_answer": "answer",
            "agent_turns": 1,
            "agent_finished": True,
        }

    fake_agent = ModuleType("adaptive_branching.src.deep_research.agent")
    fake_agent.run = fake_run
    monkeypatch.setitem(sys.modules, "adaptive_branching.src.deep_research.agent", fake_agent)
    monkeypatch.setattr(src_package, "agent", fake_agent, raising=False)
    args = _fixed_sampling_args(tmp_path, skip_judge=True, samples_per_task=2)
    args.input_mode = PREFIX_INPUT_MODE
    args.max_turns = 200

    with pytest.raises(ValueError, match="do not preserve the replay prefix"):
        asyncio.run(EVAL._run_mode(args, [_prefix_row(prefix_turn=1)], "react"))


@pytest.mark.parametrize("error_stage", ["generation", "judge"])
def test_fixed_sampling_retries_transient_errors_until_all_slots_are_valid(tmp_path, monkeypatch, error_stage):
    import adaptive_branching.src.deep_research as src_package

    calls = {"generation": 0, "judge": 0}

    async def fake_run(**_kwargs):
        calls["generation"] += 1
        if error_stage == "generation" and calls["generation"] == 1:
            raise RuntimeError("transient generation error")
        return {
            "messages": [{"role": "assistant", "content": "answer"}],
            "agent_final_answer": "answer",
            "agent_turns": 1,
            "agent_session_tokens": 4,
        }

    async def fake_score(_question, _ground_truth, _answer):
        calls["judge"] += 1
        if error_stage == "judge" and calls["judge"] == 1:
            raise RuntimeError("transient judge error")
        return {"score": 1.0, "acc": True, "judge_raw": "ok", "judge_error": None}

    fake_agent = ModuleType("adaptive_branching.src.deep_research.agent")
    fake_agent.run = fake_run
    monkeypatch.setitem(sys.modules, "adaptive_branching.src.deep_research.agent", fake_agent)
    monkeypatch.setattr(src_package, "agent", fake_agent, raising=False)
    monkeypatch.setattr(EVAL, "_score", fake_score)
    args = _fixed_sampling_args(tmp_path, skip_judge=False)
    samples = [{"task_id": "task", "question": "question", "ground_truth": "answer"}]

    asyncio.run(EVAL._run_mode(args, samples, "react"))

    records = [json.loads(line) for line in (tmp_path / "test-model.react.jsonl").read_text().splitlines()]
    valid_records = [record for record in records if EVAL._resume_record_counts_as_attempt(record, require_judge=True)]
    error_records = [record for record in records if record.get("gen_error") or record.get("judge_error")]
    assert [record["attempt"] for record in valid_records] == [1, 2, 3]
    assert [record["sample_index"] for record in valid_records] == [0, 1, 2]
    assert len(error_records) == 1


def test_fixed_sampling_bounds_persistent_errors_and_finishes_other_tasks(tmp_path, monkeypatch):
    import adaptive_branching.src.deep_research as src_package

    async def fake_run(*, prompt, **_kwargs):
        if prompt == "bad question":
            raise RuntimeError("persistent generation error")
        return {
            "messages": [{"role": "assistant", "content": "answer"}],
            "agent_final_answer": "answer",
            "agent_turns": 1,
            "agent_session_tokens": 4,
        }

    async def fake_score(_question, _ground_truth, _answer):
        return {"score": 1.0, "acc": True, "judge_raw": "ok", "judge_error": None}

    fake_agent = ModuleType("adaptive_branching.src.deep_research.agent")
    fake_agent.run = fake_run
    monkeypatch.setitem(sys.modules, "adaptive_branching.src.deep_research.agent", fake_agent)
    monkeypatch.setattr(src_package, "agent", fake_agent, raising=False)
    monkeypatch.setattr(EVAL, "_score", fake_score)
    args = _fixed_sampling_args(
        tmp_path,
        skip_judge=False,
        samples_per_task=2,
        max_incomplete_fixed_retries=1,
    )
    samples = [
        {"task_id": "bad", "question": "bad question", "ground_truth": "answer"},
        {"task_id": "good", "question": "good question", "ground_truth": "answer"},
    ]

    with pytest.raises(RuntimeError, match="fixed sampling remains incomplete for 1/2"):
        asyncio.run(EVAL._run_mode(args, samples, "react"))

    records = [json.loads(line) for line in (tmp_path / "test-model.react.jsonl").read_text().splitlines()]
    bad_records = [record for record in records if record["task_id"] == "bad"]
    good_records = [record for record in records if record["task_id"] == "good"]
    assert len(bad_records) == 2
    assert all(record.get("gen_error") for record in bad_records)
    assert [record["attempt"] for record in good_records] == [1, 2]


def test_resume_rejects_duplicate_valid_fixed_attempts(tmp_path):
    out_path = tmp_path / "records.jsonl"
    base_record = {
        "task_id": "task",
        "question": "question",
        "ground_truth": "answer",
        "model": "test-model",
        "history_mode": "react",
        "attempt": 1,
        "max_attempts": 4,
        "samples_per_task": 4,
        "metrics": {"agent_turns": 1},
        "score": 1.0,
        "acc": True,
    }
    out_path.write_text(
        json.dumps(base_record) + "\n" + json.dumps(base_record) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate valid attempt 1"):
        EVAL._load_resume_records(
            out_path,
            [{"task_id": "task", "question": "question", "ground_truth": "answer"}],
            model="test-model",
            history_mode="react",
            require_judge=True,
            max_attempts=4,
            samples_per_task=4,
            stop_on_success=False,
        )


def test_prefix_resume_rejects_changed_prefix_messages(tmp_path):
    out_path = tmp_path / "records.jsonl"
    sample = _prefix_row(prefix_turn=1)
    record = {
        "task_id": sample["task_id"],
        "question": sample["question"],
        "ground_truth": sample["ground_truth"],
        "model": "test-model",
        "history_mode": "react",
        "input_mode": PREFIX_INPUT_MODE,
        "attempt": 1,
        "max_attempts": 8,
        "samples_per_task": 8,
        "metrics": {"agent_turns": 1, "agent_finished": True},
        "score": 1.0,
        "acc": True,
        "messages": [*copy.deepcopy(sample["prefix_messages"]), {"role": "assistant", "content": "answer"}],
        **prefix_provenance(sample),
    }
    record["messages"][1]["content"] = "different question"
    out_path.write_text(json.dumps(record) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="cannot resume incompatible.*do not preserve the replay prefix"):
        EVAL._load_resume_records(
            out_path,
            [sample],
            model="test-model",
            history_mode="react",
            require_judge=True,
            max_attempts=8,
            samples_per_task=8,
            stop_on_success=False,
            input_mode=PREFIX_INPUT_MODE,
        )


def test_continuation_runner_passes_offset_and_records_provenance(tmp_path, monkeypatch):
    import adaptive_branching.src.deep_research as src_package

    calls = []

    async def fake_run(*, prompt, metadata, **_kwargs):
        calls.append((copy.deepcopy(prompt), metadata))
        return {
            "messages": [*prompt, {"role": "assistant", "content": "answer"}],
            "agent_turns": 201,
            "agent_finished": True,
            "agent_final_answer": "answer",
        }

    fake_agent = ModuleType("adaptive_branching.src.deep_research.agent")
    fake_agent.run = fake_run
    monkeypatch.setitem(sys.modules, "adaptive_branching.src.deep_research.agent", fake_agent)
    monkeypatch.setattr(src_package, "agent", fake_agent, raising=False)
    args = _fixed_sampling_args(tmp_path, skip_judge=True, samples_per_task=1)
    args.max_turns = 600
    sample = {
        "task_id": "task",
        "question": "question",
        "ground_truth": "answer",
        "continuation": {"prefix_turns": 200, "total_max_turns": 600},
        "continuation_messages": [{"role": "user", "content": "question"}],
        "source_metrics": {"agent_turns": 200},
        "initial_session_tokens": 100,
    }
    asyncio.run(EVAL._run_mode(args, [sample], "keep5"))
    assert calls[0][0] == sample["continuation_messages"]
    assert calls[0][1]["agent_turn_offset"] == 200
    assert calls[0][1]["agent_max_turns"] == 600
    assert calls[0][1]["agent_initial_session_tokens"] == 100
    record = json.loads((tmp_path / "test-model.keep5.jsonl").read_text())
    assert record["continuation"] == sample["continuation"]
    assert record["source_metrics"] == sample["source_metrics"]
    changed = copy.deepcopy(sample)
    changed["continuation"]["total_max_turns"] = 700
    assert "continuation source or total turn budget differs" in EVAL._resume_record_mismatches(
        record,
        changed,
        model="test-model",
        history_mode="keep5",
        max_attempts=1,
        samples_per_task=1,
        input_mode="question",
    )
