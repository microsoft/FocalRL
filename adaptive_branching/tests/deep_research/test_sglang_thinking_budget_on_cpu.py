from array import array
from types import SimpleNamespace

import dill
import pytest
import torch

from adaptive_branching.src.deep_research.sglang_thinking_budget import (
    Glm5ThinkingBudgetLogitProcessor,
    _open_thinking_start,
)

START = Glm5ThinkingBudgetLogitProcessor.THINKING_START_TOKEN_ID
END = Glm5ThinkingBudgetLogitProcessor.THINKING_END_TOKEN_ID
NEWLINE = Glm5ThinkingBudgetLogitProcessor.NEW_LINE_TOKEN_ID
VOCAB_SIZE = END + 2


def _request(origin_input_ids, output_ids):
    return SimpleNamespace(origin_input_ids=list(origin_input_ids), output_ids=list(output_ids))


def _apply(origin_input_ids, output_ids, budget):
    logits = torch.zeros(1, VOCAB_SIZE)
    params = [{"thinking_budget": budget, "__req__": _request(origin_input_ids, output_ids)}]
    returned = Glm5ThinkingBudgetLogitProcessor()(logits, params)
    assert returned is logits
    return logits


def test_open_thinking_start_uses_latest_unclosed_block():
    assert _open_thinking_start([START, 1, END, 2, START, 3], START, END) == 4
    assert _open_thinking_start([START, 1, END], START, END) == -1
    assert _open_thinking_start([1, 2, 3], START, END) == -1


def test_processor_does_not_cut_before_budget():
    logits = _apply([START], [7, 8, 9], budget=4)

    assert torch.isfinite(logits).all()


def test_processor_forces_newline_then_thinking_end_at_budget():
    newline_logits = _apply([START], [7, 8, 9, 10], budget=4)
    end_logits = _apply([START], [7, 8, 9, NEWLINE], budget=4)

    assert newline_logits[0, NEWLINE] == 0
    assert torch.isneginf(newline_logits).sum().item() == VOCAB_SIZE - 1
    assert end_logits[0, END] == 0
    assert torch.isneginf(end_logits).sum().item() == VOCAB_SIZE - 1


def test_processor_applies_after_a_closed_historical_thinking_block():
    logits = _apply([START, 1, END, 2, START], [7, 8], budget=2)

    assert logits[0, NEWLINE] == 0
    assert torch.isneginf(logits).sum().item() == VOCAB_SIZE - 1


def test_processor_accepts_sglang_array_token_storage():
    logits = torch.zeros(1, VOCAB_SIZE)
    req = SimpleNamespace(
        origin_input_ids=array("I", [START]),
        output_ids=array("I", [7, 8]),
    )

    Glm5ThinkingBudgetLogitProcessor()(logits, [{"thinking_budget": 2, "__req__": req}])

    assert logits[0, NEWLINE] == 0
    assert torch.isneginf(logits).sum().item() == VOCAB_SIZE - 1


@pytest.mark.parametrize(
    "params, error",
    [
        ({"thinking_budget": -1, "__req__": _request([START], [])}, "non-negative integer"),
        ({"thinking_budget": True, "__req__": _request([START], [])}, "non-negative integer"),
        ({"thinking_budget": 4}, "missing SGLang __req__"),
    ],
)
def test_processor_rejects_invalid_custom_params(params, error):
    with pytest.raises(ValueError, match=error):
        Glm5ThinkingBudgetLogitProcessor()(torch.zeros(1, VOCAB_SIZE), [params])


def test_processor_serialization_round_trip():
    serialized = Glm5ThinkingBudgetLogitProcessor.to_str()
    payload = __import__("json").loads(serialized)
    restored_class = dill.loads(bytes.fromhex(payload["callable"]))

    assert restored_class is Glm5ThinkingBudgetLogitProcessor
    assert restored_class.THINKING_START_TOKEN_ID == 154841
    assert restored_class.THINKING_END_TOKEN_ID == 154842
    assert restored_class.NEW_LINE_TOKEN_ID == 198
