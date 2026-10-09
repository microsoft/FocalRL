"""SGLang custom logit processors used by local evaluation clients."""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

import dill


def _open_thinking_start(ids: Sequence[int], start_id: int, end_id: int) -> int:
    """Return the start of the currently open thinking block, if one exists."""
    if isinstance(ids, (str, bytes)) or not isinstance(ids, Sequence):
        raise TypeError("ids must be an integer sequence")
    if not all(isinstance(token_id, int) for token_id in ids):
        raise TypeError("ids must contain only integers")
    if not all(isinstance(token_id, int) and token_id >= 0 for token_id in (start_id, end_id)):
        raise ValueError("thinking token IDs must be non-negative integers")
    if start_id == end_id:
        raise ValueError("thinking start and end token IDs must differ")

    for index in range(len(ids) - 1, -1, -1):
        if ids[index] == start_id:
            return index
        if ids[index] == end_id:
            return -1
    return -1


class Glm5ThinkingBudgetLogitProcessor:
    """Force-close an open GLM-5 thinking block once its token budget is used."""

    THINKING_START_TOKEN_ID = 154841
    THINKING_END_TOKEN_ID = 154842
    NEW_LINE_TOKEN_ID = 198

    @classmethod
    def to_str(cls) -> str:
        """Serialize this processor in the format expected by SGLang."""
        return json.dumps({"callable": dill.dumps(cls).hex()})

    def __call__(self, logits: Any, custom_param_list: list[dict[str, Any]] | None = None) -> Any:
        if not custom_param_list:
            return logits
        if getattr(logits, "ndim", None) != 2:
            raise ValueError(f"logits must be rank 2, got shape={getattr(logits, 'shape', None)!r}")
        if logits.shape[0] != len(custom_param_list):
            raise ValueError(
                f"logits batch size {logits.shape[0]} does not match custom params {len(custom_param_list)}"
            )

        for batch_index, params in enumerate(custom_param_list):
            if not isinstance(params, dict):
                raise TypeError(f"custom params at batch index {batch_index} must be a dict")
            thinking_budget = params.get("thinking_budget")
            if isinstance(thinking_budget, bool) or not isinstance(thinking_budget, int) or thinking_budget < 0:
                raise ValueError(
                    f"thinking_budget at batch index {batch_index} must be a non-negative integer, "
                    f"got {thinking_budget!r}"
                )
            req = params.get("__req__")
            if req is None:
                raise ValueError(f"custom params at batch index {batch_index} are missing SGLang __req__")
            origin_input_ids = getattr(req, "origin_input_ids", None)
            output_ids = getattr(req, "output_ids", None)
            if any(
                isinstance(token_ids, (str, bytes)) or not isinstance(token_ids, Sequence)
                for token_ids in (origin_input_ids, output_ids)
            ):
                raise TypeError(
                    f"SGLang request IDs at batch index {batch_index} must be integer sequences, "
                    f"got origin={type(origin_input_ids).__name__}, output={type(output_ids).__name__}"
                )
            current_ids = [*origin_input_ids, *output_ids]
            if not all(isinstance(token_id, int) for token_id in current_ids):
                raise TypeError(f"SGLang request IDs at batch index {batch_index} must contain only integers")

            start_index = _open_thinking_start(
                current_ids,
                self.THINKING_START_TOKEN_ID,
                self.THINKING_END_TOKEN_ID,
            )
            if start_index < 0 or len(current_ids) - start_index - 1 < thinking_budget:
                continue

            logits[batch_index, :] = -float("inf")
            forced_token_id = (
                self.THINKING_END_TOKEN_ID
                if output_ids and output_ids[-1] == self.NEW_LINE_TOKEN_ID
                else self.NEW_LINE_TOKEN_ID
            )
            if forced_token_id >= logits.shape[1]:
                raise ValueError(
                    f"forced token ID {forced_token_id} is outside logits vocabulary size {logits.shape[1]}"
                )
            logits[batch_index, forced_token_id] = 0.0

        return logits
