"""Opt-in complete-chat token guard for local judges; never remove actions or rubrics."""

from __future__ import annotations

import re
from functools import cache
from pathlib import Path


@cache
def local_tokenizer(path: str):
    if not isinstance(path, str) or not (Path(path) / "tokenizer_config.json").is_file():
        raise ValueError(f"judge tokenizer must be a local checkpoint: {path!r}")
    from miles.utils.processing_utils import load_tokenizer

    return load_tokenizer(path, trust_remote_code=True, local_files_only=True)


class JudgeTokenBudget:
    def __init__(self, tokenizer, *, max_input_tokens: int, context_length: int, max_output_tokens: int):
        for name, value in (
            ("max_input_tokens", max_input_tokens),
            ("context_length", context_length),
            ("max_output_tokens", max_output_tokens),
        ):
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer: {value!r}")
        if max_input_tokens + max_output_tokens > context_length:
            raise ValueError("judge input plus output budget exceeds context_length")
        if not callable(getattr(tokenizer, "apply_chat_template", None)):
            raise TypeError("tokenizer must implement apply_chat_template")
        self.tokenizer = tokenizer
        self.limit = max_input_tokens

    def count(self, system: str, user: str) -> int:
        if not isinstance(system, str) or not system.strip() or not isinstance(user, str) or not user.strip():
            raise ValueError("judge system and user prompts must be nonempty strings")
        ids = self.tokenizer.apply_chat_template(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=True,
            return_dict=False,
        )
        if not isinstance(ids, list) or not ids:
            raise ValueError("judge tokenizer returned invalid token IDs")
        return len(ids)

    def fit(self, system: str, user: str) -> tuple[str, int, int]:
        initial = self.count(system, user)
        current = initial
        for _ in range(12):
            if current <= self.limit:
                return user, initial, current
            reduced = shrink_observations(user, min(0.75, 0.9 * self.limit / current))
            if reduced == user:
                raise ValueError(
                    f"judge protected content cannot fit: input_tokens={current}, limit={self.limit}; "
                    "refusing to discard actions, turn IDs, question, or rubric"
                )
            user = reduced
            current = self.count(system, user)
        raise ValueError(f"judge token fitting did not converge: input_tokens={current}, limit={self.limit}")


def shrink_observations(prompt: str, ratio: float) -> str:
    """Shorten only rendered reasoning/observations, keeping structural lines and actions."""
    if not isinstance(prompt, str) or not prompt or not isinstance(ratio, (int, float)) or not 0 < ratio < 1:
        raise ValueError("nonempty prompt and ratio in (0, 1) required")
    # Renderers start each new message, turn and enclosing section at one of these boundaries.
    parts = re.split(r"(?m)(?=^(?:#{2,3} |\[|</?(?:prefix_context|local_continuation|recovery_rubric)\b))", prompt)
    changed = []
    for part in parts:
        match = re.match(r"(\[(?:reasoning|following_observation:[^\]\n]+)\] )([\s\S]*)", part)
        if match is None or len(match[2]) <= 256:
            changed.append(part)
            continue
        body = match[2]
        target = max(256, int(len(body) * ratio))
        marker = "\n... [JUDGE_TOKEN_BUDGET: observation/reasoning shortened] ...\n"
        keep = target - len(marker)
        assert keep > 0 and target < len(body)
        changed.append(match[1] + body[: (keep + 1) // 2] + marker + body[-(keep // 2) :])
    return "".join(changed)
