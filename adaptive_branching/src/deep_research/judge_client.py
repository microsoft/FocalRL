"""Minimal JSON client shared by the Value-Cliff locator and local judge."""

from __future__ import annotations

import asyncio
import json
import os
import re
from collections.abc import Callable
from typing import Any

from adaptive_branching.src.deep_research.judge_audit import audit_attempt, record_response

from adaptive_branching.src.deep_research.judge_config import (
    judge_settings,
    optional_enable_thinking,
    positive_float,
    positive_int,
    required_str,
)


class JudgeClient:
    def __init__(self, *, config_section: str) -> None:
        from openai import AsyncOpenAI

        config = judge_settings(config_section)
        self.model = required_str(config, "model")
        self.max_tokens = positive_int(config, "max_tokens")
        self.max_retries = int(config.get("max_retries", 3))
        if self.max_retries <= 0:
            raise ValueError(f"judge config max_retries must be positive, got {self.max_retries}")
        self.reasoning_effort = str(config.get("reasoning_effort") or "").strip()
        if self.reasoning_effort not in {"", "low", "medium", "high", "xhigh", "max"}:
            raise ValueError(f"invalid judge reasoning_effort {self.reasoning_effort!r}")
        self._responses_api = self.model.lower().startswith("gpt-")
        self.enable_thinking = optional_enable_thinking(config)
        self.sampling_params = {} if self._responses_api else _chat_sampling_params(config)
        self.token_budget = None
        if "max_input_tokens" in config:
            from adaptive_branching.src.deep_research.judge_token_budget import JudgeTokenBudget, local_tokenizer

            if self._responses_api:
                raise ValueError("local tokenizer budget is only supported for chat judges")
            self.token_budget = JudgeTokenBudget(
                local_tokenizer(required_str(config, "tokenizer_path")),
                max_input_tokens=config["max_input_tokens"],
                context_length=config["context_length"],
                max_output_tokens=self.max_tokens,
            )
        self._client = AsyncOpenAI(
            api_key=required_str(config, "api_key"),
            base_url=required_str(config, "base_url"),
            timeout=positive_float(config, "timeout"),
        )

    async def complete_json(
        self,
        system: str,
        user: str,
        required_keys: tuple[str, ...],
        tag: str,
        validate: Callable[[dict[str, Any]], Any],
    ) -> Any:
        last_error: Exception | None = None
        request_user = user
        for attempt in range(self.max_retries):
            budget = getattr(self, "token_budget", None)
            if budget is not None:
                request_user, before, after = await asyncio.to_thread(budget.fit, system, request_user)
                print(f"JUDGE_TOKEN_BUDGET tag={tag} before={before} after={after} limit={budget.limit}", flush=True)
            with audit_attempt(
                os.environ.get("JUDGE_AUDIT_DIR"),
                tag=tag,
                attempt=attempt + 1,
                request=(
                    {
                        "system": system,
                        "user": request_user,
                        "model": self.model,
                        "max_tokens": self.max_tokens,
                        "enable_thinking": self.enable_thinking,
                        "sampling_params": self.sampling_params,
                        "reasoning_effort": self.reasoning_effort,
                    }
                    if os.environ.get("JUDGE_AUDIT_DIR") is not None
                    else {}
                ),
            ) as audit:
                try:
                    content = await self._complete(system, request_user)
                except Exception as exc:  # noqa: BLE001 - retry bounded external calls
                    last_error = exc
                else:
                    try:
                        parsed = _parse_json_object(content)
                        missing = [key for key in required_keys if key not in parsed]
                        if missing:
                            raise ValueError(f"{tag} missing JSON keys: {missing}")
                        verdict = validate(parsed)
                        if audit is not None:
                            audit["validated"] = True
                        return verdict
                    except Exception as exc:  # noqa: BLE001 - bounded validation retry
                        last_error = exc
                        request_user = _validation_retry_prompt(user, exc)
                if audit is not None:
                    audit["validated"] = False
                    audit["error"] = f"{type(last_error).__name__}: {last_error}"
            if attempt + 1 < self.max_retries:
                print(
                    f"RETRY tag={tag} next_attempt={attempt + 2}/{self.max_retries} error={type(last_error).__name__}: {last_error}",
                    flush=True,
                )
                await asyncio.sleep(0.5 * (2**attempt))
        raise RuntimeError(f"{tag} failed after {self.max_retries} attempts: {last_error}") from last_error

    async def _complete(self, system: str, user: str) -> str:
        if self._responses_api:
            kwargs: dict[str, Any] = {}
            if self.reasoning_effort:
                kwargs["reasoning"] = {"effort": self.reasoning_effort}
            response = await self._client.responses.create(
                model=self.model,
                instructions=system,
                input=user,
                max_output_tokens=self.max_tokens,
                **kwargs,
            )
            record_response(response)
            if str(getattr(response, "status", "") or "") == "incomplete":
                raise ValueError("judge response was incomplete")
            content = str(getattr(response, "output_text", "") or "")
        else:
            kwargs = {
                "temperature": self.sampling_params["temperature"],
                "top_p": self.sampling_params["top_p"],
                "presence_penalty": self.sampling_params["presence_penalty"],
                "extra_body": {
                    "top_k": self.sampling_params["top_k"],
                    "min_p": self.sampling_params["min_p"],
                    "repetition_penalty": self.sampling_params["repetition_penalty"],
                },
            }
            if self.enable_thinking is not None:
                kwargs["extra_body"]["chat_template_kwargs"] = {"enable_thinking": self.enable_thinking}
            if self.reasoning_effort:
                kwargs["reasoning_effort"] = self.reasoning_effort
            response = await self._client.chat.completions.create(
                model=self.model,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                max_tokens=self.max_tokens,
                **kwargs,
            )
            record_response(response)
            if not response.choices:
                raise ValueError("judge response has no choices")
            choice = response.choices[0]
            if str(getattr(choice, "finish_reason", "") or "") == "length":
                raise ValueError("judge response exhausted max_tokens")
            content = str(choice.message.content or "")
        if not content.strip():
            raise ValueError("judge response is empty")
        return content


def _chat_sampling_params(config: dict[str, Any]) -> dict[str, int | float]:
    """Validate explicit OpenAI-compatible sampling parameters for local judges."""

    temperature = _bounded_number(config, "temperature", lower=0.0, upper=2.0, lower_inclusive=False)
    top_p = _bounded_number(config, "top_p", lower=0.0, upper=1.0, lower_inclusive=False)
    top_k = positive_int(config, "top_k")
    min_p = _bounded_number(config, "min_p", lower=0.0, upper=1.0)
    presence_penalty = _bounded_number(config, "presence_penalty", lower=-2.0, upper=2.0)
    repetition_penalty = _bounded_number(
        config,
        "repetition_penalty",
        lower=0.0,
        upper=None,
        lower_inclusive=False,
    )
    return {
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "min_p": min_p,
        "presence_penalty": presence_penalty,
        "repetition_penalty": repetition_penalty,
    }


def _bounded_number(
    config: dict[str, Any],
    key: str,
    *,
    lower: float,
    upper: float | None,
    lower_inclusive: bool = True,
) -> float:
    raw = config.get(key)
    if isinstance(raw, bool):
        raise ValueError(f"judge config {key!r} must be a number, got {raw!r}")
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"judge config {key!r} must be a number, got {raw!r}") from exc
    lower_ok = value >= lower if lower_inclusive else value > lower
    upper_ok = upper is None or value <= upper
    if not lower_ok or not upper_ok:
        left = "[" if lower_inclusive else "("
        right = "]" if upper is not None else ")"
        upper_text = str(upper) if upper is not None else "inf"
        raise ValueError(f"judge config {key!r} must be in {left}{lower}, {upper_text}{right}, got {value}")
    return value


def _parse_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1)
    start = text.find("{")
    if start < 0:
        raise ValueError("judge response contains no JSON object")
    try:
        parsed, _ = json.JSONDecoder().raw_decode(text[start:])
    except json.JSONDecodeError as exc:
        raise ValueError(f"judge returned invalid JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ValueError(f"judge JSON must be an object, got {type(parsed).__name__}")
    return parsed


def _validation_retry_prompt(original_user: str, error: Exception) -> str:
    if not isinstance(original_user, str) or not original_user.strip():
        raise ValueError("original_user must be a non-empty string")
    message = str(error).strip()
    if not message:
        message = type(error).__name__
    message = message[:2000]
    return f"{original_user}\n\n## Required correction\nYour previous response failed strict validation: {message}\nReturn a corrected JSON object only. Preserve the requested schema and obey every stated range and invariant."
