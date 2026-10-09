"""Sample ReAct-style OpenAI tool agent for Miles session-server rollout."""

from __future__ import annotations

import asyncio
import copy
import json
import logging
import os
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import httpx

from adaptive_branching.src.deep_research.branch_sampling import BranchDataError, entropy_score, selection_policy
from adaptive_branching.tools.deep_research.microsoft_browse_tool import FETCH_URL_TOOL_SCHEMA, MicrosoftBrowseTool
from adaptive_branching.tools.deep_research.microsoft_search_tool import SEARCH_TOOL_SCHEMA, MicrosoftSearchTool
from adaptive_branching.tools.deep_research.prismcrawl_search_tool import PrismCrawlSearchTool
from adaptive_branching.tools.deep_research.serper_search_tool import SerperSearchTool
from adaptive_branching.tools.deep_research.serpent_search_tool import SerpentSearchTool

logger = logging.getLogger(__name__)

TOOLS = [SEARCH_TOOL_SCHEMA, FETCH_URL_TOOL_SCHEMA]

SYSTEM_PROMPT = (
    "In this environment you have access to a set of tools you can use to answer the user's question.\n"
    "\n"
    "You only have access to the tools provided. You can use multiple tools per message, "
    "and will receive the results of those tools in the user's next response. "
    "You use tools step-by-step to accomplish a given task. \n"
    "# General Objective\n"
    "\n"
    "You accomplish a given task iteratively, breaking it down into clear steps and working through them methodically."
)


def _final_answer_prompt(task_description: str) -> str:
    return (
        "Summarize the above conversation, and output the FINAL ANSWER to the original question.\n\n"
        "If a clear answer has already been provided earlier in the conversation, "
        "do not rethink or recalculate it — "
        "simply extract that answer.\n"
        "If a definitive answer could not be determined, "
        "make a well-informed educated guess based on the conversation.\n\n"
        "The original question is repeated here for reference:\n\n"
        f'"{task_description}"\n\n'
        "Your final answer MUST strictly follow any formatting instructions in the original question — "
        "such as alphabetization, sequencing, units, rounding, decimal places, etc.\n\n"
        "You must absolutely not perform any MCP tool call, tool invocation, search, scrape, "
        "code execution, or similar actions.\n"
        "You can only answer the original question based on the information already retrieved "
        "and your own internal knowledge.\n"
        "If you attempt to call any tool, it will be considered a mistake."
    )


FINAL_ANSWER_PROMPT_MARGIN_TOKENS = 1024
TOOL_RESULT_OMITTED_PLACEHOLDER = "Tool result is omitted to save tokens."
_CHAT_COMPLETION_MAX_ATTEMPTS = 3
_CHAT_COMPLETION_RETRY_DELAY_SECONDS = 0.5
_CHAT_COMPLETION_RETRY_STATUSES = {500, 502, 503, 504}
_TOOL_CALL_TEXT_RE = re.compile(
    r"<\s*/?\s*tool_call\b|"
    r"<\s*/?\s*function\b|"
    r"\btool_calls\b|"
    r"\bsearch_query\b|"
    r"\bfetch_url\b|"
    r"\bbrowser\.(?:search|open)\b",
    re.IGNORECASE,
)
_CONTEXT_LENGTH_INPUT_RE = re.compile(r"([\d,]+)\s+tokens from the input messages", re.IGNORECASE)
_CONTEXT_LENGTH_MAX_RE = re.compile(r"maximum context length of\s+([\d,]+)\s+tokens", re.IGNORECASE)
_SEARCH_QUERY_REPEAT_JACCARD_THRESHOLD = 0.65

_OC_ENV_RE = re.compile(r"\$\{oc\.env:([^,}]+)(?:,([^}]*))?\}")


async def _post_chat_completion(
    client: httpx.AsyncClient,
    url: str,
    payload: dict[str, Any],
    headers: Mapping[str, str] | None = None,
) -> httpx.Response:
    for attempt in range(_CHAT_COMPLETION_MAX_ATTEMPTS):
        try:
            response = await client.post(url, json=payload, headers=headers)
        except (httpx.TimeoutException, httpx.TransportError) as exc:
            if attempt + 1 >= _CHAT_COMPLETION_MAX_ATTEMPTS:
                raise RuntimeError(
                    f"chat completion transport failed after {_CHAT_COMPLETION_MAX_ATTEMPTS} attempts: "
                    f"{type(exc).__name__}: {exc}"
                ) from exc
            logger.warning(
                "chat completion transport error, retrying (%d/%d): %s",
                attempt + 1,
                _CHAT_COMPLETION_MAX_ATTEMPTS,
                exc,
            )
            await asyncio.sleep(_CHAT_COMPLETION_RETRY_DELAY_SECONDS * (2**attempt))
            continue

        if response.status_code == 200:
            return response
        if response.status_code in _CHAT_COMPLETION_RETRY_STATUSES and attempt + 1 < _CHAT_COMPLETION_MAX_ATTEMPTS:
            logger.warning(
                "chat completion HTTP %d, retrying (%d/%d): %s",
                response.status_code,
                attempt + 1,
                _CHAT_COMPLETION_MAX_ATTEMPTS,
                response.text[:200],
            )
            await asyncio.sleep(_CHAT_COMPLETION_RETRY_DELAY_SECONDS * (2**attempt))
            continue
        return response

    raise RuntimeError("chat completion retry loop exited unexpectedly")


def _chat_completions_url(base_url: str) -> str:
    url = str(base_url).strip().rstrip("/")
    if not url:
        raise ValueError("base_url must be non-empty")

    path = urlsplit(url).path.rstrip("/")
    if path.endswith("/chat/completions"):
        return url
    if path.endswith("/v1"):
        return f"{url}/chat/completions"
    return f"{url}/v1/chat/completions"


def _responses_url(base_url: str) -> str:
    url = str(base_url).strip().rstrip("/")
    if not url:
        raise ValueError("base_url must be non-empty")

    path = urlsplit(url).path.rstrip("/")
    if path.endswith("/responses"):
        return url
    if path.endswith("/v1"):
        return f"{url}/responses"
    return f"{url}/v1/responses"


def _agent_api_mode(metadata: dict[str, Any]) -> str:
    mode = str(metadata.get("agent_api_mode") or "chat").strip().lower()
    if mode not in {"chat", "responses"}:
        raise ValueError("agent_api_mode must be 'chat' or 'responses'")
    return mode


def _responses_input(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for message in messages:
        role = str(message.get("role") or "")
        content = message.get("content")
        if role == "tool":
            items.append(
                {
                    "type": "function_call_output",
                    "call_id": str(message.get("tool_call_id") or ""),
                    "output": str(content or ""),
                }
            )
            continue
        if role not in {"system", "user", "assistant"}:
            raise ValueError(f"unsupported Responses API message role: {role!r}")
        responses_output = message.get("_responses_output")
        if isinstance(responses_output, list):
            items.extend(responses_output)
            continue
        if content:
            items.append({"role": role, "content": content})
        for tool_call in message.get("tool_calls") or []:
            function = tool_call.get("function") or {}
            items.append(
                {
                    "type": "function_call",
                    "call_id": str(tool_call.get("id") or ""),
                    "name": str(function.get("name") or ""),
                    "arguments": str(function.get("arguments") or "{}"),
                }
            )
    return items


def _responses_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    converted: list[dict[str, Any]] = []
    for tool in tools:
        function = tool.get("function") or {}
        converted.append(
            {
                "type": "function",
                "name": str(function.get("name") or ""),
                "description": str(function.get("description") or ""),
                "parameters": function.get("parameters") or {},
            }
        )
    return converted


def _responses_payload(chat_payload: dict[str, Any]) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": chat_payload["model"],
        "input": _responses_input(chat_payload["messages"]),
    }
    if chat_payload.get("tools"):
        payload["tools"] = _responses_tools(chat_payload["tools"])
        payload["tool_choice"] = chat_payload.get("tool_choice", "auto")
    if chat_payload.get("max_tokens") is not None:
        payload["max_output_tokens"] = chat_payload["max_tokens"]
    reasoning_effort = chat_payload.get("reasoning_effort")
    if reasoning_effort:
        payload["reasoning"] = {"effort": reasoning_effort}
    return payload


def _chat_data_from_responses(data: dict[str, Any]) -> dict[str, Any]:
    content_parts: list[str] = []
    tool_calls: list[dict[str, Any]] = []
    for item in data.get("output") or []:
        item_type = item.get("type")
        if item_type == "message":
            for part in item.get("content") or []:
                if part.get("type") == "output_text" and part.get("text"):
                    content_parts.append(str(part["text"]))
        elif item_type == "function_call":
            tool_calls.append(
                {
                    "id": str(item.get("call_id") or item.get("id") or ""),
                    "type": "function",
                    "function": {
                        "name": str(item.get("name") or ""),
                        "arguments": str(item.get("arguments") or "{}"),
                    },
                }
            )

    status = str(data.get("status") or "")
    incomplete_reason = str((data.get("incomplete_details") or {}).get("reason") or "")
    if tool_calls:
        finish_reason = "tool_calls"
    elif status == "incomplete" and incomplete_reason == "max_output_tokens":
        finish_reason = "length"
    else:
        finish_reason = "stop"

    message: dict[str, Any] = {"role": "assistant", "content": "\n".join(content_parts)}
    message["_responses_output"] = data.get("output") or []
    if tool_calls:
        message["tool_calls"] = tool_calls
    usage = data.get("usage") if isinstance(data.get("usage"), dict) else {}
    return {
        "choices": [{"message": message, "finish_reason": finish_reason}],
        "usage": {
            "prompt_tokens": usage.get("input_tokens"),
            "completion_tokens": usage.get("output_tokens"),
            "total_tokens": usage.get("total_tokens"),
        },
    }


def _chat_completion_headers(metadata: dict[str, Any]) -> dict[str, str] | None:
    headers: dict[str, str] = {}

    raw_headers = metadata.get("agent_chat_headers") or os.getenv("AGENT_CHAT_HEADERS_JSON")
    if raw_headers:
        if isinstance(raw_headers, str):
            parsed = json.loads(raw_headers)
        else:
            parsed = raw_headers
        if not isinstance(parsed, dict):
            raise ValueError("agent_chat_headers/AGENT_CHAT_HEADERS_JSON must be a JSON object")
        headers.update({str(k): str(v) for k, v in parsed.items() if v is not None})

    api_key = metadata.get("agent_chat_api_key") or os.getenv("AGENT_CHAT_API_KEY")
    if api_key:
        header_name = str(
            metadata.get("agent_chat_api_key_header") or os.getenv("AGENT_CHAT_API_KEY_HEADER") or "Authorization"
        )
        value = str(api_key)
        if header_name.lower() == "authorization" and not value.lower().startswith("bearer "):
            value = f"Bearer {value}"
        headers[header_name] = value

    return headers or None


def _load_tools_config() -> dict[str, Any]:
    default_path = Path(__file__).resolve().parents[2] / "config" / "microsoft_gaia_tools.yaml"
    config_path = Path(os.getenv("AGENT_TOOLS_CONFIG", str(default_path))).expanduser()
    if not config_path.is_file():
        raise FileNotFoundError(f"agent tools config not found: {config_path}")

    import yaml

    loaded = yaml.safe_load(config_path.read_text())
    if not isinstance(loaded, dict):
        raise ValueError(f"agent tools config must contain a mapping: {config_path}")
    for section in ("search", "fetch_url"):
        if not isinstance(loaded.get(section), dict):
            raise ValueError(f"agent tools config requires a {section!r} mapping")
    return loaded


def _resolve_env(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _resolve_env(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_resolve_env(item) for item in value]
    if not isinstance(value, str):
        return value

    def replace(match: re.Match[str]) -> str:
        name = match.group(1).strip()
        default = match.group(2)
        if name in os.environ:
            return os.environ[name]
        if default is not None:
            return default.strip()
        raise ValueError(f"required environment variable {name!r} is not set")

    return _OC_ENV_RE.sub(replace, value)


TOOL_CONFIG = _resolve_env(_load_tools_config())


def _build_search_tool(config: dict[str, Any]) -> MicrosoftSearchTool | SerpentSearchTool | SerperSearchTool | PrismCrawlSearchTool:
    provider = str(config.get("provider", "microsoft")).strip().lower()
    if provider == "microsoft":
        return MicrosoftSearchTool(config=config)
    if provider == "serpent":
        return SerpentSearchTool(config=config)
    if provider == "serper":
        return SerperSearchTool(config=config)
    if provider == "prismcrawl":
        return PrismCrawlSearchTool(config=config)
    raise ValueError(
        f"unsupported search provider: {provider!r}; expected microsoft, serpent, serper, or prismcrawl"
    )


SEARCH_TOOL = _build_search_tool(TOOL_CONFIG["search"])
BROWSE_TOOL = MicrosoftBrowseTool(config=TOOL_CONFIG["fetch_url"])


async def run(
    base_url: str,
    prompt: Any,
    request_kwargs: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
    **kwargs,
) -> dict[str, Any] | None:
    """Run one tool-calling trajectory through the Miles session server.

    This function is intended for:
    --custom-generate-function-path miles.rollout.generate_hub.agentic_tool_call.generate
    --custom-agent-function-path adaptive_branching.src.deep_research.agent.run
    """
    request_kwargs = request_kwargs or {}
    metadata = metadata or {}
    messages = _messages_from_prompt(prompt, system_prompt=metadata.get("agent_system_prompt"))
    original_question = _original_question_from_messages(messages)

    rk = {k: v for k, v in request_kwargs.items() if k not in {"messages", "tools", "tool_choice"}}
    api_mode = _agent_api_mode(metadata)
    capture_raw_responses = _truthy(metadata.get("agent_capture_raw_responses", False))
    if capture_raw_responses and api_mode != "responses":
        raise ValueError("agent_capture_raw_responses requires agent_api_mode='responses'")
    endpoint_base = str(
        metadata.get("agent_chat_completions_url") or os.getenv("AGENT_CHAT_COMPLETIONS_URL") or base_url
    )
    completion_url = _responses_url(endpoint_base) if api_mode == "responses" else _chat_completions_url(endpoint_base)
    chat_headers = _chat_completion_headers(metadata)
    max_turns = int(metadata.get("agent_max_turns") or os.getenv("AGENT_MAX_TURNS", 30))
    max_seq_len = _optional_int(metadata.get("max_seq_len"))
    chat_timeout = float(os.getenv("AGENT_CHAT_TIMEOUT", 1800))
    context_reserve_tokens = _optional_int(os.getenv("AGENT_CONTEXT_RESERVE_TOKENS")) or 32768
    history_mode = _history_mode(metadata)
    keep_tool_results = _optional_int(metadata.get("agent_keep_tool_results")) or 5
    return_messages = not _falsey(metadata.get("agent_return_messages", True))
    return_logprobs = _truthy(metadata.get("agent_logprobs", True))
    branch_policy = selection_policy()
    capture_branch = branch_policy != "value_cliff" and not metadata.get("ab_local_rollout")
    capture_entropy = capture_branch and branch_policy == "entropy_delta_max"
    if capture_entropy and (api_mode != "chat" or not return_logprobs):
        raise BranchDataError("entropy branching requires chat API with logprobs enabled")
    branch_history = copy.deepcopy(messages) if capture_branch else None
    branch_entropies = {}
    force_final_on_max_turns = not _falsey(metadata.get("agent_force_final_on_max_turns", True))
    force_final_on_context_reserve = not _falsey(
        metadata.get("agent_force_final_on_context_reserve", True)
    )
    tool_call_text_retries = int(metadata.get("agent_tool_call_text_retries") or 0)
    if tool_call_text_retries < 0:
        raise ValueError("agent_tool_call_text_retries must be >= 0")

    metrics: dict[str, Any] = {
        "agent_turns": 0,
        "agent_tool_call_count": 0,
        "agent_tool_call_failure_rate": 0.0,
        "agent_tool_unit_count": 0,
        "agent_tool_unit_success_count": 0,
        "agent_tool_unit_success_rate": 1.0,
        "agent_search_count": 0,
        "agent_search_query_count": 0,
        "agent_search_query_repeat_count": 0,
        "agent_search_cache_hit_count": 0,
        "agent_search_retry_429_count": 0,
        "agent_search_non_200_attempt_count": 0,
        "agent_fetch_url_count": 0,
        "agent_fetch_url_repeat_count": 0,
        # 进入“快到上下文上限，所以准备强制回答”逻辑的比例；触发 guard，
        # 不代表最终一定成功拿到答案。
        "agent_context_reserve_hit": False,
        # A large tool observation can cross the reserve boundary between two
        # model calls.  If the server reports the exact prompt size in a 400,
        # recover by issuing a dynamically capped forced-final request.
        "agent_context_length_recovery": False,
        # 强制 final answer 请求成功返回的比例；成功发出“不许再用工具，
        # 立刻回答”的最后一次 LLM 请求并拿到 200。
        "agent_forced_final_answer": False,
        # 正常工具循环用满 max_turns 后，进入 Apodex 风格 summarize/final-answer 兜底的比例。
        "agent_max_turns_hit": False,
        # Local process rollout may stop at a short horizon without asking for a
        # synthetic final answer. This is distinct from the normal max-turns
        # forced-final path above.
        "agent_local_horizon_hit": False,
        # 触发 guard 了，但剩余 token 连 final answer 请求都不够发，所以跳过。
        "agent_forced_final_answer_skipped_no_budget": False,
        # 强制 final answer 请求 HTTP 非 200 或返回空内容。
        "agent_forced_final_answer_failed": False,
        # 最终保存的 answer 是否非空；作为 final-answer 质量指标的分母。
        "agent_final_answer_present": False,
        # 最终保存的 answer 是否疑似文本版 tool call，例如 <tool_call> 或 search/fetch 调用文本。
        "agent_final_answer_tool_call_like": False,
        # 强制 final answer 那次返回的内容是否疑似文本版 tool call；聚合分母是全体样本。
        # forced 内部污染率可用它除以 agent_forced_final_answer。
        "agent_forced_final_answer_tool_call_like": False,
    }
    if max_seq_len is not None:
        metrics["agent_max_seq_len"] = max_seq_len
    final_content = ""
    last_finish_reason = ""
    turn_offset = metadata.get("agent_turn_offset", 0)
    session_tokens = metadata.get("agent_initial_session_tokens", 0)
    if isinstance(turn_offset, bool) or not isinstance(turn_offset, int) or not 0 <= turn_offset < max_turns:
        raise ValueError(f"agent_turn_offset={turn_offset!r} must be in [0, {max_turns})")
    if isinstance(session_tokens, bool) or not isinstance(session_tokens, int) or session_tokens < 0:
        raise ValueError("agent_initial_session_tokens must be a non-negative integer")
    if turn_offset:
        observed_turns = sum(message.get("role") == "assistant" for message in messages)
        if observed_turns != turn_offset or messages[-1].get("role") != "tool":
            raise ValueError(f"continuation offset={turn_offset} disagrees with prefix turns={observed_turns}/tail")
        metrics["agent_turns"] = turn_offset
        metrics["agent_continuation_prefix_turns"] = turn_offset
    search_query_history: list[frozenset[str]] = []
    fetch_url_history: set[str] = set()
    raw_responses: list[dict[str, Any]] = []

    def capture_response(
        data: dict[str, Any],
        *,
        agent_turn: int,
        request_kind: str,
        retry_index: int = 0,
    ) -> None:
        if not capture_raw_responses:
            return
        if not isinstance(data, dict):
            raise TypeError(f"Responses API payload must be an object, got {type(data).__name__}")
        if isinstance(agent_turn, bool) or not isinstance(agent_turn, int) or agent_turn <= 0:
            raise ValueError(f"agent_turn must be a positive integer, got {agent_turn!r}")
        if request_kind not in {"agent_turn", "forced_final"}:
            raise ValueError(f"unsupported raw response request_kind: {request_kind!r}")
        if isinstance(retry_index, bool) or not isinstance(retry_index, int) or retry_index < 0:
            raise ValueError(f"retry_index must be a non-negative integer, got {retry_index!r}")
        raw_responses.append(
            {
                "request_index": len(raw_responses) + 1,
                "agent_turn": agent_turn,
                "request_kind": request_kind,
                "retry_index": retry_index,
                "response": data,
            }
        )

    async with httpx.AsyncClient(timeout=chat_timeout) as client:
        async def force_final_answer(reason: str) -> None:
            nonlocal final_content, last_finish_reason, session_tokens

            if reason == "context_reserve":
                metrics["agent_context_reserve_hit"] = True
                metrics["agent_context_reserve_tokens"] = context_reserve_tokens
            elif reason == "max_turns":
                metrics["agent_max_turns_hit"] = True

            metrics["agent_pre_forced_session_tokens"] = session_tokens
            final_kwargs = _final_request_kwargs(rk, max_seq_len, session_tokens)
            if final_kwargs is None:
                last_finish_reason = "forced_answer_skipped_no_budget"
                metrics["agent_forced_final_answer_skipped_no_budget"] = True
                metrics["agent_forced_final_answer_reason"] = reason
                return

            final_messages = [
                *messages,
                {"role": "user", "content": _final_answer_prompt(original_question)},
            ]
            final_payload = {
                "messages": final_messages,
                **final_kwargs,
                "logprobs": return_logprobs,
                "stream": False,
            }

            request_payload = _responses_payload(final_payload) if api_mode == "responses" else final_payload
            response = await _post_chat_completion(client, completion_url, request_payload, headers=chat_headers)
            if response.status_code != 200:
                last_finish_reason = "forced_answer_fail"
                metrics["agent_forced_final_answer_failed"] = True
                metrics["agent_forced_final_answer_failed_status"] = response.status_code
                metrics["agent_forced_final_answer_reason"] = reason
                return

            messages.append(final_messages[-1])
            data = response.json()
            if api_mode == "responses":
                capture_response(
                    data,
                    agent_turn=int(metrics["agent_turns"]) + 1,
                    request_kind="forced_final",
                )
                data = _chat_data_from_responses(data)
            choice = data["choices"][0]
            assistant_msg = choice["message"]
            last_finish_reason = str(choice.get("finish_reason") or "")
            messages.append(assistant_msg)
            metrics["agent_forced_final_answer"] = True
            metrics["agent_forced_final_answer_reason"] = reason

            usage = data.get("usage") if isinstance(data.get("usage"), dict) else {}
            meta_info = choice.get("meta_info") if isinstance(choice.get("meta_info"), dict) else {}
            session_tokens = _session_tokens_from_response(usage, meta_info, session_tokens)
            metrics["agent_session_tokens"] = session_tokens

            content = assistant_msg.get("content") or ""
            final_content = content
            metrics["agent_forced_final_answer_tool_call_like"] = _looks_like_tool_call_text(content)
            if not content:
                last_finish_reason = "forced_answer_empty"
                metrics["agent_forced_final_answer_failed"] = True

        for turn in range(turn_offset + 1, max_turns + 1):
            if _should_force_final_answer(max_seq_len, session_tokens, context_reserve_tokens):
                if force_final_on_context_reserve:
                    await force_final_answer("context_reserve")
                else:
                    metrics["agent_context_reserve_hit"] = True
                    metrics["agent_context_reserve_tokens"] = context_reserve_tokens
                    metrics["agent_local_horizon_hit"] = True
                    last_finish_reason = "local_context_reserve"
                break

            payload = {
                "messages": messages,
                "tools": TOOLS,
                "tool_choice": "auto",
                **rk,
                # Miles Session Server/TITO 构造训练轨迹所必需
                "logprobs": return_logprobs,
                "stream": False,
            }
            if capture_entropy:
                payload["top_logprobs"] = 10
                payload["return_meta_info"] = True
            request_payload = _responses_payload(payload) if api_mode == "responses" else payload
            tool_call_text_retry_attempt = 0
            while True:
                response = await _post_chat_completion(
                    client,
                    completion_url,
                    request_payload,
                    headers=chat_headers,
                )
                if response.status_code != 200:
                    break

                data = response.json()
                if api_mode == "responses":
                    capture_response(
                        data,
                        agent_turn=turn,
                        request_kind="agent_turn",
                        retry_index=tool_call_text_retry_attempt,
                    )
                    data = _chat_data_from_responses(data)
                choice = data["choices"][0]
                assistant_msg = choice["message"]
                last_finish_reason = str(choice.get("finish_reason") or "")
                content = assistant_msg.get("content") or ""
                tool_calls = assistant_msg.get("tool_calls") or []

                malformed_text_tool_call = (
                    tool_call_text_retries > 0 and last_finish_reason not in {"abort", "length"}
                    and not tool_calls
                    and _looks_like_unparsed_tool_call(content)
                )
                if malformed_text_tool_call and tool_call_text_retry_attempt < tool_call_text_retries:
                    tool_call_text_retry_attempt += 1
                    metrics["agent_tool_call_text_retry_count"] = (
                        metrics.get("agent_tool_call_text_retry_count", 0) + 1
                    )
                    logger.warning(
                        "textual tool call was not parsed; resampling the same turn (%d/%d)",
                        tool_call_text_retry_attempt,
                        tool_call_text_retries,
                    )
                    continue
                if malformed_text_tool_call and tool_call_text_retries:
                    metrics["agent_tool_call_text_retry_exhausted"] = True
                break

            if response.status_code != 200:
                context_budget = _context_length_error_budget(response)
                if context_budget is not None:
                    server_max_tokens, input_tokens = context_budget
                    max_seq_len = min(max_seq_len or server_max_tokens, server_max_tokens)
                    session_tokens = input_tokens
                    metrics["agent_context_length_recovery"] = True
                    metrics["agent_context_length_recovery_input_tokens"] = input_tokens
                    metrics["agent_context_length_recovery_server_max_tokens"] = server_max_tokens
                    metrics["agent_session_tokens"] = session_tokens
                    if force_final_on_context_reserve:
                        await force_final_answer("context_reserve")
                    else:
                        metrics["agent_context_reserve_hit"] = True
                        metrics["agent_context_reserve_tokens"] = context_reserve_tokens
                        metrics["agent_local_horizon_hit"] = True
                        last_finish_reason = "local_context_reserve"
                    break
                raise RuntimeError(f"chat completion failed ({response.status_code}): {response.text[:500]}")

            if capture_entropy and last_finish_reason not in {"abort", "length"}:
                try:
                    branch_entropies[str(turn)] = entropy_score(
                        choice.get("meta_info"), metadata.get("ab_entropy_vocab_size")
                    )
                except BranchDataError as exc:
                    raise BranchDataError(f"branch entropy at assistant turn {turn}: {exc}") from exc
            if capture_branch:
                branch_history.append(copy.deepcopy(assistant_msg))
            messages.append(assistant_msg)
            metrics["agent_turns"] = turn

            usage = data.get("usage") if isinstance(data.get("usage"), dict) else {}
            meta_info = choice.get("meta_info") if isinstance(choice.get("meta_info"), dict) else {}
            session_tokens = _session_tokens_from_response(usage, meta_info, session_tokens)
            metrics["agent_session_tokens"] = session_tokens

            if content:
                final_content = content

            # An aborted or length-truncated assistant response is not a
            # completed action.  Even if SGLang managed to parse a tool call
            # from the partial payload, executing it would create another turn
            # after a terminal TITO record and make the trajectory invalid.
            if last_finish_reason in {"abort", "length"}:
                break

            if not tool_calls:
                break

            search_query_count, search_query_repeat_count = _count_search_queries_and_repeats(
                tool_calls,
                search_query_history,
            )
            metrics["agent_search_query_count"] += search_query_count
            metrics["agent_search_query_repeat_count"] += search_query_repeat_count
            metrics["agent_fetch_url_repeat_count"] += _count_repeated_fetch_urls(
                tool_calls,
                fetch_url_history,
            )
            tool_messages, tool_metrics = await _execute_tool_calls(tool_calls)
            if capture_branch:
                branch_history.extend(copy.deepcopy(tool_messages))
            messages.extend(tool_messages)
            if history_mode == "keep5":
                metrics["agent_history_keep5_omitted_tool_results"] = _keep_last_tool_results(
                    messages,
                    keep_tool_results,
                )
            _merge_numeric_metrics(metrics, tool_metrics)
        else:
            if max_turns > 0:
                if force_final_on_max_turns:
                    await force_final_answer("max_turns")
                else:
                    metrics["agent_local_horizon_hit"] = True
                    last_finish_reason = "local_horizon"

    tool_call_count = metrics.get("agent_tool_call_count", 0)
    tool_call_failures = metrics.pop("tool_call_failures", 0)
    if tool_call_count:
        metrics["agent_tool_call_failure_rate"] = tool_call_failures / tool_call_count

    tool_unit_count = metrics.pop("tool_unit_count", 0)
    tool_unit_successes = metrics.pop("tool_unit_successes", 0)
    metrics["agent_tool_unit_count"] = tool_unit_count
    metrics["agent_tool_unit_success_count"] = tool_unit_successes
    metrics["agent_tool_unit_success_rate"] = tool_unit_successes / tool_unit_count if tool_unit_count else 1.0

    if capture_branch:
        metrics["ab_branch_history"] = branch_history
        metrics["ab_branch_entropies"] = branch_entropies
    final_answer = _strip_thinking(final_content)
    metrics["agent_finished"] = bool(final_content) and last_finish_reason not in {
        "abort",
        "length",
        "forced_answer_fail",
        "forced_answer_empty",
        "forced_answer_skipped_no_budget",
    }
    metrics["agent_last_finish_reason"] = last_finish_reason
    metrics["agent_final_content_preview"] = final_content[-1000:]
    metrics["agent_final_answer"] = final_answer
    metrics["agent_final_answer_present"] = bool(final_answer)
    metrics["agent_final_answer_tool_call_like"] = _looks_like_tool_call_text(final_answer)
    if history_mode == "keep5":
        metrics["agent_history_mode"] = history_mode
        metrics["agent_keep_tool_results"] = keep_tool_results
        metrics.setdefault("agent_history_keep5_omitted_tool_results", 0)
    elif return_messages:
        metrics["agent_history_mode"] = history_mode
    if capture_raw_responses:
        if not raw_responses:
            raise RuntimeError("raw Responses capture was enabled but no successful response was recorded")
        metrics["agent_raw_responses"] = raw_responses
    if return_messages:
        metrics["messages"] = messages
    return metrics


def _messages_from_prompt(prompt: Any, system_prompt: Any = None) -> list[dict[str, Any]]:
    if isinstance(prompt, list):
        messages = [dict(message) for message in prompt]
    else:
        messages = [{"role": "user", "content": str(prompt)}]

    if not messages or messages[0].get("role") != "system":
        if system_prompt is None:
            messages.insert(0, {"role": "system", "content": SYSTEM_PROMPT})
        elif str(system_prompt):
            messages.insert(0, {"role": "system", "content": str(system_prompt)})
    return messages


def _original_question_from_messages(messages: list[dict[str, Any]]) -> str:
    for message in messages:
        if message.get("role") == "user":
            content = message.get("content")
            if isinstance(content, str):
                return content
            return str(content)
    return ""


def _strip_thinking(text: str) -> str:
    text = text or ""
    if "</think>" in text:
        return text.rsplit("</think>", 1)[-1].strip()
    return text.strip()


def _looks_like_tool_call_text(text: str) -> bool:
    return bool(_TOOL_CALL_TEXT_RE.search(text or ""))


def _looks_like_unparsed_tool_call(text: str) -> bool:
    text = (text or "").strip().lower()
    markers = ("dsml", "<search", "search_query", "tool:search", "web_search", "fetch_url", "tool_call")
    return not text or any(marker in text for marker in markers)


def _history_mode(metadata: dict[str, Any]) -> str:
    mode = str(metadata.get("agent_history_mode") or "react").strip().lower()
    if mode not in {"react", "keep5"}:
        raise ValueError("agent_history_mode must be 'react' or 'keep5'")
    return mode


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _falsey(value: Any) -> bool:
    if isinstance(value, bool):
        return not value
    if value is None:
        return False
    return str(value).strip().lower() in {"0", "false", "no", "n", "off"}


def _keep_last_tool_results(messages: list[dict[str, Any]], keep: int) -> int:
    tool_indices = [idx for idx, message in enumerate(messages) if message.get("role") == "tool"]
    if not tool_indices:
        return 0

    keep_set = set(tool_indices[-keep:]) if keep > 0 else set()
    omitted = 0
    for idx in tool_indices:
        if idx in keep_set:
            continue
        if messages[idx].get("content") != TOOL_RESULT_OMITTED_PLACEHOLDER:
            messages[idx]["content"] = TOOL_RESULT_OMITTED_PLACEHOLDER
        omitted += 1
    return omitted


def _optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _context_length_error_budget(response: Any) -> tuple[int, int] | None:
    if getattr(response, "status_code", None) != 400:
        return None
    text = str(getattr(response, "text", "") or "")
    if "maximum context length" not in text.lower():
        return None
    max_match = _CONTEXT_LENGTH_MAX_RE.search(text)
    input_match = _CONTEXT_LENGTH_INPUT_RE.search(text)
    if max_match is None or input_match is None:
        return None
    server_max_tokens = int(max_match.group(1).replace(",", ""))
    input_tokens = int(input_match.group(1).replace(",", ""))
    if server_max_tokens <= 0 or input_tokens <= 0 or input_tokens >= server_max_tokens:
        return None
    return server_max_tokens, input_tokens


def _should_force_final_answer(max_seq_len: int | None, session_tokens: int, reserve_tokens: int) -> bool:
    return max_seq_len is not None and session_tokens > 0 and session_tokens >= max_seq_len - reserve_tokens


def _final_request_kwargs(
    request_kwargs: dict[str, Any], max_seq_len: int | None, session_tokens: int
) -> dict[str, Any] | None:
    kwargs = {k: v for k, v in request_kwargs.items() if k not in {"min_tokens", "min_new_tokens"}}
    original_max_tokens = _optional_int(kwargs.get("max_tokens")) or 32768
    if max_seq_len is None or session_tokens <= 0:
        kwargs["max_tokens"] = original_max_tokens
        return kwargs

    available = max_seq_len - session_tokens - FINAL_ANSWER_PROMPT_MARGIN_TOKENS
    if available <= 0:
        return None
    kwargs["max_tokens"] = max(1, min(original_max_tokens, available))
    return kwargs


def _session_tokens_from_response(usage: dict[str, Any], meta_info: dict[str, Any], previous_tokens: int) -> int:
    completion_tokens = _optional_int(usage.get("completion_tokens")) or _optional_int(
        meta_info.get("completion_tokens")
    )
    if completion_tokens is None and isinstance(meta_info.get("output_token_logprobs"), list):
        completion_tokens = len(meta_info["output_token_logprobs"])
    return _optional_int(usage.get("total_tokens")) or _optional_int(meta_info.get("total_tokens")) or (
        previous_tokens + (completion_tokens or 0)
    )


async def _execute_tool_calls(tool_calls: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    results = await asyncio.gather(*(_execute_tool_call(tool_call) for tool_call in tool_calls))
    messages: list[dict[str, Any]] = []
    metrics: dict[str, Any] = {}
    for message, meta in results:
        messages.append(message)
        _merge_numeric_metrics(metrics, meta)
    return messages, metrics


async def _execute_tool_call(tool_call: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    tool_call_id = str(tool_call.get("id") or "tool_call")
    function = tool_call.get("function") or {}
    name = str(function.get("name") or "")

    try:
        arguments = _parse_arguments(function.get("arguments"))
        if name == "search":
            result = await SEARCH_TOOL.execute(arguments)
            meta = {"agent_search_count": 1, **_tool_quality_metrics(result.metadata)}
        elif name == "fetch_url":
            result = await BROWSE_TOOL.execute(arguments)
            meta = {"agent_fetch_url_count": 1, **_tool_quality_metrics(result.metadata)}
        else:
            result = _ToolResult(f"Unknown tool: {name}", {"success": False, "tool_call_failures": 1})
            meta = _tool_quality_metrics(result.metadata)
    except Exception as exc:
        logger.warning("Tool call failed: %s", exc, exc_info=True)
        result = _ToolResult(f"({name or 'tool'} error: {type(exc).__name__}: {exc})", {})
        meta = {"tool_call_failures": 1}

    meta = {"agent_tool_call_count": 1, **meta}

    return {
        "role": "tool",
        "tool_call_id": tool_call_id,
        "name": name,
        "content": result.text,
    }, meta


def _tool_quality_metrics(metadata: dict[str, Any]) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    success = metadata.get("success")
    metrics["tool_call_failures"] = int(success is False or bool(metadata.get("tool_call_failures", 0)))
    metrics["tool_unit_count"] = int(metadata.get("tool_unit_count", 0) or 0)
    metrics["tool_unit_successes"] = int(metadata.get("tool_unit_successes", 0) or 0)
    metrics["agent_search_cache_hit_count"] = int(metadata.get("cache_hits", 0) or 0)
    metrics["agent_search_retry_429_count"] = int(metadata.get("search_retry_429s", 0) or 0)
    metrics["agent_search_non_200_attempt_count"] = int(metadata.get("search_non_200_attempts", 0) or 0)
    if isinstance(success, bool):
        metrics["tool_call_failures"] = int(not success)
    return metrics


def _count_repeated_search_queries(
    tool_calls: list[dict[str, Any]],
    history: list[frozenset[str]],
) -> int:
    _, repeat_count = _count_search_queries_and_repeats(tool_calls, history)
    return repeat_count


def _count_search_queries_and_repeats(
    tool_calls: list[dict[str, Any]],
    history: list[frozenset[str]],
) -> tuple[int, int]:
    query_count = 0
    repeat_count = 0
    for tool_call in tool_calls:
        function = tool_call.get("function") or {}
        if str(function.get("name") or "") != "search":
            continue
        try:
            arguments = _parse_arguments(function.get("arguments"))
        except ValueError:
            continue

        for query in _coerce_search_queries_for_metrics(arguments.get("query")):
            tokens = _normalize_search_query_tokens(query)
            if not tokens:
                continue
            query_count += 1
            if any(
                _jaccard_similarity(tokens, previous) >= _SEARCH_QUERY_REPEAT_JACCARD_THRESHOLD
                for previous in history
            ):
                repeat_count += 1
            history.append(tokens)
    return query_count, repeat_count


def _count_repeated_fetch_urls(
    tool_calls: list[dict[str, Any]],
    history: set[str],
) -> int:
    repeat_count = 0
    for tool_call in tool_calls:
        function = tool_call.get("function") or {}
        if str(function.get("name") or "") != "fetch_url":
            continue
        try:
            arguments = _parse_arguments(function.get("arguments"))
        except ValueError:
            continue

        for url in _coerce_fetch_urls_for_metrics(arguments.get("url")):
            canonical_url = _canonical_fetch_url_for_metrics(url)
            if not canonical_url:
                continue
            if canonical_url in history:
                repeat_count += 1
            history.add(canonical_url)
    return repeat_count


def _coerce_search_queries_for_metrics(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list | tuple):
        return [str(item) for item in value if item is not None and str(item).strip()]
    return [str(value)] if str(value).strip() else []


def _normalize_search_query_tokens(query: str) -> frozenset[str]:
    text = re.sub(r"https?://\S+", " URL ", str(query).lower())
    return frozenset(token for token in re.split(r"[^\w]+", text) if token)


def _coerce_fetch_urls_for_metrics(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list | tuple):
        return [str(item) for item in value if item is not None and str(item).strip()]
    return [str(value)] if str(value).strip() else []


def _canonical_fetch_url_for_metrics(url: str) -> str:
    text = str(url).strip()
    if not text:
        return ""
    try:
        parts = urlsplit(text)
        port = parts.port
    except ValueError:
        return text.split("#", 1)[0].strip()

    if not parts.scheme or not parts.netloc:
        return text.split("#", 1)[0].strip()

    scheme = parts.scheme.lower()
    host = (parts.hostname or parts.netloc).lower()
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    default_port = (scheme == "http" and port == 80) or (scheme == "https" and port == 443)
    netloc = host if port is None or default_port else f"{host}:{port}"
    path = parts.path or "/"
    query = f"?{parts.query}" if parts.query else ""
    return f"{scheme}://{netloc}{path}{query}"


def _jaccard_similarity(left: frozenset[str], right: frozenset[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def _parse_arguments(raw: Any) -> dict[str, Any]:
    if raw is None or raw == "":
        return {}
    if isinstance(raw, dict):
        return raw
    try:
        parsed = json.loads(str(raw))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid tool arguments JSON: {raw!r}") from exc
    if not isinstance(parsed, dict):
        raise ValueError("tool arguments must decode to a JSON object")
    return parsed


def _merge_numeric_metrics(dst: dict[str, Any], src: dict[str, Any]) -> None:
    for key, value in src.items():
        if isinstance(value, bool):
            continue
        if isinstance(value, int | float):
            dst[key] = dst.get(key, 0) + value


class _ToolResult:
    def __init__(self, text: str, metadata: dict[str, Any]):
        self.text = text
        self.metadata = metadata
