"""Small Microsoft Grounding search backend for Miles agentic rollout.

This module intentionally avoids the old veRL ``BaseTool`` plumbing.  It keeps
the pieces needed by the session-server agent path: OpenAI tool schema,
parameter validation, bounded concurrency, retries, and deterministic text
rendering for model observations.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import random
from dataclasses import dataclass, field
from typing import Any

import httpx

from ._sqlite_cache import AsyncSqliteCache

logger = logging.getLogger(__name__)

_FATAL_HTTP_STATUSES = {401, 403}
_RETRY_HTTP_STATUSES = {429, 502, 503, 504}


def _required_config(config: dict[str, Any], key: str) -> Any:
    if key not in config:
        raise ValueError(f"search tool config requires {key!r}")
    return config[key]


SEARCH_TOOL_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "search",
        "description": (
            "Search the web using Microsoft Grounding. "
            "Provide one to three complementary queries. "
            "Returns result titles, URLs, and query-relevant passages."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "minLength": 1,
                    },
                    "minItems": 1,
                    "maxItems": 3,
                    "description": (
                        "One to three complementary web search query strings."
                    ),
                }
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    },
}


@dataclass
class ToolExecution:
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


def _coerce_str_list(value: Any, *, key: str, limit: int) -> list[str]:
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        raise ValueError(f"{key!r} must be a string or list of strings")
    items = [str(item).strip() for item in value if str(item).strip()]
    if not items:
        raise ValueError(f"{key!r} must contain at least one non-empty string")
    return items[:limit]


def _coerce_api_keys(*values: Any) -> list[str]:
    keys: list[str] = []
    for value in values:
        if value is None:
            continue
        if isinstance(value, str):
            stripped = value.strip()
            if not stripped:
                continue
            if stripped.startswith("["):
                try:
                    parsed = json.loads(stripped)
                except json.JSONDecodeError:
                    parsed = [stripped]
                keys.extend(str(item).strip() for item in parsed if str(item).strip())
            else:
                normalized = stripped.replace("\n", ",")
                keys.extend(item.strip() for item in normalized.split(",") if item.strip())
        elif isinstance(value, list | tuple | set):
            keys.extend(str(item).strip() for item in value if str(item).strip())
        else:
            keys.append(str(value).strip())

    return list(dict.fromkeys(key for key in keys if key))


class MicrosoftSearchTool:
    tool_schema = SEARCH_TOOL_SCHEMA

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        tool_schema: dict[str, Any] | None = None,
        *,
        api_key: str | None = None,
        api_keys: list[str] | None = None,
        endpoint: str | None = None,
        content_format: str | None = None,
        max_length: int | None = None,
        max_results_per_query: int | None = None,
        max_queries_per_call: int | None = None,
        timeout: float | None = None,
        num_retries: int | None = None,
        max_concurrency: int | None = None,
        max_qps: float | None = None,
        cache_path: str | None = None,
    ) -> None:
        if not isinstance(config, dict):
            raise ValueError("MicrosoftSearchTool requires the search YAML mapping")
        cfg = config
        self.tool_schema = tool_schema or SEARCH_TOOL_SCHEMA
        self.api_keys = _coerce_api_keys(
            api_keys,
            api_key,
            cfg.get("ms_api_keys"),
            cfg.get("ms_api_key"),
        )
        self.api_key = self.api_keys[0] if self.api_keys else ""
        self.endpoint = endpoint if endpoint is not None else str(_required_config(cfg, "ms_endpoint"))
        self.content_format = (
            content_format if content_format is not None else str(_required_config(cfg, "content_format"))
        )
        self.max_length = int(max_length if max_length is not None else _required_config(cfg, "max_length"))
        self.max_results_per_query = int(
            max_results_per_query
            if max_results_per_query is not None
            else _required_config(cfg, "max_results_per_query")
        )
        self.max_queries_per_call = int(
            max_queries_per_call
            if max_queries_per_call is not None
            else _required_config(cfg, "max_queries_per_call")
        )
        self.timeout = float(timeout if timeout is not None else _required_config(cfg, "timeout"))
        self.num_retries = int(num_retries if num_retries is not None else _required_config(cfg, "num_retries"))
        concurrency = int(
            max_concurrency if max_concurrency is not None else _required_config(cfg, "max_concurrency")
        )
        self.max_qps = float(max_qps if max_qps is not None else _required_config(cfg, "max_qps"))
        resolved_cache_path = cache_path if cache_path is not None else str(_required_config(cfg, "cache_path"))

        if not self.api_keys:
            raise RuntimeError("search tool config requires at least one ms_api_key")
        if self.max_length <= 0 or self.max_results_per_query <= 0 or self.max_queries_per_call <= 0:
            raise ValueError("search size limits must be positive")
        if self.timeout <= 0 or self.num_retries <= 0 or concurrency <= 0 or self.max_qps < 0:
            raise ValueError("search timeout/retry/concurrency must be positive and max_qps must be non-negative")

        self._sema = asyncio.Semaphore(concurrency)
        self._rate_lock = asyncio.Lock()
        self._next_request_time = 0.0
        self.cache = AsyncSqliteCache(resolved_cache_path)
        self._cache_locks: dict[str, asyncio.Lock] = {}

    def _choose_api_key(self) -> str:
        return random.choice(self.api_keys)

    async def execute(self, parameters: dict[str, Any]) -> ToolExecution:
        try:
            queries = _coerce_str_list(parameters.get("query"), key="query", limit=self.max_queries_per_call)
        except ValueError as exc:
            return ToolExecution(f"(search error: {exc})", {"success": False, "tool_call_failures": 1})

        results = await asyncio.gather(*(self._single_search(query) for query in queries), return_exceptions=True)
        rendered: list[str] = []
        successes = 0
        cache_hits = 0
        retry_429s = 0
        non_200_attempts = 0

        for query, result in zip(queries, results, strict=True):
            if isinstance(result, Exception):
                rendered.append(f"### Query: {query}\n(error: {type(result).__name__}: {result})")
                continue
            text, meta = result
            rendered.append(text)
            successes += int(bool(meta.get("success")))
            cache_hits += int(meta.get("cache_hits", 0))
            retry_429s += int(meta.get("search_retry_429s", 0))
            non_200_attempts += int(meta.get("search_non_200_attempts", 0))

        success = successes > 0
        return ToolExecution(
            "\n\n===\n\n".join(rendered),
            {
                "success": success,
                "tool_call_count": 1,
                "tool_call_successes": int(success),
                "tool_call_failures": int(not success),
                "tool_unit_count": len(queries),
                "tool_unit_successes": successes,
                "tool_unit_failures": len(queries) - successes,
                "cache_hits": cache_hits,
                "search_retry_429s": retry_429s,
                "search_non_200_attempts": non_200_attempts,
            },
        )

    async def _single_search(self, query: str) -> tuple[str, dict[str, Any]]:
        key = self._cache_key(query)
        cached = await self.cache.get(key)
        if cached is not None:
            return cached, {"success": True, "cache_hits": 1}

        lock = self._cache_locks.setdefault(key, asyncio.Lock())
        async with lock:
            cached = await self.cache.get(key)
            if cached is not None:
                return cached, {"success": True, "cache_hits": 1}

            payload = await self._call_ms_search(query)
            rendered = self._render(query, payload)
            success = bool(payload.get("_cacheable", False))
            if success:
                await self.cache.set(key, rendered)
            return rendered, {
                "success": success,
                "cache_hits": 0,
                "search_non_200_attempts": int(payload.get("_search_non_200_attempts", 0) or 0),
                "search_retry_429s": int(payload.get("_search_retry_429s", 0) or 0),
            }

    def _cache_key(self, query: str) -> str:
        payload = json.dumps(
            {
                "endpoint": self.endpoint,
                "content_format": self.content_format,
                "max_length": self.max_length,
                "max_results_per_query": self.max_results_per_query,
                "query": query,
            },
            sort_keys=True,
            ensure_ascii=False,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    async def _call_ms_search(self, query: str) -> dict[str, Any]:
        body = {
            "query": query,
            "maxResults": self.max_results_per_query,
            "contentFormat": self.content_format,
            "maxLength": self.max_length,
        }
        non_200_attempts = 0
        retry_429s = 0
        last_exc: Exception | None = None

        async with self._sema:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                for attempt in range(self.num_retries):
                    try:
                        await self._wait_for_rate_slot()
                        headers = {"x-apikey": self._choose_api_key(), "content-type": "application/json"}
                        response = await client.post(self.endpoint, headers=headers, json=body)
                        if response.status_code != 200:
                            non_200_attempts += 1
                        if response.status_code == 429:
                            retry_429s += 1
                        if response.status_code in _FATAL_HTTP_STATUSES:
                            raise RuntimeError(f"MS search HTTP {response.status_code}: {response.text[:200]}")
                        if response.status_code in _RETRY_HTTP_STATUSES:
                            await asyncio.sleep(2**attempt)
                            continue
                        if response.status_code != 200:
                            return self._with_counters(
                                {"_filtered_msg": f"(search HTTP {response.status_code}: {response.text[:200]})"},
                                non_200_attempts,
                                retry_429s,
                            )

                        data = response.json()
                        if isinstance(data, dict) and "errorCode" in data:
                            msg = data.get("userMessage") or data.get("errorCode")
                            return self._with_counters(
                                {"_filtered_msg": f"(search filtered: {data['errorCode']} - {msg})"},
                                non_200_attempts,
                                retry_429s,
                            )
                        data["_cacheable"] = bool(data.get("webResults"))
                        return self._with_counters(data, non_200_attempts, retry_429s)
                    except (httpx.TimeoutException, httpx.HTTPError) as exc:
                        last_exc = exc
                        await asyncio.sleep(2**attempt)

        logger.warning("MS search failed after %d attempts: %s", self.num_retries, last_exc)
        return self._with_counters(
            {"_filtered_msg": f"(search timeout/network error: {last_exc})"}, non_200_attempts, retry_429s
        )

    async def _wait_for_rate_slot(self) -> None:
        """Limit uncached HTTP request starts across all concurrent searches."""
        if self.max_qps <= 0:
            return
        interval = 1.0 / self.max_qps
        loop = asyncio.get_running_loop()
        async with self._rate_lock:
            now = loop.time()
            scheduled = max(now, self._next_request_time)
            self._next_request_time = scheduled + interval
            delay = scheduled - now
        if delay > 0:
            await asyncio.sleep(delay)

    @staticmethod
    def _with_counters(payload: dict[str, Any], non_200_attempts: int, retry_429s: int) -> dict[str, Any]:
        payload["_search_non_200_attempts"] = non_200_attempts
        payload["_search_retry_429s"] = retry_429s
        return payload

    @staticmethod
    def _render(query: str, payload: dict[str, Any]) -> str:
        if "_filtered_msg" in payload:
            return f"### Query: {query}\n{payload['_filtered_msg']}"

        web_results = payload.get("webResults") or []
        if not web_results:
            return f"### Query: {query}\n(no results)"

        lines = [f"### Query: {query}"]
        for idx, item in enumerate(web_results, 1):
            title = item.get("title", "(no title)")
            url = item.get("url", "")
            content = (item.get("content") or "").strip()
            lines.append(f"{idx}. [{title}]({url})\n   {content}")
        return "\n".join(lines)


_DEFAULT_TOOL: MicrosoftSearchTool | None = None


def _default_tool() -> MicrosoftSearchTool:
    global _DEFAULT_TOOL
    if _DEFAULT_TOOL is None:
        from adaptive_branching.src.deep_research.agent import SEARCH_TOOL

        _DEFAULT_TOOL = SEARCH_TOOL
    return _DEFAULT_TOOL


tool_specs = [SEARCH_TOOL_SCHEMA]


async def execute_tool(name: str, params: dict[str, Any]) -> str:
    if name != "search":
        return f"Unknown tool: {name}"
    result = await _default_tool().execute(params)
    return result.text
