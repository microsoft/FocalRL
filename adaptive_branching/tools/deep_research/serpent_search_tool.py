"""Serpent web-search backend with the same agent contract as Microsoft search."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any
from urllib.parse import urlsplit

import httpx

from ._sqlite_cache import AsyncSqliteCache
from .microsoft_search_tool import SEARCH_TOOL_SCHEMA, ToolExecution, _coerce_str_list, _required_config

logger = logging.getLogger(__name__)

_FATAL_HTTP_STATUSES = {400, 401, 402, 403}
_RETRY_HTTP_STATUSES = {429, 500, 502, 503, 504}
_SUPPORTED_ENGINES = {"google", "bing", "yahoo", "ddg", "brave"}


class SerpentSearchTool:
    """Search Serpent while preserving the existing ``search`` tool interface."""

    tool_schema = SEARCH_TOOL_SCHEMA

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        tool_schema: dict[str, Any] | None = None,
        *,
        api_key: str | None = None,
        endpoint: str | None = None,
        engine: str | None = None,
        country: str | None = None,
        result_format: str | None = None,
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
            raise ValueError("SerpentSearchTool requires the search YAML mapping")
        cfg = config
        self.tool_schema = tool_schema or SEARCH_TOOL_SCHEMA
        self.api_key = str(api_key if api_key is not None else cfg.get("serpent_api_key", "")).strip()
        self.endpoint = str(
            endpoint if endpoint is not None else _required_config(cfg, "serpent_endpoint")
        ).strip()
        self.engine = str(engine if engine is not None else _required_config(cfg, "serpent_engine")).strip().lower()
        self.country = str(
            country if country is not None else _required_config(cfg, "serpent_country")
        ).strip().lower()
        self.result_format = str(
            result_format if result_format is not None else _required_config(cfg, "serpent_format")
        ).strip().lower()
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
        resolved_cache_path = str(
            cache_path if cache_path is not None else _required_config(cfg, "serpent_cache_path")
        ).strip()

        parsed_endpoint = urlsplit(self.endpoint)
        if not self.api_key:
            raise RuntimeError("Serpent search requires a non-empty SERPENT_API_KEY")
        if parsed_endpoint.scheme != "https" or not parsed_endpoint.netloc:
            raise ValueError("serpent_endpoint must be an absolute HTTPS URL")
        if self.engine not in _SUPPORTED_ENGINES:
            raise ValueError(f"unsupported Serpent search engine: {self.engine!r}")
        if len(self.country) != 2 or not self.country.isalpha():
            raise ValueError("serpent_country must be a two-letter country code")
        if self.result_format != "full":
            raise ValueError("serpent_format must be 'full' so search snippets remain available")
        if not 1 <= self.max_results_per_query <= 100:
            raise ValueError("max_results_per_query must be in [1, 100]")
        if not 1 <= self.max_queries_per_call <= 3:
            raise ValueError("max_queries_per_call must be in [1, 3]")
        if self.max_length <= 0:
            raise ValueError("max_length must be positive")
        if self.timeout <= 0 or self.num_retries <= 0 or concurrency <= 0 or self.max_qps < 0:
            raise ValueError("search timeout/retry/concurrency must be positive and max_qps must be non-negative")
        if not resolved_cache_path:
            raise ValueError("serpent_cache_path must be non-empty")

        self._sema = asyncio.Semaphore(concurrency)
        self._rate_lock = asyncio.Lock()
        self._next_request_time = 0.0
        self.cache = AsyncSqliteCache(resolved_cache_path)
        self._cache_locks: dict[str, asyncio.Lock] = {}

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
            text, metadata = result
            rendered.append(text)
            successes += int(bool(metadata.get("success")))
            cache_hits += int(metadata.get("cache_hits", 0))
            retry_429s += int(metadata.get("search_retry_429s", 0))
            non_200_attempts += int(metadata.get("search_non_200_attempts", 0))

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

            payload = await self._call_serpent_search(query)
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
                "provider": "serpent",
                "endpoint": self.endpoint,
                "engine": self.engine,
                "country": self.country,
                "format": self.result_format,
                "max_length": self.max_length,
                "max_results_per_query": self.max_results_per_query,
                "query": query,
            },
            sort_keys=True,
            ensure_ascii=False,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    async def _call_serpent_search(self, query: str) -> dict[str, Any]:
        params = {
            "q": query,
            "num": self.max_results_per_query,
            "engine": self.engine,
            "country": self.country,
            "format": self.result_format,
        }
        headers = {"X-API-Key": self.api_key}
        non_200_attempts = 0
        retry_429s = 0
        last_exc: Exception | None = None

        async with self._sema:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                for attempt in range(self.num_retries):
                    try:
                        await self._wait_for_rate_slot()
                        response = await client.get(self.endpoint, headers=headers, params=params)
                        if response.status_code != 200:
                            non_200_attempts += 1
                        if response.status_code == 429:
                            retry_429s += 1
                        if response.status_code in _FATAL_HTTP_STATUSES:
                            raise RuntimeError(
                                f"Serpent search HTTP {response.status_code}: {response.text[:200]}"
                            )
                        if response.status_code in _RETRY_HTTP_STATUSES:
                            if attempt + 1 < self.num_retries:
                                await asyncio.sleep(self._retry_delay_seconds(response, attempt))
                                continue
                            break
                        if response.status_code != 200:
                            return self._with_counters(
                                {"_filtered_msg": f"(search HTTP {response.status_code}: {response.text[:200]})"},
                                non_200_attempts,
                                retry_429s,
                            )

                        try:
                            data = response.json()
                        except ValueError as exc:
                            raise RuntimeError("Serpent search returned invalid JSON") from exc
                        normalized = self._normalize_response(data)
                        return self._with_counters(normalized, non_200_attempts, retry_429s)
                    except (httpx.TimeoutException, httpx.TransportError) as exc:
                        last_exc = exc
                        if attempt + 1 < self.num_retries:
                            await asyncio.sleep(2**attempt)
                            continue
                        break

        detail = f"{type(last_exc).__name__}: {last_exc}" if last_exc is not None else "retryable HTTP failure"
        logger.warning("Serpent search failed after %d attempts: %s", self.num_retries, detail)
        return self._with_counters(
            {"_filtered_msg": f"(search timeout/network error: {detail})"}, non_200_attempts, retry_429s
        )

    def _normalize_response(self, data: Any) -> dict[str, Any]:
        if not isinstance(data, dict):
            raise TypeError("Serpent search response must be a JSON object")
        if data.get("success") is not True:
            message = str(data.get("error") or data.get("message") or "unknown API error")
            return {"_filtered_msg": f"(search error: {message})", "_cacheable": False}
        results = data.get("results")
        if not isinstance(results, dict):
            raise TypeError("Serpent full-format response requires a results object")
        organic = results.get("organic")
        if not isinstance(organic, list):
            raise TypeError("Serpent full-format response requires results.organic to be a list")

        web_results: list[dict[str, str]] = []
        for index, item in enumerate(organic[: self.max_results_per_query]):
            if not isinstance(item, dict):
                raise TypeError(f"Serpent organic result {index} must be an object")
            title = str(item.get("title") or "(no title)")
            url = str(item.get("url") or "")
            snippet = str(item.get("snippet") or "").strip()[: self.max_length]
            web_results.append({"title": title, "url": url, "content": snippet})
        return {"webResults": web_results, "_cacheable": bool(web_results)}

    async def _wait_for_rate_slot(self) -> None:
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
    def _retry_delay_seconds(response: httpx.Response, attempt: int) -> float:
        value = response.headers.get("Retry-After", "").strip()
        if value:
            try:
                return max(0.0, float(value))
            except ValueError:
                try:
                    retry_at = parsedate_to_datetime(value)
                    if retry_at.tzinfo is None:
                        retry_at = retry_at.replace(tzinfo=timezone.utc)
                    return max(0.0, (retry_at - datetime.now(timezone.utc)).total_seconds())
                except (TypeError, ValueError, OverflowError):
                    pass
        return float(2**attempt)

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
        for index, item in enumerate(web_results, 1):
            lines.append(f"{index}. [{item['title']}]({item['url']})\n   {item['content']}")
        return "\n".join(lines)


_DEFAULT_TOOL: SerpentSearchTool | None = None


def _default_tool() -> SerpentSearchTool:
    global _DEFAULT_TOOL
    if _DEFAULT_TOOL is None:
        from adaptive_branching.src.deep_research.agent import SEARCH_TOOL

        if not isinstance(SEARCH_TOOL, SerpentSearchTool):
            raise RuntimeError("agent search provider is not configured as 'serpent'")
        _DEFAULT_TOOL = SEARCH_TOOL
    return _DEFAULT_TOOL


tool_specs = [SEARCH_TOOL_SCHEMA]


async def execute_tool(name: str, params: dict[str, Any]) -> str:
    if name != "search":
        return f"Unknown tool: {name}"
    result = await _default_tool().execute(params)
    return result.text
