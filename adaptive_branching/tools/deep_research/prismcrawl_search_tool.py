"""PrismCrawl Google web-search backend with the existing agent search contract."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from typing import Any
from urllib.parse import urlsplit

import httpx

from .microsoft_search_tool import SEARCH_TOOL_SCHEMA
from .serpent_search_tool import SerpentSearchTool

logger = logging.getLogger(__name__)

_FATAL_HTTP_STATUSES = {400, 401, 402, 403, 405, 413}
_RETRY_HTTP_STATUSES = {429, 500, 502, 503, 504}


class PrismCrawlSearchTool(SerpentSearchTool):
    """Call PrismCrawl while preserving the current ``search`` tool observation format."""

    tool_schema = SEARCH_TOOL_SCHEMA

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        tool_schema: dict[str, Any] | None = None,
        *,
        api_key: str | None = None,
        endpoint: str | None = None,
        country: str | None = None,
        language: str | None = None,
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
            raise ValueError("PrismCrawlSearchTool requires the search YAML mapping")

        resolved_api_key = api_key if api_key is not None else config.get("prismcrawl_api_key", "")
        if not isinstance(resolved_api_key, str):
            raise ValueError("prismcrawl_api_key must be a string")
        resolved_api_key = resolved_api_key.strip()
        resolved_endpoint = str(endpoint if endpoint is not None else config.get("prismcrawl_endpoint", "")).strip()
        resolved_country = (
            str(country if country is not None else config.get("prismcrawl_country", "")).strip().lower()
        )
        resolved_language = (
            str(language if language is not None else config.get("prismcrawl_language", "")).strip().lower()
        )
        resolved_cache_path = str(
            cache_path if cache_path is not None else config.get("prismcrawl_cache_path", "")
        ).strip()

        if not resolved_api_key:
            raise RuntimeError("PrismCrawl search requires a non-empty PRISMCRAWL_API_KEY")
        parsed_endpoint = urlsplit(resolved_endpoint)
        if parsed_endpoint.scheme != "https" or not parsed_endpoint.netloc:
            raise ValueError("prismcrawl_endpoint must be an absolute HTTPS URL")
        if len(resolved_country) != 2 or not resolved_country.isalpha():
            raise ValueError("prismcrawl_country must be a two-letter country code")
        if len(resolved_language) != 2 or not resolved_language.isalpha():
            raise ValueError("prismcrawl_language must be a two-letter language code")
        if not resolved_cache_path:
            raise ValueError("prismcrawl_cache_path must be non-empty")

        super().__init__(
            config=config,
            tool_schema=tool_schema,
            api_key=resolved_api_key,
            endpoint=resolved_endpoint,
            engine="google",
            country=resolved_country,
            result_format="full",
            max_length=max_length,
            max_results_per_query=max_results_per_query,
            max_queries_per_call=max_queries_per_call,
            timeout=timeout,
            num_retries=num_retries,
            max_concurrency=max_concurrency,
            max_qps=max_qps,
            cache_path=resolved_cache_path,
        )
        self.language = resolved_language

    def _cache_key(self, query: str) -> str:
        payload = json.dumps(
            {
                "provider": "prismcrawl",
                "endpoint": self.endpoint,
                "country": self.country,
                "language": self.language,
                "max_length": self.max_length,
                "max_results_per_query": self.max_results_per_query,
                "query": query,
            },
            sort_keys=True,
            ensure_ascii=False,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    async def _call_serpent_search(self, query: str) -> dict[str, Any]:
        """Override the inherited request hook with PrismCrawl's POST contract."""
        if not isinstance(query, str) or not query.strip() or len(query.encode("utf-8")) > 8192:
            raise ValueError("PrismCrawl query must be non-empty and at most 8192 UTF-8 bytes")
        body = {
            "query": query,
            "html": False,
            "gl": self.country,
            "hl": self.language,
        }
        headers = {"x-api-key": self.api_key, "Content-Type": "application/json"}
        non_200_attempts = 0
        retry_429s = 0
        last_exc: Exception | None = None

        async with self._sema:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                for attempt in range(self.num_retries):
                    try:
                        await self._wait_for_rate_slot()
                        response = await client.post(self.endpoint, headers=headers, json=body)
                        if response.status_code != 200:
                            non_200_attempts += 1
                        if response.status_code == 429:
                            retry_429s += 1
                        if response.status_code in _FATAL_HTTP_STATUSES:
                            raise RuntimeError(f"PrismCrawl search HTTP {response.status_code}")
                        if response.status_code in _RETRY_HTTP_STATUSES:
                            if attempt + 1 < self.num_retries:
                                await asyncio.sleep(self._retry_delay_seconds(response, attempt))
                                continue
                            break
                        if response.status_code != 200:
                            return self._with_counters(
                                {"_filtered_msg": f"(search HTTP {response.status_code})"},
                                non_200_attempts,
                                retry_429s,
                            )

                        try:
                            data = response.json()
                        except ValueError as exc:
                            raise RuntimeError("PrismCrawl search returned invalid JSON") from exc
                        normalized = self._normalize_response(data)
                        return self._with_counters(normalized, non_200_attempts, retry_429s)
                    except (httpx.TimeoutException, httpx.TransportError) as exc:
                        last_exc = exc
                        if attempt + 1 < self.num_retries:
                            await asyncio.sleep(2**attempt)
                            continue
                        break

        detail = f"{type(last_exc).__name__}: {last_exc}" if last_exc is not None else "retryable HTTP failure"
        logger.warning("PrismCrawl search failed after %d attempts: %s", self.num_retries, detail)
        return self._with_counters(
            {"_filtered_msg": f"(search timeout/network error: {detail})"}, non_200_attempts, retry_429s
        )

    def _normalize_response(self, data: Any) -> dict[str, Any]:
        if not isinstance(data, dict):
            raise TypeError("PrismCrawl search response must be a JSON object")
        if data.get("success") is not True:
            raise RuntimeError("PrismCrawl search response success is not true")
        result_data = data.get("data")
        if not isinstance(result_data, dict) or result_data.get("format") != "json":
            raise TypeError("PrismCrawl response requires data.format=json")
        content = result_data.get("content")
        if not isinstance(content, dict) or not isinstance(content.get("results"), list):
            raise TypeError("PrismCrawl response requires data.content.results to be a list")

        web_results: list[dict[str, str]] = []
        for index, item in enumerate(content["results"]):
            if not isinstance(item, dict):
                raise TypeError(f"PrismCrawl result {index} must be an object")
            if item.get("type") != "organic":
                continue
            for field in ("title", "url", "snippet"):
                if item.get(field) is not None and not isinstance(item[field], str):
                    raise TypeError(f"PrismCrawl result {index}.{field} must be a string or null")
            if not item.get("url"):
                raise ValueError(f"PrismCrawl organic result {index} requires a non-empty url")
            title = item.get("title") or "(no title)"
            url = item["url"]
            # Match Serper's configured observation limits; Prism has no `num` parameter.
            snippet = (item.get("snippet") or "").strip()[: self.max_length]
            if len(web_results) < self.max_results_per_query:
                web_results.append({"title": title, "url": url, "content": snippet})
        return {"webResults": web_results, "_cacheable": bool(web_results)}
