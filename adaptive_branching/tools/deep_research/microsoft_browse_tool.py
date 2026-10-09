"""Microsoft Grounding browse backend with Browser-LLM summarisation.

This is the Miles session-server version of the old veRL browse tool.  It keeps
the useful runtime behavior: URL validation, Jina-first fetching, MS browse
fallback, raw and summary SQLite caches, retries, and deterministic tool output.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import random
import re
from dataclasses import dataclass, field
from typing import Any

import httpx
from openai import AsyncOpenAI

from ._sqlite_cache import AsyncSqliteCache

logger = logging.getLogger(__name__)

_FATAL_HTTP_STATUSES = {401, 403}
_RETRY_HTTP_STATUSES = {429, 502, 503, 504}
_TARGET_STATUS_RE = re.compile(r"Warning:\s*Target URL returned error\s+(\d{3})", re.IGNORECASE)
_ANTI_BOT_RE = re.compile(
    r"cloudflare|just a moment|captcha|access denied|forbidden|not authorized|permission denied|"
    r"security checkpoint|429:\s*too_many_requests|too many requests|rate limit|login required|"
    r"sign in to continue|log in to continue",
    re.IGNORECASE,
)


FETCH_URL_TOOL_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "fetch_url",
        "description": (
            "Fetch one to three webpages or online PDF files and return "
            "Browser-LLM-summarized content with Rationale, Evidence, "
            "and Summary sections."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "format": "uri",
                        "minLength": 1,
                    },
                    "minItems": 1,
                    "maxItems": 3,
                    "uniqueItems": True,
                    "description": (
                        "One to three distinct HTTP or HTTPS URLs to fetch."
                    ),
                },
                "purpose": {
                    "type": "string",
                    "minLength": 1,
                    "description": (
                        "A concise description of the information to extract "
                        "from the provided pages."
                    ),
                },
            },
            "required": ["url", "purpose"],
            "additionalProperties": False,
        },
    },
}


@dataclass
class ToolExecution:
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


def _required_config(config: dict[str, Any], key: str) -> Any:
    if key not in config:
        raise ValueError(f"fetch_url tool config requires {key!r}")
    return config[key]


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


def _build_browser_prompt(raw_content: str, purpose: str, max_completion_tokens: int) -> str:
    if purpose:
        context = (
            "\nIMPORTANT CONTEXT:\n"
            f'- The agent\'s IMMEDIATE PURPOSE for this page is: "{purpose}"\n\n'
            "Extract and summarize information directly relevant to that purpose.\n"
        )
    else:
        context = (
            "\nIMPORTANT CONTEXT:\n"
            "- The agent's IMMEDIATE PURPOSE for this page is to summarize key points and evidence.\n\n"
            "Extract and summarize the most useful information from this page.\n"
        )

    return f"""
Please process the following webpage or local file content and user goal.

## Webpage/Local file Content
{raw_content}

## User Goal
{context}

## Task Guidelines
1. Identify the parts of the source that directly address the goal.
2. Distinguish explicit source evidence from your own interpretation.
3. Preserve important qualifiers, dates, numbers, names, and surrounding context.
4. Do not invent facts or fill gaps using unsupported assumptions.
5. If the source does not contain enough relevant information, state that clearly.
6. Keep the response concise and within approximately {max_completion_tokens} tokens.
7. Return only the Markdown structure specified below.

## Output Format

## Rationale
Briefly explain which parts of the source are relevant and why.

## Evidence
Present the strongest source-grounded evidence. Preserve exact figures, dates,
names, and qualifications where important.

## Summary
Provide a concise answer to the goal based only on the evidence above.
"""


class MicrosoftBrowseTool:
    tool_schema = FETCH_URL_TOOL_SCHEMA

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        tool_schema: dict[str, Any] | None = None,
        *,
        api_key: str | None = None,
        api_keys: list[str] | None = None,
        endpoint: str | None = None,
        cache_path: str | None = None,
        raw_cache_path: str | None = None,
    ) -> None:
        if not isinstance(config, dict):
            raise ValueError("MicrosoftBrowseTool requires the fetch_url YAML mapping")
        cfg = config
        self.tool_schema = tool_schema or FETCH_URL_TOOL_SCHEMA

        self.api_keys = _coerce_api_keys(
            api_keys,
            api_key,
            cfg.get("ms_api_keys"),
            cfg.get("ms_api_key"),
        )
        self.api_key = self.api_keys[0] if self.api_keys else ""
        self.endpoint = endpoint if endpoint is not None else str(_required_config(cfg, "ms_endpoint"))
        self.content_format = str(_required_config(cfg, "content_format"))
        self.ms_max_length = int(_required_config(cfg, "ms_max_length"))
        self.live_crawl = str(_required_config(cfg, "live_crawl"))
        self.render_dynamic_pages = _bool_value(_required_config(cfg, "render_dynamic_pages"))

        self.timeout = float(_required_config(cfg, "timeout"))
        self.num_retries = int(_required_config(cfg, "num_retries"))
        self.max_urls_per_call = int(_required_config(cfg, "max_urls_per_call"))
        self.fallback_raw_chars = int(_required_config(cfg, "fallback_raw_chars"))

        self.jina_primary = _bool_value(_required_config(cfg, "jina_primary"))
        self.ms_fallback_enabled = cfg.get("ms_fallback_enabled", True)
        if type(self.ms_fallback_enabled) is not bool:
            raise TypeError("ms_fallback_enabled must be a boolean")
        self.jina_base_url = str(_required_config(cfg, "jina_base_url")).rstrip("/")
        self.jina_timeout = float(_required_config(cfg, "jina_timeout"))
        self.jina_num_retries = int(_required_config(cfg, "jina_num_retries"))
        self.ms_num_retries = int(_required_config(cfg, "ms_num_retries"))
        terminal_statuses = _required_config(cfg, "jina_terminal_target_statuses")
        self.jina_terminal_target_statuses = {int(status) for status in terminal_statuses}

        self.browser_llm_url = str(_required_config(cfg, "browser_llm_url"))
        self.browser_llm_key = str(_required_config(cfg, "browser_llm_key"))
        self.browser_llm_model = str(_required_config(cfg, "browser_llm_model"))
        self.browser_llm_max_input_tokens = int(_required_config(cfg, "browser_llm_max_input_tokens"))
        self.browser_llm_max_output_tokens = int(_required_config(cfg, "browser_llm_max_output_tokens"))
        browser_llm_concurrency = int(_required_config(cfg, "browser_llm_max_concurrency"))

        if self.ms_fallback_enabled and not self.api_keys:
            raise RuntimeError("fetch_url tool config requires at least one ms_api_key")
        if not self.ms_fallback_enabled and not self.jina_primary:
            raise ValueError("disabling Microsoft fallback requires jina_primary=true")
        if not self.browser_llm_url or not self.browser_llm_key or not self.browser_llm_model:
            raise RuntimeError("fetch_url tool config requires browser_llm_url, browser_llm_key, and browser_llm_model")
        if self.max_urls_per_call <= 0 or self.timeout <= 0 or self.num_retries <= 0:
            raise ValueError("browse limits, timeout, and retries must be positive")
        if self.ms_num_retries <= 0 or self.jina_num_retries <= 0 or self.jina_timeout <= 0:
            raise ValueError("fetch retry and timeout values must be positive")
        if (
            self.browser_llm_max_input_tokens <= 0
            or self.browser_llm_max_output_tokens <= 0
            or browser_llm_concurrency <= 0
        ):
            raise ValueError("Browser LLM token and concurrency limits must be positive")
        if self.jina_primary and not self.jina_base_url:
            raise ValueError("jina_base_url is required when jina_primary is enabled")

        ms_concurrency = int(_required_config(cfg, "max_concurrency"))
        jina_concurrency = int(_required_config(cfg, "jina_max_concurrency"))
        if ms_concurrency <= 0 or jina_concurrency <= 0:
            raise ValueError("fetch concurrency values must be positive")
        self._ms_sema = asyncio.Semaphore(ms_concurrency)
        self._jina_sema = asyncio.Semaphore(jina_concurrency)
        self._browser_llm_sema = asyncio.Semaphore(browser_llm_concurrency)

        self._browser_llm_client = AsyncOpenAI(api_key=self.browser_llm_key, base_url=self.browser_llm_url)
        resolved_cache_path = cache_path if cache_path is not None else str(_required_config(cfg, "cache_path"))
        self.cache = AsyncSqliteCache(resolved_cache_path)
        resolved_raw_cache_path = (
            raw_cache_path if raw_cache_path is not None else str(_required_config(cfg, "raw_cache_path"))
        )
        self.raw_cache = AsyncSqliteCache(
            resolved_raw_cache_path
        )
        self._cache_locks: dict[str, asyncio.Lock] = {}
        self._raw_cache_locks: dict[str, asyncio.Lock] = {}
        self._tiktoken_enc = None

    def _choose_api_key(self) -> str:
        return random.choice(self.api_keys)

    async def execute(self, parameters: dict[str, Any]) -> ToolExecution:
        try:
            urls = _coerce_str_list(parameters.get("url"), key="url", limit=self.max_urls_per_call)
        except ValueError as exc:
            return ToolExecution(f"(fetch_url error: {exc})", {"success": False, "tool_call_failures": 1})

        purpose = str(parameters.get("purpose") or "").strip()
        results = await asyncio.gather(
            *(self._fetch_and_summarize(url, purpose) for url in urls),
            return_exceptions=True,
        )

        blocks: list[str] = []
        cache_hits = 0
        raw_cache_hits = 0
        filtered = 0
        successes = 0
        llm_fallbacks = 0
        for url, result in zip(urls, results, strict=True):
            if isinstance(result, Exception):
                blocks.append(f"URL: {url}\n\nSummary:\n(error: {type(result).__name__}: {result})")
                continue
            text, cache_hit, raw_cache_hit, was_filtered, success, llm_fallback = result
            blocks.append(text)
            cache_hits += int(cache_hit)
            raw_cache_hits += int(raw_cache_hit)
            filtered += int(was_filtered)
            successes += int(success)
            llm_fallbacks += int(llm_fallback)

        success = successes > 0
        return ToolExecution(
            "\n\n---\n\n".join(blocks),
            {
                "success": success,
                "tool_call_count": 1,
                "tool_call_successes": int(success),
                "tool_call_failures": int(not success),
                "tool_unit_count": len(urls),
                "tool_unit_successes": successes,
                "tool_unit_failures": len(urls) - successes,
                "num_urls": len(urls),
                "cache_hits": cache_hits,
                "raw_cache_hits": raw_cache_hits,
                "filtered": filtered,
                "llm_fallbacks": llm_fallbacks,
            },
        )

    async def _fetch_and_summarize(self, url: str, purpose: str) -> tuple[str, bool, bool, bool, bool, bool]:
        key = self._cache_key(url, purpose)
        cached = await self.cache.get(key)
        if cached is not None:
            return cached, True, False, False, True, False

        lock = self._cache_locks.setdefault(key, asyncio.Lock())
        async with lock:
            cached = await self.cache.get(key)
            if cached is not None:
                return cached, True, False, False, True, False

            raw, sentinel, raw_cache_hit = await self._fetch_raw(url)
            if sentinel is not None:
                return f"URL: {url}\n\nSummary:\n{sentinel}", False, raw_cache_hit, True, False, False

            summary = await self._browser_llm_summarize(raw, purpose)
            if not summary.strip():
                raw_head = raw[: self.fallback_raw_chars]
                block = f"URL: {url}\n\nSummary:\n(browser LLM failed, returning raw head)\n{raw_head}"
                return block, False, raw_cache_hit, False, False, True

            block = f"URL: {url}\n\nSummary:\n{summary}"
            await self.cache.set(key, block)
            return block, False, raw_cache_hit, False, True, False

    def _cache_key(self, url: str, purpose: str) -> str:
        payload = json.dumps(
            {
                "endpoint": self.endpoint,
                "content_format": self.content_format,
                "ms_max_length": self.ms_max_length,
                "live_crawl": self.live_crawl,
                "render_dynamic_pages": self.render_dynamic_pages,
                "jina_primary": self.jina_primary,
                "ms_fallback_enabled": self.ms_fallback_enabled,
                "jina_base_url": self.jina_base_url,
                "jina_num_retries": self.jina_num_retries,
                "ms_num_retries": self.ms_num_retries,
                "jina_terminal_target_statuses": sorted(self.jina_terminal_target_statuses),
                "browser_llm_model": self.browser_llm_model,
                "browser_llm_max_output_tokens": self.browser_llm_max_output_tokens,
                "url": url,
                "purpose": purpose,
            },
            sort_keys=True,
            ensure_ascii=False,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    async def _fetch_raw(self, url: str) -> tuple[str, str | None, bool]:
        if not self._is_supported_url(url):
            return "", f"(invalid URL: {url})", False

        cached = await self.raw_cache.get(url)
        if cached is not None:
            return cached, None, True

        lock = self._raw_cache_locks.setdefault(url, asyncio.Lock())
        async with lock:
            cached = await self.raw_cache.get(url)
            if cached is not None:
                return cached, None, True

            raw, sentinel = await self._fetch_raw_uncached(url)
            if raw:
                await self.raw_cache.set(url, raw)
            return raw, sentinel, False

    async def _fetch_raw_uncached(self, url: str) -> tuple[str, str | None]:
        if self.jina_primary:
            raw, sentinel = await self._jina_fetch(url)
            if raw:
                return raw, None
            if sentinel is not None:
                return "", sentinel
        if not self.ms_fallback_enabled:
            return "", f"(Jina could not fetch URL: {url})"
        return await self._ms_browse(url)

    @staticmethod
    def _is_supported_url(url: str) -> bool:
        try:
            parsed = httpx.URL(url)
        except Exception:
            return False
        return parsed.scheme in {"http", "https"} and bool(parsed.host)

    async def _jina_fetch(self, url: str) -> tuple[str, str | None]:
        reader_url = f"{self.jina_base_url}/{url}"
        async with self._jina_sema:
            async with httpx.AsyncClient(timeout=self.jina_timeout) as client:
                last_error = ""
                for attempt in range(self.jina_num_retries):
                    try:
                        response = await client.get(reader_url)
                        text = response.text or ""
                        if response.status_code != 200:
                            last_error = f"Jina HTTP {response.status_code}: {text[:200]}"
                            if attempt + 1 < self.jina_num_retries:
                                await asyncio.sleep(2**attempt)
                            continue

                        target_status = _extract_target_status(text)
                        if target_status in self.jina_terminal_target_statuses:
                            return "", f"(URL not found by Jina: target HTTP {target_status} for {url})"
                        if target_status is not None or _ANTI_BOT_RE.search(text):
                            return "", None
                        if not _extract_markdown_content(text).strip():
                            return "", None
                        return text, None
                    except (httpx.TimeoutException, httpx.HTTPError) as exc:
                        last_error = f"{type(exc).__name__}: {exc}"
                        if attempt + 1 < self.jina_num_retries:
                            await asyncio.sleep(2**attempt)

                logger.info("Jina failed after %d attempts for %s: %s", self.jina_num_retries, url, last_error)
                return "", None

    async def _ms_browse(self, url: str) -> tuple[str, str | None]:
        body: dict[str, Any] = {
            "url": url,
            "contentFormat": self.content_format,
            "maxLength": self.ms_max_length,
            "liveCrawl": self.live_crawl,
        }
        if self.render_dynamic_pages:
            body["renderDynamicPages"] = True
        async with self._ms_sema:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                last_exc: Exception | None = None
                for attempt in range(self.ms_num_retries):
                    try:
                        headers = {"x-apikey": self._choose_api_key(), "content-type": "application/json"}
                        response = await client.post(self.endpoint, headers=headers, json=body)
                        if response.status_code in _FATAL_HTTP_STATUSES:
                            raise RuntimeError(f"MS browse HTTP {response.status_code}: {response.text[:200]}")
                        if response.status_code == 404:
                            return "", f"(URL not found: {url})"
                        if response.status_code == 202:
                            data = response.json() if response.content else {}
                            wait_s = _parse_retry_after(data.get("retryAfter", "10s"))
                            return "", f"(browse pending: retryAfter {wait_s:g}s for {url})"
                        if response.status_code in _RETRY_HTTP_STATUSES:
                            if attempt + 1 < self.ms_num_retries:
                                await asyncio.sleep(2**attempt)
                            continue
                        if response.status_code != 200:
                            return "", f"(browse HTTP {response.status_code}: {response.text[:200]})"

                        data = response.json()
                        if isinstance(data, dict) and "errorCode" in data:
                            msg = data.get("userMessage") or data.get("errorCode")
                            return "", f"(URL filtered by API: {data['errorCode']} - {msg})"
                        content = (data or {}).get("content", "") or ""
                        if not content:
                            return "", f"(URL returned no content: {url})"
                        return content, None
                    except (httpx.TimeoutException, httpx.HTTPError) as exc:
                        last_exc = exc
                        if attempt + 1 < self.ms_num_retries:
                            await asyncio.sleep(2**attempt)

                logger.warning("MS browse failed after %d attempts for %s: %s", self.ms_num_retries, url, last_exc)
                return "", f"(browse timeout/network error for {url}: {last_exc})"

    async def _browser_llm_summarize(self, raw_content: str, purpose: str) -> str:
        truncated = self._truncate_to_tokens(raw_content, self.browser_llm_max_input_tokens)
        prompt = _build_browser_prompt(truncated, purpose, self.browser_llm_max_output_tokens)
        last_exc: Exception | None = None

        async with self._browser_llm_sema:
            for attempt in range(self.num_retries):
                try:
                    response = await asyncio.wait_for(
                        self._browser_llm_client.responses.create(
                            model=self.browser_llm_model,
                            input=prompt,
                            max_output_tokens=self.browser_llm_max_output_tokens,
                        ),
                        timeout=self.timeout,
                    )
                    content = _extract_responses_text(response)
                    if content.strip():
                        return content
                except Exception as exc:
                    last_exc = exc
                if attempt + 1 < self.num_retries:
                    await asyncio.sleep(2**attempt)

        logger.warning("Browser LLM failed after %d attempts: %s", self.num_retries, last_exc)
        return ""

    def _truncate_to_tokens(self, text: str, max_tokens: int) -> str:
        if not text:
            return ""
        enc = self._enc()
        tokens = enc.encode(text)
        if len(tokens) <= max_tokens:
            return text
        return enc.decode(tokens[:max_tokens])

    def _enc(self):
        if self._tiktoken_enc is None:
            import tiktoken

            self._tiktoken_enc = tiktoken.get_encoding("cl100k_base")
        return self._tiktoken_enc


def _bool_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() not in {"0", "false", "no", "off", ""}
    return bool(value)


def _parse_retry_after(spec: Any) -> float:
    if isinstance(spec, (int, float)):
        return float(spec)
    if isinstance(spec, str):
        try:
            return float(spec.strip().rstrip("s"))
        except ValueError:
            return 10.0
    return 10.0


def _extract_target_status(text: str) -> int | None:
    match = _TARGET_STATUS_RE.search(text or "")
    return int(match.group(1)) if match else None


def _extract_markdown_content(text: str) -> str:
    marker = "Markdown Content:"
    return text.split(marker, 1)[1] if marker in text else text


def _extract_responses_text(response: Any) -> str:
    text = getattr(response, "output_text", None)
    if isinstance(text, str) and text.strip():
        return text

    parts: list[str] = []
    for item in getattr(response, "output", []) or []:
        if getattr(item, "type", None) != "message":
            continue
        for content in getattr(item, "content", []) or []:
            if getattr(content, "type", None) in {"output_text", "text"}:
                value = getattr(content, "text", None)
                if isinstance(value, str):
                    parts.append(value)
    return "".join(parts)


_DEFAULT_TOOL: MicrosoftBrowseTool | None = None


def _default_tool() -> MicrosoftBrowseTool:
    global _DEFAULT_TOOL
    if _DEFAULT_TOOL is None:
        from adaptive_branching.src.deep_research.agent import BROWSE_TOOL

        _DEFAULT_TOOL = BROWSE_TOOL
    return _DEFAULT_TOOL


tool_specs = [FETCH_URL_TOOL_SCHEMA]


async def execute_tool(name: str, params: dict[str, Any]) -> str:
    if name != "fetch_url":
        return f"Unknown tool: {name}"
    result = await _default_tool().execute(params)
    return result.text
