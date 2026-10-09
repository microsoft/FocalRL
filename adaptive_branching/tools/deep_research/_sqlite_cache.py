"""Small async-friendly SQLite string cache for rollout tools."""

from __future__ import annotations

import asyncio
import sqlite3
import time
from pathlib import Path


class AsyncSqliteCache:
    def __init__(self, path: str | Path):
        self.path = Path(path).expanduser()
        self._ready = False
        self._lock = asyncio.Lock()

    async def _ensure_ready(self) -> None:
        if self._ready:
            return
        async with self._lock:
            if self._ready:
                return
            await asyncio.to_thread(self._init_db)
            self._ready = True

    def _init_db(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.path) as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS cache "
                "(key TEXT PRIMARY KEY, value TEXT NOT NULL, updated_at REAL NOT NULL)"
            )
            conn.commit()

    async def get(self, key: str) -> str | None:
        await self._ensure_ready()
        return await asyncio.to_thread(self._get_sync, key)

    def _get_sync(self, key: str) -> str | None:
        with sqlite3.connect(self.path) as conn:
            row = conn.execute("SELECT value FROM cache WHERE key = ?", (key,)).fetchone()
        return row[0] if row else None

    async def set(self, key: str, value: str) -> None:
        await self._ensure_ready()
        await asyncio.to_thread(self._set_sync, key, value)

    def _set_sync(self, key: str, value: str) -> None:
        with sqlite3.connect(self.path) as conn:
            conn.execute(
                "INSERT INTO cache(key, value, updated_at) VALUES (?, ?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at",
                (key, value, time.time()),
            )
            conn.commit()
