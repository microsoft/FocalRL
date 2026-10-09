from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest


for _name in ("LLM_JUDGE_KEY", "LLM_API_KEY", "MS_API_KEYS"):
    os.environ[_name] = "test-only-unused"

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@pytest.fixture(autouse=True)
def _test_api_keys(monkeypatch):
    """Provide non-secret keys required by fail-fast YAML interpolation."""
    monkeypatch.setenv("LLM_JUDGE_KEY", "test-judge-key")
    monkeypatch.setenv("LLM_API_KEY", "test-browser-key")

    monkeypatch.setenv("MS_API_KEYS", "test-search-key")
