import importlib
import os

import pytest

import summarizer


def test_generate_summary_without_api_key(monkeypatch):
    """When no Gemini key is configured, summarizer should return an error dict, not crash."""
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    importlib.reload(summarizer)

    result = summarizer.generate_summary("test-run-without-key")

    assert isinstance(result, dict)
    assert result.get("run_id") == "test-run-without-key"
    assert "error" in result


def test_generate_summary_smoke_with_api_key(monkeypatch):
    """If a key is present, ensure generate_summary returns a structured dict (error or summary)."""
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        pytest.skip("No GEMINI_API_KEY/GOOGLE_API_KEY set; skipping smoke test.")

    monkeypatch.setenv("GEMINI_API_KEY", api_key)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    importlib.reload(summarizer)

    result = summarizer.generate_summary("test-run-with-key")

    assert isinstance(result, dict)
    assert result.get("run_id") == "test-run-with-key"
    assert ("summary_text" in result) or ("error" in result)
