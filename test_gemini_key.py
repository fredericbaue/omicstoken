import os
import pytest
from summarizer import generate_summary

"""
This test ensures that:
- If a Gemini API key exists (via .env or system env), it is valid and non-empty.
- If no key exists, the test SKIPS rather than crashing pytest.

This keeps pytest stable in both developer and CI environments.
"""

api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")


def test_gemini_api_key_present_or_skip():
    if not api_key:
        pytest.skip("No GEMINI_API_KEY/GOOGLE_API_KEY configured; skipping Gemini tests.")

    # Basic sanity check (do NOT print or slice the key)
    assert isinstance(api_key, str)
    assert len(api_key) > 0


def test_summarizer_handles_missing_key_gracefully():
    """
    Even when a key is missing or invalid, generate_summary() should return
    a structured error dict, NOT crash pytest.
    """
    result = generate_summary("FAKE_RUN_ID_FOR_TESTING")

    assert isinstance(result, dict)
    # We do not enforce success, only that the function never throws.
    assert "error" in result or "summary" in result or "summary_text" in result
