import asyncio
import os
import sys
from unittest.mock import MagicMock, patch
import httpx
from google.genai.errors import APIError

from fastapi.testclient import TestClient

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.main import app
from app.llm import execute_with_retry, is_transient_error, AssistantResponse
from app.rate_limiter import rate_limiter
from app.cache import response_cache

client = TestClient(app)


def setup_function():
    """Reset rate limiter, response cache, and environment defaults before each test."""
    rate_limiter.reset()
    response_cache.clear()
    os.environ["LLM_PROVIDER"] = "gemini"
    os.environ["RATE_LIMIT_REQUESTS"] = "10"
    os.environ["RATE_LIMIT_WINDOW_SECONDS"] = "60"


def test_is_transient_error_classification():
    """Verify that is_transient_error accurately distinguishes transient from permanent errors."""
    # Transient errors -> should be True
    assert is_transient_error(httpx.ConnectError("Connection refused")) is True
    assert is_transient_error(httpx.ReadTimeout("Read timed out")) is True
    assert is_transient_error(ConnectionResetError("Connection reset by peer")) is True
    assert is_transient_error(TimeoutError("Operation timed out")) is True
    assert is_transient_error(RuntimeError("Service unavailable 503")) is True

    # Permanent errors -> should be False
    assert is_transient_error(ValueError("GEMINI_API_KEY is not configured")) is False
    assert is_transient_error(ValueError("Unsupported LLM_PROVIDER")) is False
    assert is_transient_error(RuntimeError("401 Unauthorized: Invalid API key")) is False
    assert is_transient_error(RuntimeError("403 Permission Denied")) is False
    assert is_transient_error(RuntimeError("ResourceExhausted: 429 Quota exceeded")) is False
    assert is_transient_error(RuntimeError("rate limit reached")) is False
    print("✓ is_transient_error classification verified")


def test_transient_failure_recovery():
    """Test 1: Mock provider fails once with transient error and succeeds on retry."""
    call_count = 0

    def flaky_func():
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise httpx.ConnectError("Temporary network glitch")
        return AssistantResponse(answer="Recovered answer", topic="Reliability")

    result = execute_with_retry(flaky_func, max_retries=2, base_delay=0.01)
    assert call_count == 2
    assert result.answer == "Recovered answer"
    print("✓ Test 1 — Transient failure recovered on retry 2/3")


def test_persistent_transient_failure():
    """Test 2: Mock provider fails continuously with transient error; verify bounded to 3 attempts."""
    call_count = 0

    def always_failing_func():
        nonlocal call_count
        call_count += 1
        raise httpx.ConnectTimeout("Connection timed out repeatedly")

    try:
        execute_with_retry(always_failing_func, max_retries=2, base_delay=0.01)
        assert False, "Expected failure after exhausting retries"
    except httpx.ConnectTimeout:
        assert call_count == 3, f"Expected exactly 3 attempts (1 initial + 2 retries), got {call_count}"
        print(f"✓ Test 2 — Persistent failure bounded to exactly {call_count} attempts (no infinite retries)")


def test_permanent_authentication_failure():
    """Test 3: Permanent auth error must fail immediately with 1 attempt and 0 retries."""
    call_count = 0

    def auth_failing_func():
        nonlocal call_count
        call_count += 1
        raise RuntimeError("401 Unauthorized: invalid_argument for api_key")

    try:
        execute_with_retry(auth_failing_func, max_retries=2, base_delay=0.01)
        assert False, "Expected immediate failure"
    except RuntimeError:
        assert call_count == 1, f"Expected exactly 1 attempt with 0 retries, got {call_count}"
        print(f"✓ Test 3 — Permanent failure stopped immediately after {call_count} attempt (0 retries)")


def test_quota_rate_limit_protection():
    """Test 4: Provider quota/rate-limit error must NOT be retried (Gemini Quota Protection)."""
    call_count = 0

    def quota_exceeded_func():
        nonlocal call_count
        call_count += 1
        raise RuntimeError("429 ResourceExhausted: Quota exceeded for model")

    try:
        execute_with_retry(quota_exceeded_func, max_retries=2, base_delay=0.01)
        assert False, "Expected immediate quota error"
    except RuntimeError:
        assert call_count == 1, f"Expected exactly 1 attempt with 0 retries for quota errors, got {call_count}"
        print(f"✓ Test 4 — Quota protection verified: {call_count} attempt (0 retries on 429/quota)")


def test_api_endpoint_error_handling():
    """Test 5 & 6: Test HTTP status codes and error responses from FastAPI /ask endpoint."""
    rate_limiter.reset()
    response_cache.clear()
    os.environ["LLM_PROVIDER"] = "gemini"

    # 5.1 Invalid payload validation
    res = client.post("/ask", json={"question": ""})
    assert res.status_code == 422
    res = client.post("/ask", json={"question": "   "})
    assert res.status_code == 422
    print("✓ Test 5.1 — Empty/whitespace input validation (HTTP 422) verified")

    # 5.2 Missing API Key handling
    with patch("app.llm.os.getenv") as mock_env:
        mock_env.side_effect = lambda k, default=None: "gemini" if k == "LLM_PROVIDER" else None
        res = client.post("/ask", json={"question": "What is Python?"})
        assert res.status_code == 500
        assert "API key is not configured" in res.json()["detail"]
        print("✓ Test 5.2 — Missing API key returns safe HTTP 500 error")

    # 5.3 Upstream Quota exceeded error (HTTP 429)
    with patch("app.llm.retrieve_relevant_chunks", return_value=[]), \
         patch("app.llm._call_gemini", side_effect=RuntimeError("429 ResourceExhausted: Quota exceeded")):
        res = client.post("/ask", json={"question": "Test question"})
        assert res.status_code == 429
        assert "quota or rate limit exceeded" in res.json()["detail"].lower()
        print("✓ Test 5.3 — Upstream quota exhaustion mapped to HTTP 429")

    # 5.4 Upstream Transient error exhausted (HTTP 503)
    with patch("app.llm.retrieve_relevant_chunks", return_value=[]), \
         patch("app.llm._call_gemini", side_effect=RuntimeError("Service unavailable after retries")), \
         patch("app.llm._call_ollama", side_effect=RuntimeError("Ollama fallback failed")), \
         patch("app.llm.time.sleep", return_value=None):
        res = client.post("/ask", json={"question": "Test question"})
        assert res.status_code == 503
        assert "temporarily unavailable" in res.json()["detail"].lower()
        print("✓ Test 5.4 — Exhausted transient error mapped to HTTP 503")


if __name__ == "__main__":
    print("=== STARTING TASK 2 PHASE 3 RETRY & RELIABILITY TESTS ===\n")
    test_is_transient_error_classification()
    test_transient_failure_recovery()
    test_persistent_transient_failure()
    test_permanent_authentication_failure()
    test_quota_rate_limit_protection()
    test_api_endpoint_error_handling()
    print("\nALL RETRY AND ERROR HANDLING TESTS PASSED SUCCESSFULLY!")
