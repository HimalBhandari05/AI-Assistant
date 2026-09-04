import asyncio
import os
import sys
import time
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.main import app
from app.llm import AssistantResponse
from app.rate_limiter import InMemoryRateLimiter, rate_limiter
from app.cache import response_cache

client = TestClient(app)


def test_requests_under_limit():
    """Test 1: Send requests within the configured limit; verify all are accepted."""
    rate_limiter.reset()
    response_cache.clear()
    rate_limiter.requests_limit = 3
    rate_limiter.window_seconds = 60.0

    mock_response = AssistantResponse(answer="Accepted answer", topic="General")

    with patch("app.main.get_llm_response", return_value=mock_response) as mock_llm:
        headers = {"X-Forwarded-For": "10.0.0.1"}
        for i in range(3):
            res = client.post("/ask", json={"question": f"Question {i}"}, headers=headers)
            assert res.status_code == 200, f"Request {i+1} failed with status {res.status_code}: {res.text}"
            assert res.json()["answer"] == "Accepted answer"

        assert mock_llm.call_count == 3
        print("✓ Test 1 — All 3 requests under limit were successfully accepted (HTTP 200)")


def test_request_over_limit_and_pipeline_not_executed():
    """Test 2: Exceed rate limit on 4th request; verify HTTP 429 and LLM pipeline NOT executed."""
    rate_limiter.reset()
    response_cache.clear()
    rate_limiter.requests_limit = 3
    rate_limiter.window_seconds = 60.0

    mock_response = AssistantResponse(answer="Accepted answer", topic="General")

    with patch("app.main.get_llm_response", return_value=mock_response) as mock_llm:
        headers = {"X-Forwarded-For": "10.0.0.2"}

        # First 3 requests succeed
        for i in range(3):
            res = client.post("/ask", json={"question": f"Valid query {i}"}, headers=headers)
            assert res.status_code == 200

        assert mock_llm.call_count == 3

        # 4th request must be rejected with HTTP 429
        res_rejected = client.post("/ask", json={"question": "Rejected query"}, headers=headers)
        assert res_rejected.status_code == 429
        assert "Rate limit exceeded" in res_rejected.json()["detail"]
        assert "Retry-After" in res_rejected.headers
        retry_after = int(res_rejected.headers["Retry-After"])
        assert 0 < retry_after <= 60

        # Verify LLM pipeline was NOT executed for the 4th request
        assert mock_llm.call_count == 3, f"Expected 3 LLM calls, but got {mock_llm.call_count}"
        print(f"✓ Test 2 — 4th request rejected with HTTP 429 (Retry-After: {retry_after}s), LLM NOT executed")


def test_different_client_ips_independent():
    """Test 3: Verify separate client IPs have independent rate limit tracking."""
    rate_limiter.reset()
    response_cache.clear()
    rate_limiter.requests_limit = 2
    rate_limiter.window_seconds = 60.0

    mock_response = AssistantResponse(answer="Accepted", topic="General")

    with patch("app.main.get_llm_response", return_value=mock_response) as mock_llm:
        headers_a = {"X-Forwarded-For": "192.168.1.100"}
        headers_b = {"X-Forwarded-For": "192.168.1.200"}

        # Client A exhausts quota (2 requests)
        assert client.post("/ask", json={"question": "A1"}, headers=headers_a).status_code == 200
        assert client.post("/ask", json={"question": "A2"}, headers=headers_a).status_code == 200
        # Client A 3rd request -> 429
        assert client.post("/ask", json={"question": "A3"}, headers=headers_a).status_code == 429

        # Client B sends first request -> should succeed (HTTP 200)
        res_b = client.post("/ask", json={"question": "B1"}, headers=headers_b)
        assert res_b.status_code == 200

        print("✓ Test 3 — Independent limits per client IP verified (Client A blocked, Client B allowed)")


def test_window_expiration():
    """Test 4: Verify requests become available again once the rate limit window expires."""
    rate_limiter.reset()
    response_cache.clear()
    rate_limiter.requests_limit = 2
    rate_limiter.window_seconds = 0.25  # 250ms window for fast testing

    mock_response = AssistantResponse(answer="Accepted", topic="General")

    with patch("app.main.get_llm_response", return_value=mock_response):
        headers = {"X-Forwarded-For": "10.0.0.50"}

        # Request 1 & 2 succeed
        assert client.post("/ask", json={"question": "Q1"}, headers=headers).status_code == 200
        assert client.post("/ask", json={"question": "Q2"}, headers=headers).status_code == 200

        # Request 3 fails immediately
        assert client.post("/ask", json={"question": "Q3"}, headers=headers).status_code == 429

        # Sleep past the window
        time.sleep(0.30)

        # Request 4 should now succeed
        res_after = client.post("/ask", json={"question": "Q4"}, headers=headers)
        assert res_after.status_code == 200
        print("✓ Test 4 — Window expiration verified: requests allowed after window elapses")


def test_health_check_not_rate_limited():
    """Test 5: Verify GET / health check endpoint is never rate limited."""
    rate_limiter.reset()
    response_cache.clear()
    rate_limiter.requests_limit = 1
    rate_limiter.window_seconds = 60.0

    mock_response = AssistantResponse(answer="Accepted", topic="General")

    with patch("app.main.get_llm_response", return_value=mock_response):
        headers = {"X-Forwarded-For": "10.0.0.99"}

        # Exhaust POST /ask
        assert client.post("/ask", json={"question": "Q1"}, headers=headers).status_code == 200
        assert client.post("/ask", json={"question": "Q2"}, headers=headers).status_code == 429

        # GET / must still succeed with HTTP 200
        for _ in range(5):
            res_health = client.get("/", headers=headers)
            assert res_health.status_code == 200
            assert res_health.json()["status"] == "ok"

        print("✓ Test 5 — GET / health check remains accessible and un-throttled")


def test_memory_cleanup():
    """Test 6: Verify internal rate limiter purges stale/expired entries."""
    limiter = InMemoryRateLimiter(requests_limit=5, window_seconds=0.1)

    # Populate multiple simulated client IPs
    current_time = time.time()
    limiter._requests["1.1.1.1"] = [current_time - 1.0]  # Expired
    limiter._requests["2.2.2.2"] = [current_time - 0.5]  # Expired
    limiter._requests["3.3.3.3"] = [current_time]        # Active

    limiter._cleanup_stale_entries(current_time)

    assert "1.1.1.1" not in limiter._requests
    assert "2.2.2.2" not in limiter._requests
    assert "3.3.3.3" in limiter._requests
    print("✓ Test 6 — In-memory stale entry cleanup verified")


if __name__ == "__main__":
    print("=== STARTING TASK 2 PHASE 4 RATE LIMITING TESTS ===\n")
    test_requests_under_limit()
    test_request_over_limit_and_pipeline_not_executed()
    test_different_client_ips_independent()
    test_window_expiration()
    test_health_check_not_rate_limited()
    test_memory_cleanup()
    print("\nALL RATE LIMITING TESTS PASSED SUCCESSFULLY!")
