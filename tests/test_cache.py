import asyncio
import os
import sys
import time
from unittest.mock import MagicMock, patch
import httpx
from fastapi.testclient import TestClient

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.main import app
from app.llm import AssistantResponse
from app.cache import response_cache, normalize_question, make_cache_key
from app.rate_limiter import rate_limiter

client = TestClient(app)


def setup_function():
    """Reset rate limiter, response cache, and environment before each test."""
    rate_limiter.reset()
    response_cache.clear()
    response_cache.enabled = True
    response_cache.ttl_seconds = 300.0
    response_cache.max_size = 100
    os.environ["LLM_PROVIDER"] = "gemini"
    os.environ["CACHE_ENABLED"] = "true"
    os.environ["CACHE_TTL_SECONDS"] = "300"
    os.environ["CACHE_MAX_SIZE"] = "100"


def test_cache_miss():
    """Test 1: First request is a Cache MISS; LLM executes and stores response."""
    setup_function()
    mock_resp = AssistantResponse(answer="RAG combines search with generation.", topic="RAG")

    with patch("app.main.get_llm_response", return_value=mock_resp) as mock_llm:
        res = client.post("/ask", json={"question": "What is RAG?"})
        assert res.status_code == 200
        assert res.json()["answer"] == "RAG combines search with generation."

        # LLM pipeline should have been invoked exactly once
        assert mock_llm.call_count == 1
        assert response_cache.hits == 0
        assert response_cache.misses == 1
        assert len(response_cache._cache) == 1
        print("✓ Test 1 — Cache Miss verified: LLM executed and response cached")


def test_cache_hit():
    """Test 2: Second identical request is a Cache HIT; skips LLM and RAG."""
    setup_function()
    mock_resp = AssistantResponse(answer="RAG combines search with generation.", topic="RAG")

    with patch("app.main.get_llm_response", return_value=mock_resp) as mock_llm:
        # Request 1: Miss
        res1 = client.post("/ask", json={"question": "What is RAG?"})
        assert res1.status_code == 200
        assert mock_llm.call_count == 1

        # Request 2: Hit
        res2 = client.post("/ask", json={"question": "What is RAG?"})
        assert res2.status_code == 200
        assert res2.json()["answer"] == "RAG combines search with generation."

        # LLM pipeline must NOT have been called again
        assert mock_llm.call_count == 1
        assert response_cache.hits == 1
        assert response_cache.misses == 1
        print("✓ Test 2 — Cache Hit verified: LLM execution bypassed")


def test_key_normalization():
    """Test 3: Different casing and whitespace variants map to the same cache key."""
    setup_function()
    mock_resp = AssistantResponse(answer="Normalized Answer", topic="AI")

    with patch("app.main.get_llm_response", return_value=mock_resp) as mock_llm:
        variations = [
            "What is RAG?",
            "  what is rag?  ",
            "WHAT IS RAG?",
            "What   is   RAG?",
        ]

        # First variant triggers MISS
        res1 = client.post("/ask", json={"question": variations[0]})
        assert res1.status_code == 200
        assert mock_llm.call_count == 1

        # All subsequent variants must trigger HITs
        for q in variations[1:]:
            res = client.post("/ask", json={"question": q})
            assert res.status_code == 200
            assert res.json()["answer"] == "Normalized Answer"

        assert mock_llm.call_count == 1
        assert response_cache.hits == 3
        print(f"✓ Test 3 — Question normalization verified across {len(variations)} variants")


def test_ttl_expiration():
    """Test 4: Expired cache entries are removed and LLM executes again."""
    setup_function()
    response_cache.ttl_seconds = 0.20  # 200ms TTL
    mock_resp = AssistantResponse(answer="Fresh Answer", topic="General")

    with patch("app.main.get_llm_response", return_value=mock_resp) as mock_llm:
        # Request 1: MISS & cached
        assert client.post("/ask", json={"question": "What is binary search?"}).status_code == 200
        assert mock_llm.call_count == 1

        # Immediate repeat: HIT
        assert client.post("/ask", json={"question": "What is binary search?"}).status_code == 200
        assert mock_llm.call_count == 1

        # Sleep past TTL
        time.sleep(0.25)

        # Request 3: Expired -> MISS & LLM re-invoked
        res3 = client.post("/ask", json={"question": "What is binary search?"})
        assert res3.status_code == 200
        assert mock_llm.call_count == 2
        print("✓ Test 4 — TTL expiration verified: expired item evicted and recomputed")


def test_maximum_size_lru_eviction():
    """Test 5: When cache exceeds CACHE_MAX_SIZE, oldest entry is evicted."""
    setup_function()
    response_cache.max_size = 2
    mock_resp = AssistantResponse(answer="Generic Answer", topic="Test")

    with patch("app.main.get_llm_response", return_value=mock_resp):
        # Insert Q1 and Q2 (cache size = 2)
        client.post("/ask", json={"question": "Question 1"})
        client.post("/ask", json={"question": "Question 2"})
        assert len(response_cache._cache) == 2

        # Insert Q3 -> Q1 should be evicted (cache size remains 2)
        client.post("/ask", json={"question": "Question 3"})
        assert len(response_cache._cache) == 2

        # Check keys in cache
        key1 = make_cache_key("gemini", "Question 1")
        key2 = make_cache_key("gemini", "Question 2")
        key3 = make_cache_key("gemini", "Question 3")

        assert key1 not in response_cache._cache
        assert key2 in response_cache._cache
        assert key3 in response_cache._cache
        print("✓ Test 5 — Cache max size (2) and LRU eviction verified")


def test_errors_are_not_cached():
    """Test 6: Failed LLM requests are never stored in cache."""
    setup_function()

    # Step 1: Failed request
    with patch("app.main.get_llm_response", side_effect=httpx.ConnectError("Service unavailable")):
        res1 = client.post("/ask", json={"question": "Failing query"})
        assert res1.status_code == 503
        assert len(response_cache._cache) == 0

    # Step 2: Next request should attempt LLM again
    mock_success = AssistantResponse(answer="Recovered Answer", topic="Reliability")
    with patch("app.main.get_llm_response", return_value=mock_success) as mock_llm:
        res2 = client.post("/ask", json={"question": "Failing query"})
        assert res2.status_code == 200
        assert res2.json()["answer"] == "Recovered Answer"
        assert mock_llm.call_count == 1
        assert len(response_cache._cache) == 1
        print("✓ Test 6 — Errors are NOT cached verified")


def test_provider_isolation():
    """Test 7: Cache entries are partitioned by provider (Gemini cache not returned to Ollama)."""
    setup_function()
    gemini_resp = AssistantResponse(answer="Gemini Answer", topic="Cloud")
    ollama_resp = AssistantResponse(answer="Ollama Answer", topic="Local")

    # 1. Query under Gemini
    os.environ["LLM_PROVIDER"] = "gemini"
    with patch("app.main.get_llm_response", return_value=gemini_resp) as mock_llm:
        res1 = client.post("/ask", json={"question": "What is Python?"})
        assert res1.status_code == 200
        assert res1.json()["answer"] == "Gemini Answer"
        assert mock_llm.call_count == 1

    # 2. Switch provider to Ollama and query same question
    os.environ["LLM_PROVIDER"] = "ollama"
    with patch("app.main.get_llm_response", return_value=ollama_resp) as mock_llm:
        res2 = client.post("/ask", json={"question": "What is Python?"})
        assert res2.status_code == 200
        assert res2.json()["answer"] == "Ollama Answer"
        # Ollama must have been executed because the cache key is provider-isolated
        assert mock_llm.call_count == 1

    print("✓ Test 7 — Provider isolation verified (separate cache keys per provider)")


def test_rate_limiter_compatibility():
    """Test 8: Rate limiter takes precedence before cache lookup."""
    setup_function()
    rate_limiter.requests_limit = 2
    rate_limiter.window_seconds = 60.0

    mock_resp = AssistantResponse(answer="Cached Answer", topic="General")

    with patch("app.main.get_llm_response", return_value=mock_resp):
        headers = {"X-Forwarded-For": "10.0.0.88"}

        # Request 1 (MISS) & Request 2 (HIT) succeed
        assert client.post("/ask", json={"question": "Test query"}, headers=headers).status_code == 200
        assert client.post("/ask", json={"question": "Test query"}, headers=headers).status_code == 200

        # Request 3 must fail with HTTP 429 despite response being in cache
        res3 = client.post("/ask", json={"question": "Test query"}, headers=headers)
        assert res3.status_code == 429
        assert "Rate limit exceeded" in res3.json()["detail"]
        print("✓ Test 8 — Rate limiter verified to apply before cache lookup")


def test_cache_performance_comparison():
    """Test 9: Measure latency improvement of Cache HIT vs simulated Cache MISS."""
    setup_function()
    simulated_delay = 0.05  # 50ms simulated LLM work

    def slow_llm_response(question):
        time.sleep(simulated_delay)
        return AssistantResponse(answer="Timed Answer", topic="Performance")

    with patch("app.main.get_llm_response", side_effect=slow_llm_response):
        # Measure Cache MISS
        t0 = time.perf_counter()
        res_miss = client.post("/ask", json={"question": "Benchmark query"})
        miss_duration = time.perf_counter() - t0
        assert res_miss.status_code == 200

        # Measure Cache HIT
        t0 = time.perf_counter()
        res_hit = client.post("/ask", json={"question": "Benchmark query"})
        hit_duration = time.perf_counter() - t0
        assert res_hit.status_code == 200

        speedup = miss_duration / max(hit_duration, 0.0001)
        print(f"✓ Test 9 — Performance comparison: MISS={miss_duration*1000:.2f}ms vs HIT={hit_duration*1000:.2f}ms ({speedup:.1f}x speedup)")


if __name__ == "__main__":
    print("=== STARTING TASK 2 PHASE 6 RESPONSE CACHE TESTS ===\n")
    test_cache_miss()
    test_cache_hit()
    test_key_normalization()
    test_ttl_expiration()
    test_maximum_size_lru_eviction()
    test_errors_are_not_cached()
    test_provider_isolation()
    test_rate_limiter_compatibility()
    test_cache_performance_comparison()
    print("\nALL RESPONSE CACHE TESTS PASSED SUCCESSFULLY!")
