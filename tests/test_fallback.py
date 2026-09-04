import os
import sys
import time
from unittest.mock import MagicMock, patch
import httpx
from fastapi.testclient import TestClient

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.main import app
from app.llm import AssistantResponse, get_llm_response
from app.rate_limiter import rate_limiter
from app.cache import response_cache

client = TestClient(app)


def setup_function():
    """Reset rate limiter, response cache, and environment defaults before each test."""
    rate_limiter.reset()
    response_cache.clear()
    os.environ["LLM_PROVIDER"] = "gemini"
    os.environ["FALLBACK_ENABLED"] = "true"
    os.environ["FALLBACK_PROVIDER"] = "ollama"


def test_primary_success():
    """Test 1: When Gemini succeeds, Ollama is NOT called; returns HTTP 200."""
    setup_function()
    gemini_resp = AssistantResponse(answer="Gemini answer", topic="Primary")

    with patch("app.llm._call_gemini", return_value=gemini_resp) as mock_gemini, \
         patch("app.llm._call_ollama") as mock_ollama, \
         patch("app.llm.time.sleep", return_value=None), \
         patch("app.llm.retrieve_relevant_chunks", return_value=[]):

        res = client.post("/ask", json={"question": "What is Python?"})
        assert res.status_code == 200
        data = res.json()
        assert data["answer"] == "Gemini answer"
        assert data["topic"] == "Primary"

        assert mock_gemini.call_count == 1
        assert mock_ollama.call_count == 0
        print("✓ Test 1 — Primary success verified (Gemini called, Ollama NOT called)")


def test_gemini_transient_failure_ollama_success():
    """Test 2: When Gemini fails with transient error after retries, Ollama succeeds; returns HTTP 200."""
    setup_function()
    ollama_resp = AssistantResponse(answer="Ollama fallback answer", topic="Fallback")

    with patch("app.llm._call_gemini", side_effect=httpx.ConnectError("Gemini server unavailable")) as mock_gemini, \
         patch("app.llm._call_ollama", return_value=ollama_resp) as mock_ollama, \
         patch("app.llm.time.sleep", return_value=None), \
         patch("app.llm.retrieve_relevant_chunks", return_value=[]):

        res = client.post("/ask", json={"question": "What is Python?"})
        assert res.status_code == 200
        data = res.json()
        assert data["answer"] == "Ollama fallback answer"
        assert data["topic"] == "Fallback"

        # Gemini should have attempted 3 times (1 initial + 2 retries)
        assert mock_gemini.call_count == 3
        # Ollama should have been called exactly once as fallback
        assert mock_ollama.call_count == 1
        print("✓ Test 2 — Gemini transient failure -> Ollama fallback success verified")


def test_gemini_and_ollama_both_fail():
    """Test 3: When Gemini and Ollama both fail, returns HTTP 503 without additional loops."""
    setup_function()

    with patch("app.llm._call_gemini", side_effect=httpx.ConnectError("Gemini down")) as mock_gemini, \
         patch("app.llm._call_ollama", side_effect=httpx.ConnectError("Ollama also down")) as mock_ollama, \
         patch("app.llm.time.sleep", return_value=None), \
         patch("app.llm.retrieve_relevant_chunks", return_value=[]):

        res = client.post("/ask", json={"question": "What is Python?"})
        assert res.status_code == 503
        assert "temporarily unavailable" in res.json()["detail"].lower()

        assert mock_gemini.call_count == 3
        assert mock_ollama.call_count == 1
        print("✓ Test 3 — Gemini + Ollama both fail -> HTTP 503 returned cleanly")


def test_gemini_quota_error_no_fallback():
    """Test 4: Gemini quota/rate-limit error (429/ResourceExhausted) must NOT trigger fallback."""
    setup_function()

    with patch("app.llm._call_gemini", side_effect=RuntimeError("429 ResourceExhausted: Quota exceeded")) as mock_gemini, \
         patch("app.llm._call_ollama") as mock_ollama, \
         patch("app.llm.time.sleep", return_value=None), \
         patch("app.llm.retrieve_relevant_chunks", return_value=[]):

        res = client.post("/ask", json={"question": "What is Python?"})
        assert res.status_code == 429
        assert "quota or rate limit" in res.json()["detail"].lower()

        # Quota errors fail immediately with 1 attempt (0 retries) and NO fallback
        assert mock_gemini.call_count == 1
        assert mock_ollama.call_count == 0
        print("✓ Test 4 — Gemini Quota 429 protected: 1 attempt, 0 retries, Ollama NOT called")


def test_authentication_error_no_fallback():
    """Test 5: Authentication failure (401/Invalid Key) must NOT trigger fallback."""
    setup_function()

    with patch("app.llm._call_gemini", side_effect=RuntimeError("401 Unauthorized: Invalid API Key")) as mock_gemini, \
         patch("app.llm._call_ollama") as mock_ollama, \
         patch("app.llm.time.sleep", return_value=None), \
         patch("app.llm.retrieve_relevant_chunks", return_value=[]):

        res = client.post("/ask", json={"question": "What is Python?"})
        assert res.status_code in (400, 500, 502)

        # Auth errors fail immediately (0 retries) and NO fallback
        assert mock_gemini.call_count == 1
        assert mock_ollama.call_count == 0
        print("✓ Test 5 — Auth failure stopped immediately: Ollama NOT called")


def test_explicit_ollama_provider():
    """Test 6: When LLM_PROVIDER=ollama is explicitly selected, Ollama is used directly."""
    setup_function()
    os.environ["LLM_PROVIDER"] = "ollama"
    ollama_resp = AssistantResponse(answer="Direct Ollama answer", topic="Local")

    with patch("app.llm._call_gemini") as mock_gemini, \
         patch("app.llm._call_ollama", return_value=ollama_resp) as mock_ollama, \
         patch("app.llm.time.sleep", return_value=None), \
         patch("app.llm.retrieve_relevant_chunks", return_value=[]):

        res = client.post("/ask", json={"question": "What is Python?"})
        assert res.status_code == 200
        assert res.json()["answer"] == "Direct Ollama answer"

        assert mock_gemini.call_count == 0
        assert mock_ollama.call_count == 1
        print("✓ Test 6 — Explicit LLM_PROVIDER=ollama uses Ollama directly without Gemini")


def test_rag_context_preservation_on_fallback():
    """Test 7: Verify RAG retrieval is executed only once and identical context is passed to fallback."""
    setup_function()
    sample_chunks = [{"source": "docs/guide.txt", "chunk_id": 42, "text": "Binary search has O(log n) complexity."}]
    ollama_resp = AssistantResponse(answer="RAG answer via Ollama", topic="Algorithms")

    with patch("app.llm.retrieve_relevant_chunks", return_value=sample_chunks) as mock_rag, \
         patch("app.llm._call_gemini", side_effect=httpx.ConnectError("Gemini down")), \
         patch("app.llm.time.sleep", return_value=None), \
         patch("app.llm._call_ollama", return_value=ollama_resp) as mock_ollama:

        res = client.post("/ask", json={"question": "What is binary search complexity?"})
        assert res.status_code == 200

        # RAG retrieval must have been called exactly once
        assert mock_rag.call_count == 1

        # Check prompt passed to Ollama contains the retrieved RAG context
        ollama_call_args = mock_ollama.call_args[1]
        formatted_prompt = ollama_call_args["formatted_user_prompt"]
        assert "Binary search has O(log n) complexity." in formatted_prompt
        assert "docs/guide.txt" in formatted_prompt
        print("✓ Test 7 — RAG context preserved across fallback (retrieval executed once)")


def test_fallback_disabled():
    """Test 8: When FALLBACK_ENABLED=false, transient Gemini failures do NOT call Ollama."""
    setup_function()
    os.environ["FALLBACK_ENABLED"] = "false"

    with patch("app.llm._call_gemini", side_effect=httpx.ConnectError("Gemini down")) as mock_gemini, \
         patch("app.llm._call_ollama") as mock_ollama, \
         patch("app.llm.time.sleep", return_value=None), \
         patch("app.llm.retrieve_relevant_chunks", return_value=[]):

        res = client.post("/ask", json={"question": "What is Python?"})
        assert res.status_code == 503
        assert "temporarily unavailable" in res.json()["detail"].lower()

        # Gemini retries exhausted
        assert mock_gemini.call_count == 3
        # Ollama must NOT be called since fallback is disabled
        assert mock_ollama.call_count == 0
        print("✓ Test 8 — Fallback disabled verified (FALLBACK_ENABLED=false prevents Ollama invocation)")


if __name__ == "__main__":
    print("=== STARTING TASK 2 PHASE 5 FALLBACK PROVIDER TESTS ===\n")
    test_primary_success()
    test_gemini_transient_failure_ollama_success()
    test_gemini_and_ollama_both_fail()
    test_gemini_quota_error_no_fallback()
    test_authentication_error_no_fallback()
    test_explicit_ollama_provider()
    test_rag_context_preservation_on_fallback()
    test_fallback_disabled()
    print("\nALL FALLBACK PROVIDER TESTS PASSED SUCCESSFULLY!")
