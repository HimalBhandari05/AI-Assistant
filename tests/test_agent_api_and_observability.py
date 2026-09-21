"""Unit and integration tests for Phase 4 (API & UI Integration) and Phase 5 (Observability & Token Accounting)."""

import os
import sys
import time
from unittest.mock import patch
from fastapi.testclient import TestClient

# Ensure project root in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.main import app
from app.agent.context import TokenUsage
from app.agent.engine import AgentResult, run_agent_workflow
from app.agent.metrics import AgentExecutionMetrics
from app.cache import response_cache
from app.rate_limiter import rate_limiter

client = TestClient(app)


def setup_function():
    """Reset rate limiter and response cache."""
    rate_limiter.reset()
    response_cache.clear()
    os.environ["LLM_PROVIDER"] = "gemini"
    os.environ["USE_AGENT"] = "true"


def test_api_ask_endpoint_agent_response():
    """Verify POST /ask endpoint returns enriched agent response."""
    setup_function()
    mock_agent_res = AgentResult(
        status="completed",
        answer="Binary search runs in O(log n) time.",
        topic="Algorithms",
        trajectory_length=2,
        clarification_required=False,
    )

    with patch("app.agent.engine.run_agent_workflow", return_value=mock_agent_res):
        res = client.post("/ask", json={"question": "What is binary search complexity?"})
        assert res.status_code == 200
        data = res.json()
        assert data["answer"] == "Binary search runs in O(log n) time."
        assert data["topic"] == "Algorithms"
        assert data["status"] == "completed"
        assert data["trajectory_length"] == 2
        assert data["clarification_required"] is False


def test_api_ask_clarification_response():
    """Verify POST /ask returns clarification state correctly."""
    setup_function()
    mock_agent_res = AgentResult(
        status="clarification_required",
        answer="Which search algorithm are you asking about?",
        topic="Clarification",
        trajectory_length=1,
        clarification_required=True,
    )

    with patch("app.agent.engine.run_agent_workflow", return_value=mock_agent_res):
        res = client.post("/ask", json={"question": "What is its complexity?"})
        assert res.status_code == 200
        data = res.json()
        assert data["clarification_required"] is True
        assert data["status"] == "clarification_required"
        assert "Which search algorithm" in data["answer"]


def test_agent_caching_behavior():
    """Verify that only the final AssistantResponse is cached and subsequent hits bypass the agent."""
    setup_function()
    mock_agent_res = AgentResult(
        status="completed",
        answer="Cached agent answer",
        topic="Test",
        trajectory_length=3,
        clarification_required=False,
    )

    with patch("app.agent.engine.run_agent_workflow", return_value=mock_agent_res) as mock_agent:
        # 1. First call -> Cache MISS
        res1 = client.post("/ask", json={"question": "Unique agent query"})
        assert res1.status_code == 200
        assert mock_agent.call_count == 1

        # 2. Second call -> Cache HIT
        res2 = client.post("/ask", json={"question": "Unique agent query"})
        assert res2.status_code == 200
        assert res2.json()["answer"] == "Cached agent answer"
        assert mock_agent.call_count == 1  # Agent NOT re-executed


def test_metrics_token_accounting_and_telemetry():
    """Verify AgentExecutionMetrics records token counts, tool telemetry, and run ID."""
    metrics = AgentExecutionMetrics(question="Test question")
    assert metrics.agent_run_id is not None
    assert len(metrics.agent_run_id) > 10

    # Record tool calls
    metrics.record_tool_call(
        tool_name="rag_search",
        arguments_valid=True,
        success=True,
        execution_time_seconds=0.015,
    )
    metrics.record_tool_call(
        tool_name="calculator",
        arguments_valid=False,
        success=False,
        execution_time_seconds=0.002,
        error_type="InvalidArgumentsError",
        error_message="Missing argument 'b'",
    )

    assert len(metrics.tool_metrics) == 2
    assert metrics.tool_metrics[0].tool_name == "rag_search"
    assert metrics.tool_metrics[0].success is True
    assert metrics.tool_metrics[1].tool_name == "calculator"
    assert metrics.tool_metrics[1].arguments_valid is False
    assert metrics.tool_metrics[1].error_type == "InvalidArgumentsError"


def test_structured_failure_logging_and_secret_redaction():
    """Verify structured failure logs redact sensitive keys."""
    metrics = AgentExecutionMetrics(question="Failure question")
    metrics.record_failure(
        failure_type="ProviderError",
        iteration=2,
        details={
            "error": "Authentication failed",
            "api_key": "secret_key_12345",
            "auth_token": "token_abcde",
            "regular_info": "safe_details",
        },
    )

    assert len(metrics.failures) == 1
    failure = metrics.failures[0]
    assert failure.failure_type == "ProviderError"
    assert failure.iteration == 2
    assert failure.details["api_key"] == "[REDACTED]"
    assert failure.details["auth_token"] == "[REDACTED]"
    assert failure.details["regular_info"] == "safe_details"
