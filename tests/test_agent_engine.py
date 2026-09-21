"""Unit and trajectory tests for Phase 3 Agent Execution Engine."""

import os
import sys
import time
from unittest.mock import MagicMock, patch
import pytest

# Ensure project root in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.agent.engine import (
    run_agent_workflow,
    parse_model_action_json,
    AgentResult,
)
from app.agent.tools import ToolRegistry, default_tool_registry


def test_parse_model_action_json():
    """Verify robust JSON action parsing across formats."""
    # Plain JSON
    a1 = parse_model_action_json('{"action": "rag_search", "arguments": {"query": "test"}}')
    assert a1["action"] == "rag_search"

    # Markdown fenced code block
    a2 = parse_model_action_json('```json\n{"action": "final_answer", "answer": "Hello", "topic": "General"}\n```')
    assert a2["action"] == "final_answer"
    assert a2["answer"] == "Hello"

    # Embedded JSON in text
    a3 = parse_model_action_json('Here is my thought: {"action": "calculator", "arguments": {"a": 1, "b": 2, "operation": "add"}}')
    assert a3["action"] == "calculator"


def test_trajectory_direct_calculation():
    """Test A: Direct calculation trajectory: calculator -> final_answer."""
    responses = [
        ({"action": "calculator", "arguments": {"a": 45, "b": 12, "operation": "multiply"}}, 50, 20),
        ({"action": "final_answer", "answer": "45 multiplied by 12 is 540.", "topic": "Arithmetic"}, 60, 15),
    ]
    resp_idx = 0

    def mock_query(*args, **kwargs):
        nonlocal resp_idx
        res = responses[resp_idx]
        resp_idx += 1
        return res

    with patch("app.agent.engine._query_provider_for_action", side_effect=mock_query):
        result = run_agent_workflow(question="What is 45 multiplied by 12?")
        assert result.status == "completed"
        assert "540" in result.answer
        assert result.topic == "Arithmetic"
        assert result.trajectory_length == 2
        assert result.trajectory[0].action == "calculator"
        assert result.trajectory[1].action == "final_answer"
        assert result.token_usage.total_tokens == 145


def test_trajectory_single_retrieval():
    """Test B: Single retrieval trajectory: rag_search -> final_answer."""
    responses = [
        ({"action": "rag_search", "arguments": {"query": "binary search"}}, 40, 20),
        ({"action": "final_answer", "answer": "Binary search is O(log n).", "topic": "Algorithms"}, 80, 25),
    ]
    resp_idx = 0

    def mock_query(*args, **kwargs):
        nonlocal resp_idx
        res = responses[resp_idx]
        resp_idx += 1
        return res

    with patch("app.agent.engine._query_provider_for_action", side_effect=mock_query), \
         patch("app.agent.tools.retrieve_relevant_chunks", return_value=[{"source": "doc.txt", "chunk_id": 1, "text": "Binary search is O(log n)."}]):
        result = run_agent_workflow(question="What is binary search complexity?")
        assert result.status == "completed"
        assert result.trajectory_length == 2
        assert result.trajectory[0].action == "rag_search"
        assert result.trajectory[1].action == "final_answer"


def test_trajectory_multiple_retrievals():
    """Test C: Multiple retrievals trajectory: rag_search(A) -> rag_search(B) -> final_answer."""
    responses = [
        ({"action": "rag_search", "arguments": {"query": "linear search complexity"}}, 40, 15),
        ({"action": "rag_search", "arguments": {"query": "binary search complexity"}}, 60, 20),
        ({"action": "final_answer", "answer": "Linear search is O(n) while binary search is O(log n).", "topic": "Algorithms"}, 100, 30),
    ]
    resp_idx = 0

    def mock_query(*args, **kwargs):
        nonlocal resp_idx
        res = responses[resp_idx]
        resp_idx += 1
        return res

    with patch("app.agent.engine._query_provider_for_action", side_effect=mock_query), \
         patch("app.agent.tools.retrieve_relevant_chunks", return_value=[{"source": "doc.txt", "chunk_id": 1, "text": "Search info"}]):
        result = run_agent_workflow(question="Compare linear and binary search complexity.")
        assert result.status == "completed"
        assert result.trajectory_length == 3
        assert result.trajectory[0].action == "rag_search"
        assert result.trajectory[0].arguments["query"] == "linear search complexity"
        assert result.trajectory[1].action == "rag_search"
        assert result.trajectory[1].arguments["query"] == "binary search complexity"
        assert result.trajectory[2].action == "final_answer"


def test_trajectory_tool_chaining():
    """Test D: Tool chaining: rag_search -> calculator -> final_answer."""
    responses = [
        ({"action": "rag_search", "arguments": {"query": "dataset size"}}, 50, 20),
        ({"action": "calculator", "arguments": {"a": 100, "b": 10, "operation": "divide"}}, 80, 25),
        ({"action": "final_answer", "answer": "The result after calculation is 10.", "topic": "Data"}, 110, 20),
    ]
    resp_idx = 0

    def mock_query(*args, **kwargs):
        nonlocal resp_idx
        res = responses[resp_idx]
        resp_idx += 1
        return res

    with patch("app.agent.engine._query_provider_for_action", side_effect=mock_query), \
         patch("app.agent.tools.retrieve_relevant_chunks", return_value=[{"source": "doc.txt", "chunk_id": 1, "text": "Dataset size is 100."}]):
        result = run_agent_workflow(question="Find dataset size and divide by 10.")
        assert result.status == "completed"
        assert result.trajectory_length == 3
        assert result.trajectory[0].action == "rag_search"
        assert result.trajectory[1].action == "calculator"
        assert result.trajectory[2].action == "final_answer"


def test_trajectory_clarification():
    """Test E: Clarification action: ask_user_clarification."""
    responses = [
        ({"action": "ask_user_clarification", "question": "Which algorithm are you referring to?"}, 40, 15),
    ]

    with patch("app.agent.engine._query_provider_for_action", return_value=responses[0]):
        result = run_agent_workflow(question="What is its complexity?")
        assert result.status == "clarification_required"
        assert result.clarification_required is True
        assert "Which algorithm are you referring to?" in result.answer
        assert result.trajectory_length == 1
        assert result.trajectory[0].action == "ask_user_clarification"


def test_trajectory_max_iterations_exceeded():
    """Test F: Max iterations exceeded without infinite loop."""
    call_count = 0

    def distinct_searches(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        return ({"action": "rag_search", "arguments": {"query": f"distinct query {call_count}"}}, 40, 10)

    with patch("app.agent.engine._query_provider_for_action", side_effect=distinct_searches), \
         patch("app.agent.tools.retrieve_relevant_chunks", return_value=[]):
        result = run_agent_workflow(question="Endless query", max_iterations=3)
        assert result.status == "max_iterations_exceeded"
        assert result.trajectory_length == 3
        assert "maximum reasoning iterations" in result.answer


def test_trajectory_duplicate_action_loop_protection():
    """Test G: Protection against duplicate action loops."""
    duplicate_search = ({"action": "rag_search", "arguments": {"query": "fixed same query"}}, 30, 10)

    with patch("app.agent.engine._query_provider_for_action", return_value=duplicate_search), \
         patch("app.agent.tools.retrieve_relevant_chunks", return_value=[]):
        result = run_agent_workflow(question="Loop query", max_iterations=5)
        # Should break when duplicate count exceeds limit before hitting max_iterations
        assert result.status == "max_iterations_exceeded"
        assert len(result.trajectory) <= 3
