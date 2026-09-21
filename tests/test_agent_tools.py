"""Unit and contract tests for Phase 1 Tool Registry and tools."""

import os
import sys
from unittest.mock import patch
import pytest

# Ensure project root in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.agent.tools import (
    ToolRegistry,
    ToolResult,
    ToolError,
    rag_search_tool,
    calculator_tool,
    ask_user_clarification_tool,
    default_tool_registry,
)


def test_calculator_tool_success():
    """Verify calculator tool executes valid arithmetic operations."""
    res = calculator_tool(a=25, b=18, operation="multiply")
    assert res["result"] == 450.0
    assert res["operation"] == "multiply"

    res_div = calculator_tool(a=10, b=2, operation="divide")
    assert res_div["result"] == 5.0


def test_calculator_tool_errors():
    """Verify calculator tool errors on zero division and invalid inputs."""
    with pytest.raises(ValueError, match="Division by zero"):
        calculator_tool(a=10, b=0, operation="divide")

    with pytest.raises(ValueError, match="Unsupported operation"):
        calculator_tool(a=10, b=2, operation="modulo")


def test_rag_search_tool_success():
    """Verify rag_search tool returns structured chunks."""
    mock_chunks = [
        {"text": "Binary search is O(log n).", "source": "sample.txt", "chunk_id": 0, "distance": 0.1}
    ]
    with patch("app.agent.tools.retrieve_relevant_chunks", return_value=mock_chunks) as mock_retrieve:
        res = rag_search_tool(query="binary search", top_k=2)
        assert res["query"] == "binary search"
        assert res["count"] == 1
        assert len(res["results"]) == 1
        assert res["results"][0]["text"] == "Binary search is O(log n)."
        mock_retrieve.assert_called_once_with(question="binary search", top_k=2)


def test_rag_search_tool_empty_query():
    """Verify rag_search tool errors on empty query."""
    with pytest.raises(ValueError, match="Search query cannot be empty"):
        rag_search_tool(query="")


def test_ask_user_clarification_tool():
    """Verify clarification tool returns clarification required structure."""
    res = ask_user_clarification_tool(question="Which algorithm do you mean?")
    assert res["clarification_required"] is True
    assert res["question"] == "Which algorithm do you mean?"

    with pytest.raises(ValueError, match="cannot be empty"):
        ask_user_clarification_tool(question="")


def test_tool_registry_contract():
    """Verify ToolRegistry execution contract, error trapping, and timing."""
    registry = default_tool_registry

    # 1. Successful calculator call
    res_calc = registry.execute("calculator", {"a": 10, "b": 5, "operation": "add"})
    assert isinstance(res_calc, ToolResult)
    assert res_calc.success is True
    assert res_calc.tool == "calculator"
    assert res_calc.result == {"a": 10.0, "b": 5.0, "operation": "add", "result": 15.0}
    assert res_calc.error is None
    assert res_calc.execution_time_seconds >= 0.0

    # 2. Handled exception (division by zero)
    res_div0 = registry.execute("calculator", {"a": 10, "b": 0, "operation": "divide"})
    assert res_div0.success is False
    assert res_div0.error is not None
    assert res_div0.error.type == "ValueError"
    assert "Division by zero" in res_div0.error.message

    # 3. Invalid arguments
    res_invalid_args = registry.execute("calculator", {"invalid_param": 123})
    assert res_invalid_args.success is False
    assert res_invalid_args.error.type == "InvalidArgumentsError"

    # 4. Unknown tool
    res_unknown = registry.execute("non_existent_tool", {})
    assert res_unknown.success is False
    assert res_unknown.error.type == "UnknownToolError"


def test_tool_registry_listing():
    """Verify list_tools exposes registered metadata."""
    tools = default_tool_registry.list_tools()
    names = [t["name"] for t in tools]
    assert "rag_search" in names
    assert "calculator" in names
    assert "ask_user_clarification" in names
