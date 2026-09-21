"""Unit and integration tests for Phase 7 Failure Injection and Fault Recovery."""

import os
import sys
from unittest.mock import MagicMock, patch
import pytest

# Ensure project root in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.agent.engine import run_agent_workflow, AgentResult
from app.agent.tools import ToolRegistry, default_tool_registry
from tests.evaluation.failure_injection import (
    FAILURE_INJECTION_CASES,
    run_failure_injection_evaluation,
    FailureInjectionSummary,
)


def test_failure_injection_dataset_structure():
    """Verify all 4 failure injection cases are properly configured."""
    assert len(FAILURE_INJECTION_CASES) == 4
    ids = [c.id for c in FAILURE_INJECTION_CASES]
    assert "FI-01" in ids
    assert "FI-02" in ids
    assert "FI-03" in ids
    assert "FI-04" in ids


def test_scenario_1_empty_retrieval_and_query_reformulation():
    """Test 1: Injected empty RAG search triggers query reformulation and successful final answer."""
    # Step 1: Model searches with vague query -> receives 0 chunks
    # Step 2: Model observes empty result, reformulates query -> receives valid chunk
    # Step 3: Model outputs verified final answer
    model_actions = [
        ({"action": "rag_search", "arguments": {"query": "vague keyword"}}, 50, 20),
        ({"action": "rag_search", "arguments": {"query": "Retrieval-Augmented Generation definition"}}, 70, 25),
        ({"action": "final_answer", "answer": "RAG retrieves facts from external knowledge bases.", "topic": "NLP"}, 110, 30),
    ]
    call_idx = 0

    def mock_query(*args, **kwargs):
        nonlocal call_idx
        action = model_actions[call_idx]
        call_idx += 1
        return action

    # Retrieve mock: call 1 returns empty, call 2 returns real chunk
    retrieval_responses = [
        [],  # Injected failure (empty retrieval)
        [{"source": "sample.txt", "chunk_id": 0, "text": "RAG retrieves facts from external knowledge bases."}],
    ]
    retrieval_idx = 0

    def mock_retrieve(question, top_k=3):
        nonlocal retrieval_idx
        res = retrieval_responses[retrieval_idx]
        retrieval_idx += 1
        return res

    with patch("app.agent.engine._query_provider_for_action", side_effect=mock_query), \
         patch("app.agent.tools.retrieve_relevant_chunks", side_effect=mock_retrieve):
        result = run_agent_workflow(question="What is RAG?")
        assert result.status == "completed"
        assert result.trajectory_length == 3
        # Verify first step was empty
        assert "0 relevant document chunks found" in result.trajectory[0].observation
        # Verify second step succeeded
        assert "sample.txt" in result.trajectory[1].observation
        assert "RAG retrieves facts" in result.answer


def test_scenario_2_calculator_division_by_zero_recovery():
    """Test 2: Injected calculator zero division returns structured error and agent recovers."""
    # Step 1: Model calls calculator with invalid b=0 -> receives ToolError observation
    # Step 2: Model observes error, corrects b=16 -> receives valid result
    # Step 3: Model outputs final answer
    model_actions = [
        ({"action": "calculator", "arguments": {"a": 256, "b": 0, "operation": "divide"}}, 40, 15),
        ({"action": "calculator", "arguments": {"a": 256, "b": 16, "operation": "divide"}}, 60, 20),
        ({"action": "final_answer", "answer": "256 divided by 16 is 16.", "topic": "Arithmetic"}, 80, 20),
    ]
    call_idx = 0

    def mock_query(*args, **kwargs):
        nonlocal call_idx
        action = model_actions[call_idx]
        call_idx += 1
        return action

    with patch("app.agent.engine._query_provider_for_action", side_effect=mock_query):
        result = run_agent_workflow(question="Calculate 256 divided by 16.")
        assert result.status == "completed"
        assert result.trajectory_length == 3
        # Verify step 1 trapped error cleanly
        assert "[Tool Error (calculator)]: ValueError" in result.trajectory[0].observation
        assert result.trajectory[0].success is False
        # Verify step 2 executed successfully
        assert "[Calculator Result]: 256.0 divide 16.0 = 16.0" in result.trajectory[1].observation
        assert result.trajectory[1].success is True
        assert "16" in result.answer


def test_scenario_3_cascading_error_recovery():
    """Test 3: Cascading multi-step error recovery across retrieval and tool execution."""
    # Step 1: Empty search -> recovers in step 2
    # Step 3: Bad calculator call -> recovers in step 4
    # Step 5: Final answer
    model_actions = [
        ({"action": "rag_search", "arguments": {"query": "bad query 1"}}, 40, 15),
        ({"action": "rag_search", "arguments": {"query": "linear search 500 binary 9"}}, 60, 20),
        ({"action": "calculator", "arguments": {"a": 500, "b": 0, "operation": "divide"}}, 80, 20),
        ({"action": "calculator", "arguments": {"a": 500, "b": 9, "operation": "subtract"}}, 100, 20),
        ({"action": "final_answer", "answer": "Difference is 491.", "topic": "Algorithms"}, 120, 25),
    ]
    call_idx = 0

    def mock_query(*args, **kwargs):
        nonlocal call_idx
        action = model_actions[call_idx]
        call_idx += 1
        return action

    retrieval_responses = [
        [],
        [{"source": "sample.txt", "chunk_id": 0, "text": "Linear search 500, binary search 9."}],
    ]
    retrieval_idx = 0

    def mock_retrieve(question, top_k=3):
        nonlocal retrieval_idx
        res = retrieval_responses[retrieval_idx]
        retrieval_idx += 1
        return res

    with patch("app.agent.engine._query_provider_for_action", side_effect=mock_query), \
         patch("app.agent.tools.retrieve_relevant_chunks", side_effect=mock_retrieve):
        result = run_agent_workflow(question="Compare operations and calculate difference.")
        assert result.status == "completed"
        assert result.trajectory_length == 5
        assert result.trajectory[0].observation.startswith("[RAG Search Results")
        assert result.trajectory[2].success is False
        assert result.trajectory[3].success is True
        assert "491" in result.answer


def test_scenario_4_unrecoverable_budget_protection():
    """Test 4: Unrecoverable empty retrieval hits max iterations and terminates gracefully without hanging."""
    call_count = 0

    def endless_queries(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        return ({"action": "rag_search", "arguments": {"query": f"unsupported query attempt {call_count}"}}, 50, 15)

    with patch("app.agent.engine._query_provider_for_action", side_effect=endless_queries), \
         patch("app.agent.tools.retrieve_relevant_chunks", return_value=[]):
        result = run_agent_workflow(question="Non-existent topic", max_iterations=5)
        assert result.status == "max_iterations_exceeded"
        assert result.trajectory_length == 5
        assert "maximum reasoning iterations" in result.answer
        # Verify hallucination prevented
        assert "non-existent" not in result.answer.lower() or "not able to fully resolve" in result.answer


def test_failure_injection_benchmark_runner():
    """Verify run_failure_injection_evaluation generates report and calculates expected rates."""
    report_path = "tests/evaluation/results/failure_injection_results.md"
    summary = run_failure_injection_evaluation(output_report_path=report_path)

    assert isinstance(summary, FailureInjectionSummary)
    assert summary.total_injected_cases == 4
    assert summary.failure_detection_rate == 100.0
    assert summary.failure_recovery_rate == 100.0
    assert summary.soft_failures == 2
    assert summary.cascading_soft_failures == 1
    assert summary.hard_failures == 1

    # Verify Markdown report file generated
    assert os.path.exists(report_path)
    with open(report_path, "r", encoding="utf-8") as f:
        content = f.read()

    assert "# W16 Fault Recovery & Failure Injection Report" in content
    assert "Failure Detection Rate" in content
    assert "100.0%" in content
    assert "FI-01" in content
    assert "FI-04" in content
    assert "Cost of Recovery Analysis" in content
