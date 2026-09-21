"""Unit tests for the Phase 6 Evaluation Harness and Metric Calculator."""

import os
import sys
import pytest

# Ensure project root in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.agent.context import TokenUsage, TrajectoryStep
from app.agent.engine import AgentResult
from tests.evaluation.benchmark_cases import BenchmarkCase, BENCHMARK_CASES
from tests.evaluation.evaluator import (
    evaluate_single_case,
    compute_benchmark_summary,
    generate_markdown_report,
    check_numerical_result_in_text,
    check_keywords_in_text,
    CaseEvaluationResult,
)


def test_benchmark_dataset_integrity():
    """Verify that benchmark dataset has at least 15 valid curated cases across categories."""
    assert len(BENCHMARK_CASES) >= 15
    categories = set(c.category for c in BENCHMARK_CASES)
    assert "single_step_calc" in categories
    assert "single_step_rag" in categories
    assert "multi_step_retrieval" in categories
    assert "retrieval_and_calculator" in categories
    assert "clarification" in categories
    assert "boundary_stop" in categories


def test_check_numerical_result_in_text():
    """Verify numerical extraction and matching."""
    assert check_numerical_result_in_text(540.0, "The answer is 540.") is True
    assert check_numerical_result_in_text(540.0, "Result: 540.0 exactly") is True
    assert check_numerical_result_in_text(540.0, "The value is 120.") is False
    assert check_numerical_result_in_text(16.0, "256 divided by 16 is 16.") is True


def test_check_keywords_in_text():
    """Verify keyword presence matching."""
    text = "Binary search runs in O(log n) time while linear search takes O(n)."
    assert check_keywords_in_text(["binary search", "O(log n)"], text) is True
    assert check_keywords_in_text(["photosynthesis"], text) is False


def test_evaluate_single_case_success():
    """Verify clean success evaluation."""
    case = BenchmarkCase(
        id="test_calc",
        category="single_step_calc",
        question="What is 10 + 20?",
        expected_status="completed",
        required_tools=["calculator"],
        expected_numerical_result=30.0,
    )

    trajectory = [
        TrajectoryStep(iteration=1, action="calculator", arguments={"a": 10, "b": 20, "operation": "add"}, observation="10 + 20 = 30", success=True),
        TrajectoryStep(iteration=2, action="final_answer", arguments={"answer": "10 + 20 is 30.", "topic": "Math"}, observation="Done", success=True),
    ]
    agent_res = AgentResult(
        status="completed",
        answer="10 + 20 is 30.",
        topic="Math",
        trajectory_length=2,
        trajectory=trajectory,
        token_usage=TokenUsage(prompt_tokens=50, completion_tokens=20, total_tokens=70),
    )

    eval_res = evaluate_single_case(case, agent_res)
    assert eval_res.success is True
    assert eval_res.failure_classification is None
    assert eval_res.tool_calls_count == 1
    assert eval_res.correct_tool_calls_count == 1
    assert eval_res.total_tokens == 70


def test_evaluate_single_case_hard_failure():
    """Verify hard failure detection on missing required tool or wrong answer."""
    case = BenchmarkCase(
        id="test_req_tool",
        category="single_step_calc",
        question="What is 10 + 20?",
        expected_status="completed",
        required_tools=["calculator"],
        expected_numerical_result=30.0,
    )

    # Missing calculator tool
    trajectory = [
        TrajectoryStep(iteration=1, action="final_answer", arguments={"answer": "The answer is 999.", "topic": "Math"}, observation="Done", success=True),
    ]
    agent_res = AgentResult(
        status="completed",
        answer="The answer is 999.",
        topic="Math",
        trajectory_length=1,
        trajectory=trajectory,
        token_usage=TokenUsage(prompt_tokens=30, completion_tokens=10, total_tokens=40),
    )

    eval_res = evaluate_single_case(case, agent_res)
    assert eval_res.success is False
    assert eval_res.failure_classification == "Hard Failure"
    assert "Missing required tools" in eval_res.failure_details


def test_evaluate_single_case_soft_failure_recovery():
    """Verify soft failure detection when an initial tool error is recovered in next step."""
    case = BenchmarkCase(
        id="test_soft_fail",
        category="single_step_calc",
        question="What is 10 divided by 2?",
        expected_status="completed",
        required_tools=["calculator"],
        expected_numerical_result=5.0,
    )

    trajectory = [
        TrajectoryStep(iteration=1, action="calculator", arguments={"a": 10, "b": 0, "operation": "divide"}, observation="[Tool Error (calculator)]: ValueError - Division by zero", success=False),
        TrajectoryStep(iteration=2, action="calculator", arguments={"a": 10, "b": 2, "operation": "divide"}, observation="10 / 2 = 5", success=True),
        TrajectoryStep(iteration=3, action="final_answer", arguments={"answer": "10 divided by 2 is 5.", "topic": "Math"}, observation="Done", success=True),
    ]
    agent_res = AgentResult(
        status="completed",
        answer="10 divided by 2 is 5.",
        topic="Math",
        trajectory_length=3,
        trajectory=trajectory,
        token_usage=TokenUsage(prompt_tokens=120, completion_tokens=35, total_tokens=155),
    )

    eval_res = evaluate_single_case(case, agent_res)
    assert eval_res.success is True
    assert eval_res.failure_classification == "Soft Failure"
    assert "Recovered from tool error" in eval_res.failure_details


def test_compute_benchmark_summary():
    """Verify aggregate summary calculations (TCR, TCC, trajectory stats, token counts)."""
    results = [
        CaseEvaluationResult(
            case_id="c1", category="single_step_calc", question="q1", success=True, status="completed",
            trajectory_length=2, tool_calls_count=1, correct_tool_calls_count=1,
            prompt_tokens=50, completion_tokens=20, total_tokens=70, failure_classification=None,
            failure_details=None, answer="ans 1", execution_time_seconds=0.1,
        ),
        CaseEvaluationResult(
            case_id="c2", category="single_step_rag", question="q2", success=False, status="error",
            trajectory_length=1, tool_calls_count=1, correct_tool_calls_count=0,
            prompt_tokens=100, completion_tokens=30, total_tokens=130, failure_classification="Hard Failure",
            failure_details="Error", answer="ans 2", execution_time_seconds=0.2,
        ),
    ]

    summary = compute_benchmark_summary(results)
    assert summary.total_tasks == 2
    assert summary.successful_tasks == 1
    assert summary.task_completion_rate == 50.0
    assert summary.total_tool_calls == 2
    assert summary.correct_tool_calls == 1
    assert summary.tool_call_correctness == 50.0
    assert summary.avg_trajectory_length == 1.5
    assert summary.min_trajectory_length == 1
    assert summary.max_trajectory_length == 2
    assert summary.total_tokens == 200
    assert summary.avg_tokens_per_query == 100.0
    assert summary.hard_failures == 1
    assert summary.soft_failures == 0


def test_generate_markdown_report():
    """Verify Markdown report generation format and headings."""
    results = [
        CaseEvaluationResult(
            case_id="case_calc_01", category="single_step_calc", question="What is 45 * 12?", success=True, status="completed",
            trajectory_length=2, tool_calls_count=1, correct_tool_calls_count=1,
            prompt_tokens=50, completion_tokens=20, total_tokens=70, failure_classification=None,
            failure_details=None, answer="540", execution_time_seconds=0.1,
        ),
    ]
    summary = compute_benchmark_summary(results)
    report_md = generate_markdown_report(summary, {"provider": "gemini", "model": "gemini-2.5-flash"})

    assert "# W16 Agent Evaluation Report" in report_md
    assert "## 1. Evaluation Configuration" in report_md
    assert "## 2. Overall Results" in report_md
    assert "## 3. Per-Query Results" in report_md
    assert "## 4. Failure Analysis" in report_md
    assert "## 5. Trajectory Analysis" in report_md
    assert "## 6. Token Analysis" in report_md
    assert "Task Completion Rate (TCR)" in report_md
    assert "`case_calc_01`" in report_md
