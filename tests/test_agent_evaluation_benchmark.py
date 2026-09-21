"""End-to-End Benchmark Execution Test for Phase 6 Evaluation Harness."""

import os
import sys
import pytest

# Ensure project root in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.agent.context import TokenUsage, TrajectoryStep
from app.agent.engine import AgentResult
from tests.evaluation.benchmark_cases import BENCHMARK_CASES
from tests.evaluation.evaluator import run_benchmark_evaluation, BenchmarkSummary


def simulated_agent_runner(question: str) -> AgentResult:
    """Deterministic agent runner simulating accurate multi-step trajectories for the 16 benchmark cases."""
    q_lower = question.lower()

    # 1. case_calc_01: 45 * 12
    if "45 multiplied by 12" in q_lower or "45 * 12" in q_lower:
        traj = [
            TrajectoryStep(iteration=1, action="calculator", arguments={"a": 45.0, "b": 12.0, "operation": "multiply"}, observation="[Calculator Result]: 45.0 multiply 12.0 = 540.0", success=True, execution_time_seconds=0.001),
            TrajectoryStep(iteration=2, action="final_answer", arguments={"answer": "45 multiplied by 12 is 540.", "topic": "Arithmetic"}, observation="Final answer generated.", success=True, execution_time_seconds=0.001),
        ]
        return AgentResult(
            status="completed",
            answer="45 multiplied by 12 is 540.",
            topic="Arithmetic",
            trajectory_length=2,
            trajectory=traj,
            token_usage=TokenUsage(prompt_tokens=65, completion_tokens=25, total_tokens=90),
        )

    # 2. case_calc_02: 256 / 16
    if "256 divided by 16" in q_lower:
        traj = [
            TrajectoryStep(iteration=1, action="calculator", arguments={"a": 256.0, "b": 16.0, "operation": "divide"}, observation="[Calculator Result]: 256.0 divide 16.0 = 16.0", success=True, execution_time_seconds=0.001),
            TrajectoryStep(iteration=2, action="final_answer", arguments={"answer": "256 divided by 16 is 16.", "topic": "Arithmetic"}, observation="Final answer generated.", success=True, execution_time_seconds=0.001),
        ]
        return AgentResult(
            status="completed",
            answer="256 divided by 16 is 16.",
            topic="Arithmetic",
            trajectory_length=2,
            trajectory=traj,
            token_usage=TokenUsage(prompt_tokens=60, completion_tokens=20, total_tokens=80),
        )

    # 3. case_calc_03: 1450 + 3250
    if "1450 added to 3250" in q_lower or "1450" in q_lower:
        traj = [
            TrajectoryStep(iteration=1, action="calculator", arguments={"a": 1450.0, "b": 3250.0, "operation": "add"}, observation="[Calculator Result]: 1450.0 add 3250.0 = 4700.0", success=True, execution_time_seconds=0.001),
            TrajectoryStep(iteration=2, action="final_answer", arguments={"answer": "1450 added to 3250 is 4700.", "topic": "Arithmetic"}, observation="Final answer generated.", success=True, execution_time_seconds=0.001),
        ]
        return AgentResult(
            status="completed",
            answer="1450 added to 3250 is 4700.",
            topic="Arithmetic",
            trajectory_length=2,
            trajectory=traj,
            token_usage=TokenUsage(prompt_tokens=65, completion_tokens=22, total_tokens=87),
        )

    # 4. case_rag_01: RAG definition
    if "retrieval-augmented generation (rag)" in q_lower:
        traj = [
            TrajectoryStep(iteration=1, action="rag_search", arguments={"query": "Retrieval-Augmented Generation definition", "top_k": 3}, observation="[RAG Search Results]: RAG retrieves relevant facts from external knowledge bases... reduces model hallucinations and enables source attribution.", success=True, execution_time_seconds=0.01),
            TrajectoryStep(iteration=2, action="final_answer", arguments={"answer": "Retrieval-Augmented Generation (RAG) is an architectural framework that retrieves relevant facts from external knowledge bases before generating responses, reducing model hallucinations and enabling source attribution.", "topic": "NLP"}, observation="Final answer generated.", success=True, execution_time_seconds=0.001),
        ]
        return AgentResult(
            status="completed",
            answer="Retrieval-Augmented Generation (RAG) is an architectural framework that retrieves relevant facts from external knowledge bases before generating responses, reducing model hallucinations and enabling source attribution.",
            topic="NLP",
            trajectory_length=2,
            trajectory=traj,
            token_usage=TokenUsage(prompt_tokens=110, completion_tokens=45, total_tokens=155),
        )

    # 5. case_rag_02: deep learning unstructured data
    if "deep learning" in q_lower and "unstructured" in q_lower:
        traj = [
            TrajectoryStep(iteration=1, action="rag_search", arguments={"query": "deep learning neural networks unstructured data", "top_k": 3}, observation="[RAG Search Results]: Deep learning utilizes multi-layered artificial neural networks... processing unstructured data such as text, images, and audio.", success=True, execution_time_seconds=0.01),
            TrajectoryStep(iteration=2, action="final_answer", arguments={"answer": "Deep learning uses multi-layered artificial neural networks to process unstructured data including text, images, and audio.", "topic": "Machine Learning"}, observation="Final answer generated.", success=True, execution_time_seconds=0.001),
        ]
        return AgentResult(
            status="completed",
            answer="Deep learning uses multi-layered artificial neural networks to process unstructured data including text, images, and audio.",
            topic="Machine Learning",
            trajectory_length=2,
            trajectory=traj,
            token_usage=TokenUsage(prompt_tokens=120, completion_tokens=40, total_tokens=160),
        )

    # 6. case_rag_03: linear search time complexity
    if "time complexity of linear search" in q_lower:
        traj = [
            TrajectoryStep(iteration=1, action="rag_search", arguments={"query": "linear search time complexity", "top_k": 3}, observation="[RAG Search Results]: Linear search checks every element sequentially... resulting in O(n) time complexity.", success=True, execution_time_seconds=0.01),
            TrajectoryStep(iteration=2, action="final_answer", arguments={"answer": "According to the documents, linear search checks every element sequentially, resulting in O(n) time complexity.", "topic": "Algorithms"}, observation="Final answer generated.", success=True, execution_time_seconds=0.001),
        ]
        return AgentResult(
            status="completed",
            answer="According to the documents, linear search checks every element sequentially, resulting in O(n) time complexity.",
            topic="Algorithms",
            trajectory_length=2,
            trajectory=traj,
            token_usage=TokenUsage(prompt_tokens=105, completion_tokens=35, total_tokens=140),
        )

    # 7. case_multi_01: compare linear and binary search
    if "compare linear search and binary search" in q_lower:
        traj = [
            TrajectoryStep(iteration=1, action="rag_search", arguments={"query": "linear search time complexity", "top_k": 2}, observation="[RAG Search Results]: Linear search has O(n) time complexity.", success=True, execution_time_seconds=0.01),
            TrajectoryStep(iteration=2, action="rag_search", arguments={"query": "binary search time complexity", "top_k": 2}, observation="[RAG Search Results]: Binary search operates in O(log n) time by dividing dataset in half.", success=True, execution_time_seconds=0.01),
            TrajectoryStep(iteration=3, action="final_answer", arguments={"answer": "Linear search checks elements sequentially with O(n) complexity, while binary search repeatedly divides sorted data in half with O(log n) complexity.", "topic": "Algorithms"}, observation="Final answer generated.", success=True, execution_time_seconds=0.001),
        ]
        return AgentResult(
            status="completed",
            answer="Linear search checks elements sequentially with O(n) complexity, while binary search repeatedly divides sorted data in half with O(log n) complexity.",
            topic="Algorithms",
            trajectory_length=3,
            trajectory=traj,
            token_usage=TokenUsage(prompt_tokens=220, completion_tokens=65, total_tokens=285),
        )

    # 8. case_multi_02: AI vs ML vs DL
    if "differences between artificial intelligence, machine learning" in q_lower or "relationships and differences" in q_lower:
        traj = [
            TrajectoryStep(iteration=1, action="rag_search", arguments={"query": "Artificial Intelligence Machine Learning relationship", "top_k": 2}, observation="[RAG Search Results]: Machine learning is a subset of AI focused on learning from data without explicit programming.", success=True, execution_time_seconds=0.01),
            TrajectoryStep(iteration=2, action="rag_search", arguments={"query": "Deep Learning neural networks", "top_k": 2}, observation="[RAG Search Results]: Deep learning is a specialized branch using multi-layered neural networks.", success=True, execution_time_seconds=0.01),
            TrajectoryStep(iteration=3, action="final_answer", arguments={"answer": "AI creates systems mimicking human intelligence; Machine Learning is a subset of AI that learns from data; and Deep Learning is a specialized branch of ML utilizing neural networks for unstructured data.", "topic": "AI Overview"}, observation="Final answer generated.", success=True, execution_time_seconds=0.001),
        ]
        return AgentResult(
            status="completed",
            answer="AI creates systems mimicking human intelligence; Machine Learning is a subset of AI that learns from data; and Deep Learning is a specialized branch of ML utilizing neural networks for unstructured data.",
            topic="AI Overview",
            trajectory_length=3,
            trajectory=traj,
            token_usage=TokenUsage(prompt_tokens=240, completion_tokens=70, total_tokens=310),
        )

    # 9. case_multi_03: RAG improvements without retraining
    if "how does retrieval-augmented generation improve" in q_lower:
        traj = [
            TrajectoryStep(iteration=1, action="rag_search", arguments={"query": "RAG external knowledge bases hallucinations", "top_k": 2}, observation="[RAG Search Results]: Retrieves facts from external knowledge bases, reducing model hallucinations.", success=True, execution_time_seconds=0.01),
            TrajectoryStep(iteration=2, action="rag_search", arguments={"query": "RAG source attribution retraining", "top_k": 2}, observation="[RAG Search Results]: Enables source attribution and keeps responses up to date without retraining.", success=True, execution_time_seconds=0.01),
            TrajectoryStep(iteration=3, action="final_answer", arguments={"answer": "RAG connects models to external knowledge bases, reducing hallucinations, providing source attribution, and keeping information current without requiring expensive retraining.", "topic": "RAG"}, observation="Final answer generated.", success=True, execution_time_seconds=0.001),
        ]
        return AgentResult(
            status="completed",
            answer="RAG connects models to external knowledge bases, reducing hallucinations, providing source attribution, and keeping information current without requiring expensive retraining.",
            topic="RAG",
            trajectory_length=3,
            trajectory=traj,
            token_usage=TokenUsage(prompt_tokens=230, completion_tokens=60, total_tokens=290),
        )

    # 10. case_chain_01: 500 - 9 = 491
    if "linear search examines 500 items" in q_lower:
        traj = [
            TrajectoryStep(iteration=1, action="rag_search", arguments={"query": "linear search sequential 500 binary 9", "top_k": 2}, observation="[RAG Search Results]: Linear search O(n), binary search O(log n).", success=True, execution_time_seconds=0.01),
            TrajectoryStep(iteration=2, action="calculator", arguments={"a": 500.0, "b": 9.0, "operation": "subtract"}, observation="[Calculator Result]: 500.0 subtract 9.0 = 491.0", success=True, execution_time_seconds=0.001),
            TrajectoryStep(iteration=3, action="final_answer", arguments={"answer": "Linear search takes 500 operations and binary search takes 9, so linear search takes 491 more operations.", "topic": "Algorithm Comparison"}, observation="Final answer generated.", success=True, execution_time_seconds=0.001),
        ]
        return AgentResult(
            status="completed",
            answer="Linear search takes 500 operations and binary search takes 9, so linear search takes 491 more operations.",
            topic="Algorithm Comparison",
            trajectory_length=3,
            trajectory=traj,
            token_usage=TokenUsage(prompt_tokens=210, completion_tokens=55, total_tokens=265),
        )

    # 11. case_chain_02: 250 * 50 = 12500
    if "array of 250 elements" in q_lower and "50 times" in q_lower:
        traj = [
            TrajectoryStep(iteration=1, action="calculator", arguments={"a": 250.0, "b": 50.0, "operation": "multiply"}, observation="[Calculator Result]: 250.0 multiply 50.0 = 12500.0", success=True, execution_time_seconds=0.001),
            TrajectoryStep(iteration=2, action="final_answer", arguments={"answer": "The total number of operations across 50 runs is 12500.", "topic": "Computation"}, observation="Final answer generated.", success=True, execution_time_seconds=0.001),
        ]
        return AgentResult(
            status="completed",
            answer="The total number of operations across 50 runs is 12500.",
            topic="Computation",
            trajectory_length=2,
            trajectory=traj,
            token_usage=TokenUsage(prompt_tokens=90, completion_tokens=25, total_tokens=115),
        )

    # 12. case_chain_03: 4 * 8 = 32
    if "4 layers of neural networks" in q_lower or "feature maps" in q_lower:
        traj = [
            TrajectoryStep(iteration=1, action="calculator", arguments={"a": 4.0, "b": 8.0, "operation": "multiply"}, observation="[Calculator Result]: 4.0 multiply 8.0 = 32.0", success=True, execution_time_seconds=0.001),
            TrajectoryStep(iteration=2, action="final_answer", arguments={"answer": "With 4 layers and 8 feature maps per layer, the total is 32 feature maps.", "topic": "Neural Networks"}, observation="Final answer generated.", success=True, execution_time_seconds=0.001),
        ]
        return AgentResult(
            status="completed",
            answer="With 4 layers and 8 feature maps per layer, the total is 32 feature maps.",
            topic="Neural Networks",
            trajectory_length=2,
            trajectory=traj,
            token_usage=TokenUsage(prompt_tokens=95, completion_tokens=28, total_tokens=123),
        )

    # 13. case_clarify_01: ambiguous its complexity
    if "what is its exact time complexity" in q_lower:
        traj = [
            TrajectoryStep(iteration=1, action="ask_user_clarification", arguments={"question": "Which algorithm or data structure are you asking about?"}, observation="[Clarification Prompt]: Which algorithm or data structure are you asking about?", success=True, execution_time_seconds=0.001),
        ]
        return AgentResult(
            status="clarification_required",
            answer="Which algorithm or data structure are you asking about?",
            topic="Clarification",
            trajectory_length=1,
            trajectory=traj,
            clarification_required=True,
            token_usage=TokenUsage(prompt_tokens=55, completion_tokens=18, total_tokens=73),
        )

    # 14. case_clarify_02: ambiguous how many layers
    if "how many layers does it have" in q_lower:
        traj = [
            TrajectoryStep(iteration=1, action="ask_user_clarification", arguments={"question": "Which specific neural network model or architecture are you referring to?"}, observation="[Clarification Prompt]: Which specific neural network model or architecture are you referring to?", success=True, execution_time_seconds=0.001),
        ]
        return AgentResult(
            status="clarification_required",
            answer="Which specific neural network model or architecture are you referring to?",
            topic="Clarification",
            trajectory_length=1,
            trajectory=traj,
            clarification_required=True,
            token_usage=TokenUsage(prompt_tokens=55, completion_tokens=20, total_tokens=75),
        )

    # 15. case_bound_01: 100 - 37 = 63
    if "100 minus 37" in q_lower:
        traj = [
            TrajectoryStep(iteration=1, action="calculator", arguments={"a": 100.0, "b": 37.0, "operation": "subtract"}, observation="[Calculator Result]: 100.0 subtract 37.0 = 63.0", success=True, execution_time_seconds=0.001),
            TrajectoryStep(iteration=2, action="final_answer", arguments={"answer": "100 minus 37 is 63.", "topic": "Arithmetic"}, observation="Final answer generated.", success=True, execution_time_seconds=0.001),
        ]
        return AgentResult(
            status="completed",
            answer="100 minus 37 is 63.",
            topic="Arithmetic",
            trajectory_length=2,
            trajectory=traj,
            token_usage=TokenUsage(prompt_tokens=60, completion_tokens=18, total_tokens=78),
        )

    # 16. case_bound_02: primary benefit of RAG
    if "primary benefit of rag" in q_lower:
        traj = [
            TrajectoryStep(iteration=1, action="rag_search", arguments={"query": "primary benefit of RAG", "top_k": 2}, observation="[RAG Search Results]: Reduces model hallucinations and keeps responses up to date without retraining.", success=True, execution_time_seconds=0.01),
            TrajectoryStep(iteration=2, action="final_answer", arguments={"answer": "The primary benefit of RAG is reducing hallucinations and keeping knowledge up to date by retrieving facts from external knowledge bases.", "topic": "RAG"}, observation="Final answer generated.", success=True, execution_time_seconds=0.001),
        ]
        return AgentResult(
            status="completed",
            answer="The primary benefit of RAG is reducing hallucinations and keeping knowledge up to date by retrieving facts from external knowledge bases.",
            topic="RAG",
            trajectory_length=2,
            trajectory=traj,
            token_usage=TokenUsage(prompt_tokens=100, completion_tokens=30, total_tokens=130),
        )

    # Default fallback
    return AgentResult(
        status="completed",
        answer="Generic response.",
        topic="General",
        trajectory_length=1,
        trajectory=[TrajectoryStep(iteration=1, action="final_answer", arguments={}, observation="Done", success=True)],
        token_usage=TokenUsage(prompt_tokens=50, completion_tokens=15, total_tokens=65),
    )


def test_full_benchmark_run():
    """Run evaluation benchmark suite against all 16 cases and verify summary metrics and report generation."""
    report_path = "tests/evaluation/results/agent_evaluation_results.md"
    summary = run_benchmark_evaluation(
        cases=BENCHMARK_CASES,
        agent_runner_fn=simulated_agent_runner,
        output_report_path=report_path,
    )

    assert isinstance(summary, BenchmarkSummary)
    assert summary.total_tasks == 16
    assert summary.successful_tasks == 16
    assert summary.task_completion_rate == 100.0
    assert summary.tool_call_correctness == 100.0
    assert 1.5 <= summary.avg_trajectory_length <= 3.5
    assert summary.hard_failures == 0
    assert summary.soft_failures == 0
    assert summary.cascading_soft_failures == 0
    assert summary.total_tokens > 0

    # Verify that Markdown report was written
    assert os.path.exists(report_path)
    with open(report_path, "r", encoding="utf-8") as f:
        content = f.read()

    assert "# W16 Agent Evaluation Report" in content
    assert "Task Completion Rate (TCR)" in content
    assert "100.0%" in content
    assert "case_calc_01" in content
    assert "case_multi_01" in content
    assert "case_clarify_01" in content
