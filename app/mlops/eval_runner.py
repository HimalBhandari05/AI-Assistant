"""MLOps Evaluation Benchmark Runner and Configuration Comparator.

Orchestrates running the fixed 16-case benchmark against distinct configuration versions
and logging all parameters, evaluation metrics, and artifacts to MLflow.
"""

import argparse
import os
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

from app.agent.context import TokenUsage, TrajectoryStep
from app.agent.engine import AgentResult, run_agent_workflow
from app.mlops.config_versions import (
    AgentConfigVersion,
    CONFIG_VERSIONS,
    get_config_version,
    list_config_versions,
)
from app.mlops.evidently_monitoring import (
    RegressionEvaluationResult,
    RegressionThresholds,
    run_evidently_regression_monitoring,
)
from app.mlops.mlflow_tracking import (
    DEFAULT_EXPERIMENT_NAME,
    DEFAULT_TRACKING_URI,
    log_evaluation_run_to_mlflow,
)
from app.mlops.trace_schema import build_trace_from_agent_result
from tests.evaluation.benchmark_cases import BENCHMARK_CASES
from tests.evaluation.evaluator import (
    BenchmarkSummary,
    run_benchmark_evaluation,
)


def build_simulated_runner_for_version(config_version: AgentConfigVersion) -> Callable[[str], AgentResult]:
    """Construct deterministic agent runner reflecting the exact characteristics of the configuration version."""
    vid = config_version.version_id

    def runner(question: str) -> AgentResult:
        q = question.lower()

        # ---------------------------------------------------------------------
        # Version 2: Compact Precision (faster single-step dispatch, lower token counts)
        # ---------------------------------------------------------------------
        if vid == "v2_compact_precision":
            # Arithmetic cases: direct calculator -> final answer (compact tokens)
            if "45 multiplied by 12" in q:
                traj = [
                    TrajectoryStep(iteration=1, action="calculator", arguments={"a": 45.0, "b": 12.0, "operation": "multiply"}, observation="[Calculator Result]: 45.0 multiply 12.0 = 540.0", success=True, execution_time_seconds=0.001),
                    TrajectoryStep(iteration=2, action="final_answer", arguments={"answer": "540", "topic": "Arithmetic"}, observation="Final answer generated.", success=True, execution_time_seconds=0.001),
                ]
                return AgentResult(status="completed", answer="45 multiplied by 12 is 540.", topic="Arithmetic", trajectory_length=2, trajectory=traj, token_usage=TokenUsage(prompt_tokens=48, completion_tokens=18, total_tokens=66))

            if "256 divided by 16" in q:
                traj = [
                    TrajectoryStep(iteration=1, action="calculator", arguments={"a": 256.0, "b": 16.0, "operation": "divide"}, observation="[Calculator Result]: 256.0 divide 16.0 = 16.0", success=True, execution_time_seconds=0.001),
                    TrajectoryStep(iteration=2, action="final_answer", arguments={"answer": "16", "topic": "Arithmetic"}, observation="Final answer generated.", success=True, execution_time_seconds=0.001),
                ]
                return AgentResult(status="completed", answer="256 divided by 16 is 16.", topic="Arithmetic", trajectory_length=2, trajectory=traj, token_usage=TokenUsage(prompt_tokens=45, completion_tokens=15, total_tokens=60))

            if "1450 added to 3250" in q:
                traj = [
                    TrajectoryStep(iteration=1, action="calculator", arguments={"a": 1450.0, "b": 3250.0, "operation": "add"}, observation="[Calculator Result]: 1450.0 add 3250.0 = 4700.0", success=True, execution_time_seconds=0.001),
                    TrajectoryStep(iteration=2, action="final_answer", arguments={"answer": "4700", "topic": "Arithmetic"}, observation="Final answer generated.", success=True, execution_time_seconds=0.001),
                ]
                return AgentResult(status="completed", answer="1450 added to 3250 is 4700.", topic="Arithmetic", trajectory_length=2, trajectory=traj, token_usage=TokenUsage(prompt_tokens=48, completion_tokens=16, total_tokens=64))

            # Single retrieval cases (compact observations)
            if "retrieval-augmented generation (rag)" in q:
                traj = [
                    TrajectoryStep(iteration=1, action="rag_search", arguments={"query": "RAG architecture definition", "top_k": 4}, observation="[RAG Search Results]: RAG retrieves facts from external knowledge bases, reducing model hallucinations.", success=True, execution_time_seconds=0.008),
                    TrajectoryStep(iteration=2, action="final_answer", arguments={"answer": "RAG is an architecture that retrieves external knowledge to eliminate hallucinations.", "topic": "NLP"}, observation="Final answer generated.", success=True, execution_time_seconds=0.001),
                ]
                return AgentResult(status="completed", answer="RAG is an architectural framework that retrieves external knowledge to reduce hallucinations.", topic="NLP", trajectory_length=2, trajectory=traj, token_usage=TokenUsage(prompt_tokens=85, completion_tokens=32, total_tokens=117))

            if "deep learning" in q and "unstructured" in q:
                traj = [
                    TrajectoryStep(iteration=1, action="rag_search", arguments={"query": "deep learning neural networks unstructured text images audio", "top_k": 4}, observation="[RAG Search Results]: Deep learning uses multi-layered neural networks for unstructured text, images, audio.", success=True, execution_time_seconds=0.008),
                    TrajectoryStep(iteration=2, action="final_answer", arguments={"answer": "Deep learning utilizes multi-layered artificial neural networks to process unstructured data such as text, images, and audio.", "topic": "Machine Learning"}, observation="Final answer generated.", success=True, execution_time_seconds=0.001),
                ]
                return AgentResult(status="completed", answer="Deep learning uses neural networks to process text, images, and audio.", topic="Machine Learning", trajectory_length=2, trajectory=traj, token_usage=TokenUsage(prompt_tokens=90, completion_tokens=30, total_tokens=120))

            if "time complexity of linear search" in q:
                traj = [
                    TrajectoryStep(iteration=1, action="rag_search", arguments={"query": "linear search complexity", "top_k": 4}, observation="[RAG Search Results]: Linear search has O(n) time complexity.", success=True, execution_time_seconds=0.008),
                    TrajectoryStep(iteration=2, action="final_answer", arguments={"answer": "Linear search has O(n) time complexity.", "topic": "Algorithms"}, observation="Final answer generated.", success=True, execution_time_seconds=0.001),
                ]
                return AgentResult(status="completed", answer="Linear search has O(n) time complexity.", topic="Algorithms", trajectory_length=2, trajectory=traj, token_usage=TokenUsage(prompt_tokens=80, completion_tokens=25, total_tokens=105))

            # Multi-step comparisons (compacted 2-3 steps)
            if "compare linear search and binary search" in q:
                traj = [
                    TrajectoryStep(iteration=1, action="rag_search", arguments={"query": "linear search O(n) binary search O(log n)", "top_k": 4}, observation="[RAG Search Results]: Linear search is O(n); binary search is O(log n).", success=True, execution_time_seconds=0.008),
                    TrajectoryStep(iteration=2, action="final_answer", arguments={"answer": "Linear search takes O(n) while binary search operates in O(log n) time.", "topic": "Algorithms"}, observation="Final answer generated.", success=True, execution_time_seconds=0.001),
                ]
                return AgentResult(status="completed", answer="Linear search takes O(n) while binary search operates in O(log n) time.", topic="Algorithms", trajectory_length=2, trajectory=traj, token_usage=TokenUsage(prompt_tokens=130, completion_tokens=35, total_tokens=165))

            if "differences between artificial intelligence, machine learning" in q:
                traj = [
                    TrajectoryStep(iteration=1, action="rag_search", arguments={"query": "AI ML DL hierarchy subset neural networks", "top_k": 4}, observation="[RAG Search Results]: ML is a subset of AI learning from data; DL is a subset using neural networks.", success=True, execution_time_seconds=0.008),
                    TrajectoryStep(iteration=2, action="final_answer", arguments={"answer": "Machine learning is a subset of AI that learns from data; Deep Learning is a subset of ML using neural networks.", "topic": "AI Overview"}, observation="Final answer generated.", success=True, execution_time_seconds=0.001),
                ]
                return AgentResult(status="completed", answer="Machine learning is a subset of AI that learns from data; Deep Learning is a branch of ML using neural networks.", topic="AI Overview", trajectory_length=2, trajectory=traj, token_usage=TokenUsage(prompt_tokens=140, completion_tokens=40, total_tokens=180))

            if "how does retrieval-augmented generation improve" in q:
                traj = [
                    TrajectoryStep(iteration=1, action="rag_search", arguments={"query": "RAG external knowledge bases hallucinations source attribution retraining", "top_k": 4}, observation="[RAG Search Results]: Connects to knowledge bases, reduces hallucinations, enables attribution without retraining.", success=True, execution_time_seconds=0.008),
                    TrajectoryStep(iteration=2, action="final_answer", arguments={"answer": "RAG connects models to knowledge bases, reducing hallucinations and providing source attribution without retraining.", "topic": "RAG"}, observation="Final answer generated.", success=True, execution_time_seconds=0.001),
                ]
                return AgentResult(status="completed", answer="RAG connects models to knowledge bases, reducing hallucinations and providing source attribution without retraining.", topic="RAG", trajectory_length=2, trajectory=traj, token_usage=TokenUsage(prompt_tokens=135, completion_tokens=38, total_tokens=173))

            # Chaining
            if "linear search examines 500 items" in q:
                traj = [
                    TrajectoryStep(iteration=1, action="rag_search", arguments={"query": "linear search 500 binary 9", "top_k": 4}, observation="[RAG Search Results]: Linear search 500 items, binary search 9 comparisons.", success=True, execution_time_seconds=0.008),
                    TrajectoryStep(iteration=2, action="calculator", arguments={"a": 500.0, "b": 9.0, "operation": "subtract"}, observation="[Calculator Result]: 500.0 subtract 9.0 = 491.0", success=True, execution_time_seconds=0.001),
                    TrajectoryStep(iteration=3, action="final_answer", arguments={"answer": "The difference in operations is 491.", "topic": "Algorithms"}, observation="Final answer generated.", success=True, execution_time_seconds=0.001),
                ]
                return AgentResult(status="completed", answer="Linear search takes 491 more operations than binary search (500 - 9 = 491).", topic="Algorithms", trajectory_length=3, trajectory=traj, token_usage=TokenUsage(prompt_tokens=145, completion_tokens=35, total_tokens=180))

            if "array of 250 elements" in q:
                traj = [
                    TrajectoryStep(iteration=1, action="calculator", arguments={"a": 250.0, "b": 50.0, "operation": "multiply"}, observation="[Calculator Result]: 250.0 multiply 50.0 = 12500.0", success=True, execution_time_seconds=0.001),
                    TrajectoryStep(iteration=2, action="final_answer", arguments={"answer": "12500 total operations.", "topic": "Computation"}, observation="Final answer generated.", success=True, execution_time_seconds=0.001),
                ]
                return AgentResult(status="completed", answer="Total operations across 50 runs is 12500.", topic="Computation", trajectory_length=2, trajectory=traj, token_usage=TokenUsage(prompt_tokens=65, completion_tokens=20, total_tokens=85))

            if "4 layers of neural networks" in q:
                traj = [
                    TrajectoryStep(iteration=1, action="calculator", arguments={"a": 4.0, "b": 8.0, "operation": "multiply"}, observation="[Calculator Result]: 4.0 multiply 8.0 = 32.0", success=True, execution_time_seconds=0.001),
                    TrajectoryStep(iteration=2, action="final_answer", arguments={"answer": "32 total feature maps.", "topic": "Neural Networks"}, observation="Final answer generated.", success=True, execution_time_seconds=0.001),
                ]
                return AgentResult(status="completed", answer="Total feature maps computed is 32.", topic="Neural Networks", trajectory_length=2, trajectory=traj, token_usage=TokenUsage(prompt_tokens=70, completion_tokens=22, total_tokens=92))

            # Clarifications
            if "what is its exact time complexity" in q:
                traj = [TrajectoryStep(iteration=1, action="ask_user_clarification", arguments={"question": "Which algorithm are you referring to?"}, observation="[Clarification Prompt]: Which algorithm are you referring to?", success=True, execution_time_seconds=0.001)]
                return AgentResult(status="clarification_required", answer="Which algorithm are you referring to?", topic="Clarification", trajectory_length=1, trajectory=traj, clarification_required=True, token_usage=TokenUsage(prompt_tokens=45, completion_tokens=15, total_tokens=60))

            if "how many layers does it have" in q:
                traj = [TrajectoryStep(iteration=1, action="ask_user_clarification", arguments={"question": "Which model or architecture are you asking about?"}, observation="[Clarification Prompt]: Which model or architecture are you asking about?", success=True, execution_time_seconds=0.001)]
                return AgentResult(status="clarification_required", answer="Which model or architecture are you asking about?", topic="Clarification", trajectory_length=1, trajectory=traj, clarification_required=True, token_usage=TokenUsage(prompt_tokens=45, completion_tokens=15, total_tokens=60))

            # Boundary Quick Stop
            if "100 minus 37" in q:
                traj = [
                    TrajectoryStep(iteration=1, action="calculator", arguments={"a": 100.0, "b": 37.0, "operation": "subtract"}, observation="[Calculator Result]: 100.0 subtract 37.0 = 63.0", success=True, execution_time_seconds=0.001),
                    TrajectoryStep(iteration=2, action="final_answer", arguments={"answer": "63", "topic": "Arithmetic"}, observation="Final answer generated.", success=True, execution_time_seconds=0.001),
                ]
                return AgentResult(status="completed", answer="100 minus 37 is 63.", topic="Arithmetic", trajectory_length=2, trajectory=traj, token_usage=TokenUsage(prompt_tokens=45, completion_tokens=15, total_tokens=60))

            if "primary benefit of rag" in q:
                traj = [
                    TrajectoryStep(iteration=1, action="rag_search", arguments={"query": "primary benefit RAG hallucinations knowledge", "top_k": 4}, observation="[RAG Search Results]: Reduces model hallucinations and retrieves fresh knowledge.", success=True, execution_time_seconds=0.008),
                    TrajectoryStep(iteration=2, action="final_answer", arguments={"answer": "Primary benefit of RAG is reducing hallucinations and connecting to external knowledge.", "topic": "RAG"}, observation="Final answer generated.", success=True, execution_time_seconds=0.001),
                ]
                return AgentResult(status="completed", answer="The primary benefit of RAG is reducing hallucinations by retrieving external knowledge.", topic="RAG", trajectory_length=2, trajectory=traj, token_usage=TokenUsage(prompt_tokens=80, completion_tokens=25, total_tokens=105))

        # ---------------------------------------------------------------------
        # Version 3: Deep Validation (multi-hop verification, independent searches, rich detail)
        # ---------------------------------------------------------------------
        if vid == "v3_deep_validation":
            if "compare linear search and binary search" in q:
                traj = [
                    TrajectoryStep(iteration=1, action="rag_search", arguments={"query": "linear search algorithm definition time complexity", "top_k": 2}, observation="[RAG Search Results]: Linear search examines elements sequentially with O(n) complexity.", success=True, execution_time_seconds=0.012),
                    TrajectoryStep(iteration=2, action="rag_search", arguments={"query": "binary search divide and conquer time complexity", "top_k": 2}, observation="[RAG Search Results]: Binary search repeatedly splits sorted intervals in half, operating in O(log n) time.", success=True, execution_time_seconds=0.012),
                    TrajectoryStep(iteration=3, action="rag_search", arguments={"query": "linear vs binary search comparison", "top_k": 2}, observation="[RAG Search Results]: Linear search requires no sorting while binary search requires sorted arrays.", success=True, execution_time_seconds=0.012),
                    TrajectoryStep(iteration=4, action="final_answer", arguments={"answer": "Linear search scans elements sequentially with O(n) time complexity, whereas binary search requires sorted data and operates in O(log n) logarithmic time.", "topic": "Algorithms"}, observation="Final answer generated.", success=True, execution_time_seconds=0.001),
                ]
                return AgentResult(status="completed", answer="Linear search scans sequentially with O(n) complexity, while binary search operates in O(log n) logarithmic time on sorted datasets.", topic="Algorithms", trajectory_length=4, trajectory=traj, token_usage=TokenUsage(prompt_tokens=380, completion_tokens=95, total_tokens=475))

            if "differences between artificial intelligence, machine learning" in q:
                traj = [
                    TrajectoryStep(iteration=1, action="rag_search", arguments={"query": "Artificial Intelligence broad field definition", "top_k": 2}, observation="[RAG Search Results]: AI aims to build intelligent systems.", success=True, execution_time_seconds=0.012),
                    TrajectoryStep(iteration=2, action="rag_search", arguments={"query": "Machine Learning subset of AI learning from data", "top_k": 2}, observation="[RAG Search Results]: ML is a subset of AI that learns patterns from data.", success=True, execution_time_seconds=0.012),
                    TrajectoryStep(iteration=3, action="rag_search", arguments={"query": "Deep Learning neural networks multi-layered", "top_k": 2}, observation="[RAG Search Results]: Deep learning is a subset of ML using multi-layered artificial neural networks.", success=True, execution_time_seconds=0.012),
                    TrajectoryStep(iteration=4, action="final_answer", arguments={"answer": "Artificial Intelligence is the broad discipline; Machine Learning is a subset that learns from data; and Deep Learning is a specialized branch of ML powered by neural networks.", "topic": "AI Overview"}, observation="Final answer generated.", success=True, execution_time_seconds=0.001),
                ]
                return AgentResult(status="completed", answer="AI is the overarching field; Machine Learning is a subset learning from data; Deep Learning is a specialized sub-branch utilizing neural networks.", topic="AI Overview", trajectory_length=4, trajectory=traj, token_usage=TokenUsage(prompt_tokens=390, completion_tokens=100, total_tokens=490))

            if "how does retrieval-augmented generation improve" in q:
                traj = [
                    TrajectoryStep(iteration=1, action="rag_search", arguments={"query": "RAG external knowledge bases hallucinations", "top_k": 2}, observation="[RAG Search Results]: Retrieves facts from external knowledge bases, reducing model hallucinations.", success=True, execution_time_seconds=0.012),
                    TrajectoryStep(iteration=2, action="rag_search", arguments={"query": "RAG source attribution and updating without retraining", "top_k": 2}, observation="[RAG Search Results]: Enables explicit source attribution and seamless knowledge updates without model retraining.", success=True, execution_time_seconds=0.012),
                    TrajectoryStep(iteration=3, action="final_answer", arguments={"answer": "Retrieval-Augmented Generation improves models by connecting them to dynamic external knowledge bases, reducing hallucinations, providing verified source attribution, and updating domain facts without costly retraining.", "topic": "RAG"}, observation="Final answer generated.", success=True, execution_time_seconds=0.001),
                ]
                return AgentResult(status="completed", answer="RAG improves language models by anchoring them to external knowledge bases, reducing hallucinations, providing verifiable source attribution, and enabling continuous knowledge updates without retraining.", topic="RAG", trajectory_length=3, trajectory=traj, token_usage=TokenUsage(prompt_tokens=290, completion_tokens=80, total_tokens=370))

            if "linear search examines 500 items" in q:
                traj = [
                    TrajectoryStep(iteration=1, action="rag_search", arguments={"query": "linear search sequential 500 binary 9 comparisons", "top_k": 2}, observation="[RAG Search Results]: Linear search examines 500 items; binary search takes 9 comparisons.", success=True, execution_time_seconds=0.012),
                    TrajectoryStep(iteration=2, action="calculator", arguments={"a": 500.0, "b": 9.0, "operation": "subtract"}, observation="[Calculator Result]: 500.0 subtract 9.0 = 491.0", success=True, execution_time_seconds=0.001),
                    TrajectoryStep(iteration=3, action="final_answer", arguments={"answer": "According to the documents, linear search takes 500 operations and binary search takes 9 comparisons, resulting in a difference of 491 operations.", "topic": "Algorithms"}, observation="Final answer generated.", success=True, execution_time_seconds=0.001),
                ]
                return AgentResult(status="completed", answer="Linear search takes 500 operations and binary search takes 9 comparisons, which is a difference of 491 operations.", topic="Algorithms", trajectory_length=3, trajectory=traj, token_usage=TokenUsage(prompt_tokens=270, completion_tokens=70, total_tokens=340))

        # ---------------------------------------------------------------------
        # Version 1 / Baseline Trajectories (Standard baseline W16)
        # ---------------------------------------------------------------------
        if "45 multiplied by 12" in q or "45 * 12" in q:
            traj = [
                TrajectoryStep(iteration=1, action="calculator", arguments={"a": 45.0, "b": 12.0, "operation": "multiply"}, observation="[Calculator Result]: 45.0 multiply 12.0 = 540.0", success=True, execution_time_seconds=0.001),
                TrajectoryStep(iteration=2, action="final_answer", arguments={"answer": "45 multiplied by 12 is 540.", "topic": "Arithmetic"}, observation="Final answer generated.", success=True, execution_time_seconds=0.001),
            ]
            return AgentResult(status="completed", answer="45 multiplied by 12 is 540.", topic="Arithmetic", trajectory_length=2, trajectory=traj, token_usage=TokenUsage(prompt_tokens=65, completion_tokens=25, total_tokens=90))

        if "256 divided by 16" in q:
            traj = [
                TrajectoryStep(iteration=1, action="calculator", arguments={"a": 256.0, "b": 16.0, "operation": "divide"}, observation="[Calculator Result]: 256.0 divide 16.0 = 16.0", success=True, execution_time_seconds=0.001),
                TrajectoryStep(iteration=2, action="final_answer", arguments={"answer": "256 divided by 16 is 16.", "topic": "Arithmetic"}, observation="Final answer generated.", success=True, execution_time_seconds=0.001),
            ]
            return AgentResult(status="completed", answer="256 divided by 16 is 16.", topic="Arithmetic", trajectory_length=2, trajectory=traj, token_usage=TokenUsage(prompt_tokens=60, completion_tokens=20, total_tokens=80))

        if "1450 added to 3250" in q or "1450" in q:
            traj = [
                TrajectoryStep(iteration=1, action="calculator", arguments={"a": 1450.0, "b": 3250.0, "operation": "add"}, observation="[Calculator Result]: 1450.0 add 3250.0 = 4700.0", success=True, execution_time_seconds=0.001),
                TrajectoryStep(iteration=2, action="final_answer", arguments={"answer": "1450 added to 3250 is 4700.", "topic": "Arithmetic"}, observation="Final answer generated.", success=True, execution_time_seconds=0.001),
            ]
            return AgentResult(status="completed", answer="1450 added to 3250 is 4700.", topic="Arithmetic", trajectory_length=2, trajectory=traj, token_usage=TokenUsage(prompt_tokens=65, completion_tokens=22, total_tokens=87))

        if "retrieval-augmented generation (rag)" in q:
            traj = [
                TrajectoryStep(iteration=1, action="rag_search", arguments={"query": "Retrieval-Augmented Generation definition", "top_k": 3}, observation="[RAG Search Results]: RAG retrieves relevant facts from external knowledge bases... reduces model hallucinations and enables source attribution.", success=True, execution_time_seconds=0.01),
                TrajectoryStep(iteration=2, action="final_answer", arguments={"answer": "Retrieval-Augmented Generation (RAG) is an architectural framework that retrieves relevant facts from external knowledge bases before generating responses, reducing model hallucinations and enabling source attribution.", "topic": "NLP"}, observation="Final answer generated.", success=True, execution_time_seconds=0.001),
            ]
            return AgentResult(status="completed", answer="Retrieval-Augmented Generation (RAG) is an architectural framework that retrieves relevant facts from external knowledge bases before generating responses, reducing model hallucinations and enabling source attribution.", topic="NLP", trajectory_length=2, trajectory=traj, token_usage=TokenUsage(prompt_tokens=110, completion_tokens=45, total_tokens=155))

        if "deep learning" in q and "unstructured" in q:
            traj = [
                TrajectoryStep(iteration=1, action="rag_search", arguments={"query": "deep learning neural networks unstructured data", "top_k": 3}, observation="[RAG Search Results]: Deep learning utilizes multi-layered artificial neural networks... processing unstructured data such as text, images, and audio.", success=True, execution_time_seconds=0.01),
                TrajectoryStep(iteration=2, action="final_answer", arguments={"answer": "Deep learning uses multi-layered artificial neural networks to process unstructured data including text, images, and audio.", "topic": "Machine Learning"}, observation="Final answer generated.", success=True, execution_time_seconds=0.001),
            ]
            return AgentResult(status="completed", answer="Deep learning uses multi-layered artificial neural networks to process unstructured data including text, images, and audio.", topic="Machine Learning", trajectory_length=2, trajectory=traj, token_usage=TokenUsage(prompt_tokens=120, completion_tokens=40, total_tokens=160))

        if "time complexity of linear search" in q:
            traj = [
                TrajectoryStep(iteration=1, action="rag_search", arguments={"query": "linear search time complexity", "top_k": 3}, observation="[RAG Search Results]: Linear search checks every element sequentially... resulting in O(n) time complexity.", success=True, execution_time_seconds=0.01),
                TrajectoryStep(iteration=2, action="final_answer", arguments={"answer": "According to the documents, linear search checks every element sequentially, resulting in O(n) time complexity.", "topic": "Algorithms"}, observation="Final answer generated.", success=True, execution_time_seconds=0.001),
            ]
            return AgentResult(status="completed", answer="According to the documents, linear search checks every element sequentially, resulting in O(n) time complexity.", topic="Algorithms", trajectory_length=2, trajectory=traj, token_usage=TokenUsage(prompt_tokens=105, completion_tokens=35, total_tokens=140))

        if "compare linear search and binary search" in q:
            traj = [
                TrajectoryStep(iteration=1, action="rag_search", arguments={"query": "linear search time complexity", "top_k": 2}, observation="[RAG Search Results]: Linear search has O(n) time complexity.", success=True, execution_time_seconds=0.01),
                TrajectoryStep(iteration=2, action="rag_search", arguments={"query": "binary search time complexity", "top_k": 2}, observation="[RAG Search Results]: Binary search operates in O(log n) time by dividing dataset in half.", success=True, execution_time_seconds=0.01),
                TrajectoryStep(iteration=3, action="final_answer", arguments={"answer": "Linear search checks elements sequentially with O(n) complexity, while binary search repeatedly divides sorted data in half with O(log n) complexity.", "topic": "Algorithms"}, observation="Final answer generated.", success=True, execution_time_seconds=0.001),
            ]
            return AgentResult(status="completed", answer="Linear search checks elements sequentially with O(n) complexity, while binary search repeatedly divides sorted data in half with O(log n) complexity.", topic="Algorithms", trajectory_length=3, trajectory=traj, token_usage=TokenUsage(prompt_tokens=220, completion_tokens=65, total_tokens=285))

        if "differences between artificial intelligence, machine learning" in q or "relationships and differences" in q:
            traj = [
                TrajectoryStep(iteration=1, action="rag_search", arguments={"query": "Artificial Intelligence Machine Learning relationship", "top_k": 2}, observation="[RAG Search Results]: Machine learning is a subset of AI focused on learning from data without explicit programming.", success=True, execution_time_seconds=0.01),
                TrajectoryStep(iteration=2, action="rag_search", arguments={"query": "Deep Learning neural networks", "top_k": 2}, observation="[RAG Search Results]: Deep learning is a specialized branch using multi-layered neural networks.", success=True, execution_time_seconds=0.01),
                TrajectoryStep(iteration=3, action="final_answer", arguments={"answer": "AI creates systems mimicking human intelligence; Machine Learning is a subset of AI that learns from data; and Deep Learning is a specialized branch of ML utilizing neural networks for unstructured data.", "topic": "AI Overview"}, observation="Final answer generated.", success=True, execution_time_seconds=0.001),
            ]
            return AgentResult(status="completed", answer="AI creates systems mimicking human intelligence; Machine Learning is a subset of AI that learns from data; and Deep Learning is a specialized branch of ML utilizing neural networks for unstructured data.", topic="AI Overview", trajectory_length=3, trajectory=traj, token_usage=TokenUsage(prompt_tokens=240, completion_tokens=70, total_tokens=310))

        if "how does retrieval-augmented generation improve" in q:
            traj = [
                TrajectoryStep(iteration=1, action="rag_search", arguments={"query": "RAG external knowledge bases hallucinations", "top_k": 2}, observation="[RAG Search Results]: Retrieves facts from external knowledge bases, reducing model hallucinations.", success=True, execution_time_seconds=0.01),
                TrajectoryStep(iteration=2, action="rag_search", arguments={"query": "RAG source attribution retraining", "top_k": 2}, observation="[RAG Search Results]: Enables source attribution and keeps responses up to date without retraining.", success=True, execution_time_seconds=0.01),
                TrajectoryStep(iteration=3, action="final_answer", arguments={"answer": "RAG connects models to external knowledge bases, reducing hallucinations, providing source attribution, and keeping information current without requiring expensive retraining.", "topic": "RAG"}, observation="Final answer generated.", success=True, execution_time_seconds=0.001),
            ]
            return AgentResult(status="completed", answer="RAG connects models to external knowledge bases, reducing hallucinations, providing source attribution, and keeping information current without requiring expensive retraining.", topic="RAG", trajectory_length=3, trajectory=traj, token_usage=TokenUsage(prompt_tokens=230, completion_tokens=60, total_tokens=290))

        if "linear search examines 500 items" in q:
            traj = [
                TrajectoryStep(iteration=1, action="rag_search", arguments={"query": "linear search sequential 500 binary 9", "top_k": 2}, observation="[RAG Search Results]: Linear search O(n), binary search O(log n).", success=True, execution_time_seconds=0.01),
                TrajectoryStep(iteration=2, action="calculator", arguments={"a": 500.0, "b": 9.0, "operation": "subtract"}, observation="[Calculator Result]: 500.0 subtract 9.0 = 491.0", success=True, execution_time_seconds=0.001),
                TrajectoryStep(iteration=3, action="final_answer", arguments={"answer": "Linear search takes 500 operations and binary search takes 9, so linear search takes 491 more operations.", "topic": "Algorithm Comparison"}, observation="Final answer generated.", success=True, execution_time_seconds=0.001),
            ]
            return AgentResult(status="completed", answer="Linear search takes 500 operations and binary search takes 9, so linear search takes 491 more operations.", topic="Algorithm Comparison", trajectory_length=3, trajectory=traj, token_usage=TokenUsage(prompt_tokens=210, completion_tokens=55, total_tokens=265))

        if "array of 250 elements" in q and "50 times" in q:
            traj = [
                TrajectoryStep(iteration=1, action="calculator", arguments={"a": 250.0, "b": 50.0, "operation": "multiply"}, observation="[Calculator Result]: 250.0 multiply 50.0 = 12500.0", success=True, execution_time_seconds=0.001),
                TrajectoryStep(iteration=2, action="final_answer", arguments={"answer": "The total number of operations across 50 runs is 12500.", "topic": "Computation"}, observation="Final answer generated.", success=True, execution_time_seconds=0.001),
            ]
            return AgentResult(status="completed", answer="The total number of operations across 50 runs is 12500.", topic="Computation", trajectory_length=2, trajectory=traj, token_usage=TokenUsage(prompt_tokens=90, completion_tokens=25, total_tokens=115))

        if "4 layers of neural networks" in q or "feature maps" in q:
            traj = [
                TrajectoryStep(iteration=1, action="calculator", arguments={"a": 4.0, "b": 8.0, "operation": "multiply"}, observation="[Calculator Result]: 4.0 multiply 8.0 = 32.0", success=True, execution_time_seconds=0.001),
                TrajectoryStep(iteration=2, action="final_answer", arguments={"answer": "With 4 layers and 8 feature maps per layer, the total is 32 feature maps.", "topic": "Neural Networks"}, observation="Final answer generated.", success=True, execution_time_seconds=0.001),
            ]
            return AgentResult(status="completed", answer="With 4 layers and 8 feature maps per layer, the total is 32 feature maps.", topic="Neural Networks", trajectory_length=2, trajectory=traj, token_usage=TokenUsage(prompt_tokens=95, completion_tokens=28, total_tokens=123))

        if "what is its exact time complexity" in q:
            traj = [
                TrajectoryStep(iteration=1, action="ask_user_clarification", arguments={"question": "Which algorithm or data structure are you asking about?"}, observation="[Clarification Prompt]: Which algorithm or data structure are you asking about?", success=True, execution_time_seconds=0.001),
            ]
            return AgentResult(status="clarification_required", answer="Which algorithm or data structure are you asking about?", topic="Clarification", trajectory_length=1, trajectory=traj, clarification_required=True, token_usage=TokenUsage(prompt_tokens=55, completion_tokens=18, total_tokens=73))

        if "how many layers does it have" in q:
            traj = [
                TrajectoryStep(iteration=1, action="ask_user_clarification", arguments={"question": "Which specific neural network model or architecture are you referring to?"}, observation="[Clarification Prompt]: Which specific neural network model or architecture are you referring to?", success=True, execution_time_seconds=0.001),
            ]
            return AgentResult(status="clarification_required", answer="Which specific neural network model or architecture are you referring to?", topic="Clarification", trajectory_length=1, trajectory=traj, clarification_required=True, token_usage=TokenUsage(prompt_tokens=55, completion_tokens=20, total_tokens=75))

        if "100 minus 37" in q:
            traj = [
                TrajectoryStep(iteration=1, action="calculator", arguments={"a": 100.0, "b": 37.0, "operation": "subtract"}, observation="[Calculator Result]: 100.0 subtract 37.0 = 63.0", success=True, execution_time_seconds=0.001),
                TrajectoryStep(iteration=2, action="final_answer", arguments={"answer": "100 minus 37 is 63.", "topic": "Arithmetic"}, observation="Final answer generated.", success=True, execution_time_seconds=0.001),
            ]
            return AgentResult(status="completed", answer="100 minus 37 is 63.", topic="Arithmetic", trajectory_length=2, trajectory=traj, token_usage=TokenUsage(prompt_tokens=60, completion_tokens=18, total_tokens=78))

        if "primary benefit of rag" in q:
            traj = [
                TrajectoryStep(iteration=1, action="rag_search", arguments={"query": "primary benefit of RAG", "top_k": 2}, observation="[RAG Search Results]: Reduces model hallucinations and keeps responses up to date without retraining.", success=True, execution_time_seconds=0.01),
                TrajectoryStep(iteration=2, action="final_answer", arguments={"answer": "The primary benefit of RAG is reducing hallucinations and keeping knowledge up to date by retrieving facts from external knowledge bases.", "topic": "RAG"}, observation="Final answer generated.", success=True, execution_time_seconds=0.001),
            ]
            return AgentResult(status="completed", answer="The primary benefit of RAG is reducing hallucinations and keeping knowledge up to date by retrieving facts from external knowledge bases.", topic="RAG", trajectory_length=2, trajectory=traj, token_usage=TokenUsage(prompt_tokens=100, completion_tokens=30, total_tokens=130))

        # Default fallback
        return AgentResult(
            status="completed",
            answer="Generic response.",
            topic="General",
            trajectory_length=1,
            trajectory=[TrajectoryStep(iteration=1, action="final_answer", arguments={}, observation="Done", success=True)],
            token_usage=TokenUsage(prompt_tokens=50, completion_tokens=15, total_tokens=65),
        )

    return runner


def build_agent_runner_for_config(
    config_version: AgentConfigVersion,
    use_simulation: bool = True,
) -> Callable[[str], AgentResult]:
    """Create an agent runner callable applying the configuration version settings."""
    if use_simulation:
        base_runner = build_simulated_runner_for_version(config_version)
    else:
        # Live agent runner
        def base_runner(question: str) -> AgentResult:
            return run_agent_workflow(
                question=question,
                max_iterations=config_version.max_iterations,
                max_execution_time_seconds=config_version.agent_timeout_seconds,
                provider=config_version.llm_provider,
                temperature=config_version.temperature,
                prompt_version=config_version.prompt_version,
            )

    def traced_runner(question: str) -> AgentResult:
        res = base_runner(question)
        if getattr(res, "trace", None) is None:
            res.trace = build_trace_from_agent_result(
                agent_result=res,
                query=question,
                config_version=config_version.version_id,
                prompt_version=config_version.prompt_version,
                model=config_version.model,
                temperature=config_version.temperature,
            )
        return res

    return traced_runner


def run_version_evaluation(
    version_name: str,
    reference_summary: Optional[BenchmarkSummary] = None,
    tracking_uri: Optional[str] = None,
    experiment_name: str = DEFAULT_EXPERIMENT_NAME,
    use_simulation: bool = True,
    output_dir: str = "tests/evaluation/results",
    reports_dir: str = "reports/evidently",
    thresholds: Optional[RegressionThresholds] = None,
) -> Tuple[BenchmarkSummary, Dict[str, Any], RegressionEvaluationResult]:
    """Execute the benchmark for a specific version and log the run to MLflow."""
    config_version = get_config_version(version_name)
    runner = build_agent_runner_for_config(config_version, use_simulation=use_simulation)

    report_filename = f"benchmark_results_{config_version.version_id}.md"
    report_path = os.path.join(output_dir, report_filename)

    print(f"\n================================================================================")
    print(f"MLOps Evaluation Run: {config_version.version_id} ({config_version.prompt_version})")
    print(f"Description: {config_version.description}")
    print(f"Temp: {config_version.temperature} | MaxIter: {config_version.max_iterations} | ChunkSize: {config_version.chunk_size} | TopK: {config_version.rag_top_k}")
    print(f"================================================================================\n")

    summary = run_benchmark_evaluation(
        cases=BENCHMARK_CASES,
        agent_runner_fn=runner,
        output_report_path=report_path,
    )

    # Establish or resolve reference baseline summary
    ref_summary = reference_summary
    if ref_summary is None:
        if config_version.version_id == "v1_baseline":
            ref_summary = summary
        else:
            ref_cfg = get_config_version("v1_baseline")
            ref_runner = build_agent_runner_for_config(ref_cfg, use_simulation=use_simulation)
            ref_summary = run_benchmark_evaluation(
                cases=BENCHMARK_CASES,
                agent_runner_fn=ref_runner,
                output_report_path=None,
            )

    # Run Evidently regression monitoring and HTML report generation
    reg_result, html_path = run_evidently_regression_monitoring(
        current_summary=summary,
        reference_summary=ref_summary,
        current_version=config_version.version_id,
        reference_version="v1_baseline",
        reports_dir=reports_dir,
        thresholds=thresholds,
    )

    mlflow_info = log_evaluation_run_to_mlflow(
        config_version=config_version,
        summary=summary,
        traces=getattr(summary, "traces", None),
        regression_result=reg_result,
        evidently_html_path=html_path,
        experiment_name=experiment_name,
        tracking_uri=tracking_uri,
    )

    reg_status_str = "REGRESSION DETECTED" if reg_result.regression_detected else "PASSED (No Regression)"
    print(f"\n>>> Evidently Regression Status: {reg_status_str} | Pass Rate: {reg_result.pct_tests_passed:.1f}%")
    print(f">>> Evidently HTML Report: {html_path}")
    print(f">>> MLflow Run Logged: RunID = {mlflow_info['run_id']} | Experiment = {experiment_name}")
    print(f">>> Artifacts stored at: {mlflow_info['artifact_uri']}\n")

    return summary, mlflow_info, reg_result


def run_all_versions_evaluation(
    tracking_uri: Optional[str] = None,
    experiment_name: str = DEFAULT_EXPERIMENT_NAME,
    use_simulation: bool = True,
    output_dir: str = "tests/evaluation/results",
    reports_dir: str = "reports/evidently",
    thresholds: Optional[RegressionThresholds] = None,
) -> Dict[str, Tuple[BenchmarkSummary, Dict[str, Any], RegressionEvaluationResult]]:
    """Execute evaluation for all 3 configured versions and generate comparison summary."""
    versions = list_config_versions()
    results: Dict[str, Tuple[BenchmarkSummary, Dict[str, Any], RegressionEvaluationResult]] = {}

    print("\n" + "=" * 80)
    print(f"Starting Week 17 Track B MLOps Evaluation across {len(versions)} Configuration Versions")
    print("=" * 80)

    # First evaluate reference baseline (v1_baseline)
    baseline_v = get_config_version("v1_baseline")
    ref_summary, ref_info, ref_reg = run_version_evaluation(
        version_name="v1_baseline",
        reference_summary=None,
        tracking_uri=tracking_uri,
        experiment_name=experiment_name,
        use_simulation=use_simulation,
        output_dir=output_dir,
        reports_dir=reports_dir,
        thresholds=thresholds,
    )
    results["v1_baseline"] = (ref_summary, ref_info, ref_reg)

    # Evaluate remaining versions against v1_baseline reference
    for v in versions:
        if v.version_id == "v1_baseline":
            continue
        summary, mlflow_info, reg_result = run_version_evaluation(
            version_name=v.version_id,
            reference_summary=ref_summary,
            tracking_uri=tracking_uri,
            experiment_name=experiment_name,
            use_simulation=use_simulation,
            output_dir=output_dir,
            reports_dir=reports_dir,
            thresholds=thresholds,
        )
        results[v.version_id] = (summary, mlflow_info, reg_result)

    # Print Comparison Table
    print("\n" + "=" * 105)
    print("MLOps Evaluation & Evidently AI Regression Cross-Configuration Comparison Summary")
    print("=" * 105)
    header = f"{'Version':<22} | {'Prompt Ver':<25} | {'Temp':<5} | {'TCR':<7} | {'TCC':<7} | {'Avg Steps':<10} | {'Avg Tok':<8} | {'Pass %':<7} | {'Regression':<12} | {'Run ID':<32}"
    print(header)
    print("-" * len(header))

    for vid, (s, info, reg) in results.items():
        cfg = get_config_version(vid)
        run_id = info["run_id"]
        reg_flag = "❌ DETECTED" if reg.regression_detected else "✅ NONE"
        row = (
            f"{vid:<22} | "
            f"{cfg.prompt_version:<25} | "
            f"{cfg.temperature:<5.1f} | "
            f"{s.task_completion_rate:>5.1f}% | "
            f"{s.tool_call_correctness:>5.1f}% | "
            f"{s.avg_trajectory_length:>8.2f}s | "
            f"{s.avg_tokens_per_query:>6.1f}t | "
            f"{reg.pct_tests_passed:>5.1f}% | "
            f"{reg_flag:<12} | "
            f"{run_id:<32}"
        )
        print(row)
    print("=" * 105 + "\n")

    return results


def main() -> None:
    """CLI entrypoint for running versioned evaluations."""
    parser = argparse.ArgumentParser(description="Week 17 Track B Agentic AI MLOps Evaluation Runner")
    parser.add_argument(
        "--version",
        "-v",
        type=str,
        default="all",
        help="Configuration version to evaluate ('v1_baseline', 'v2_compact_precision', 'v3_deep_validation', or 'all').",
    )
    parser.add_argument(
        "--tracking-uri",
        "-u",
        type=str,
        default=DEFAULT_TRACKING_URI,
        help=f"MLflow tracking URI (default: '{DEFAULT_TRACKING_URI}').",
    )
    parser.add_argument(
        "--experiment-name",
        "-e",
        type=str,
        default=DEFAULT_EXPERIMENT_NAME,
        help=f"MLflow experiment name (default: '{DEFAULT_EXPERIMENT_NAME}').",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Use live LLM API calls instead of simulated deterministic evaluation.",
    )

    args = parser.parse_args()

    if args.version.lower().strip() == "all":
        run_all_versions_evaluation(
            tracking_uri=args.tracking_uri,
            experiment_name=args.experiment_name,
            use_simulation=not args.live,
        )
    else:
        run_version_evaluation(
            version_name=args.version,
            tracking_uri=args.tracking_uri,
            experiment_name=args.experiment_name,
            use_simulation=not args.live,
        )


if __name__ == "__main__":
    main()
