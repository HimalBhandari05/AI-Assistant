"""Unit and Integration Tests for Agent Trace Logging and MLflow Trace Artifacts.

Tests trace schema validation, sensitive credential redaction, per-iteration
telemetry extraction, multi-step trace construction, recovery event logging,
and MLflow artifact persistence across configuration versions.
"""

import json
import os
import tempfile
import pytest

from app.agent.context import TokenUsage, TrajectoryStep
from app.agent.engine import AgentResult
from app.agent.metrics import AgentExecutionMetrics
from app.mlops.config_versions import get_config_version
from app.mlops.eval_runner import build_agent_runner_for_config
from app.mlops.mlflow_tracking import (
    log_evaluation_run_to_mlflow,
    setup_mlflow_experiment,
)
from app.mlops.trace_schema import (
    AgentExecutionTrace,
    AgentIterationTrace,
    build_markdown_trace_summary,
    build_trace_from_agent_result,
    is_sensitive_key,
    sanitize_trace_payload,
)
from tests.evaluation.benchmark_cases import BENCHMARK_CASES
from tests.evaluation.evaluator import (
    BenchmarkSummary,
    CaseEvaluationResult,
    evaluate_single_case,
    run_benchmark_evaluation,
)


def test_trace_schema_models():
    """Verify AgentIterationTrace and AgentExecutionTrace models instantiate and validate correctly."""
    step = AgentIterationTrace(
        iteration=1,
        decision_type="tool_call",
        tool_name="calculator",
        tool_arguments={"a": 10.0, "b": 5.0, "operation": "add"},
        observation="[Calculator Result]: 10.0 add 5.0 = 15.0",
        success=True,
        execution_time_seconds=0.005,
    )
    assert step.iteration == 1
    assert step.tool_name == "calculator"
    assert step.success is True

    trace = AgentExecutionTrace(
        case_id="test_case_01",
        query="What is 10 + 5?",
        config_version="v1_baseline",
        prompt_version="v1_baseline",
        model="gemini-2.5-flash",
        temperature=0.1,
        status="completed",
        termination_reason="Final answer synthesized.",
        final_answer="10 + 5 = 15",
        total_iterations=1,
        total_execution_time_seconds=0.05,
        token_usage={"prompt_tokens": 50, "completion_tokens": 15, "total_tokens": 65},
        iterations=[step],
    )
    assert trace.case_id == "test_case_01"
    assert len(trace.iterations) == 1
    assert trace.token_usage["total_tokens"] == 65
    
    # Test JSON serialization roundtrip
    dumped = trace.model_dump()
    assert dumped["case_id"] == "test_case_01"
    assert dumped["iterations"][0]["tool_name"] == "calculator"


def test_sensitive_credential_redaction():
    """Verify that credentials, tokens, passwords, and API keys are redacted from trace payloads."""
    assert is_sensitive_key("api_key") is True
    assert is_sensitive_key("gemini_api_key") is True
    assert is_sensitive_key("Authorization") is True
    assert is_sensitive_key("token") is True
    assert is_sensitive_key("password") is True
    assert is_sensitive_key("user_query") is False
    assert is_sensitive_key("tool_name") is False

    raw_payload = {
        "query": "search query",
        "api_key": "AIzaSySecretKey12345678901234567890",
        "config": {
            "bearer_token": "Bearer eyJhbGciOi...",
            "password": "supersecretpassword",
            "safe_param": 42,
        },
        "nested_list": [
            {"secret_auth": "xyz123", "public_id": "item_1"},
            "Normal text string",
        ],
    }

    sanitized = sanitize_trace_payload(raw_payload)

    assert sanitized["api_key"] == "[REDACTED]"
    assert sanitized["config"]["bearer_token"] == "[REDACTED]"
    assert sanitized["config"]["password"] == "[REDACTED]"
    assert sanitized["config"]["safe_param"] == 42
    assert sanitized["nested_list"][0]["secret_auth"] == "[REDACTED]"
    assert sanitized["nested_list"][0]["public_id"] == "item_1"
    assert sanitized["query"] == "search query"


def test_build_trace_from_agent_result_single_tool():
    """Verify trace building for a single calculator tool execution."""
    traj = [
        TrajectoryStep(
            iteration=1,
            action="calculator",
            arguments={"a": 45.0, "b": 12.0, "operation": "multiply"},
            observation="[Calculator Result]: 45.0 multiply 12.0 = 540.0",
            success=True,
            execution_time_seconds=0.002,
        ),
        TrajectoryStep(
            iteration=2,
            action="final_answer",
            arguments={"answer": "540", "topic": "Arithmetic"},
            observation="Final answer generated.",
            success=True,
            execution_time_seconds=0.001,
        ),
    ]
    res = AgentResult(
        status="completed",
        answer="45 multiplied by 12 is 540.",
        topic="Arithmetic",
        trajectory_length=2,
        trajectory=traj,
        token_usage=TokenUsage(prompt_tokens=50, completion_tokens=20, total_tokens=70),
    )

    trace = build_trace_from_agent_result(
        agent_result=res,
        query="What is 45 multiplied by 12?",
        config_version="v2_compact_precision",
        prompt_version="v2_compact",
        model="gemini-2.5-flash",
        temperature=0.0,
        case_id="case_calc_01",
    )

    assert trace.case_id == "case_calc_01"
    assert trace.config_version == "v2_compact_precision"
    assert trace.status == "completed"
    assert trace.total_iterations == 2
    assert trace.token_usage["total_tokens"] == 70
    assert len(trace.iterations) == 2
    assert trace.iterations[0].decision_type == "tool_call"
    assert trace.iterations[0].tool_name == "calculator"
    assert trace.iterations[1].decision_type == "final_answer"
    assert "successfully synthesized" in trace.termination_reason


def test_build_trace_from_agent_result_multi_step_chain():
    """Verify trace building for multi-step RAG + calculator chaining."""
    traj = [
        TrajectoryStep(
            iteration=1,
            action="rag_search",
            arguments={"query": "linear search vs binary search count", "top_k": 5},
            observation="[RAG Search Results]: Linear search 500 items, binary search 9 comparisons.",
            success=True,
            execution_time_seconds=0.012,
        ),
        TrajectoryStep(
            iteration=2,
            action="calculator",
            arguments={"a": 500.0, "b": 9.0, "operation": "subtract"},
            observation="[Calculator Result]: 500.0 subtract 9.0 = 491.0",
            success=True,
            execution_time_seconds=0.002,
        ),
        TrajectoryStep(
            iteration=3,
            action="final_answer",
            arguments={"answer": "491", "topic": "Algorithms"},
            observation="Final answer generated.",
            success=True,
            execution_time_seconds=0.001,
        ),
    ]
    res = AgentResult(
        status="completed",
        answer="The difference in search operations is 491.",
        topic="Algorithms",
        trajectory_length=3,
        trajectory=traj,
        token_usage=TokenUsage(prompt_tokens=180, completion_tokens=45, total_tokens=225),
    )

    trace = build_trace_from_agent_result(
        agent_result=res,
        query="How many more operations does linear search take?",
        config_version="v1_baseline",
        case_id="case_chain_01",
    )

    assert trace.total_iterations == 3
    assert [step.tool_name for step in trace.iterations] == ["rag_search", "calculator", None]
    assert trace.iterations[0].tool_arguments["top_k"] == 5
    assert trace.iterations[1].tool_arguments["operation"] == "subtract"


def test_build_trace_from_agent_result_clarification():
    """Verify trace building for ambiguous queries requiring clarification."""
    traj = [
        TrajectoryStep(
            iteration=1,
            action="ask_user_clarification",
            arguments={"question": "Which algorithm are you referring to?"},
            observation="[Clarification Prompt]: Which algorithm are you referring to?",
            success=True,
            execution_time_seconds=0.001,
        )
    ]
    res = AgentResult(
        status="clarification_required",
        answer="Which algorithm are you referring to?",
        topic="Clarification",
        trajectory_length=1,
        trajectory=traj,
        clarification_required=True,
        token_usage=TokenUsage(prompt_tokens=40, completion_tokens=15, total_tokens=55),
    )

    trace = build_trace_from_agent_result(
        agent_result=res,
        query="What is its exact time complexity?",
        config_version="v1_baseline",
        case_id="case_clarify_01",
    )

    assert trace.status == "clarification_required"
    assert trace.total_iterations == 1
    assert "Ambiguous user query detected" in trace.termination_reason
    assert trace.iterations[0].decision_type == "ask_user_clarification"


def test_build_trace_from_agent_result_error_and_recovery():
    """Verify error and recovery event logging in execution traces."""
    traj = [
        TrajectoryStep(
            iteration=1,
            action="calculator",
            arguments={"a": 10.0, "b": 0.0, "operation": "divide"},
            observation="[Tool Error (calculator)]: InvalidOperation - Division by zero is undefined.",
            success=False,
            execution_time_seconds=0.002,
        ),
        TrajectoryStep(
            iteration=2,
            action="calculator",
            arguments={"a": 10.0, "b": 2.0, "operation": "divide"},
            observation="[Calculator Result]: 10.0 divide 2.0 = 5.0",
            success=True,
            execution_time_seconds=0.002,
        ),
        TrajectoryStep(
            iteration=3,
            action="final_answer",
            arguments={"answer": "5.0", "topic": "Arithmetic"},
            observation="Final answer generated.",
            success=True,
            execution_time_seconds=0.001,
        ),
    ]
    metrics = AgentExecutionMetrics(question="Divide 10 by zero then recover with 2")
    metrics.record_tool_call("calculator", arguments_valid=True, success=False, execution_time_seconds=0.002, error_type="ZeroDivisionError", error_message="Division by zero")
    metrics.record_tool_call("calculator", arguments_valid=True, success=True, execution_time_seconds=0.002)

    res = AgentResult(
        status="completed",
        answer="The recovered result is 5.0.",
        topic="Arithmetic",
        trajectory_length=3,
        trajectory=traj,
        token_usage=TokenUsage(prompt_tokens=120, completion_tokens=30, total_tokens=150),
        metrics=metrics,
    )

    trace = build_trace_from_agent_result(
        agent_result=res,
        query="Divide 10 by zero then 2",
        config_version="v3_deep_validation",
        case_id="case_recovery_01",
    )

    assert len(trace.errors_and_recoveries) >= 1
    assert trace.errors_and_recoveries[0]["recovered"] is True
    assert trace.iterations[0].error_type == "ToolExecutionError"
    assert trace.iterations[0].success is False


def test_build_markdown_trace_summary():
    """Verify markdown trace report generation produces well-formatted output."""
    traj = [
        TrajectoryStep(iteration=1, action="calculator", arguments={"a": 2.0, "b": 2.0, "operation": "add"}, observation="4.0", success=True, execution_time_seconds=0.001),
        TrajectoryStep(iteration=2, action="final_answer", arguments={"answer": "4"}, observation="Final answer", success=True, execution_time_seconds=0.001),
    ]
    res = AgentResult(status="completed", answer="4", topic="Math", trajectory_length=2, trajectory=traj, token_usage=TokenUsage(prompt_tokens=20, completion_tokens=10, total_tokens=30))
    trace = build_trace_from_agent_result(res, "2+2", case_id="case_calc_01")

    md = build_markdown_trace_summary([trace])
    assert "# Agent Execution Trace Summary" in md
    assert "| `case_calc_01` | `completed` | 2 |" in md
    assert "### Trace 1: case_calc_01" in md
    assert "**Iter 1 [Tool Call]**" in md


def test_mlflow_trace_artifact_logging():
    """Verify that MLflow logs individual and aggregated trace artifacts under evaluation/traces/."""
    with tempfile.TemporaryDirectory() as tmp_tracking_dir:
        sqlite_uri = f"sqlite:///{os.path.join(tmp_tracking_dir, 'test_mlflow.db')}"
        exp_name = "test-phase3-tracing-experiment"
        cfg = get_config_version("v2_compact_precision")

        # Create dummy summary and traces
        traj = [
            TrajectoryStep(iteration=1, action="calculator", arguments={"a": 10.0, "b": 2.0, "operation": "multiply"}, observation="20.0", success=True, execution_time_seconds=0.002),
            TrajectoryStep(iteration=2, action="final_answer", arguments={"answer": "20"}, observation="Done", success=True, execution_time_seconds=0.001),
        ]
        res = AgentResult(status="completed", answer="20", topic="Math", trajectory_length=2, trajectory=traj, token_usage=TokenUsage(prompt_tokens=40, completion_tokens=10, total_tokens=50))
        t1 = build_trace_from_agent_result(res, "10 * 2", config_version=cfg.version_id, case_id="case_calc_01")

        case_res = CaseEvaluationResult(
            case_id="case_calc_01",
            category="Arithmetic",
            question="10 * 2",
            success=True,
            status="completed",
            trajectory_length=2,
            tool_calls_count=1,
            correct_tool_calls_count=1,
            prompt_tokens=40,
            completion_tokens=10,
            total_tokens=50,
            answer="20",
            execution_time_seconds=0.003,
        )
        summary = BenchmarkSummary(
            total_tasks=1,
            successful_tasks=1,
            task_completion_rate=100.0,
            total_tool_calls=1,
            correct_tool_calls=1,
            tool_call_correctness=100.0,
            avg_trajectory_length=2.0,
            min_trajectory_length=2,
            max_trajectory_length=2,
            avg_tokens_per_query=50.0,
            total_tokens=50,
            hard_failures=0,
            soft_failures=0,
            cascading_soft_failures=0,
            results=[case_res],
            traces=[t1],
        )

        log_info = log_evaluation_run_to_mlflow(
            config_version=cfg,
            summary=summary,
            traces=[t1],
            experiment_name=exp_name,
            tracking_uri=sqlite_uri,
        )

        assert "run_id" in log_info
        assert log_info["trace_count"] == 1

        # Verify artifacts exist in the MLflow run artifact store
        import mlflow
        client = mlflow.tracking.MlflowClient(tracking_uri=sqlite_uri)
        artifacts = client.list_artifacts(log_info["run_id"], path="evaluation")
        artifact_names = [a.path for a in artifacts]
        assert "evaluation/summary.json" in artifact_names
        assert "evaluation/traces" in artifact_names

        trace_artifacts = client.list_artifacts(log_info["run_id"], path="evaluation/traces")
        trace_file_names = [a.path for a in trace_artifacts]
        assert "evaluation/traces/case_calc_01.json" in trace_file_names
        assert "evaluation/traces/all_traces.jsonl" in trace_file_names
        assert "evaluation/traces/representative_traces.json" in trace_file_names
        assert "evaluation/traces/traces_summary.md" in trace_file_names


def test_eval_runner_produces_traces_for_all_versions():
    """Verify that simulated eval runners produce 16 complete traces for each configuration version."""
    for vid in ["v1_baseline", "v2_compact_precision", "v3_deep_validation"]:
        cfg = get_config_version(vid)
        runner = build_agent_runner_for_config(cfg, use_simulation=True)

        summary = run_benchmark_evaluation(
            cases=BENCHMARK_CASES,
            agent_runner_fn=runner,
            output_report_path=None,
        )

        assert len(summary.traces) == 16
        for trace in summary.traces:
            assert isinstance(trace, AgentExecutionTrace)
            assert trace.config_version == vid
            assert trace.total_iterations >= 1
            assert len(trace.iterations) == trace.total_iterations
            assert trace.status in ("completed", "clarification_required")
            assert trace.termination_reason != ""
