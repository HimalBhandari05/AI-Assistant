"""Unit and Integration Tests for Evidently AI Regression Monitoring and MLflow Reporting.

Tests dataset conversion, custom agentic metrics, regression threshold checks,
Evidently HTML report generation, failed-case analysis, and MLflow artifact persistence.
"""

import json
import os
import tempfile
import pandas as pd
import pytest

from app.agent.context import TokenUsage, TrajectoryStep
from app.agent.engine import AgentResult
from app.mlops.config_versions import get_config_version
from app.mlops.eval_runner import (
    build_agent_runner_for_config,
    run_all_versions_evaluation,
    run_version_evaluation,
)
from app.mlops.evidently_monitoring import (
    RegressionEvaluationResult,
    RegressionThresholds,
    calculate_custom_metrics,
    convert_benchmark_results_to_dataframe,
    evaluate_regression,
    generate_evidently_html_report,
    generate_markdown_regression_summary,
    run_evidently_regression_monitoring,
)
from app.mlops.mlflow_tracking import log_evaluation_run_to_mlflow
from tests.evaluation.benchmark_cases import BENCHMARK_CASES
from tests.evaluation.evaluator import (
    BenchmarkSummary,
    CaseEvaluationResult,
    run_benchmark_evaluation,
)


def _build_dummy_summary(
    success_rate: float = 100.0,
    avg_tokens: float = 150.0,
    avg_traj: float = 2.0,
    inject_hard_failure: bool = False,
) -> BenchmarkSummary:
    """Helper to construct deterministic BenchmarkSummary objects for testing."""
    results = []
    num_cases = 16
    fail_index = 0 if inject_hard_failure else -1

    for i in range(num_cases):
        is_pass = (i != fail_index)
        results.append(
            CaseEvaluationResult(
                case_id=f"case_{i+1:02d}",
                category="test_category",
                question=f"Test Question {i+1}",
                success=is_pass,
                status="completed" if is_pass else "error",
                trajectory_length=int(avg_traj),
                tool_calls_count=1,
                correct_tool_calls_count=1 if is_pass else 0,
                prompt_tokens=int(avg_tokens * 0.75),
                completion_tokens=int(avg_tokens * 0.25),
                total_tokens=int(avg_tokens),
                failure_classification="Hard Failure" if not is_pass else None,
                failure_details="Injected test failure" if not is_pass else None,
                answer="Answer",
                execution_time_seconds=0.01,
            )
        )

    passed_count = sum(1 for r in results if r.success)
    tcr = (passed_count / num_cases) * 100.0

    return BenchmarkSummary(
        total_tasks=num_cases,
        successful_tasks=passed_count,
        task_completion_rate=tcr,
        total_tool_calls=num_cases,
        correct_tool_calls=passed_count,
        tool_call_correctness=tcr,
        avg_trajectory_length=avg_traj,
        min_trajectory_length=int(avg_traj),
        max_trajectory_length=int(avg_traj),
        avg_tokens_per_query=avg_tokens,
        total_tokens=int(avg_tokens * num_cases),
        hard_failures=1 if inject_hard_failure else 0,
        soft_failures=0,
        cascading_soft_failures=0,
        results=results,
    )


def test_convert_benchmark_results_to_dataframe():
    """Verify conversion of BenchmarkSummary to standardized pandas DataFrame."""
    summary = _build_dummy_summary(avg_tokens=120.0, avg_traj=2.0)
    df = convert_benchmark_results_to_dataframe(summary)

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 16
    expected_cols = [
        "case_id",
        "category",
        "task_success",
        "status",
        "trajectory_length",
        "tool_calls_count",
        "correct_tool_calls_count",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "execution_time_seconds",
        "hard_failure",
        "soft_failure",
    ]
    for col in expected_cols:
        assert col in df.columns
    assert df["task_success"].sum() == 16


def test_custom_metrics_calculation():
    """Verify custom metrics (pct_tests_passed, trajectory_efficiency_score, token_cost_ratio)."""
    ref = _build_dummy_summary(avg_tokens=150.0, avg_traj=2.0)
    curr = _build_dummy_summary(avg_tokens=100.0, avg_traj=1.8)

    metrics = calculate_custom_metrics(current_summary=curr, reference_summary=ref)

    assert metrics["pct_tests_passed"] == 100.0
    assert metrics["trajectory_efficiency_score"] == round(100.0 / 1.8, 2)
    assert metrics["token_cost_ratio"] == round(100.0 / 150.0, 3)


def test_evaluate_regression_clean_pass():
    """Verify regression evaluation on matching / improved configurations returns no regression."""
    ref = _build_dummy_summary(avg_tokens=150.0, avg_traj=2.0)
    curr = _build_dummy_summary(avg_tokens=105.0, avg_traj=1.9)

    res = evaluate_regression(
        current_summary=curr,
        reference_summary=ref,
        current_version="v2_compact_precision",
        reference_version="v1_baseline",
    )

    assert isinstance(res, RegressionEvaluationResult)
    assert res.regression_detected is False
    assert len(res.regression_reasons) == 0
    assert len(res.failed_cases) == 0
    assert res.pct_tests_passed == 100.0


def test_evaluate_regression_detected_on_failure():
    """Verify regression detection when benchmark test cases fail."""
    ref = _build_dummy_summary(avg_tokens=150.0, avg_traj=2.0)
    curr_with_fail = _build_dummy_summary(avg_tokens=150.0, avg_traj=2.0, inject_hard_failure=True)

    res = evaluate_regression(
        current_summary=curr_with_fail,
        reference_summary=ref,
        current_version="v_broken",
        reference_version="v1_baseline",
    )

    assert res.regression_detected is True
    assert res.pct_tests_passed == (15 / 16) * 100.0
    assert len(res.failed_cases) == 1
    assert res.failed_cases[0]["case_id"] == "case_01"
    assert any("Test Pass Percentage" in r for r in res.regression_reasons)


def test_evaluate_regression_detected_on_token_growth():
    """Verify regression detection when token usage exceeds growth multiplier."""
    ref = _build_dummy_summary(avg_tokens=100.0, avg_traj=2.0)
    curr_inflated = _build_dummy_summary(avg_tokens=250.0, avg_traj=2.0)  # 2.5x token inflation

    res = evaluate_regression(
        current_summary=curr_inflated,
        reference_summary=ref,
        current_version="v_inflated",
        reference_version="v1_baseline",
    )

    assert res.regression_detected is True
    assert res.token_cost_ratio == 2.5
    assert any("Average tokens per query" in r for r in res.regression_reasons)


def test_generate_evidently_html_report():
    """Verify Evidently TestSuite runs and generates valid HTML report file."""
    ref_summary = _build_dummy_summary(avg_tokens=150.0)
    curr_summary = _build_dummy_summary(avg_tokens=110.0)

    ref_df = convert_benchmark_results_to_dataframe(ref_summary)
    curr_df = convert_benchmark_results_to_dataframe(curr_summary)

    with tempfile.TemporaryDirectory() as tmp_dir:
        html_file = os.path.join(tmp_dir, "test_regression_report.html")
        out_path = generate_evidently_html_report(
            reference_df=ref_df,
            current_df=curr_df,
            output_html_path=html_file,
            current_version="v2_compact_precision",
            reference_version="v1_baseline",
        )

        assert os.path.exists(out_path)
        assert os.path.getsize(out_path) > 1000
        with open(out_path, "r", encoding="utf-8") as f:
            content = f.read()
            assert "<html" in content.lower()


def test_markdown_regression_summary_generation():
    """Verify markdown regression summary contains overview table and thresholds."""
    ref = _build_dummy_summary(avg_tokens=150.0)
    curr = _build_dummy_summary(avg_tokens=105.0)

    res = evaluate_regression(curr, ref, "v2_compact_precision", "v1_baseline")
    md = generate_markdown_regression_summary(res)

    assert "# Evidently AI Regression Analysis Report" in md
    assert "NO REGRESSION DETECTED" in md
    assert "| **Benchmark Pass Rate** |" in md
    assert "| **Trajectory Efficiency Score (TES)** |" in md


def test_evidently_mlflow_artifact_logging():
    """Verify that Evidently HTML report and regression summaries are logged into MLflow artifacts."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        sqlite_uri = f"sqlite:///{os.path.join(tmp_dir, 'evidently_mlflow.db')}"
        exp_name = "test-phase4-evidently-experiment"
        cfg = get_config_version("v2_compact_precision")

        ref_summary = _build_dummy_summary(avg_tokens=150.0)
        curr_summary = _build_dummy_summary(avg_tokens=105.0)

        html_report_path = os.path.join(tmp_dir, "v2_vs_v1.html")
        generate_evidently_html_report(
            reference_df=convert_benchmark_results_to_dataframe(ref_summary),
            current_df=convert_benchmark_results_to_dataframe(curr_summary),
            output_html_path=html_report_path,
            current_version=cfg.version_id,
        )

        reg_result = evaluate_regression(curr_summary, ref_summary, cfg.version_id, "v1_baseline")

        log_info = log_evaluation_run_to_mlflow(
            config_version=cfg,
            summary=curr_summary,
            regression_result=reg_result,
            evidently_html_path=html_report_path,
            experiment_name=exp_name,
            tracking_uri=sqlite_uri,
        )

        assert "run_id" in log_info
        assert log_info["regression_detected"] is False

        # Verify artifacts in MLflow
        import mlflow
        client = mlflow.tracking.MlflowClient(tracking_uri=sqlite_uri)
        artifacts = client.list_artifacts(log_info["run_id"], path="evaluation/evidently")
        artifact_paths = [a.path for a in artifacts]

        assert "evaluation/evidently/v2_vs_v1.html" in artifact_paths
        assert "evaluation/evidently/regression_summary.json" in artifact_paths
        assert "evaluation/evidently/regression_summary.md" in artifact_paths


def test_full_version_evaluation_with_evidently():
    """Verify run_version_evaluation executes benchmark, runs Evidently, and generates reports."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        sqlite_uri = f"sqlite:///{os.path.join(tmp_dir, 'mlflow_test.db')}"
        exp_name = "test-eval-runner-evidently"
        rep_dir = os.path.join(tmp_dir, "reports")

        summary, mlflow_info, reg_result = run_version_evaluation(
            version_name="v2_compact_precision",
            reference_summary=None,
            tracking_uri=sqlite_uri,
            experiment_name=exp_name,
            use_simulation=True,
            output_dir=os.path.join(tmp_dir, "eval_results"),
            reports_dir=rep_dir,
        )

        assert summary.task_completion_rate == 100.0
        assert reg_result.pct_tests_passed == 100.0
        assert reg_result.regression_detected is False
        assert os.path.exists(os.path.join(rep_dir, "v2_compact_precision_vs_v1_baseline.html"))
        assert os.path.exists(os.path.join(rep_dir, "v2_compact_precision_regression_summary.json"))
        assert os.path.exists(os.path.join(rep_dir, "v2_compact_precision_regression_summary.md"))
