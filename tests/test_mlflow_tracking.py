"""Unit and integration tests for MLflow experiment tracking, configuration versioning, and evaluation runs."""

import os
import sys
import pytest
import mlflow

# Ensure project root in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.agent.context import TokenUsage, TrajectoryStep
from app.agent.engine import AgentResult
from app.mlops.config_versions import (
    AgentConfigVersion,
    CONFIG_VERSIONS,
    get_config_version,
    list_config_versions,
    VERSION_1_BASELINE,
    VERSION_2_COMPACT_PRECISION,
    VERSION_3_DEEP_VALIDATION,
)
from app.mlops.mlflow_tracking import (
    setup_mlflow_experiment,
    log_evaluation_run_to_mlflow,
    build_results_csv,
)
from app.mlops.eval_runner import (
    run_version_evaluation,
    run_all_versions_evaluation,
    build_simulated_runner_for_version,
)
from tests.evaluation.benchmark_cases import BENCHMARK_CASES
from tests.evaluation.evaluator import (
    BenchmarkSummary,
    CaseEvaluationResult,
    compute_benchmark_summary,
)


def test_config_versions_registry():
    """Verify registry contains exactly 3 canonical versions and handles aliases."""
    versions = list_config_versions()
    assert len(versions) == 3
    version_ids = [v.version_id for v in versions]
    assert "v1_baseline" in version_ids
    assert "v2_compact_precision" in version_ids
    assert "v3_deep_validation" in version_ids

    # Alias checks
    assert get_config_version("v1").version_id == "v1_baseline"
    assert get_config_version("prompt_v1").version_id == "v1_baseline"
    assert get_config_version("v2").version_id == "v2_compact_precision"
    assert get_config_version("prompt_v2").version_id == "v2_compact_precision"
    assert get_config_version("v3").version_id == "v3_deep_validation"
    assert get_config_version("prompt_v3").version_id == "v3_deep_validation"

    # Unknown version raises ValueError
    with pytest.raises(ValueError):
        get_config_version("non_existent_version")


def test_config_version_parameters_distinction():
    """Verify that all 3 versions have genuinely distinct hyperparameter configurations."""
    v1 = get_config_version("v1_baseline")
    v2 = get_config_version("v2_compact_precision")
    v3 = get_config_version("v3_deep_validation")

    # Distinct temperatures
    assert v1.temperature != v2.temperature
    assert v2.temperature != v3.temperature

    # Distinct iteration budgets
    assert v1.max_iterations == 5
    assert v2.max_iterations == 4
    assert v3.max_iterations == 6

    # Distinct chunk sizes & top_k
    assert v1.chunk_size == 1000 and v1.rag_top_k == 3
    assert v2.chunk_size == 800 and v2.rag_top_k == 4
    assert v3.chunk_size == 1200 and v3.rag_top_k == 2

    # Verify MLflow parameter export
    params = v2.to_mlflow_params()
    assert params["version_id"] == "v2_compact_precision"
    assert params["temperature"] == 0.0
    assert params["max_iterations"] == 4
    assert params["chunk_size"] == 800
    assert params["top_k"] == 4


def test_setup_mlflow_experiment(tmp_path):
    """Verify local MLflow experiment initialization."""
    db_path = f"sqlite:///{tmp_path}/test_mlflow.db"
    exp_name = "test-agentic-experiment"

    exp_id = setup_mlflow_experiment(experiment_name=exp_name, tracking_uri=db_path)
    assert exp_id is not None

    client = mlflow.tracking.MlflowClient(tracking_uri=db_path)
    exp = client.get_experiment(exp_id)
    assert exp.name == exp_name


def test_log_evaluation_run_to_mlflow(tmp_path):
    """Verify logging parameters, metrics, tags, and artifacts to MLflow."""
    db_path = f"sqlite:///{tmp_path}/test_mlflow.db"
    exp_name = "test-eval-run-exp"

    cfg = get_config_version("v1_baseline")
    case_res = CaseEvaluationResult(
        case_id="case_calc_01",
        category="single_step_calc",
        question="What is 45 * 12?",
        success=True,
        status="completed",
        trajectory_length=2,
        tool_calls_count=1,
        correct_tool_calls_count=1,
        prompt_tokens=65,
        completion_tokens=25,
        total_tokens=90,
        failure_classification=None,
        failure_details=None,
        answer="540",
        execution_time_seconds=0.01,
    )
    summary = compute_benchmark_summary([case_res])

    info = log_evaluation_run_to_mlflow(
        config_version=cfg,
        summary=summary,
        experiment_name=exp_name,
        tracking_uri=db_path,
        run_name="test_v1_run",
    )

    assert info["run_id"] is not None
    assert info["experiment_id"] is not None

    client = mlflow.tracking.MlflowClient(tracking_uri=db_path)
    run = client.get_run(info["run_id"])

    # Verify Parameters
    assert run.data.params["version_id"] == "v1_baseline"
    assert run.data.params["prompt_version"] == "w16_react_v1"
    assert float(run.data.params["temperature"]) == 0.1
    assert int(run.data.params["max_iterations"]) == 5

    # Verify Metrics
    assert run.data.metrics["task_completion_rate"] == 100.0
    assert run.data.metrics["tool_call_correctness"] == 100.0
    assert run.data.metrics["total_tokens"] == 90.0

    # Verify Artifacts
    artifacts = client.list_artifacts(info["run_id"], path="evaluation")
    artifact_filenames = [a.path.split("/")[-1] for a in artifacts]
    assert "summary.json" in artifact_filenames
    assert "results.json" in artifact_filenames
    assert "results.csv" in artifact_filenames
    assert "evaluation_report.md" in artifact_filenames
    assert "config.json" in artifact_filenames
    assert "benchmark_cases.json" in artifact_filenames


def test_eval_runner_version_execution(tmp_path):
    """Verify execution of evaluation runner for a single version with simulated runner."""
    db_path = f"sqlite:///{tmp_path}/test_mlflow.db"
    exp_name = "test-runner-exp"
    out_dir = str(tmp_path / "results")

    summary, info, *rest = run_version_evaluation(
        version_name="v2_compact_precision",
        tracking_uri=db_path,
        experiment_name=exp_name,
        use_simulation=True,
        output_dir=out_dir,
    )

    assert isinstance(summary, BenchmarkSummary)
    assert summary.total_tasks == 16
    assert summary.successful_tasks == 16
    assert summary.task_completion_rate == 100.0
    assert summary.avg_trajectory_length < 2.1  # Compact v2 trajectory
    assert info["run_id"] is not None


def test_eval_runner_cross_version_comparison(tmp_path):
    """Verify comparative evaluation across all 3 versions and metric differences."""
    db_path = f"sqlite:///{tmp_path}/test_mlflow.db"
    exp_name = "test-comparator-exp"
    out_dir = str(tmp_path / "results")

    all_results = run_all_versions_evaluation(
        tracking_uri=db_path,
        experiment_name=exp_name,
        use_simulation=True,
        output_dir=out_dir,
    )

    assert "v1_baseline" in all_results
    assert "v2_compact_precision" in all_results
    assert "v3_deep_validation" in all_results

    s1, *_ = all_results["v1_baseline"]
    s2, *_ = all_results["v2_compact_precision"]
    s3, *_ = all_results["v3_deep_validation"]

    # All achieve 100% completion on curated benchmarks
    assert s1.task_completion_rate == 100.0
    assert s2.task_completion_rate == 100.0
    assert s3.task_completion_rate == 100.0

    # Token footprint ordering: v2 (compact) < v1 (baseline) < v3 (deep validation)
    assert s2.total_tokens < s1.total_tokens < s3.total_tokens

    # Average steps ordering: v2 <= v1 < v3
    assert s2.avg_trajectory_length <= s1.avg_trajectory_length < s3.avg_trajectory_length
