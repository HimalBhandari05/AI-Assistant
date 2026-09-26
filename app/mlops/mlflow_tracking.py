"""MLflow Experiment Tracking and Run Management for Agentic AI Evaluation."""

import csv
import datetime
import json
import logging
import os
import tempfile
from typing import Any, Dict, List, Optional
import shutil
import mlflow
from mlflow.entities import ViewType

from app.mlops.config_versions import AgentConfigVersion
from app.mlops.evidently_monitoring import (
    RegressionEvaluationResult,
    generate_markdown_regression_summary,
)
from app.mlops.trace_schema import (
    AgentExecutionTrace,
    build_markdown_trace_summary,
    sanitize_trace_payload,
)
from tests.evaluation.benchmark_cases import BENCHMARK_CASES
from tests.evaluation.evaluator import BenchmarkSummary, generate_markdown_report

os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

logger = logging.getLogger("ai_assistant.mlops.tracking")

DEFAULT_EXPERIMENT_NAME = "week17-track-b-agentic-mlops"
DEFAULT_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")


def setup_mlflow_experiment(
    experiment_name: str = DEFAULT_EXPERIMENT_NAME,
    tracking_uri: Optional[str] = None,
) -> str:
    """Initialize and retrieve or create the target MLflow experiment.

    Args:
        experiment_name: Name of the experiment to log to.
        tracking_uri: Local directory path or remote tracking URI (defaults to file:./mlruns).

    Returns:
        The experiment_id string.
    """
    uri = tracking_uri or DEFAULT_TRACKING_URI
    mlflow.set_tracking_uri(uri)
    logger.info(f"Configured MLflow tracking URI: {uri}")

    exp = mlflow.get_experiment_by_name(experiment_name)
    if exp is not None:
        if exp.lifecycle_stage == "deleted":
            # If previously deleted, create or use unique name
            exp_id = mlflow.create_experiment(f"{experiment_name}_{int(datetime.datetime.now().timestamp())}")
        else:
            exp_id = exp.experiment_id
    else:
        exp_id = mlflow.create_experiment(experiment_name)
        logger.info(f"Created new MLflow experiment '{experiment_name}' (ID: {exp_id})")

    mlflow.set_experiment(experiment_name)
    return exp_id


def build_results_csv(summary: BenchmarkSummary, output_filepath: str) -> None:
    """Write structured tabular CSV of per-query benchmark results."""
    fieldnames = [
        "case_id",
        "category",
        "question",
        "success",
        "status",
        "trajectory_length",
        "tool_calls_count",
        "correct_tool_calls_count",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "failure_classification",
        "failure_details",
        "execution_time_seconds",
    ]
    with open(output_filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in summary.results:
            writer.writerow({
                "case_id": r.case_id,
                "category": r.category,
                "question": r.question,
                "success": r.success,
                "status": r.status,
                "trajectory_length": r.trajectory_length,
                "tool_calls_count": r.tool_calls_count,
                "correct_tool_calls_count": r.correct_tool_calls_count,
                "prompt_tokens": r.prompt_tokens,
                "completion_tokens": r.completion_tokens,
                "total_tokens": r.total_tokens,
                "failure_classification": r.failure_classification or "None",
                "failure_details": r.failure_details or "",
                "execution_time_seconds": r.execution_time_seconds,
            })


def log_evaluation_run_to_mlflow(
    config_version: AgentConfigVersion,
    summary: BenchmarkSummary,
    traces: Optional[List[AgentExecutionTrace]] = None,
    regression_result: Optional[RegressionEvaluationResult] = None,
    evidently_html_path: Optional[str] = None,
    experiment_name: str = DEFAULT_EXPERIMENT_NAME,
    tracking_uri: Optional[str] = None,
    run_name: Optional[str] = None,
    tags: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Log an evaluation benchmark run to MLflow including parameters, metrics, traces, and Evidently artifacts.

    Args:
        config_version: The AgentConfigVersion used during the evaluation.
        summary: The aggregated BenchmarkSummary produced by the evaluation harness.
        traces: Optional list of AgentExecutionTrace instances. If None, extracts from summary.traces.
        regression_result: Optional RegressionEvaluationResult from Evidently regression monitoring.
        evidently_html_path: Optional path to Evidently HTML report.
        experiment_name: Target experiment name.
        tracking_uri: MLflow tracking URI.
        run_name: Custom run name (defaults to 'eval_{version_id}_{timestamp}').
        tags: Additional custom tags.

    Returns:
        Dictionary containing run_id, experiment_id, artifact_uri, logged params, and logged metrics.
    """
    exp_id = setup_mlflow_experiment(
        experiment_name=experiment_name,
        tracking_uri=tracking_uri,
    )

    timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    name = run_name or f"eval_{config_version.version_id}_{timestamp_str}"

    # Extract flat parameters and metrics
    params = config_version.to_mlflow_params()
    metrics = {
        "task_completion_rate": float(summary.task_completion_rate),
        "tool_call_correctness": float(summary.tool_call_correctness),
        "avg_trajectory_length": float(summary.avg_trajectory_length),
        "min_trajectory_length": float(summary.min_trajectory_length),
        "max_trajectory_length": float(summary.max_trajectory_length),
        "avg_tokens_per_query": float(summary.avg_tokens_per_query),
        "total_tokens": float(summary.total_tokens),
        "total_tasks": float(summary.total_tasks),
        "successful_tasks": float(summary.successful_tasks),
        "total_tool_calls": float(summary.total_tool_calls),
        "correct_tool_calls": float(summary.correct_tool_calls),
        "hard_failures": float(summary.hard_failures),
        "soft_failures": float(summary.soft_failures),
        "cascading_soft_failures": float(summary.cascading_soft_failures),
    }

    # Add Evidently regression metrics if available
    if regression_result is not None:
        metrics.update({
            "pct_tests_passed": float(regression_result.pct_tests_passed),
            "trajectory_efficiency_score": float(regression_result.trajectory_efficiency_score),
            "token_cost_ratio": float(regression_result.token_cost_ratio),
            "regression_detected": 1.0 if regression_result.regression_detected else 0.0,
        })

    run_tags = {
        "version_id": config_version.version_id,
        "prompt_version": config_version.prompt_version,
        "model": config_version.model,
        "temperature": str(config_version.temperature),
        "assignment": "Week 17 Track B - Agentic AI MLOps",
        "phase": "Phase 4 - Evidently Regression Testing & MLOps Tracking",
    }
    if tags:
        run_tags.update(tags)

    # Resolve traces list
    active_traces: List[AgentExecutionTrace] = traces if traces is not None else getattr(summary, "traces", []) or []

    with mlflow.start_run(experiment_id=exp_id, run_name=name) as run:
        run_id = run.info.run_id

        # 1. Log Parameters
        mlflow.log_params(params)

        # 2. Log Metrics
        mlflow.log_metrics(metrics)

        # 3. Log Tags
        mlflow.set_tags(run_tags)

        # 4. Generate and Log Artifacts
        with tempfile.TemporaryDirectory() as tmp_dir:
            # (a) Summary JSON
            summary_path = os.path.join(tmp_dir, "summary.json")
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(summary.model_dump(), f, indent=2)

            # (b) Detailed Results JSON
            results_path = os.path.join(tmp_dir, "results.json")
            with open(results_path, "w", encoding="utf-8") as f:
                json.dump([r.model_dump() for r in summary.results], f, indent=2)

            # (c) Tabular CSV
            csv_path = os.path.join(tmp_dir, "results.csv")
            build_results_csv(summary, csv_path)

            # (d) Comprehensive Markdown Report
            report_path = os.path.join(tmp_dir, "evaluation_report.md")
            report_md = generate_markdown_report(
                summary=summary,
                config={
                    "model": config_version.model,
                    "provider": config_version.llm_provider,
                    "max_iterations": config_version.max_iterations,
                    "timeout": config_version.agent_timeout_seconds,
                    "prompt_version": config_version.prompt_version,
                },
            )
            with open(report_path, "w", encoding="utf-8") as f:
                f.write(report_md)

            # (e) Configuration Snapshot JSON
            config_path = os.path.join(tmp_dir, "config.json")
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(config_version.model_dump(), f, indent=2)

            # (f) Benchmark Cases Definition JSON
            cases_path = os.path.join(tmp_dir, "benchmark_cases.json")
            with open(cases_path, "w", encoding="utf-8") as f:
                json.dump([c.model_dump() for c in BENCHMARK_CASES], f, indent=2)

            # (g) Traces Artifacts Subdirectory
            traces_dir = os.path.join(tmp_dir, "traces")
            os.makedirs(traces_dir, exist_ok=True)

            if active_traces:
                # 1. Individual JSON traces for every case
                all_traces_data = []
                for t in active_traces:
                    t_dict = t.model_dump() if hasattr(t, "model_dump") else dict(t)
                    t_dict_sanitized = sanitize_trace_payload(t_dict)
                    all_traces_data.append(t_dict_sanitized)

                    case_file_name = f"{t.case_id or t.trace_id}.json"
                    single_trace_path = os.path.join(traces_dir, case_file_name)
                    with open(single_trace_path, "w", encoding="utf-8") as f:
                        json.dump(t_dict_sanitized, f, indent=2)

                # 2. Line-delimited JSONL trace file containing all traces
                jsonl_path = os.path.join(traces_dir, "all_traces.jsonl")
                with open(jsonl_path, "w", encoding="utf-8") as f:
                    for td in all_traces_data:
                        f.write(json.dumps(td) + "\n")

                # 3. Representative traces (Single tool, Tool chaining, Clarification)
                rep_cases = {}
                for t in active_traces:
                    cid = t.case_id or ""
                    # Pick calculator/rag single-step
                    if "calc" in cid or "rag_01" in cid:
                        if "single_step" not in rep_cases:
                            rep_cases["single_step"] = t.model_dump()
                    # Pick multi-step chaining
                    elif "chain" in cid or "rag_04" in cid or "comp" in cid:
                        if "multi_step_chain" not in rep_cases:
                            rep_cases["multi_step_chain"] = t.model_dump()
                    # Pick clarification
                    elif "clarify" in cid or t.status == "clarification_required":
                        if "clarification" not in rep_cases:
                            rep_cases["clarification"] = t.model_dump()

                # Fallback to first 3 traces if representative keys not filled
                if len(rep_cases) < 2 and active_traces:
                    for idx, t in enumerate(active_traces[:3], 1):
                        rep_cases[f"sample_{idx}"] = t.model_dump()

                rep_traces_path = os.path.join(traces_dir, "representative_traces.json")
                with open(rep_traces_path, "w", encoding="utf-8") as f:
                    json.dump(sanitize_trace_payload(rep_cases), f, indent=2)

                # 4. Traces Markdown Summary
                traces_summary_path = os.path.join(traces_dir, "traces_summary.md")
                traces_md = build_markdown_trace_summary(active_traces)
                with open(traces_summary_path, "w", encoding="utf-8") as f:
                    f.write(traces_md)

            # (h) Evidently AI Regression Artifacts Subdirectory
            evidently_dir = os.path.join(tmp_dir, "evidently")
            os.makedirs(evidently_dir, exist_ok=True)

            if regression_result is not None:
                # 1. Regression Summary JSON
                reg_json_path = os.path.join(evidently_dir, "regression_summary.json")
                with open(reg_json_path, "w", encoding="utf-8") as f:
                    json.dump(regression_result.model_dump(), f, indent=2)

                # 2. Regression Summary Markdown
                reg_md_path = os.path.join(evidently_dir, "regression_summary.md")
                with open(reg_md_path, "w", encoding="utf-8") as f:
                    f.write(generate_markdown_regression_summary(regression_result))

                # 3. Copy Evidently HTML report if available
                if evidently_html_path and os.path.exists(evidently_html_path):
                    html_dest_name = os.path.basename(evidently_html_path)
                    shutil.copyfile(evidently_html_path, os.path.join(evidently_dir, html_dest_name))

            # Upload all artifacts to MLflow
            mlflow.log_artifacts(tmp_dir, artifact_path="evaluation")

        artifact_uri = run.info.artifact_uri
        logger.info(f"Successfully logged MLflow run '{name}' (RunID: {run_id}) with Evidently artifacts")

    return {
        "run_id": run_id,
        "run_name": name,
        "experiment_id": exp_id,
        "artifact_uri": artifact_uri,
        "params": params,
        "metrics": metrics,
        "trace_count": len(active_traces),
        "regression_detected": regression_result.regression_detected if regression_result else False,
    }
