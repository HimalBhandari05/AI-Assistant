"""Evidently AI Regression Testing, Evaluation Monitoring, and MLflow Artifact Integration.

Provides structured regression testing comparing candidate agent configuration versions
against the reference baseline (v1_baseline) using the fixed 16-case benchmark suite.
Generates interactive Evidently HTML reports, machine-readable regression summaries,
and logs regression metrics and artifacts to MLflow.
"""

import datetime
import json
import logging
import os
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
from pydantic import BaseModel, Field

# Graceful import for Evidently (supports both modern and legacy paths)
try:
    from evidently.test_suite import TestSuite
    from evidently.report import Report
    from evidently.test_preset import DataDriftTestPreset, NoTargetPerformanceTestPreset
    from evidently.metric_preset import DataDriftPreset, DataQualityPreset
    from evidently.tests import (
        TestNumberOfColumns,
        TestNumberOfRows,
        TestColumnsType,
        TestColumnValueMean,
        TestColumnValueMin,
        TestColumnValueMax,
    )
    from evidently.pipeline.column_mapping import ColumnMapping
except (ImportError, ModuleNotFoundError):
    from evidently.legacy.test_suite import TestSuite
    from evidently.legacy.report import Report
    from evidently.legacy.test_preset import DataDriftTestPreset, NoTargetPerformanceTestPreset
    from evidently.legacy.metric_preset import DataDriftPreset, DataQualityPreset
    from evidently.legacy.tests import (
        TestNumberOfColumns,
        TestNumberOfRows,
        TestColumnsType,
        TestColumnValueMean,
        TestColumnValueMin,
        TestColumnValueMax,
    )
    from evidently.legacy.pipeline.column_mapping import ColumnMapping

from tests.evaluation.benchmark_cases import BENCHMARK_CASES, BenchmarkCase
from tests.evaluation.evaluator import BenchmarkSummary, CaseEvaluationResult

logger = logging.getLogger("ai_assistant.mlops.evidently")

DEFAULT_REPORTS_DIR = "reports/evidently"


class RegressionThresholds(BaseModel):
    """Configurable project regression thresholds evaluated against baseline."""
    min_pct_tests_passed: float = Field(default=95.0, description="Minimum allowable benchmark test pass percentage (%).")
    min_task_completion_rate: float = Field(default=95.0, description="Minimum allowable Task Completion Rate (%).")
    min_tool_call_correctness: float = Field(default=95.0, description="Minimum allowable Tool-Call Correctness (%).")
    max_trajectory_length_multiplier: float = Field(default=1.5, description="Max allowed multiplier on average trajectory length vs reference.")
    max_token_growth_multiplier: float = Field(default=1.75, description="Max allowed multiplier on average tokens per query vs reference.")
    max_hard_failures: int = Field(default=0, description="Maximum allowed hard failures.")


class RegressionEvaluationResult(BaseModel):
    """Machine-readable regression evaluation report comparing candidate vs reference."""
    reference_version: str = Field(..., description="Reference baseline version (e.g. v1_baseline).")
    current_version: str = Field(..., description="Candidate version evaluated.")
    timestamp: str = Field(default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat())
    total_cases: int = Field(..., description="Total benchmark cases evaluated.")
    passed_cases: int = Field(..., description="Number of cases passing all criteria.")
    pct_tests_passed: float = Field(..., description="Percentage of benchmark cases passed (0-100%).")
    task_completion_rate: float = Field(..., description="Task Completion Rate TCR (%).")
    tool_call_correctness: float = Field(..., description="Tool-Call Correctness TCC (%).")
    avg_trajectory_length: float = Field(..., description="Candidate average trajectory length.")
    ref_avg_trajectory_length: float = Field(..., description="Reference average trajectory length.")
    avg_tokens_per_query: float = Field(..., description="Candidate average tokens per query.")
    ref_avg_tokens_per_query: float = Field(..., description="Reference average tokens per query.")
    trajectory_efficiency_score: float = Field(..., description="Custom metric: TCR / avg_trajectory_length.")
    token_cost_ratio: float = Field(..., description="Custom metric: Candidate avg tokens / Reference avg tokens.")
    regression_detected: bool = Field(default=False, description="True if any regression threshold was breached.")
    regression_reasons: List[str] = Field(default_factory=list, description="List of reasons triggering regression.")
    failed_cases: List[Dict[str, Any]] = Field(default_factory=list, description="Details of failed benchmark cases.")
    metrics: Dict[str, float] = Field(default_factory=dict, description="Flat regression metrics dictionary.")
    thresholds: Dict[str, Any] = Field(default_factory=dict, description="Applied threshold parameters.")
    html_report_path: Optional[str] = Field(default=None, description="Path to generated Evidently HTML report.")


def convert_benchmark_results_to_dataframe(summary: BenchmarkSummary) -> pd.DataFrame:
    """Convert BenchmarkSummary case results into a standardized pandas DataFrame for Evidently."""
    rows = []
    for r in summary.results:
        rows.append({
            "case_id": r.case_id,
            "category": r.category,
            "task_success": 1 if r.success else 0,
            "status": r.status,
            "trajectory_length": r.trajectory_length,
            "tool_calls_count": r.tool_calls_count,
            "correct_tool_calls_count": r.correct_tool_calls_count,
            "prompt_tokens": r.prompt_tokens,
            "completion_tokens": r.completion_tokens,
            "total_tokens": r.total_tokens,
            "execution_time_seconds": r.execution_time_seconds,
            "hard_failure": 1 if r.failure_classification == "Hard Failure" else 0,
            "soft_failure": 1 if r.failure_classification in ("Soft Failure", "Cascading Soft Failure") else 0,
        })
    return pd.DataFrame(rows)


def calculate_custom_metrics(
    current_summary: BenchmarkSummary,
    reference_summary: BenchmarkSummary,
) -> Dict[str, float]:
    """Compute domain-specific custom agentic metrics.

    Metrics:
    1. pct_tests_passed: Percentage of benchmark test cases that passed all ground truth assertions.
    2. trajectory_efficiency_score: (TCR / avg_trajectory_length) - Measures goal completion density per reasoning step.
    3. token_cost_ratio: (current_tokens / reference_tokens) - Measures token inflation vs reference baseline.
    """
    total_cases = len(current_summary.results)
    passed_cases = sum(1 for r in current_summary.results if r.success)
    pct_passed = (passed_cases / total_cases * 100.0) if total_cases > 0 else 0.0

    avg_traj = current_summary.avg_trajectory_length
    tcr = current_summary.task_completion_rate
    tes = (tcr / avg_traj) if avg_traj > 0 else 0.0

    ref_tokens = reference_summary.avg_tokens_per_query
    curr_tokens = current_summary.avg_tokens_per_query
    token_ratio = (curr_tokens / ref_tokens) if ref_tokens > 0 else 1.0

    return {
        "pct_tests_passed": round(pct_passed, 2),
        "trajectory_efficiency_score": round(tes, 2),
        "token_cost_ratio": round(token_ratio, 3),
    }


def evaluate_regression(
    current_summary: BenchmarkSummary,
    reference_summary: BenchmarkSummary,
    current_version: str,
    reference_version: str = "v1_baseline",
    thresholds: Optional[RegressionThresholds] = None,
) -> RegressionEvaluationResult:
    """Evaluate candidate benchmark summary against reference baseline and check regression thresholds."""
    th = thresholds or RegressionThresholds()
    custom = calculate_custom_metrics(current_summary, reference_summary)

    total_cases = len(current_summary.results)
    passed_cases = sum(1 for r in current_summary.results if r.success)
    pct_passed = custom["pct_tests_passed"]
    tcr = current_summary.task_completion_rate
    tcc = current_summary.tool_call_correctness
    avg_traj = current_summary.avg_trajectory_length
    ref_avg_traj = reference_summary.avg_trajectory_length
    avg_tok = current_summary.avg_tokens_per_query
    ref_avg_tok = reference_summary.avg_tokens_per_query

    regression_reasons: List[str] = []
    failed_cases: List[Dict[str, Any]] = []

    # Check failed cases
    for r in current_summary.results:
        if not r.success:
            failed_cases.append({
                "case_id": r.case_id,
                "category": r.category,
                "question": r.question,
                "status": r.status,
                "failure_classification": r.failure_classification,
                "failure_details": r.failure_details,
                "trajectory_length": r.trajectory_length,
                "total_tokens": r.total_tokens,
            })

    # Threshold checks
    if pct_passed < th.min_pct_tests_passed:
        regression_reasons.append(
            f"Test Pass Percentage ({pct_passed:.1f}%) is below minimum threshold ({th.min_pct_tests_passed:.1f}%)"
        )
    if tcr < th.min_task_completion_rate:
        regression_reasons.append(
            f"Task Completion Rate TCR ({tcr:.1f}%) is below minimum threshold ({th.min_task_completion_rate:.1f}%)"
        )
    if tcc < th.min_tool_call_correctness:
        regression_reasons.append(
            f"Tool-Call Correctness TCC ({tcc:.1f}%) is below minimum threshold ({th.min_tool_call_correctness:.1f}%)"
        )
    if current_summary.hard_failures > th.max_hard_failures:
        regression_reasons.append(
            f"Hard failures count ({current_summary.hard_failures}) exceeded max allowed ({th.max_hard_failures})"
        )
    if ref_avg_traj > 0 and avg_traj > (ref_avg_traj * th.max_trajectory_length_multiplier):
        regression_reasons.append(
            f"Average trajectory length ({avg_traj:.2f}) exceeded allowed multiplier ({th.max_trajectory_length_multiplier}x of ref {ref_avg_traj:.2f})"
        )
    if ref_avg_tok > 0 and avg_tok > (ref_avg_tok * th.max_token_growth_multiplier):
        regression_reasons.append(
            f"Average tokens per query ({avg_tok:.1f}) exceeded allowed growth multiplier ({th.max_token_growth_multiplier}x of ref {ref_avg_tok:.1f})"
        )

    regression_detected = len(regression_reasons) > 0

    metrics_dict = {
        "pct_tests_passed": pct_passed,
        "task_completion_rate": tcr,
        "tool_call_correctness": tcc,
        "avg_trajectory_length": avg_traj,
        "avg_tokens_per_query": avg_tok,
        "trajectory_efficiency_score": custom["trajectory_efficiency_score"],
        "token_cost_ratio": custom["token_cost_ratio"],
        "hard_failures": float(current_summary.hard_failures),
        "soft_failures": float(current_summary.soft_failures),
        "regression_detected": 1.0 if regression_detected else 0.0,
    }

    return RegressionEvaluationResult(
        reference_version=reference_version,
        current_version=current_version,
        total_cases=total_cases,
        passed_cases=passed_cases,
        pct_tests_passed=pct_passed,
        task_completion_rate=tcr,
        tool_call_correctness=tcc,
        avg_trajectory_length=avg_traj,
        ref_avg_trajectory_length=ref_avg_traj,
        avg_tokens_per_query=avg_tok,
        ref_avg_tokens_per_query=ref_avg_tok,
        trajectory_efficiency_score=custom["trajectory_efficiency_score"],
        token_cost_ratio=custom["token_cost_ratio"],
        regression_detected=regression_detected,
        regression_reasons=regression_reasons,
        failed_cases=failed_cases,
        metrics=metrics_dict,
        thresholds=th.model_dump(),
    )


def generate_evidently_html_report(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    output_html_path: str,
    current_version: str,
    reference_version: str = "v1_baseline",
) -> str:
    """Run Evidently TestSuite and Report on reference vs current dataframes and save merged HTML."""
    os.makedirs(os.path.dirname(output_html_path), exist_ok=True)

    cm = ColumnMapping(
        numerical_features=[
            "task_success",
            "trajectory_length",
            "tool_calls_count",
            "correct_tool_calls_count",
            "prompt_tokens",
            "completion_tokens",
            "total_tokens",
            "execution_time_seconds",
            "hard_failure",
            "soft_failure",
        ],
        categorical_features=["case_id", "category", "status"],
    )

    # Initialize Evidently TestSuite with agent-specific assertions
    test_suite = TestSuite(tests=[
        TestNumberOfRows(),
        TestColumnsType(),
        TestColumnValueMin(column_name="task_success", gte=0.0),
        TestColumnValueMax(column_name="hard_failure", lte=0.0),
        TestColumnValueMean(column_name="task_success", gte=0.95),
        DataDriftTestPreset(),
    ])

    test_suite.run(reference_data=reference_df, current_data=current_df, column_mapping=cm)
    test_suite.save_html(output_html_path)
    logger.info(f"Saved Evidently HTML report to: {output_html_path}")

    return output_html_path


def generate_markdown_regression_summary(reg_result: RegressionEvaluationResult) -> str:
    """Format human-readable markdown regression analysis summary."""
    status_badge = "🚨 REGRESSION DETECTED" if reg_result.regression_detected else "✅ NO REGRESSION DETECTED"
    lines = [
        f"# Evidently AI Regression Analysis Report: `{reg_result.current_version}` vs `{reg_result.reference_version}`",
        "",
        f"**Overall Status**: {status_badge}",
        f"**Timestamp**: `{reg_result.timestamp}`",
        "",
        "## 1. Regression Metrics Overview",
        "",
        "| Metric | Reference (`" + reg_result.reference_version + "`) | Current (`" + reg_result.current_version + "`) | Status | Threshold |",
        "| :--- | :---: | :---: | :---: | :--- |",
        f"| **Benchmark Pass Rate** | `100.0%` | `{reg_result.pct_tests_passed:.1f}%` | {'✅' if reg_result.pct_tests_passed >= reg_result.thresholds['min_pct_tests_passed'] else '❌'} | $\\ge {reg_result.thresholds['min_pct_tests_passed']:.1f}\\%$ |",
        f"| **Task Completion Rate (TCR)** | `100.0%` | `{reg_result.task_completion_rate:.1f}%` | {'✅' if reg_result.task_completion_rate >= reg_result.thresholds['min_task_completion_rate'] else '❌'} | $\\ge {reg_result.thresholds['min_task_completion_rate']:.1f}\\%$ |",
        f"| **Tool-Call Correctness (TCC)** | `100.0%` | `{reg_result.tool_call_correctness:.1f}%` | {'✅' if reg_result.tool_call_correctness >= reg_result.thresholds['min_tool_call_correctness'] else '❌'} | $\\ge {reg_result.thresholds['min_tool_call_correctness']:.1f}\\%$ |",
        f"| **Average Trajectory Length** | `{reg_result.ref_avg_trajectory_length:.2f}` | `{reg_result.avg_trajectory_length:.2f}` | {'✅' if reg_result.avg_trajectory_length <= reg_result.ref_avg_trajectory_length * reg_result.thresholds['max_trajectory_length_multiplier'] else '❌'} | $\\le {reg_result.ref_avg_trajectory_length * reg_result.thresholds['max_trajectory_length_multiplier']:.2f}$ |",
        f"| **Average Tokens per Query** | `{reg_result.ref_avg_tokens_per_query:.1f}` | `{reg_result.avg_tokens_per_query:.1f}` | {'✅' if reg_result.avg_tokens_per_query <= reg_result.ref_avg_tokens_per_query * reg_result.thresholds['max_token_growth_multiplier'] else '❌'} | $\\le {reg_result.ref_avg_tokens_per_query * reg_result.thresholds['max_token_growth_multiplier']:.1f}$ |",
        f"| **Trajectory Efficiency Score (TES)** | `{100.0 / reg_result.ref_avg_trajectory_length if reg_result.ref_avg_trajectory_length > 0 else 0:.1f}` | `{reg_result.trajectory_efficiency_score:.1f}` | {'✅' if reg_result.trajectory_efficiency_score >= 35.0 else '⚠️'} | $\\ge 35.0$ |",
        f"| **Token Cost Ratio** | `1.000` | `{reg_result.token_cost_ratio:.3f}` | {'✅' if reg_result.token_cost_ratio <= reg_result.thresholds['max_token_growth_multiplier'] else '❌'} | $\\le {reg_result.thresholds['max_token_growth_multiplier']:.2f}$ |",
        "",
    ]

    if reg_result.regression_detected:
        lines.append("## 2. Regression Breach Reasons")
        lines.append("")
        for reason in reg_result.regression_reasons:
            lines.append(f"- ❌ **{reason}**")
        lines.append("")
    else:
        lines.append("## 2. Regression Verification")
        lines.append("")
        lines.append("All evaluation metrics and custom agentic performance scores remained within configured regression thresholds.")
        lines.append("")

    if reg_result.failed_cases:
        lines.append("## 3. Failed Benchmark Case Analysis")
        lines.append("")
        for fc in reg_result.failed_cases:
            lines.append(f"### Case `{fc['case_id']}` ({fc['category']})")
            lines.append(f"- **Query**: *\"{fc['question']}\"*")
            lines.append(f"- **Status**: `{fc['status']}`")
            lines.append(f"- **Classification**: `{fc['failure_classification']}`")
            lines.append(f"- **Failure Details**: {fc['failure_details']}")
            lines.append(f"- **Trajectory Length**: {fc['trajectory_length']} steps | **Tokens**: {fc['total_tokens']}")
            lines.append("")
    else:
        lines.append("## 3. Benchmark Correctness Analysis")
        lines.append("")
        lines.append(f"All **{reg_result.total_cases}/{reg_result.total_cases}** fixed benchmark test cases completed successfully with zero functional regressions.")
        lines.append("")

    return "\n".join(lines)


def run_evidently_regression_monitoring(
    current_summary: BenchmarkSummary,
    reference_summary: BenchmarkSummary,
    current_version: str,
    reference_version: str = "v1_baseline",
    reports_dir: str = DEFAULT_REPORTS_DIR,
    thresholds: Optional[RegressionThresholds] = None,
) -> Tuple[RegressionEvaluationResult, str]:
    """Execute complete Evidently regression evaluation, generate HTML report, and return result object."""
    ref_df = convert_benchmark_results_to_dataframe(reference_summary)
    curr_df = convert_benchmark_results_to_dataframe(current_summary)

    html_filename = f"{current_version}_vs_{reference_version}.html"
    html_path = os.path.join(reports_dir, html_filename)

    generate_evidently_html_report(
        reference_df=ref_df,
        current_df=curr_df,
        output_html_path=html_path,
        current_version=current_version,
        reference_version=reference_version,
    )

    reg_result = evaluate_regression(
        current_summary=current_summary,
        reference_summary=reference_summary,
        current_version=current_version,
        reference_version=reference_version,
        thresholds=thresholds,
    )
    reg_result.html_report_path = html_path

    # Persist JSON & Markdown regression summaries
    os.makedirs(reports_dir, exist_ok=True)
    json_path = os.path.join(reports_dir, f"{current_version}_regression_summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(reg_result.model_dump(), f, indent=2)

    md_path = os.path.join(reports_dir, f"{current_version}_regression_summary.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(generate_markdown_regression_summary(reg_result))

    return reg_result, html_path
