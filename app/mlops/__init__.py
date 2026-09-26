"""MLOps and Experiment Tracking package for AI Assistant."""

from app.mlops.config_versions import (
    AgentConfigVersion,
    CONFIG_VERSIONS,
    get_config_version,
    list_config_versions,
    VERSION_1_BASELINE,
    VERSION_2_COMPACT_PRECISION,
    VERSION_3_DEEP_VALIDATION,
)
from app.mlops.trace_schema import (
    AgentExecutionTrace,
    AgentIterationTrace,
    build_markdown_trace_summary,
    build_trace_from_agent_result,
    is_sensitive_key,
    sanitize_trace_payload,
)
from app.mlops.evidently_monitoring import (
    RegressionEvaluationResult,
    RegressionThresholds,
    calculate_custom_metrics,
    evaluate_regression,
    generate_evidently_html_report,
    generate_markdown_regression_summary,
    run_evidently_regression_monitoring,
)

__all__ = [
    "AgentConfigVersion",
    "CONFIG_VERSIONS",
    "get_config_version",
    "list_config_versions",
    "VERSION_1_BASELINE",
    "VERSION_2_COMPACT_PRECISION",
    "VERSION_3_DEEP_VALIDATION",
    "AgentExecutionTrace",
    "AgentIterationTrace",
    "build_markdown_trace_summary",
    "build_trace_from_agent_result",
    "is_sensitive_key",
    "sanitize_trace_payload",
    "RegressionEvaluationResult",
    "RegressionThresholds",
    "calculate_custom_metrics",
    "evaluate_regression",
    "generate_evidently_html_report",
    "generate_markdown_regression_summary",
    "run_evidently_regression_monitoring",
]
