"""Observability, Token Accounting, Tool Metrics, and Structured Failure Logging for Agentic Assistant."""

import logging
import time
import uuid
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from app.agent.context import TokenUsage

logger = logging.getLogger("ai_assistant.agent.metrics")


class ToolExecutionMetric(BaseModel):
    """Execution telemetry for an individual tool dispatch."""
    tool_name: str
    arguments_valid: bool
    success: bool
    execution_time_seconds: float
    error_type: Optional[str] = None
    error_message: Optional[str] = None


class AgentFailureRecord(BaseModel):
    """Structured failure log entry for debugging and evaluation."""
    failure_type: str = Field(..., description="Classification of failure (e.g. MaxIterationsExceeded, ToolExecutionError).")
    agent_run_id: str = Field(..., description="Unique run identifier.")
    iteration: int = Field(..., description="Iteration during which failure occurred.")
    question: str = Field(..., description="The user prompt.")
    details: Dict[str, Any] = Field(default_factory=dict, description="Contextual debugging details (excluding secrets).")
    timestamp: float = Field(default_factory=time.time)


class AgentExecutionMetrics(BaseModel):
    """Comprehensive observability record for a complete agent run."""
    agent_run_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    question: str
    status: str = "in_progress"
    trajectory_length: int = 0
    total_execution_time_seconds: float = 0.0
    token_usage: TokenUsage = Field(default_factory=TokenUsage)
    tool_metrics: List[ToolExecutionMetric] = Field(default_factory=list)
    failures: List[AgentFailureRecord] = Field(default_factory=list)

    def record_tool_call(
        self,
        tool_name: str,
        arguments_valid: bool,
        success: bool,
        execution_time_seconds: float,
        error_type: Optional[str] = None,
        error_message: Optional[str] = None,
    ) -> None:
        """Log telemetry for a tool call."""
        metric = ToolExecutionMetric(
            tool_name=tool_name,
            arguments_valid=arguments_valid,
            success=success,
            execution_time_seconds=execution_time_seconds,
            error_type=error_type,
            error_message=error_message,
        )
        self.tool_metrics.append(metric)

    def record_failure(
        self,
        failure_type: str,
        iteration: int,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record structured failure event safely."""
        clean_details = {}
        if details:
            # Strip potential sensitive keys
            for k, v in details.items():
                if any(sec in k.lower() for sec in ["key", "secret", "token", "password", "auth"]):
                    clean_details[k] = "[REDACTED]"
                else:
                    clean_details[k] = v

        record = AgentFailureRecord(
            failure_type=failure_type,
            agent_run_id=self.agent_run_id,
            iteration=iteration,
            question=self.question,
            details=clean_details,
        )
        self.failures.append(record)
        logger.warning(
            f"[Agent Failure] RunID={self.agent_run_id} Type={failure_type} "
            f"Iteration={iteration}: {clean_details}"
        )
