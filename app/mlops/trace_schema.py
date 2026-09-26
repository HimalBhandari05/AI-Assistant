"""Trace Schema, Data Models, and Redaction Utilities for Agent Telemetry.

Provides structured models for capturing full observable agent execution traces
including per-iteration model decisions, tool calls, tool results, token counts,
execution timing, and recovery events. All trace payloads are strictly sanitized
to redact sensitive keys (passwords, tokens, API keys, secrets).
"""

import copy
import datetime
import json
import logging
import re
import uuid
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field

SAFE_KEYS = {
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "token_usage",
    "token_count",
    "tokens",
    "avg_tokens_per_query",
    "total_tokens_consumed",
}

SENSITIVE_KEY_PATTERNS = [
    re.compile(r"api[_-]?key", re.IGNORECASE),
    re.compile(r"secret", re.IGNORECASE),
    re.compile(r"(auth|bearer|access|refresh|session|jwt|api|security)[_-]?token", re.IGNORECASE),
    re.compile(r"^token$", re.IGNORECASE),
    re.compile(r"password", re.IGNORECASE),
    re.compile(r"auth(orization)?", re.IGNORECASE),
    re.compile(r"credential", re.IGNORECASE),
    re.compile(r"private[_-]?key", re.IGNORECASE),
]


def is_sensitive_key(key: str) -> bool:
    """Check if a dictionary key matches any sensitive naming patterns."""
    if not isinstance(key, str):
        return False
    norm_key = key.lower().strip()
    if norm_key in SAFE_KEYS:
        return False
    return any(pattern.search(key) for pattern in SENSITIVE_KEY_PATTERNS)


def sanitize_trace_payload(data: Any) -> Any:
    """Recursively traverse and redact sensitive values from dictionaries, lists, and strings.

    Args:
        data: Arbitrary object (dict, list, str, primitive) to sanitize.

    Returns:
        Deep-copied and redacted version of the data payload.
    """
    if isinstance(data, dict):
        sanitized = {}
        for k, v in data.items():
            if is_sensitive_key(str(k)):
                sanitized[k] = "[REDACTED]"
            else:
                sanitized[k] = sanitize_trace_payload(v)
        return sanitized
    elif isinstance(data, list):
        return [sanitize_trace_payload(item) for item in data]
    elif isinstance(data, tuple):
        return [sanitize_trace_payload(item) for item in data]
    elif isinstance(data, str):
        # Basic check for inline tokens or credentials in strings
        cleaned = data
        for pattern in [r"AIza[0-9A-Za-z-_]{35}", r"Bearer\s+[A-Za-z0-9-_.]+", r"key=[A-Za-z0-9-_]+"]:
            cleaned = re.sub(pattern, "[REDACTED_CREDENTIAL]", cleaned)
        return cleaned
    else:
        return data


class AgentIterationTrace(BaseModel):
    """Detailed trace of a single agent reasoning and action step."""
    iteration: int = Field(..., description="1-indexed iteration counter.")
    decision_type: str = Field(..., description="Decision type: 'tool_call', 'final_answer', 'ask_user_clarification', 'max_iterations'.")
    tool_name: Optional[str] = Field(default=None, description="Name of the selected tool if applicable.")
    tool_arguments: Optional[Dict[str, Any]] = Field(default=None, description="Sanitized input arguments passed to tool.")
    raw_result: Optional[Any] = Field(default=None, description="Sanitized raw tool result or summary.")
    observation: Optional[str] = Field(default=None, description="Bounded structured observation returned to agent.")
    success: bool = Field(default=True, description="Whether the step or tool completed successfully.")
    execution_time_seconds: float = Field(default=0.0, description="Execution time of this step in seconds.")
    error_type: Optional[str] = Field(default=None, description="Error type if failure occurred.")
    error_message: Optional[str] = Field(default=None, description="Sanitized error description.")
    prompt_tokens: int = Field(default=0, description="Estimated/measured prompt tokens for this step.")
    completion_tokens: int = Field(default=0, description="Estimated/measured completion tokens for this step.")


class AgentExecutionTrace(BaseModel):
    """Complete structured telemetry trace for a full agent request workflow."""
    trace_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique identifier for trace.")
    timestamp: str = Field(default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat(), description="ISO 8601 execution timestamp.")
    case_id: Optional[str] = Field(default=None, description="Benchmark test case ID if part of evaluation suite.")
    query: str = Field(..., description="Original user prompt or test question.")
    config_version: str = Field(default="v1_baseline", description="Configuration version ID.")
    prompt_version: str = Field(default="v1_baseline", description="Prompt template version.")
    model: str = Field(default="gemini-2.5-flash", description="Underlying model identifier.")
    temperature: float = Field(default=0.1, description="Sampling temperature.")
    status: str = Field(..., description="Final status: 'completed', 'clarification_required', 'max_iterations_exceeded', 'timeout', 'error'.")
    termination_reason: str = Field(..., description="Human-readable reason for workflow completion or termination.")
    final_answer: str = Field(..., description="Final synthesized answer or clarification prompt.")
    total_iterations: int = Field(..., description="Total iterations executed.")
    total_execution_time_seconds: float = Field(default=0.0, description="Total wall-clock execution time.")
    token_usage: Dict[str, int] = Field(default_factory=dict, description="Token consumption dictionary.")
    iterations: List[AgentIterationTrace] = Field(default_factory=list, description="Chronological list of iteration traces.")
    errors_and_recoveries: List[Dict[str, Any]] = Field(default_factory=list, description="Logged tool errors and recovery events.")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional context parameters and run metadata.")


def build_trace_from_agent_result(
    agent_result: Any,
    query: str,
    config_version: str = "v1_baseline",
    prompt_version: str = "v1_baseline",
    model: str = "gemini-2.5-flash",
    temperature: float = 0.1,
    case_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> AgentExecutionTrace:
    """Construct a validated AgentExecutionTrace from an AgentResult and configuration metadata.

    Args:
        agent_result: The AgentResult instance returned by engine.
        query: The user prompt or benchmark question.
        config_version: Configuration version identifier.
        prompt_version: Prompt version tag.
        model: Underlying LLM model name.
        temperature: Sampling temperature.
        case_id: Optional benchmark case ID.
        metadata: Optional extra metadata dictionary.

    Returns:
        AgentExecutionTrace object with sanitized telemetry.
    """
    iteration_traces: List[AgentIterationTrace] = []
    errors_recoveries: List[Dict[str, Any]] = []

    trajectory = getattr(agent_result, "trajectory", []) or []
    for step in trajectory:
        action = getattr(step, "action", "unknown")
        args = sanitize_trace_payload(getattr(step, "arguments", {}) or {})
        obs = getattr(step, "observation", None)
        success = getattr(step, "success", True)
        step_time = float(getattr(step, "execution_time_seconds", 0.0))

        if action == "final_answer":
            decision_type = "final_answer"
            tool_name = None
        elif action == "ask_user_clarification":
            decision_type = "ask_user_clarification"
            tool_name = "ask_user_clarification"
        else:
            decision_type = "tool_call"
            tool_name = action

        err_type = None
        err_msg = None
        if not success or (obs and "[Tool Error" in obs):
            err_type = "ToolExecutionError"
            err_msg = obs
            errors_recoveries.append({
                "iteration": getattr(step, "iteration", len(iteration_traces) + 1),
                "tool": tool_name,
                "error": err_msg,
                "recovered": True if agent_result.status == "completed" else False,
            })

        iteration_traces.append(
            AgentIterationTrace(
                iteration=getattr(step, "iteration", len(iteration_traces) + 1),
                decision_type=decision_type,
                tool_name=tool_name,
                tool_arguments=args,
                raw_result=None,
                observation=obs,
                success=success,
                execution_time_seconds=step_time,
                error_type=err_type,
                error_message=err_msg,
            )
        )

    # Determine termination reason
    status = getattr(agent_result, "status", "completed")
    total_iters = len(iteration_traces)
    if status == "completed":
        term_reason = f"Final answer successfully synthesized after {total_iters} iterations."
    elif status == "clarification_required":
        term_reason = f"Ambiguous user query detected; requested user clarification at iteration {total_iters}."
    elif status == "max_iterations_exceeded":
        term_reason = f"Agent reached maximum iteration budget ({total_iters}) without converging."
    elif status == "timeout":
        term_reason = "Agent execution timed out exceeding max allowed execution seconds."
    else:
        term_reason = f"Agent execution ended with status: {status}."

    # Extract token usage
    token_usage_obj = getattr(agent_result, "token_usage", None)
    if token_usage_obj:
        token_usage_dict = {
            "prompt_tokens": getattr(token_usage_obj, "prompt_tokens", 0),
            "completion_tokens": getattr(token_usage_obj, "completion_tokens", 0),
            "total_tokens": getattr(token_usage_obj, "total_tokens", 0),
        }
    else:
        token_usage_dict = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    # Extract total execution time
    metrics_obj = getattr(agent_result, "metrics", None)
    if metrics_obj and hasattr(metrics_obj, "total_execution_time_seconds"):
        exec_time = float(metrics_obj.total_execution_time_seconds)
    else:
        exec_time = sum(t.execution_time_seconds for t in iteration_traces)

    # Merge failure events from metrics if present
    if metrics_obj and hasattr(metrics_obj, "failure_events"):
        for fe in metrics_obj.failure_events:
            errors_recoveries.append(sanitize_trace_payload(fe))

    meta = sanitize_trace_payload(metadata or {})

    return AgentExecutionTrace(
        case_id=case_id,
        query=query,
        config_version=config_version,
        prompt_version=prompt_version,
        model=model,
        temperature=temperature,
        status=status,
        termination_reason=term_reason,
        final_answer=getattr(agent_result, "answer", ""),
        total_iterations=total_iters,
        total_execution_time_seconds=round(exec_time, 4),
        token_usage=token_usage_dict,
        iterations=iteration_traces,
        errors_and_recoveries=errors_recoveries,
        metadata=meta,
    )


def build_markdown_trace_summary(traces: List[AgentExecutionTrace]) -> str:
    """Generate a clean, human-readable Markdown summary report of execution traces.

    Args:
        traces: List of AgentExecutionTrace objects.

    Returns:
        Formatted Markdown text document.
    """
    lines = [
        "# Agent Execution Trace Summary",
        "",
        f"- **Total Traces Logged**: {len(traces)}",
        f"- **Generated At**: {datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}",
        "",
        "## Summary Table",
        "",
        "| Case ID | Status | Iterations | Total Time (s) | Prompt Tokens | Completion Tokens | Total Tokens | Termination Reason |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |",
    ]

    for t in traces:
        cid = t.case_id or "N/A"
        p_tok = t.token_usage.get("prompt_tokens", 0)
        c_tok = t.token_usage.get("completion_tokens", 0)
        tot_tok = t.token_usage.get("total_tokens", 0)
        lines.append(
            f"| `{cid}` | `{t.status}` | {t.total_iterations} | {t.total_execution_time_seconds:.3f} | {p_tok} | {c_tok} | {tot_tok} | {t.termination_reason} |"
        )

    lines.append("")
    lines.append("## Detailed Representative Traces")
    lines.append("")

    for idx, t in enumerate(traces[:5], start=1):
        lines.append(f"### Trace {idx}: {t.case_id or t.trace_id[:8]} (`{t.status}`)")
        lines.append(f"- **Query**: *\"{t.query}\"*")
        lines.append(f"- **Config Version**: `{t.config_version}` | **Model**: `{t.model}`")
        lines.append(f"- **Final Answer**: {t.final_answer}")
        lines.append(f"- **Iterations ({t.total_iterations})**:")
        for step in t.iterations:
            if step.decision_type == "tool_call":
                args_str = json.dumps(step.tool_arguments)
                lines.append(f"  - **Iter {step.iteration} [Tool Call]**: `{step.tool_name}({args_str})` -> `{step.observation}` ({step.execution_time_seconds:.3f}s)")
            elif step.decision_type == "ask_user_clarification":
                lines.append(f"  - **Iter {step.iteration} [Clarification]**: `{step.observation}` ({step.execution_time_seconds:.3f}s)")
            elif step.decision_type == "final_answer":
                lines.append(f"  - **Iter {step.iteration} [Final Answer Decision]**: ({step.execution_time_seconds:.3f}s)")
            else:
                lines.append(f"  - **Iter {step.iteration} [{step.decision_type}]**: ({step.execution_time_seconds:.3f}s)")

        if t.errors_and_recoveries:
            lines.append("- **Errors & Recoveries**:")
            for err in t.errors_and_recoveries:
                lines.append(f"  - `{json.dumps(err)}`")

        lines.append("")

    return "\n".join(lines)
