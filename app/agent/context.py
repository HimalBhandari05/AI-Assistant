"""Structured Context Engineering, Agent State, Trajectory Tracking, and Scratchpad Compaction.

================================================================================
Context Engineering Architecture & Design
================================================================================
Problem:
    In an iterative multi-step agent workflow, repeated document searches (RAG),
    intermediate tool outputs, and successive reasoning iterations produce rapidly
    inflating context windows. Unbounded context growth leads to high token costs,
    latency inflation, and potential context window degradation.

Technique:
    1. Structured Trajectory Tracking: Represents each agent step strictly as
       (iteration, action, arguments, observation, success, timing) without storing
       unnecessary private chain-of-thought tokens.
    2. Observation Truncation: Binds raw tool outputs (e.g. retrieved document chunks)
       to deterministic character and item budgets (max 400 chars/chunk, max 1200 chars/obs).
    3. Lightweight Compaction: When the trajectory length grows beyond 3 iterations,
       older steps are rolled up into compact single-line summaries while retaining full
       observation fidelity for the most recent steps.

Benefit:
    Maintains a deterministic token footprint per iteration, preserves critical factual
    evidence, and ensures consistent decision-making performance across long trajectories.
================================================================================
"""

import json
import logging
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

logger = logging.getLogger("ai_assistant.agent.context")

# Context budget limits
MAX_CHUNK_CHARS = 400
MAX_OBSERVATION_CHARS = 1200
MAX_DETAILED_STEPS = 2  # Keep full details for the last N steps, compact older


class TokenUsage(BaseModel):
    """Token consumption accounting per step and aggregate request."""
    prompt_tokens: int = Field(default=0, description="Total input/prompt tokens consumed.")
    completion_tokens: int = Field(default=0, description="Total output/completion tokens consumed.")
    total_tokens: int = Field(default=0, description="Sum of prompt and completion tokens.")
    token_usage_available: bool = Field(default=True, description="False if provider does not expose token metrics.")

    def add(self, prompt: int, completion: int) -> None:
        """Accumulate token counts."""
        self.prompt_tokens += max(0, prompt)
        self.completion_tokens += max(0, completion)
        self.total_tokens = self.prompt_tokens + self.completion_tokens


class TrajectoryStep(BaseModel):
    """Structured representation of an individual agent action-observation step."""
    iteration: int = Field(..., description="1-indexed iteration counter.")
    action: str = Field(..., description="The tool or action name decided by the agent.")
    arguments: Dict[str, Any] = Field(default_factory=dict, description="Input arguments passed to the action.")
    observation: Optional[str] = Field(default=None, description="Bounded structured outcome of the action.")
    success: bool = Field(default=True, description="Whether the action/tool completed successfully.")
    execution_time_seconds: float = Field(default=0.0, description="Execution time of the step in seconds.")


class AgentState(BaseModel):
    """Complete state snapshot of an active agent execution."""
    user_question: str = Field(..., description="Original user prompt.")
    iteration: int = Field(default=0, description="Current iteration count.")
    trajectory: List[TrajectoryStep] = Field(default_factory=list, description="Ordered history of steps taken.")
    available_tools: List[dict] = Field(default_factory=list, description="Tool metadata schemas exposed to agent.")
    final_answer: Optional[str] = Field(default=None, description="Final synthesised answer.")
    topic: Optional[str] = Field(default=None, description="Classified topic.")
    clarification_required: bool = Field(default=False, description="True if clarification was requested.")
    clarification_question: Optional[str] = Field(default=None, description="The clarification prompt if requested.")
    status: str = Field(default="in_progress", description="Status: 'in_progress', 'completed', 'clarification_required', 'max_iterations_exceeded', 'timeout', 'error'.")
    token_usage: TokenUsage = Field(default_factory=TokenUsage, description="Aggregated token usage metrics.")


def truncate_observation(tool_name: str, raw_result: Any, error: Optional[dict] = None) -> str:
    """Transform raw tool results or errors into bounded, structured observation strings.

    Args:
        tool_name: Name of the executed tool.
        raw_result: The raw output data payload from tool execution.
        error: Error dictionary if tool execution failed.

    Returns:
        A concise, bounded observation string suitable for the agent scratchpad.
    """
    if error:
        return f"[Tool Error ({tool_name})]: {error.get('type', 'Error')} - {error.get('message', 'Unknown error')}"

    if raw_result is None:
        return f"[Tool Output ({tool_name})]: Success with no return payload."

    if tool_name == "rag_search":
        if isinstance(raw_result, dict):
            query = raw_result.get("query", "")
            chunks = raw_result.get("results", [])
            count = raw_result.get("count", len(chunks))

            if count == 0 or not chunks:
                return f"[RAG Search Results for '{query}']: 0 relevant document chunks found."

            obs_lines = [f"[RAG Search Results for '{query}' ({count} chunks found)]:"]
            for idx, c in enumerate(chunks[:3], 1):
                src = c.get("source", "doc")
                chunk_id = c.get("chunk_id", 0)
                text = c.get("text", "").strip()
                # Truncate individual chunk text if too large
                if len(text) > MAX_CHUNK_CHARS:
                    text = text[:MAX_CHUNK_CHARS] + "... [truncated]"
                obs_lines.append(f"  ({idx}) [Source: {src}, Chunk: {chunk_id}]: {text}")

            combined = "\n".join(obs_lines)
            if len(combined) > MAX_OBSERVATION_CHARS:
                return combined[:MAX_OBSERVATION_CHARS] + "\n... [Observation Truncated]"
            return combined

    if tool_name == "calculator":
        if isinstance(raw_result, dict):
            a = raw_result.get("a")
            b = raw_result.get("b")
            op = raw_result.get("operation")
            res = raw_result.get("result")
            return f"[Calculator Result]: {a} {op} {b} = {res}"

    if tool_name == "ask_user_clarification":
        if isinstance(raw_result, dict):
            q = raw_result.get("question", "")
            return f"[Clarification Requested]: {q}"

    # Generic JSON fallback
    try:
        raw_str = json.dumps(raw_result, ensure_ascii=False)
    except Exception:
        raw_str = str(raw_result)

    if len(raw_str) > MAX_OBSERVATION_CHARS:
        raw_str = raw_str[:MAX_OBSERVATION_CHARS] + "... [truncated]"
    return f"[Tool Output ({tool_name})]: {raw_str}"


def build_compacted_scratchpad(trajectory: List[TrajectoryStep]) -> str:
    """Build a scratchpad representation of past trajectory steps with lightweight compaction.

    Steps older than MAX_DETAILED_STEPS are compacted to single-line summaries to bound context.
    """
    if not trajectory:
        return "No actions taken yet."

    total_steps = len(trajectory)
    cutoff_index = max(0, total_steps - MAX_DETAILED_STEPS)

    lines = []
    for idx, step in enumerate(trajectory):
        is_older = idx < cutoff_index

        if is_older:
            # Compact summary for older steps
            args_brief = ", ".join(f"{k}={v}" for k, v in step.arguments.items())
            obs_brief = (step.observation or "").split("\n")[0][:100]
            lines.append(
                f"Step {step.iteration} (Compacted): Action `{step.action}({args_brief})` -> Outcome: {obs_brief}..."
            )
        else:
            # Full structured detail for recent steps
            args_str = json.dumps(step.arguments)
            lines.append(f"Step {step.iteration}:")
            lines.append(f"  Action: {step.action}")
            lines.append(f"  Arguments: {args_str}")
            lines.append(f"  Observation:\n    {step.observation}")

    return "\n\n".join(lines)


def build_agent_system_prompt(available_tools: List[dict]) -> str:
    """Construct deterministic system instructions detailing tool schema and action protocol."""
    tools_desc = []
    for t in available_tools:
        tools_desc.append(
            f"- Tool: `{t['name']}`\n"
            f"  Description: {t['description']}\n"
            f"  Parameters Schema: {json.dumps(t['parameters'])}"
        )
    tools_block = "\n".join(tools_desc)

    return (
        "You are an autonomous, factually grounded AI research assistant. "
        "Your task is to answer the user's question by performing iterative multi-step reasoning, "
        "dynamically selecting tools, observing intermediate evidence, and synthesizing a verified answer.\n\n"
        "### Available Actions / Tools:\n"
        f"{tools_block}\n\n"
        "- Action: `final_answer`\n"
        "  Description: Output the final answer when you have sufficient verified evidence.\n"
        "  Parameters: {\"answer\": \"string\", \"topic\": \"string\"}\n\n"
        "- Action: `ask_user_clarification`\n"
        "  Description: Request clarification if the user prompt is underspecified or ambiguous.\n"
        "  Parameters: {\"question\": \"string\"}\n\n"
        "### Operational Guidelines:\n"
        "1. Dynamic Retrieval: If a document search returns insufficient or partial facts, reformulate your query and search again.\n"
        "2. Tool Chaining: Use the calculator to perform mathematical computations on retrieved numbers.\n"
        "3. Intermediate Evaluation: Review observations carefully before deciding the next step.\n"
        "4. No Hallucinations: If documents do not contain the answer, state so clearly in your final answer.\n"
        "5. Output Protocol: At each step, output EXACTLY ONE JSON action object with 'action' and 'arguments' (or 'answer' and 'topic' for final_answer).\n"
        "Do NOT output markdown commentary outside the JSON object."
    )


def build_agent_iteration_prompt(state: AgentState) -> str:
    """Construct the complete prompt payload for the next LLM iteration."""
    scratchpad = build_compacted_scratchpad(state.trajectory)

    return (
        f"User Question:\n{state.user_question}\n\n"
        f"Execution History & Observations (Iteration {state.iteration}):\n"
        f"{scratchpad}\n\n"
        f"Instruction for Next Step (Iteration {state.iteration + 1}):\n"
        "Based on the execution history and observations above, choose your next action. "
        "If you have gathered all necessary information, use `final_answer`. "
        "If you need more evidence, call `rag_search` with a refined query or another tool.\n"
        "Output ONLY a valid JSON object matching the chosen action schema."
    )
