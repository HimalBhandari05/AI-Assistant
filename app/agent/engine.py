"""Agent Execution Controller: Bounded ReAct Loop, Dynamic Tool Selection, and Safety Guardrails."""

import json
import logging
import os
import time
from typing import Any, Dict, Optional, Tuple
from google import genai
from google.genai import types
from google.genai.errors import APIError
import httpx
from pydantic import BaseModel, Field

from app.agent.context import (
    AgentState,
    TrajectoryStep,
    TokenUsage,
    build_agent_system_prompt,
    build_agent_iteration_prompt,
    truncate_observation,
)
from app.agent.metrics import AgentExecutionMetrics
from app.agent.tools import ToolRegistry, default_tool_registry
from app.llm import execute_with_retry, is_transient_error

logger = logging.getLogger("ai_assistant.agent.engine")

DEFAULT_MAX_ITERATIONS = int(os.getenv("AGENT_MAX_ITERATIONS", "5"))
DEFAULT_MAX_EXECUTION_TIME = float(os.getenv("AGENT_TIMEOUT_SECONDS", "30.0"))


class AgentResult(BaseModel):
    """Standardized result returned by the agent execution engine."""
    status: str = Field(..., description="Execution status: 'completed', 'clarification_required', 'max_iterations_exceeded', 'timeout', 'error'.")
    answer: str = Field(..., description="The synthesized final answer or clarification prompt.")
    topic: str = Field(default="General", description="The classified topic.")
    trajectory_length: int = Field(default=0, description="Total number of reasoning/action steps executed.")
    trajectory: list = Field(default_factory=list, description="List of TrajectoryStep objects.")
    clarification_required: bool = Field(default=False, description="True if clarification from the user is required.")
    token_usage: TokenUsage = Field(default_factory=TokenUsage, description="Aggregated token usage across iterations.")
    metrics: Optional[AgentExecutionMetrics] = Field(default=None, description="Observability telemetry.")


def parse_model_action_json(raw_text: str) -> dict:
    """Safely parse structured JSON action dictionary from model response string."""
    if not raw_text or not raw_text.strip():
        raise ValueError("Model returned an empty response.")

    clean_text = raw_text.strip()
    if clean_text.startswith("```json"):
        clean_text = clean_text[7:]
    elif clean_text.startswith("```"):
        clean_text = clean_text[3:]
    if clean_text.endswith("```"):
        clean_text = clean_text[:-3]
    clean_text = clean_text.strip()

    try:
        data = json.loads(clean_text)
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError as exc:
        # Attempt to find JSON object substring
        start_idx = clean_text.find("{")
        end_idx = clean_text.rfind("}")
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            sub = clean_text[start_idx : end_idx + 1]
            try:
                data = json.loads(sub)
                if isinstance(data, dict):
                    return data
            except Exception:
                pass
        raise ValueError(f"Failed to parse JSON action from model output: {exc}. Raw text: {raw_text[:200]}") from exc

    raise ValueError(f"Model output did not contain a valid JSON object. Received: {clean_text[:200]}")


def _query_provider_for_action(
    system_prompt: str,
    iteration_prompt: str,
    provider: str = "gemini",
) -> Tuple[dict, int, int]:
    """Dispatch prompt to configured LLM provider and return (action_dict, prompt_tokens, completion_tokens)."""
    norm_provider = (provider or "gemini").lower().strip()

    if norm_provider == "gemini":
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key or not api_key.strip() or api_key.strip() in ("your_api_key", "your_gemini_api_key", "your_gemini_api_key_here"):
            raise ValueError("GEMINI_API_KEY is not configured. Please set a valid API key in your .env file.")

        model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        client = genai.Client(api_key=api_key)

        config = types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=0.1,
            response_mime_type="application/json",
        )

        contents = [types.Content(role="user", parts=[types.Part.from_text(text=iteration_prompt)])]

        try:
            response = client.models.generate_content(
                model=model,
                contents=contents,
                config=config,
            )
        except APIError as exc:
            raise RuntimeError(f"Gemini API request failed: {exc.message or str(exc)}") from exc
        except Exception as exc:
            raise RuntimeError(f"Gemini service error: {str(exc)}") from exc

        raw_text = response.text or ""
        action_dict = parse_model_action_json(raw_text)

        # Extract token usage if reported
        p_tokens = 0
        c_tokens = 0
        if getattr(response, "usage_metadata", None):
            p_tokens = getattr(response.usage_metadata, "prompt_token_count", 0) or 0
            c_tokens = getattr(response.usage_metadata, "candidates_token_count", 0) or 0

        return action_dict, p_tokens, c_tokens

    elif norm_provider == "ollama":
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
        model = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")
        timeout = float(os.getenv("OLLAMA_TIMEOUT", "120.0"))

        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": iteration_prompt},
            ],
            "options": {"temperature": 0.1},
            "stream": False,
        }

        with httpx.Client(timeout=timeout) as client:
            resp = client.post(f"{base_url}/api/chat", json=payload)
            resp.raise_for_status()
            data = resp.json()

        raw_content = data.get("message", {}).get("content", "")
        action_dict = parse_model_action_json(raw_content)

        p_tokens = data.get("prompt_eval_count", 0) or 0
        c_tokens = data.get("eval_count", 0) or 0
        return action_dict, p_tokens, c_tokens

    elif norm_provider == "vllm":
        base_url = os.getenv("VLLM_BASE_URL", "http://localhost:8001/v1").rstrip("/")
        model = os.getenv("VLLM_MODEL", "Qwen/Qwen2.5-3B-Instruct")
        timeout = float(os.getenv("VLLM_TIMEOUT", "120.0"))

        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": iteration_prompt},
            ],
            "temperature": 0.1,
            "response_format": {"type": "json_object"},
        }

        with httpx.Client(timeout=timeout) as client:
            resp = client.post(f"{base_url}/chat/completions", json=payload)
            resp.raise_for_status()
            data = resp.json()

        choices = data.get("choices", [])
        if not choices or "message" not in choices[0]:
            raise RuntimeError("Invalid response structure from vLLM.")

        raw_content = choices[0]["message"].get("content", "")
        action_dict = parse_model_action_json(raw_content)

        usage = data.get("usage", {})
        p_tokens = usage.get("prompt_tokens", 0) or 0
        c_tokens = usage.get("completion_tokens", 0) or 0
        return action_dict, p_tokens, c_tokens

    else:
        raise ValueError(f"Unsupported LLM_PROVIDER '{provider}'.")


def run_agent_workflow(
    question: str,
    tool_registry: Optional[ToolRegistry] = None,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
    max_execution_time_seconds: float = DEFAULT_MAX_EXECUTION_TIME,
    provider: Optional[str] = None,
) -> AgentResult:
    """Execute the multi-step iterative research and verification agent loop.

    Args:
        question: The user input query to solve.
        tool_registry: ToolRegistry instance (defaults to default_tool_registry).
        max_iterations: Maximum allowed reasoning steps (default: 5).
        max_execution_time_seconds: Timeout ceiling in seconds (default: 30.0).
        provider: Overriding LLM provider (defaults to os.getenv('LLM_PROVIDER', 'gemini')).

    Returns:
        Structured AgentResult containing final answer, trajectory, metrics, and token usage.
    """
    start_time = time.perf_counter()
    reg = tool_registry or default_tool_registry
    raw_prov = provider or os.getenv("LLM_PROVIDER", "gemini")
    active_provider = (raw_prov or "gemini").lower().strip()

    available_tools = reg.list_tools()
    system_prompt = build_agent_system_prompt(available_tools)

    state = AgentState(
        user_question=question,
        iteration=0,
        available_tools=available_tools,
    )
    metrics = AgentExecutionMetrics(question=question)

    # Track recent action signatures for duplicate-loop detection
    seen_action_signatures: Dict[str, int] = {}

    while state.iteration < max_iterations:
        # Check execution timeout
        elapsed_so_far = time.perf_counter() - start_time
        if elapsed_so_far > max_execution_time_seconds:
            logger.warning(f"Agent execution timed out after {elapsed_so_far:.2f}s (Max: {max_execution_time_seconds}s)")
            metrics.record_failure("ExecutionTimeout", state.iteration, {"elapsed": elapsed_so_far})
            state.status = "timeout"
            return AgentResult(
                status="timeout",
                answer="The request could not be completed within the allocated time limit.",
                topic="System",
                trajectory_length=len(state.trajectory),
                trajectory=state.trajectory,
                token_usage=state.token_usage,
                metrics=metrics,
            )

        state.iteration += 1
        iter_prompt = build_agent_iteration_prompt(state)

        # 1. Ask LLM for next action with retry/fallback resilience
        try:
            action_dict, prompt_toks, comp_toks = execute_with_retry(
                _query_provider_for_action,
                system_prompt=system_prompt,
                iteration_prompt=iter_prompt,
                provider=active_provider,
            )
            state.token_usage.add(prompt_toks, comp_toks)
        except Exception as exc:
            # Check fallback if Gemini failed on eligible transient error
            fallback_enabled = os.getenv("FALLBACK_ENABLED", "true").lower().strip() in ("true", "1", "yes")
            if active_provider == "gemini" and fallback_enabled and is_transient_error(exc):
                fallback_prov = os.getenv("FALLBACK_PROVIDER", "ollama").lower().strip()
                logger.warning(f"Primary provider Gemini failed on step {state.iteration}: {exc}. Trying fallback: {fallback_prov}")
                try:
                    action_dict, prompt_toks, comp_toks = _query_provider_for_action(
                        system_prompt=system_prompt,
                        iteration_prompt=iter_prompt,
                        provider=fallback_prov,
                    )
                    state.token_usage.add(prompt_toks, comp_toks)
                except Exception as fb_exc:
                    metrics.record_failure("ProviderError", state.iteration, {"error": str(fb_exc)})
                    raise RuntimeError(f"Both primary and fallback providers failed during agent loop: {fb_exc}") from fb_exc
            else:
                metrics.record_failure("ProviderError", state.iteration, {"error": str(exc)})
                raise exc

        # 2. Inspect decided action
        action_name = action_dict.get("action", "").strip()

        # Direct final_answer shorthand: if dict has 'answer' directly or action is final_answer
        if action_name == "final_answer" or ("answer" in action_dict and "action" not in action_dict):
            final_ans = action_dict.get("answer") or action_dict.get("arguments", {}).get("answer", "")
            topic_str = action_dict.get("topic") or action_dict.get("arguments", {}).get("topic", "General")

            state.status = "completed"
            state.final_answer = str(final_ans)
            state.topic = str(topic_str)

            # Record final completion step
            step_record = TrajectoryStep(
                iteration=state.iteration,
                action="final_answer",
                arguments={"answer": final_ans, "topic": topic_str},
                observation="Final answer generated.",
                success=True,
                execution_time_seconds=round(time.perf_counter() - start_time, 4),
            )
            state.trajectory.append(step_record)

            metrics.status = "completed"
            metrics.trajectory_length = len(state.trajectory)
            metrics.total_execution_time_seconds = round(time.perf_counter() - start_time, 4)
            metrics.token_usage = state.token_usage

            return AgentResult(
                status="completed",
                answer=state.final_answer,
                topic=state.topic,
                trajectory_length=len(state.trajectory),
                trajectory=state.trajectory,
                clarification_required=False,
                token_usage=state.token_usage,
                metrics=metrics,
            )

        # Clarification action
        if action_name == "ask_user_clarification":
            clarify_q = (
                action_dict.get("question")
                or action_dict.get("arguments", {}).get("question", "")
            )
            state.status = "clarification_required"
            state.clarification_required = True
            state.clarification_question = str(clarify_q)

            step_record = TrajectoryStep(
                iteration=state.iteration,
                action="ask_user_clarification",
                arguments={"question": clarify_q},
                observation=f"[Clarification Prompt]: {clarify_q}",
                success=True,
                execution_time_seconds=round(time.perf_counter() - start_time, 4),
            )
            state.trajectory.append(step_record)

            metrics.status = "clarification_required"
            metrics.trajectory_length = len(state.trajectory)
            metrics.total_execution_time_seconds = round(time.perf_counter() - start_time, 4)
            metrics.token_usage = state.token_usage

            return AgentResult(
                status="clarification_required",
                answer=state.clarification_question,
                topic="Clarification",
                trajectory_length=len(state.trajectory),
                trajectory=state.trajectory,
                clarification_required=True,
                token_usage=state.token_usage,
                metrics=metrics,
            )

        # 3. Handle Tool Dispatch
        action_args = action_dict.get("arguments", {})
        if not isinstance(action_args, dict):
            action_args = {}

        # Duplicate Action Loop Detection
        action_sig = f"{action_name}:{json.dumps(action_args, sort_keys=True)}"
        seen_action_signatures[action_sig] = seen_action_signatures.get(action_sig, 0) + 1

        if seen_action_signatures[action_sig] > 2:
            # Break infinite loop if same action executed 3 times
            logger.warning(f"Loop detected for action '{action_sig}'. Terminating gracefully.")
            metrics.record_failure("DuplicateActionError", state.iteration, {"action_sig": action_sig})
            state.status = "max_iterations_exceeded"
            break

        # Execute tool via Registry
        tool_res = reg.execute(action_name, action_args)

        # Record tool metrics
        err_dict = tool_res.error.model_dump() if tool_res.error else None
        metrics.record_tool_call(
            tool_name=action_name,
            arguments_valid=bool(tool_res.error is None or tool_res.error.type != "InvalidArgumentsError"),
            success=tool_res.success,
            execution_time_seconds=tool_res.execution_time_seconds,
            error_type=tool_res.error.type if tool_res.error else None,
            error_message=tool_res.error.message if tool_res.error else None,
        )

        # Create structured bounded observation
        observation_str = truncate_observation(
            tool_name=action_name,
            raw_result=tool_res.result,
            error=err_dict,
        )

        # Append to trajectory
        step = TrajectoryStep(
            iteration=state.iteration,
            action=action_name,
            arguments=action_args,
            observation=observation_str,
            success=tool_res.success,
            execution_time_seconds=tool_res.execution_time_seconds,
        )
        state.trajectory.append(step)

    # If loop terminates without final_answer or clarification -> max iterations exceeded
    logger.warning(f"Agent reached maximum iterations ({max_iterations}) for question: '{question}'")
    metrics.record_failure("MaxIterationsExceeded", max_iterations, {"question": question})
    state.status = "max_iterations_exceeded"
    metrics.status = "max_iterations_exceeded"
    metrics.trajectory_length = len(state.trajectory)
    metrics.total_execution_time_seconds = round(time.perf_counter() - start_time, 4)
    metrics.token_usage = state.token_usage

    return AgentResult(
        status="max_iterations_exceeded",
        answer="I reached the maximum reasoning iterations without being able to fully resolve your request. Please try refining your question.",
        topic="General",
        trajectory_length=len(state.trajectory),
        trajectory=state.trajectory,
        clarification_required=False,
        token_usage=state.token_usage,
        metrics=metrics,
    )
