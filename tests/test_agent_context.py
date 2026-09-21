"""Unit tests for Phase 2 Context Engineering, Trajectory, and Scratchpad Compaction."""

import os
import sys
import pytest

# Ensure project root in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.agent.context import (
    AgentState,
    TrajectoryStep,
    TokenUsage,
    truncate_observation,
    build_compacted_scratchpad,
    build_agent_system_prompt,
    build_agent_iteration_prompt,
    MAX_CHUNK_CHARS,
    MAX_OBSERVATION_CHARS,
)


def test_agent_state_initialization():
    """Verify clean AgentState initialization."""
    state = AgentState(user_question="What is binary search?")
    assert state.user_question == "What is binary search?"
    assert state.iteration == 0
    assert len(state.trajectory) == 0
    assert state.status == "in_progress"
    assert state.token_usage.total_tokens == 0


def test_token_usage_accumulation():
    """Verify TokenUsage addition and aggregation."""
    usage = TokenUsage()
    usage.add(100, 50)
    assert usage.prompt_tokens == 100
    assert usage.completion_tokens == 50
    assert usage.total_tokens == 150

    usage.add(200, 75)
    assert usage.prompt_tokens == 300
    assert usage.completion_tokens == 125
    assert usage.total_tokens == 425


def test_truncate_observation_rag():
    """Verify observation truncation bounds large chunk content and chunks count."""
    long_text = "A" * 1000  # 1000 chars, exceeds MAX_CHUNK_CHARS (400)
    raw_rag = {
        "query": "search query",
        "results": [
            {"source": "doc1.txt", "chunk_id": 1, "text": long_text},
            {"source": "doc2.txt", "chunk_id": 2, "text": "Normal short chunk"},
        ],
        "count": 2,
    }

    obs = truncate_observation("rag_search", raw_rag)
    assert "[RAG Search Results for 'search query' (2 chunks found)]:" in obs
    assert "... [truncated]" in obs
    assert len(obs) <= MAX_OBSERVATION_CHARS + 100


def test_truncate_observation_calculator():
    """Verify calculator observation formatting."""
    raw_calc = {"a": 25.0, "b": 18.0, "operation": "multiply", "result": 450.0}
    obs = truncate_observation("calculator", raw_calc)
    assert obs == "[Calculator Result]: 25.0 multiply 18.0 = 450.0"


def test_truncate_observation_error():
    """Verify error observation formatting."""
    err = {"type": "ValueError", "message": "Division by zero"}
    obs = truncate_observation("calculator", None, error=err)
    assert "[Tool Error (calculator)]: ValueError - Division by zero" == obs


def test_scratchpad_compaction():
    """Verify that older steps are compacted into single-line summaries while recent steps are detailed."""
    steps = [
        TrajectoryStep(iteration=1, action="rag_search", arguments={"query": "step 1"}, observation="Obs 1 line 1\nObs 1 line 2"),
        TrajectoryStep(iteration=2, action="rag_search", arguments={"query": "step 2"}, observation="Obs 2 line 1\nObs 2 line 2"),
        TrajectoryStep(iteration=3, action="calculator", arguments={"a": 1, "b": 2, "operation": "add"}, observation="Obs 3"),
        TrajectoryStep(iteration=4, action="rag_search", arguments={"query": "step 4"}, observation="Obs 4 detailed"),
    ]

    scratchpad = build_compacted_scratchpad(steps)

    # Steps 1 and 2 (older than last 2 steps) should be compacted
    assert "Step 1 (Compacted):" in scratchpad
    assert "Step 2 (Compacted):" in scratchpad

    # Steps 3 and 4 (last 2 steps) should have full detail
    assert "Step 3:" in scratchpad
    assert "  Action: calculator" in scratchpad
    assert "Step 4:" in scratchpad
    assert "  Action: rag_search" in scratchpad


def test_prompt_builders():
    """Verify agent system prompt and iteration prompt generation."""
    tools = [
        {"name": "rag_search", "description": "Search docs", "parameters": {"type": "object"}},
        {"name": "calculator", "description": "Math calc", "parameters": {"type": "object"}},
    ]
    sys_prompt = build_agent_system_prompt(tools)
    assert "Tool: `rag_search`" in sys_prompt
    assert "Tool: `calculator`" in sys_prompt
    assert "Action: `final_answer`" in sys_prompt

    state = AgentState(user_question="Compare X and Y", iteration=0, available_tools=tools)
    iter_prompt = build_agent_iteration_prompt(state)
    assert "User Question:\nCompare X and Y" in iter_prompt
    assert "Instruction for Next Step (Iteration 1):" in iter_prompt
