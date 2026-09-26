"""Unit tests for centralized configuration and MLOps tracking parameter export."""

import os
import sys
import pytest

# Ensure project root in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.config import Settings, get_settings


def test_default_settings_initialization():
    """Verify default configuration values."""
    s = Settings()
    assert s.prompt_version == "w16_react_v1"
    assert s.gemini_model in ("gemini-2.5-flash", os.getenv("GEMINI_MODEL", "gemini-2.5-flash"))
    assert s.agent_max_iterations == 5
    assert s.rag_top_k == 3
    assert s.chunk_size == 1000
    assert s.agent_temperature == 0.1
    assert s.llm_temperature == 0.2
    assert s.max_chunk_chars == 400
    assert s.max_observation_chars == 1200
    assert s.max_detailed_steps == 2


def test_settings_environment_override(monkeypatch):
    """Verify that environment variables dynamically override defaults."""
    monkeypatch.setenv("AGENT_MAX_ITERATIONS", "10")
    monkeypatch.setenv("AGENT_TEMPERATURE", "0.7")
    monkeypatch.setenv("RAG_TOP_K", "5")
    monkeypatch.setenv("CHUNK_SIZE", "500")
    monkeypatch.setenv("PROMPT_VERSION", "v2_experimental")

    s = get_settings()
    assert s.agent_max_iterations == 10
    assert s.agent_temperature == 0.7
    assert s.rag_top_k == 5
    assert s.chunk_size == 500
    assert s.prompt_version == "v2_experimental"


def test_mlflow_tracking_params_export():
    """Verify export of configuration parameters for MLflow experiment tracking."""
    s = Settings()
    params = s.get_mlflow_tracking_params()

    expected_keys = [
        "prompt_version",
        "llm_provider",
        "gemini_model",
        "gemini_embedding_model",
        "agent_temperature",
        "llm_temperature",
        "agent_max_iterations",
        "agent_timeout_seconds",
        "rag_chunk_size",
        "rag_chunk_overlap",
        "rag_top_k",
        "chroma_path",
        "fallback_enabled",
        "fallback_provider",
        "max_retries",
    ]

    for key in expected_keys:
        assert key in params, f"Missing expected MLflow tracking parameter: {key}"
        assert params[key] is not None
