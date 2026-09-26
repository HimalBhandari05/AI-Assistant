"""Configuration and Prompt Version Registry for MLOps Tracking and Experimentation.

Defines standardized, reproducible configuration versions across:
- Prompt Templates & System Instructions
- LLM Hyperparameters (temperature, sampling)
- Agent Controller Parameters (iteration budget, timeout)
- RAG Pipeline Parameters (chunk size, overlap, top-k)
- Context Engineering Limits (chunk character caps, scratchpad compaction)
"""

import os
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class AgentConfigVersion(BaseModel):
    """Immutable specification for a tracked agent configuration version."""
    version_id: str = Field(..., description="Unique version identifier (e.g. 'v1_baseline').")
    prompt_version: str = Field(..., description="Prompt template version tag (e.g. 'w16_react_v1').")
    description: str = Field(..., description="Rationale and operational characteristics of this version.")
    llm_provider: str = Field(default="gemini", description="Underlying model provider.")
    model: str = Field(default="gemini-2.5-flash", description="Model name identifier.")
    embedding_model: str = Field(default="text-embedding-004", description="Embedding model name.")
    temperature: float = Field(default=0.1, description="Sampling temperature.")
    max_iterations: int = Field(default=5, description="Maximum reasoning iterations allowed.")
    agent_timeout_seconds: float = Field(default=30.0, description="Hard execution timeout ceiling in seconds.")
    chunk_size: int = Field(default=1000, description="RAG document chunk size in characters.")
    chunk_overlap: int = Field(default=100, description="RAG document chunk overlap.")
    rag_top_k: int = Field(default=3, description="Number of vector chunks retrieved per search.")
    max_chunk_chars: int = Field(default=400, description="Character budget per individual chunk observation.")
    max_observation_chars: int = Field(default=1200, description="Maximum character budget per tool observation.")
    max_detailed_steps: int = Field(default=2, description="Number of recent trajectory steps to retain in detail.")
    fallback_enabled: bool = Field(default=True, description="Whether transient error fallback is active.")
    fallback_provider: str = Field(default="ollama", description="Fallback provider identifier.")

    def to_mlflow_params(self) -> Dict[str, Any]:
        """Convert configuration fields to a flat dictionary for MLflow parameter logging."""
        return {
            "version_id": self.version_id,
            "prompt_version": self.prompt_version,
            "description": self.description,
            "llm_provider": self.llm_provider,
            "model": self.model,
            "embedding_model": self.embedding_model,
            "temperature": self.temperature,
            "max_iterations": self.max_iterations,
            "agent_timeout_seconds": self.agent_timeout_seconds,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "top_k": self.rag_top_k,
            "max_chunk_chars": self.max_chunk_chars,
            "max_observation_chars": self.max_observation_chars,
            "max_detailed_steps": self.max_detailed_steps,
            "fallback_enabled": self.fallback_enabled,
            "fallback_provider": self.fallback_provider,
        }


# -----------------------------------------------------------------------------
# 1. Version 1: Baseline W16 ReAct Agent
# -----------------------------------------------------------------------------
VERSION_1_BASELINE = AgentConfigVersion(
    version_id="v1_baseline",
    prompt_version="w16_react_v1",
    description="Baseline W16 ReAct configuration with standard system prompt, greedy temperature (0.1), 5 max iterations, and chunk size 1000.",
    temperature=0.1,
    max_iterations=5,
    agent_timeout_seconds=30.0,
    chunk_size=1000,
    chunk_overlap=100,
    rag_top_k=3,
    max_chunk_chars=400,
    max_observation_chars=1200,
    max_detailed_steps=2,
)

# -----------------------------------------------------------------------------
# 2. Version 2: Compact Precision & Granular Retrieval
# -----------------------------------------------------------------------------
VERSION_2_COMPACT_PRECISION = AgentConfigVersion(
    version_id="v2_compact_precision",
    prompt_version="w17_compact_precision_v2",
    description="Optimized precision configuration with concise prompt, deterministic temperature (0.0), reduced iteration budget (4), granular chunking (800), and top-k 4.",
    temperature=0.0,
    max_iterations=4,
    agent_timeout_seconds=25.0,
    chunk_size=800,
    chunk_overlap=80,
    rag_top_k=4,
    max_chunk_chars=350,
    max_observation_chars=1000,
    max_detailed_steps=2,
)

# -----------------------------------------------------------------------------
# 3. Version 3: Deep Validation & Multi-Hop Exploration
# -----------------------------------------------------------------------------
VERSION_3_DEEP_VALIDATION = AgentConfigVersion(
    version_id="v3_deep_validation",
    prompt_version="w17_deep_validation_v3",
    description="Deep validation configuration with multi-hop verification prompt, exploratory temperature (0.3), expanded iteration budget (6), broad chunking (1200), and top-k 2.",
    temperature=0.3,
    max_iterations=6,
    agent_timeout_seconds=35.0,
    chunk_size=1200,
    chunk_overlap=150,
    rag_top_k=2,
    max_chunk_chars=500,
    max_observation_chars=1500,
    max_detailed_steps=3,
)

# Canonical Registry
CONFIG_VERSIONS: Dict[str, AgentConfigVersion] = {
    "v1_baseline": VERSION_1_BASELINE,
    "v2_compact_precision": VERSION_2_COMPACT_PRECISION,
    "v3_deep_validation": VERSION_3_DEEP_VALIDATION,
}

# Alias Map for user convenience and CLI flexibility
_VERSION_ALIASES: Dict[str, str] = {
    "v1": "v1_baseline",
    "prompt_v1": "v1_baseline",
    "baseline": "v1_baseline",
    "w16_react_v1": "v1_baseline",
    "v2": "v2_compact_precision",
    "prompt_v2": "v2_compact_precision",
    "compact": "v2_compact_precision",
    "w17_compact_precision_v2": "v2_compact_precision",
    "v2_optimized_retrieval": "v2_compact_precision",
    "v3": "v3_deep_validation",
    "prompt_v3": "v3_deep_validation",
    "deep": "v3_deep_validation",
    "w17_deep_validation_v3": "v3_deep_validation",
    "v3_exploratory_resilient": "v3_deep_validation",
}


def get_config_version(version_name: str) -> AgentConfigVersion:
    """Retrieve an AgentConfigVersion by exact ID or common alias."""
    clean_name = str(version_name).strip().lower()
    canonical_id = _VERSION_ALIASES.get(clean_name, clean_name)
    if canonical_id not in CONFIG_VERSIONS:
        available = list(CONFIG_VERSIONS.keys()) + list(_VERSION_ALIASES.keys())
        raise ValueError(
            f"Unknown configuration version '{version_name}'. Available versions: {available}"
        )
    return CONFIG_VERSIONS[canonical_id]


def list_config_versions() -> List[AgentConfigVersion]:
    """Return list of all registered canonical configuration versions."""
    return list(CONFIG_VERSIONS.values())
