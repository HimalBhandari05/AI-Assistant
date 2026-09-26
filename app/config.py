"""Centralized Configuration & Settings Management for AI Assistant and MLOps.

Provides strongly-typed, environment-aware configuration for:
- Prompts & Prompt Versioning (for MLflow tracking)
- LLM Providers & Models (Gemini, Ollama, vLLM)
- Model Hyperparameters (temperature, max retries, timeout)
- Agent Controller Parameters (max iterations, timeout, scratchpad limits)
- RAG Pipeline (chunk size, overlap, top-k, ChromaDB path)
- Caching & Rate Limiting Guardrails
"""

import os
from typing import Any, Dict, Optional
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Unified application settings with environment variable loading."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # 1. Prompt Versioning & Configuration
    prompt_version: str = Field(default="w16_react_v1", description="Prompt template version identifier for MLflow tracking")

    # 2. LLM Provider & Model Selection
    llm_provider: str = Field(default="gemini", alias="LLM_PROVIDER")
    gemini_api_key: Optional[str] = Field(default=None, alias="GEMINI_API_KEY")
    gemini_model: str = Field(default="gemini-2.5-flash", alias="GEMINI_MODEL")
    gemini_embedding_model: str = Field(default="text-embedding-004", alias="GEMINI_EMBEDDING_MODEL")

    ollama_base_url: str = Field(default="http://localhost:11434", alias="OLLAMA_BASE_URL")
    ollama_model: str = Field(default="qwen2.5:3b", alias="OLLAMA_MODEL")
    ollama_timeout: float = Field(default=120.0, alias="OLLAMA_TIMEOUT")

    vllm_base_url: str = Field(default="http://localhost:8001/v1", alias="VLLM_BASE_URL")
    vllm_model: str = Field(default="Qwen/Qwen2.5-3B-Instruct", alias="VLLM_MODEL")
    vllm_timeout: float = Field(default=120.0, alias="VLLM_TIMEOUT")

    # 3. Model Hyperparameters
    agent_temperature: float = Field(default=0.1, alias="AGENT_TEMPERATURE")
    llm_temperature: float = Field(default=0.2, alias="LLM_TEMPERATURE")
    max_retries: int = Field(default=2, alias="MAX_RETRIES")
    retry_base_delay: float = Field(default=0.5, alias="RETRY_BASE_DELAY")
    fallback_enabled: bool = Field(default=True, alias="FALLBACK_ENABLED")
    fallback_provider: str = Field(default="ollama", alias="FALLBACK_PROVIDER")

    # 4. Agent Controller Configuration
    agent_max_iterations: int = Field(default=5, alias="AGENT_MAX_ITERATIONS")
    agent_timeout_seconds: float = Field(default=30.0, alias="AGENT_TIMEOUT_SECONDS")
    use_agent: bool = Field(default=True, alias="USE_AGENT")

    # 5. RAG & Vector Store Configuration
    chunk_size: int = Field(default=1000, alias="CHUNK_SIZE")
    chunk_overlap: int = Field(default=100, alias="CHUNK_OVERLAP")
    rag_top_k: int = Field(default=3, alias="RAG_TOP_K")
    chroma_path: str = Field(default="data/chroma", alias="CHROMA_PATH")
    collection_name: str = Field(default="documents", alias="COLLECTION_NAME")

    # 6. Context Engineering & Scratchpad Budgets
    max_chunk_chars: int = Field(default=400, alias="MAX_CHUNK_CHARS")
    max_observation_chars: int = Field(default=1200, alias="MAX_OBSERVATION_CHARS")
    max_detailed_steps: int = Field(default=2, alias="MAX_DETAILED_STEPS")

    # 7. Cache Configuration
    cache_enabled: bool = Field(default=True, alias="CACHE_ENABLED")
    cache_ttl_seconds: float = Field(default=300.0, alias="CACHE_TTL_SECONDS")
    cache_max_size: int = Field(default=100, alias="CACHE_MAX_SIZE")

    # 8. Rate Limiting Configuration
    rate_limit_requests: int = Field(default=10, alias="RATE_LIMIT_REQUESTS")
    rate_limit_window_seconds: float = Field(default=60.0, alias="RATE_LIMIT_WINDOW_SECONDS")

    def get_mlflow_tracking_params(self) -> Dict[str, Any]:
        """Export parameters dictionary structured for MLflow logging."""
        return {
            "prompt_version": self.prompt_version,
            "llm_provider": self.llm_provider,
            "gemini_model": self.gemini_model,
            "gemini_embedding_model": self.gemini_embedding_model,
            "agent_temperature": self.agent_temperature,
            "llm_temperature": self.llm_temperature,
            "agent_max_iterations": self.agent_max_iterations,
            "agent_timeout_seconds": self.agent_timeout_seconds,
            "rag_chunk_size": self.chunk_size,
            "rag_chunk_overlap": self.chunk_overlap,
            "rag_top_k": self.rag_top_k,
            "chroma_path": self.chroma_path,
            "fallback_enabled": self.fallback_enabled,
            "fallback_provider": self.fallback_provider,
            "max_retries": self.max_retries,
        }


def get_settings() -> Settings:
    """Instantiate and return fresh Settings loaded from environment variables and .env."""
    return Settings()


# Default singleton instance
settings = get_settings()
