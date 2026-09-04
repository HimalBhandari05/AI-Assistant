import asyncio
import logging
import os
import time
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request, status
import httpx
from pydantic import BaseModel, Field, field_validator

from app.cache import response_cache
from app.llm import AssistantResponse, get_llm_response
from app.rate_limiter import rate_limiter

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ai_assistant")

app = FastAPI(
    title="AI Assistant API",
    description="Production-grade AI Assistant Backend with RAG, Multi-Provider LLM, Fallback, Caching, and Rate Limiting",
    version="2.7.0",
)


@app.middleware("http")
async def measure_request_latency(request: Request, call_next):
    """Middleware to measure, record, and header-tag request latency."""
    start_time = time.perf_counter()
    response = await call_next(request)
    process_time = time.perf_counter() - start_time

    response.headers["X-Process-Time"] = f"{process_time:.4f}s"
    logger.info(
        f"{request.method} {request.url.path} completed in {process_time:.4f}s (HTTP {response.status_code})"
    )
    return response


class QuestionRequest(BaseModel):
    question: str = Field(..., description="The question to ask the AI assistant.")

    @field_validator("question")
    @classmethod
    def validate_question_not_empty(cls, value: str) -> str:
        trimmed = value.strip()
        if not trimmed:
            raise ValueError("Question cannot be empty or contain only whitespace.")
        return trimmed


@app.get("/")
async def read_root():
    """Health check endpoint (not rate-limited)."""
    return {"status": "ok", "message": "AI Assistant API is running"}


@app.post("/ask", response_model=AssistantResponse)
async def ask_question(request: Request, payload: QuestionRequest):
    """Ask a question to the AI Assistant asynchronously with rate limiting, response caching, and bounded retries."""
    # 1. Enforce in-memory rate limiting before cache lookup or expensive operations
    await rate_limiter.check_rate_limit(request)

    # 2. Check response cache (skips RAG, embeddings, tools, and LLM on hit)
    raw_provider = os.getenv("LLM_PROVIDER", "gemini")
    provider = (raw_provider or "gemini").lower().strip()
    cached_response = await response_cache.get(provider, payload.question)
    if cached_response is not None:
        return cached_response

    # 3. Proceed with LLM / RAG processing on cache miss
    try:
        # Offload synchronous blocking operations (ChromaDB + LLM API + retries + fallback) to worker thread
        response = await asyncio.to_thread(get_llm_response, payload.question)
        # 4. Store successful response in cache
        await response_cache.set(provider, payload.question, response)
        return response
    except ValueError as exc:
        err_str = str(exc)
        if "GEMINI_API_KEY" in err_str or "API_KEY" in err_str:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="AI provider API key is not configured. Please configure it in your .env file.",
            )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=err_str,
        )
    except Exception as exc:
        err_str = str(exc)
        err_lower = err_str.lower()
        logger.error(f"Exception processing request: {err_str}")

        if any(term in err_lower for term in ["quota", "rate_limit", "rate limit", "resource_exhausted", "429", "too many requests"]):
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="AI service quota or rate limit exceeded. Please check your account quota or try again later.",
            )
        if (
            isinstance(exc, (httpx.ConnectError, httpx.TimeoutException, ConnectionError, TimeoutError))
            or any(term in err_lower for term in ["unavailable", "connection", "timeout", "timed out", "connect", "both failed", "service unavailable"])
        ):
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="The AI service is temporarily unavailable. Please try again later.",
            )
        if "401" in err_str or "unauthorized" in err_lower or "forbidden" in err_lower:
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="Upstream authentication or access error.",
            )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred while processing your request.",
        )
