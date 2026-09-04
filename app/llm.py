import json
import logging
import os
import time
from typing import Any, Callable
from google import genai
from google.genai import types
from google.genai.errors import APIError
import httpx
from pydantic import BaseModel, Field, ValidationError

from app.prompts import SYSTEM_PROMPT
from app.rag.retrieve import retrieve_relevant_chunks
from app.tools import calculator

logger = logging.getLogger("ai_assistant.llm")

MAX_RETRIES = 2  # At most 3 total attempts
BASE_BACKOFF_DELAY = float(os.getenv("RETRY_BASE_DELAY", "0.5"))


class AssistantResponse(BaseModel):
    answer: str = Field(..., description="A concise, factual answer to the question.")
    topic: str = Field(..., description="The classified subject or topic of the question.")


def is_transient_error(exc: Exception) -> bool:
    """Determine whether an exception is transient and safe to retry."""
    # 1. Non-retryable: validation and input errors
    if isinstance(exc, (ValueError, TypeError, ValidationError)):
        return False

    err_msg = str(exc).lower()

    # 2. Non-retryable: Authentication, permission, and configuration errors
    if any(term in err_msg for term in ["api_key", "api key", "unauthenticated", "permission_denied", "invalid_argument", "401", "403"]):
        return False

    # 3. Non-retryable: Quota & Rate limits (Gemini Quota Protection)
    if any(term in err_msg for term in ["quota", "rate_limit", "rate limit", "resource_exhausted", "429", "too many requests"]):
        return False

    # 4. Retryable: Network, connection, and timeout exceptions
    if isinstance(
        exc,
        (
            httpx.ConnectError,
            httpx.TimeoutException,
            httpx.ConnectTimeout,
            httpx.ReadTimeout,
            httpx.RemoteProtocolError,
            ConnectionError,
            TimeoutError,
        ),
    ):
        return True

    # 5. Retryable: Upstream HTTP 5xx errors
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in (500, 502, 503, 504)

    if isinstance(exc, APIError):
        code = getattr(exc, "code", None)
        if code in (500, 502, 503, 504):
            return True
        if any(term in err_msg for term in ["unavailable", "503", "502", "504", "deadline_exceeded", "timed out", "timeout"]):
            return True
        return False

    # 6. Generic transient error keywords
    if any(term in err_msg for term in ["connection reset", "connection refused", "timeout", "timed out", "service unavailable", "503", "502"]):
        return True

    return False


def execute_with_retry(
    func: Callable[..., Any],
    *args: Any,
    max_retries: int = MAX_RETRIES,
    base_delay: float = BASE_BACKOFF_DELAY,
    **kwargs: Any,
) -> Any:
    """Execute a callable with bounded exponential backoff for transient failures."""
    last_exception = None

    for attempt in range(max_retries + 1):
        try:
            return func(*args, **kwargs)
        except Exception as exc:
            last_exception = exc
            if attempt < max_retries and is_transient_error(exc):
                delay = base_delay * (2 ** attempt)
                logger.warning(
                    f"Transient error on attempt {attempt + 1}/{max_retries + 1}: {exc}. "
                    f"Retrying in {delay:.2f}s..."
                )
                time.sleep(delay)
            else:
                if not is_transient_error(exc):
                    logger.info(f"Non-retryable error on attempt {attempt + 1}: {exc}")
                raise exc

    if last_exception:
        raise last_exception


def execute_tool_call(name: str, args: dict) -> dict:
    """Execute an explicitly allowed tool and return a structured tool response dictionary."""
    if name != "calculator":
        raise RuntimeError(f"Unknown tool requested by model: {name}")

    try:
        a = float(args["a"])
        b = float(args["b"])
        operation = str(args["operation"])
        result = calculator(a=a, b=b, operation=operation)
        return {"result": result}
    except (KeyError, ValueError, TypeError) as exc:
        return {"error": str(exc)}


def format_rag_prompt(question: str, chunks: list[dict]) -> str:
    """Format retrieved document chunks and user question for the LLM."""
    if chunks:
        context_items = [
            f"[Source: {c['source']} (Chunk {c['chunk_id']})]\n{c['text']}"
            for c in chunks
        ]
        context_text = "\n\n".join(context_items)
        return f"Retrieved Context:\n{context_text}\n\nUser Question:\n{question}"

    return f"Retrieved Context:\nNo relevant document context found.\n\nUser Question:\n{question}"


def _call_gemini(question: str, formatted_user_prompt: str) -> AssistantResponse:
    """Execute inference using Google Gemini provider with tool calling and structured output."""
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key or not api_key.strip() or api_key.strip() in ("your_api_key", "your_gemini_api_key", "your_gemini_api_key_here"):
        raise ValueError("GEMINI_API_KEY is not configured. Please set a valid API key in your .env file.")

    model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    client = genai.Client(api_key=api_key)

    initial_config = types.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT,
        temperature=0.2,
        tools=[calculator],
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
    )

    contents = [
        types.Content(
            role="user",
            parts=[types.Part.from_text(text=formatted_user_prompt)],
        )
    ]

    try:
        response = client.models.generate_content(
            model=model,
            contents=contents,
            config=initial_config,
        )
    except APIError as exc:
        raise RuntimeError(f"Gemini API request failed: {exc.message or str(exc)}") from exc
    except Exception as exc:
        raise RuntimeError(f"Gemini service error: {str(exc)}") from exc

    # Handle Gemini function calling if requested
    if response.function_calls:
        fc = response.function_calls[0]
        tool_response_payload = execute_tool_call(fc.name, fc.args or {})

        contents.append(response.candidates[0].content)
        contents.append(
            types.Content(
                role="user",
                parts=[
                    types.Part.from_function_response(
                        name=fc.name,
                        response=tool_response_payload,
                    )
                ],
            )
        )

        final_config = types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=0.2,
            response_mime_type="application/json",
            response_schema=AssistantResponse,
        )

        try:
            response = client.models.generate_content(
                model=model,
                contents=contents,
                config=final_config,
            )
        except APIError as exc:
            raise RuntimeError(f"Gemini API request failed during tool completion: {exc.message or str(exc)}") from exc
        except Exception as exc:
            raise RuntimeError(f"Gemini service error during tool completion: {str(exc)}") from exc

    raw_text = response.text
    if not raw_text or not raw_text.strip():
        raise RuntimeError("Received an empty response from Gemini.")

    clean_text = raw_text.strip()
    if clean_text.startswith("```json"):
        clean_text = clean_text[7:]
    elif clean_text.startswith("```"):
        clean_text = clean_text[3:]
    if clean_text.endswith("```"):
        clean_text = clean_text[:-3]
    clean_text = clean_text.strip()

    try:
        return AssistantResponse.model_validate_json(clean_text)
    except (ValidationError, Exception):
        try:
            parsed = json.loads(clean_text)
            if isinstance(parsed, dict) and "answer" in parsed:
                return AssistantResponse(
                    answer=str(parsed.get("answer", "")),
                    topic=str(parsed.get("topic", "General")),
                )
        except Exception:
            pass
        return AssistantResponse(answer=clean_text, topic="General")


def _call_ollama(formatted_user_prompt: str) -> AssistantResponse:
    """Execute inference using local Ollama provider."""
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
    model = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": formatted_user_prompt},
        ],
        "format": AssistantResponse.model_json_schema(),
        "options": {
            "temperature": 0.2,
        },
        "stream": False,
    }

    timeout = float(os.getenv("OLLAMA_TIMEOUT", "120.0"))
    try:
        with httpx.Client(timeout=timeout) as client:
            resp = client.post(f"{base_url}/api/chat", json=payload)
            resp.raise_for_status()
            data = resp.json()
    except httpx.ConnectError as exc:
        raise RuntimeError(f"Could not connect to Ollama server at {base_url}. Ensure Ollama is running.") from exc
    except httpx.HTTPStatusError as exc:
        raise RuntimeError(f"Ollama server returned HTTP {exc.response.status_code}: {exc.response.text}") from exc
    except Exception as exc:
        raise RuntimeError(f"Ollama request failed: {str(exc)}") from exc

    raw_content = data.get("message", {}).get("content", "")
    if not raw_content or not raw_content.strip():
        raise RuntimeError("Received an empty response from Ollama.")

    try:
        return AssistantResponse.model_validate_json(raw_content)
    except ValidationError as exc:
        raise RuntimeError(f"Ollama structured response validation failed: {str(exc)}") from exc
    except Exception as exc:
        raise RuntimeError(f"Failed to parse Ollama structured response: {str(exc)}") from exc


def _call_vllm(formatted_user_prompt: str) -> AssistantResponse:
    """Execute inference using local vLLM OpenAI-compatible endpoint."""
    base_url = os.getenv("VLLM_BASE_URL", "http://localhost:8001/v1").rstrip("/")
    model = os.getenv("VLLM_MODEL", "Qwen/Qwen2.5-3B-Instruct")

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": formatted_user_prompt},
        ],
        "temperature": 0.2,
        "response_format": {"type": "json_object"},
    }

    timeout = float(os.getenv("VLLM_TIMEOUT", "120.0"))
    try:
        with httpx.Client(timeout=timeout) as client:
            resp = client.post(f"{base_url}/chat/completions", json=payload)
            resp.raise_for_status()
            data = resp.json()
    except httpx.ConnectError as exc:
        raise RuntimeError(f"Could not connect to vLLM server at {base_url}. Ensure vLLM is running.") from exc
    except httpx.HTTPStatusError as exc:
        raise RuntimeError(f"vLLM server returned HTTP {exc.response.status_code}: {exc.response.text}") from exc
    except Exception as exc:
        raise RuntimeError(f"vLLM request failed: {str(exc)}") from exc

    choices = data.get("choices", [])
    if not choices or "message" not in choices[0]:
        raise RuntimeError("Received invalid response structure from vLLM.")

    raw_content = choices[0]["message"].get("content", "")
    if not raw_content or not raw_content.strip():
        raise RuntimeError("Received empty message content from vLLM.")

    try:
        return AssistantResponse.model_validate_json(raw_content)
    except ValidationError as exc:
        raise RuntimeError(f"vLLM structured response validation failed: {str(exc)}") from exc
    except Exception as exc:
        raise RuntimeError(f"Failed to parse vLLM structured response: {str(exc)}") from exc


def _is_fallback_enabled() -> bool:
    """Check if fallback provider strategy is enabled via environment."""
    val = os.getenv("FALLBACK_ENABLED", "true")
    if val is None:
        return True
    return str(val).lower().strip() in ("true", "1", "yes")


def _execute_fallback_provider(fallback_provider: str, formatted_user_prompt: str) -> AssistantResponse:
    """Execute a single fallback attempt to the configured fallback provider."""
    provider_name = (fallback_provider or "ollama").lower().strip()
    if provider_name == "ollama":
        return _call_ollama(formatted_user_prompt=formatted_user_prompt)
    elif provider_name == "vllm":
        return _call_vllm(formatted_user_prompt=formatted_user_prompt)
    else:
        raise ValueError(f"Unsupported FALLBACK_PROVIDER '{fallback_provider}'. Supported providers: ollama, vllm.")


def get_llm_response(question: str) -> AssistantResponse:
    """Send a question with retrieved RAG context to the configured LLM provider with bounded retry and fallback."""
    raw_provider = os.getenv("LLM_PROVIDER", "gemini")
    provider = (raw_provider or "gemini").lower().strip()
    if provider not in ("gemini", "ollama", "vllm"):
        raise ValueError(f"Unsupported LLM_PROVIDER '{provider}'. Supported providers: gemini, ollama, vllm.")

    # Retrieve relevant document context from ChromaDB if available (executed ONCE)
    try:
        chunks = retrieve_relevant_chunks(question)
    except Exception:
        chunks = []

    formatted_user_prompt = format_rag_prompt(question, chunks)

    # 1. Primary provider execution
    if provider == "gemini":
        try:
            return execute_with_retry(_call_gemini, question=question, formatted_user_prompt=formatted_user_prompt)
        except Exception as primary_exc:
            # Check if fallback is enabled and error is eligible (transient failure, not quota or auth)
            fallback_enabled = _is_fallback_enabled()
            if fallback_enabled and is_transient_error(primary_exc):
                fallback_provider = os.getenv("FALLBACK_PROVIDER", "ollama").lower().strip()
                logger.warning(
                    f"Primary provider Gemini failed after retries: {primary_exc}. "
                    f"Attempting fallback provider: {fallback_provider}."
                )
                try:
                    fallback_response = _execute_fallback_provider(fallback_provider, formatted_user_prompt)
                    logger.info(f"Fallback provider {fallback_provider} succeeded.")
                    return fallback_response
                except Exception as fb_exc:
                    logger.error(f"Fallback provider {fallback_provider} failed: {fb_exc}")
                    raise RuntimeError(
                        f"Primary provider Gemini and fallback provider {fallback_provider} both failed (service unavailable)."
                    ) from fb_exc
            else:
                # Non-eligible error (e.g. Quota 429, Auth 401, Validation) or Fallback Disabled
                raise primary_exc

    elif provider == "ollama":
        return execute_with_retry(_call_ollama, formatted_user_prompt=formatted_user_prompt)
    elif provider == "vllm":
        return execute_with_retry(_call_vllm, formatted_user_prompt=formatted_user_prompt)
    else:
        raise ValueError(f"Unsupported LLM_PROVIDER '{provider}'. Supported providers: gemini, ollama, vllm.")
