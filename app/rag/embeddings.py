import os
from google import genai
from google.genai.errors import APIError


def get_embedding_client_and_model() -> tuple[genai.Client, str]:
    """Retrieve initialized Gemini client and configured embedding model name."""
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key or not api_key.strip() or api_key.strip() in ("your_api_key", "your_gemini_api_key", "your_gemini_api_key_here"):
        raise ValueError("GEMINI_API_KEY is not configured. Please set a valid API key in your .env file.")

    model = os.getenv("GEMINI_EMBEDDING_MODEL", "text-embedding-004")
    client = genai.Client(api_key=api_key)
    return client, model


def generate_embedding(text: str) -> list[float]:
    """Generate vector embedding for a single text string using Gemini API."""
    if not text or not text.strip():
        raise ValueError("Text for embedding generation cannot be empty.")

    client, model = get_embedding_client_and_model()

    try:
        response = client.models.embed_content(
            model=model,
            contents=text,
        )
    except Exception as exc:
        err_msg = str(exc).lower()
        if "404" in err_msg or "not found" in err_msg:
            # Fallback to standard gemini-embedding-001
            try:
                response = client.models.embed_content(
                    model="gemini-embedding-001",
                    contents=text,
                )
            except Exception as fb_exc:
                raise RuntimeError(f"Gemini embedding API error: {str(fb_exc)}") from fb_exc
        else:
            raise RuntimeError(f"Gemini embedding API error: {str(exc)}") from exc

    if not response.embeddings or not response.embeddings[0].values:
        raise RuntimeError("Received empty embedding vector from Gemini embedding API.")

    return [float(val) for val in response.embeddings[0].values]


def generate_embeddings(texts: list[str]) -> list[list[float]]:
    """Generate vector embeddings for a list of text strings in a single batch API call."""
    if not texts:
        return []

    client, model = get_embedding_client_and_model()

    try:
        response = client.models.embed_content(
            model=model,
            contents=texts,
        )
    except Exception as exc:
        err_msg = str(exc).lower()
        if "404" in err_msg or "not found" in err_msg:
            try:
                response = client.models.embed_content(
                    model="gemini-embedding-001",
                    contents=texts,
                )
            except Exception as fb_exc:
                raise RuntimeError(f"Gemini embedding API error: {str(fb_exc)}") from fb_exc
        else:
            raise RuntimeError(f"Gemini embedding API error: {str(exc)}") from exc

    if not response.embeddings:
        raise RuntimeError("Received empty embeddings list from Gemini embedding API.")

    return [[float(val) for val in emb.values] for emb in response.embeddings]
