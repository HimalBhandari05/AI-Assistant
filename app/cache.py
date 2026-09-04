import asyncio
from collections import OrderedDict
import logging
import os
import time
from typing import Dict, Optional, Tuple

from app.llm import AssistantResponse

logger = logging.getLogger("ai_assistant.cache")


def normalize_question(question: str) -> str:
    """Normalize question string by stripping whitespace, collapsing internal spaces, and lowercasing."""
    return " ".join(question.strip().split()).lower()


def make_cache_key(provider: str, question: str) -> str:
    """Generate deterministic composite cache key containing provider and normalized question."""
    norm_provider = (provider or "gemini").strip().lower()
    norm_question = normalize_question(question)
    return f"{norm_provider}:{norm_question}"


class InMemoryResponseCache:
    """In-memory, LRU response cache with TTL expiration and provider isolation."""

    def __init__(
        self,
        enabled: Optional[bool] = None,
        ttl_seconds: Optional[float] = None,
        max_size: Optional[int] = None,
    ):
        self._load_config(enabled, ttl_seconds, max_size)
        self._cache: OrderedDict[str, Tuple[AssistantResponse, float]] = OrderedDict()
        self._lock: asyncio.Lock = asyncio.Lock()
        self.hits: int = 0
        self.misses: int = 0

    def _load_config(
        self,
        enabled: Optional[bool] = None,
        ttl_seconds: Optional[float] = None,
        max_size: Optional[int] = None,
    ) -> None:
        """Load cache configuration from parameters or environment variables."""
        if enabled is not None:
            self.enabled = bool(enabled)
        else:
            raw_enabled = os.getenv("CACHE_ENABLED", "true")
            self.enabled = str(raw_enabled).lower().strip() in ("true", "1", "yes")

        if ttl_seconds is not None:
            self.ttl_seconds = float(ttl_seconds)
        else:
            self.ttl_seconds = float(os.getenv("CACHE_TTL_SECONDS", "300"))

        if max_size is not None:
            self.max_size = int(max_size)
        else:
            self.max_size = int(os.getenv("CACHE_MAX_SIZE", "100"))

    async def get(self, provider: str, question: str) -> Optional[AssistantResponse]:
        """Lookup cached response for the provider and question. Returns None on miss or expiration."""
        # Re-check environment toggle dynamically if needed
        if not self._is_enabled():
            return None

        key = make_cache_key(provider, question)
        now = time.time()

        async with self._lock:
            if key in self._cache:
                response, timestamp = self._cache[key]
                if now - timestamp <= self.ttl_seconds:
                    # Cache HIT - mark recently used (LRU)
                    self._cache.move_to_end(key)
                    self.hits += 1
                    logger.info(f"Cache HIT for key '{key[:40]}...'")
                    return response
                else:
                    # Expired entry - evict
                    del self._cache[key]
                    logger.info(f"Cache EXPIRED for key '{key[:40]}...'")

            self.misses += 1
            logger.info(f"Cache MISS for key '{key[:40]}...'")
            return None

    async def set(self, provider: str, question: str, response: AssistantResponse) -> None:
        """Store a successful AssistantResponse in the cache."""
        if not self._is_enabled():
            return

        key = make_cache_key(provider, question)
        now = time.time()

        async with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
            elif len(self._cache) >= self.max_size:
                # Evict oldest entry (LRU eviction)
                evicted_key, _ = self._cache.popitem(last=False)
                logger.info(f"Cache EVICTED oldest entry '{evicted_key[:40]}...'")

            self._cache[key] = (response, now)
            logger.info(f"Cache STORED entry '{key[:40]}...' (Total: {len(self._cache)}/{self.max_size})")

    def _is_enabled(self) -> bool:
        """Check if cache is enabled."""
        raw = os.getenv("CACHE_ENABLED")
        if raw is not None:
            return str(raw).lower().strip() in ("true", "1", "yes")
        return self.enabled

    def clear(self) -> None:
        """Clear cache state and metrics (useful for test isolation)."""
        self._cache.clear()
        self.hits = 0
        self.misses = 0

    def get_metrics(self) -> dict:
        """Return cache performance statistics."""
        return {
            "hits": self.hits,
            "misses": self.misses,
            "size": len(self._cache),
            "max_size": self.max_size,
            "ttl_seconds": self.ttl_seconds,
            "enabled": self._is_enabled(),
        }


# Singleton cache instance
response_cache = InMemoryResponseCache()
