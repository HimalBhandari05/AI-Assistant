import asyncio
import logging
import math
import os
import time
from typing import Dict, List, Optional
from fastapi import HTTPException, Request, status

logger = logging.getLogger("ai_assistant.rate_limiter")


def get_client_ip(request: Request) -> str:
    """Extract client IP from X-Forwarded-For header or direct client host."""
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        # X-Forwarded-For can be a comma-separated list; first entry is the original client
        return forwarded.split(",")[0].strip()
    if request.client and request.client.host:
        return request.client.host
    return "127.0.0.1"


class InMemoryRateLimiter:
    """In-memory, sliding-window rate limiter per client IP.
    
    Tracks request timestamps within a configurable rolling window.
    Thread-safe and async-safe via asyncio.Lock.
    """

    def __init__(
        self,
        requests_limit: Optional[int] = None,
        window_seconds: Optional[float] = None,
    ):
        self._load_config(requests_limit, window_seconds)
        self._requests: Dict[str, List[float]] = {}
        self._lock: asyncio.Lock = asyncio.Lock()

    def _load_config(
        self,
        requests_limit: Optional[int] = None,
        window_seconds: Optional[float] = None,
    ):
        """Read rate limit configuration from parameters or environment variables."""
        if requests_limit is not None:
            self.requests_limit = int(requests_limit)
        else:
            self.requests_limit = int(os.getenv("RATE_LIMIT_REQUESTS", "10"))

        if window_seconds is not None:
            self.window_seconds = float(window_seconds)
        else:
            self.window_seconds = float(os.getenv("RATE_LIMIT_WINDOW_SECONDS", "60"))

    async def check_rate_limit(self, request: Request) -> None:
        """Check if incoming request from client IP is within rate limits.
        
        Raises HTTPException(429) with Retry-After header if limit exceeded.
        """
        client_ip = get_client_ip(request)
        now = time.time()

        async with self._lock:
            # Clean up expired timestamps for this IP
            cutoff = now - self.window_seconds
            timestamps = self._requests.get(client_ip, [])
            valid_timestamps = [t for t in timestamps if t > cutoff]

            if len(valid_timestamps) >= self.requests_limit:
                # Calculate remaining seconds until the oldest request falls out of the window
                oldest_timestamp = valid_timestamps[0]
                retry_after = max(1, math.ceil(oldest_timestamp + self.window_seconds - now))
                logger.warning(
                    f"Rate limit exceeded for IP {client_ip}: {len(valid_timestamps)}/{self.requests_limit} "
                    f"requests in {self.window_seconds}s window. Retry-After: {retry_after}s"
                )
                self._requests[client_ip] = valid_timestamps
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail="Rate limit exceeded. Please try again later.",
                    headers={"Retry-After": str(retry_after)},
                )

            # Accept request and register current timestamp
            valid_timestamps.append(now)
            self._requests[client_ip] = valid_timestamps

            # Memory management: opportunistic cleanup if tracker dictionary grows large
            if len(self._requests) > 1000:
                self._cleanup_stale_entries(now)

    def _cleanup_stale_entries(self, current_time: float) -> None:
        """Prune client IPs that have no active timestamps within the current window."""
        cutoff = current_time - self.window_seconds
        stale_ips = [
            ip for ip, timestamps in self._requests.items()
            if not any(t > cutoff for t in timestamps)
        ]
        for ip in stale_ips:
            del self._requests[ip]

    def reset(self) -> None:
        """Reset internal request state and reload config from environment (useful for test isolation)."""
        self._requests.clear()
        self._load_config()


# Default singleton rate limiter instance
rate_limiter = InMemoryRateLimiter()
