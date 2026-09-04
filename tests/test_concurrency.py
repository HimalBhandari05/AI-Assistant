import asyncio
import os
import sys
import time
from unittest.mock import MagicMock, patch
import httpx

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.main import app
from app.cache import response_cache
from app.rate_limiter import rate_limiter

TEST_QUESTIONS = [
    "What is binary search?",
    "What is photosynthesis?",
    "What is Retrieval-Augmented Generation?",
    "What is 25 multiplied by 18?",
]


async def run_sequential(
    client: httpx.AsyncClient, questions: list[str]
) -> tuple[float, list[float], list[dict]]:
    """Execute requests sequentially and record timing."""
    latencies = []
    responses = []
    start_total = time.perf_counter()

    for q in questions:
        req_start = time.perf_counter()
        resp = await client.post("/ask", json={"question": q})
        req_elapsed = time.perf_counter() - req_start
        latencies.append(req_elapsed)
        responses.append(
            {
                "status": resp.status_code,
                "data": resp.json() if resp.status_code == 200 else resp.text,
                "latency": req_elapsed,
                "process_header": resp.headers.get("X-Process-Time"),
            }
        )

    total_time = time.perf_counter() - start_total
    return total_time, latencies, responses


async def run_concurrent(
    client: httpx.AsyncClient, questions: list[str]
) -> tuple[float, list[float], list[dict]]:
    """Execute requests concurrently with asyncio.gather and record timing."""
    start_total = time.perf_counter()

    async def fetch_single(q: str):
        req_start = time.perf_counter()
        resp = await client.post("/ask", json={"question": q})
        req_elapsed = time.perf_counter() - req_start
        return {
            "status": resp.status_code,
            "data": resp.json() if resp.status_code == 200 else resp.text,
            "latency": req_elapsed,
            "process_header": resp.headers.get("X-Process-Time"),
        }

    results = await asyncio.gather(*(fetch_single(q) for q in questions))
    total_time = time.perf_counter() - start_total
    latencies = [r["latency"] for r in results]
    return total_time, latencies, list(results)


def print_performance_summary(
    mode: str, total_time: float, latencies: list[float], count: int
):
    """Print clean summary metrics for performance tests."""
    avg_latency = sum(latencies) / len(latencies) if latencies else 0
    throughput = count / total_time if total_time > 0 else 0
    print(f"=== {mode} Benchmark Results ===")
    print(f"Completed Requests:  {count}")
    print(f"Total Elapsed Time:  {total_time:.4f}s")
    print(f"Average Latency:     {avg_latency:.4f}s")
    print(f"Min / Max Latency:   {min(latencies):.4f}s / {max(latencies):.4f}s")
    print(f"Throughput:          {throughput:.2f} req/s\n")


import pytest


@pytest.mark.asyncio
async def test_concurrency_and_latency():
    """Run benchmark comparing sequential vs concurrent request handling."""
    rate_limiter.reset()
    response_cache.clear()

    # Simulated I/O delay (e.g. 0.2s representing model/network roundtrip)
    def mock_get_llm_response(question: str):
        time.sleep(0.2)
        return {"answer": f"Answer for: {question}", "topic": "General"}

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        with patch("app.main.get_llm_response", side_effect=mock_get_llm_response):
            print("Running Sequential Execution Test (4 requests)...")
            seq_total, seq_latencies, seq_resps = await run_sequential(client, TEST_QUESTIONS)
            print_performance_summary("Sequential", seq_total, seq_latencies, len(TEST_QUESTIONS))

            # Clear cache between sequential and concurrent runs to benchmark raw concurrency
            response_cache.clear()
            rate_limiter.reset()

            print("Running Concurrent Execution Test (4 requests)...")
            con_total, con_latencies, con_resps = await run_concurrent(client, TEST_QUESTIONS)
            print_performance_summary("Concurrent", con_total, con_latencies, len(TEST_QUESTIONS))

            # Verify all responses succeeded
            assert all(r["status"] == 200 for r in seq_resps)
            assert all(r["status"] == 200 for r in con_resps)

            # Verify X-Process-Time header
            for r in con_resps:
                assert r["process_header"] is not None and "s" in r["process_header"]
            print("✓ Latency middleware (X-Process-Time header) verified")

            # Concurrent should be substantially faster than sequential (approaching ~1x delay instead of 4x delay)
            assert con_total < seq_total, f"Expected concurrent ({con_total:.2f}s) < sequential ({seq_total:.2f}s)"
            print(f"✓ Concurrency speedup verified: {seq_total:.2f}s -> {con_total:.2f}s ({(seq_total/con_total):.2f}x faster)")

        # Error isolation test: 1 invalid request along with 2 valid requests
        mixed_payloads = [
            "Valid question 1",
            "",  # Invalid empty question -> should return 422
            "Valid question 2",
        ]
        with patch("app.main.get_llm_response", return_value={"answer": "ok", "topic": "test"}):
            _, _, mixed_resps = await run_concurrent(client, mixed_payloads)
            assert mixed_resps[0]["status"] == 200
            assert mixed_resps[1]["status"] == 422
            assert mixed_resps[2]["status"] == 200
            print("✓ Concurrent error isolation verified (failed request does not affect concurrent sibling requests)")


if __name__ == "__main__":
    asyncio.run(test_concurrency_and_latency())
