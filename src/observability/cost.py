"""Cost & performance tracking — aggregates the token/latency data already
emitted by spans (see spans.set_llm_usage) into per-request and cumulative
cost, latency, and cache-hit metrics.

Pricing is per 1M tokens (USD), editable below. Costs are ESTIMATES.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass

# Per 1M tokens, USD (input, output). Update as prices change.
PRICING: dict[str, tuple[float, float]] = {
    "gpt-5-mini": (0.25, 2.00),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
    "gemini-flash-latest": (0.075, 0.30),
    "gemini-2.5-flash": (0.075, 0.30),
    "local": (0.0, 0.0),
}


def price_for(model: str) -> tuple[float, float]:
    m = (model or "").lower()
    for key, price in PRICING.items():
        if key in m:
            return price
    return (0.0, 0.0)


def estimate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    in_price, out_price = price_for(model)
    return (prompt_tokens / 1_000_000) * in_price + (completion_tokens / 1_000_000) * out_price


@dataclass
class _Totals:
    requests: int = 0
    llm_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_cost_usd: float = 0.0
    total_latency_ms: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0


class CostTracker:
    """Process-global accumulator for cost/latency/cache metrics."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._t = _Totals()

    def record_llm(self, model: str, prompt_tokens: int, completion_tokens: int) -> float:
        cost = estimate_cost(model, prompt_tokens, completion_tokens)
        with self._lock:
            self._t.llm_calls += 1
            self._t.prompt_tokens += prompt_tokens
            self._t.completion_tokens += completion_tokens
            self._t.total_cost_usd += cost
        return cost

    def record_request(self, latency_ms: float) -> None:
        with self._lock:
            self._t.requests += 1
            self._t.total_latency_ms += latency_ms

    def record_cache(self, hit: bool) -> None:
        with self._lock:
            if hit:
                self._t.cache_hits += 1
            else:
                self._t.cache_misses += 1

    def snapshot(self) -> dict:
        with self._lock:
            t = self._t
            req = max(t.requests, 1)
            cache_total = t.cache_hits + t.cache_misses
            return {
                "requests": t.requests,
                "llm_calls": t.llm_calls,
                "prompt_tokens": t.prompt_tokens,
                "completion_tokens": t.completion_tokens,
                "total_tokens": t.prompt_tokens + t.completion_tokens,
                "total_cost_usd": round(t.total_cost_usd, 6),
                "avg_cost_per_request_usd": round(t.total_cost_usd / req, 6),
                "avg_latency_ms": round(t.total_latency_ms / req, 1),
                "cache_hits": t.cache_hits,
                "cache_misses": t.cache_misses,
                "cache_hit_rate": round(t.cache_hits / cache_total, 3) if cache_total else None,
            }

    def reset(self) -> None:
        with self._lock:
            self._t = _Totals()


COST = CostTracker()


class track_request_latency:
    """Context manager: times a request and records it on exit."""

    def __enter__(self):
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        COST.record_request((time.perf_counter() - self._t0) * 1000)
        return False