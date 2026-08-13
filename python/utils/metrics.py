"""Prometheus metrics definitions and FastAPI middleware for FunnelRAG.

All metrics share the ``funnelrag_`` prefix so they can be easily
distinguished in a multi-service Prometheus scrape target.

Typical usage::

    from python.utils.metrics import track_query, track_error, track_cache

    track_query()
    track_error("timeout")
    track_cache("coarse", hit=True)

Or mount the auto-instrumenting middleware::

    from python.utils.metrics import MetricsMiddleware

    app.add_middleware(MetricsMiddleware)
"""

from __future__ import annotations

import time
from typing import Awaitable, Callable, Dict, Optional

from prometheus_client import Counter, Gauge, Histogram
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

# ---------------------------------------------------------------------------
# Metric definitions
# ---------------------------------------------------------------------------

QUERY_TOTAL = Counter(
    "funnelrag_query_total",
    "Total number of RAG queries processed",
)

QUERY_LATENCY = Histogram(
    "funnelrag_query_latency_seconds",
    "End-to-end query latency in seconds",
    buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0),
)

QUERY_ERRORS = Counter(
    "funnelrag_query_errors_total",
    "Total query errors",
    labelnames=("error_type",),
)

MILVUS_SEARCH_LATENCY = Histogram(
    "funnelrag_milvus_search_latency_seconds",
    "Milvus vector search latency in seconds",
    buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0),
)

MILVUS_SEARCH_RESULTS = Histogram(
    "funnelrag_milvus_search_results_count",
    "Number of results returned by Milvus search",
    buckets=(1, 5, 10, 25, 50, 100, 200, 500),
)

RERANK_LATENCY = Histogram(
    "funnelrag_rerank_latency_seconds",
    "Cross-encoder reranking latency in seconds",
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0),
)

RERANK_DOCUMENTS_COUNT = Histogram(
    "funnelrag_rerank_documents_count",
    "Number of documents submitted for reranking",
    buckets=(1, 5, 10, 20, 50, 100, 200),
)

LLM_GENERATE_LATENCY = Histogram(
    "funnelrag_llm_generate_latency_seconds",
    "LLM answer generation latency in seconds",
    buckets=(0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0),
)

LLM_TOKEN_USAGE = Counter(
    "funnelrag_llm_token_usage_total",
    "Total LLM token consumption",
    labelnames=("type",),  # type = "input" | "output"
)

CACHE_HITS = Counter(
    "funnelrag_cache_hits_total",
    "Cache hit count",
    labelnames=("cache_type",),  # cache_type = "coarse" | "fine"
)

CACHE_MISSES = Counter(
    "funnelrag_cache_misses_total",
    "Cache miss count",
    labelnames=("cache_type",),
)

ACTIVE_CONNECTIONS = Gauge(
    "funnelrag_active_connections",
    "Number of currently active connections",
    labelnames=("service",),  # service = "milvus" | "postgres" | "redis"
)

DOCUMENT_STORE_OPS = Counter(
    "funnelrag_document_store_ops_total",
    "Document storage operation count",
    labelnames=("operation",),  # operation = "insert" | "delete" | "update"
)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def track_query() -> None:
    """Increment the global query counter."""
    QUERY_TOTAL.inc()


def track_error(error_type: str) -> None:
    """Increment the query error counter for the given *error_type*.

    Args:
        error_type: Categorisation of the error (e.g. ``"timeout"``,
            ``"validation"``, ``"upstream"``).
    """
    QUERY_ERRORS.labels(error_type=error_type).inc()


def track_cache(cache_type: str, hit: bool) -> None:
    """Record a cache hit or miss.

    Args:
        cache_type: Cache namespace (``"coarse"`` or ``"fine"``).
        hit: ``True`` for a cache hit, ``False`` for a miss.
    """
    if hit:
        CACHE_HITS.labels(cache_type=cache_type).inc()
    else:
        CACHE_MISSES.labels(cache_type=cache_type).inc()


def track_llm_tokens(token_type: str, count: int) -> None:
    """Record LLM token usage.

    Args:
        token_type: ``"input"`` or ``"output"``.
        count: Number of tokens consumed.
    """
    LLM_TOKEN_USAGE.labels(type=token_type).inc(count)


def track_document_op(operation: str, count: int = 1) -> None:
    """Record a document storage operation.

    Args:
        operation: One of ``"insert"``, ``"delete"``, ``"update"``.
        count: Number of documents affected.
    """
    DOCUMENT_STORE_OPS.labels(operation=operation).inc(count)


# ---------------------------------------------------------------------------
# FastAPI / Starlette middleware
# ---------------------------------------------------------------------------


class MetricsMiddleware(BaseHTTPMiddleware):
    """Auto-instrumenting middleware that records request counts and latencies.

    For every inbound HTTP request the middleware:

    1. Starts a high-resolution timer.
    2. Observes the response status code and records the latency into
       :data:`QUERY_LATENCY` (only for ``/query`` paths) or a general
       HTTP latency metric.
    3. Increments :data:`QUERY_TOTAL` for ``/query`` requests.

    Usage::

        from fastapi import FastAPI
        from python.utils.metrics import MetricsMiddleware

        app = FastAPI()
        app.add_middleware(MetricsMiddleware)
    """

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        """Intercept each request, measure latency, and record metrics."""
        start = time.perf_counter()
        response = await call_next(request)
        elapsed = time.perf_counter() - start

        path = request.url.path

        # Only instrument the main query endpoint to keep QUERY_* metrics
        # semantically clean.  Other paths are tracked with a lightweight
        # path-level counter to help with operational visibility.
        if "/query" in path:
            QUERY_TOTAL.inc()
            QUERY_LATENCY.observe(elapsed)

        response.headers["X-Process-Time-Ms"] = f"{elapsed * 1000:.2f}"
        return response
