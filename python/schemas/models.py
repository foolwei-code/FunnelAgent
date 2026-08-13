"""Pydantic request / response schemas for the FunnelRAG API.

Every model includes strict validation rules so that malformed data is
rejected before it reaches business logic.  Serialisation helpers
(``model_dump``, ``model_dump_json``) are provided by Pydantic v2 out
of the box.

Typical usage::

    from python.schemas.models import QueryRequest, QueryResponse

    req = QueryRequest(question="What is RAG?")
    resp = QueryResponse(answer="...", sources=[...])
"""

from __future__ import annotations

import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Shared / composite models
# ---------------------------------------------------------------------------


class SourceDocument(BaseModel):
    """A single retrieved document returned as part of a query response.

    Attributes:
        doc_id: Unique document identifier (e.g. Milvus primary key).
        title: Human-readable document title.
        content: Full or truncated document text.
        source: Origin identifier (URL, file path, etc.).
        score: Relevance score assigned by retrieval / reranking.
        metadata: Arbitrary key-value pairs for additional context.
    """

    doc_id: str = Field(..., min_length=1, description="Unique document identifier")
    title: str = Field(default="", description="Document title")
    content: str = Field(default="", description="Document content excerpt")
    source: str = Field(default="", description="Origin of the document")
    score: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Relevance score in [0, 1]"
    )
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Extra metadata")


class ThinkingStep(BaseModel):
    """Represents a single reasoning / tool-use step in the answer chain.

    Attributes:
        tool: Name of the tool or pipeline stage invoked.
        input: Input provided to the tool (serialised as string).
        output: Tool output (serialised as string).
        duration_ms: Wall-clock duration of this step in milliseconds.
    """

    tool: str = Field(..., min_length=1, description="Tool or stage name")
    input: str = Field(default="", description="Tool input (stringified)")
    output: str = Field(default="", description="Tool output (stringified)")
    duration_ms: float = Field(default=0.0, ge=0.0, description="Step duration in ms")


class StreamEvent(BaseModel):
    """A single event in a server-sent-event (SSE) stream.

    Attributes:
        type: Event type discriminator (e.g. ``"token"``, ``"source"``,
            ``"thinking"``, ``"done"``, ``"error"``).
        data: Payload associated with the event.
    """

    type: str = Field(..., min_length=1, description="Event type")
    data: Dict[str, Any] = Field(default_factory=dict, description="Event payload")


# ---------------------------------------------------------------------------
# Query models
# ---------------------------------------------------------------------------


class QueryRequest(BaseModel):
    """Inbound RAG query request.

    Attributes:
        question: The user's natural-language question.
        top_k: Maximum number of documents to return in the answer context.
        conversation_id: Optional conversation identifier for multi-turn
            dialogue continuity.
        max_tokens: Override the default LLM generation token limit.
        temperature_override: Override the default LLM sampling temperature.
        include_sources: Whether to include source documents in the response.
    """

    question: str = Field(..., min_length=1, description="User question")
    top_k: int = Field(
        default=5, ge=1, le=100, description="Number of documents to retrieve"
    )
    conversation_id: Optional[str] = Field(
        default=None,
        description="Conversation ID for multi-turn context",
    )
    max_tokens: Optional[int] = Field(
        default=None,
        ge=1,
        le=32768,
        description="Override LLM max_tokens for this request",
    )
    temperature_override: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=2.0,
        description="Override LLM temperature for this request",
    )
    include_sources: bool = Field(
        default=True,
        description="Include source documents in the response",
    )

    @field_validator("question")
    @classmethod
    def _normalise_question(cls, v: str) -> str:
        """Strip surrounding whitespace from the question."""
        return v.strip()


class QueryMetadata(BaseModel):
    """Operational metadata about a completed query.

    Attributes:
        token_usage: Token counts consumed by the LLM.
        latency_ms: End-to-end query latency in milliseconds.
        model_name: Name of the LLM model used to generate the answer.
    """

    token_usage: Dict[str, int] = Field(
        default_factory=dict,
        description="Token usage breakdown, e.g. {'input': 120, 'output': 85}",
    )
    latency_ms: float = Field(
        default=0.0, ge=0.0, description="End-to-end latency in ms"
    )
    model_name: str = Field(default="", description="LLM model name used")


class QueryResponse(BaseModel):
    """Outbound RAG query response.

    Attributes:
        answer: The generated answer text.
        sources: List of source documents that grounded the answer.
        thinking_chain: Ordered list of reasoning / tool-use steps.
        metadata: Operational metadata (token usage, latency, model).
    """

    answer: str = Field(default="", description="Generated answer text")
    sources: List[SourceDocument] = Field(
        default_factory=list,
        description="Source documents grounding the answer",
    )
    thinking_chain: Optional[List[ThinkingStep]] = Field(
        default=None,
        description="Ordered chain-of-thought steps",
    )
    metadata: QueryMetadata = Field(
        default_factory=QueryMetadata,
        description="Operational response metadata",
    )


# ---------------------------------------------------------------------------
# Ingestion models
# ---------------------------------------------------------------------------


class IngestDocument(BaseModel):
    """A single document to be ingested into the knowledge base.

    Attributes:
        doc_id: Client-assigned document identifier.  Must be unique within
            the target collection.
        title: Document title.
        content: Full document text content.
        source: Origin of the document (URL, file path, etc.).
        metadata: Arbitrary key-value metadata to store alongside the document.
    """

    doc_id: str = Field(..., min_length=1, description="Unique document identifier")
    title: str = Field(default="", description="Document title")
    content: str = Field(..., min_length=1, description="Document text content")
    source: str = Field(default="", description="Origin of the document")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Extra metadata")

    @field_validator("content")
    @classmethod
    def _content_not_blank(cls, v: str) -> str:
        """Reject blank / whitespace-only content."""
        if not v.strip():
            raise ValueError("content must not be blank")
        return v


class IngestRequest(BaseModel):
    """Batch document-ingestion request.

    Attributes:
        documents: List of documents to ingest.  Must contain at least one
            document and no more than 500 per request.
    """

    documents: List[IngestDocument] = Field(
        ...,
        min_length=1,
        max_length=500,
        description="Documents to ingest (1–500 per request)",
    )

    @model_validator(mode="after")
    def _unique_doc_ids(self) -> "IngestRequest":
        """Ensure all doc_ids within a single request are unique."""
        ids = [doc.doc_id for doc in self.documents]
        if len(ids) != len(set(ids)):
            duplicates = [did for did in ids if ids.count(did) > 1]
            raise ValueError(f"Duplicate doc_ids found: {set(duplicates)}")
        return self


class IngestResponse(BaseModel):
    """Response returned after a successful ingestion.

    Attributes:
        ingested: Number of documents successfully ingested.
        failed: Number of documents that failed to ingest.
        errors: Per-document error details for any failures.
    """

    ingested: int = Field(default=0, ge=0, description="Number of documents ingested")
    failed: int = Field(default=0, ge=0, description="Number of documents that failed")
    errors: Dict[str, str] = Field(
        default_factory=dict,
        description="Map of doc_id → error message for failed documents",
    )


# ---------------------------------------------------------------------------
# System health / error models
# ---------------------------------------------------------------------------


class HealthResponse(BaseModel):
    """Health-check response payload.

    Attributes:
        status: Overall health status (``"healthy"``, ``"degraded"``, ``"unhealthy"``).
        version: Application version string.
        uptime_seconds: Seconds since the application process started.
        services: Per-service health mapping (e.g. ``{"milvus": True}``).
    """

    status: str = Field(default="healthy", description="Overall health status")
    version: str = Field(default="0.1.0", description="Application version")
    uptime_seconds: float = Field(
        default=0.0,
        ge=0.0,
        description="Process uptime in seconds",
    )
    services: Dict[str, bool] = Field(
        default_factory=dict,
        description="Per-service reachability mapping",
    )

    @field_validator("status")
    @classmethod
    def _validate_status(cls, v: str) -> str:
        """Ensure status is one of the allowed values."""
        allowed = {"healthy", "degraded", "unhealthy"}
        if v not in allowed:
            raise ValueError(f"status must be one of {allowed}, got {v!r}")
        return v


class ErrorResponse(BaseModel):
    """Standard error envelope returned for all API errors.

    Attributes:
        error: Short error code or category (e.g. ``"validation_error"``).
        detail: Human-readable explanation of what went wrong.
        request_id: Correlation ID for log-tracing.
    """

    error: str = Field(..., min_length=1, description="Error code or category")
    detail: str = Field(default="", description="Human-readable error detail")
    request_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Correlation ID for tracing",
    )
