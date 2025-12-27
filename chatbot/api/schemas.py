# -*- coding: utf-8 -*-
# =================================================================================================
# api/schemas.py — Pydantic Request/Response Models
# =================================================================================================
# Production-grade Pydantic models implementing:
#
#   1. INPUT VALIDATION: Strict type checking with meaningful error messages.
#   2. SERIALIZATION: JSON-compatible output with camelCase aliases.
#   3. DOCUMENTATION: Field descriptions for OpenAPI/Swagger generation.
#   4. SECURITY: Size limits to prevent DoS via large payloads.
#   5. OPTIONAL FIELDS: Sensible defaults reduce required client input.
#
# Naming Conventions:
# -------------------
#   - *Request: Input models for POST/PUT/PATCH endpoints.
#   - *Response: Output models for API responses.
#   - *Base: Shared fields inherited by request/response models.
#   - *Create: Input for resource creation.
#   - *Update: Input for resource updates (partial).
#
# =================================================================================================

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Literal, Optional, Union
from pydantic import BaseModel, ConfigDict, Field, field_validator
import time


# -----------------------------------------------------------------------------
# Configuration for All Models
# -----------------------------------------------------------------------------
class BaseSchema(BaseModel):
    """
    Base schema with shared configuration.
    
    Model Config:
    -------------
    - from_attributes: Enable ORM mode for SQLAlchemy compatibility.
    - populate_by_name: Accept both snake_case and alias names.
    - str_strip_whitespace: Auto-strip whitespace from strings.
    - validate_default: Validate default values.
    """
    model_config = ConfigDict(
        from_attributes=True,
        populate_by_name=True,
        str_strip_whitespace=True,
        validate_default=True,
    )


# =============================================================================
# Session Schemas
# =============================================================================

class SessionCreate(BaseSchema):
    """
    Request model for creating a new chat session.
    
    Usage:
    ------
    POST /api/v1/sessions
    
    Example:
    --------
    {
        "user_id": "user_123",
        "title": "Trip Planning",
        "metadata": {"source": "web"}
    }
    """
    user_id: Optional[str] = Field(
        default=None,
        max_length=256,
        description="Optional user identifier for session ownership.",
    )
    title: Optional[str] = Field(
        default=None,
        max_length=512,
        description="Optional session title for display.",
    )
    conv_id: Optional[str] = Field(
        default=None,
        max_length=64,
        description="Custom conversation ID (auto-generated if not provided).",
    )
    branch_id: str = Field(
        default="main",
        max_length=64,
        description="Branch ID for conversation forking.",
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Custom metadata dictionary.",
    )


class SessionUpdate(BaseSchema):
    """Request model for updating a session."""
    title: Optional[str] = Field(default=None, max_length=512)
    metadata: Optional[Dict[str, Any]] = None
    version: Optional[int] = Field(
        default=None,
        ge=1,
        description="Version for optimistic locking (optional).",
    )


class SessionResponse(BaseSchema):
    """
    Response model for session data.
    
    Returned by:
    ------------
    - POST /api/v1/sessions (create)
    - GET /api/v1/sessions/{session_id} (retrieve)
    """
    id: str = Field(description="Unique session identifier (UUID).")
    user_id: Optional[str] = Field(default=None, description="User identifier.")
    conv_id: str = Field(description="Conversation ID for history engine.")
    branch_id: str = Field(description="Branch ID for conversation forking.")
    title: Optional[str] = Field(default=None, description="Session title.")
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: float = Field(description="Creation timestamp (Unix epoch).")
    updated_at: float = Field(description="Last update timestamp.")
    is_active: bool = Field(default=True, description="False if soft-deleted.")
    version: int = Field(default=1, description="Version for optimistic locking.")
    
    @field_validator("created_at", "updated_at", mode="before")
    @classmethod
    def convert_timestamp(cls, v: Any) -> float:
        """Ensure timestamps are float."""
        if isinstance(v, datetime):
            return v.timestamp()
        return float(v) if v is not None else time.time()


class SessionListResponse(BaseSchema):
    """Response model for session list."""
    sessions: List[SessionResponse]
    total: int = Field(description="Total number of sessions.")
    limit: int = Field(description="Maximum items per page.")
    offset: int = Field(description="Current offset.")


# =============================================================================
# History Schemas (Query-Answer Pairs)
# =============================================================================

class MessageCreate(BaseSchema):
    """
    Request model for adding a message to history.
    
    Usage:
    ------
    POST /api/v1/sessions/{session_id}/messages
    """
    query: str = Field(
        min_length=1,
        max_length=32768,  # 32KB limit for security
        description="User query or message content.",
    )
    role: Literal["user", "assistant", "system"] = Field(
        default="user",
        description="Message role in conversation.",
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Custom metadata (e.g., client info, session state).",
    )


class HistoryEntry(BaseSchema):
    """
    Response model for a history entry.
    
    Represents a single query-answer pair in conversation history.
    """
    id: str = Field(description="Unique history entry ID (UUID).")
    session_id: str = Field(description="Parent session ID.")
    query: str = Field(description="User query text.")
    answer: Optional[str] = Field(default=None, description="AI response text.")
    role: str = Field(description="Message role (user/assistant/system).")
    metadata: Dict[str, Any] = Field(default_factory=dict)
    retrieves: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Retrieved context chunks used for response.",
    )
    tokens_query: Optional[int] = Field(default=None, description="Query token count.")
    tokens_answer: Optional[int] = Field(default=None, description="Answer token count.")
    latency_ms: Optional[float] = Field(default=None, description="Response latency.")
    created_at: float
    updated_at: float


class HistoryListResponse(BaseSchema):
    """Response model for history listing."""
    entries: List[HistoryEntry]
    total: int
    session_id: str


# =============================================================================
# Chat/Generation Schemas
# =============================================================================

class FeedbackRequest(BaseSchema):
    """
    Request model for message feedback.
    """
    score: int = Field(description="Feedback score: 1 (like) or -1 (dislike).", ge=-1, le=1)
    comment: Optional[str] = Field(default=None, description="Optional feedback comment.")


class ChatRequest(BaseSchema):
    """
    Request model for chat completion.
    
    Usage:
    ------
    POST /api/v1/chat
    
    Example:
    --------
    {
        "session_id": "abc-123",
        "message": "What is the capital of France?",
        "temperature": 0.7,
        "max_tokens": 1024
    }
    """
    session_id: Optional[str] = Field(
        default=None,
        description="Session ID for context. Creates new session if not provided.",
    )
    message: str = Field(
        min_length=1,
        max_length=32768,
        description="User message to process.",
    )
    system_prompt: Optional[str] = Field(
        default=None,
        max_length=8192,
        description="Optional system prompt override.",
    )
    temperature: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=2.0,
        description="Sampling temperature (0.0-2.0).",
    )
    max_tokens: Optional[int] = Field(
        default=None,
        ge=1,
        le=16384,
        description="Maximum tokens to generate.",
    )
    include_context: bool = Field(
        default=True,
        description="Whether to include retrieved context.",
    )
    context_token_budget: Optional[int] = Field(
        default=None,
        ge=0,
        le=32768,
        description="Token budget for context retrieval.",
    )
    model: Optional[str] = Field(
        default=None,
        max_length=256,
        description="Model override for this request.",
    )


class ContextChunk(BaseSchema):
    """A chunk of context retrieved for the response."""
    text: str = Field(description="Chunk text content.")
    score: float = Field(description="Relevance score (0-1).")
    doc_id: Optional[str] = Field(default=None, description="Source document ID.")
    chunk_index: Optional[int] = Field(default=None, description="Chunk position in document.")
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ChatResponse(BaseSchema):
    """
    Response model for chat completion.
    
    Contains the AI response along with context and metadata.
    """
    id: str = Field(description="Response ID (UUID).")
    session_id: str = Field(description="Session ID used.")
    message: str = Field(description="AI response message.")
    context: List[ContextChunk] = Field(
        default_factory=list,
        description="Retrieved context chunks.",
    )
    model: str = Field(description="Model used for generation.")
    tokens_prompt: Optional[int] = Field(default=None, description="Prompt tokens used.")
    tokens_completion: Optional[int] = Field(default=None, description="Completion tokens.")
    latency_ms: float = Field(description="Total response latency.")
    finish_reason: Optional[str] = Field(
        default=None,
        description="Why generation stopped (stop/length/error).",
    )
    metadata: Dict[str, Any] = Field(default_factory=dict)


class CompletionRequest(BaseSchema):
    """Request model for text completion (non-chat)."""
    prompt: str = Field(min_length=1, max_length=32768)
    temperature: Optional[float] = Field(default=None, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(default=None, ge=1, le=16384)
    model: Optional[str] = Field(default=None, max_length=256)


class CompletionResponse(BaseSchema):
    """Response model for text completion."""
    id: str
    text: str
    model: str
    tokens_prompt: Optional[int] = None
    tokens_completion: Optional[int] = None
    latency_ms: float
    finish_reason: Optional[str] = None


# =============================================================================
# Search/Retrieval Schemas
# =============================================================================

class SearchRequest(BaseSchema):
    """
    Request model for semantic search.
    
    Usage:
    ------
    POST /api/v1/search
    """
    query: str = Field(
        min_length=1,
        max_length=8192,
        description="Search query text.",
    )
    top_k: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Number of results to return.",
    )
    collection: Optional[str] = Field(
        default=None,
        max_length=128,
        description="Collection to search (uses default if not provided).",
    )
    filters: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Metadata filters for search.",
    )
    include_text: bool = Field(
        default=True,
        description="Include full text in results.",
    )
    rerank: bool = Field(
        default=False,
        description="Apply Rethinker reranking.",
    )


class SearchResult(BaseSchema):
    """A single search result."""
    id: str = Field(description="Chunk ID.")
    text: str = Field(description="Chunk text content.")
    score: float = Field(description="Relevance score.")
    doc_id: Optional[str] = Field(default=None, description="Source document ID.")
    chunk_index: Optional[int] = Field(default=None)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class SearchResponse(BaseSchema):
    """Response model for search results."""
    query: str = Field(description="Original query.")
    results: List[SearchResult] = Field(description="Search results.")
    total: int = Field(description="Total results found.")
    latency_ms: float = Field(description="Search latency.")


# =============================================================================
# Document Processing Schemas
# =============================================================================

class DocumentUploadResponse(BaseSchema):
    """Response after document upload."""
    id: str = Field(description="Document ID for status tracking.")
    filename: str
    status: Literal["pending", "processing", "completed", "failed"]
    message: str = Field(default="Document queued for processing.")


class DocumentStatus(BaseSchema):
    """Document processing status with detailed progress tracking."""
    id: str
    filename: str
    status: Literal["pending", "processing", "completed", "failed"]
    stage: Literal["pending", "converting", "chunking", "indexing", "completed", "failed"] = Field(
        default="pending",
        description="Current processing stage for detailed progress."
    )
    progress: int = Field(
        default=0,
        ge=0,
        le=100,
        description="Processing progress percentage (0-100)."
    )
    chunk_count: int = Field(default=0)
    file_size: Optional[int] = Field(default=None, description="File size in bytes.")
    error: Optional[str] = None
    created_at: float
    updated_at: float


class DocumentListResponse(BaseSchema):
    """Paginated list of documents."""
    documents: List[DocumentStatus]
    total: int
    limit: int
    offset: int


class ChunkTextRequest(BaseSchema):
    """Request to chunk raw text."""
    text: str = Field(min_length=1, max_length=1048576)  # 1MB limit
    chunk_size: int = Field(default=1024, ge=64, le=8192)
    chunk_overlap: int = Field(default=128, ge=0, le=2048)
    doc_id: Optional[str] = Field(default=None, max_length=256)


class ChunkResult(BaseSchema):
    """A text chunk."""
    index: int
    text: str
    start: int
    end: int
    hash64: Optional[int] = None


class ChunkTextResponse(BaseSchema):
    """Response with chunked text."""
    doc_id: str
    chunks: List[ChunkResult]
    total_chunks: int


class ChunkingStatusResponse(BaseSchema):
    """
    Response for chunking status check.
    
    Returned by:
    ------------
    GET /api/v1/documents/{doc_id}/chunking-status
    
    Fields:
    -------
    - doc_id: Unique document identifier
    - is_chunked: True if JSONL chunks file exists
    - chunk_count: Number of chunks generated (0 if not chunked)
    - chunks_file: Absolute path to the JSONL chunks file
    - status: Current document processing status
    """
    doc_id: str = Field(description="Document ID.")
    is_chunked: bool = Field(description="True if chunking is complete.")
    chunk_count: int = Field(default=0, description="Number of chunks generated.")
    chunks_file: Optional[str] = Field(default=None, description="Path to JSONL chunks file.")
    status: str = Field(description="Document processing status.")


# =============================================================================
# Health/System Schemas
# =============================================================================

class HealthResponse(BaseSchema):
    """Health check response."""
    status: Literal["healthy", "degraded", "unhealthy"]
    timestamp: float = Field(default_factory=time.time)
    version: str = Field(default="1.0.0")
    components: Dict[str, str] = Field(
        default_factory=dict,
        description="Status of individual components.",
    )


class MetricsResponse(BaseSchema):
    """System metrics response."""
    uptime_seconds: float
    requests_total: int
    requests_active: int
    avg_latency_ms: float
    error_rate: float
    db_connections_active: int
    db_connections_pool: int


# =============================================================================
# Error Response Schema
# =============================================================================

class ErrorDetail(BaseSchema):
    """Detailed error information."""
    code: str = Field(description="Error code for programmatic handling.")
    message: str = Field(description="Human-readable error message.")
    field: Optional[str] = Field(default=None, description="Field causing error.")


class ErrorResponse(BaseSchema):
    """
    Standard error response format.
    
    Consistent error structure across all API endpoints.
    """
    error: str = Field(description="Error type/category.")
    message: str = Field(description="Human-readable error description.")
    details: List[ErrorDetail] = Field(
        default_factory=list,
        description="Additional error details.",
    )
    request_id: Optional[str] = Field(
        default=None,
        description="Request ID for debugging.",
    )
    timestamp: float = Field(default_factory=time.time)


# =============================================================================
# Batch Operations Schemas
# =============================================================================

class BatchChatRequest(BaseSchema):
    """Request for batch chat completions."""
    messages: List[str] = Field(
        min_length=1,
        max_length=100,
        description="List of messages to process.",
    )
    session_id: Optional[str] = None
    system_prompt: Optional[str] = Field(default=None, max_length=8192)
    temperature: Optional[float] = Field(default=None, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(default=None, ge=1, le=8192)


class BatchChatResponse(BaseSchema):
    """Response for batch chat completions."""
    results: List[ChatResponse]
    total: int
    successful: int
    failed: int
    latency_ms: float


# =============================================================================
# Context Building Schemas
# =============================================================================

class ContextRequest(BaseSchema):
    """Request to build context for a query."""
    query: str = Field(min_length=1, max_length=8192)
    session_id: str
    token_budget: int = Field(default=4096, ge=64, le=32768)
    include_recent: int = Field(
        default=10,
        ge=0,
        le=100,
        description="Number of recent messages to include.",
    )


class ContextResponse(BaseSchema):
    """Built context for a query."""
    messages: List[HistoryEntry] = Field(description="Context messages.")
    total_tokens: int = Field(description="Estimated token count.")
    sources: List[ContextChunk] = Field(default_factory=list)
