# =============================================================================
# SCHEMAS - Pydantic Models for Data Validation
# =============================================================================
# Type-safe data models for the AI Core system.
# =============================================================================

from __future__ import annotations
from typing import List, Dict, Any, Optional, Union, Literal
from pydantic import BaseModel, Field
from datetime import datetime


# =============================================================================
# CHAT MODELS
# =============================================================================

class ChatMessage(BaseModel):
    """Chat message model."""
    role: Literal["system", "user", "assistant", "function", "tool"]
    content: Union[str, List[Dict[str, Any]], None]
    name: Optional[str] = None
    function_call: Optional[Dict[str, Any]] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    
    class Config:
        extra = "allow"


class GenerationConfig(BaseModel):
    """Generation parameters."""
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=4096, ge=1)
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)
    frequency_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)
    presence_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)
    stop: Optional[List[str]] = None


class ChatRequest(BaseModel):
    """Chat request model."""
    query: str = Field(..., min_length=1)
    session_id: Optional[str] = None
    system_prompt: Optional[str] = None
    include_history: bool = True
    documents: Optional[List[str]] = None
    stream: bool = False
    generation: Optional[GenerationConfig] = None
    metadata: Optional[Dict[str, Any]] = None


class ChatResponse(BaseModel):
    """Chat response model."""
    content: str
    role: str = "assistant"
    model: Optional[str] = None
    finish_reason: Optional[str] = None
    usage: Optional[Dict[str, int]] = None
    session_id: Optional[str] = None
    message_id: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.now)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ChatChunk(BaseModel):
    """Streaming chunk model."""
    content: str
    role: Optional[str] = None
    finish_reason: Optional[str] = None
    chunk_index: int = 0


# =============================================================================
# SESSION MODELS
# =============================================================================

class SessionConfig(BaseModel):
    """Session configuration."""
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    system_prompt: Optional[str] = None
    model: Optional[str] = None
    ttl_seconds: int = 3600
    max_history: int = 50
    metadata: Dict[str, Any] = Field(default_factory=dict)


# =============================================================================
# DOCUMENT MODELS
# =============================================================================

class DocumentContext(BaseModel):
    """Document context for RAG."""
    documents: List[str] = Field(default_factory=list)
    top_k: int = 5
    relevance_threshold: float = 0.7
    include_sources: bool = True


# =============================================================================
# FEEDBACK MODELS
# =============================================================================

class FeedbackData(BaseModel):
    """User feedback model."""
    session_id: str
    message_id: Optional[str] = None
    feedback_type: Literal["thumbs_up", "thumbs_down", "rating", "text"]
    value: Optional[Union[bool, int, str]] = None
    comment: Optional[str] = None
    categories: List[str] = Field(default_factory=list)


# =============================================================================
# PROVIDER MODELS
# =============================================================================

class ProviderConfig(BaseModel):
    """Provider configuration."""
    provider: str
    model: str
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    timeout: int = 60
    max_retries: int = 3
    custom_headers: Dict[str, str] = Field(default_factory=dict)
