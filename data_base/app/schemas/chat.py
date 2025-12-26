# ==============================================================================
# CHAT SCHEMAS - Chatbot Functionality
# ==============================================================================
# Request/Response schemas for chat sessions and messages
# ==============================================================================

from __future__ import annotations

from typing import List, Optional

from pydantic import Field

from app.schemas.base import BaseSchema, TimestampSchema


class ChatMessageCreate(BaseSchema):
    """Schema for creating a chat message."""
    
    role: str = Field(
        ...,
        pattern="^(user|assistant|system)$",
        description="Message role (user/assistant/system)",
    )
    content: str = Field(
        ...,
        min_length=1,
        max_length=100000,
        description="Message content",
    )


class ChatMessageResponse(TimestampSchema):
    """Schema for chat message response."""
    
    id: str = Field(
        ...,
        description="Message unique identifier",
    )
    session_id: str = Field(
        ...,
        description="Parent session ID",
    )
    role: str = Field(
        ...,
        description="Message role",
    )
    content: str = Field(
        ...,
        description="Message content",
    )
    tokens: Optional[int] = Field(
        None,
        description="Token count for AI processing",
    )
    model: Optional[str] = Field(
        None,
        description="AI model used for response",
    )


class ChatSessionCreate(BaseSchema):
    """Schema for creating a chat session."""
    
    title: str = Field(
        "New Chat",
        max_length=255,
        description="Session display title",
    )
    description: Optional[str] = Field(
        None,
        max_length=1000,
        description="Session description",
    )


class ChatSessionUpdate(BaseSchema):
    """Schema for updating a chat session."""
    
    title: Optional[str] = Field(
        None,
        max_length=255,
        description="Session display title",
    )
    description: Optional[str] = Field(
        None,
        max_length=1000,
        description="Session description",
    )
    is_active: Optional[bool] = Field(
        None,
        description="Session active status",
    )


class ChatSessionResponse(TimestampSchema):
    """Schema for chat session response."""
    
    id: str = Field(
        ...,
        description="Session unique identifier",
    )
    user_id: str = Field(
        ...,
        description="Session owner ID",
    )
    title: str = Field(
        ...,
        description="Session display title",
    )
    description: Optional[str] = Field(
        None,
        description="Session description",
    )
    is_active: bool = Field(
        ...,
        description="Whether session is active",
    )
    message_count: int = Field(
        ...,
        description="Number of messages in session",
    )
    messages: Optional[List[ChatMessageResponse]] = Field(
        None,
        description="Session messages (when requested)",
    )


class ChatCompletionRequest(BaseSchema):
    """Schema for chat completion request."""
    
    session_id: str = Field(
        ...,
        description="Target session ID",
    )
    message: str = Field(
        ...,
        min_length=1,
        max_length=50000,
        description="User message content",
    )
    model: Optional[str] = Field(
        "gpt-4",
        description="AI model to use",
    )
    temperature: Optional[float] = Field(
        0.7,
        ge=0,
        le=2,
        description="Response randomness",
    )
    max_tokens: Optional[int] = Field(
        2048,
        ge=1,
        le=128000,
        description="Maximum response tokens",
    )


class ChatCompletionResponse(BaseSchema):
    """Schema for chat completion response."""
    
    message: ChatMessageResponse = Field(
        ...,
        description="AI response message",
    )
    usage: dict = Field(
        ...,
        description="Token usage statistics",
    )
