# ==============================================================================
# CHAT MODELS - Chatbot Functionality
# ==============================================================================
# Chat sessions and messages for AI-powered chatbot features
# ==============================================================================

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional

from sqlalchemy import ForeignKey, String, Text, Integer
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.domain_models.base import SQLBase, TimestampMixin

if TYPE_CHECKING:
    from app.domain_models.user import User


class ChatSession(SQLBase, TimestampMixin):
    """
    Chat session representing a conversation thread.
    
    Groups related messages together under a single conversation
    context for a specific user.
    
    Attributes:
        user_id: Owner of this chat session
        title: Display title for the session
        description: Optional session description
        is_active: Whether session is still active
        
    Relationships:
        user: Session owner
        messages: Messages in this session
    """
    
    __tablename__ = "chat_sessions"
    
    # Foreign keys
    user_id: Mapped[str] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"),
        index=True,
        nullable=False,
    )
    
    # Session metadata
    title: Mapped[str] = mapped_column(
        String(255),
        default="New Chat",
        nullable=False,
    )
    description: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
    )
    
    # Status
    is_active: Mapped[bool] = mapped_column(
        default=True,
        nullable=False,
    )
    
    # Message count for quick reference
    message_count: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False,
    )
    
    # Relationships
    user: Mapped["User"] = relationship(
        "User",
        back_populates="chat_sessions",
    )
    messages: Mapped[List["ChatMessage"]] = relationship(
        "ChatMessage",
        back_populates="session",
        cascade="all, delete-orphan",
        order_by="ChatMessage.created_at",
    )
    
    def __repr__(self) -> str:
        return f"<ChatSession(id={self.id}, title={self.title})>"


class ChatMessage(SQLBase, TimestampMixin):
    """
    Individual message within a chat session.
    
    Stores message content with role designation (user/assistant/system)
    and optional metadata for AI processing.
    
    Attributes:
        session_id: Parent chat session
        role: Message sender role (user/assistant/system)
        content: Message text content
        tokens: Token count for AI processing
        
    Relationships:
        session: Parent chat session
    """
    
    __tablename__ = "chat_messages"
    
    # Foreign keys
    session_id: Mapped[str] = mapped_column(
        ForeignKey("chat_sessions.id", ondelete="CASCADE"),
        index=True,
        nullable=False,
    )
    
    # Message content
    role: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
    )
    content: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )
    
    # AI metadata
    tokens: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True,
    )
    model: Mapped[Optional[str]] = mapped_column(
        String(100),
        nullable=True,
    )
    
    # Relationships
    session: Mapped["ChatSession"] = relationship(
        "ChatSession",
        back_populates="messages",
    )
    
    def __repr__(self) -> str:
        return f"<ChatMessage(id={self.id}, role={self.role})>"
