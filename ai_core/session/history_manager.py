# =============================================================================
# HISTORY MANAGER - Conversation History Tracking
# =============================================================================
# Manages conversation history with windowing, summarization, and persistence.
# =============================================================================

from __future__ import annotations
import threading
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
import logging

from clients.base_client import ChatMessage

logger = logging.getLogger(__name__)


@dataclass
class ConversationHistory:
    """
    Represents conversation history for a session.
    
    Attributes:
        session_id: Associated session identifier
        messages: List of conversation messages
        max_messages: Maximum messages to retain
        created_at: History creation timestamp
        summary: Optional summary of older messages
    """
    session_id: str
    messages: List[ChatMessage] = field(default_factory=list)
    max_messages: int = 50
    created_at: datetime = field(default_factory=datetime.now)
    summary: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def message_count(self) -> int:
        """Get number of messages."""
        return len(self.messages)
    
    @property
    def is_empty(self) -> bool:
        """Check if history is empty."""
        return len(self.messages) == 0
    
    def add_message(self, message: ChatMessage) -> None:
        """Add a message to history."""
        self.messages.append(message)
        
        # Trim if exceeds max
        if len(self.messages) > self.max_messages:
            self._trim_history()
    
    def add_user_message(self, content: str, **metadata) -> ChatMessage:
        """Add a user message."""
        msg = ChatMessage(role="user", content=content, metadata=metadata)
        self.add_message(msg)
        return msg
    
    def add_assistant_message(self, content: str, **metadata) -> ChatMessage:
        """Add an assistant message."""
        msg = ChatMessage(role="assistant", content=content, metadata=metadata)
        self.add_message(msg)
        return msg
    
    def add_system_message(self, content: str) -> ChatMessage:
        """Add a system message."""
        msg = ChatMessage(role="system", content=content)
        self.add_message(msg)
        return msg
    
    def _trim_history(self) -> None:
        """Trim history to max messages, preserving system message."""
        if len(self.messages) <= self.max_messages:
            return
        
        # Preserve system message if exists
        system_msg = None
        if self.messages and self.messages[0].role == "system":
            system_msg = self.messages[0]
        
        # Keep recent messages
        keep_count = self.max_messages - (1 if system_msg else 0)
        self.messages = (
            ([system_msg] if system_msg else []) + 
            self.messages[-keep_count:]
        )
    
    def get_window(
        self,
        n: Optional[int] = None,
        include_system: bool = True
    ) -> List[ChatMessage]:
        """
        Get recent messages window.
        
        Args:
            n: Number of recent messages (None = all)
            include_system: Include system message
            
        Returns:
            List of messages
        """
        if n is None:
            return self.messages.copy()
        
        result = []
        
        # Add system message if exists and requested
        if include_system and self.messages and self.messages[0].role == "system":
            result.append(self.messages[0])
            remaining = self.messages[1:]
        else:
            remaining = self.messages
        
        # Add recent messages
        result.extend(remaining[-n:])
        return result
    
    def get_last_user_message(self) -> Optional[ChatMessage]:
        """Get the last user message."""
        for msg in reversed(self.messages):
            if msg.role == "user":
                return msg
        return None
    
    def get_last_assistant_message(self) -> Optional[ChatMessage]:
        """Get the last assistant message."""
        for msg in reversed(self.messages):
            if msg.role == "assistant":
                return msg
        return None
    
    def clear(self, keep_system: bool = True) -> None:
        """Clear history, optionally keeping system message."""
        if keep_system and self.messages and self.messages[0].role == "system":
            self.messages = [self.messages[0]]
        else:
            self.messages = []
    
    def to_dict_list(self) -> List[Dict[str, Any]]:
        """Convert messages to list of dicts."""
        return [msg.to_dict() for msg in self.messages]


class HistoryManager:
    """
    Manages conversation histories for multiple sessions.
    
    Features:
        - Per-session history tracking
        - Automatic history windowing
        - Thread-safe operations
        - History export/import
    
    Example:
        >>> manager = HistoryManager(max_messages_per_session=50)
        >>> 
        >>> # Add messages
        >>> manager.add_message("session-1", ChatMessage(role="user", content="Hi"))
        >>> manager.add_message("session-1", ChatMessage(role="assistant", content="Hello!"))
        >>> 
        >>> # Get history for API call
        >>> messages = manager.get_messages("session-1")
    """
    
    def __init__(
        self,
        max_messages_per_session: int = 50,
        default_system_prompt: Optional[str] = None
    ):
        """
        Initialize history manager.
        
        Args:
            max_messages_per_session: Maximum messages per session
            default_system_prompt: Default system prompt for new histories
        """
        self.max_messages = max_messages_per_session
        self.default_system_prompt = default_system_prompt
        
        self._histories: Dict[str, ConversationHistory] = {}
        self._lock = threading.RLock()
    
    def get_or_create_history(
        self,
        session_id: str,
        system_prompt: Optional[str] = None
    ) -> ConversationHistory:
        """
        Get existing history or create new one.
        
        Args:
            session_id: Session identifier
            system_prompt: System prompt for new history
            
        Returns:
            ConversationHistory object
        """
        with self._lock:
            if session_id not in self._histories:
                history = ConversationHistory(
                    session_id=session_id,
                    max_messages=self.max_messages
                )
                
                # Add system prompt if provided
                prompt = system_prompt or self.default_system_prompt
                if prompt:
                    history.add_system_message(prompt)
                
                self._histories[session_id] = history
            
            return self._histories[session_id]
    
    def add_message(
        self,
        session_id: str,
        message: ChatMessage
    ) -> None:
        """
        Add a message to session history.
        
        Args:
            session_id: Session identifier
            message: Message to add
        """
        with self._lock:
            history = self.get_or_create_history(session_id)
            history.add_message(message)
    
    def add_user_message(self, session_id: str, content: str) -> ChatMessage:
        """Add a user message."""
        with self._lock:
            history = self.get_or_create_history(session_id)
            return history.add_user_message(content)
    
    def add_assistant_message(self, session_id: str, content: str) -> ChatMessage:
        """Add an assistant message."""
        with self._lock:
            history = self.get_or_create_history(session_id)
            return history.add_assistant_message(content)
    
    def get_messages(
        self,
        session_id: str,
        n: Optional[int] = None,
        include_system: bool = True
    ) -> List[ChatMessage]:
        """
        Get messages for a session.
        
        Args:
            session_id: Session identifier
            n: Number of recent messages (None = all)
            include_system: Include system message
            
        Returns:
            List of ChatMessage objects
        """
        with self._lock:
            history = self._histories.get(session_id)
            if history is None:
                return []
            return history.get_window(n, include_system)
    
    def set_system_prompt(
        self,
        session_id: str,
        prompt: str
    ) -> None:
        """
        Set system prompt for a session.
        
        Args:
            session_id: Session identifier
            prompt: System prompt text
        """
        with self._lock:
            history = self.get_or_create_history(session_id)
            
            # Remove existing system message if present
            if history.messages and history.messages[0].role == "system":
                history.messages.pop(0)
            
            # Insert new system message at start
            history.messages.insert(0, ChatMessage(role="system", content=prompt))
    
    def clear_history(
        self,
        session_id: str,
        keep_system: bool = True
    ) -> None:
        """
        Clear history for a session.
        
        Args:
            session_id: Session identifier
            keep_system: Keep system message
        """
        with self._lock:
            history = self._histories.get(session_id)
            if history:
                history.clear(keep_system)
    
    def delete_history(self, session_id: str) -> bool:
        """
        Delete history for a session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            True if deleted, False if not found
        """
        with self._lock:
            if session_id in self._histories:
                del self._histories[session_id]
                return True
            return False
    
    def get_message_count(self, session_id: str) -> int:
        """Get message count for session."""
        with self._lock:
            history = self._histories.get(session_id)
            return history.message_count if history else 0
    
    def export_history(self, session_id: str) -> List[Dict[str, Any]]:
        """Export history as list of dicts."""
        with self._lock:
            history = self._histories.get(session_id)
            return history.to_dict_list() if history else []
    
    def import_history(
        self,
        session_id: str,
        messages: List[Dict[str, Any]]
    ) -> None:
        """
        Import history from list of dicts.
        
        Args:
            session_id: Session identifier
            messages: List of message dicts
        """
        with self._lock:
            history = self.get_or_create_history(session_id)
            history.messages = [
                ChatMessage(
                    role=msg["role"],
                    content=msg["content"],
                    name=msg.get("name"),
                    function_call=msg.get("function_call"),
                    tool_calls=msg.get("tool_calls"),
                )
                for msg in messages
            ]
