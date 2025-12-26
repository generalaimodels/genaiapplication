# ==============================================================================
# CHAT SERVICE - Chatbot Functionality
# ==============================================================================
# Business logic for chat sessions and messages
# ==============================================================================

from __future__ import annotations

from typing import Any, Dict, List, Optional

from app.database.adapters.base_adapter import BaseDatabaseAdapter
from app.core.exceptions import NotFoundError, AuthorizationError
from app.schemas.chat import (
    ChatSessionCreate,
    ChatSessionUpdate,
    ChatSessionResponse,
    ChatMessageCreate,
    ChatMessageResponse,
)
from app.services.base_service import BaseService


class ChatService:
    """
    Chat service for managing sessions and messages.
    
    Provides chat session creation, message handling,
    and conversation management.
    """
    
    def __init__(self, adapter: BaseDatabaseAdapter) -> None:
        """Initialize chat service."""
        self._adapter = adapter
        self._sessions_collection = "chat_sessions"
        self._messages_collection = "chat_messages"
    
    # ==========================================================================
    # SESSIONS
    # ==========================================================================
    
    def _session_to_response(self, entity: Any) -> ChatSessionResponse:
        """Convert session entity to response."""
        if hasattr(entity, "to_dict"):
            return ChatSessionResponse.model_validate(entity.to_dict())
        if isinstance(entity, dict):
            return ChatSessionResponse.model_validate(entity)
        return ChatSessionResponse.model_validate(entity, from_attributes=True)
    
    async def create_session(
        self,
        user_id: str,
        schema: ChatSessionCreate,
    ) -> ChatSessionResponse:
        """
        Create a new chat session.
        
        Args:
            user_id: Owner user ID
            schema: Session creation data
            
        Returns:
            Created session response
        """
        data = schema.model_dump(exclude_unset=True)
        data["user_id"] = user_id
        data["is_active"] = True
        data["message_count"] = 0
        
        result = await self._adapter.create(self._sessions_collection, data)
        return self._session_to_response(result)
    
    async def get_session(
        self,
        session_id: str,
        user_id: str,
    ) -> ChatSessionResponse:
        """
        Get a chat session by ID.
        
        Args:
            session_id: Session ID
            user_id: Requesting user ID (for authorization)
            
        Returns:
            Session response
            
        Raises:
            NotFoundError: If session not found
            AuthorizationError: If user doesn't own session
        """
        result = await self._adapter.get_by_id(
            self._sessions_collection,
            session_id,
        )
        
        if not result:
            raise NotFoundError(
                message="Chat session not found",
                resource_type="chat_session",
                resource_id=session_id,
            )
        
        # Check ownership
        owner_id = (
            result.get("user_id")
            if isinstance(result, dict)
            else getattr(result, "user_id")
        )
        
        if owner_id != user_id:
            raise AuthorizationError(message="Not authorized to access this session")
        
        return self._session_to_response(result)
    
    async def get_user_sessions(
        self,
        user_id: str,
        skip: int = 0,
        limit: int = 20,
    ) -> List[ChatSessionResponse]:
        """Get all sessions for a user."""
        results = await self._adapter.get_all(
            self._sessions_collection,
            skip=skip,
            limit=limit,
            filters={"user_id": user_id},
            sort_by="created_at",
            sort_order="desc",
        )
        return [self._session_to_response(r) for r in results]
    
    async def update_session(
        self,
        session_id: str,
        user_id: str,
        schema: ChatSessionUpdate,
    ) -> ChatSessionResponse:
        """Update a chat session."""
        # Verify ownership first
        await self.get_session(session_id, user_id)
        
        data = schema.model_dump(exclude_unset=True)
        result = await self._adapter.update(
            self._sessions_collection,
            session_id,
            data,
        )
        return self._session_to_response(result)
    
    async def delete_session(
        self,
        session_id: str,
        user_id: str,
    ) -> bool:
        """Delete a chat session and its messages."""
        # Verify ownership first
        await self.get_session(session_id, user_id)
        
        # Delete messages first
        await self._adapter.bulk_delete(
            self._messages_collection,
            {"session_id": session_id},
        )
        
        # Delete session
        return await self._adapter.delete(self._sessions_collection, session_id)
    
    # ==========================================================================
    # MESSAGES
    # ==========================================================================
    
    def _message_to_response(self, entity: Any) -> ChatMessageResponse:
        """Convert message entity to response."""
        if isinstance(entity, dict):
            return ChatMessageResponse.model_validate(entity)
        return ChatMessageResponse.model_validate(entity, from_attributes=True)
    
    async def add_message(
        self,
        session_id: str,
        user_id: str,
        schema: ChatMessageCreate,
    ) -> ChatMessageResponse:
        """
        Add a message to a session.
        
        Args:
            session_id: Target session ID
            user_id: User adding message (for authorization)
            schema: Message data
            
        Returns:
            Created message response
        """
        # Verify session ownership
        await self.get_session(session_id, user_id)
        
        data = schema.model_dump(exclude_unset=True)
        data["session_id"] = session_id
        
        result = await self._adapter.create(self._messages_collection, data)
        
        # Update message count
        session = await self._adapter.get_by_id(
            self._sessions_collection,
            session_id,
        )
        if session:
            current_count = (
                session.get("message_count", 0)
                if isinstance(session, dict)
                else getattr(session, "message_count", 0)
            )
            await self._adapter.update(
                self._sessions_collection,
                session_id,
                {"message_count": current_count + 1},
            )
        
        return self._message_to_response(result)
    
    async def get_session_messages(
        self,
        session_id: str,
        user_id: str,
        skip: int = 0,
        limit: int = 100,
    ) -> List[ChatMessageResponse]:
        """Get all messages in a session."""
        # Verify session ownership
        await self.get_session(session_id, user_id)
        
        results = await self._adapter.get_all(
            self._messages_collection,
            skip=skip,
            limit=limit,
            filters={"session_id": session_id},
            sort_by="created_at",
            sort_order="asc",
        )
        return [self._message_to_response(r) for r in results]
    
    async def get_session_with_messages(
        self,
        session_id: str,
        user_id: str,
    ) -> ChatSessionResponse:
        """Get session with all messages included."""
        session = await self.get_session(session_id, user_id)
        messages = await self.get_session_messages(session_id, user_id)
        
        return ChatSessionResponse(
            **session.model_dump(exclude={"messages"}),
            messages=messages,
        )
