# ==============================================================================
# CHAT ENDPOINTS - Chat Session & Message Routes
# ==============================================================================
# Chat session management and messaging endpoints
# ==============================================================================

from __future__ import annotations

from typing import List

from fastapi import APIRouter, HTTPException, Query, status

from app.api.dependencies import ChatServiceDep, CurrentUserID
from app.core.exceptions import NotFoundError, AuthorizationError
from app.schemas.base import APIResponse
from app.schemas.chat import (
    ChatSessionCreate,
    ChatSessionUpdate,
    ChatSessionResponse,
    ChatMessageCreate,
    ChatMessageResponse,
)

router = APIRouter(prefix="/chat", tags=["Chat"])


# ==============================================================================
# SESSION ENDPOINTS
# ==============================================================================

@router.post(
    "/sessions",
    response_model=APIResponse[ChatSessionResponse],
    status_code=status.HTTP_201_CREATED,
    summary="Create chat session",
    description="Create a new chat session for the current user.",
)
async def create_session(
    user_id: CurrentUserID,
    schema: ChatSessionCreate,
    service: ChatServiceDep,
) -> APIResponse[ChatSessionResponse]:
    """Create a new chat session."""
    session = await service.create_session(user_id, schema)
    return APIResponse.ok(data=session, message="Session created successfully")


@router.get(
    "/sessions",
    response_model=APIResponse[List[ChatSessionResponse]],
    summary="List chat sessions",
    description="Get all chat sessions for the current user.",
)
async def list_sessions(
    user_id: CurrentUserID,
    service: ChatServiceDep,
    skip: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100),
) -> APIResponse[List[ChatSessionResponse]]:
    """Get user's chat sessions."""
    sessions = await service.get_user_sessions(user_id, skip, limit)
    return APIResponse.ok(data=sessions)


@router.get(
    "/sessions/{session_id}",
    response_model=APIResponse[ChatSessionResponse],
    summary="Get chat session",
    description="Get a specific chat session with optional messages.",
)
async def get_session(
    session_id: str,
    user_id: CurrentUserID,
    service: ChatServiceDep,
    include_messages: bool = Query(False),
) -> APIResponse[ChatSessionResponse]:
    """Get a chat session by ID."""
    try:
        if include_messages:
            session = await service.get_session_with_messages(session_id, user_id)
        else:
            session = await service.get_session(session_id, user_id)
        return APIResponse.ok(data=session)
    except NotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found",
        )
    except AuthorizationError:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to access this session",
        )


@router.patch(
    "/sessions/{session_id}",
    response_model=APIResponse[ChatSessionResponse],
    summary="Update chat session",
    description="Update a chat session's title or status.",
)
async def update_session(
    session_id: str,
    user_id: CurrentUserID,
    schema: ChatSessionUpdate,
    service: ChatServiceDep,
) -> APIResponse[ChatSessionResponse]:
    """Update a chat session."""
    try:
        session = await service.update_session(session_id, user_id, schema)
        return APIResponse.ok(data=session, message="Session updated successfully")
    except NotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found",
        )
    except AuthorizationError:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to update this session",
        )


@router.delete(
    "/sessions/{session_id}",
    response_model=APIResponse[dict],
    summary="Delete chat session",
    description="Delete a chat session and all its messages.",
)
async def delete_session(
    session_id: str,
    user_id: CurrentUserID,
    service: ChatServiceDep,
) -> APIResponse[dict]:
    """Delete a chat session."""
    try:
        await service.delete_session(session_id, user_id)
        return APIResponse.ok(
            data={"deleted": True},
            message="Session deleted successfully",
        )
    except NotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found",
        )
    except AuthorizationError:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to delete this session",
        )


# ==============================================================================
# MESSAGE ENDPOINTS
# ==============================================================================

@router.post(
    "/sessions/{session_id}/messages",
    response_model=APIResponse[ChatMessageResponse],
    status_code=status.HTTP_201_CREATED,
    summary="Add message",
    description="Add a message to a chat session.",
)
async def add_message(
    session_id: str,
    user_id: CurrentUserID,
    schema: ChatMessageCreate,
    service: ChatServiceDep,
) -> APIResponse[ChatMessageResponse]:
    """Add a message to a session."""
    try:
        message = await service.add_message(session_id, user_id, schema)
        return APIResponse.ok(data=message, message="Message added successfully")
    except NotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found",
        )
    except AuthorizationError:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to add messages to this session",
        )


@router.get(
    "/sessions/{session_id}/messages",
    response_model=APIResponse[List[ChatMessageResponse]],
    summary="Get messages",
    description="Get all messages in a chat session.",
)
async def get_messages(
    session_id: str,
    user_id: CurrentUserID,
    service: ChatServiceDep,
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=500),
) -> APIResponse[List[ChatMessageResponse]]:
    """Get messages in a session."""
    try:
        messages = await service.get_session_messages(
            session_id, user_id, skip, limit
        )
        return APIResponse.ok(data=messages)
    except NotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found",
        )
    except AuthorizationError:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to view this session",
        )
