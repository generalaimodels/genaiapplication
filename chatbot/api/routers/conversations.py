# -*- coding: utf-8 -*-
# =================================================================================================
# api/routers/conversations.py — Chat Session & History Management Endpoints
# =================================================================================================
# Production-grade conversation management implementing:
#
#   1. SESSION CRUD: Create, read, update, delete chat sessions.
#   2. HISTORY MANAGEMENT: Track query-answer pairs within sessions.
#   3. CONTEXT BUILDING: Build RAG context for queries.
#   4. BRANCHING: Fork conversations for exploring alternatives.
#
# Endpoints:
# ----------
#   POST   /sessions               Create new session
#   GET    /sessions               List sessions
#   GET    /sessions/{id}          Get session details
#   PATCH  /sessions/{id}          Update session
#   DELETE /sessions/{id}          Soft delete session
#   GET    /sessions/{id}/history  Get conversation history
#   POST   /sessions/{id}/messages Add message to history
#   POST   /sessions/{id}/context  Build context for query
#   POST   /sessions/{id}/branch   Create branch from session
#
# =================================================================================================

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status

from api.schemas import (
    ContextResponse,
    HistoryEntry,
    HistoryListResponse,
    MessageCreate,
    SessionCreate,
    SessionResponse,
    SessionListResponse,
    SessionUpdate,
    ContextChunk,
    FeedbackRequest,
)
from api.exceptions import NotFoundError, ValidationError, assert_found
from api.dependencies import (
    get_session_repo,
    get_history_repo,
    get_chat_history,
    get_request_id,
    verify_api_key,
)
from api.database import SessionRepository, HistoryRepository

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
_LOG = logging.getLogger("api.routers.conversations")

# -----------------------------------------------------------------------------
# Router Configuration
# -----------------------------------------------------------------------------
router = APIRouter(
    prefix="/sessions",
    tags=["Sessions"],
    responses={
        404: {"description": "Session not found"},
        422: {"description": "Validation error"},
    },
)


# =============================================================================
# Session CRUD Endpoints
# =============================================================================

@router.post(
    "",
    response_model=SessionResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create Session",
    description="""
    Create a new chat session.
    
    **Features**:
    - Auto-generates session ID (UUID v4)
    - Auto-generates conversation ID if not provided
    - Supports custom metadata
    - Returns complete session object
    
    **Example**:
    ```json
    {
        "user_id": "user_123",
        "title": "Trip Planning to Japan",
        "metadata": {"source": "web_app", "locale": "en-US"}
    }
    ```
    """,
)
async def create_session(
    request: SessionCreate,
    sessions: SessionRepository = Depends(get_session_repo),
) -> SessionResponse:
    """Create a new chat session."""
    session = await sessions.create(
        user_id=request.user_id,
        conv_id=request.conv_id,
        branch_id=request.branch_id,
        title=request.title,
        metadata=request.metadata,
    )
    
    _LOG.info("Session created: %s (user=%s)", session["id"], request.user_id)
    
    return SessionResponse(**session)


@router.get(
    "",
    response_model=SessionListResponse,
    summary="List Sessions",
    description="""
    List all active sessions with pagination.
    
    **Query Parameters**:
    - `user_id`: Filter by user (optional)
    - `limit`: Max results (default: 50, max: 100)
    - `offset`: Pagination offset
    """,
)
async def list_sessions(
    user_id: Optional[str] = Query(None, max_length=256),
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0, ge=0),
    sessions: SessionRepository = Depends(get_session_repo),
) -> SessionListResponse:
    """List sessions with optional user filter."""
    session_list = await sessions.list(user_id=user_id, limit=limit, offset=offset)
    
    return SessionListResponse(
        sessions=[SessionResponse(**s) for s in session_list],
        total=len(session_list),  # Would need count query for accurate total
        limit=limit,
        offset=offset,
    )


@router.get(
    "/{session_id}",
    response_model=SessionResponse,
    summary="Get Session",
    description="Get session details by ID.",
)
async def get_session(
    session_id: str,
    sessions: SessionRepository = Depends(get_session_repo),
) -> SessionResponse:
    """Get session by ID."""
    session = assert_found(
        await sessions.get(session_id),
        "Session",
        session_id,
    )
    return SessionResponse(**session)


@router.patch(
    "/{session_id}",
    response_model=SessionResponse,
    summary="Update Session",
    description="""
    Update session properties.
    
    **Optimistic Locking**:
    Pass the current `version` to prevent lost updates in concurrent scenarios.
    If version doesn't match, returns 409 Conflict.
    """,
)
async def update_session(
    session_id: str,
    request: SessionUpdate,
    sessions: SessionRepository = Depends(get_session_repo),
) -> SessionResponse:
    """Update session with optimistic locking."""
    # Verify session exists
    existing = await sessions.get(session_id)
    if existing is None:
        raise NotFoundError("Session", session_id)
    
    # Update with version check
    updated = await sessions.update(
        session_id=session_id,
        title=request.title,
        metadata=request.metadata,
        version=request.version,
    )
    
    if updated is None and request.version is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Version mismatch - session was modified by another request",
        )
    
    return SessionResponse(**updated or existing)


@router.delete(
    "/{session_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete Session",
    description="Soft delete a session (can be recovered).",
)
async def delete_session(
    session_id: str,
    sessions: SessionRepository = Depends(get_session_repo),
) -> None:
    """Soft delete session."""
    deleted = await sessions.delete(session_id)
    if not deleted:
        raise NotFoundError("Session", session_id)
    
    _LOG.info("Session deleted: %s", session_id)


# =============================================================================
# History Endpoints
# =============================================================================

@router.get(
    "/{session_id}/history",
    response_model=HistoryListResponse,
    summary="Get Session History",
    description="""
    Get conversation history for a session.
    
    Returns messages in chronological order with:
    - User queries
    - AI answers
    - Retrieved context references
    - Token counts and latency
    """,
)
async def get_session_history(
    session_id: str,
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    sessions: SessionRepository = Depends(get_session_repo),
    history_repo: HistoryRepository = Depends(get_history_repo),
) -> HistoryListResponse:
    """Get conversation history for session."""
    # Verify session exists
    session = assert_found(
        await sessions.get(session_id),
        "Session",
        session_id,
    )
    
    entries = await history_repo.list_by_session(
        session_id=session_id,
        limit=limit,
        offset=offset,
    )
    
    return HistoryListResponse(
        entries=[HistoryEntry(**e) for e in entries],
        total=len(entries),
        session_id=session_id,
    )


@router.post(
    "/{session_id}/messages",
    response_model=HistoryEntry,
    status_code=status.HTTP_201_CREATED,
    summary="Add Message",
    description="""
    Add a message to the conversation history.
    
    Used for:
    - Recording user queries
    - Recording AI responses
    - Adding system messages
    
    For chat with AI response, use `/chat` endpoint instead.
    """,
)
async def add_message(
    session_id: str,
    request: MessageCreate,
    sessions: SessionRepository = Depends(get_session_repo),
    history_repo: HistoryRepository = Depends(get_history_repo),
) -> HistoryEntry:
    """Add message to session history."""
    # Verify session exists
    session = assert_found(
        await sessions.get(session_id),
        "Session",
        session_id,
    )
    
    # Estimate token count (simple approximation)
    tokens = max(1, len(request.query) // 4)
    
    entry = await history_repo.create(
        session_id=session_id,
        query=request.query,
        role=request.role,
        metadata=request.metadata,
        tokens_query=tokens,
    )
    
    # Also add to chat history engine if available
    try:
        chat_history = get_chat_history()
        from history import Role
        role_map = {"user": Role.USER, "assistant": Role.ASSISTANT, "system": Role.SYSTEM}
        role = role_map.get(request.role, Role.USER)
        
        # Get next message ID
        entries = await history_repo.list_by_session(session_id)
        msg_id = len(entries)
        
        chat_history.add_message(
            conv_id=session["conv_id"],
            branch_id=session["branch_id"],
            msg_id=msg_id,
            role=role,
            content=request.query,
            tokens=tokens,
        )
    except Exception as e:
        _LOG.warning("Failed to add to chat history engine: %s", e)
    
    return HistoryEntry(**entry)


@router.post(
    "/{session_id}/messages/{message_id}/feedback",
    response_model=HistoryEntry,
    summary="Submit Feedback",
    description="Submit user feedback (like/dislike) for a specific message.",
)
async def submit_feedback(
    session_id: str,
    message_id: str,
    request: FeedbackRequest,
    sessions: SessionRepository = Depends(get_session_repo),
    history_repo: HistoryRepository = Depends(get_history_repo),
) -> HistoryEntry:
    """Submit feedback for a message."""
    # Verify session exists
    await assert_found(await sessions.get(session_id), "Session", session_id)
    
    # Update feedback
    updated = await history_repo.update_feedback(
        history_id=message_id,
        score=request.score,
        comment=request.comment,
    )
    
    if not updated:
        raise NotFoundError("Message", message_id)
        
    return HistoryEntry(**updated)


# =============================================================================
# Context Building Endpoint
# =============================================================================

@router.post(
    "/{session_id}/context",
    response_model=ContextResponse,
    summary="Build Context",
    description="""
    Build RAG context for a query.
    
    Retrieves relevant messages from:
    - Semantic search (similar content)
    - Recent conversation history
    - Deduplicated and ranked by relevance
    
    **Token Budget**: Maximum tokens for context (prevents exceeding model limits).
    """,
)
async def build_context(
    session_id: str,
    request: ContextRequest,
    sessions: SessionRepository = Depends(get_session_repo),
    history_repo: HistoryRepository = Depends(get_history_repo),
) -> ContextResponse:
    """Build context for query using chat history engine."""
    # Verify session exists
    session = assert_found(
        await sessions.get(session_id),
        "Session",
        session_id,
    )
    
    messages: List[HistoryEntry] = []
    sources: List[ContextChunk] = []
    total_tokens = 0
    
    try:
        # Use chat history engine for context building
        chat_history = get_chat_history()
        
        context_messages = chat_history.build_context(
            conv_id=session["conv_id"],
            branch_id=session["branch_id"],
            query_text=request.query,
            budget_ctx=request.token_budget,
        )
        
        # Convert to response format
        for conv_id, branch_id, msg_id, content in context_messages:
            # Estimate tokens
            tokens = max(1, len(content) // 4)
            total_tokens += tokens
            
            sources.append(ContextChunk(
                text=content,
                score=1.0,  # Would need actual scores from engine
                metadata={"conv_id": conv_id, "branch_id": branch_id, "msg_id": msg_id},
            ))
    except Exception as e:
        _LOG.warning("Chat history context failed, using recent messages: %s", e)
        
        # Fallback to recent messages from DB
        entries = await history_repo.list_by_session(
            session_id=session_id,
            limit=request.include_recent,
        )
        messages = [HistoryEntry(**e) for e in entries]
        total_tokens = sum(
            (e.tokens_query or 0) + (e.tokens_answer or 0)
            for e in messages
        )
    
    return ContextResponse(
        messages=messages,
        total_tokens=total_tokens,
        sources=sources,
    )


# =============================================================================
# Branching Endpoint
# =============================================================================

@router.post(
    "/{session_id}/branch",
    response_model=SessionResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Branch Session",
    description="""
    Create a new branch from the current session.
    
    **Use Case**: Explore alternative conversation paths without modifying original.
    
    The new session:
    - Gets a new session ID
    - Inherits conversation history up to branch point
    - Uses a new branch_id internally
    """,
)
async def branch_session(
    session_id: str,
    branch_name: str = Query(..., min_length=1, max_length=64),
    from_message: Optional[int] = Query(None, ge=0, description="Branch from this message index"),
    sessions: SessionRepository = Depends(get_session_repo),
) -> SessionResponse:
    """Create a branch from existing session."""
    # Verify source session exists
    source = assert_found(
        await sessions.get(session_id),
        "Session",
        session_id,
    )
    
    # Create new session with branch
    new_branch_id = f"{source['branch_id']}/{branch_name}"
    
    new_session = await sessions.create(
        user_id=source.get("user_id"),
        conv_id=source["conv_id"],
        branch_id=new_branch_id,
        title=f"{source.get('title', 'Untitled')} (Branch: {branch_name})",
        metadata={
            "branched_from": session_id,
            "branch_name": branch_name,
            "branch_point": from_message,
        },
    )
    
    # Create branch in chat history engine
    try:
        chat_history = get_chat_history()
        chat_history.branch(
            conv_id=source["conv_id"],
            src_branch=source["branch_id"],
            new_branch=new_branch_id,
            from_msg_id=from_message,
        )
        _LOG.info("Branch created: %s -> %s", source["branch_id"], new_branch_id)
    except Exception as e:
        _LOG.warning("Failed to create branch in chat history: %s", e)
    
    return SessionResponse(**new_session)
