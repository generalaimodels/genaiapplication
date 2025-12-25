# =============================================================================
# SOTA AUTHENTICATION SYSTEM - SESSION ROUTES
# =============================================================================
# File: api/v1/session_routes.py
# Description: Session management API endpoints
# =============================================================================

from typing import Optional

from fastapi import APIRouter, Request, Depends, status, Path

from auth.schemas import (
    SessionResponse,
    SessionListResponse,
    MessageResponse,
)
from auth.service import AuthService
from auth.dependencies import (
    AuthServiceDep,
    ActiveUser,
    CurrentToken,
    get_client_ip,
    get_user_agent,
)


router = APIRouter(prefix="/sessions", tags=["Sessions"])


# =============================================================================
# SESSION LIST
# =============================================================================

@router.get(
    "/",
    response_model=SessionListResponse,
    summary="List all active sessions",
    description="Get a list of all active sessions for the authenticated user.",
)
async def list_sessions(
    user: ActiveUser,
    token: CurrentToken,
    auth_service: AuthServiceDep,
) -> SessionListResponse:
    """
    List all active sessions.
    
    Returns sessions with device info, IP addresses, and activity timestamps.
    Current session is marked with `is_current: true`.
    """
    return await auth_service.get_user_sessions(
        user_id=user.id,
        current_session_id=token.session_id,
    )


# =============================================================================
# SESSION DETAILS
# =============================================================================

@router.get(
    "/{session_id}",
    response_model=SessionResponse,
    summary="Get session details",
    description="Get detailed information about a specific session.",
)
async def get_session(
    session_id: str,
    user: ActiveUser,
    token: CurrentToken,
    auth_service: AuthServiceDep,
) -> SessionResponse:
    """
    Get session details.
    
    Returns detailed information about the specified session.
    """
    sessions = await auth_service.get_user_sessions(
        user_id=user.id,
        current_session_id=token.session_id,
    )
    
    for session in sessions.sessions:
        if session.session_id == session_id:
            return session
    
    from core.exceptions import SessionNotFoundError
    raise SessionNotFoundError(session_id=session_id)


# =============================================================================
# SESSION REVOCATION
# =============================================================================

@router.delete(
    "/{session_id}",
    response_model=MessageResponse,
    summary="Revoke a session",
    description="Revoke a specific session by ID.",
)
async def revoke_session(
    request: Request,
    session_id: str,
    user: ActiveUser,
    auth_service: AuthServiceDep,
    user_agent: Optional[str] = Depends(get_user_agent),
) -> MessageResponse:
    """
    Revoke a specific session.
    
    Terminates the specified session simultaneously logging out
    that device/browser.
    """
    ip_address = get_client_ip(request)
    
    await auth_service.revoke_session(
        user_id=user.id,
        session_id=session_id,
        ip_address=ip_address,
        user_agent=user_agent,
    )
    
    return MessageResponse(
        message=f"Session {session_id} revoked successfully",
        success=True,
    )


@router.delete(
    "/",
    response_model=MessageResponse,
    summary="Revoke all sessions",
    description="Revoke all sessions except the current one.",
)
async def revoke_all_sessions(
    request: Request,
    user: ActiveUser,
    token: CurrentToken,
    auth_service: AuthServiceDep,
    user_agent: Optional[str] = Depends(get_user_agent),
) -> MessageResponse:
    """
    Revoke all sessions except current.
    
    Logs out from all other devices while keeping the current session active.
    """
    ip_address = get_client_ip(request)
    
    count = await auth_service.logout_all(
        user_id=user.id,
        ip_address=ip_address,
        user_agent=user_agent,
        except_current=token.session_id,
    )
    
    return MessageResponse(
        message=f"Revoked {count} sessions",
        success=True,
    )
