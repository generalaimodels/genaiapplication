# =============================================================================
# SOTA AUTHENTICATION SYSTEM - USER ROUTES
# =============================================================================
# File: api/v1/user_routes.py
# Description: User profile and management API endpoints
# =============================================================================

from typing import Optional

from fastapi import APIRouter, Request, Depends, status

from auth.schemas import (
    UserResponse,
    UserUpdate,
    MessageResponse,
    SessionListResponse,
)
from auth.service import AuthService
from auth.dependencies import (
    AuthServiceDep,
    ActiveUser,
    CurrentToken,
    get_client_ip,
    get_user_agent,
)


router = APIRouter(prefix="/users", tags=["Users"])


# =============================================================================
# CURRENT USER PROFILE
# =============================================================================

@router.get(
    "/me",
    response_model=UserResponse,
    summary="Get current user profile",
    description="Retrieve the profile of the currently authenticated user.",
)
async def get_current_user_profile(
    user: ActiveUser,
) -> UserResponse:
    """
    Get current user's profile.
    
    Returns user information for the authenticated user.
    """
    return UserResponse.model_validate(user)


@router.patch(
    "/me",
    response_model=UserResponse,
    summary="Update current user profile",
    description="Update profile information for the currently authenticated user.",
)
async def update_current_user_profile(
    update_data: UserUpdate,
    user: ActiveUser,
    auth_service: AuthServiceDep,
) -> UserResponse:
    """
    Update current user's profile.
    
    - **username**: New username (optional)
    - **email**: New email address (optional, requires re-verification)
    """
    return await auth_service.update_user(
        user_id=user.id,
        email=update_data.email,
        username=update_data.username,
    )


@router.delete(
    "/me",
    response_model=MessageResponse,
    summary="Delete current user account",
    description="Permanently delete the authenticated user's account.",
)
async def delete_current_user(
    request: Request,
    user: ActiveUser,
    auth_service: AuthServiceDep,
    user_agent: Optional[str] = Depends(get_user_agent),
) -> MessageResponse:
    """
    Delete current user's account.
    
    This action is permanent and cannot be undone.
    """
    ip_address = get_client_ip(request)
    
    await auth_service.delete_user(
        user_id=user.id,
        ip_address=ip_address,
        user_agent=user_agent,
    )
    
    return MessageResponse(
        message="Account deleted successfully",
        success=True,
    )


# =============================================================================
# USER SESSIONS
# =============================================================================

@router.get(
    "/me/sessions",
    response_model=SessionListResponse,
    summary="Get user's active sessions",
    description="List all active sessions for the current user.",
)
async def get_user_sessions(
    user: ActiveUser,
    token: CurrentToken,
    auth_service: AuthServiceDep,
) -> SessionListResponse:
    """
    Get all active sessions.
    
    Returns a list of all active sessions for the current user,
    including device information and activity timestamps.
    """
    return await auth_service.get_user_sessions(
        user_id=user.id,
        current_session_id=token.session_id,
    )
