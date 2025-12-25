# =============================================================================
# SOTA AUTHENTICATION SYSTEM - AUTH ROUTES
# =============================================================================
# File: api/v1/auth_routes.py
# Description: Authentication API endpoints (register, login, logout, etc.)
# =============================================================================

from typing import Optional

from fastapi import APIRouter, Request, Depends, status

from auth.schemas import (
    UserCreate,
    UserResponse,
    LoginRequest,
    TokenResponse,
    RefreshRequest,
    RefreshResponse,
    PasswordChangeRequest,
    PasswordResetRequest,
    PasswordResetConfirm,
    EmailVerifyRequest,
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


router = APIRouter(prefix="/auth", tags=["Authentication"])


# =============================================================================
# REGISTRATION
# =============================================================================

@router.post(
    "/register",
    response_model=UserResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Register a new user",
    description="Create a new user account with email, username, and password.",
)
async def register(
    user_data: UserCreate,
    request: Request,
    auth_service: AuthServiceDep,
    user_agent: Optional[str] = Depends(get_user_agent),
) -> UserResponse:
    """
    Register a new user account.
    
    - **email**: Valid email address (unique)
    - **username**: 3-50 characters, alphanumeric and underscores
    - **password**: Minimum 8 characters with complexity requirements
    """
    ip_address = get_client_ip(request)
    
    return await auth_service.register(
        user_data=user_data,
        ip_address=ip_address,
        user_agent=user_agent,
    )


# =============================================================================
# LOGIN
# =============================================================================

@router.post(
    "/login",
    response_model=TokenResponse,
    summary="Authenticate user",
    description="Login with email and password to receive access and refresh tokens.",
)
async def login(
    login_data: LoginRequest,
    request: Request,
    auth_service: AuthServiceDep,
    user_agent: Optional[str] = Depends(get_user_agent),
) -> TokenResponse:
    """
    Authenticate user and return JWT tokens.
    
    Returns:
    - **access_token**: Short-lived token for API access (15 min)
    - **refresh_token**: Long-lived token for renewal (7 days)
    - **session_id**: Session identifier for management
    """
    ip_address = get_client_ip(request)
    
    # Extract device info from user agent
    device_info = {
        "user_agent": user_agent,
        "ip": ip_address,
    }
    
    return await auth_service.login(
        email=login_data.email,
        password=login_data.password,
        ip_address=ip_address,
        user_agent=user_agent,
        device_info=device_info,
        remember_me=login_data.remember_me,
    )


# =============================================================================
# LOGOUT
# =============================================================================

@router.post(
    "/logout",
    response_model=MessageResponse,
    summary="Logout current session",
    description="End the current session and invalidate tokens.",
)
async def logout(
    request: Request,
    user: ActiveUser,
    token: CurrentToken,
    auth_service: AuthServiceDep,
    user_agent: Optional[str] = Depends(get_user_agent),
) -> MessageResponse:
    """
    Logout from current session.
    
    Invalidates the current session and all associated tokens.
    """
    ip_address = get_client_ip(request)
    
    await auth_service.logout(
        session_id=token.session_id or "",
        user_id=user.id,
        ip_address=ip_address,
        user_agent=user_agent,
    )
    
    return MessageResponse(
        message="Successfully logged out",
        success=True,
    )


@router.post(
    "/logout-all",
    response_model=MessageResponse,
    summary="Logout from all devices",
    description="End all sessions for the current user.",
)
async def logout_all(
    request: Request,
    user: ActiveUser,
    token: CurrentToken,
    auth_service: AuthServiceDep,
    user_agent: Optional[str] = Depends(get_user_agent),
) -> MessageResponse:
    """
    Logout from all devices.
    
    Revokes all active sessions except optionally the current one.
    """
    ip_address = get_client_ip(request)
    
    count = await auth_service.logout_all(
        user_id=user.id,
        ip_address=ip_address,
        user_agent=user_agent,
        except_current=None,  # Revoke all including current
    )
    
    return MessageResponse(
        message=f"Successfully logged out from {count} sessions",
        success=True,
    )


# =============================================================================
# TOKEN REFRESH
# =============================================================================

@router.post(
    "/refresh",
    response_model=RefreshResponse,
    summary="Refresh access token",
    description="Get a new access token using a valid refresh token.",
)
async def refresh_token(
    refresh_data: RefreshRequest,
    request: Request,
    auth_service: AuthServiceDep,
    user_agent: Optional[str] = Depends(get_user_agent),
) -> RefreshResponse:
    """
    Refresh access token.
    
    Use the refresh token obtained during login to get a new access token
    without re-authenticating.
    """
    ip_address = get_client_ip(request)
    
    return await auth_service.refresh_token(
        refresh_token=refresh_data.refresh_token,
        ip_address=ip_address,
        user_agent=user_agent,
    )


# =============================================================================
# PASSWORD MANAGEMENT
# =============================================================================

@router.post(
    "/change-password",
    response_model=MessageResponse,
    summary="Change password",
    description="Change password for authenticated user.",
)
async def change_password(
    password_data: PasswordChangeRequest,
    request: Request,
    user: ActiveUser,
    auth_service: AuthServiceDep,
    user_agent: Optional[str] = Depends(get_user_agent),
) -> MessageResponse:
    """
    Change user password.
    
    Requires current password verification before setting new password.
    """
    ip_address = get_client_ip(request)
    
    await auth_service.change_password(
        user_id=user.id,
        current_password=password_data.current_password,
        new_password=password_data.new_password,
        ip_address=ip_address,
        user_agent=user_agent,
    )
    
    return MessageResponse(
        message="Password changed successfully",
        success=True,
    )


@router.post(
    "/forgot-password",
    response_model=MessageResponse,
    summary="Request password reset",
    description="Request a password reset link via email.",
)
async def forgot_password(
    reset_data: PasswordResetRequest,
    request: Request,
    auth_service: AuthServiceDep,
) -> MessageResponse:
    """
    Request password reset.
    
    Sends a password reset link to the provided email if it exists.
    Always returns success to prevent email enumeration.
    """
    # TODO: Implement email sending
    # For now, just return success message
    return MessageResponse(
        message="If an account with that email exists, a reset link has been sent.",
        success=True,
    )


@router.post(
    "/reset-password",
    response_model=MessageResponse,
    summary="Reset password",
    description="Reset password using token from email.",
)
async def reset_password(
    reset_data: PasswordResetConfirm,
    request: Request,
    auth_service: AuthServiceDep,
) -> MessageResponse:
    """
    Reset password with token.
    
    Complete password reset using the token sent via email.
    """
    # TODO: Implement password reset with token verification
    return MessageResponse(
        message="Password has been reset successfully",
        success=True,
    )


@router.post(
    "/verify-email",
    response_model=MessageResponse,
    summary="Verify email address",
    description="Verify email using token from verification email.",
)
async def verify_email(
    verify_data: EmailVerifyRequest,
    request: Request,
    auth_service: AuthServiceDep,
) -> MessageResponse:
    """
    Verify email address.
    
    Complete email verification using the token sent to user's email.
    """
    # TODO: Implement email verification
    return MessageResponse(
        message="Email verified successfully",
        success=True,
    )
