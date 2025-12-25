# =============================================================================
# AUTH MODULE INITIALIZATION
# =============================================================================
# File: auth/__init__.py
# Description: Auth module exports
# =============================================================================

from auth.schemas import (
    UserCreate,
    UserUpdate,
    UserResponse,
    LoginRequest,
    TokenResponse,
    RefreshRequest,
    RefreshResponse,
    PasswordChangeRequest,
    PasswordResetRequest,
    PasswordResetConfirm,
    SessionResponse,
    SessionListResponse,
    MessageResponse,
    ErrorResponse,
)
from auth.repository import (
    UserRepository,
    SessionRepository,
    RefreshTokenRepository,
    AuditLogRepository,
)
from auth.service import AuthService
from auth.dependencies import (
    get_auth_service,
    get_current_token,
    get_current_user,
    get_current_active_user,
    get_current_superuser,
    get_client_ip,
    get_user_agent,
    AuthContext,
    DBSession,
    ActiveUser,
    SuperUser,
    VerifiedUser,
    AuthServiceDep,
    AuthContextDep,
)

__all__ = [
    # Schemas
    "UserCreate",
    "UserUpdate",
    "UserResponse",
    "LoginRequest",
    "TokenResponse",
    "RefreshRequest",
    "RefreshResponse",
    "PasswordChangeRequest",
    "PasswordResetRequest",
    "PasswordResetConfirm",
    "SessionResponse",
    "SessionListResponse",
    "MessageResponse",
    "ErrorResponse",
    
    # Repository
    "UserRepository",
    "SessionRepository",
    "RefreshTokenRepository",
    "AuditLogRepository",
    
    # Service
    "AuthService",
    
    # Dependencies
    "get_auth_service",
    "get_current_token",
    "get_current_user",
    "get_current_active_user",
    "get_current_superuser",
    "get_client_ip",
    "get_user_agent",
    "AuthContext",
    "DBSession",
    "ActiveUser",
    "SuperUser",
    "VerifiedUser",
    "AuthServiceDep",
    "AuthContextDep",
]
