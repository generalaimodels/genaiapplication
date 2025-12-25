# =============================================================================
# CORE MODULE INITIALIZATION
# =============================================================================
# File: core/__init__.py
# Description: Core module exports for centralized access
# =============================================================================

from core.config import settings, get_settings, Settings
from core.exceptions import (
    # Base
    AuthSystemException,
    
    # Authentication
    AuthenticationError,
    InvalidCredentialsError,
    TokenError,
    TokenExpiredError,
    TokenInvalidError,
    TokenBlacklistedError,
    TokenMissingError,
    
    # Authorization
    AuthorizationError,
    InsufficientPermissionsError,
    
    # User
    UserError,
    UserNotFoundError,
    UserExistsError,
    UserInactiveError,
    UserNotVerifiedError,
    AccountLockedError,
    
    # Session
    SessionError,
    SessionNotFoundError,
    SessionExpiredError,
    SessionInvalidError,
    
    # Rate Limiting
    RateLimitExceededError,
    
    # Validation
    ValidationError,
    PasswordValidationError,
    EmailValidationError,
    
    # Database
    DatabaseError,
    DatabaseConnectionError,
    DatabaseQueryError,
    
    # Redis
    RedisError,
    RedisConnectionError,
)
from core.security import (
    PasswordManager,
    JWTManager,
    PasswordValidator,
    TokenPayload,
    TokenPair,
    password_manager,
    jwt_manager,
    password_validator,
    generate_secure_token,
    generate_session_id,
    generate_verification_code,
)

__all__ = [
    # Config
    "settings",
    "get_settings",
    "Settings",
    
    # Exceptions
    "AuthSystemException",
    "AuthenticationError",
    "InvalidCredentialsError",
    "TokenError",
    "TokenExpiredError",
    "TokenInvalidError",
    "TokenBlacklistedError",
    "TokenMissingError",
    "AuthorizationError",
    "InsufficientPermissionsError",
    "UserError",
    "UserNotFoundError",
    "UserExistsError",
    "UserInactiveError",
    "UserNotVerifiedError",
    "AccountLockedError",
    "SessionError",
    "SessionNotFoundError",
    "SessionExpiredError",
    "SessionInvalidError",
    "RateLimitExceededError",
    "ValidationError",
    "PasswordValidationError",
    "EmailValidationError",
    "DatabaseError",
    "DatabaseConnectionError",
    "DatabaseQueryError",
    "RedisError",
    "RedisConnectionError",
    
    # Security
    "PasswordManager",
    "JWTManager",
    "PasswordValidator",
    "TokenPayload",
    "TokenPair",
    "password_manager",
    "jwt_manager",
    "password_validator",
    "generate_secure_token",
    "generate_session_id",
    "generate_verification_code",
]

