# =============================================================================
# SOTA AUTHENTICATION SYSTEM - CORE EXCEPTIONS MODULE
# =============================================================================
# File: core/exceptions.py
# Description: Custom exception hierarchy for the authentication system
#              Provides granular error handling with HTTP status code mapping
# =============================================================================

from typing import Optional, Dict, Any
from fastapi import HTTPException, status


class AuthSystemException(Exception):
    """
    ┌─────────────────────────────────────────────────────────────────────────┐
    │                    BASE EXCEPTION CLASS                                  │
    │  All custom exceptions inherit from this base class                      │
    │  Provides consistent error structure across the application             │
    └─────────────────────────────────────────────────────────────────────────┘
    
    Attributes:
        message: Human-readable error description
        error_code: Machine-readable error identifier
        status_code: HTTP status code for API responses
        details: Additional context for debugging
    """
    
    def __init__(
        self,
        message: str = "An error occurred",
        error_code: str = "AUTH_ERROR",
        status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR,
        details: Optional[Dict[str, Any]] = None
    ):
        self.message = message
        self.error_code = error_code
        self.status_code = status_code
        self.details = details or {}
        super().__init__(self.message)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for JSON serialization."""
        return {
            "error": True,
            "error_code": self.error_code,
            "message": self.message,
            "details": self.details
        }
    
    def to_http_exception(self) -> HTTPException:
        """Convert to FastAPI HTTPException for API responses."""
        return HTTPException(
            status_code=self.status_code,
            detail=self.to_dict()
        )


# =============================================================================
# AUTHENTICATION EXCEPTIONS
# =============================================================================

class AuthenticationError(AuthSystemException):
    """
    Raised when authentication fails (invalid credentials, expired token, etc.)
    
    Examples:
        - Invalid email/password combination
        - Expired or malformed JWT token
        - Missing authentication header
    """
    
    def __init__(
        self,
        message: str = "Authentication failed",
        error_code: str = "AUTHENTICATION_FAILED",
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            message=message,
            error_code=error_code,
            status_code=status.HTTP_401_UNAUTHORIZED,
            details=details
        )


class InvalidCredentialsError(AuthenticationError):
    """Raised when email/password combination is invalid."""
    
    def __init__(self, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message="Invalid email or password",
            error_code="INVALID_CREDENTIALS",
            details=details
        )


class TokenError(AuthenticationError):
    """Base class for all token-related errors."""
    
    def __init__(
        self,
        message: str = "Token error",
        error_code: str = "TOKEN_ERROR",
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            message=message,
            error_code=error_code,
            details=details
        )


class TokenExpiredError(TokenError):
    """Raised when JWT token has expired."""
    
    def __init__(self, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message="Token has expired",
            error_code="TOKEN_EXPIRED",
            details=details
        )


class TokenInvalidError(TokenError):
    """Raised when JWT token is malformed or signature is invalid."""
    
    def __init__(self, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message="Invalid token",
            error_code="TOKEN_INVALID",
            details=details
        )


class TokenBlacklistedError(TokenError):
    """Raised when JWT token has been revoked/blacklisted."""
    
    def __init__(self, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message="Token has been revoked",
            error_code="TOKEN_BLACKLISTED",
            details=details
        )


class TokenMissingError(TokenError):
    """Raised when authentication token is not provided."""
    
    def __init__(self, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message="Authentication token is required",
            error_code="TOKEN_MISSING",
            details=details
        )


# =============================================================================
# AUTHORIZATION EXCEPTIONS
# =============================================================================

class AuthorizationError(AuthSystemException):
    """
    Raised when user lacks permission to access a resource.
    
    Examples:
        - Non-admin accessing admin-only endpoint
        - User trying to access another user's data
    """
    
    def __init__(
        self,
        message: str = "Access denied",
        error_code: str = "AUTHORIZATION_FAILED",
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            message=message,
            error_code=error_code,
            status_code=status.HTTP_403_FORBIDDEN,
            details=details
        )


class InsufficientPermissionsError(AuthorizationError):
    """Raised when user doesn't have required permissions."""
    
    def __init__(
        self,
        required_permission: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        message = "Insufficient permissions"
        if required_permission:
            message = f"Insufficient permissions. Required: {required_permission}"
        super().__init__(
            message=message,
            error_code="INSUFFICIENT_PERMISSIONS",
            details=details
        )


# =============================================================================
# USER EXCEPTIONS
# =============================================================================

class UserError(AuthSystemException):
    """Base class for user-related errors."""
    
    def __init__(
        self,
        message: str = "User error",
        error_code: str = "USER_ERROR",
        status_code: int = status.HTTP_400_BAD_REQUEST,
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            message=message,
            error_code=error_code,
            status_code=status_code,
            details=details
        )


class UserNotFoundError(UserError):
    """Raised when requested user does not exist."""
    
    def __init__(self, user_id: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        message = "User not found"
        if user_id:
            message = f"User with ID '{user_id}' not found"
        super().__init__(
            message=message,
            error_code="USER_NOT_FOUND",
            status_code=status.HTTP_404_NOT_FOUND,
            details=details
        )


class UserExistsError(UserError):
    """Raised when attempting to create a user that already exists."""
    
    def __init__(
        self,
        field: str = "email",
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            message=f"User with this {field} already exists",
            error_code="USER_EXISTS",
            status_code=status.HTTP_409_CONFLICT,
            details=details
        )


class UserInactiveError(UserError):
    """Raised when attempting to authenticate an inactive user."""
    
    def __init__(self, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message="User account is inactive",
            error_code="USER_INACTIVE",
            status_code=status.HTTP_403_FORBIDDEN,
            details=details
        )


class UserNotVerifiedError(UserError):
    """Raised when unverified user attempts restricted action."""
    
    def __init__(self, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message="Email verification required",
            error_code="USER_NOT_VERIFIED",
            status_code=status.HTTP_403_FORBIDDEN,
            details=details
        )


class AccountLockedError(UserError):
    """Raised when user account is locked due to too many failed attempts."""
    
    def __init__(
        self,
        locked_until: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        message = "Account is temporarily locked"
        if locked_until:
            message = f"Account is locked until {locked_until}"
        super().__init__(
            message=message,
            error_code="ACCOUNT_LOCKED",
            status_code=status.HTTP_423_LOCKED,
            details=details
        )


# =============================================================================
# SESSION EXCEPTIONS
# =============================================================================

class SessionError(AuthSystemException):
    """Base class for session-related errors."""
    
    def __init__(
        self,
        message: str = "Session error",
        error_code: str = "SESSION_ERROR",
        status_code: int = status.HTTP_400_BAD_REQUEST,
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            message=message,
            error_code=error_code,
            status_code=status_code,
            details=details
        )


class SessionNotFoundError(SessionError):
    """Raised when requested session does not exist."""
    
    def __init__(self, session_id: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        message = "Session not found"
        if session_id:
            message = f"Session '{session_id}' not found"
        super().__init__(
            message=message,
            error_code="SESSION_NOT_FOUND",
            status_code=status.HTTP_404_NOT_FOUND,
            details=details
        )


class SessionExpiredError(SessionError):
    """Raised when session has expired."""
    
    def __init__(self, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message="Session has expired",
            error_code="SESSION_EXPIRED",
            status_code=status.HTTP_401_UNAUTHORIZED,
            details=details
        )


class SessionInvalidError(SessionError):
    """Raised when session is invalid or corrupted."""
    
    def __init__(self, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message="Invalid session",
            error_code="SESSION_INVALID",
            status_code=status.HTTP_401_UNAUTHORIZED,
            details=details
        )


# =============================================================================
# RATE LIMITING & SECURITY EXCEPTIONS
# =============================================================================

class RateLimitExceededError(AuthSystemException):
    """Raised when rate limit is exceeded."""
    
    def __init__(
        self,
        retry_after: Optional[int] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        message = "Rate limit exceeded. Please try again later"
        if retry_after:
            message = f"Rate limit exceeded. Retry after {retry_after} seconds"
            if details is None:
                details = {}
            details["retry_after"] = retry_after
        super().__init__(
            message=message,
            error_code="RATE_LIMIT_EXCEEDED",
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            details=details
        )


# =============================================================================
# VALIDATION EXCEPTIONS
# =============================================================================

class ValidationError(AuthSystemException):
    """Raised when input validation fails."""
    
    def __init__(
        self,
        message: str = "Validation failed",
        field: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        if field:
            message = f"Validation failed for field: {field}"
        super().__init__(
            message=message,
            error_code="VALIDATION_ERROR",
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            details=details
        )


class PasswordValidationError(ValidationError):
    """Raised when password does not meet requirements."""
    
    def __init__(
        self,
        message: str = "Password does not meet requirements",
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            message=message,
            field="password",
            details=details
        )
        self.error_code = "PASSWORD_VALIDATION_ERROR"


class EmailValidationError(ValidationError):
    """Raised when email format is invalid."""
    
    def __init__(self, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message="Invalid email format",
            field="email",
            details=details
        )
        self.error_code = "EMAIL_VALIDATION_ERROR"


# =============================================================================
# DATABASE EXCEPTIONS
# =============================================================================

class DatabaseError(AuthSystemException):
    """Base class for database-related errors."""
    
    def __init__(
        self,
        message: str = "Database error",
        error_code: str = "DATABASE_ERROR",
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            message=message,
            error_code=error_code,
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            details=details
        )


class DatabaseConnectionError(DatabaseError):
    """Raised when database connection fails."""
    
    def __init__(self, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message="Failed to connect to database",
            error_code="DATABASE_CONNECTION_ERROR",
            details=details
        )


class DatabaseQueryError(DatabaseError):
    """Raised when database query fails."""
    
    def __init__(self, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message="Database query failed",
            error_code="DATABASE_QUERY_ERROR",
            details=details
        )


# =============================================================================
# REDIS EXCEPTIONS
# =============================================================================

class RedisError(AuthSystemException):
    """Base class for Redis-related errors."""
    
    def __init__(
        self,
        message: str = "Redis error",
        error_code: str = "REDIS_ERROR",
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            message=message,
            error_code=error_code,
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            details=details
        )


class RedisConnectionError(RedisError):
    """Raised when Redis connection fails."""
    
    def __init__(self, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message="Failed to connect to Redis",
            error_code="REDIS_CONNECTION_ERROR",
            details=details
        )
