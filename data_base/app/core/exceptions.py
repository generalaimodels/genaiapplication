# ==============================================================================
# CUSTOM EXCEPTIONS - Application Error Hierarchy
# ==============================================================================
# Structured exception classes for consistent error handling
# Each exception maps to appropriate HTTP status codes
# ==============================================================================

from __future__ import annotations

from typing import Any, Dict, Optional


class AppException(Exception):
    """
    Base exception for all application errors.
    
    Provides a consistent interface for error handling with:
    - Error code for programmatic identification
    - HTTP status code mapping
    - Detailed message and optional context
    
    Attributes:
        message: Human-readable error description
        error_code: Machine-readable error identifier
        status_code: HTTP status code to return
        details: Additional context dictionary
        
    Example:
        >>> raise AppException(
        ...     message="Something went wrong",
        ...     error_code="INTERNAL_ERROR",
        ...     status_code=500
        ... )
    """
    
    def __init__(
        self,
        message: str = "An unexpected error occurred",
        error_code: str = "INTERNAL_ERROR",
        status_code: int = 500,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.message = message
        self.error_code = error_code
        self.status_code = status_code
        self.details = details or {}
        super().__init__(self.message)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert exception to dictionary format for JSON response.
        
        Returns:
            Dictionary containing error details
        """
        return {
            "success": False,
            "error": {
                "code": self.error_code,
                "message": self.message,
                "details": self.details,
            }
        }
    
    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"message='{self.message}', "
            f"error_code='{self.error_code}', "
            f"status_code={self.status_code})"
        )


# ==============================================================================
# DATABASE EXCEPTIONS
# ==============================================================================

class DatabaseError(AppException):
    """
    Base exception for database-related errors.
    
    Raised when database operations fail due to:
    - Connection issues
    - Query execution failures
    - Transaction errors
    """
    
    def __init__(
        self,
        message: str = "Database operation failed",
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            message=message,
            error_code="DATABASE_ERROR",
            status_code=503,
            details=details,
        )


class ConnectionError(DatabaseError):
    """
    Raised when database connection cannot be established.
    """
    
    def __init__(
        self,
        message: str = "Failed to connect to database",
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(message=message, details=details)
        self.error_code = "DATABASE_CONNECTION_ERROR"


class TransactionError(DatabaseError):
    """
    Raised when database transaction fails.
    
    Indicates that a transaction could not be committed
    and has been rolled back.
    """
    
    def __init__(
        self,
        message: str = "Transaction failed",
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(message=message, details=details)
        self.error_code = "TRANSACTION_ERROR"
        self.status_code = 500


# ==============================================================================
# RESOURCE EXCEPTIONS
# ==============================================================================

class NotFoundError(AppException):
    """
    Raised when a requested resource does not exist.
    
    Maps to HTTP 404 Not Found.
    
    Attributes:
        resource_type: Type of resource that was not found
        resource_id: Identifier of the missing resource
    """
    
    def __init__(
        self,
        message: str = "Resource not found",
        resource_type: Optional[str] = None,
        resource_id: Optional[Any] = None,
    ) -> None:
        details = {}
        if resource_type:
            details["resource_type"] = resource_type
        if resource_id:
            details["resource_id"] = str(resource_id)
        
        super().__init__(
            message=message,
            error_code="NOT_FOUND",
            status_code=404,
            details=details,
        )
        self.resource_type = resource_type
        self.resource_id = resource_id


class AlreadyExistsError(AppException):
    """
    Raised when attempting to create a resource that already exists.
    
    Maps to HTTP 409 Conflict.
    """
    
    def __init__(
        self,
        message: str = "Resource already exists",
        resource_type: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        _details = details or {}
        if resource_type:
            _details["resource_type"] = resource_type
        
        super().__init__(
            message=message,
            error_code="ALREADY_EXISTS",
            status_code=409,
            details=_details,
        )


# ==============================================================================
# VALIDATION EXCEPTIONS
# ==============================================================================

class ValidationError(AppException):
    """
    Raised when input validation fails.
    
    Maps to HTTP 422 Unprocessable Entity.
    Contains field-level validation errors.
    """
    
    def __init__(
        self,
        message: str = "Validation error",
        errors: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            message=message,
            error_code="VALIDATION_ERROR",
            status_code=422,
            details={"validation_errors": errors or {}},
        )
        self.errors = errors or {}


class BadRequestError(AppException):
    """
    Raised for malformed or invalid requests.
    
    Maps to HTTP 400 Bad Request.
    """
    
    def __init__(
        self,
        message: str = "Bad request",
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            message=message,
            error_code="BAD_REQUEST",
            status_code=400,
            details=details,
        )


# ==============================================================================
# AUTHENTICATION & AUTHORIZATION EXCEPTIONS
# ==============================================================================

class AuthenticationError(AppException):
    """
    Raised when authentication fails.
    
    Maps to HTTP 401 Unauthorized.
    
    Common causes:
    - Invalid or expired token
    - Missing authentication header
    - Invalid credentials
    """
    
    def __init__(
        self,
        message: str = "Authentication failed",
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            message=message,
            error_code="AUTHENTICATION_ERROR",
            status_code=401,
            details=details,
        )


class AuthorizationError(AppException):
    """
    Raised when user lacks permission for an action.
    
    Maps to HTTP 403 Forbidden.
    
    The user is authenticated but not authorized
    to access the requested resource.
    """
    
    def __init__(
        self,
        message: str = "Permission denied",
        required_permission: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        _details = details or {}
        if required_permission:
            _details["required_permission"] = required_permission
        
        super().__init__(
            message=message,
            error_code="AUTHORIZATION_ERROR",
            status_code=403,
            details=_details,
        )


class TokenExpiredError(AuthenticationError):
    """
    Raised when JWT token has expired.
    """
    
    def __init__(
        self,
        message: str = "Token has expired",
    ) -> None:
        super().__init__(message=message)
        self.error_code = "TOKEN_EXPIRED"


class InvalidTokenError(AuthenticationError):
    """
    Raised when JWT token is invalid or malformed.
    """
    
    def __init__(
        self,
        message: str = "Invalid token",
    ) -> None:
        super().__init__(message=message)
        self.error_code = "INVALID_TOKEN"


# ==============================================================================
# RATE LIMITING EXCEPTIONS
# ==============================================================================

class RateLimitError(AppException):
    """
    Raised when rate limit is exceeded.
    
    Maps to HTTP 429 Too Many Requests.
    
    Attributes:
        retry_after: Seconds until the client can retry
    """
    
    def __init__(
        self,
        message: str = "Rate limit exceeded",
        retry_after: int = 60,
    ) -> None:
        super().__init__(
            message=message,
            error_code="RATE_LIMIT_EXCEEDED",
            status_code=429,
            details={"retry_after_seconds": retry_after},
        )
        self.retry_after = retry_after


# ==============================================================================
# BUSINESS LOGIC EXCEPTIONS
# ==============================================================================

class InsufficientFundsError(AppException):
    """
    Raised when a financial transaction fails due to insufficient funds.
    
    Maps to HTTP 400 Bad Request.
    """
    
    def __init__(
        self,
        message: str = "Insufficient funds",
        required_amount: Optional[float] = None,
        available_amount: Optional[float] = None,
    ) -> None:
        details = {}
        if required_amount is not None:
            details["required_amount"] = required_amount
        if available_amount is not None:
            details["available_amount"] = available_amount
        
        super().__init__(
            message=message,
            error_code="INSUFFICIENT_FUNDS",
            status_code=400,
            details=details,
        )


class BusinessRuleError(AppException):
    """
    Raised when a business rule is violated.
    
    Maps to HTTP 400 Bad Request.
    """
    
    def __init__(
        self,
        message: str = "Business rule violation",
        rule: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        _details = details or {}
        if rule:
            _details["violated_rule"] = rule
        
        super().__init__(
            message=message,
            error_code="BUSINESS_RULE_ERROR",
            status_code=400,
            details=_details,
        )


class ServiceUnavailableError(AppException):
    """
    Raised when an external service is unavailable.
    
    Maps to HTTP 503 Service Unavailable.
    """
    
    def __init__(
        self,
        message: str = "Service temporarily unavailable",
        service_name: Optional[str] = None,
        retry_after: Optional[int] = None,
    ) -> None:
        details = {}
        if service_name:
            details["service"] = service_name
        if retry_after:
            details["retry_after_seconds"] = retry_after
        
        super().__init__(
            message=message,
            error_code="SERVICE_UNAVAILABLE",
            status_code=503,
            details=details,
        )
