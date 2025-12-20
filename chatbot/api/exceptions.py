# -*- coding: utf-8 -*-
# =================================================================================================
# api/exceptions.py — Custom Exception Hierarchy with FastAPI Handlers
# =================================================================================================
# Production-grade exception handling implementing:
#
#   1. TYPED EXCEPTIONS: Semantic exception classes for different error categories.
#   2. HTTP MAPPING: Each exception maps to appropriate HTTP status code.
#   3. ERROR CODES: Machine-readable codes for client-side handling.
#   4. STACK TRACES: Debug mode includes traces; production hides internals.
#   5. REQUEST CONTEXT: Errors include request ID for debugging.
#
# Exception Hierarchy:
# --------------------
#   APIException (base)
#   ├── ValidationError (400)
#   ├── AuthenticationError (401)
#   ├── AuthorizationError (403)
#   ├── NotFoundError (404)
#   ├── ConflictError (409)
#   ├── RateLimitError (429)
#   ├── ServiceError (500)
#   ├── ServiceUnavailableError (503)
#   └── TimeoutError (504)
#
# =================================================================================================

from __future__ import annotations

import logging
import traceback
import time
from typing import Any, Dict, List, Optional, Type, Union

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import ValidationError as PydanticValidationError

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
_LOG = logging.getLogger("api.exceptions")


# =============================================================================
# Base Exception Class
# =============================================================================

class APIException(Exception):
    """
    Base exception for all API errors.
    
    Attributes:
    -----------
    status_code : HTTP status code.
    error_code : Machine-readable error code.
    message : Human-readable error message.
    details : Additional error details.
    headers : Optional response headers.
    
    Usage:
    ------
    raise APIException(
        status_code=400,
        error_code="INVALID_INPUT",
        message="Invalid input data",
        details=[{"field": "email", "message": "Invalid format"}]
    )
    """
    
    def __init__(
        self,
        status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR,
        error_code: str = "INTERNAL_ERROR",
        message: str = "An unexpected error occurred",
        details: Optional[List[Dict[str, Any]]] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.error_code = error_code
        self.message = message
        self.details = details or []
        self.headers = headers
    
    def to_dict(self, request_id: Optional[str] = None) -> Dict[str, Any]:
        """Convert exception to response dictionary."""
        return {
            "error": self.error_code,
            "message": self.message,
            "details": self.details,
            "request_id": request_id,
            "timestamp": time.time(),
        }


# =============================================================================
# Client Error Exceptions (4xx)
# =============================================================================

class ValidationError(APIException):
    """
    Input validation failed (400 Bad Request).
    
    Use for:
    --------
    - Invalid request body format
    - Missing required fields
    - Field value out of range
    - Type conversion errors
    """
    
    def __init__(
        self,
        message: str = "Validation error",
        details: Optional[List[Dict[str, Any]]] = None,
        field: Optional[str] = None,
    ) -> None:
        if field and not details:
            details = [{"field": field, "message": message}]
        super().__init__(
            status_code=status.HTTP_400_BAD_REQUEST,
            error_code="VALIDATION_ERROR",
            message=message,
            details=details,
        )


class AuthenticationError(APIException):
    """
    Authentication failed (401 Unauthorized).
    
    Use for:
    --------
    - Missing API key
    - Invalid API key
    - Expired token
    """
    
    def __init__(
        self,
        message: str = "Authentication required",
        details: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        super().__init__(
            status_code=status.HTTP_401_UNAUTHORIZED,
            error_code="AUTHENTICATION_ERROR",
            message=message,
            details=details,
            headers={"WWW-Authenticate": "Bearer"},
        )


class AuthorizationError(APIException):
    """
    Not authorized to access resource (403 Forbidden).
    
    Use for:
    --------
    - Insufficient permissions
    - Resource belongs to another user
    - Feature not enabled for account
    """
    
    def __init__(
        self,
        message: str = "Access forbidden",
        details: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        super().__init__(
            status_code=status.HTTP_403_FORBIDDEN,
            error_code="AUTHORIZATION_ERROR",
            message=message,
            details=details,
        )


class NotFoundError(APIException):
    """
    Resource not found (404 Not Found).
    
    Use for:
    --------
    - Session not found
    - Document not found
    - Invalid endpoint
    """
    
    def __init__(
        self,
        resource: str = "Resource",
        resource_id: Optional[str] = None,
        message: Optional[str] = None,
    ) -> None:
        if message is None:
            if resource_id:
                message = f"{resource} with ID '{resource_id}' not found"
            else:
                message = f"{resource} not found"
        super().__init__(
            status_code=status.HTTP_404_NOT_FOUND,
            error_code="NOT_FOUND",
            message=message,
        )


class ConflictError(APIException):
    """
    Resource conflict (409 Conflict).
    
    Use for:
    --------
    - Duplicate resource creation
    - Optimistic locking failure
    - State transition conflict
    """
    
    def __init__(
        self,
        message: str = "Resource conflict",
        details: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        super().__init__(
            status_code=status.HTTP_409_CONFLICT,
            error_code="CONFLICT_ERROR",
            message=message,
            details=details,
        )


class RateLimitError(APIException):
    """
    Rate limit exceeded (429 Too Many Requests).
    
    Use for:
    --------
    - API rate limit exceeded
    - Concurrent request limit
    - Quota exhausted
    """
    
    def __init__(
        self,
        message: str = "Rate limit exceeded",
        retry_after: Optional[int] = None,
    ) -> None:
        headers = {}
        if retry_after:
            headers["Retry-After"] = str(retry_after)
        super().__init__(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            error_code="RATE_LIMIT_EXCEEDED",
            message=message,
            headers=headers if headers else None,
        )


class UnprocessableEntityError(APIException):
    """
    Request understood but cannot be processed (422 Unprocessable Entity).
    
    Use for:
    --------
    - Semantic validation errors
    - Business logic violations
    - Invalid state transitions
    """
    
    def __init__(
        self,
        message: str = "Unprocessable entity",
        details: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        super().__init__(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            error_code="UNPROCESSABLE_ENTITY",
            message=message,
            details=details,
        )


# =============================================================================
# Server Error Exceptions (5xx)
# =============================================================================

class ServiceError(APIException):
    """
    Internal service error (500 Internal Server Error).
    
    Use for:
    --------
    - Unexpected exceptions
    - Database errors
    - Internal logic errors
    """
    
    def __init__(
        self,
        message: str = "Internal server error",
        details: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        super().__init__(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            error_code="SERVICE_ERROR",
            message=message,
            details=details,
        )


class ServiceUnavailableError(APIException):
    """
    Service temporarily unavailable (503 Service Unavailable).
    
    Use for:
    --------
    - External service down
    - Database connection failed
    - Circuit breaker open
    """
    
    def __init__(
        self,
        message: str = "Service temporarily unavailable",
        retry_after: Optional[int] = None,
    ) -> None:
        headers = {}
        if retry_after:
            headers["Retry-After"] = str(retry_after)
        super().__init__(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            error_code="SERVICE_UNAVAILABLE",
            message=message,
            headers=headers if headers else None,
        )


class TimeoutError(APIException):
    """
    Request timeout (504 Gateway Timeout).
    
    Use for:
    --------
    - LLM request timeout
    - External API timeout
    - Long-running operation timeout
    """
    
    def __init__(
        self,
        message: str = "Request timeout",
        operation: Optional[str] = None,
    ) -> None:
        if operation:
            message = f"Operation '{operation}' timed out"
        super().__init__(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            error_code="TIMEOUT_ERROR",
            message=message,
        )


# =============================================================================
# Exception Handlers for FastAPI
# =============================================================================

def get_request_id(request: Request) -> Optional[str]:
    """Extract request ID from headers or state."""
    # Try header first
    request_id = request.headers.get("X-Request-ID")
    if request_id:
        return request_id
    # Try state (set by middleware)
    if hasattr(request.state, "request_id"):
        return request.state.request_id
    return None


async def api_exception_handler(request: Request, exc: APIException) -> JSONResponse:
    """
    Handle APIException and its subclasses.
    
    Response Format:
    ----------------
    {
        "error": "ERROR_CODE",
        "message": "Human-readable message",
        "details": [...],
        "request_id": "abc-123",
        "timestamp": 1234567890.123
    }
    """
    request_id = get_request_id(request)
    
    # Log error with context
    _LOG.warning(
        "API error: %s %s -> %d %s",
        request.method,
        request.url.path,
        exc.status_code,
        exc.error_code,
        extra={"request_id": request_id, "details": exc.details},
    )
    
    return JSONResponse(
        status_code=exc.status_code,
        content=exc.to_dict(request_id),
        headers=exc.headers,
    )


async def validation_exception_handler(
    request: Request,
    exc: RequestValidationError,
) -> JSONResponse:
    """
    Handle Pydantic validation errors from FastAPI.
    
    Transforms Pydantic errors into our standard error format.
    """
    request_id = get_request_id(request)
    
    # Transform Pydantic errors to our format
    details = []
    for error in exc.errors():
        field = ".".join(str(loc) for loc in error.get("loc", []))
        details.append({
            "field": field,
            "message": error.get("msg", "Validation error"),
            "type": error.get("type", "unknown"),
        })
    
    _LOG.warning(
        "Validation error: %s %s -> %d errors",
        request.method,
        request.url.path,
        len(details),
        extra={"request_id": request_id, "details": details},
    )
    
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": "VALIDATION_ERROR",
            "message": "Request validation failed",
            "details": details,
            "request_id": request_id,
            "timestamp": time.time(),
        },
    )


async def pydantic_exception_handler(
    request: Request,
    exc: PydanticValidationError,
) -> JSONResponse:
    """Handle direct Pydantic ValidationError (not from FastAPI)."""
    request_id = get_request_id(request)
    
    details = []
    for error in exc.errors():
        field = ".".join(str(loc) for loc in error.get("loc", []))
        details.append({
            "field": field,
            "message": error.get("msg", "Validation error"),
        })
    
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": "VALIDATION_ERROR",
            "message": "Data validation failed",
            "details": details,
            "request_id": request_id,
            "timestamp": time.time(),
        },
    )


async def generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """
    Catch-all handler for unexpected exceptions.
    
    Security:
    ---------
    - Never expose stack traces in production.
    - Log full details for debugging.
    - Return generic message to client.
    """
    request_id = get_request_id(request)
    
    # Log full exception with traceback
    _LOG.exception(
        "Unhandled exception: %s %s",
        request.method,
        request.url.path,
        extra={"request_id": request_id},
    )
    
    # Check if debug mode (would need to import config)
    show_details = False
    try:
        from api.config import get_settings
        show_details = get_settings().debug
    except Exception:
        pass
    
    details = []
    if show_details:
        details.append({
            "exception": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        })
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "INTERNAL_ERROR",
            "message": "An unexpected error occurred",
            "details": details,
            "request_id": request_id,
            "timestamp": time.time(),
        },
    )


# =============================================================================
# Registration Function
# =============================================================================

def register_exception_handlers(app: FastAPI) -> None:
    """
    Register all exception handlers with FastAPI app.
    
    Call during app initialization:
    -------------------------------
    from api.exceptions import register_exception_handlers
    
    app = FastAPI()
    register_exception_handlers(app)
    """
    app.add_exception_handler(APIException, api_exception_handler)
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
    app.add_exception_handler(PydanticValidationError, pydantic_exception_handler)
    app.add_exception_handler(Exception, generic_exception_handler)
    
    _LOG.info("Exception handlers registered")


# =============================================================================
# Utility Functions
# =============================================================================

def assert_found(
    obj: Optional[Any],
    resource: str = "Resource",
    resource_id: Optional[str] = None,
) -> Any:
    """
    Assert that a resource was found, raising NotFoundError if None.
    
    Usage:
    ------
    session = assert_found(
        await session_repo.get(session_id),
        "Session",
        session_id
    )
    """
    if obj is None:
        raise NotFoundError(resource=resource, resource_id=resource_id)
    return obj


def wrap_service_error(exc: Exception, message: str = "Service operation failed") -> ServiceError:
    """
    Wrap a generic exception in a ServiceError.
    
    Usage:
    ------
    try:
        await external_service.call()
    except Exception as e:
        raise wrap_service_error(e, "External service call failed")
    """
    _LOG.error("Service error: %s: %s", message, str(exc), exc_info=True)
    return ServiceError(
        message=message,
        details=[{"original_error": str(exc)}],
    )
