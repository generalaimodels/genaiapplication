"""
Error Handling Module: Result Types and Error Variants
=======================================================

Design Philosophy:
    - NO EXCEPTIONS for control flow (exceptions reserved for bugs)
    - Result[T, E] pattern for explicit error handling
    - Exhaustive pattern matching enforced via type system
    - Atomic rollback guarantees for partial operations

Error Hierarchy:
    InferenceError (base)
    ├── ProviderError (upstream API failures)
    ├── ValidationError (request validation failures)
    ├── TimeoutError (deadline exceeded)
    ├── RateLimitError (429 responses)
    ├── QueueFullError (backpressure rejection)
    └── CancellationError (client disconnect)

Result Type:
    - Ok[T]: Success with value
    - Err[E]: Failure with error
    - Monadic operations: map, map_err, and_then, or_else
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Callable,
    Final,
    Generic,
    TypeVar,
    overload,
)
import time


# =============================================================================
# TYPE VARIABLES: For generic Result type
# =============================================================================
T = TypeVar("T")  # Success type
U = TypeVar("U")  # Mapped success type
E = TypeVar("E", bound="InferenceError")  # Error type (covariant)
F = TypeVar("F", bound="InferenceError")  # Mapped error type


class ErrorCode(Enum):
    """
    Numeric error codes for programmatic handling.
    Ranges:
        1000-1999: Provider errors
        2000-2999: Validation errors
        3000-3999: Resource errors (timeout, rate limit, queue)
        4000-4999: Client errors (cancellation)
    """
    # Provider errors
    PROVIDER_UNAVAILABLE = 1001
    PROVIDER_AUTH_FAILED = 1002
    PROVIDER_INVALID_RESPONSE = 1003
    PROVIDER_MODEL_NOT_FOUND = 1004
    
    # Validation errors
    VALIDATION_INVALID_REQUEST = 2001
    VALIDATION_MISSING_FIELD = 2002
    VALIDATION_INVALID_TYPE = 2003
    VALIDATION_OUT_OF_RANGE = 2004
    
    # Resource errors
    TIMEOUT_EXCEEDED = 3001
    RATE_LIMIT_EXCEEDED = 3002
    QUEUE_FULL = 3003
    RESOURCE_EXHAUSTED = 3004
    
    # Client errors
    CANCELLED = 4001
    CLIENT_DISCONNECT = 4002


@dataclass(slots=True, frozen=True)
class InferenceError:
    """
    Base error class with immutability for thread-safety.
    
    Attributes:
        code: Numeric error code for programmatic handling
        message: Human-readable error description
        timestamp_ns: Nanosecond-precision error timestamp
        request_id: Optional correlation ID for tracing
        details: Optional structured error details
    """
    code: ErrorCode
    message: str
    timestamp_ns: int = field(default_factory=time.perf_counter_ns)
    request_id: str | None = None
    details: dict[str, object] | None = None
    
    def to_dict(self) -> dict[str, object]:
        """Serialize to JSON-compatible dict."""
        return {
            "error": {
                "code": self.code.value,
                "type": self.code.name,
                "message": self.message,
                "request_id": self.request_id,
                "details": self.details,
            }
        }


# =============================================================================
# SPECIALIZED ERROR TYPES: Using inheritance for type discrimination
# =============================================================================

@dataclass(slots=True, frozen=True)
class ProviderError(InferenceError):
    """
    Upstream provider communication failure.
    
    Retryable Codes:
        - PROVIDER_UNAVAILABLE (503, connection refused)
        - RATE_LIMIT_EXCEEDED (429)
    
    Non-Retryable Codes:
        - PROVIDER_AUTH_FAILED (401, 403)
        - PROVIDER_MODEL_NOT_FOUND (404)
    """
    status_code: int | None = None
    provider_name: str | None = None
    
    @property
    def is_retryable(self) -> bool:
        """Check if error is transient and retryable."""
        return self.code in {
            ErrorCode.PROVIDER_UNAVAILABLE,
            ErrorCode.RATE_LIMIT_EXCEEDED,
        }


@dataclass(slots=True, frozen=True)
class ValidationError(InferenceError):
    """
    Request validation failure.
    
    Never retryable - client must fix request.
    """
    field_name: str | None = None
    field_value: object = None
    constraint: str | None = None


@dataclass(slots=True, frozen=True)
class TimeoutError(InferenceError):
    """
    Deadline exceeded during request processing.
    
    Retryable with exponential backoff.
    """
    timeout_seconds: float | None = None
    elapsed_seconds: float | None = None


@dataclass(slots=True, frozen=True)
class RateLimitError(InferenceError):
    """
    Rate limit exceeded (local or upstream).
    
    Retryable after retry_after_seconds.
    """
    retry_after_seconds: float | None = None
    limit_type: str | None = None  # "local" or "upstream"


@dataclass(slots=True, frozen=True)
class QueueFullError(InferenceError):
    """
    Request queue at capacity (backpressure).
    
    Client should retry with backoff or reduce load.
    """
    queue_size: int | None = None
    queue_capacity: int | None = None


@dataclass(slots=True, frozen=True)
class CancellationError(InferenceError):
    """
    Request cancelled by client disconnect.
    
    Not retryable - intentional cancellation.
    """
    reason: str | None = None


# =============================================================================
# RESULT TYPE: Monadic error handling (Ok | Err)
# =============================================================================

@dataclass(slots=True, frozen=True)
class Ok(Generic[T]):
    """
    Success variant of Result.
    
    Immutable and hashable for use as dict keys.
    """
    value: T
    
    def is_ok(self) -> bool:
        return True
    
    def is_err(self) -> bool:
        return False
    
    def unwrap(self) -> T:
        """
        Extract value. Safe because this is Ok.
        """
        return self.value
    
    def unwrap_or(self, default: T) -> T:
        """Return value, ignoring default."""
        return self.value
    
    def map(self, fn: Callable[[T], U]) -> Ok[U]:
        """Transform success value."""
        return Ok(fn(self.value))
    
    def map_err(self, fn: Callable[[E], F]) -> Ok[T]:
        """No-op for Ok variant."""
        return self
    
    def and_then(self, fn: Callable[[T], Result[U, E]]) -> Result[U, E]:
        """Chain operations that may fail."""
        return fn(self.value)


@dataclass(slots=True, frozen=True)
class Err(Generic[E]):
    """
    Failure variant of Result.
    
    Immutable and hashable for use as dict keys.
    """
    error: E
    
    def is_ok(self) -> bool:
        return False
    
    def is_err(self) -> bool:
        return True
    
    def unwrap(self) -> T:
        """
        Raises the error. Use pattern matching instead.
        """
        raise RuntimeError(f"Called unwrap on Err: {self.error}")
    
    def unwrap_or(self, default: T) -> T:
        """Return default value."""
        return default
    
    def map(self, fn: Callable[[T], U]) -> Err[E]:
        """No-op for Err variant."""
        return self
    
    def map_err(self, fn: Callable[[E], F]) -> Err[F]:
        """Transform error value."""
        return Err(fn(self.error))
    
    def and_then(self, fn: Callable[[T], Result[U, E]]) -> Err[E]:
        """Short-circuit on error."""
        return self


# Type alias for Result
Result = Ok[T] | Err[E]


# =============================================================================
# FACTORY FUNCTIONS: Create errors with proper defaults
# =============================================================================

def provider_unavailable(
    message: str,
    *,
    status_code: int | None = None,
    provider_name: str | None = None,
    request_id: str | None = None,
) -> ProviderError:
    """Create provider unavailable error."""
    return ProviderError(
        code=ErrorCode.PROVIDER_UNAVAILABLE,
        message=message,
        status_code=status_code,
        provider_name=provider_name,
        request_id=request_id,
    )


def provider_auth_failed(
    message: str = "Authentication failed",
    *,
    provider_name: str | None = None,
    request_id: str | None = None,
) -> ProviderError:
    """Create authentication failure error."""
    return ProviderError(
        code=ErrorCode.PROVIDER_AUTH_FAILED,
        message=message,
        provider_name=provider_name,
        request_id=request_id,
    )


def validation_error(
    message: str,
    *,
    field_name: str | None = None,
    field_value: object = None,
    constraint: str | None = None,
    request_id: str | None = None,
) -> ValidationError:
    """Create validation error."""
    return ValidationError(
        code=ErrorCode.VALIDATION_INVALID_REQUEST,
        message=message,
        field_name=field_name,
        field_value=field_value,
        constraint=constraint,
        request_id=request_id,
    )


def timeout_error(
    message: str = "Request timeout exceeded",
    *,
    timeout_seconds: float | None = None,
    elapsed_seconds: float | None = None,
    request_id: str | None = None,
) -> TimeoutError:
    """Create timeout error."""
    return TimeoutError(
        code=ErrorCode.TIMEOUT_EXCEEDED,
        message=message,
        timeout_seconds=timeout_seconds,
        elapsed_seconds=elapsed_seconds,
        request_id=request_id,
    )


def rate_limit_error(
    message: str = "Rate limit exceeded",
    *,
    retry_after_seconds: float | None = None,
    limit_type: str | None = None,
    request_id: str | None = None,
) -> RateLimitError:
    """Create rate limit error."""
    return RateLimitError(
        code=ErrorCode.RATE_LIMIT_EXCEEDED,
        message=message,
        retry_after_seconds=retry_after_seconds,
        limit_type=limit_type,
        request_id=request_id,
    )


def queue_full_error(
    message: str = "Request queue at capacity",
    *,
    queue_size: int | None = None,
    queue_capacity: int | None = None,
    request_id: str | None = None,
) -> QueueFullError:
    """Create queue full error."""
    return QueueFullError(
        code=ErrorCode.QUEUE_FULL,
        message=message,
        queue_size=queue_size,
        queue_capacity=queue_capacity,
        request_id=request_id,
    )


def cancellation_error(
    message: str = "Request cancelled",
    *,
    reason: str | None = None,
    request_id: str | None = None,
) -> CancellationError:
    """Create cancellation error."""
    return CancellationError(
        code=ErrorCode.CANCELLED,
        message=message,
        reason=reason,
        request_id=request_id,
    )


# =============================================================================
# PATTERN MATCHING UTILITIES
# =============================================================================

def match_result(
    result: Result[T, E],
    *,
    on_ok: Callable[[T], U],
    on_err: Callable[[E], U],
) -> U:
    """
    Exhaustive pattern matching on Result.
    
    Guarantees both branches are handled.
    
    Example:
        value = match_result(
            result,
            on_ok=lambda v: f"Success: {v}",
            on_err=lambda e: f"Error: {e.message}",
        )
    """
    match result:
        case Ok(value):
            return on_ok(value)
        case Err(error):
            return on_err(error)


# HTTP status code mapping
ERROR_CODE_TO_HTTP_STATUS: Final[dict[ErrorCode, int]] = {
    ErrorCode.PROVIDER_UNAVAILABLE: 503,
    ErrorCode.PROVIDER_AUTH_FAILED: 401,
    ErrorCode.PROVIDER_INVALID_RESPONSE: 502,
    ErrorCode.PROVIDER_MODEL_NOT_FOUND: 404,
    ErrorCode.VALIDATION_INVALID_REQUEST: 400,
    ErrorCode.VALIDATION_MISSING_FIELD: 400,
    ErrorCode.VALIDATION_INVALID_TYPE: 400,
    ErrorCode.VALIDATION_OUT_OF_RANGE: 400,
    ErrorCode.TIMEOUT_EXCEEDED: 504,
    ErrorCode.RATE_LIMIT_EXCEEDED: 429,
    ErrorCode.QUEUE_FULL: 503,
    ErrorCode.RESOURCE_EXHAUSTED: 503,
    ErrorCode.CANCELLED: 499,
    ErrorCode.CLIENT_DISCONNECT: 499,
}


def get_http_status(error: InferenceError) -> int:
    """Map error code to HTTP status code."""
    return ERROR_CODE_TO_HTTP_STATUS.get(error.code, 500)
