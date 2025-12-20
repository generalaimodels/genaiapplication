# =============================================================================
# EXCEPTIONS - Custom Exception Hierarchy
# =============================================================================
# Structured exceptions for the AI Core system.
# =============================================================================

from __future__ import annotations
from typing import Optional, Dict, Any


class AICorError(Exception):
    """
    Base exception for AI Core.
    
    Attributes:
        message: Error message
        code: Error code
        details: Additional error details
    """
    
    def __init__(
        self,
        message: str,
        code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        self.message = message
        self.code = code or "AICORE_ERROR"
        self.details = details or {}
        super().__init__(self.message)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "error": self.code,
            "message": self.message,
            "details": self.details,
        }


class ConfigurationError(AICorError):
    """Configuration-related errors."""
    
    def __init__(self, message: str, details: Optional[Dict] = None):
        super().__init__(message, "CONFIGURATION_ERROR", details)


class ProviderError(AICorError):
    """LLM provider-related errors."""
    
    def __init__(
        self,
        message: str,
        provider: Optional[str] = None,
        details: Optional[Dict] = None
    ):
        details = details or {}
        details["provider"] = provider
        super().__init__(message, "PROVIDER_ERROR", details)


class AuthenticationError(AICorError):
    """Authentication and authorization errors."""
    
    def __init__(self, message: str, provider: Optional[str] = None):
        super().__init__(
            message, 
            "AUTHENTICATION_ERROR",
            {"provider": provider}
        )


class RateLimitError(AICorError):
    """Rate limit exceeded errors."""
    
    def __init__(
        self,
        message: str,
        retry_after: Optional[int] = None,
        provider: Optional[str] = None
    ):
        super().__init__(
            message,
            "RATE_LIMIT_ERROR",
            {"retry_after": retry_after, "provider": provider}
        )


class ModelNotFoundError(AICorError):
    """Model not found or not accessible."""
    
    def __init__(self, model: str, provider: Optional[str] = None):
        super().__init__(
            f"Model not found: {model}",
            "MODEL_NOT_FOUND",
            {"model": model, "provider": provider}
        )


class SessionError(AICorError):
    """Session-related errors."""
    
    def __init__(self, message: str, session_id: Optional[str] = None):
        super().__init__(
            message,
            "SESSION_ERROR",
            {"session_id": session_id}
        )


class DocumentError(AICorError):
    """Document processing errors."""
    
    def __init__(self, message: str, source: Optional[str] = None):
        super().__init__(
            message,
            "DOCUMENT_ERROR",
            {"source": source}
        )


class StreamError(AICorError):
    """Streaming-related errors."""
    
    def __init__(self, message: str, chunk_index: Optional[int] = None):
        super().__init__(
            message,
            "STREAM_ERROR",
            {"chunk_index": chunk_index}
        )


class ValidationError(AICorError):
    """Input/output validation errors."""
    
    def __init__(self, message: str, field: Optional[str] = None):
        super().__init__(
            message,
            "VALIDATION_ERROR",
            {"field": field}
        )
