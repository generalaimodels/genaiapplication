# ==============================================================================
# CORE PACKAGE INITIALIZATION
# ==============================================================================
# Core utilities: Settings, Security, Exceptions, Constants
# ==============================================================================

"""
Core Module
===========

Contains core utilities and configurations for the application:
- settings: Environment configuration management
- security: JWT authentication and password hashing
- exceptions: Custom exception classes
- constants: Application-wide constants
"""

from app.core.settings import settings, get_settings, DatabaseType
from app.core.exceptions import (
    AppException,
    DatabaseError,
    NotFoundError,
    ValidationError,
    AuthenticationError,
    AuthorizationError,
    RateLimitError,
    TransactionError,
    InsufficientFundsError,
)

__all__ = [
    "settings",
    "get_settings",
    "DatabaseType",
    "AppException",
    "DatabaseError",
    "NotFoundError",
    "ValidationError",
    "AuthenticationError",
    "AuthorizationError",
    "RateLimitError",
    "TransactionError",
    "InsufficientFundsError",
]
