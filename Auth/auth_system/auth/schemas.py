# =============================================================================
# SOTA AUTHENTICATION SYSTEM - AUTH SCHEMAS
# =============================================================================
# File: auth/schemas.py
# Description: Pydantic models for request/response validation
#              Type-safe data transfer objects for the authentication API
# =============================================================================

from typing import Optional, List
from datetime import datetime
from uuid import UUID
import re

from pydantic import BaseModel, EmailStr, Field, field_validator, ConfigDict


# =============================================================================
# BASE SCHEMAS
# =============================================================================

class BaseSchema(BaseModel):
    """Base schema with common configuration."""
    model_config = ConfigDict(
        from_attributes=True,
        str_strip_whitespace=True,
    )


# =============================================================================
# USER SCHEMAS
# =============================================================================

class UserCreate(BaseSchema):
    """
    Schema for user registration request.
    
    Validation Rules:
        - email: Valid email format (RFC 5322)
        - username: 3-50 characters, alphanumeric with underscores
        - password: 8-128 characters, complexity requirements
    """
    email: EmailStr = Field(
        ...,
        description="User's email address",
        examples=["user@example.com"]
    )
    username: str = Field(
        ...,
        min_length=3,
        max_length=50,
        description="Unique username (alphanumeric and underscores)",
        examples=["john_doe"]
    )
    password: str = Field(
        ...,
        min_length=8,
        max_length=128,
        description="Password (min 8 chars, requires uppercase, lowercase, digit, special)",
        examples=["SecurePass123!"]
    )
    
    @field_validator("username")
    @classmethod
    def validate_username(cls, v: str) -> str:
        """Validate username format."""
        if not re.match(r"^[a-zA-Z][a-zA-Z0-9_]*$", v):
            raise ValueError(
                "Username must start with a letter and contain only "
                "letters, numbers, and underscores"
            )
        return v.lower()
    
    @field_validator("password")
    @classmethod
    def validate_password_complexity(cls, v: str) -> str:
        """Validate password complexity requirements."""
        errors = []
        
        if not any(c.isupper() for c in v):
            errors.append("at least one uppercase letter")
        if not any(c.islower() for c in v):
            errors.append("at least one lowercase letter")
        if not any(c.isdigit() for c in v):
            errors.append("at least one digit")
        if not any(c in "!@#$%^&*()_+-=[]{}|;':\",./<>?`~" for c in v):
            errors.append("at least one special character")
        
        if errors:
            raise ValueError(f"Password must contain: {', '.join(errors)}")
        
        return v


class UserUpdate(BaseSchema):
    """Schema for updating user profile."""
    username: Optional[str] = Field(
        None,
        min_length=3,
        max_length=50,
        description="New username"
    )
    email: Optional[EmailStr] = Field(
        None,
        description="New email address"
    )
    
    @field_validator("username")
    @classmethod
    def validate_username(cls, v: Optional[str]) -> Optional[str]:
        """Validate username format if provided."""
        if v is not None:
            if not re.match(r"^[a-zA-Z][a-zA-Z0-9_]*$", v):
                raise ValueError(
                    "Username must start with a letter and contain only "
                    "letters, numbers, and underscores"
                )
            return v.lower()
        return v


class UserResponse(BaseSchema):
    """Schema for user response (excludes sensitive data)."""
    id: str = Field(..., description="User UUID")
    email: str = Field(..., description="Email address")
    username: str = Field(..., description="Username")
    is_active: bool = Field(..., description="Account active status")
    is_verified: bool = Field(..., description="Email verified status")
    is_superuser: bool = Field(False, description="Admin status")
    created_at: datetime = Field(..., description="Account creation date")
    last_login: Optional[datetime] = Field(None, description="Last login time")


class UserInDB(UserResponse):
    """Full user model with password hash (internal use)."""
    password_hash: str
    failed_attempts: int = 0
    locked_until: Optional[datetime] = None


# =============================================================================
# AUTHENTICATION SCHEMAS
# =============================================================================

class LoginRequest(BaseSchema):
    """Schema for login request."""
    email: EmailStr = Field(
        ...,
        description="User's email address",
        examples=["user@example.com"]
    )
    password: str = Field(
        ...,
        min_length=1,
        description="User's password"
    )
    remember_me: bool = Field(
        False,
        description="Extend session duration"
    )


class TokenResponse(BaseSchema):
    """Schema for authentication token response."""
    access_token: str = Field(
        ...,
        description="JWT access token"
    )
    refresh_token: str = Field(
        ...,
        description="JWT refresh token"
    )
    token_type: str = Field(
        "bearer",
        description="Token type"
    )
    expires_in: int = Field(
        ...,
        description="Access token expiration in seconds"
    )
    session_id: str = Field(
        ...,
        description="Session identifier"
    )


class RefreshRequest(BaseSchema):
    """Schema for token refresh request."""
    refresh_token: str = Field(
        ...,
        description="Refresh token"
    )


class RefreshResponse(BaseSchema):
    """Schema for token refresh response."""
    access_token: str = Field(
        ...,
        description="New JWT access token"
    )
    token_type: str = Field(
        "bearer",
        description="Token type"
    )
    expires_in: int = Field(
        ...,
        description="Access token expiration in seconds"
    )


# =============================================================================
# PASSWORD SCHEMAS
# =============================================================================

class PasswordChangeRequest(BaseSchema):
    """Schema for password change request."""
    current_password: str = Field(
        ...,
        description="Current password"
    )
    new_password: str = Field(
        ...,
        min_length=8,
        max_length=128,
        description="New password"
    )
    
    @field_validator("new_password")
    @classmethod
    def validate_password(cls, v: str) -> str:
        """Validate new password complexity."""
        errors = []
        
        if not any(c.isupper() for c in v):
            errors.append("uppercase letter")
        if not any(c.islower() for c in v):
            errors.append("lowercase letter")
        if not any(c.isdigit() for c in v):
            errors.append("digit")
        if not any(c in "!@#$%^&*()_+-=[]{}|;':\",./<>?`~" for c in v):
            errors.append("special character")
        
        if errors:
            raise ValueError(f"Password must contain: {', '.join(errors)}")
        
        return v


class PasswordResetRequest(BaseSchema):
    """Schema for password reset request (forgot password)."""
    email: EmailStr = Field(
        ...,
        description="User's email address"
    )


class PasswordResetConfirm(BaseSchema):
    """Schema for password reset confirmation."""
    token: str = Field(
        ...,
        description="Password reset token"
    )
    new_password: str = Field(
        ...,
        min_length=8,
        max_length=128,
        description="New password"
    )
    
    @field_validator("new_password")
    @classmethod
    def validate_password(cls, v: str) -> str:
        """Validate new password complexity."""
        errors = []
        
        if not any(c.isupper() for c in v):
            errors.append("uppercase letter")
        if not any(c.islower() for c in v):
            errors.append("lowercase letter")
        if not any(c.isdigit() for c in v):
            errors.append("digit")
        if not any(c in "!@#$%^&*()_+-=[]{}|;':\",./<>?`~" for c in v):
            errors.append("special character")
        
        if errors:
            raise ValueError(f"Password must contain: {', '.join(errors)}")
        
        return v


class EmailVerifyRequest(BaseSchema):
    """Schema for email verification request."""
    token: str = Field(
        ...,
        description="Email verification token"
    )


# =============================================================================
# SESSION SCHEMAS
# =============================================================================

class SessionResponse(BaseSchema):
    """Schema for session information."""
    session_id: str = Field(
        ...,
        description="Session identifier"
    )
    device_info: Optional[dict] = Field(
        None,
        description="Device information"
    )
    ip_address: str = Field(
        ...,
        description="Client IP address"
    )
    created_at: datetime = Field(
        ...,
        description="Session creation time"
    )
    last_activity: datetime = Field(
        ...,
        description="Last activity time"
    )
    is_current: bool = Field(
        False,
        description="Whether this is the current session"
    )


class SessionListResponse(BaseSchema):
    """Schema for session list response."""
    sessions: List[SessionResponse] = Field(
        ...,
        description="List of active sessions"
    )
    total: int = Field(
        ...,
        description="Total number of sessions"
    )


# =============================================================================
# API RESPONSE SCHEMAS
# =============================================================================

class MessageResponse(BaseSchema):
    """Generic message response."""
    message: str = Field(
        ...,
        description="Response message"
    )
    success: bool = Field(
        True,
        description="Operation success status"
    )


class ErrorResponse(BaseSchema):
    """Error response schema."""
    error: bool = Field(
        True,
        description="Error flag"
    )
    error_code: str = Field(
        ...,
        description="Error code"
    )
    message: str = Field(
        ...,
        description="Error message"
    )
    details: Optional[dict] = Field(
        None,
        description="Additional error details"
    )


# =============================================================================
# ADMIN SCHEMAS
# =============================================================================

class AdminUserUpdate(BaseSchema):
    """Schema for admin user update."""
    is_active: Optional[bool] = Field(
        None,
        description="Account active status"
    )
    is_verified: Optional[bool] = Field(
        None,
        description="Email verified status"
    )
    is_superuser: Optional[bool] = Field(
        None,
        description="Admin status"
    )


class UserListResponse(BaseSchema):
    """Schema for paginated user list."""
    users: List[UserResponse] = Field(
        ...,
        description="List of users"
    )
    total: int = Field(
        ...,
        description="Total number of users"
    )
    page: int = Field(
        ...,
        description="Current page"
    )
    page_size: int = Field(
        ...,
        description="Page size"
    )
