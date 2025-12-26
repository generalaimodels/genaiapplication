# ==============================================================================
# USER SCHEMAS - Authentication & Profile
# ==============================================================================
# Request/Response schemas for user management
# ==============================================================================

from __future__ import annotations

from typing import Optional

from pydantic import EmailStr, Field, field_validator

from app.schemas.base import BaseSchema, TimestampSchema


class UserCreate(BaseSchema):
    """Schema for user registration."""
    
    email: EmailStr = Field(
        ...,
        description="User email address",
        examples=["user@example.com"],
    )
    password: str = Field(
        ...,
        min_length=8,
        max_length=128,
        description="User password (min 8 chars)",
    )
    full_name: Optional[str] = Field(
        None,
        max_length=255,
        description="Full display name",
    )
    
    @field_validator("password")
    @classmethod
    def validate_password_strength(cls, v: str) -> str:
        """Ensure password meets security requirements."""
        if len(v) < 8:
            raise ValueError("Password must be at least 8 characters")
        if not any(c.isupper() for c in v):
            raise ValueError("Password must contain at least one uppercase letter")
        if not any(c.islower() for c in v):
            raise ValueError("Password must contain at least one lowercase letter")
        if not any(c.isdigit() for c in v):
            raise ValueError("Password must contain at least one digit")
        return v


class UserUpdate(BaseSchema):
    """Schema for updating user profile."""
    
    full_name: Optional[str] = Field(
        None,
        max_length=255,
        description="Full display name",
    )
    bio: Optional[str] = Field(
        None,
        max_length=1000,
        description="User bio/description",
    )
    avatar_url: Optional[str] = Field(
        None,
        max_length=500,
        description="Profile picture URL",
    )


class UserResponse(TimestampSchema):
    """Schema for user response (public profile)."""
    
    id: str = Field(
        ...,
        description="User unique identifier",
    )
    email: EmailStr = Field(
        ...,
        description="User email address",
    )
    full_name: Optional[str] = Field(
        None,
        description="Full display name",
    )
    bio: Optional[str] = Field(
        None,
        description="User bio/description",
    )
    avatar_url: Optional[str] = Field(
        None,
        description="Profile picture URL",
    )
    is_active: bool = Field(
        ...,
        description="Account activation status",
    )
    is_verified: bool = Field(
        ...,
        description="Email verification status",
    )


class UserLogin(BaseSchema):
    """Schema for user login request."""
    
    email: EmailStr = Field(
        ...,
        description="User email address",
    )
    password: str = Field(
        ...,
        description="User password",
    )


class TokenResponse(BaseSchema):
    """Schema for authentication token response."""
    
    access_token: str = Field(
        ...,
        description="JWT access token",
    )
    refresh_token: str = Field(
        ...,
        description="JWT refresh token",
    )
    token_type: str = Field(
        "bearer",
        description="Token type",
    )
    expires_in: int = Field(
        ...,
        description="Token expiry in seconds",
    )


class PasswordChange(BaseSchema):
    """Schema for password change request."""
    
    current_password: str = Field(
        ...,
        description="Current password",
    )
    new_password: str = Field(
        ...,
        min_length=8,
        max_length=128,
        description="New password",
    )
    
    @field_validator("new_password")
    @classmethod
    def validate_new_password(cls, v: str) -> str:
        """Validate new password strength."""
        if len(v) < 8:
            raise ValueError("Password must be at least 8 characters")
        return v


class PasswordReset(BaseSchema):
    """Schema for password reset request."""
    
    email: EmailStr = Field(
        ...,
        description="User email address",
    )


class PasswordResetConfirm(BaseSchema):
    """Schema for password reset confirmation."""
    
    token: str = Field(
        ...,
        description="Reset token",
    )
    new_password: str = Field(
        ...,
        min_length=8,
        max_length=128,
        description="New password",
    )
