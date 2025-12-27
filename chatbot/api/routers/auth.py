# -*- coding: utf-8 -*-
# =================================================================================================
# api/routers/auth.py — Authentication Router
# =================================================================================================
# Implements user authentication with:
#   - User registration with unique username validation
#   - User login with JWT token generation
#   - Password hashing using SHA256 with salt
#   - Cross-browser compatible (backend storage)
#
# Endpoints:
#   POST /api/v1/auth/register — Create new user account
#   POST /api/v1/auth/login    — Login and get access token
#   GET  /api/v1/auth/me       — Get current user info
#
# =================================================================================================

from __future__ import annotations

import hashlib
import json
import logging
import secrets
import time
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field, field_validator

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
_LOG = logging.getLogger("api.routers.auth")

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
# Secret key for JWT tokens (in production, load from environment)
JWT_SECRET = "cca_chatbot_secret_key_2024"
JWT_EXPIRY_DAYS = 7
PASSWORD_SALT = "cca_auth_salt_v1"

# -----------------------------------------------------------------------------
# Pydantic Schemas
# -----------------------------------------------------------------------------

class UserRegisterRequest(BaseModel):
    """Request model for user registration."""
    username: str = Field(
        min_length=3,
        max_length=50,
        description="Unique username (3-50 characters, alphanumeric and underscores)"
    )
    password: str = Field(
        min_length=4,
        max_length=128,
        description="Password (minimum 4 characters)"
    )
    
    @field_validator('username')
    @classmethod
    def validate_username(cls, v: str) -> str:
        """Validate username format."""
        if not v.replace('_', '').isalnum():
            raise ValueError('Username can only contain letters, numbers, and underscores')
        return v.lower()  # Normalize to lowercase


class UserLoginRequest(BaseModel):
    """Request model for user login."""
    username: str = Field(min_length=1, description="Username")
    password: str = Field(min_length=1, description="Password")


class UserResponse(BaseModel):
    """Response model for user data."""
    id: str
    username: str
    created_at: float


class TokenResponse(BaseModel):
    """Response model for authentication token."""
    access_token: str
    token_type: str = "bearer"
    expires_in: int = Field(description="Token validity in seconds")
    user: UserResponse


class MessageResponse(BaseModel):
    """Simple message response."""
    message: str
    success: bool = True


# -----------------------------------------------------------------------------
# Utility Functions
# -----------------------------------------------------------------------------

def hash_password(password: str) -> str:
    """
    Hash password using SHA256 with salt.
    
    Note: For production, use bcrypt or argon2.
    This is a simpler approach that doesn't require additional dependencies.
    """
    salted = f"{PASSWORD_SALT}:{password}"
    return hashlib.sha256(salted.encode()).hexdigest()


def verify_password(password: str, password_hash: str) -> bool:
    """Verify password against stored hash."""
    return hash_password(password) == password_hash


def generate_token(user_id: str, username: str) -> str:
    """
    Generate a simple JWT-like token.
    
    Format: base64(user_id:username:timestamp:signature)
    
    Note: For production, use proper JWT library (python-jose).
    This is a simpler approach without additional dependencies.
    """
    import base64
    
    expires_at = int(time.time()) + (JWT_EXPIRY_DAYS * 24 * 60 * 60)
    payload = f"{user_id}:{username}:{expires_at}"
    signature = hashlib.sha256(f"{payload}:{JWT_SECRET}".encode()).hexdigest()[:16]
    token_data = f"{payload}:{signature}"
    return base64.urlsafe_b64encode(token_data.encode()).decode()


def verify_token(token: str) -> Optional[Dict[str, Any]]:
    """
    Verify and decode token.
    
    Returns user info if valid, None if invalid/expired.
    """
    import base64
    
    try:
        token_data = base64.urlsafe_b64decode(token.encode()).decode()
        parts = token_data.split(":")
        if len(parts) != 4:
            return None
        
        user_id, username, expires_at_str, signature = parts
        expires_at = int(expires_at_str)
        
        # Check expiry
        if time.time() > expires_at:
            return None
        
        # Verify signature
        payload = f"{user_id}:{username}:{expires_at}"
        expected_sig = hashlib.sha256(f"{payload}:{JWT_SECRET}".encode()).hexdigest()[:16]
        if signature != expected_sig:
            return None
        
        return {
            "user_id": user_id,
            "username": username,
            "expires_at": expires_at
        }
    except Exception:
        return None


def generate_user_id() -> str:
    """Generate unique user ID."""
    import uuid
    return str(uuid.uuid4())


# -----------------------------------------------------------------------------
# Database Operations
# -----------------------------------------------------------------------------

async def get_user_by_username(db, username: str) -> Optional[Dict[str, Any]]:
    """Get user from database by username."""
    rows = await db.execute(
        "SELECT * FROM users WHERE username = ? AND is_active = 1",
        (username.lower(),),
        fetch=True
    )
    return rows[0] if rows else None


async def get_user_by_id(db, user_id: str) -> Optional[Dict[str, Any]]:
    """Get user from database by ID."""
    rows = await db.execute(
        "SELECT * FROM users WHERE id = ? AND is_active = 1",
        (user_id,),
        fetch=True
    )
    return rows[0] if rows else None


async def create_user(db, username: str, password: str) -> Dict[str, Any]:
    """Create new user in database."""
    user_id = generate_user_id()
    password_hash = hash_password(password)
    now = time.time()
    
    await db.execute(
        """
        INSERT INTO users (id, username, password_hash, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?)
        """,
        (user_id, username.lower(), password_hash, now, now)
    )
    
    return {
        "id": user_id,
        "username": username.lower(),
        "created_at": now,
        "updated_at": now,
        "is_active": True
    }


# -----------------------------------------------------------------------------
# Router
# -----------------------------------------------------------------------------
router = APIRouter(prefix="/auth", tags=["Authentication"])


@router.post("/register", response_model=TokenResponse, status_code=status.HTTP_201_CREATED)
async def register_user(request: UserRegisterRequest):
    """
    Register a new user account.
    
    - Validates username uniqueness
    - Hashes password securely
    - Returns access token on success
    """
    from api.dependencies import get_async_db
    db = get_async_db()
    
    # Check if username already exists
    existing_user = await get_user_by_username(db, request.username)
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={
                "error": "USERNAME_TAKEN",
                "message": "This username is already registered. Please choose a different one."
            }
        )
    
    # Create user
    user = await create_user(db, request.username, request.password)
    _LOG.info("User registered: %s", user["username"])
    
    # Generate token
    token = generate_token(user["id"], user["username"])
    expires_in = JWT_EXPIRY_DAYS * 24 * 60 * 60
    
    return TokenResponse(
        access_token=token,
        expires_in=expires_in,
        user=UserResponse(
            id=user["id"],
            username=user["username"],
            created_at=user["created_at"]
        )
    )


@router.post("/login", response_model=TokenResponse)
async def login_user(request: UserLoginRequest):
    """
    Login with username and password.
    
    - Validates credentials
    - Returns access token on success
    """
    from api.dependencies import get_async_db
    db = get_async_db()
    
    # Get user by username
    user = await get_user_by_username(db, request.username)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "error": "INVALID_CREDENTIALS",
                "message": "Invalid username or password"
            }
        )
    
    # Verify password
    if not verify_password(request.password, user["password_hash"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "error": "INVALID_CREDENTIALS",
                "message": "Invalid username or password"
            }
        )
    
    _LOG.info("User logged in: %s", user["username"])
    
    # Generate token
    token = generate_token(user["id"], user["username"])
    expires_in = JWT_EXPIRY_DAYS * 24 * 60 * 60
    
    return TokenResponse(
        access_token=token,
        expires_in=expires_in,
        user=UserResponse(
            id=user["id"],
            username=user["username"],
            created_at=user["created_at"]
        )
    )


@router.get("/me", response_model=UserResponse)
async def get_current_user(authorization: str = None):
    """
    Get current authenticated user info.
    
    Requires Authorization header with Bearer token.
    """
    from fastapi import Header
    from api.dependencies import get_async_db
    
    # This is a simplified version - in production, use proper dependency injection
    # with OAuth2PasswordBearer
    
    if not authorization:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"error": "UNAUTHORIZED", "message": "Authorization header required"}
        )
    
    # Extract token from "Bearer <token>"
    parts = authorization.split()
    if len(parts) != 2 or parts[0].lower() != "bearer":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"error": "INVALID_TOKEN", "message": "Invalid authorization header format"}
        )
    
    token = parts[1]
    token_data = verify_token(token)
    
    if not token_data:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"error": "INVALID_TOKEN", "message": "Token is invalid or expired"}
        )
    
    db = get_async_db()
    user = await get_user_by_id(db, token_data["user_id"])
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"error": "USER_NOT_FOUND", "message": "User no longer exists"}
        )
    
    return UserResponse(
        id=user["id"],
        username=user["username"],
        created_at=user["created_at"]
    )


@router.post("/verify", response_model=UserResponse)
async def verify_user_token(authorization: str = None):
    """
    Verify token and return user info.
    
    Used by frontend to check if stored token is still valid.
    """
    return await get_current_user(authorization)
