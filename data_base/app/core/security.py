# ==============================================================================
# SECURITY MODULE - Authentication & Authorization
# ==============================================================================
# JWT Token Management, Password Hashing, OAuth2 Integration
# Production-ready security implementation
# ==============================================================================

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional, Union

from jose import JWTError, jwt
from passlib.context import CryptContext

from app.core.settings import settings
from app.core.exceptions import (
    AuthenticationError,
    TokenExpiredError,
    InvalidTokenError,
)


# ==============================================================================
# PASSWORD HASHING
# ==============================================================================

# Configure password hashing with bcrypt (industry standard)
pwd_context = CryptContext(
    schemes=["bcrypt"],
    deprecated="auto",
    bcrypt__rounds=12,  # Balanced security and performance
)


def hash_password(password: str) -> str:
    """
    Hash a plaintext password using bcrypt.
    
    Uses the bcrypt algorithm with a work factor of 12,
    providing strong protection against brute-force attacks.
    
    Args:
        password: Plaintext password to hash
        
    Returns:
        Hashed password string safe for storage
        
    Example:
        >>> hashed = hash_password("my_secure_password")
        >>> verify_password("my_secure_password", hashed)
        True
    """
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    Verify a plaintext password against its hash.
    
    Uses constant-time comparison to prevent timing attacks.
    
    Args:
        plain_password: Plaintext password to verify
        hashed_password: Stored password hash
        
    Returns:
        True if password matches, False otherwise
    """
    return pwd_context.verify(plain_password, hashed_password)


# ==============================================================================
# JWT TOKEN MANAGEMENT
# ==============================================================================

class TokenType:
    """Token type constants."""
    ACCESS = "access"
    REFRESH = "refresh"


def create_access_token(
    subject: Union[str, Any],
    expires_delta: Optional[timedelta] = None,
    additional_claims: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Create a JWT access token.
    
    Access tokens are short-lived and used for API authentication.
    They contain the user identifier and optional additional claims.
    
    Args:
        subject: Token subject (usually user ID)
        expires_delta: Custom expiration time (default from settings)
        additional_claims: Extra claims to include in token
        
    Returns:
        Encoded JWT access token string
        
    Example:
        >>> token = create_access_token(subject="user-uuid-123")
        >>> payload = decode_token(token)
        >>> payload["sub"]
        'user-uuid-123'
    """
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(
            minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES
        )
    
    to_encode: Dict[str, Any] = {
        "sub": str(subject),
        "exp": expire,
        "iat": datetime.now(timezone.utc),
        "type": TokenType.ACCESS,
    }
    
    if additional_claims:
        to_encode.update(additional_claims)
    
    return jwt.encode(
        to_encode,
        settings.SECRET_KEY,
        algorithm=settings.ALGORITHM,
    )


def create_refresh_token(
    subject: Union[str, Any],
    expires_delta: Optional[timedelta] = None,
) -> str:
    """
    Create a JWT refresh token.
    
    Refresh tokens are long-lived and used to obtain new access tokens
    without requiring re-authentication.
    
    Args:
        subject: Token subject (usually user ID)
        expires_delta: Custom expiration time (default from settings)
        
    Returns:
        Encoded JWT refresh token string
    """
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(
            days=settings.REFRESH_TOKEN_EXPIRE_DAYS
        )
    
    to_encode: Dict[str, Any] = {
        "sub": str(subject),
        "exp": expire,
        "iat": datetime.now(timezone.utc),
        "type": TokenType.REFRESH,
    }
    
    return jwt.encode(
        to_encode,
        settings.SECRET_KEY,
        algorithm=settings.ALGORITHM,
    )


def decode_token(token: str) -> Dict[str, Any]:
    """
    Decode and validate a JWT token.
    
    Verifies the token signature and expiration time.
    
    Args:
        token: JWT token string to decode
        
    Returns:
        Dictionary containing token payload
        
    Raises:
        TokenExpiredError: If token has expired
        InvalidTokenError: If token is invalid or malformed
    """
    try:
        payload = jwt.decode(
            token,
            settings.SECRET_KEY,
            algorithms=[settings.ALGORITHM],
        )
        return payload
    except jwt.ExpiredSignatureError:
        raise TokenExpiredError()
    except JWTError as e:
        raise InvalidTokenError(message=f"Invalid token: {str(e)}")


def verify_token(token: str) -> Optional[Dict[str, Any]]:
    """
    Verify and decode a JWT token, returning None on failure.
    
    A lenient version of decode_token that catches exceptions
    and returns None instead of raising.
    
    Args:
        token: JWT token string to verify
        
    Returns:
        Token payload dictionary if valid, None otherwise
    """
    try:
        return decode_token(token)
    except (TokenExpiredError, InvalidTokenError, AuthenticationError):
        return None


def verify_access_token(token: str) -> Dict[str, Any]:
    """
    Verify that a token is a valid access token.
    
    Args:
        token: JWT token string to verify
        
    Returns:
        Token payload dictionary
        
    Raises:
        InvalidTokenError: If token is not an access token
        TokenExpiredError: If token has expired
    """
    payload = decode_token(token)
    
    if payload.get("type") != TokenType.ACCESS:
        raise InvalidTokenError(message="Invalid token type: expected access token")
    
    return payload


def verify_refresh_token(token: str) -> Dict[str, Any]:
    """
    Verify that a token is a valid refresh token.
    
    Args:
        token: JWT token string to verify
        
    Returns:
        Token payload dictionary
        
    Raises:
        InvalidTokenError: If token is not a refresh token
        TokenExpiredError: If token has expired
    """
    payload = decode_token(token)
    
    if payload.get("type") != TokenType.REFRESH:
        raise InvalidTokenError(message="Invalid token type: expected refresh token")
    
    return payload


def get_token_subject(token: str) -> Optional[str]:
    """
    Extract the subject (user ID) from a token.
    
    Args:
        token: JWT token string
        
    Returns:
        Subject string if valid, None otherwise
    """
    payload = verify_token(token)
    if payload:
        return payload.get("sub")
    return None


# ==============================================================================
# TOKEN RESPONSE HELPERS
# ==============================================================================

def create_token_pair(
    subject: Union[str, Any],
    additional_claims: Optional[Dict[str, Any]] = None,
) -> Dict[str, str]:
    """
    Create both access and refresh tokens.
    
    Convenience function to generate a complete token pair
    for authentication responses.
    
    Args:
        subject: Token subject (usually user ID)
        additional_claims: Extra claims for access token
        
    Returns:
        Dictionary with access_token, refresh_token, and token_type
    """
    return {
        "access_token": create_access_token(
            subject=subject,
            additional_claims=additional_claims,
        ),
        "refresh_token": create_refresh_token(subject=subject),
        "token_type": "bearer",
    }
