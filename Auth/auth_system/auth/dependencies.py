# =============================================================================
# SOTA AUTHENTICATION SYSTEM - AUTH DEPENDENCIES
# =============================================================================
# File: auth/dependencies.py
# Description: FastAPI dependencies for authentication and authorization
#              Provides reusable dependency injection for protected routes
# =============================================================================

from typing import Optional, Annotated
from fastapi import Depends, Header, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from sqlalchemy.ext.asyncio import AsyncSession

from db.factory import get_db_session, get_redis
from db.adapters.redis_adapter import RedisAdapter
from db.models import User
from auth.repository import UserRepository, SessionRepository
from auth.service import AuthService
from core.security import jwt_manager, TokenPayload
from core.config import settings
from core.exceptions import (
    TokenMissingError,
    TokenInvalidError,
    TokenExpiredError,
    TokenBlacklistedError,
    UserNotFoundError,
    UserInactiveError,
    AuthorizationError,
    InsufficientPermissionsError,
)


# =============================================================================
# SECURITY SCHEME
# =============================================================================

# OAuth2 Bearer token scheme
bearer_scheme = HTTPBearer(
    scheme_name="Bearer",
    description="JWT access token",
    auto_error=False,
)


# =============================================================================
# DATABASE DEPENDENCIES
# =============================================================================

async def get_db_session_dep() -> AsyncSession:
    """
    Dependency for database session.
    
    Yields:
        AsyncSession: Database session with auto-commit/rollback
    """
    async for session in get_db_session():
        yield session


DBSession = Annotated[AsyncSession, Depends(get_db_session_dep)]


async def get_redis_dep() -> RedisAdapter:
    """
    Dependency for Redis client.
    
    Returns:
        RedisAdapter: Redis client instance
    """
    return await get_redis()


Redis = Annotated[RedisAdapter, Depends(get_redis_dep)]


# =============================================================================
# SERVICE DEPENDENCIES
# =============================================================================

async def get_auth_service(
    session: DBSession,
) -> AuthService:
    """
    Dependency for authentication service.
    
    Args:
        session: Database session
        
    Returns:
        AuthService: Configured auth service
    """
    return AuthService(session)


AuthServiceDep = Annotated[AuthService, Depends(get_auth_service)]


# =============================================================================
# TOKEN EXTRACTION
# =============================================================================

async def get_token_from_header(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme),
) -> str:
    """
    Extract JWT token from Authorization header.
    
    Args:
        credentials: Bearer credentials from HTTPBearer
        
    Returns:
        str: JWT token
        
    Raises:
        TokenMissingError: If token not provided
    """
    if credentials is None:
        raise TokenMissingError()
    
    return credentials.credentials


Token = Annotated[str, Depends(get_token_from_header)]


async def get_optional_token(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme),
) -> Optional[str]:
    """
    Optionally extract JWT token (for public endpoints).
    
    Args:
        credentials: Bearer credentials from HTTPBearer
        
    Returns:
        Optional token string
    """
    if credentials is None:
        return None
    return credentials.credentials


OptionalToken = Annotated[Optional[str], Depends(get_optional_token)]


# =============================================================================
# TOKEN VALIDATION
# =============================================================================

async def get_current_token(
    token: Token,
    redis: Redis,
) -> TokenPayload:
    """
    Validate and decode JWT token.
    
    Args:
        token: JWT token string
        redis: Redis client for blacklist check
        
    Returns:
        TokenPayload: Decoded token payload
        
    Raises:
        TokenExpiredError: If token expired
        TokenInvalidError: If token invalid
        TokenBlacklistedError: If token revoked
    """
    # Decode and verify token
    payload = jwt_manager.verify_token(token, expected_type="access")
    
    # Check if token is blacklisted
    is_blacklisted = await redis.exists(f"blacklist:{payload.jti}")
    if is_blacklisted:
        raise TokenBlacklistedError()
    
    return payload


CurrentToken = Annotated[TokenPayload, Depends(get_current_token)]


# =============================================================================
# USER DEPENDENCIES
# =============================================================================

async def get_current_user(
    token_payload: CurrentToken,
    session: DBSession,
) -> User:
    """
    Get current authenticated user from token.
    
    Args:
        token_payload: Validated token payload
        session: Database session
        
    Returns:
        User: Current user entity
        
    Raises:
        UserNotFoundError: If user not found
    """
    user_repo = UserRepository(session)
    user = await user_repo.get_by_id(token_payload.sub)
    
    if not user:
        raise UserNotFoundError(user_id=token_payload.sub)
    
    return user


CurrentUser = Annotated[User, Depends(get_current_user)]


async def get_current_active_user(
    user: CurrentUser,
) -> User:
    """
    Get current user ensuring they are active.
    
    Args:
        user: Current user
        
    Returns:
        User: Active user
        
    Raises:
        UserInactiveError: If user is inactive
    """
    if not user.is_active:
        raise UserInactiveError()
    
    return user


ActiveUser = Annotated[User, Depends(get_current_active_user)]


async def get_current_verified_user(
    user: ActiveUser,
) -> User:
    """
    Get current user ensuring email is verified.
    
    Args:
        user: Active user
        
    Returns:
        User: Verified user
        
    Raises:
        AuthorizationError: If email not verified
    """
    if not user.is_verified:
        raise AuthorizationError(
            message="Email verification required",
            error_code="EMAIL_NOT_VERIFIED"
        )
    
    return user


VerifiedUser = Annotated[User, Depends(get_current_verified_user)]


async def get_current_superuser(
    user: ActiveUser,
) -> User:
    """
    Get current user ensuring they are a superuser.
    
    Args:
        user: Active user
        
    Returns:
        User: Superuser
        
    Raises:
        InsufficientPermissionsError: If not superuser
    """
    if not user.is_superuser:
        raise InsufficientPermissionsError(required_permission="admin")
    
    return user


SuperUser = Annotated[User, Depends(get_current_superuser)]


# =============================================================================
# SESSION DEPENDENCIES
# =============================================================================

async def get_current_session(
    token_payload: CurrentToken,
    session: DBSession,
):
    """
    Get current session from token.
    
    Args:
        token_payload: Validated token payload
        session: Database session
        
    Returns:
        Session: Current session or None
    """
    if not token_payload.session_id:
        return None
    
    session_repo = SessionRepository(session)
    session_obj = await session_repo.get_by_id(token_payload.session_id)
    
    if session_obj and session_obj.is_active and not session_obj.is_expired:
        # Update activity
        await session_repo.update_activity(session_obj)
        return session_obj
    
    return None


# =============================================================================
# REQUEST CONTEXT
# =============================================================================

def get_client_ip(request: Request) -> str:
    """
    Extract client IP from request.
    
    Handles proxy headers (X-Forwarded-For, X-Real-IP).
    
    Args:
        request: FastAPI request
        
    Returns:
        str: Client IP address
    """
    # Check proxy headers
    forwarded_for = request.headers.get("X-Forwarded-For")
    if forwarded_for:
        # Take first IP in chain
        return forwarded_for.split(",")[0].strip()
    
    real_ip = request.headers.get("X-Real-IP")
    if real_ip:
        return real_ip.strip()
    
    # Fallback to direct client
    return request.client.host if request.client else "unknown"


def get_user_agent(
    user_agent: Optional[str] = Header(None, alias="User-Agent"),
) -> Optional[str]:
    """
    Extract user agent from request headers.
    
    Args:
        user_agent: User-Agent header
        
    Returns:
        User agent string or None
    """
    return user_agent


ClientIP = Annotated[str, Depends(get_client_ip)]
UserAgent = Annotated[Optional[str], Depends(get_user_agent)]


# =============================================================================
# COMBINED DEPENDENCIES
# =============================================================================

class AuthContext:
    """Container for authentication context."""
    
    def __init__(
        self,
        user: User,
        token: TokenPayload,
        session_id: Optional[str],
        ip_address: str,
        user_agent: Optional[str],
    ):
        self.user = user
        self.token = token
        self.session_id = session_id
        self.ip_address = ip_address
        self.user_agent = user_agent


async def get_auth_context(
    user: ActiveUser,
    token: CurrentToken,
    request: Request,
    user_agent: UserAgent,
) -> AuthContext:
    """
    Get complete authentication context.
    
    Args:
        user: Current active user
        token: Current token payload
        request: FastAPI request
        user_agent: User agent string
        
    Returns:
        AuthContext: Complete auth context
    """
    return AuthContext(
        user=user,
        token=token,
        session_id=token.session_id,
        ip_address=get_client_ip(request),
        user_agent=user_agent,
    )


AuthContextDep = Annotated[AuthContext, Depends(get_auth_context)]
