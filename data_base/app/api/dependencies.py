# ==============================================================================
# API DEPENDENCIES - Dependency Injection
# ==============================================================================
# FastAPI dependencies for authentication and database access
# ==============================================================================

from __future__ import annotations

from typing import Annotated, Optional

from fastapi import Depends, Header, HTTPException, status
from fastapi.security import OAuth2PasswordBearer

from app.core.settings import settings
from app.core.security import decode_token, TokenType
from app.core.exceptions import AuthenticationError, TokenExpiredError, InvalidTokenError
from app.database.factory import DatabaseFactory
from app.database.adapters.base_adapter import BaseDatabaseAdapter
from app.services.user_service import UserService
from app.services.chat_service import ChatService
from app.services.transaction_service import TransactionService

# OAuth2 scheme for JWT tokens
oauth2_scheme = OAuth2PasswordBearer(
    tokenUrl=f"{settings.API_V1_PREFIX}/auth/login",
    auto_error=False,
)


# ==============================================================================
# DATABASE DEPENDENCIES
# ==============================================================================

async def get_adapter() -> BaseDatabaseAdapter:
    """
    Get database adapter dependency.
    
    Returns initialized adapter from factory.
    """
    return DatabaseFactory.get_adapter()


# Annotated type for database adapter
DatabaseDep = Annotated[BaseDatabaseAdapter, Depends(get_adapter)]


# ==============================================================================
# AUTHENTICATION DEPENDENCIES
# ==============================================================================

async def get_current_user_id(
    token: Annotated[Optional[str], Depends(oauth2_scheme)],
) -> str:
    """
    Extract user ID from JWT token.
    
    Args:
        token: JWT access token from Authorization header
        
    Returns:
        User ID string
        
    Raises:
        HTTPException: If token invalid or missing
    """
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    try:
        payload = decode_token(token)
        
        # Verify it's an access token
        if payload.get("type") != TokenType.ACCESS:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token type",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        user_id = payload.get("sub")
        if not user_id:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token payload",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        return user_id
        
    except TokenExpiredError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except InvalidTokenError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(e),
            headers={"WWW-Authenticate": "Bearer"},
        )


async def get_optional_user_id(
    token: Annotated[Optional[str], Depends(oauth2_scheme)],
) -> Optional[str]:
    """
    Extract user ID if token provided, otherwise None.
    
    Useful for endpoints that work for both authenticated
    and unauthenticated users.
    """
    if not token:
        return None
    
    try:
        return await get_current_user_id(token)
    except HTTPException:
        return None


# Annotated types
CurrentUserID = Annotated[str, Depends(get_current_user_id)]
OptionalUserID = Annotated[Optional[str], Depends(get_optional_user_id)]


# ==============================================================================
# SERVICE DEPENDENCIES
# ==============================================================================

async def get_user_service(
    adapter: DatabaseDep,
) -> UserService:
    """Get user service instance."""
    return UserService(adapter)


async def get_chat_service(
    adapter: DatabaseDep,
) -> ChatService:
    """Get chat service instance."""
    return ChatService(adapter)


async def get_transaction_service(
    adapter: DatabaseDep,
) -> TransactionService:
    """Get transaction service instance."""
    return TransactionService(adapter)


# Annotated service types
UserServiceDep = Annotated[UserService, Depends(get_user_service)]
ChatServiceDep = Annotated[ChatService, Depends(get_chat_service)]
TransactionServiceDep = Annotated[TransactionService, Depends(get_transaction_service)]
