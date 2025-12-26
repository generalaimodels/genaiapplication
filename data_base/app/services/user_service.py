# ==============================================================================
# USER SERVICE - Authentication & User Management
# ==============================================================================
# Business logic for user operations and authentication
# ==============================================================================

from __future__ import annotations

from typing import Any, Dict, Optional

from app.core.security import (
    create_token_pair,
    hash_password,
    verify_password,
)
from app.core.exceptions import (
    AuthenticationError,
    AlreadyExistsError,
    NotFoundError,
)
from app.core.constants import ErrorMessages
from app.database.adapters.base_adapter import BaseDatabaseAdapter
from app.schemas.user import (
    UserCreate,
    UserUpdate,
    UserResponse,
    TokenResponse,
)
from app.services.base_service import BaseService


class UserService(
    BaseService[Any, UserCreate, UserUpdate, UserResponse]
):
    """
    User service for authentication and profile management.
    
    Provides user registration, authentication, profile updates,
    and account management functionality.
    """
    
    def __init__(self, adapter: BaseDatabaseAdapter) -> None:
        """Initialize user service."""
        super().__init__(adapter, "users")
    
    def _to_response(self, entity: Any) -> UserResponse:
        """Convert user entity to response schema."""
        if isinstance(entity, dict):
            return UserResponse.model_validate(entity)
        return UserResponse.model_validate(entity, from_attributes=True)
    
    # ==========================================================================
    # AUTHENTICATION
    # ==========================================================================
    
    async def register(self, schema: UserCreate) -> UserResponse:
        """
        Register a new user.
        
        Args:
            schema: User registration data
            
        Returns:
            Created user response
            
        Raises:
            AlreadyExistsError: If email already registered
        """
        # Check if email exists
        existing = await self._adapter.find_one(
            self._collection_name,
            {"email": schema.email},
        )
        if existing:
            raise AlreadyExistsError(
                message="Email already registered",
                resource_type="user",
            )
        
        # Hash password and create user
        data = schema.model_dump(exclude={"password"})
        data["hashed_password"] = hash_password(schema.password)
        data["is_active"] = True
        data["is_verified"] = False
        
        result = await self._adapter.create(self._collection_name, data)
        return self._to_response(result)
    
    async def authenticate(
        self,
        email: str,
        password: str,
    ) -> Dict[str, Any]:
        """
        Authenticate user and return tokens.
        
        Args:
            email: User email
            password: User password
            
        Returns:
            Dict with user and tokens
            
        Raises:
            AuthenticationError: If credentials invalid
        """
        # Find user by email
        user = await self._adapter.find_one(
            self._collection_name,
            {"email": email},
        )
        
        if not user:
            raise AuthenticationError(message=ErrorMessages.INVALID_CREDENTIALS)
        
        # Get hashed password
        hashed_password = (
            user.get("hashed_password")
            if isinstance(user, dict)
            else getattr(user, "hashed_password", None)
        )
        
        # Verify password
        if not hashed_password or not verify_password(password, hashed_password):
            raise AuthenticationError(message=ErrorMessages.INVALID_CREDENTIALS)
        
        # Check if active
        is_active = (
            user.get("is_active")
            if isinstance(user, dict)
            else getattr(user, "is_active", True)
        )
        
        if not is_active:
            raise AuthenticationError(message="Account is disabled")
        
        # Get user ID
        user_id = (
            user.get("id")
            if isinstance(user, dict)
            else getattr(user, "id")
        )
        
        # Create tokens
        tokens = create_token_pair(subject=user_id)
        
        return {
            "user": self._to_response(user),
            "tokens": TokenResponse(
                access_token=tokens["access_token"],
                refresh_token=tokens["refresh_token"],
                token_type=tokens["token_type"],
                expires_in=30 * 60,  # 30 minutes in seconds
            ),
        }
    
    # ==========================================================================
    # USER OPERATIONS
    # ==========================================================================
    
    async def get_by_email(self, email: str) -> Optional[UserResponse]:
        """Find user by email."""
        result = await self._adapter.find_one(
            self._collection_name,
            {"email": email},
        )
        return self._to_response(result) if result else None
    
    async def update_password(
        self,
        user_id: str,
        current_password: str,
        new_password: str,
    ) -> bool:
        """
        Update user password.
        
        Args:
            user_id: User ID
            current_password: Current password for verification
            new_password: New password to set
            
        Returns:
            True if successful
            
        Raises:
            AuthenticationError: If current password wrong
            NotFoundError: If user not found
        """
        user = await self._adapter.get_by_id(self._collection_name, user_id)
        
        if not user:
            raise NotFoundError(
                message="User not found",
                resource_type="user",
                resource_id=user_id,
            )
        
        # Get hashed password
        hashed_password = (
            user.get("hashed_password")
            if isinstance(user, dict)
            else getattr(user, "hashed_password")
        )
        
        # Verify current password
        if not verify_password(current_password, hashed_password):
            raise AuthenticationError(message="Current password is incorrect")
        
        # Update password
        await self._adapter.update(
            self._collection_name,
            user_id,
            {"hashed_password": hash_password(new_password)},
        )
        
        return True
    
    async def deactivate(self, user_id: str) -> bool:
        """Deactivate a user account."""
        result = await self._adapter.update(
            self._collection_name,
            user_id,
            {"is_active": False},
        )
        if not result:
            raise NotFoundError(
                message="User not found",
                resource_type="user",
                resource_id=user_id,
            )
        return True
    
    async def activate(self, user_id: str) -> bool:
        """Activate a user account."""
        result = await self._adapter.update(
            self._collection_name,
            user_id,
            {"is_active": True},
        )
        if not result:
            raise NotFoundError(
                message="User not found",
                resource_type="user",
                resource_id=user_id,
            )
        return True
