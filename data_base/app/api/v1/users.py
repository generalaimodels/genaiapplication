# ==============================================================================
# USERS ENDPOINTS - User Profile Routes
# ==============================================================================
# User profile management endpoints
# ==============================================================================

from __future__ import annotations

from fastapi import APIRouter, HTTPException, status

from app.api.dependencies import UserServiceDep, CurrentUserID
from app.core.exceptions import NotFoundError, AuthenticationError
from app.schemas.base import APIResponse
from app.schemas.user import UserUpdate, UserResponse, PasswordChange

router = APIRouter(prefix="/users", tags=["Users"])


@router.get(
    "/me",
    response_model=APIResponse[UserResponse],
    summary="Get current user",
    description="Get the profile of the currently authenticated user.",
)
async def get_current_user(
    user_id: CurrentUserID,
    service: UserServiceDep,
) -> APIResponse[UserResponse]:
    """Get current user profile."""
    try:
        user = await service.get_by_id(user_id)
        return APIResponse.ok(data=user)
    except NotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found",
        )


@router.patch(
    "/me",
    response_model=APIResponse[UserResponse],
    summary="Update current user",
    description="Update the profile of the currently authenticated user.",
)
async def update_current_user(
    user_id: CurrentUserID,
    schema: UserUpdate,
    service: UserServiceDep,
) -> APIResponse[UserResponse]:
    """Update current user profile."""
    try:
        user = await service.update(user_id, schema)
        return APIResponse.ok(data=user, message="Profile updated successfully")
    except NotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found",
        )


@router.post(
    "/me/password",
    response_model=APIResponse[dict],
    summary="Change password",
    description="Change the password of the currently authenticated user.",
)
async def change_password(
    user_id: CurrentUserID,
    schema: PasswordChange,
    service: UserServiceDep,
) -> APIResponse[dict]:
    """Change user password."""
    try:
        await service.update_password(
            user_id=user_id,
            current_password=schema.current_password,
            new_password=schema.new_password,
        )
        return APIResponse.ok(
            data={"success": True},
            message="Password changed successfully",
        )
    except AuthenticationError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e.message),
        )
    except NotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found",
        )


@router.get(
    "/{user_id}",
    response_model=APIResponse[UserResponse],
    summary="Get user by ID",
    description="Get a user's public profile by ID.",
)
async def get_user(
    user_id: str,
    service: UserServiceDep,
) -> APIResponse[UserResponse]:
    """Get user by ID."""
    try:
        user = await service.get_by_id(user_id)
        return APIResponse.ok(data=user)
    except NotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found",
        )
