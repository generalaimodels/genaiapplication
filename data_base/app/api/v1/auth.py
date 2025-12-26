# ==============================================================================
# AUTH ENDPOINTS - Authentication Routes
# ==============================================================================
# Login, register, token refresh endpoints
# ==============================================================================

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from typing import Annotated

from app.api.dependencies import UserServiceDep
from app.core.exceptions import AuthenticationError, AlreadyExistsError
from app.schemas.base import APIResponse
from app.schemas.user import (
    UserCreate,
    UserResponse,
    UserLogin,
    TokenResponse,
)

router = APIRouter(prefix="/auth", tags=["Authentication"])


@router.post(
    "/register",
    response_model=APIResponse[UserResponse],
    status_code=status.HTTP_201_CREATED,
    summary="Register new user",
    description="Create a new user account with email and password.",
)
async def register(
    schema: UserCreate,
    service: UserServiceDep,
) -> APIResponse[UserResponse]:
    """Register a new user."""
    try:
        user = await service.register(schema)
        return APIResponse.ok(data=user, message="User registered successfully")
    except AlreadyExistsError as e:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(e.message),
        )


@router.post(
    "/login",
    response_model=dict,
    summary="User login",
    description="Authenticate with email and password to receive JWT tokens.",
)
async def login(
    form_data: Annotated[OAuth2PasswordRequestForm, Depends()],
    service: UserServiceDep,
) -> dict:
    """Authenticate user and return tokens."""
    try:
        # OAuth2PasswordRequestForm uses 'username' field for email
        result = await service.authenticate(
            email=form_data.username,
            password=form_data.password,
        )
        return {
            "access_token": result["tokens"].access_token,
            "refresh_token": result["tokens"].refresh_token,
            "token_type": result["tokens"].token_type,
        }
    except AuthenticationError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(e.message),
            headers={"WWW-Authenticate": "Bearer"},
        )


@router.post(
    "/login/json",
    response_model=APIResponse[dict],
    summary="User login (JSON)",
    description="Authenticate with JSON payload instead of form data.",
)
async def login_json(
    credentials: UserLogin,
    service: UserServiceDep,
) -> APIResponse[dict]:
    """Authenticate user with JSON credentials."""
    try:
        result = await service.authenticate(
            email=credentials.email,
            password=credentials.password,
        )
        return APIResponse.ok(
            data={
                "user": result["user"].model_dump(),
                "tokens": result["tokens"].model_dump(),
            },
            message="Login successful",
        )
    except AuthenticationError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(e.message),
        )
