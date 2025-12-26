# ==============================================================================
# MAIN API ROUTER - Route Aggregation
# ==============================================================================
# Combines all API version routers
# ==============================================================================

from __future__ import annotations

from fastapi import APIRouter

from app.core.settings import settings
from app.api.v1 import (
    auth_router,
    users_router,
    chat_router,
    transactions_router,
)

# Create main API router
api_router = APIRouter()

# Include v1 routers with API prefix
api_router.include_router(auth_router, prefix=settings.API_V1_PREFIX)
api_router.include_router(users_router, prefix=settings.API_V1_PREFIX)
api_router.include_router(chat_router, prefix=settings.API_V1_PREFIX)
api_router.include_router(transactions_router, prefix=settings.API_V1_PREFIX)
