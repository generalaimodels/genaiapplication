# =============================================================================
# API V1 MODULE INITIALIZATION
# =============================================================================
# File: api/v1/__init__.py
# Description: API v1 module exports and router aggregation
# =============================================================================

from fastapi import APIRouter

from api.v1.auth_routes import router as auth_router
from api.v1.user_routes import router as user_router
from api.v1.session_routes import router as session_router
from api.v1.health_routes import router as health_router


# Create main API v1 router
api_router = APIRouter(prefix="/api/v1")

# Include sub-routers
api_router.include_router(auth_router)
api_router.include_router(user_router)
api_router.include_router(session_router)

# Health routes at root level (no /api/v1 prefix)
health_api_router = APIRouter()
health_api_router.include_router(health_router)


__all__ = [
    "api_router",
    "auth_router",
    "user_router",
    "session_router",
]
