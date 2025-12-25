# =============================================================================
# SOTA AUTHENTICATION SYSTEM - HEALTH ROUTES
# =============================================================================
# File: api/v1/health_routes.py
# Description: Health check endpoints for monitoring and orchestration
# =============================================================================

from datetime import datetime, timezone
from typing import Dict, Any

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from db.factory import DBFactory
from core.config import settings


router = APIRouter(tags=["Health"])


# =============================================================================
# RESPONSE MODELS
# =============================================================================

class HealthResponse(BaseModel):
    """Health check response model."""
    status: str = Field(..., description="Overall status: healthy, degraded, unhealthy")
    timestamp: datetime = Field(..., description="Check timestamp")
    version: str = Field(..., description="Application version")
    environment: str = Field(..., description="Environment name")


class DetailedHealthResponse(HealthResponse):
    """Detailed health check with component statuses."""
    components: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict,
        description="Individual component health"
    )


# =============================================================================
# HEALTH ENDPOINTS
# =============================================================================

@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Basic health check",
    description="Quick health check for load balancers.",
)
async def health_check() -> HealthResponse:
    """
    Basic health check.
    
    Returns minimal health status for load balancer probes.
    This endpoint should be fast and not check dependencies.
    """
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now(timezone.utc),
        version="1.0.0",
        environment=settings.app_env,
    )


@router.get(
    "/health/ready",
    response_model=DetailedHealthResponse,
    summary="Readiness check",
    description="Check if the service is ready to handle requests.",
)
async def readiness_check() -> DetailedHealthResponse:
    """
    Readiness check.
    
    Verifies all dependencies are available and the service can
    handle requests. Used by Kubernetes readiness probes.
    """
    components = {}
    overall_status = "healthy"
    
    # Check database
    try:
        db_adapter = DBFactory.get_db_adapter()
        async with db_adapter.get_session() as session:
            await session.execute("SELECT 1")
        components["database"] = {
            "status": "healthy",
            "type": settings.db_type,
        }
    except Exception as e:
        components["database"] = {
            "status": "unhealthy",
            "error": str(e),
        }
        overall_status = "degraded"
    
    # Check Redis
    try:
        redis_adapter = DBFactory.get_redis_adapter()
        if await redis_adapter.check_health():
            components["redis"] = {"status": "healthy"}
        else:
            components["redis"] = {"status": "unhealthy"}
            overall_status = "degraded"
    except Exception as e:
        components["redis"] = {
            "status": "unhealthy",
            "error": str(e),
        }
        overall_status = "degraded"
    
    return DetailedHealthResponse(
        status=overall_status,
        timestamp=datetime.now(timezone.utc),
        version="1.0.0",
        environment=settings.app_env,
        components=components,
    )


@router.get(
    "/health/live",
    response_model=HealthResponse,
    summary="Liveness check",
    description="Check if the service is alive.",
)
async def liveness_check() -> HealthResponse:
    """
    Liveness check.
    
    Simple check that the process is running.
    Used by Kubernetes liveness probes.
    """
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now(timezone.utc),
        version="1.0.0",
        environment=settings.app_env,
    )
