# -*- coding: utf-8 -*-
# =================================================================================================
# api/routers/health.py — Health Check & Metrics Endpoints
# =================================================================================================
# Production-grade health endpoints implementing:
#
#   1. LIVENESS PROBE: Is the process running? (/health/live)
#   2. READINESS PROBE: Is the service ready to accept traffic? (/health/ready)
#   3. HEALTH CHECK: Overall health status with components (/health)
#   4. METRICS: Prometheus-style metrics for monitoring (/metrics)
#
# Kubernetes Integration:
# -----------------------
#   livenessProbe:
#     httpGet:
#       path: /api/v1/health/live
#       port: 8000
#     initialDelaySeconds: 5
#     periodSeconds: 10
#   
#   readinessProbe:
#     httpGet:
#       path: /api/v1/health/ready
#       port: 8000
#     initialDelaySeconds: 10
#     periodSeconds: 5
#
# =================================================================================================

from __future__ import annotations

import logging
import os
import platform
import time
from typing import Dict

from fastapi import APIRouter, Depends, status
from fastapi.responses import JSONResponse

from api.schemas import HealthResponse, MetricsResponse
from api.dependencies import get_service_health

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
_LOG = logging.getLogger("api.routers.health")

# -----------------------------------------------------------------------------
# Router Configuration
# -----------------------------------------------------------------------------
router = APIRouter(
    prefix="/health",
    tags=["Health"],
    responses={
        503: {"description": "Service unavailable"},
    },
)

# Startup time for uptime calculation
_startup_time = time.time()

# Request counters (simple in-memory metrics)
_request_count = 0
_error_count = 0
_total_latency_ms = 0.0


def increment_request_count(latency_ms: float = 0.0, is_error: bool = False) -> None:
    """Increment request metrics (called from middleware)."""
    global _request_count, _error_count, _total_latency_ms
    _request_count += 1
    _total_latency_ms += latency_ms
    if is_error:
        _error_count += 1


# =============================================================================
# Liveness Probe
# =============================================================================

@router.get(
    "/live",
    response_model=HealthResponse,
    summary="Liveness Probe",
    description="""
    Lightweight check to determine if the process is alive.
    
    **Use Case**: Kubernetes liveness probe to detect deadlocks or hung processes.
    
    **Returns**: Always returns 200 if the process is running.
    """,
)
async def liveness() -> HealthResponse:
    """
    Liveness check - is the process running?
    
    This endpoint should:
    - Return immediately (no I/O operations).
    - Never fail unless the process is completely hung.
    - Be called frequently by orchestrators.
    """
    return HealthResponse(
        status="healthy",
        timestamp=time.time(),
        version=_get_version(),
        components={},
    )


# =============================================================================
# Readiness Probe
# =============================================================================

@router.get(
    "/ready",
    response_model=HealthResponse,
    summary="Readiness Probe",
    description="""
    Check if the service is ready to accept traffic.
    
    **Use Case**: Kubernetes readiness probe to manage traffic routing.
    
    **Checks**:
    - Database connectivity
    - Required services initialized
    
    **Returns**: 200 if ready, 503 if not ready.
    """,
)
async def readiness() -> JSONResponse:
    """
    Readiness check - can we serve traffic?
    
    Checks critical dependencies:
    - Database connection
    - Essential services
    
    Returns 503 if any critical dependency is unhealthy.
    """
    components = await get_service_health()
    
    # Determine overall status
    critical_services = ["database"]
    all_healthy = all(
        components.get(svc) == "healthy"
        for svc in critical_services
    )
    
    status_str = "healthy" if all_healthy else "unhealthy"
    status_code = status.HTTP_200_OK if all_healthy else status.HTTP_503_SERVICE_UNAVAILABLE
    
    response = HealthResponse(
        status=status_str,
        timestamp=time.time(),
        version=_get_version(),
        components=components,
    )
    
    return JSONResponse(
        status_code=status_code,
        content=response.model_dump(),
    )


# =============================================================================
# Full Health Check
# =============================================================================

@router.get(
    "",
    response_model=HealthResponse,
    summary="Health Check",
    description="""
    Comprehensive health check with component status.
    
    **Use Case**: Monitoring dashboards and alerting systems.
    
    **Checks all components**:
    - Database
    - LLM service
    - Vector search
    - Chat history engine
    
    **Returns**: Detailed status of all components.
    """,
)
async def health_check() -> HealthResponse:
    """
    Full health check with all component statuses.
    
    Unlike readiness, this always returns 200 but indicates
    degraded components in the response body.
    """
    components = await get_service_health()
    
    # Determine overall status
    unhealthy_count = sum(1 for v in components.values() if v == "unhealthy")
    unavailable_count = sum(1 for v in components.values() if v == "unavailable")
    
    if unhealthy_count > 0:
        status_str = "unhealthy"
    elif unavailable_count > 0:
        status_str = "degraded"
    else:
        status_str = "healthy"
    
    return HealthResponse(
        status=status_str,
        timestamp=time.time(),
        version=_get_version(),
        components=components,
    )


# =============================================================================
# Metrics Endpoint
# =============================================================================

@router.get(
    "/metrics",
    response_model=MetricsResponse,
    summary="Service Metrics",
    description="""
    Prometheus-style metrics for monitoring.
    
    **Metrics included**:
    - Uptime
    - Total requests
    - Active requests
    - Average latency
    - Error rate
    - Database connection stats
    """,
    tags=["Metrics"],
)
async def metrics() -> MetricsResponse:
    """
    Get service metrics for monitoring.
    
    Returns simplified metrics suitable for Prometheus scraping
    or custom monitoring dashboards.
    """
    uptime = time.time() - _startup_time
    avg_latency = _total_latency_ms / max(1, _request_count)
    error_rate = _error_count / max(1, _request_count)
    
    # Get DB connection info
    db_active = 0
    db_pool = 0
    try:
        from api.dependencies import get_db_manager
        db_manager = get_db_manager()
        db_pool = len(db_manager._connections)
    except Exception:
        pass
    
    return MetricsResponse(
        uptime_seconds=uptime,
        requests_total=_request_count,
        requests_active=0,  # Would need middleware tracking
        avg_latency_ms=avg_latency,
        error_rate=error_rate,
        db_connections_active=db_active,
        db_connections_pool=db_pool,
    )


# =============================================================================
# System Info (Debug Only)
# =============================================================================

@router.get(
    "/info",
    summary="System Information",
    description="Runtime information (debug builds only).",
    include_in_schema=False,  # Hide from OpenAPI docs
)
async def system_info() -> Dict:
    """
    System information for debugging.
    
    Hidden from OpenAPI docs to avoid exposing in production.
    """
    try:
        from api.config import get_settings
        settings = get_settings()
        if not settings.debug:
            return {"message": "Debug mode disabled"}
    except Exception:
        pass
    
    import sys
    
    return {
        "python_version": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "hostname": platform.node(),
        "pid": os.getpid(),
        "cwd": os.getcwd(),
        "uptime_seconds": time.time() - _startup_time,
    }


# =============================================================================
# Helper Functions
# =============================================================================

def _get_version() -> str:
    """Get application version."""
    try:
        from api import __version__
        return __version__
    except Exception:
        return "1.0.0"
