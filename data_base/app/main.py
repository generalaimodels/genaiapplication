# ==============================================================================
# MAIN APPLICATION - FastAPI Entry Point
# ==============================================================================
# Application factory with lifespan events, middleware, and routing
# ==============================================================================

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import logging

from app.core.settings import settings
from app.core.exceptions import AppException
from app.database.factory import DatabaseFactory
from app.api.router import api_router
from app.schemas.base import HealthResponse

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ==============================================================================
# LIFESPAN MANAGEMENT
# ==============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """
    Application lifespan manager.
    
    Handles startup and shutdown events:
    - Startup: Initialize database connection
    - Shutdown: Close database connections
    """
    # Startup
    logger.info(f"Starting {settings.APP_NAME} v{settings.APP_VERSION}")
    logger.info(f"Environment: {settings.ENVIRONMENT}")
    logger.info(f"Database: {settings.DATABASE_TYPE}")
    
    try:
        await DatabaseFactory.initialize()
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        # Continue anyway for development
        if settings.ENVIRONMENT == "production":
            raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down application...")
    await DatabaseFactory.shutdown()
    logger.info("Application shutdown complete")


# ==============================================================================
# APPLICATION FACTORY
# ==============================================================================

def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application.
    
    Returns:
        Configured FastAPI application instance
    """
    app = FastAPI(
        title=settings.API_TITLE,
        description=settings.API_DESCRIPTION,
        version=settings.APP_VERSION,
        debug=settings.DEBUG,
        lifespan=lifespan,
        docs_url="/docs" if settings.DEBUG else None,
        redoc_url="/redoc" if settings.DEBUG else None,
        openapi_url="/openapi.json" if settings.DEBUG else None,
    )
    
    # Configure CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=settings.CORS_ALLOW_CREDENTIALS,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Register exception handlers
    register_exception_handlers(app)
    
    # Include routers
    app.include_router(api_router)
    
    # Register health endpoints
    register_health_endpoints(app)
    
    return app


# ==============================================================================
# EXCEPTION HANDLERS
# ==============================================================================

def register_exception_handlers(app: FastAPI) -> None:
    """Register global exception handlers."""
    
    @app.exception_handler(AppException)
    async def app_exception_handler(
        request: Request,
        exc: AppException,
    ) -> JSONResponse:
        """Handle application exceptions."""
        return JSONResponse(
            status_code=exc.status_code,
            content=exc.to_dict(),
        )
    
    @app.exception_handler(Exception)
    async def generic_exception_handler(
        request: Request,
        exc: Exception,
    ) -> JSONResponse:
        """Handle unexpected exceptions."""
        logger.exception(f"Unexpected error: {exc}")
        
        if settings.DEBUG:
            detail = str(exc)
        else:
            detail = "An unexpected error occurred"
        
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "success": False,
                "error": {
                    "code": "INTERNAL_ERROR",
                    "message": detail,
                }
            },
        )


# ==============================================================================
# HEALTH ENDPOINTS
# ==============================================================================

def register_health_endpoints(app: FastAPI) -> None:
    """Register health check endpoints."""
    
    @app.get(
        "/health",
        response_model=HealthResponse,
        tags=["Health"],
        summary="Health check",
        description="Check application and database health.",
    )
    async def health_check() -> HealthResponse:
        """Application health check."""
        db_healthy = await DatabaseFactory.health_check()
        
        return HealthResponse(
            status="healthy" if db_healthy else "degraded",
            version=settings.APP_VERSION,
            database="connected" if db_healthy else "disconnected",
        )
    
    @app.get(
        "/",
        tags=["Health"],
        summary="Root endpoint",
        description="Welcome message and API information.",
    )
    async def root() -> dict:
        """Root endpoint with API info."""
        return {
            "name": settings.APP_NAME,
            "version": settings.APP_VERSION,
            "docs": "/docs" if settings.DEBUG else "Disabled in production",
            "health": "/health",
        }


# Create application instance
app = create_app()


# ==============================================================================
# DEVELOPMENT RUNNER
# ==============================================================================

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.DEBUG,
        log_level=settings.LOG_LEVEL.lower(),
    )
