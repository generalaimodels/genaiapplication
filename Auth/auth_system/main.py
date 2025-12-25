# =============================================================================
# SOTA AUTHENTICATION SYSTEM - MAIN APPLICATION
# =============================================================================
# File: main.py
# Description: FastAPI application entry point with lifecycle management
# =============================================================================

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from api.v1 import api_router
from api.v1.health_routes import router as health_router
from api.middleware import (
    RateLimiterMiddleware,
    SecurityHeadersMiddleware,
    RequestIDMiddleware,
    AuthenticationMiddleware,
    LoggingMiddleware,
)
from db.factory import DBFactory
from core.config import settings
from core.exceptions import AuthSystemException


# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)


# =============================================================================
# APPLICATION LIFECYCLE
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifecycle manager.
    
    Handles startup and shutdown events:
    - Startup: Initialize database connections, create tables
    - Shutdown: Close all connections gracefully
    """
    # STARTUP
    logger.info(f"Starting {settings.app_name} in {settings.app_env} mode")
    
    try:
        # Connect to databases
        await DBFactory.connect_all()
        logger.info("Database connections established")
        
        # Create tables (in development only)
        if settings.is_development:
            await DBFactory.create_tables()
            logger.info("Database tables created/verified")
        
        logger.info(f"{settings.app_name} started successfully")
        
    except Exception as e:
        logger.error(f"Startup failed: {e}")
        raise
    
    yield  # Application runs here
    
    # SHUTDOWN
    logger.info(f"Shutting down {settings.app_name}")
    
    try:
        await DBFactory.disconnect_all()
        logger.info("Database connections closed")
    except Exception as e:
        logger.error(f"Shutdown error: {e}")
    
    logger.info(f"{settings.app_name} shutdown complete")


# =============================================================================
# APPLICATION FACTORY
# =============================================================================

def create_application() -> FastAPI:
    """
    Create and configure FastAPI application.
    
    Returns:
        FastAPI: Configured application instance
    """
    app = FastAPI(
        title=settings.app_name,
        description="SOTA Authentication System with JWT, Sessions, and Multi-DB Support",
        version="1.0.0",
        docs_url="/docs" if settings.is_development else None,
        redoc_url="/redoc" if settings.is_development else None,
        openapi_url="/openapi.json" if settings.is_development else None,
        lifespan=lifespan,
    )
    
    # =========================================================================
    # MIDDLEWARE STACK (order matters - first added = outermost)
    # =========================================================================
    
    # Request ID (outermost - adds tracking ID)
    app.add_middleware(RequestIDMiddleware)
    
    # Logging (captures request/response info)
    app.add_middleware(LoggingMiddleware)
    
    # Security Headers
    app.add_middleware(SecurityHeadersMiddleware)
    
    # Rate Limiting
    app.add_middleware(RateLimiterMiddleware)
    
    # Authentication (validates JWT tokens)
    app.add_middleware(AuthenticationMiddleware)
    
    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["X-Request-ID", "X-RateLimit-Limit", "X-RateLimit-Remaining"],
    )
    
    # =========================================================================
    # EXCEPTION HANDLERS
    # =========================================================================
    
    @app.exception_handler(AuthSystemException)
    async def auth_exception_handler(
        request: Request,
        exc: AuthSystemException,
    ) -> JSONResponse:
        """Handle custom authentication exceptions."""
        return JSONResponse(
            status_code=exc.status_code,
            content=exc.to_dict(),
        )
    
    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(
        request: Request,
        exc: StarletteHTTPException,
    ) -> JSONResponse:
        """Handle standard HTTP exceptions."""
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": True,
                "error_code": f"HTTP_{exc.status_code}",
                "message": exc.detail,
                "details": {},
            },
        )
    
    @app.exception_handler(Exception)
    async def general_exception_handler(
        request: Request,
        exc: Exception,
    ) -> JSONResponse:
        """Handle unexpected exceptions."""
        logger.exception("Unhandled exception")
        
        # Don't expose internal errors in production
        if settings.is_production:
            message = "An internal error occurred"
        else:
            message = str(exc)
        
        return JSONResponse(
            status_code=500,
            content={
                "error": True,
                "error_code": "INTERNAL_ERROR",
                "message": message,
                "details": {},
            },
        )
    
    # =========================================================================
    # ROUTES
    # =========================================================================
    
    # API v1 routes
    app.include_router(api_router)
    
    # Health routes at root level
    app.include_router(health_router)
    
    # Root endpoint
    @app.get("/", include_in_schema=False)
    async def root():
        """Root endpoint with API info."""
        return {
            "name": settings.app_name,
            "version": "1.0.0",
            "status": "running",
            "docs": "/docs" if settings.is_development else None,
        }
    
    return app


# =============================================================================
# APPLICATION INSTANCE
# =============================================================================

app = create_application()


# =============================================================================
# ENTRYPOINT
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.is_development,
        log_level=settings.log_level.lower(),
    )
