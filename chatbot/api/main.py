# -*- coding: utf-8 -*-
# =================================================================================================
# api/main.py — FastAPI Application Entry Point
# =================================================================================================
# Production-grade FastAPI application implementing:
#
#   1. LIFESPAN MANAGEMENT: Proper startup/shutdown with resource cleanup.
#   2. MIDDLEWARE STACK: Request ID, timing, logging, rate limiting, CORS.
#   3. EXCEPTION HANDLERS: Consistent error responses across all endpoints.
#   4. API VERSIONING: Versioned prefix (/api/v1) for backward compatibility.
#   5. DOCUMENTATION: OpenAPI/Swagger with rich descriptions.
#
# Running the Server:
# -------------------
#   Development:
#     python -m uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
#
#   Production:
#     python -m uvicorn api.main:app --workers 4 --host 0.0.0.0 --port 8000
#
#   Or directly:
#     python api/main.py
#
# API Documentation:
# ------------------
#   Swagger UI: http://localhost:8000/docs
#   ReDoc:      http://localhost:8000/redoc
#   OpenAPI:    http://localhost:8000/openapi.json
#
# =================================================================================================

from __future__ import annotations

import logging
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

# -----------------------------------------------------------------------------
# Add parent directory to path for imports (when running as script)
# -----------------------------------------------------------------------------
_PARENT_DIR = Path(__file__).parent.parent
if str(_PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(_PARENT_DIR))

# -----------------------------------------------------------------------------
# Logging Configuration
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
_LOG = logging.getLogger("api.main")


# =============================================================================
# Application Lifespan (Startup/Shutdown)
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Application lifespan context manager.
    
    Startup:
    --------
    - Initialize database connections.
    - Create data directories.
    - Pre-warm critical services (optional).
    
    Shutdown:
    ---------
    - Close database connections.
    - Cancel background tasks.
    - Release resources.
    """
    startup_time = time.time()
    _LOG.info("=" * 60)
    _LOG.info("Starting CCA Chatbot API...")
    _LOG.info("=" * 60)
    
    # ----- STARTUP -----
    try:
        from api.config import get_settings
        settings = get_settings()
        
        # Ensure data directories exist
        settings.data_dir.mkdir(parents=True, exist_ok=True)
        (settings.data_dir / "uploads").mkdir(exist_ok=True)
        
        _LOG.info("Data directory: %s", settings.data_dir)
        _LOG.info("Database path: %s", settings.db_path)
        _LOG.info("API prefix: %s", settings.api_prefix)
        _LOG.info("Debug mode: %s", settings.debug)
        
        # Initialize database
        from api.dependencies import get_db_manager
        db_manager = get_db_manager()
        _LOG.info("Database initialized")
        
        # Log startup time
        _LOG.info("Startup complete in %.2f seconds", time.time() - startup_time)
        _LOG.info("=" * 60)
        
    except Exception as e:
        _LOG.error("Startup failed: %s", e, exc_info=True)
        raise
    
    # ----- YIELD (Application Running) -----
    yield
    
    # ----- SHUTDOWN -----
    _LOG.info("=" * 60)
    _LOG.info("Shutting down CCA Chatbot API...")
    
    try:
        from api.dependencies import cleanup_services
        await cleanup_services()
        _LOG.info("Cleanup complete")
    except Exception as e:
        _LOG.warning("Cleanup error: %s", e)
    
    _LOG.info("Goodbye!")
    _LOG.info("=" * 60)


# =============================================================================
# FastAPI Application Factory
# =============================================================================

def create_app() -> FastAPI:
    """
    Create and configure FastAPI application.
    
    Factory Pattern:
    ----------------
    Enables creating multiple app instances (useful for testing).
    
    Configuration:
    --------------
    All settings loaded from environment/config file via get_settings().
    """
    # Load settings
    try:
        from api.config import get_settings
        settings = get_settings()
        debug = settings.debug
        api_prefix = settings.api_prefix
    except Exception as e:
        _LOG.warning("Failed to load settings: %s (using defaults)", e)
        debug = False
        api_prefix = "/api/v1"
    
    # Create app
    app = FastAPI(
        title="CCA Chatbot API",
        description="""
## Advanced Generalized Backend API for AI Chatbot

Production-grade API integrating:
- **Document Processing**: PDF/text conversion and chunking
- **Chat History**: Conversation management with branching
- **Vector Search**: Semantic search and retrieval
- **LLM Generation**: Chat completion with context

### Features
- Session management with conversation history
- RAG (Retrieval-Augmented Generation) context building
- Streaming responses via Server-Sent Events
- Batch processing for bulk operations
- Comprehensive health checks and metrics

### Architecture
- FastAPI with async/await for high concurrency
- SQLite with WAL mode for ACID compliance
- Token bucket rate limiting
- Structured logging with request tracing
        """,
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        debug=debug,
        lifespan=lifespan,
    )
    
    # ----- Register Exception Handlers -----
    from api.exceptions import register_exception_handlers
    register_exception_handlers(app)
    
    # ----- Register Middleware -----
    from api.middleware import register_middleware
    register_middleware(app)
    
    # ----- Include Routers -----
    from api.routers import health, conversations, documents, search, generation
    
    # Health endpoints (no prefix for Kubernetes probes)
    app.include_router(health.router, prefix=api_prefix)
    
    # Main API routers
    app.include_router(conversations.router, prefix=api_prefix)
    app.include_router(documents.router, prefix=api_prefix)
    app.include_router(search.router, prefix=api_prefix)
    app.include_router(generation.router, prefix=api_prefix)
    
    # ----- Root Endpoint -----
    @app.get("/", tags=["Root"])
    async def root() -> dict:
        """API root - returns basic info."""
        return {
            "name": "CCA Chatbot API",
            "version": "1.0.0",
            "docs": "/docs",
            "health": f"{api_prefix}/health",
        }
    
    # ----- Catch-All for Unmatched Routes -----
    @app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"], include_in_schema=False)
    async def catch_all(path: str, request: Request) -> JSONResponse:
        """Handle unmatched routes with 404."""
        return JSONResponse(
            status_code=404,
            content={
                "error": "NOT_FOUND",
                "message": f"Path '/{path}' not found",
                "hint": "Check /docs for available endpoints",
            },
        )
    
    _LOG.info("FastAPI application created")
    return app


# =============================================================================
# Application Instance
# =============================================================================

app = create_app()


# =============================================================================
# Direct Execution
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    
    # Load settings for uvicorn configuration
    try:
        from api.config import get_settings
        settings = get_settings()
        host = settings.uvicorn_host
        port = settings.uvicorn_port
        reload = settings.uvicorn_reload or settings.debug
        workers = settings.uvicorn_workers
        log_level = settings.log_level.lower()
    except Exception:
        host = "0.0.0.0"
        port = 8000
        reload = True
        workers = 1
        log_level = "info"
    
    _LOG.info("Starting Uvicorn server...")
    _LOG.info("  Host: %s", host)
    _LOG.info("  Port: %d", port)
    _LOG.info("  Reload: %s", reload)
    _LOG.info("  Workers: %d", workers)
    
    uvicorn.run(
        "api.main:app",
        host=host,
        port=port,
        reload=reload,
        workers=workers if not reload else 1,  # Workers not compatible with reload
        log_level=log_level,
        access_log=True,
    )
