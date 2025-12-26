# ==============================================================================
# REQUEST LOGGER MIDDLEWARE
# ==============================================================================
# Structured request/response logging
# ==============================================================================

from __future__ import annotations

import time
import uuid
from typing import Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
import logging

from app.core.settings import settings

logger = logging.getLogger(__name__)


class RequestLoggerMiddleware(BaseHTTPMiddleware):
    """
    Middleware for structured request/response logging.
    
    Logs request details, response status, and timing information.
    Adds request ID header for tracing.
    """
    
    async def dispatch(
        self,
        request: Request,
        call_next: Callable,
    ) -> Response:
        """Process request with logging."""
        # Generate request ID
        request_id = str(uuid.uuid4())[:8]
        
        # Start timing
        start_time = time.time()
        
        # Log request
        logger.info(
            f"[{request_id}] {request.method} {request.url.path} - Started"
        )
        
        # Process request
        try:
            response = await call_next(request)
            
            # Calculate duration
            duration_ms = (time.time() - start_time) * 1000
            
            # Log response
            log_level = logging.INFO if response.status_code < 400 else logging.WARNING
            logger.log(
                log_level,
                f"[{request_id}] {request.method} {request.url.path} "
                f"- {response.status_code} ({duration_ms:.2f}ms)"
            )
            
            # Add headers
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Response-Time"] = f"{duration_ms:.2f}ms"
            
            return response
            
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            logger.error(
                f"[{request_id}] {request.method} {request.url.path} "
                f"- Error ({duration_ms:.2f}ms): {str(e)}"
            )
            raise
