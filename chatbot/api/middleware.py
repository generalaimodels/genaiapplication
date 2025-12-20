# -*- coding: utf-8 -*-
# =================================================================================================
# api/middleware.py — Production Middleware Stack
# =================================================================================================
# High-performance middleware implementing:
#
#   1. REQUEST LOGGING: Structured logs with timing, status, and request ID.
#   2. TIMING HEADERS: X-Response-Time header for client-side metrics.
#   3. RATE LIMITING: Token bucket algorithm with per-IP tracking.
#   4. REQUEST ID: Unique ID generation for request tracing.
#   5. ERROR RECOVERY: Graceful handling of middleware failures.
#
# Middleware Execution Order:
# ---------------------------
#   1. RequestIDMiddleware (adds request ID for tracing)
#   2. TimingMiddleware (measures response time)
#   3. RequestLoggingMiddleware (logs request/response)
#   4. RateLimitMiddleware (enforces rate limits)
#
# Performance Notes:
# ------------------
#   - Middleware runs on every request; keep it minimal.
#   - Heavy operations (DB, external calls) NEVER in middleware.
#   - Use async where possible to avoid blocking event loop.
#
# =================================================================================================

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable, DefaultDict, Dict, Optional, Set

from fastapi import FastAPI, Request, Response, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.types import ASGIApp

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
_LOG = logging.getLogger("api.middleware")


# =============================================================================
# Request ID Middleware
# =============================================================================

class RequestIDMiddleware(BaseHTTPMiddleware):
    """
    Adds unique request ID to each request for distributed tracing.
    
    Request ID Source:
    ------------------
    1. X-Request-ID header from client (for tracing across services).
    2. Auto-generated UUID v4 if not provided.
    
    The request ID is:
    - Stored in request.state.request_id for handlers.
    - Added to response as X-Request-ID header.
    - Included in all log messages for correlation.
    """
    
    async def dispatch(
        self,
        request: Request,
        call_next: RequestResponseEndpoint,
    ) -> Response:
        # Use client-provided ID or generate new one
        request_id = request.headers.get("X-Request-ID")
        if not request_id:
            request_id = str(uuid.uuid4())
        
        # Store in request state for access in handlers
        request.state.request_id = request_id
        
        # Process request
        response = await call_next(request)
        
        # Add to response headers
        response.headers["X-Request-ID"] = request_id
        
        return response


# =============================================================================
# Timing Middleware
# =============================================================================

class TimingMiddleware(BaseHTTPMiddleware):
    """
    Measures and reports request processing time.
    
    Timing Details:
    ---------------
    - Start time: When middleware receives request.
    - End time: When response is ready to send.
    - Overhead: ~0.01ms per request (negligible).
    
    Headers Added:
    --------------
    - X-Response-Time: Processing time in milliseconds.
    - X-Request-Start: Unix timestamp when request started.
    """
    
    async def dispatch(
        self,
        request: Request,
        call_next: RequestResponseEndpoint,
    ) -> Response:
        start_time = time.perf_counter()
        start_ts = time.time()
        
        # Store start time for use in handlers
        request.state.start_time = start_time
        request.state.start_ts = start_ts
        
        # Process request
        response = await call_next(request)
        
        # Calculate duration
        duration_ms = (time.perf_counter() - start_time) * 1000
        
        # Add timing headers
        response.headers["X-Response-Time"] = f"{duration_ms:.2f}ms"
        response.headers["X-Request-Start"] = str(start_ts)
        
        return response


# =============================================================================
# Request Logging Middleware
# =============================================================================

class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """
    Structured request/response logging for observability.
    
    Log Format:
    -----------
    - method: HTTP method (GET, POST, etc.)
    - path: Request path
    - status: Response status code
    - duration_ms: Processing time
    - request_id: Unique request identifier
    - client_ip: Client IP address
    - user_agent: Client user agent (truncated)
    
    Log Levels:
    -----------
    - INFO: Successful requests (2xx)
    - WARNING: Client errors (4xx)
    - ERROR: Server errors (5xx)
    """
    
    # Paths to skip logging (health checks, metrics)
    SKIP_PATHS: Set[str] = {
        "/health",
        "/health/live",
        "/health/ready",
        "/metrics",
        "/favicon.ico",
    }
    
    async def dispatch(
        self,
        request: Request,
        call_next: RequestResponseEndpoint,
    ) -> Response:
        # Skip logging for health checks
        if request.url.path in self.SKIP_PATHS:
            return await call_next(request)
        
        start_time = time.perf_counter()
        
        # Extract request details
        method = request.method
        path = request.url.path
        query = str(request.query_params) if request.query_params else ""
        client_ip = self._get_client_ip(request)
        user_agent = request.headers.get("User-Agent", "")[:100]  # Truncate
        request_id = getattr(request.state, "request_id", "unknown")
        
        # Process request
        try:
            response = await call_next(request)
            status_code = response.status_code
            error_msg = None
        except Exception as exc:
            # Log exception and re-raise (let exception handlers deal with it)
            status_code = 500
            error_msg = str(exc)
            raise
        finally:
            duration_ms = (time.perf_counter() - start_time) * 1000
            
            # Prepare log message
            log_data = {
                "method": method,
                "path": path,
                "query": query,
                "status": status_code,
                "duration_ms": round(duration_ms, 2),
                "request_id": request_id,
                "client_ip": client_ip,
            }
            
            # Choose log level based on status
            if status_code >= 500:
                _LOG.error(
                    "%s %s -> %d (%.2fms) [%s]",
                    method, path, status_code, duration_ms, request_id,
                    extra=log_data,
                )
            elif status_code >= 400:
                _LOG.warning(
                    "%s %s -> %d (%.2fms) [%s]",
                    method, path, status_code, duration_ms, request_id,
                    extra=log_data,
                )
            else:
                _LOG.info(
                    "%s %s -> %d (%.2fms) [%s]",
                    method, path, status_code, duration_ms, request_id,
                    extra=log_data,
                )
        
        return response
    
    @staticmethod
    def _get_client_ip(request: Request) -> str:
        """
        Extract client IP, handling proxy headers.
        
        Priority:
        ---------
        1. X-Forwarded-For (first IP in chain)
        2. X-Real-IP
        3. Client host from connection
        """
        # X-Forwarded-For can contain multiple IPs
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            return forwarded.split(",")[0].strip()
        
        # X-Real-IP is a single IP
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip.strip()
        
        # Fall back to connection client
        if request.client:
            return request.client.host
        
        return "unknown"


# =============================================================================
# Rate Limiting Middleware
# =============================================================================

@dataclass
class TokenBucket:
    """
    Token bucket rate limiter state for a single client.
    
    Algorithm:
    ----------
    - Bucket holds up to `burst` tokens.
    - Tokens refill at `rate` per second.
    - Each request consumes 1 token.
    - Request blocked if tokens < 1.
    
    Benefits:
    ---------
    - Allows short bursts above sustained rate.
    - Smooth rate limiting (no hard cutoffs).
    - Memory efficient (single float per client).
    """
    tokens: float
    last_update: float
    rate: float  # tokens per second
    burst: int   # max tokens (bucket size)
    
    def consume(self, now: float) -> bool:
        """
        Try to consume a token.
        
        Returns:
        --------
        True if token consumed, False if rate limited.
        """
        # Refill tokens based on time elapsed
        elapsed = now - self.last_update
        self.tokens = min(self.burst, self.tokens + elapsed * self.rate)
        self.last_update = now
        
        # Try to consume
        if self.tokens >= 1.0:
            self.tokens -= 1.0
            return True
        return False
    
    def retry_after(self, now: float) -> int:
        """Calculate seconds until a token is available."""
        if self.tokens >= 1.0:
            return 0
        needed = 1.0 - self.tokens
        return int(needed / self.rate) + 1


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Token bucket rate limiting per client IP.
    
    Configuration:
    --------------
    - requests_per_minute: Sustained request rate.
    - burst: Max requests in short burst.
    - enabled: Toggle rate limiting on/off.
    
    Response on Limit:
    ------------------
    - Status: 429 Too Many Requests
    - Header: Retry-After (seconds until reset)
    - Body: JSON error response
    
    Exclusions:
    -----------
    - Health check endpoints (always allowed)
    - Whitelisted IPs (if configured)
    """
    
    # Paths excluded from rate limiting
    EXCLUDED_PATHS: Set[str] = {
        "/health",
        "/health/live",
        "/health/ready",
        "/docs",
        "/redoc",
        "/openapi.json",
    }
    
    def __init__(
        self,
        app: ASGIApp,
        requests_per_minute: int = 100,
        burst: int = 20,
        enabled: bool = True,
    ) -> None:
        super().__init__(app)
        self.rate = requests_per_minute / 60.0  # Convert to per-second
        self.burst = burst
        self.enabled = enabled
        self.buckets: DefaultDict[str, TokenBucket] = defaultdict(
            lambda: TokenBucket(
                tokens=float(self.burst),
                last_update=time.time(),
                rate=self.rate,
                burst=self.burst,
            )
        )
        self._cleanup_task: Optional[asyncio.Task] = None
        self._lock = asyncio.Lock()
    
    async def dispatch(
        self,
        request: Request,
        call_next: RequestResponseEndpoint,
    ) -> Response:
        # Skip if disabled or excluded path
        if not self.enabled:
            return await call_next(request)
        
        if request.url.path in self.EXCLUDED_PATHS:
            return await call_next(request)
        
        # Get client identifier
        client_id = self._get_client_id(request)
        now = time.time()
        
        # Check rate limit
        async with self._lock:
            bucket = self.buckets[client_id]
            if not bucket.consume(now):
                # Rate limited
                retry_after = bucket.retry_after(now)
                request_id = getattr(request.state, "request_id", "unknown")
                
                _LOG.warning(
                    "Rate limit exceeded for %s [%s]",
                    client_id,
                    request_id,
                )
                
                return JSONResponse(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    content={
                        "error": "RATE_LIMIT_EXCEEDED",
                        "message": "Too many requests. Please slow down.",
                        "retry_after": retry_after,
                        "request_id": request_id,
                        "timestamp": now,
                    },
                    headers={"Retry-After": str(retry_after)},
                )
        
        # Schedule cleanup if not running
        if self._cleanup_task is None or self._cleanup_task.done():
            self._cleanup_task = asyncio.create_task(self._cleanup_old_buckets())
        
        return await call_next(request)
    
    def _get_client_id(self, request: Request) -> str:
        """Get client identifier for rate limiting."""
        # Use IP + optional API key for more granular limiting
        ip = RequestLoggingMiddleware._get_client_ip(request)
        api_key = request.headers.get("X-API-Key", "")[:8]  # First 8 chars only
        if api_key:
            return f"{ip}:{api_key}"
        return ip
    
    async def _cleanup_old_buckets(self) -> None:
        """Remove stale buckets to prevent memory leak."""
        await asyncio.sleep(60)  # Run every minute
        
        now = time.time()
        stale_threshold = 300  # 5 minutes of inactivity
        
        async with self._lock:
            stale_keys = [
                key for key, bucket in self.buckets.items()
                if now - bucket.last_update > stale_threshold
            ]
            for key in stale_keys:
                del self.buckets[key]
            
            if stale_keys:
                _LOG.debug("Cleaned up %d stale rate limit buckets", len(stale_keys))


# =============================================================================
# CORS Middleware Configuration
# =============================================================================

def configure_cors(
    app: FastAPI,
    origins: list[str] = ["*"],
    allow_credentials: bool = True,
    allow_methods: list[str] = ["*"],
    allow_headers: list[str] = ["*"],
) -> None:
    """
    Configure CORS middleware for the FastAPI app.
    
    Security Notes:
    ---------------
    - Production: Use specific origins, not "*".
    - Credentials: Only enable if using cookies/auth headers.
    - Methods: Restrict to actually used methods.
    
    Parameters:
    -----------
    origins : Allowed origin domains.
    allow_credentials : Allow cookies/auth headers.
    allow_methods : Allowed HTTP methods.
    allow_headers : Allowed request headers.
    """
    from fastapi.middleware.cors import CORSMiddleware
    
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=allow_credentials,
        allow_methods=allow_methods,
        allow_headers=allow_headers,
        expose_headers=["X-Request-ID", "X-Response-Time"],
    )
    
    _LOG.info("CORS configured: origins=%s", origins)


# =============================================================================
# Middleware Registration
# =============================================================================

def register_middleware(app: FastAPI) -> None:
    """
    Register all middleware in correct order.
    
    Order Matters:
    --------------
    Middleware is executed in reverse order of registration.
    Last registered = first to process request.
    
    Execution Order:
    ----------------
    1. RequestIDMiddleware (first in, adds ID)
    2. TimingMiddleware (measures total time)
    3. RequestLoggingMiddleware (logs with timing)
    4. RateLimitMiddleware (may reject early)
    5. CORSMiddleware (handled by FastAPI)
    """
    # Import settings
    try:
        from api.config import get_settings
        settings = get_settings()
        rate_limit_enabled = settings.rate_limit_enabled
        rate_limit_rpm = settings.rate_limit_requests_per_minute
        rate_limit_burst = settings.rate_limit_burst
        cors_origins = settings.cors_origins
    except Exception:
        # Use defaults if config not available
        rate_limit_enabled = True
        rate_limit_rpm = 100
        rate_limit_burst = 20
        cors_origins = ["*"]
    
    # Register in reverse execution order
    # 4. Rate limiting (last registered = processed first after CORS)
    app.add_middleware(
        RateLimitMiddleware,
        requests_per_minute=rate_limit_rpm,
        burst=rate_limit_burst,
        enabled=rate_limit_enabled,
    )
    
    # 3. Request logging
    app.add_middleware(RequestLoggingMiddleware)
    
    # 2. Timing
    app.add_middleware(TimingMiddleware)
    
    # 1. Request ID (first registered = last processed inbound, first outbound)
    app.add_middleware(RequestIDMiddleware)
    
    # Configure CORS (uses FastAPI's built-in CORS middleware)
    configure_cors(app, origins=cors_origins)
    
    _LOG.info(
        "Middleware registered: rate_limit=%s (rpm=%d, burst=%d)",
        rate_limit_enabled,
        rate_limit_rpm,
        rate_limit_burst,
    )
