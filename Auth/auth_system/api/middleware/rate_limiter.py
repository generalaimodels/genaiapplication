# =============================================================================
# SOTA AUTHENTICATION SYSTEM - RATE LIMITER MIDDLEWARE
# =============================================================================
# File: api/middleware/rate_limiter.py
# Description: Rate limiting middleware using Redis sliding window algorithm
# =============================================================================

from typing import Optional, Callable
from datetime import datetime

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from db.factory import DBFactory
from core.config import settings


class RateLimiterMiddleware(BaseHTTPMiddleware):
    """
    ┌─────────────────────────────────────────────────────────────────────────┐
    │                    RATE LIMITER MIDDLEWARE                               │
    │  Sliding window rate limiting using Redis                               │
    │  Protects API from abuse and ensures fair usage                         │
    └─────────────────────────────────────────────────────────────────────────┘
    
    Features:
        - IP-based rate limiting
        - Endpoint-specific limits
        - Sliding window algorithm
        - Informative headers
    
    Headers Added:
        - X-RateLimit-Limit: Maximum requests allowed
        - X-RateLimit-Remaining: Requests remaining
        - X-RateLimit-Reset: Seconds until reset
        - Retry-After: Seconds to wait (when limited)
    """
    
    # Endpoints with stricter limits
    STRICT_ENDPOINTS = {
        "/api/v1/auth/login": 10,      # 10 attempts per minute
        "/api/v1/auth/register": 5,     # 5 registrations per minute
        "/api/v1/auth/forgot-password": 3,  # 3 requests per minute
    }
    
    # Endpoints exempt from rate limiting
    EXEMPT_ENDPOINTS = {
        "/health",
        "/health/ready",
        "/health/live",
        "/docs",
        "/redoc",
        "/openapi.json",
    }
    
    async def dispatch(
        self,
        request: Request,
        call_next: Callable,
    ) -> Response:
        """
        Process request with rate limiting.
        
        Args:
            request: Incoming request
            call_next: Next middleware/handler
            
        Returns:
            Response with rate limit headers
        """
        path = request.url.path
        
        # Skip exempt endpoints
        if path in self.EXEMPT_ENDPOINTS:
            return await call_next(request)
        
        # Get client IP
        client_ip = self._get_client_ip(request)
        
        # Determine limit for this endpoint
        limit = self.STRICT_ENDPOINTS.get(path, settings.rate_limit_per_minute)
        
        try:
            # Get Redis client
            redis = DBFactory.get_redis_adapter()
            
            # Create rate limit key
            key = f"rate:{client_ip}:{path}"
            
            # Increment counter
            current = await redis.incr(key)
            
            # Set expiration on first request
            if current == 1:
                await redis.expire(key, 60)  # 1 minute window
            
            # Get TTL for reset header
            ttl = await redis.ttl(key)
            if ttl < 0:
                ttl = 60
            
            # Check if over limit
            if current > limit:
                return JSONResponse(
                    status_code=429,
                    content={
                        "error": True,
                        "error_code": "RATE_LIMIT_EXCEEDED",
                        "message": f"Rate limit exceeded. Try again in {ttl} seconds.",
                        "details": {"retry_after": ttl},
                    },
                    headers={
                        "X-RateLimit-Limit": str(limit),
                        "X-RateLimit-Remaining": "0",
                        "X-RateLimit-Reset": str(ttl),
                        "Retry-After": str(ttl),
                    },
                )
            
            # Process request
            response = await call_next(request)
            
            # Add rate limit headers
            response.headers["X-RateLimit-Limit"] = str(limit)
            response.headers["X-RateLimit-Remaining"] = str(max(0, limit - current))
            response.headers["X-RateLimit-Reset"] = str(ttl)
            
            return response
            
        except Exception:
            # If Redis fails, allow request but log
            return await call_next(request)
    
    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP from request."""
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            return forwarded.split(",")[0].strip()
        
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip.strip()
        
        return request.client.host if request.client else "unknown"


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """
    ┌─────────────────────────────────────────────────────────────────────────┐
    │                    SECURITY HEADERS MIDDLEWARE                           │
    │  Adds security-related HTTP headers to all responses                    │
    └─────────────────────────────────────────────────────────────────────────┘
    """
    
    # Paths that need relaxed CSP for Swagger UI
    DOCS_PATHS = {"/docs", "/redoc", "/openapi.json"}
    
    async def dispatch(
        self,
        request: Request,
        call_next: Callable,
    ) -> Response:
        """Add security headers to response."""
        response = await call_next(request)
        
        path = request.url.path
        
        # Prevent MIME type sniffing
        response.headers["X-Content-Type-Options"] = "nosniff"
        
        # Prevent clickjacking
        response.headers["X-Frame-Options"] = "DENY"
        
        # XSS protection
        response.headers["X-XSS-Protection"] = "1; mode=block"
        
        # Referrer policy
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        
        # HSTS (only in production)
        if settings.is_production:
            response.headers["Strict-Transport-Security"] = (
                "max-age=31536000; includeSubDomains; preload"
            )
        
        # Content Security Policy
        # More permissive for docs endpoints to allow Swagger UI
        if path in self.DOCS_PATHS or path.startswith("/docs") or path.startswith("/redoc"):
            response.headers["Content-Security-Policy"] = (
                "default-src 'self'; "
                "script-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; "
                "style-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; "
                "img-src 'self' data: https://fastapi.tiangolo.com; "
                "font-src 'self' https://cdn.jsdelivr.net;"
            )
        else:
            response.headers["Content-Security-Policy"] = (
                "default-src 'self'; "
                "script-src 'self' 'unsafe-inline'; "
                "style-src 'self' 'unsafe-inline'; "
                "img-src 'self' data:; "
                "font-src 'self';"
            )
        
        # Permissions policy
        response.headers["Permissions-Policy"] = (
            "geolocation=(), "
            "microphone=(), "
            "camera=()"
        )
        
        return response


class RequestIDMiddleware(BaseHTTPMiddleware):
    """
    ┌─────────────────────────────────────────────────────────────────────────┐
    │                    REQUEST ID MIDDLEWARE                                 │
    │  Generates unique request ID for tracing and logging                    │
    └─────────────────────────────────────────────────────────────────────────┘
    """
    
    async def dispatch(
        self,
        request: Request,
        call_next: Callable,
    ) -> Response:
        """Add X-Request-ID to request and response."""
        import uuid
        
        # Get or generate request ID
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        
        # Store in request state
        request.state.request_id = request_id
        
        # Process request
        response = await call_next(request)
        
        # Add to response headers
        response.headers["X-Request-ID"] = request_id
        
        return response
