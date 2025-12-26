# ==============================================================================
# RATE LIMITER MIDDLEWARE
# ==============================================================================
# Token bucket rate limiting implementation
# ==============================================================================

from __future__ import annotations

import time
from collections import defaultdict
from typing import Callable, Dict, Tuple

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from app.core.settings import settings


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Rate limiting middleware using token bucket algorithm.
    
    Limits requests per client based on IP address.
    Configurable through settings.
    
    Attributes:
        requests_limit: Maximum requests per window
        window_seconds: Time window in seconds
        _tokens: Token bucket for each client
    """
    
    def __init__(
        self,
        app,
        requests_limit: int = None,
        window_seconds: int = None,
    ):
        super().__init__(app)
        self.requests_limit = requests_limit or settings.RATE_LIMIT_REQUESTS
        self.window_seconds = window_seconds or settings.RATE_LIMIT_WINDOW
        self._tokens: Dict[str, Tuple[int, float]] = defaultdict(
            lambda: (self.requests_limit, time.time())
        )
    
    def _get_client_id(self, request: Request) -> str:
        """Get client identifier from request."""
        # Try to get real IP from forwarded headers
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            return forwarded.split(",")[0].strip()
        
        if request.client:
            return request.client.host
        
        return "unknown"
    
    def _check_rate_limit(self, client_id: str) -> Tuple[bool, int, int]:
        """
        Check if request is within rate limit.
        
        Returns:
            Tuple of (allowed, remaining, reset_time)
        """
        current_time = time.time()
        tokens, last_update = self._tokens[client_id]
        
        # Refill tokens based on time elapsed
        elapsed = current_time - last_update
        refill = int(elapsed / self.window_seconds * self.requests_limit)
        tokens = min(self.requests_limit, tokens + refill)
        
        if tokens > 0:
            # Allow request
            tokens -= 1
            self._tokens[client_id] = (tokens, current_time)
            reset_time = int(self.window_seconds - (current_time - last_update))
            return True, tokens, max(0, reset_time)
        else:
            # Rate limit exceeded
            self._tokens[client_id] = (0, last_update)
            reset_time = int(self.window_seconds - (current_time - last_update))
            return False, 0, max(0, reset_time) or self.window_seconds
    
    async def dispatch(
        self,
        request: Request,
        call_next: Callable,
    ) -> Response:
        """Process request through rate limiter."""
        # Skip rate limiting if disabled
        if not settings.RATE_LIMIT_ENABLED:
            return await call_next(request)
        
        # Skip health endpoints
        if request.url.path in ["/health", "/"]:
            return await call_next(request)
        
        client_id = self._get_client_id(request)
        allowed, remaining, reset_time = self._check_rate_limit(client_id)
        
        if not allowed:
            return JSONResponse(
                status_code=429,
                content={
                    "success": False,
                    "error": {
                        "code": "RATE_LIMIT_EXCEEDED",
                        "message": "Too many requests",
                        "details": {
                            "retry_after_seconds": reset_time,
                        }
                    }
                },
                headers={
                    "X-RateLimit-Limit": str(self.requests_limit),
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Reset": str(reset_time),
                    "Retry-After": str(reset_time),
                },
            )
        
        response = await call_next(request)
        
        # Add rate limit headers
        response.headers["X-RateLimit-Limit"] = str(self.requests_limit)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        response.headers["X-RateLimit-Reset"] = str(reset_time)
        
        return response
