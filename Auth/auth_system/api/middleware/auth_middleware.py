# =============================================================================
# SOTA AUTHENTICATION SYSTEM - AUTH MIDDLEWARE
# =============================================================================
# File: api/middleware/auth_middleware.py
# Description: Authentication middleware for token validation
# =============================================================================

from typing import Optional, Callable
import logging

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from core.security import jwt_manager
from db.factory import DBFactory


logger = logging.getLogger(__name__)


class AuthenticationMiddleware(BaseHTTPMiddleware):
    """
    ┌─────────────────────────────────────────────────────────────────────────┐
    │                    AUTHENTICATION MIDDLEWARE                             │
    │  Validates JWT tokens and attaches user context to requests             │
    │  Works alongside FastAPI dependencies for flexible auth                 │
    └─────────────────────────────────────────────────────────────────────────┘
    
    This middleware:
        1. Extracts Bearer token from Authorization header
        2. Validates JWT signature and expiration
        3. Checks token blacklist in Redis
        4. Attaches token payload to request.state
    
    Protected routes still use dependencies for fine-grained control,
    but this middleware provides early validation and context setup.
    """
    
    # Public endpoints that don't require authentication
    PUBLIC_ENDPOINTS = {
        # Auth endpoints
        "/api/v1/auth/register",
        "/api/v1/auth/login",
        "/api/v1/auth/forgot-password",
        "/api/v1/auth/reset-password",
        "/api/v1/auth/verify-email",
        
        # Health checks
        "/health",
        "/health/ready",
        "/health/live",
        
        # Documentation
        "/",
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
        Process request with optional authentication.
        
        Args:
            request: Incoming request
            call_next: Next middleware/handler
            
        Returns:
            Response from handler
        """
        path = request.url.path
        
        # Initialize auth state
        request.state.user_id = None
        request.state.session_id = None
        request.state.token_payload = None
        request.state.is_authenticated = False
        
        # Skip auth for public endpoints
        if path in self.PUBLIC_ENDPOINTS:
            return await call_next(request)
        
        # Extract authorization header
        auth_header = request.headers.get("Authorization")
        
        if auth_header and auth_header.startswith("Bearer "):
            token = auth_header[7:]  # Remove "Bearer " prefix
            
            try:
                # Validate token
                payload = jwt_manager.verify_token(token, expected_type="access")
                
                # Check blacklist
                try:
                    redis = DBFactory.get_redis_adapter()
                    is_blacklisted = await redis.exists(f"blacklist:{payload.jti}")
                    if is_blacklisted:
                        # Token is blacklisted, but don't block yet
                        # Let the route handler decide via dependencies
                        pass
                    else:
                        # Attach to request state
                        request.state.user_id = payload.sub
                        request.state.session_id = payload.session_id
                        request.state.token_payload = payload
                        request.state.is_authenticated = True
                except Exception:
                    # Redis error, continue without blacklist check
                    request.state.user_id = payload.sub
                    request.state.session_id = payload.session_id
                    request.state.token_payload = payload
                    request.state.is_authenticated = True
                    
            except Exception as e:
                # Token validation failed
                # Don't block - let dependencies handle auth errors
                logger.debug(f"Token validation failed: {e}")
        
        return await call_next(request)


class LoggingMiddleware(BaseHTTPMiddleware):
    """
    ┌─────────────────────────────────────────────────────────────────────────┐
    │                    LOGGING MIDDLEWARE                                    │
    │  Logs request/response details for monitoring and debugging             │
    └─────────────────────────────────────────────────────────────────────────┘
    """
    
    async def dispatch(
        self,
        request: Request,
        call_next: Callable,
    ) -> Response:
        """Log request and response details."""
        import time
        
        # Start timer
        start_time = time.perf_counter()
        
        # Get request info
        method = request.method
        path = request.url.path
        client_ip = self._get_client_ip(request)
        request_id = getattr(request.state, "request_id", "unknown")
        
        # Process request
        try:
            response = await call_next(request)
            
            # Calculate duration
            duration = time.perf_counter() - start_time
            duration_ms = round(duration * 1000, 2)
            
            # Log successful request
            logger.info(
                f"{method} {path} - {response.status_code} - {duration_ms}ms - "
                f"IP: {client_ip} - RequestID: {request_id}"
            )
            
            return response
            
        except Exception as e:
            # Calculate duration
            duration = time.perf_counter() - start_time
            duration_ms = round(duration * 1000, 2)
            
            # Log error
            logger.error(
                f"{method} {path} - ERROR - {duration_ms}ms - "
                f"IP: {client_ip} - RequestID: {request_id} - {str(e)}"
            )
            
            raise
    
    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP from request."""
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            return forwarded.split(",")[0].strip()
        
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip.strip()
        
        return request.client.host if request.client else "unknown"
