# =============================================================================
# SOTA AUTHENTICATION SYSTEM - SESSION STORAGE
# =============================================================================
# File: session/storage.py
# Description: Session storage abstraction using Redis
#              Provides high-performance session caching
# =============================================================================

from typing import Optional, Set
from datetime import datetime, timezone
import json

from db.adapters.redis_adapter import RedisAdapter
from session.models import SessionData
from core.config import settings


class SessionStorage:
    """
    ┌─────────────────────────────────────────────────────────────────────────┐
    │                    SESSION STORAGE                                       │
    │  Redis-backed session storage for high-performance access               │
    │  Provides caching layer on top of database sessions                     │
    └─────────────────────────────────────────────────────────────────────────┘
    
    Key Patterns:
        - session:{session_id}     → Hash with session data
        - user_sessions:{user_id}  → Set of session IDs
        - blacklist:{token_jti}    → Blacklisted token marker
    """
    
    # Key prefixes
    SESSION_PREFIX = "session:"
    USER_SESSIONS_PREFIX = "user_sessions:"
    BLACKLIST_PREFIX = "blacklist:"
    
    def __init__(self, redis: RedisAdapter):
        """
        Initialize session storage.
        
        Args:
            redis: Redis adapter instance
        """
        self._redis = redis
    
    # =========================================================================
    # SESSION OPERATIONS
    # =========================================================================
    
    async def create_session(
        self,
        session_id: str,
        data: SessionData,
        ttl: Optional[int] = None,
    ) -> bool:
        """
        Create a new session in Redis.
        
        Args:
            session_id: Unique session identifier
            data: Session data to store
            ttl: Time-to-live in seconds (default: settings.session_ttl)
            
        Returns:
            bool: True if successful
        """
        session_key = f"{self.SESSION_PREFIX}{session_id}"
        user_sessions_key = f"{self.USER_SESSIONS_PREFIX}{data.user_id}"
        
        # Convert to dict for storage
        session_dict = {
            "user_id": data.user_id,
            "email": data.email,
            "username": data.username,
            "roles": json.dumps(data.roles),
            "created_at": data.created_at.isoformat(),
            "last_activity": data.last_activity.isoformat(),
            "device_info": json.dumps(data.device_info) if data.device_info else "{}",
            "ip_address": data.ip_address,
        }
        
        # Store session hash
        await self._redis.hset(session_key, session_dict)
        
        # Set TTL
        await self._redis.expire(session_key, ttl or settings.session_ttl)
        
        # Add to user's session index
        await self._redis.sadd(user_sessions_key, session_id)
        
        return True
    
    async def get_session(self, session_id: str) -> Optional[SessionData]:
        """
        Retrieve session data from Redis.
        
        Args:
            session_id: Session identifier
            
        Returns:
            SessionData if exists and valid, None otherwise
        """
        session_key = f"{self.SESSION_PREFIX}{session_id}"
        
        data = await self._redis.hgetall(session_key)
        
        if not data:
            return None
        
        # Parse stored data
        try:
            roles = json.loads(data.get("roles", "[]"))
            device_info = json.loads(data.get("device_info", "{}"))
            
            return SessionData(
                user_id=data["user_id"],
                email=data["email"],
                username=data.get("username", ""),
                roles=roles,
                created_at=datetime.fromisoformat(data["created_at"]),
                last_activity=datetime.fromisoformat(data["last_activity"]),
                device_info=device_info,
                ip_address=data["ip_address"],
            )
        except (KeyError, json.JSONDecodeError):
            return None
    
    async def update_activity(self, session_id: str) -> bool:
        """
        Update session last activity timestamp.
        
        Args:
            session_id: Session identifier
            
        Returns:
            bool: True if successful
        """
        session_key = f"{self.SESSION_PREFIX}{session_id}"
        
        # Check if session exists
        if not await self._redis.exists(session_key):
            return False
        
        # Update last_activity
        await self._redis.hset(session_key, {
            "last_activity": datetime.now(timezone.utc).isoformat()
        })
        
        return True
    
    async def delete_session(self, session_id: str, user_id: str) -> bool:
        """
        Delete a session from Redis.
        
        Args:
            session_id: Session identifier
            user_id: User ID for index cleanup
            
        Returns:
            bool: True if successful
        """
        session_key = f"{self.SESSION_PREFIX}{session_id}"
        user_sessions_key = f"{self.USER_SESSIONS_PREFIX}{user_id}"
        
        # Delete session hash
        await self._redis.delete(session_key)
        
        # Remove from user's session index
        await self._redis.srem(user_sessions_key, session_id)
        
        return True
    
    async def delete_all_user_sessions(self, user_id: str) -> int:
        """
        Delete all sessions for a user.
        
        Args:
            user_id: User ID
            
        Returns:
            int: Number of sessions deleted
        """
        user_sessions_key = f"{self.USER_SESSIONS_PREFIX}{user_id}"
        
        # Get all session IDs
        session_ids = await self._redis.smembers(user_sessions_key)
        
        count = 0
        for session_id in session_ids:
            session_key = f"{self.SESSION_PREFIX}{session_id}"
            await self._redis.delete(session_key)
            count += 1
        
        # Clear the user sessions set
        await self._redis.delete(user_sessions_key)
        
        return count
    
    async def get_user_session_ids(self, user_id: str) -> Set[str]:
        """
        Get all session IDs for a user.
        
        Args:
            user_id: User ID
            
        Returns:
            Set of session IDs
        """
        user_sessions_key = f"{self.USER_SESSIONS_PREFIX}{user_id}"
        return await self._redis.smembers(user_sessions_key)
    
    async def session_exists(self, session_id: str) -> bool:
        """
        Check if session exists.
        
        Args:
            session_id: Session identifier
            
        Returns:
            bool: True if session exists
        """
        session_key = f"{self.SESSION_PREFIX}{session_id}"
        return await self._redis.exists(session_key)
    
    async def get_session_ttl(self, session_id: str) -> int:
        """
        Get remaining TTL for a session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            int: TTL in seconds, -1 if no TTL, -2 if not exists
        """
        session_key = f"{self.SESSION_PREFIX}{session_id}"
        return await self._redis.ttl(session_key)
    
    async def extend_session(
        self,
        session_id: str,
        ttl: Optional[int] = None,
    ) -> bool:
        """
        Extend session TTL.
        
        Args:
            session_id: Session identifier
            ttl: New TTL in seconds (default: settings.session_ttl)
            
        Returns:
            bool: True if successful
        """
        session_key = f"{self.SESSION_PREFIX}{session_id}"
        return await self._redis.expire(session_key, ttl or settings.session_ttl)
    
    # =========================================================================
    # TOKEN BLACKLIST
    # =========================================================================
    
    async def blacklist_token(
        self,
        token_jti: str,
        ttl: int,
    ) -> bool:
        """
        Add token to blacklist.
        
        Args:
            token_jti: Token unique identifier
            ttl: Time-to-live (should match token's remaining lifetime)
            
        Returns:
            bool: True if successful
        """
        key = f"{self.BLACKLIST_PREFIX}{token_jti}"
        return await self._redis.set(key, "revoked", ttl=ttl)
    
    async def is_token_blacklisted(self, token_jti: str) -> bool:
        """
        Check if token is blacklisted.
        
        Args:
            token_jti: Token unique identifier
            
        Returns:
            bool: True if blacklisted
        """
        key = f"{self.BLACKLIST_PREFIX}{token_jti}"
        return await self._redis.exists(key)
    
    # =========================================================================
    # RATE LIMITING
    # =========================================================================
    
    async def check_rate_limit(
        self,
        key: str,
        limit: int,
        window: int = 60,
    ) -> tuple[bool, int]:
        """
        Check and update rate limit counter.
        
        Args:
            key: Rate limit key (e.g., IP:endpoint)
            limit: Maximum requests allowed
            window: Time window in seconds
            
        Returns:
            tuple[bool, int]: (is_allowed, current_count)
        """
        current = await self._redis.incr(f"rate:{key}")
        
        if current == 1:
            # First request, set expiration
            await self._redis.expire(f"rate:{key}", window)
        
        return current <= limit, current
    
    async def get_rate_limit_remaining(
        self,
        key: str,
        limit: int,
    ) -> int:
        """
        Get remaining requests for rate limit.
        
        Args:
            key: Rate limit key
            limit: Maximum requests allowed
            
        Returns:
            int: Remaining requests
        """
        current = await self._redis.get(f"rate:{key}")
        if current is None:
            return limit
        return max(0, limit - int(current))
