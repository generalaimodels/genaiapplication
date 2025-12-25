# =============================================================================
# SOTA AUTHENTICATION SYSTEM - REDIS ADAPTER
# =============================================================================
# File: db/adapters/redis_adapter.py
# Description: Redis adapter for session storage, caching, and rate limiting
#              Uses redis-py async client with hiredis parser for performance
# =============================================================================

from typing import Any, Dict, Optional, Set
import json

from redis.asyncio import Redis, ConnectionPool
from redis.exceptions import RedisError, ConnectionError as RedisConnectionError

from db.base import IRedisAdapter
from core.config import settings
from core.exceptions import RedisConnectionError as CustomRedisConnectionError


class RedisAdapter(IRedisAdapter):
    """
    ┌─────────────────────────────────────────────────────────────────────────┐
    │                    REDIS ADAPTER                                         │
    │  High-performance Redis client for sessions, caching, and rate limiting │
    │  Uses hiredis parser for optimal performance                            │
    └─────────────────────────────────────────────────────────────────────────┘
    
    Data Structures Used:
        - HASH:   Session data storage
        - SET:    User sessions index
        - STRING: Rate limiting counters, token blacklist
    
    Key Patterns:
        - session:{session_id}     → Session hash
        - user_sessions:{user_id}  → Set of session IDs
        - rate_limit:{ip}:{path}   → Rate limit counter
        - blacklist:{token_jti}    → Blacklisted token
    
    Features:
        - Connection pooling
        - Automatic reconnection
        - Pipelining support
        - Lua script execution
        - Health checking
    """
    
    def __init__(
        self,
        redis_url: Optional[str] = None,
        **kwargs: Any
    ):
        """
        Initialize Redis adapter with connection pool.
        
        Args:
            redis_url: Optional Redis URL, defaults to settings.redis_url
            **kwargs: Additional redis-py options
                - max_connections: int - Pool size (default: 10)
                - socket_timeout: float - Socket timeout (default: 5.0)
                - socket_connect_timeout: float - Connection timeout (default: 5.0)
                - retry_on_timeout: bool - Retry on timeout (default: True)
        """
        self._redis_url = redis_url or settings.redis_url
        self._options = kwargs
        
        # Default connection options
        self._default_options = {
            "max_connections": 10,
            "socket_timeout": 5.0,
            "socket_connect_timeout": 5.0,
            "retry_on_timeout": True,
            "decode_responses": True,  # Return strings instead of bytes
        }
        
        self._pool: Optional[ConnectionPool] = None
        self._client: Optional[Redis] = None
        self._is_connected = False
    
    async def connect(self) -> None:
        """
        Establish connection to Redis server.
        
        Creates connection pool and verifies connectivity.
        
        Raises:
            RedisConnectionError: If connection fails
        """
        if self._is_connected:
            return
        
        try:
            # Merge default options with provided options
            options = {**self._default_options, **self._options}
            
            # Create connection pool
            self._pool = ConnectionPool.from_url(
                self._redis_url,
                **options
            )
            
            # Create client with pool
            self._client = Redis(connection_pool=self._pool)
            
            # Verify connection
            await self._client.ping()
            
            self._is_connected = True
            
        except (RedisError, RedisConnectionError) as e:
            raise CustomRedisConnectionError(
                details={"error": str(e), "url": self._redis_url}
            )
    
    async def disconnect(self) -> None:
        """
        Close Redis connection and cleanup resources.
        """
        if self._client:
            await self._client.close()
            self._client = None
        
        if self._pool:
            await self._pool.disconnect()
            self._pool = None
        
        self._is_connected = False
    
    def _ensure_connected(self) -> Redis:
        """Ensure client is connected and return it."""
        if not self._client or not self._is_connected:
            raise RuntimeError("Redis not connected. Call connect() first.")
        return self._client
    
    # =========================================================================
    # STRING OPERATIONS
    # =========================================================================
    
    async def get(self, key: str) -> Optional[str]:
        """
        Get string value by key.
        
        Args:
            key: Redis key
            
        Returns:
            Value if exists, None otherwise
        """
        client = self._ensure_connected()
        return await client.get(key)
    
    async def set(
        self,
        key: str,
        value: str,
        ttl: Optional[int] = None
    ) -> bool:
        """
        Set string value with optional TTL.
        
        Args:
            key: Redis key
            value: String value
            ttl: Time-to-live in seconds
            
        Returns:
            True if successful
        """
        client = self._ensure_connected()
        if ttl:
            return await client.setex(key, ttl, value)
        return await client.set(key, value)
    
    async def delete(self, key: str) -> bool:
        """
        Delete a key.
        
        Args:
            key: Redis key
            
        Returns:
            True if key was deleted
        """
        client = self._ensure_connected()
        result = await client.delete(key)
        return result > 0
    
    async def exists(self, key: str) -> bool:
        """
        Check if key exists.
        
        Args:
            key: Redis key
            
        Returns:
            True if key exists
        """
        client = self._ensure_connected()
        return await client.exists(key) > 0
    
    async def incr(self, key: str) -> int:
        """
        Increment counter.
        
        Args:
            key: Redis key
            
        Returns:
            New value after increment
        """
        client = self._ensure_connected()
        return await client.incr(key)
    
    async def ttl(self, key: str) -> int:
        """
        Get remaining TTL.
        
        Args:
            key: Redis key
            
        Returns:
            TTL in seconds, -1 if no TTL, -2 if key doesn't exist
        """
        client = self._ensure_connected()
        return await client.ttl(key)
    
    async def expire(self, name: str, ttl: int) -> bool:
        """
        Set key expiration.
        
        Args:
            name: Redis key
            ttl: Time-to-live in seconds
            
        Returns:
            True if TTL was set
        """
        client = self._ensure_connected()
        return await client.expire(name, ttl)
    
    # =========================================================================
    # HASH OPERATIONS (for sessions)
    # =========================================================================
    
    async def hset(self, name: str, mapping: Dict[str, Any]) -> int:
        """
        Set multiple hash fields.
        
        Args:
            name: Hash key
            mapping: Field-value pairs
            
        Returns:
            Number of fields added
        """
        client = self._ensure_connected()
        # Convert non-string values to JSON
        serialized = {
            k: json.dumps(v) if not isinstance(v, str) else v
            for k, v in mapping.items()
        }
        return await client.hset(name, mapping=serialized)
    
    async def hget(self, name: str, key: str) -> Optional[str]:
        """
        Get hash field value.
        
        Args:
            name: Hash key
            key: Field name
            
        Returns:
            Field value if exists
        """
        client = self._ensure_connected()
        return await client.hget(name, key)
    
    async def hgetall(self, name: str) -> Dict[str, str]:
        """
        Get all hash fields and values.
        
        Args:
            name: Hash key
            
        Returns:
            Dictionary of field-value pairs
        """
        client = self._ensure_connected()
        return await client.hgetall(name)
    
    async def hdel(self, name: str, *keys: str) -> int:
        """
        Delete hash fields.
        
        Args:
            name: Hash key
            *keys: Field names to delete
            
        Returns:
            Number of fields deleted
        """
        client = self._ensure_connected()
        if not keys:
            return 0
        return await client.hdel(name, *keys)
    
    # =========================================================================
    # SET OPERATIONS (for user sessions index)
    # =========================================================================
    
    async def sadd(self, name: str, *values: str) -> int:
        """
        Add values to set.
        
        Args:
            name: Set key
            *values: Values to add
            
        Returns:
            Number of new elements added
        """
        client = self._ensure_connected()
        if not values:
            return 0
        return await client.sadd(name, *values)
    
    async def srem(self, name: str, *values: str) -> int:
        """
        Remove values from set.
        
        Args:
            name: Set key
            *values: Values to remove
            
        Returns:
            Number of elements removed
        """
        client = self._ensure_connected()
        if not values:
            return 0
        return await client.srem(name, *values)
    
    async def smembers(self, name: str) -> Set[str]:
        """
        Get all set members.
        
        Args:
            name: Set key
            
        Returns:
            Set of all members
        """
        client = self._ensure_connected()
        return await client.smembers(name)
    
    async def scard(self, name: str) -> int:
        """
        Get set cardinality (size).
        
        Args:
            name: Set key
            
        Returns:
            Number of elements in set
        """
        client = self._ensure_connected()
        return await client.scard(name)
    
    # =========================================================================
    # PIPELINE OPERATIONS
    # =========================================================================
    
    async def pipeline_execute(self, commands: list) -> list:
        """
        Execute multiple commands in a pipeline.
        
        Args:
            commands: List of (method_name, args, kwargs) tuples
            
        Returns:
            List of results
        """
        client = self._ensure_connected()
        pipe = client.pipeline()
        
        for cmd in commands:
            method_name, args, kwargs = cmd
            method = getattr(pipe, method_name)
            method(*args, **kwargs)
        
        return await pipe.execute()
    
    # =========================================================================
    # UTILITY METHODS
    # =========================================================================
    
    async def check_health(self) -> bool:
        """
        Check Redis connection health.
        
        Returns:
            True if Redis is healthy
        """
        try:
            client = self._ensure_connected()
            return await client.ping()
        except Exception:
            return False
    
    async def flush_db(self) -> bool:
        """
        Flush current database (use with caution!).
        
        Returns:
            True if successful
        """
        client = self._ensure_connected()
        return await client.flushdb()
    
    async def get_info(self) -> Dict[str, Any]:
        """
        Get Redis server info.
        
        Returns:
            Server info dictionary
        """
        client = self._ensure_connected()
        return await client.info()
    
    async def keys_count(self, pattern: str = "*") -> int:
        """
        Count keys matching pattern.
        
        Args:
            pattern: Key pattern (e.g., "session:*")
            
        Returns:
            Number of matching keys
        """
        client = self._ensure_connected()
        keys = await client.keys(pattern)
        return len(keys)
    
    @property
    def is_connected(self) -> bool:
        """Check if connected to Redis."""
        return self._is_connected


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def get_redis_adapter(**kwargs: Any) -> RedisAdapter:
    """
    Factory function to create Redis adapter.
    
    Args:
        **kwargs: Options passed to RedisAdapter
        
    Returns:
        RedisAdapter: Configured adapter instance
    """
    return RedisAdapter(**kwargs)
