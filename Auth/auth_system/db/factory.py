# =============================================================================
# SOTA AUTHENTICATION SYSTEM - DATABASE FACTORY
# =============================================================================
# File: db/factory.py
# Description: Factory pattern for database adapter instantiation
#              Provides unified interface for switching between database backends
# =============================================================================

from typing import Optional, Union
from enum import Enum

from db.base import BaseDBAdapter, IRedisAdapter
from db.adapters.sqlite_adapter import SQLiteAdapter
from db.adapters.postgres_adapter import PostgresAdapter
from db.adapters.redis_adapter import RedisAdapter
from core.config import settings


class DatabaseType(str, Enum):
    """Supported database types."""
    SQLITE = "sqlite"
    POSTGRESQL = "postgresql"


class DBFactory:
    """
    ┌─────────────────────────────────────────────────────────────────────────┐
    │                    DATABASE FACTORY                                      │
    │  Factory pattern implementation for creating database adapters          │
    │  Supports runtime switching between SQLite and PostgreSQL               │
    └─────────────────────────────────────────────────────────────────────────┘
    
    The factory maintains singleton instances of adapters to ensure
    consistent connection management across the application.
    
    Usage:
        # Get adapter based on settings
        db = DBFactory.get_db_adapter()
        await db.connect()
        
        # Get Redis adapter
        redis = DBFactory.get_redis_adapter()
        await redis.connect()
    """
    
    # Singleton instances
    _db_adapter: Optional[BaseDBAdapter] = None
    _redis_adapter: Optional[RedisAdapter] = None
    
    @classmethod
    def get_db_adapter(
        cls,
        db_type: Optional[str] = None,
        force_new: bool = False,
        **kwargs
    ) -> BaseDBAdapter:
        """
        Get database adapter based on configuration or specified type.
        
        Args:
            db_type: Override database type (sqlite/postgresql)
                    Defaults to settings.db_type
            force_new: Force creation of new adapter instance
            **kwargs: Additional options passed to adapter
            
        Returns:
            BaseDBAdapter: Configured database adapter
            
        Raises:
            ValueError: If unsupported database type specified
            
        Example:
            # Use configured database
            db = DBFactory.get_db_adapter()
            
            # Force SQLite for testing
            test_db = DBFactory.get_db_adapter(
                db_type="sqlite",
                force_new=True
            )
        """
        # Use singleton unless force_new
        if not force_new and cls._db_adapter is not None:
            return cls._db_adapter
        
        # Determine database type
        selected_type = db_type or settings.db_type
        
        # Create appropriate adapter
        if selected_type == DatabaseType.SQLITE or selected_type == "sqlite":
            adapter = SQLiteAdapter(**kwargs)
        elif selected_type == DatabaseType.POSTGRESQL or selected_type == "postgresql":
            adapter = PostgresAdapter(**kwargs)
        else:
            raise ValueError(
                f"Unsupported database type: {selected_type}. "
                f"Supported types: {[t.value for t in DatabaseType]}"
            )
        
        # Store as singleton if not forced new
        if not force_new:
            cls._db_adapter = adapter
        
        return adapter
    
    @classmethod
    def get_redis_adapter(
        cls,
        force_new: bool = False,
        **kwargs
    ) -> RedisAdapter:
        """
        Get Redis adapter instance.
        
        Args:
            force_new: Force creation of new adapter instance
            **kwargs: Additional options passed to adapter
            
        Returns:
            RedisAdapter: Configured Redis adapter
            
        Example:
            redis = DBFactory.get_redis_adapter()
            await redis.connect()
        """
        # Use singleton unless force_new
        if not force_new and cls._redis_adapter is not None:
            return cls._redis_adapter
        
        adapter = RedisAdapter(**kwargs)
        
        # Store as singleton if not forced new
        if not force_new:
            cls._redis_adapter = adapter
        
        return adapter
    
    @classmethod
    async def connect_all(cls) -> None:
        """
        Connect to all configured databases (SQL and Redis).
        
        In development mode, Redis connection failure is non-fatal.
        Convenience method for application startup.
        """
        import logging
        logger = logging.getLogger(__name__)
        
        db_adapter = cls.get_db_adapter()
        await db_adapter.connect()
        logger.info("Database connection established")
        
        # Redis is optional in development
        try:
            redis_adapter = cls.get_redis_adapter()
            await redis_adapter.connect()
            logger.info("Redis connection established")
        except Exception as e:
            if settings.is_development:
                logger.warning(f"Redis connection failed (optional in dev): {e}")
                cls._redis_adapter = None
            else:
                raise
    
    @classmethod
    async def disconnect_all(cls) -> None:
        """
        Disconnect from all databases.
        
        Convenience method for application shutdown.
        """
        if cls._db_adapter:
            await cls._db_adapter.disconnect()
            cls._db_adapter = None
        
        if cls._redis_adapter:
            try:
                await cls._redis_adapter.disconnect()
            except Exception:
                pass
            cls._redis_adapter = None
    
    @classmethod
    async def create_tables(cls) -> None:
        """
        Create all database tables using SQLAlchemy metadata.
        """
        db_adapter = cls.get_db_adapter()
        await db_adapter.create_tables()
    
    @classmethod
    async def health_check(cls) -> dict:
        """
        Check health of all database connections.
        
        Returns:
            Dict with health status of each component
        """
        results = {
            "database": False,
            "redis": False,
        }
        
        try:
            if cls._db_adapter:
                async with cls._db_adapter.get_session() as session:
                    await session.execute("SELECT 1")
                results["database"] = True
        except Exception:
            pass
        
        try:
            if cls._redis_adapter:
                results["redis"] = await cls._redis_adapter.check_health()
        except Exception:
            pass
        
        return results
    
    @classmethod
    def reset(cls) -> None:
        """
        Reset singleton instances (useful for testing).
        """
        cls._db_adapter = None
        cls._redis_adapter = None


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def get_db_adapter(**kwargs) -> BaseDBAdapter:
    """Get database adapter via factory."""
    return DBFactory.get_db_adapter(**kwargs)


def get_redis_adapter(**kwargs) -> RedisAdapter:
    """Get Redis adapter via factory."""
    return DBFactory.get_redis_adapter(**kwargs)


async def get_db_session():
    """
    FastAPI dependency for database session.
    
    Yields:
        AsyncSession: Database session
        
    Usage:
        @app.get("/users")
        async def get_users(
            session: AsyncSession = Depends(get_db_session)
        ):
            ...
    """
    adapter = DBFactory.get_db_adapter()
    async with adapter.get_session() as session:
        yield session


async def get_redis():
    """
    FastAPI dependency for Redis client.
    
    Returns:
        RedisAdapter: Redis adapter instance
        
    Usage:
        @app.get("/cache")
        async def get_cached(
            redis: RedisAdapter = Depends(get_redis)
        ):
            ...
    """
    return DBFactory.get_redis_adapter()
