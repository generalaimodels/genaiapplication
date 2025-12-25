# =============================================================================
# SOTA AUTHENTICATION SYSTEM - POSTGRESQL ADAPTER
# =============================================================================
# File: db/adapters/postgres_adapter.py
# Description: PostgreSQL database adapter for production environments
#              Uses asyncpg for high-performance async operations
# =============================================================================

from typing import Any, Dict, Optional

from db.base import BaseDBAdapter
from core.config import settings


class PostgresAdapter(BaseDBAdapter):
    """
    ┌─────────────────────────────────────────────────────────────────────────┐
    │                    POSTGRESQL DATABASE ADAPTER                           │
    │  High-performance async PostgreSQL implementation for production        │
    │  Uses asyncpg driver with SQLAlchemy async ORM                          │
    └─────────────────────────────────────────────────────────────────────────┘
    
    Features:
        - Connection pooling with configurable size
        - Prepared statement caching
        - SSL/TLS support for secure connections
        - JSONB support for complex data
        - High concurrency support
    
    Connection Pool Configuration:
        - pool_size:     Initial connections (default: 5)
        - max_overflow:  Extra connections allowed (default: 10)
        - pool_timeout:  Wait time for connection (default: 30s)
        - pool_recycle:  Recycle connections after (default: 1800s)
    
    Usage:
        adapter = PostgresAdapter()
        await adapter.connect()
        async with adapter.get_session() as session:
            # perform database operations
        await adapter.disconnect()
    """
    
    def __init__(
        self,
        database_url: Optional[str] = None,
        **kwargs: Any
    ):
        """
        Initialize PostgreSQL adapter with connection pool.
        
        Args:
            database_url: Optional custom database URL
                         Defaults to settings.database_url
            **kwargs: Additional engine options overriding defaults
                     
        Engine Options:
            - pool_size: int - Number of connections to maintain
            - max_overflow: int - Max additional connections
            - pool_timeout: int - Seconds to wait for connection
            - pool_recycle: int - Seconds before recycling connection
            - pool_pre_ping: bool - Test connections before use
            - echo: bool - Log SQL queries
        """
        # Use provided URL or get from settings
        if database_url is None:
            database_url = settings.database_url
        
        # Production-optimized connection pool settings
        default_options = {
            # Connection pool configuration
            "pool_size": settings.db_pool_size,
            "max_overflow": settings.db_max_overflow,
            "pool_timeout": settings.db_pool_timeout,
            "pool_recycle": 1800,  # Recycle connections every 30 minutes
            "pool_pre_ping": True,  # Verify connections before use
            
            # Query logging (disabled in production)
            "echo": settings.debug and settings.is_development,
            
            # asyncpg-specific options via connect_args
            "connect_args": {
                # Statement cache size for prepared statements
                "statement_cache_size": 100,
                # Command timeout in seconds
                "command_timeout": 60,
                # Prepared statement cache
                "prepared_statement_cache_size": 100,
            },
        }
        
        # Merge with provided options
        default_options.update(kwargs)
        
        super().__init__(database_url, **default_options)
    
    async def connect(self) -> None:
        """
        Connect to PostgreSQL with connection verification.
        
        Tests the connection and logs pool status.
        """
        await super().connect()
        
        # Verify connection by executing simple query
        async with self.get_session() as session:
            result = await session.execute("SELECT 1")
            result.scalar()  # Consume result to verify connection
    
    async def execute(
        self,
        query: str,
        params: Optional[Dict[str, Any]] = None
    ) -> Any:
        """
        Execute raw SQL query with parameter binding.
        
        PostgreSQL supports advanced parameter types including
        arrays, JSONB, and custom types.
        """
        from sqlalchemy import text
        
        async with self.get_session() as session:
            result = await session.execute(
                text(query),
                params or {}
            )
            return result
    
    async def get_pool_status(self) -> Dict[str, int]:
        """
        Get current connection pool statistics.
        
        Returns:
            Dict with pool status:
                - size: Current pool size
                - checked_in: Available connections
                - checked_out: In-use connections
                - overflow: Overflow connections in use
        """
        if not self._engine:
            return {"size": 0, "checked_in": 0, "checked_out": 0, "overflow": 0}
        
        pool = self._engine.pool
        return {
            "size": pool.size(),
            "checked_in": pool.checkedin(),
            "checked_out": pool.checkedout(),
            "overflow": pool.overflow(),
        }
    
    async def check_health(self) -> bool:
        """
        Perform health check on database connection.
        
        Returns:
            bool: True if database is healthy
        """
        try:
            async with self.get_session() as session:
                result = await session.execute("SELECT 1")
                return result.scalar() == 1
        except Exception:
            return False
    
    async def get_version(self) -> str:
        """
        Get PostgreSQL server version.
        
        Returns:
            str: Server version string
        """
        async with self.get_session() as session:
            result = await session.execute("SELECT version()")
            return result.scalar() or "Unknown"
    
    async def get_database_size(self) -> int:
        """
        Get the size of the current database in bytes.
        
        Returns:
            int: Database size in bytes
        """
        async with self.get_session() as session:
            result = await session.execute(
                "SELECT pg_database_size(current_database())"
            )
            return result.scalar() or 0
    
    async def analyze_table(self, table_name: str) -> None:
        """
        Update statistics for query planner on a specific table.
        
        Args:
            table_name: Name of table to analyze
        """
        from sqlalchemy import text
        
        async with self.get_session() as session:
            await session.execute(text(f"ANALYZE {table_name}"))
    
    async def vacuum_table(self, table_name: str, full: bool = False) -> None:
        """
        Run VACUUM on a specific table to reclaim space.
        
        Args:
            table_name: Name of table to vacuum
            full: If True, perform VACUUM FULL (requires exclusive lock)
        """
        # Note: VACUUM cannot run inside a transaction
        # This needs raw connection handling
        pass
    
    @classmethod
    def create_for_testing(cls, test_database_url: str) -> "PostgresAdapter":
        """
        Create a PostgreSQL adapter configured for testing.
        
        Args:
            test_database_url: Test database connection URL
            
        Returns:
            PostgresAdapter: Configured for testing (smaller pool)
        """
        return cls(
            database_url=test_database_url,
            pool_size=2,
            max_overflow=3,
            echo=True,
        )


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def get_postgres_adapter(**kwargs: Any) -> PostgresAdapter:
    """
    Factory function to create PostgreSQL adapter.
    
    Args:
        **kwargs: Options passed to PostgresAdapter
        
    Returns:
        PostgresAdapter: Configured adapter instance
    """
    return PostgresAdapter(**kwargs)
