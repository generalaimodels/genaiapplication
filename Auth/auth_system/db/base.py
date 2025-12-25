# =============================================================================
# SOTA AUTHENTICATION SYSTEM - DATABASE BASE MODULE
# =============================================================================
# File: db/base.py
# Description: Abstract database interface defining the adapter pattern
#              All database implementations must conform to this interface
# =============================================================================

from abc import ABC, abstractmethod
from typing import Any, Optional, List, Dict, TypeVar, Generic, AsyncGenerator
from contextlib import asynccontextmanager

from sqlalchemy.ext.asyncio import (
    create_async_engine,
    AsyncSession,
    AsyncEngine,
    async_sessionmaker,
)
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy import MetaData


# =============================================================================
# SQLALCHEMY BASE CONFIGURATION
# =============================================================================

# Naming convention for constraints (important for migrations)
NAMING_CONVENTION = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s"
}

metadata = MetaData(naming_convention=NAMING_CONVENTION)


class Base(DeclarativeBase):
    """
    SQLAlchemy declarative base with custom metadata.
    All ORM models inherit from this base class.
    """
    metadata = metadata


# =============================================================================
# TYPE DEFINITIONS
# =============================================================================

T = TypeVar("T")


# =============================================================================
# ABSTRACT DATABASE ADAPTER INTERFACE
# =============================================================================

class IDBAdapter(ABC):
    """
    ┌─────────────────────────────────────────────────────────────────────────┐
    │                    ABSTRACT DATABASE ADAPTER INTERFACE                   │
    │  Defines the contract that all database implementations must follow     │
    │  Enables seamless switching between SQLite, PostgreSQL, etc.            │
    └─────────────────────────────────────────────────────────────────────────┘
    
    This interface follows the Adapter Pattern to provide a unified API
    for different database backends. Each implementation handles the
    specifics of its database while exposing the same methods.
    
    Methods:
        connect()      - Establish database connection
        disconnect()   - Close database connection
        get_session()  - Get async session for operations
        execute()      - Execute raw SQL
        create_tables() - Initialize database schema
    """
    
    @abstractmethod
    async def connect(self) -> None:
        """
        Establish connection to the database.
        
        Should initialize connection pool and verify connectivity.
        
        Raises:
            DatabaseConnectionError: If connection fails
        """
        pass
    
    @abstractmethod
    async def disconnect(self) -> None:
        """
        Close database connection and cleanup resources.
        
        Should properly close connection pool and release resources.
        """
        pass
    
    @abstractmethod
    @asynccontextmanager
    async def get_session(self) -> AsyncGenerator[AsyncSession, None]:
        """
        Provide an async database session context manager.
        
        Yields:
            AsyncSession: SQLAlchemy async session for database operations
            
        Usage:
            async with adapter.get_session() as session:
                result = await session.execute(query)
        """
        pass
    
    @abstractmethod
    async def execute(
        self,
        query: str,
        params: Optional[Dict[str, Any]] = None
    ) -> Any:
        """
        Execute a raw SQL query.
        
        Args:
            query: SQL query string
            params: Query parameters for safe interpolation
            
        Returns:
            Query result
        """
        pass
    
    @abstractmethod
    async def create_tables(self) -> None:
        """
        Create all defined tables in the database.
        
        Uses SQLAlchemy metadata to create tables if they don't exist.
        """
        pass
    
    @property
    @abstractmethod
    def engine(self) -> AsyncEngine:
        """
        Get the underlying SQLAlchemy async engine.
        
        Returns:
            AsyncEngine: Database engine instance
        """
        pass
    
    @property
    @abstractmethod
    def session_factory(self) -> async_sessionmaker[AsyncSession]:
        """
        Get the session factory for creating new sessions.
        
        Returns:
            async_sessionmaker: Session factory
        """
        pass


# =============================================================================
# BASE ADAPTER IMPLEMENTATION
# =============================================================================

class BaseDBAdapter(IDBAdapter):
    """
    Base implementation of database adapter with common functionality.
    Concrete adapters (SQLite, PostgreSQL) extend this class.
    """
    
    def __init__(self, database_url: str, **engine_options: Any):
        """
        Initialize base adapter with database URL.
        
        Args:
            database_url: Async-compatible database URL
            **engine_options: Additional SQLAlchemy engine options
        """
        self._database_url = database_url
        self._engine_options = engine_options
        self._engine: Optional[AsyncEngine] = None
        self._session_factory: Optional[async_sessionmaker[AsyncSession]] = None
        self._is_connected = False
    
    async def connect(self) -> None:
        """
        Create async engine and session factory.
        
        Initializes connection pool with configured options.
        """
        if self._is_connected:
            return
        
        # Create async engine with provided options
        self._engine = create_async_engine(
            self._database_url,
            **self._engine_options
        )
        
        # Create session factory
        self._session_factory = async_sessionmaker(
            bind=self._engine,
            class_=AsyncSession,
            expire_on_commit=False,
            autocommit=False,
            autoflush=False,
        )
        
        self._is_connected = True
    
    async def disconnect(self) -> None:
        """
        Dispose of engine and cleanup connections.
        """
        if self._engine:
            await self._engine.dispose()
            self._engine = None
            self._session_factory = None
            self._is_connected = False
    
    @asynccontextmanager
    async def get_session(self) -> AsyncGenerator[AsyncSession, None]:
        """
        Provide session with automatic commit/rollback handling.
        
        Commits on successful exit, rolls back on exception.
        """
        if not self._session_factory:
            await self.connect()
        
        session = self._session_factory()  # type: ignore
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()
    
    async def execute(
        self,
        query: str,
        params: Optional[Dict[str, Any]] = None
    ) -> Any:
        """
        Execute raw SQL query using a session.
        """
        from sqlalchemy import text
        
        async with self.get_session() as session:
            result = await session.execute(
                text(query),
                params or {}
            )
            return result
    
    async def create_tables(self) -> None:
        """
        Create all tables defined in SQLAlchemy metadata.
        """
        if not self._engine:
            await self.connect()
        
        async with self._engine.begin() as conn:  # type: ignore
            await conn.run_sync(Base.metadata.create_all)
    
    @property
    def engine(self) -> AsyncEngine:
        """Get the SQLAlchemy engine."""
        if not self._engine:
            raise RuntimeError("Database not connected. Call connect() first.")
        return self._engine
    
    @property
    def session_factory(self) -> async_sessionmaker[AsyncSession]:
        """Get the session factory."""
        if not self._session_factory:
            raise RuntimeError("Database not connected. Call connect() first.")
        return self._session_factory
    
    @property
    def is_connected(self) -> bool:
        """Check if database is connected."""
        return self._is_connected


# =============================================================================
# REDIS ADAPTER INTERFACE
# =============================================================================

class IRedisAdapter(ABC):
    """
    ┌─────────────────────────────────────────────────────────────────────────┐
    │                    ABSTRACT REDIS ADAPTER INTERFACE                      │
    │  Defines key-value operations for session storage and caching          │
    └─────────────────────────────────────────────────────────────────────────┘
    """
    
    @abstractmethod
    async def connect(self) -> None:
        """Establish Redis connection."""
        pass
    
    @abstractmethod
    async def disconnect(self) -> None:
        """Close Redis connection."""
        pass
    
    @abstractmethod
    async def get(self, key: str) -> Optional[str]:
        """Get value by key."""
        pass
    
    @abstractmethod
    async def set(
        self,
        key: str,
        value: str,
        ttl: Optional[int] = None
    ) -> bool:
        """Set value with optional TTL."""
        pass
    
    @abstractmethod
    async def delete(self, key: str) -> bool:
        """Delete key."""
        pass
    
    @abstractmethod
    async def exists(self, key: str) -> bool:
        """Check if key exists."""
        pass
    
    @abstractmethod
    async def hset(
        self,
        name: str,
        mapping: Dict[str, Any]
    ) -> int:
        """Set hash fields."""
        pass
    
    @abstractmethod
    async def hget(self, name: str, key: str) -> Optional[str]:
        """Get hash field."""
        pass
    
    @abstractmethod
    async def hgetall(self, name: str) -> Dict[str, str]:
        """Get all hash fields."""
        pass
    
    @abstractmethod
    async def hdel(self, name: str, *keys: str) -> int:
        """Delete hash fields."""
        pass
    
    @abstractmethod
    async def sadd(self, name: str, *values: str) -> int:
        """Add to set."""
        pass
    
    @abstractmethod
    async def srem(self, name: str, *values: str) -> int:
        """Remove from set."""
        pass
    
    @abstractmethod
    async def smembers(self, name: str) -> set:
        """Get all set members."""
        pass
    
    @abstractmethod
    async def expire(self, name: str, ttl: int) -> bool:
        """Set key expiration."""
        pass
    
    @abstractmethod
    async def incr(self, key: str) -> int:
        """Increment counter."""
        pass
    
    @abstractmethod
    async def ttl(self, key: str) -> int:
        """Get remaining TTL."""
        pass
