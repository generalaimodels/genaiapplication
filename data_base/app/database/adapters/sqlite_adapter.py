# ==============================================================================
# SQLITE ADAPTER - SQLAlchemy Async with aiosqlite
# ==============================================================================
# Lightweight database adapter for development and testing
# Full async support using aiosqlite driver
# ==============================================================================

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Dict, List, Optional, Type

from sqlalchemy import and_, delete, func, select, text, update
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase

from app.core.settings import settings
from app.core.exceptions import DatabaseError
from app.database.adapters.base_adapter import BaseDatabaseAdapter

logger = logging.getLogger(__name__)


class SQLiteAdapter(BaseDatabaseAdapter[Any]):
    """
    SQLite database adapter using SQLAlchemy async with aiosqlite.
    
    Ideal for development, testing, and small-scale deployments.
    Provides the same interface as PostgreSQLAdapter for seamless
    database switching.
    
    Features:
        - Async SQLite operations using aiosqlite
        - Automatic table creation on connect
        - Same API as PostgreSQLAdapter for compatibility
        - File-based or in-memory database support
        
    Attributes:
        _database_url: SQLite connection string
        _engine: SQLAlchemy async engine
        _session_factory: Session factory for creating sessions
        _model_registry: Mapping of collection names to model classes
        
    Example:
        >>> adapter = SQLiteAdapter()
        >>> await adapter.connect()  # Creates tables automatically
        >>> adapter.register_model("users", User)
        >>> user = await adapter.create("users", {"email": "test@example.com"})
    """
    
    def __init__(self, database_url: Optional[str] = None) -> None:
        """
        Initialize SQLite adapter.
        
        Args:
            database_url: SQLite connection URL (defaults to settings)
        """
        # Ensure async driver is used
        url = database_url or settings.SQLITE_URL
        if "sqlite://" in url and "aiosqlite" not in url:
            url = url.replace("sqlite://", "sqlite+aiosqlite://")
        
        self._database_url = url
        self._engine: Optional[AsyncEngine] = None
        self._session_factory: Optional[async_sessionmaker[AsyncSession]] = None
        self._model_registry: Dict[str, Type[DeclarativeBase]] = {}
    
    # ==========================================================================
    # MODEL REGISTRY
    # ==========================================================================
    
    def register_model(
        self,
        name: str,
        model: Type[DeclarativeBase],
    ) -> None:
        """
        Register a SQLAlchemy model for table mapping.
        
        Args:
            name: Collection/table identifier
            model: SQLAlchemy model class
        """
        self._model_registry[name] = model
        logger.debug(f"Registered model '{name}' -> {model.__name__}")
    
    def _get_model(self, collection: str) -> Type[DeclarativeBase]:
        """
        Get registered model by collection name.
        
        Args:
            collection: Collection/table identifier
            
        Returns:
            Registered model class
            
        Raises:
            ValueError: If model not registered
        """
        if collection not in self._model_registry:
            raise ValueError(
                f"Model '{collection}' not registered. "
                f"Available models: {list(self._model_registry.keys())}"
            )
        return self._model_registry[collection]
    
    # ==========================================================================
    # LIFECYCLE METHODS
    # ==========================================================================
    
    async def connect(self) -> None:
        """
        Initialize database engine and create tables.
        
        Creates an async engine and automatically creates all
        registered tables if they don't exist.
        """
        try:
            self._engine = create_async_engine(
                self._database_url,
                echo=settings.DEBUG,
                # SQLite-specific settings
                connect_args={"check_same_thread": False},
            )
            
            self._session_factory = async_sessionmaker(
                bind=self._engine,
                class_=AsyncSession,
                expire_on_commit=False,
                autoflush=False,
            )
            
            # Create tables
            async with self._engine.begin() as conn:
                from app.domain_models.base import SQLBase
                await conn.run_sync(SQLBase.metadata.create_all)
            
            logger.info("SQLite adapter connected successfully")
            
        except Exception as e:
            logger.error(f"Failed to connect to SQLite: {e}")
            raise DatabaseError(f"SQLite connection failed: {e}")
    
    async def disconnect(self) -> None:
        """Close database connections and dispose engine."""
        if self._engine:
            await self._engine.dispose()
            logger.info("SQLite adapter disconnected")
    
    async def health_check(self) -> bool:
        """
        Verify database connectivity.
        
        Returns:
            True if connection is healthy
        """
        try:
            async with self.session() as session:
                await session.execute(text("SELECT 1"))
            return True
        except Exception as e:
            logger.warning(f"SQLite health check failed: {e}")
            return False
    
    # ==========================================================================
    # SESSION MANAGEMENT
    # ==========================================================================
    
    @asynccontextmanager
    async def session(self) -> AsyncIterator[AsyncSession]:
        """
        Provide transactional session scope.
        
        Commits on successful exit, rolls back on exception.
        
        Yields:
            AsyncSession instance
            
        Raises:
            RuntimeError: If database not connected
        """
        if not self._session_factory:
            raise RuntimeError(
                "Database not connected. Call connect() first."
            )
        
        session: AsyncSession = self._session_factory()
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()
    
    # ==========================================================================
    # CRUD OPERATIONS
    # ==========================================================================
    
    async def create(
        self,
        collection: str,
        data: Dict[str, Any],
    ) -> Any:
        """Create a new record."""
        model = self._get_model(collection)
        
        async with self.session() as session:
            instance = model(**data)
            session.add(instance)
            await session.flush()
            await session.refresh(instance)
            return instance
    
    async def get_by_id(
        self,
        collection: str,
        id: Any,
    ) -> Optional[Any]:
        """Retrieve record by primary key."""
        model = self._get_model(collection)
        
        async with self.session() as session:
            return await session.get(model, id)
    
    async def get_all(
        self,
        collection: str,
        skip: int = 0,
        limit: int = 100,
        filters: Optional[Dict[str, Any]] = None,
        sort_by: Optional[str] = None,
        sort_order: str = "asc",
    ) -> List[Any]:
        """Retrieve multiple records with pagination and filtering."""
        model = self._get_model(collection)
        
        async with self.session() as session:
            query = select(model)
            
            # Apply filters
            if filters:
                conditions = [
                    getattr(model, key) == value
                    for key, value in filters.items()
                    if hasattr(model, key)
                ]
                if conditions:
                    query = query.where(and_(*conditions))
            
            # Apply sorting
            if sort_by and hasattr(model, sort_by):
                order_column = getattr(model, sort_by)
                if sort_order.lower() == "desc":
                    order_column = order_column.desc()
                query = query.order_by(order_column)
            
            # Apply pagination
            query = query.offset(skip).limit(limit)
            
            result = await session.execute(query)
            return list(result.scalars().all())
    
    async def update(
        self,
        collection: str,
        id: Any,
        data: Dict[str, Any],
    ) -> Optional[Any]:
        """Update an existing record."""
        model = self._get_model(collection)
        
        async with self.session() as session:
            instance = await session.get(model, id)
            if not instance:
                return None
            
            for key, value in data.items():
                if hasattr(instance, key):
                    setattr(instance, key, value)
            
            await session.flush()
            await session.refresh(instance)
            return instance
    
    async def delete(
        self,
        collection: str,
        id: Any,
    ) -> bool:
        """Delete a record by ID."""
        model = self._get_model(collection)
        
        async with self.session() as session:
            instance = await session.get(model, id)
            if not instance:
                return False
            
            await session.delete(instance)
            return True
    
    # ==========================================================================
    # QUERY OPERATIONS
    # ==========================================================================
    
    async def count(
        self,
        collection: str,
        filters: Optional[Dict[str, Any]] = None,
    ) -> int:
        """Count records matching filters."""
        model = self._get_model(collection)
        
        async with self.session() as session:
            query = select(func.count()).select_from(model)
            
            if filters:
                conditions = [
                    getattr(model, key) == value
                    for key, value in filters.items()
                    if hasattr(model, key)
                ]
                if conditions:
                    query = query.where(and_(*conditions))
            
            result = await session.execute(query)
            return result.scalar() or 0
    
    async def exists(
        self,
        collection: str,
        filters: Dict[str, Any],
    ) -> bool:
        """Check if any record matches filters."""
        count = await self.count(collection, filters)
        return count > 0
    
    async def find_one(
        self,
        collection: str,
        filters: Dict[str, Any],
    ) -> Optional[Any]:
        """Find a single record matching filters."""
        results = await self.get_all(
            collection,
            skip=0,
            limit=1,
            filters=filters,
        )
        return results[0] if results else None
    
    # ==========================================================================
    # BULK OPERATIONS
    # ==========================================================================
    
    async def bulk_create(
        self,
        collection: str,
        data: List[Dict[str, Any]],
    ) -> List[Any]:
        """Bulk insert records."""
        model = self._get_model(collection)
        
        async with self.session() as session:
            instances = [model(**item) for item in data]
            session.add_all(instances)
            await session.flush()
            
            for instance in instances:
                await session.refresh(instance)
            
            return instances
    
    async def bulk_update(
        self,
        collection: str,
        filters: Dict[str, Any],
        data: Dict[str, Any],
    ) -> int:
        """Bulk update records matching filters."""
        model = self._get_model(collection)
        
        async with self.session() as session:
            conditions = [
                getattr(model, key) == value
                for key, value in filters.items()
                if hasattr(model, key)
            ]
            
            stmt = update(model).where(and_(*conditions)).values(**data)
            result = await session.execute(stmt)
            return result.rowcount
    
    async def bulk_delete(
        self,
        collection: str,
        filters: Dict[str, Any],
    ) -> int:
        """Bulk delete records matching filters."""
        model = self._get_model(collection)
        
        async with self.session() as session:
            conditions = [
                getattr(model, key) == value
                for key, value in filters.items()
                if hasattr(model, key)
            ]
            
            stmt = delete(model).where(and_(*conditions))
            result = await session.execute(stmt)
            return result.rowcount
    
    # ==========================================================================
    # RAW QUERY EXECUTION
    # ==========================================================================
    
    async def execute_raw(
        self,
        query: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """Execute raw SQL query."""
        async with self.session() as session:
            result = await session.execute(text(query), params or {})
            return result.fetchall()
