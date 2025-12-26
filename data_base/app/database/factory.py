# ==============================================================================
# DATABASE FACTORY - Adapter Instantiation & Lifecycle Management
# ==============================================================================
# Factory Pattern for creating and managing database adapters
# Singleton caching for efficient resource utilization
# ==============================================================================

from __future__ import annotations

import logging
from typing import Dict, Optional

from app.core.settings import settings, DatabaseType
from app.core.exceptions import DatabaseError
from app.database.adapters.base_adapter import BaseDatabaseAdapter
from app.database.adapters.postgresql_adapter import PostgreSQLAdapter
from app.database.adapters.mongodb_adapter import MongoDBAdapter
from app.database.adapters.sqlite_adapter import SQLiteAdapter

logger = logging.getLogger(__name__)


class DatabaseFactory:
    """
    Factory class for creating and managing database adapters.
    
    Implements the Factory Pattern with singleton caching to ensure
    efficient resource utilization across the application.
    
    Features:
        - Dynamic adapter creation based on configuration
        - Singleton caching for adapter instances
        - Lifecycle management (initialize/shutdown)
        - Type-safe adapter retrieval
        
    Class Attributes:
        _instances: Cache of initialized adapter instances
        
    Example:
        >>> # Initialize at application startup
        >>> await DatabaseFactory.initialize()
        >>>
        >>> # Get adapter for database operations
        >>> adapter = DatabaseFactory.get_adapter()
        >>> user = await adapter.get_by_id("users", user_id)
        >>>
        >>> # Shutdown at application exit
        >>> await DatabaseFactory.shutdown()
    """
    
    _instances: Dict[DatabaseType, BaseDatabaseAdapter] = {}
    
    @classmethod
    def create_adapter(
        cls,
        db_type: Optional[DatabaseType] = None,
        **kwargs,
    ) -> BaseDatabaseAdapter:
        """
        Create and return appropriate database adapter.
        
        Returns cached instance if available, otherwise creates new.
        
        Args:
            db_type: Database type (defaults to settings.DATABASE_TYPE)
            **kwargs: Additional adapter configuration
                - database_url: Custom connection URL
                - connection_url: MongoDB connection URL
                - database_name: MongoDB database name
                
        Returns:
            Database adapter instance
            
        Raises:
            ValueError: If database type is not supported
        """
        db_type = db_type or settings.DATABASE_TYPE
        
        # Return cached instance if available
        if db_type in cls._instances:
            return cls._instances[db_type]
        
        # Create new adapter based on type
        adapter: BaseDatabaseAdapter
        
        if db_type == DatabaseType.SQLITE:
            adapter = SQLiteAdapter(
                database_url=kwargs.get("database_url")
            )
            logger.info("Created SQLite adapter")
        
        elif db_type == DatabaseType.POSTGRESQL:
            adapter = PostgreSQLAdapter(
                database_url=kwargs.get("database_url")
            )
            logger.info("Created PostgreSQL adapter")
        
        elif db_type == DatabaseType.MONGODB:
            adapter = MongoDBAdapter(
                connection_url=kwargs.get("connection_url"),
                database_name=kwargs.get("database_name"),
            )
            logger.info("Created MongoDB adapter")
        
        else:
            raise ValueError(f"Unsupported database type: {db_type}")
        
        cls._instances[db_type] = adapter
        return adapter
    
    @classmethod
    async def initialize(
        cls,
        db_type: Optional[DatabaseType] = None,
    ) -> BaseDatabaseAdapter:
        """
        Initialize database connection.
        
        Creates adapter and establishes database connection.
        Should be called at application startup.
        
        Args:
            db_type: Database type (defaults to settings.DATABASE_TYPE)
            
        Returns:
            Initialized database adapter
            
        Raises:
            DatabaseError: If connection fails
        """
        adapter = cls.create_adapter(db_type)
        
        try:
            await adapter.connect()
            
            # Register models for SQL adapters
            db_type = db_type or settings.DATABASE_TYPE
            if db_type in [DatabaseType.SQLITE, DatabaseType.POSTGRESQL]:
                cls._register_models(adapter)
            
            logger.info(
                f"Database initialized: {db_type or settings.DATABASE_TYPE}"
            )
            return adapter
        except Exception as e:
            logger.error(f"Database initialization failed: {e}")
            raise DatabaseError(f"Failed to initialize database: {e}")
    
    @classmethod
    def _register_models(cls, adapter: BaseDatabaseAdapter) -> None:
        """Register all domain models with the adapter."""
        try:
            from app.domain_models.user import User
            from app.domain_models.chat import ChatSession, ChatMessage
            from app.domain_models.transaction import Transaction
            from app.domain_models.product import Product
            from app.domain_models.order import Order, OrderItem
            from app.domain_models.project import Project, Task
            from app.domain_models.course import Course, CourseSection, Enrollment
            
            # Register models
            adapter.register_model("users", User)
            adapter.register_model("chat_sessions", ChatSession)
            adapter.register_model("chat_messages", ChatMessage)
            adapter.register_model("transactions", Transaction)
            adapter.register_model("products", Product)
            adapter.register_model("orders", Order)
            adapter.register_model("order_items", OrderItem)
            adapter.register_model("projects", Project)
            adapter.register_model("tasks", Task)
            adapter.register_model("courses", Course)
            adapter.register_model("course_sections", CourseSection)
            adapter.register_model("enrollments", Enrollment)
            
            logger.info("Registered all domain models with adapter")
        except Exception as e:
            logger.warning(f"Could not register models: {e}")
    
    @classmethod
    async def shutdown(cls) -> None:
        """
        Close all database connections.
        
        Releases all resources and clears adapter cache.
        Should be called at application shutdown.
        """
        for db_type, adapter in cls._instances.items():
            try:
                await adapter.disconnect()
                logger.info(f"Disconnected: {db_type}")
            except Exception as e:
                logger.error(f"Error disconnecting {db_type}: {e}")
        
        cls._instances.clear()
        logger.info("All database connections closed")
    
    @classmethod
    def get_adapter(
        cls,
        db_type: Optional[DatabaseType] = None,
    ) -> BaseDatabaseAdapter:
        """
        Get existing adapter instance.
        
        Args:
            db_type: Database type (defaults to settings.DATABASE_TYPE)
            
        Returns:
            Initialized adapter instance
            
        Raises:
            RuntimeError: If adapter not initialized
        """
        db_type = db_type or settings.DATABASE_TYPE
        
        if db_type not in cls._instances:
            raise RuntimeError(
                f"Database adapter for {db_type} not initialized. "
                f"Call DatabaseFactory.initialize() first."
            )
        
        return cls._instances[db_type]
    
    @classmethod
    def is_initialized(
        cls,
        db_type: Optional[DatabaseType] = None,
    ) -> bool:
        """
        Check if adapter is initialized.
        
        Args:
            db_type: Database type to check
            
        Returns:
            True if adapter exists in cache
        """
        db_type = db_type or settings.DATABASE_TYPE
        return db_type in cls._instances
    
    @classmethod
    async def health_check(
        cls,
        db_type: Optional[DatabaseType] = None,
    ) -> bool:
        """
        Check database health.
        
        Args:
            db_type: Database type to check
            
        Returns:
            True if database is healthy
        """
        try:
            adapter = cls.get_adapter(db_type)
            return await adapter.health_check()
        except Exception:
            return False
    
    @classmethod
    def reset(cls) -> None:
        """
        Reset factory state.
        
        Clears adapter cache without disconnecting.
        Primarily for testing purposes.
        """
        cls._instances.clear()
