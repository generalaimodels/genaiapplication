# ==============================================================================
# BASE DATABASE ADAPTER - Abstract Interface
# ==============================================================================
# Defines the contract for all database adapters
# Ensures consistent API across SQLite, PostgreSQL, MongoDB
# ==============================================================================

from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from typing import (
    Any,
    AsyncIterator,
    Dict,
    Generic,
    List,
    Optional,
    TypeVar,
)

# Type variable for generic database records
T = TypeVar("T")


class BaseDatabaseAdapter(ABC, Generic[T]):
    """
    Abstract Base Class for Database Adapters.
    
    Provides a unified interface for CRUD operations across different
    database backends. All concrete adapters must implement these methods
    to ensure consistent behavior.
    
    Generic Parameters:
        T: The type of records returned by the adapter
        
    Design Pattern:
        Implements the Adapter Pattern to provide a uniform interface
        for heterogeneous database systems.
        
    Thread Safety:
        All methods are async and designed for concurrent access.
        Connection pooling is handled by the underlying driver.
        
    Example:
        >>> adapter = PostgreSQLAdapter()
        >>> await adapter.connect()
        >>> user = await adapter.create("users", {"email": "test@example.com"})
        >>> await adapter.disconnect()
    """
    
    # ==========================================================================
    # LIFECYCLE METHODS
    # ==========================================================================
    
    @abstractmethod
    async def connect(self) -> None:
        """
        Establish database connection.
        
        Initializes the database engine/client and connection pool.
        Must be called before any database operations.
        
        Raises:
            ConnectionError: If connection cannot be established
        """
        pass
    
    @abstractmethod
    async def disconnect(self) -> None:
        """
        Close database connection.
        
        Releases all connections in the pool and cleans up resources.
        Should be called when shutting down the application.
        """
        pass
    
    @abstractmethod
    async def health_check(self) -> bool:
        """
        Check if database connection is healthy.
        
        Performs a lightweight query to verify connectivity.
        
        Returns:
            True if connection is healthy, False otherwise
        """
        pass
    
    # ==========================================================================
    # SESSION MANAGEMENT
    # ==========================================================================
    
    @abstractmethod
    @asynccontextmanager
    async def session(self) -> AsyncIterator[Any]:
        """
        Provide a transactional session scope.
        
        Creates a new session/transaction context. Changes are committed
        on successful exit or rolled back on exception.
        
        Yields:
            Session object appropriate for the database type
            
        Raises:
            RuntimeError: If database is not connected
            
        Example:
            >>> async with adapter.session() as session:
            ...     # Perform operations within transaction
            ...     await session.execute(query)
        """
        pass
    
    # ==========================================================================
    # CRUD OPERATIONS
    # ==========================================================================
    
    @abstractmethod
    async def create(
        self,
        collection: str,
        data: Dict[str, Any],
    ) -> T:
        """
        Create a new record.
        
        Args:
            collection: Table/collection name
            data: Record data as dictionary
            
        Returns:
            Created record with generated ID
            
        Raises:
            DatabaseError: If creation fails
            ValidationError: If data is invalid
        """
        pass
    
    @abstractmethod
    async def get_by_id(
        self,
        collection: str,
        id: Any,
    ) -> Optional[T]:
        """
        Retrieve a record by its primary identifier.
        
        Args:
            collection: Table/collection name
            id: Primary key value (UUID, ObjectId, int)
            
        Returns:
            Record if found, None otherwise
        """
        pass
    
    @abstractmethod
    async def get_all(
        self,
        collection: str,
        skip: int = 0,
        limit: int = 100,
        filters: Optional[Dict[str, Any]] = None,
        sort_by: Optional[str] = None,
        sort_order: str = "asc",
    ) -> List[T]:
        """
        Retrieve multiple records with pagination and filtering.
        
        Args:
            collection: Table/collection name
            skip: Number of records to skip (offset)
            limit: Maximum number of records to return
            filters: Field-value pairs for filtering
            sort_by: Field name to sort by
            sort_order: Sort direction ("asc" or "desc")
            
        Returns:
            List of matching records
        """
        pass
    
    @abstractmethod
    async def update(
        self,
        collection: str,
        id: Any,
        data: Dict[str, Any],
    ) -> Optional[T]:
        """
        Update an existing record.
        
        Args:
            collection: Table/collection name
            id: Primary key of record to update
            data: Fields to update (partial update supported)
            
        Returns:
            Updated record if found, None if not exists
        """
        pass
    
    @abstractmethod
    async def delete(
        self,
        collection: str,
        id: Any,
    ) -> bool:
        """
        Delete a record by ID.
        
        Args:
            collection: Table/collection name
            id: Primary key of record to delete
            
        Returns:
            True if deleted, False if not found
        """
        pass
    
    # ==========================================================================
    # QUERY OPERATIONS
    # ==========================================================================
    
    @abstractmethod
    async def count(
        self,
        collection: str,
        filters: Optional[Dict[str, Any]] = None,
    ) -> int:
        """
        Count records matching filters.
        
        Args:
            collection: Table/collection name
            filters: Field-value pairs for filtering
            
        Returns:
            Number of matching records
        """
        pass
    
    @abstractmethod
    async def exists(
        self,
        collection: str,
        filters: Dict[str, Any],
    ) -> bool:
        """
        Check if any record matches the filters.
        
        Args:
            collection: Table/collection name
            filters: Field-value pairs for filtering
            
        Returns:
            True if at least one record matches
        """
        pass
    
    @abstractmethod
    async def find_one(
        self,
        collection: str,
        filters: Dict[str, Any],
    ) -> Optional[T]:
        """
        Find a single record matching filters.
        
        Args:
            collection: Table/collection name
            filters: Field-value pairs for filtering
            
        Returns:
            First matching record, None if no match
        """
        pass
    
    # ==========================================================================
    # BULK OPERATIONS
    # ==========================================================================
    
    @abstractmethod
    async def bulk_create(
        self,
        collection: str,
        data: List[Dict[str, Any]],
    ) -> List[T]:
        """
        Bulk insert multiple records.
        
        Args:
            collection: Table/collection name
            data: List of record dictionaries
            
        Returns:
            List of created records with IDs
        """
        pass
    
    @abstractmethod
    async def bulk_update(
        self,
        collection: str,
        filters: Dict[str, Any],
        data: Dict[str, Any],
    ) -> int:
        """
        Bulk update records matching filters.
        
        Args:
            collection: Table/collection name
            filters: Field-value pairs for matching records
            data: Fields to update
            
        Returns:
            Number of records updated
        """
        pass
    
    @abstractmethod
    async def bulk_delete(
        self,
        collection: str,
        filters: Dict[str, Any],
    ) -> int:
        """
        Bulk delete records matching filters.
        
        Args:
            collection: Table/collection name
            filters: Field-value pairs for matching records
            
        Returns:
            Number of records deleted
        """
        pass
    
    # ==========================================================================
    # RAW QUERY EXECUTION
    # ==========================================================================
    
    @abstractmethod
    async def execute_raw(
        self,
        query: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """
        Execute a raw query for complex operations.
        
        Use with caution - bypasses the abstraction layer.
        
        Args:
            query: Raw query string (SQL or aggregation pipeline)
            params: Query parameters
            
        Returns:
            Query results (format varies by database)
        """
        pass
