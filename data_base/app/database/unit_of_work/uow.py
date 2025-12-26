# ==============================================================================
# UNIT OF WORK - Transaction Coordination
# ==============================================================================
# Manages transactional boundaries across multiple repositories
# Ensures atomic operations and data consistency
# ==============================================================================

from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from typing import Any, Dict, Optional, Type

from app.database.adapters.base_adapter import BaseDatabaseAdapter
from app.database.factory import DatabaseFactory
from app.database.repositories.base_repository import BaseRepository


class AbstractUnitOfWork(ABC):
    """
    Abstract Unit of Work pattern interface.
    
    Defines the contract for managing transactional boundaries
    and coordinating repository access.
    """
    
    @abstractmethod
    async def __aenter__(self) -> "AbstractUnitOfWork":
        """Enter transactional context."""
        pass
    
    @abstractmethod
    async def __aexit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Any,
    ) -> None:
        """Exit transactional context."""
        pass
    
    @abstractmethod
    async def commit(self) -> None:
        """Commit the transaction."""
        pass
    
    @abstractmethod
    async def rollback(self) -> None:
        """Rollback the transaction."""
        pass


class UnitOfWork(AbstractUnitOfWork):
    """
    Concrete Unit of Work implementation.
    
    Coordinates database transactions across multiple repositories,
    ensuring atomic operations and data consistency.
    
    Features:
        - Repository registration and retrieval
        - Automatic commit/rollback on context exit
        - Session sharing across repositories
        - Support for multiple database types
        
    Attributes:
        _adapter: Database adapter for operations
        _repositories: Registered repository instances
        _session: Active database session
        
    Example:
        >>> async with UnitOfWork() as uow:
        ...     uow.register_repository("users", UserRepository, "users")
        ...     repo = uow.get_repository("users")
        ...     await repo.create(user_data)
        ...     # Commits automatically on successful exit
    """
    
    def __init__(
        self,
        adapter: Optional[BaseDatabaseAdapter] = None,
    ) -> None:
        """
        Initialize Unit of Work.
        
        Args:
            adapter: Database adapter (defaults to factory adapter)
        """
        self._adapter = adapter
        self._repositories: Dict[str, BaseRepository] = {}
        self._session: Any = None
        self._is_active = False
    
    @property
    def adapter(self) -> BaseDatabaseAdapter:
        """Get database adapter, initializing if needed."""
        if self._adapter is None:
            self._adapter = DatabaseFactory.get_adapter()
        return self._adapter
    
    # ==========================================================================
    # REPOSITORY MANAGEMENT
    # ==========================================================================
    
    def register_repository(
        self,
        name: str,
        repository_class: Type[BaseRepository],
        collection_name: str,
    ) -> BaseRepository:
        """
        Register a repository for use within this unit of work.
        
        Args:
            name: Repository identifier for later retrieval
            repository_class: Repository class to instantiate
            collection_name: Database table/collection name
            
        Returns:
            Registered repository instance
        """
        repo = repository_class(
            adapter=self.adapter,
            collection_name=collection_name,
        )
        self._repositories[name] = repo
        return repo
    
    def get_repository(self, name: str) -> BaseRepository:
        """
        Get a registered repository.
        
        Args:
            name: Repository identifier
            
        Returns:
            Repository instance
            
        Raises:
            ValueError: If repository not registered
        """
        if name not in self._repositories:
            raise ValueError(
                f"Repository '{name}' not registered. "
                f"Available: {list(self._repositories.keys())}"
            )
        return self._repositories[name]
    
    def has_repository(self, name: str) -> bool:
        """Check if repository is registered."""
        return name in self._repositories
    
    # ==========================================================================
    # CONTEXT MANAGEMENT
    # ==========================================================================
    
    async def __aenter__(self) -> "UnitOfWork":
        """
        Enter transactional context.
        
        Starts a new database session/transaction.
        
        Returns:
            Self for context manager usage
        """
        self._session = self.adapter.session()
        await self._session.__aenter__()
        self._is_active = True
        return self
    
    async def __aexit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Any,
    ) -> None:
        """
        Exit transactional context.
        
        Commits on successful exit, rolls back on exception.
        
        Args:
            exc_type: Exception type if error occurred
            exc_val: Exception value if error occurred
            exc_tb: Exception traceback if error occurred
        """
        if exc_type:
            await self.rollback()
        
        if self._session:
            await self._session.__aexit__(exc_type, exc_val, exc_tb)
        
        self._is_active = False
    
    async def commit(self) -> None:
        """
        Commit the transaction.
        
        Note: In the current implementation, commit is handled
        automatically by the session context manager.
        """
        # Session auto-commits on successful context exit
        pass
    
    async def rollback(self) -> None:
        """
        Rollback the transaction.
        
        Note: In the current implementation, rollback is handled
        automatically by the session context manager on exception.
        """
        # Session auto-rollbacks on exception
        pass
    
    @property
    def is_active(self) -> bool:
        """Check if unit of work has an active session."""
        return self._is_active


@asynccontextmanager
async def get_unit_of_work():
    """
    Dependency injection helper for Unit of Work.
    
    Creates and manages a Unit of Work instance within
    an async context manager.
    
    Yields:
        UnitOfWork instance
        
    Example:
        >>> async with get_unit_of_work() as uow:
        ...     # Perform operations
        ...     pass
    """
    uow = UnitOfWork()
    async with uow:
        yield uow
