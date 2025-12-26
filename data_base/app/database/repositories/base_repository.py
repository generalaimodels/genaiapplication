# ==============================================================================
# BASE REPOSITORY - Generic Data Access Abstraction
# ==============================================================================
# Repository Pattern implementation for consistent data access
# Works with both SQL and NoSQL database adapters
# ==============================================================================

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import (
    Any,
    Dict,
    Generic,
    List,
    Optional,
    TypeVar,
)
from uuid import UUID

from pydantic import BaseModel

from app.database.adapters.base_adapter import BaseDatabaseAdapter

# Type variables for generic repository
ModelType = TypeVar("ModelType")
CreateSchemaType = TypeVar("CreateSchemaType", bound=BaseModel)
UpdateSchemaType = TypeVar("UpdateSchemaType", bound=BaseModel)


class BaseRepository(ABC, Generic[ModelType, CreateSchemaType, UpdateSchemaType]):
    """
    Abstract base repository providing standard CRUD operations.
    
    Implements the Repository Pattern for data access abstraction,
    decoupling business logic from database implementation details.
    
    Generic Parameters:
        ModelType: Domain entity/model type
        CreateSchemaType: Pydantic schema for entity creation
        UpdateSchemaType: Pydantic schema for entity updates
        
    Design Pattern:
        Repository Pattern - abstracts data access logic
        
    Attributes:
        _adapter: Database adapter for database operations
        _collection_name: Table/collection identifier
        
    Example:
        >>> class UserRepository(BaseRepository[User, UserCreate, UserUpdate]):
        ...     def _to_entity(self, data):
        ...         return User(**data) if isinstance(data, dict) else data
        ...
        >>> repo = UserRepository(adapter, "users")
        >>> user = await repo.create(UserCreate(email="test@example.com"))
    """
    
    def __init__(
        self,
        adapter: BaseDatabaseAdapter,
        collection_name: str,
    ) -> None:
        """
        Initialize repository.
        
        Args:
            adapter: Database adapter instance
            collection_name: Table/collection name for operations
        """
        self._adapter = adapter
        self._collection_name = collection_name
    
    # ==========================================================================
    # ABSTRACT METHODS - Must be implemented by subclasses
    # ==========================================================================
    
    @abstractmethod
    def _to_entity(self, data: Any) -> ModelType:
        """
        Convert database record to domain entity.
        
        Args:
            data: Raw database record (dict or ORM model)
            
        Returns:
            Domain entity instance
        """
        pass
    
    def _to_dict(self, entity: ModelType) -> Dict[str, Any]:
        """
        Convert domain entity to dictionary.
        
        Default implementation handles Pydantic models and
        SQLAlchemy models. Override for custom behavior.
        
        Args:
            entity: Domain entity instance
            
        Returns:
            Dictionary representation
        """
        if hasattr(entity, "model_dump"):
            # Pydantic model
            return entity.model_dump(exclude_unset=True)
        elif hasattr(entity, "__dict__"):
            # SQLAlchemy model or regular object
            return {
                k: v for k, v in entity.__dict__.items()
                if not k.startswith("_")
            }
        return dict(entity)
    
    # ==========================================================================
    # CRUD OPERATIONS
    # ==========================================================================
    
    async def create(self, schema: CreateSchemaType) -> ModelType:
        """
        Create a new entity.
        
        Args:
            schema: Pydantic schema with entity data
            
        Returns:
            Created entity with generated ID
        """
        data = schema.model_dump(exclude_unset=True)
        result = await self._adapter.create(self._collection_name, data)
        return self._to_entity(result)
    
    async def get_by_id(self, id: Any) -> Optional[ModelType]:
        """
        Retrieve entity by ID.
        
        Args:
            id: Primary key value
            
        Returns:
            Entity if found, None otherwise
        """
        result = await self._adapter.get_by_id(self._collection_name, id)
        return self._to_entity(result) if result else None
    
    async def get_all(
        self,
        skip: int = 0,
        limit: int = 100,
        filters: Optional[Dict[str, Any]] = None,
        sort_by: Optional[str] = None,
        sort_order: str = "asc",
    ) -> List[ModelType]:
        """
        Retrieve multiple entities with pagination.
        
        Args:
            skip: Number of records to skip
            limit: Maximum records to return
            filters: Field-value pairs for filtering
            sort_by: Field to sort by
            sort_order: Sort direction ("asc" or "desc")
            
        Returns:
            List of matching entities
        """
        results = await self._adapter.get_all(
            self._collection_name,
            skip=skip,
            limit=limit,
            filters=filters,
            sort_by=sort_by,
            sort_order=sort_order,
        )
        return [self._to_entity(r) for r in results]
    
    async def update(
        self,
        id: Any,
        schema: UpdateSchemaType,
    ) -> Optional[ModelType]:
        """
        Update an existing entity.
        
        Args:
            id: Primary key of entity to update
            schema: Pydantic schema with fields to update
            
        Returns:
            Updated entity if found, None otherwise
        """
        data = schema.model_dump(exclude_unset=True)
        result = await self._adapter.update(self._collection_name, id, data)
        return self._to_entity(result) if result else None
    
    async def delete(self, id: Any) -> bool:
        """
        Delete an entity by ID.
        
        Args:
            id: Primary key of entity to delete
            
        Returns:
            True if deleted, False if not found
        """
        return await self._adapter.delete(self._collection_name, id)
    
    # ==========================================================================
    # QUERY OPERATIONS
    # ==========================================================================
    
    async def count(
        self,
        filters: Optional[Dict[str, Any]] = None,
    ) -> int:
        """
        Count entities matching filters.
        
        Args:
            filters: Field-value pairs for filtering
            
        Returns:
            Number of matching entities
        """
        return await self._adapter.count(self._collection_name, filters)
    
    async def exists(self, id: Any) -> bool:
        """
        Check if entity exists by ID.
        
        Args:
            id: Primary key to check
            
        Returns:
            True if entity exists
        """
        result = await self.get_by_id(id)
        return result is not None
    
    async def find_one(
        self,
        filters: Dict[str, Any],
    ) -> Optional[ModelType]:
        """
        Find a single entity matching filters.
        
        Args:
            filters: Field-value pairs for matching
            
        Returns:
            First matching entity, None if not found
        """
        result = await self._adapter.find_one(self._collection_name, filters)
        return self._to_entity(result) if result else None
    
    # ==========================================================================
    # BULK OPERATIONS
    # ==========================================================================
    
    async def bulk_create(
        self,
        schemas: List[CreateSchemaType],
    ) -> List[ModelType]:
        """
        Bulk create entities.
        
        Args:
            schemas: List of Pydantic schemas
            
        Returns:
            List of created entities
        """
        data = [s.model_dump(exclude_unset=True) for s in schemas]
        results = await self._adapter.bulk_create(self._collection_name, data)
        return [self._to_entity(r) for r in results]
    
    async def bulk_update(
        self,
        filters: Dict[str, Any],
        data: Dict[str, Any],
    ) -> int:
        """
        Bulk update entities matching filters.
        
        Args:
            filters: Field-value pairs for matching
            data: Fields to update
            
        Returns:
            Number of entities updated
        """
        return await self._adapter.bulk_update(
            self._collection_name,
            filters,
            data,
        )
    
    async def bulk_delete(
        self,
        filters: Dict[str, Any],
    ) -> int:
        """
        Bulk delete entities matching filters.
        
        Args:
            filters: Field-value pairs for matching
            
        Returns:
            Number of entities deleted
        """
        return await self._adapter.bulk_delete(self._collection_name, filters)
