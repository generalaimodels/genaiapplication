# ==============================================================================
# BASE SERVICE - Generic Business Logic Layer
# ==============================================================================
# Abstract service providing common CRUD operations
# ==============================================================================

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Generic, List, Optional, TypeVar
from uuid import UUID

from pydantic import BaseModel

from app.database.adapters.base_adapter import BaseDatabaseAdapter
from app.core.exceptions import NotFoundError

# Type variables for generic service
EntityType = TypeVar("EntityType")
CreateSchemaType = TypeVar("CreateSchemaType", bound=BaseModel)
UpdateSchemaType = TypeVar("UpdateSchemaType", bound=BaseModel)
ResponseSchemaType = TypeVar("ResponseSchemaType", bound=BaseModel)


class BaseService(ABC, Generic[EntityType, CreateSchemaType, UpdateSchemaType, ResponseSchemaType]):
    """
    Abstract base service providing standard business operations.
    
    Encapsulates business logic and database operations for a
    specific domain entity, providing a clean interface for
    API endpoints.
    
    Generic Parameters:
        EntityType: Domain entity/model type
        CreateSchemaType: Pydantic schema for creation
        UpdateSchemaType: Pydantic schema for updates
        ResponseSchemaType: Pydantic schema for responses
        
    Attributes:
        _adapter: Database adapter for operations
        _collection_name: Table/collection identifier
        
    Example:
        >>> class UserService(BaseService[User, UserCreate, UserUpdate, UserResponse]):
        ...     def _to_response(self, entity):
        ...         return UserResponse.model_validate(entity)
    """
    
    def __init__(
        self,
        adapter: BaseDatabaseAdapter,
        collection_name: str,
    ) -> None:
        """
        Initialize service.
        
        Args:
            adapter: Database adapter instance
            collection_name: Table/collection name
        """
        self._adapter = adapter
        self._collection_name = collection_name
    
    # ==========================================================================
    # ABSTRACT METHODS
    # ==========================================================================
    
    @abstractmethod
    def _to_response(self, entity: Any) -> ResponseSchemaType:
        """
        Convert entity to response schema.
        
        Args:
            entity: Database entity/record
            
        Returns:
            Response schema instance
        """
        pass
    
    # ==========================================================================
    # CRUD OPERATIONS
    # ==========================================================================
    
    async def create(self, schema: CreateSchemaType) -> ResponseSchemaType:
        """
        Create a new entity.
        
        Args:
            schema: Creation schema with entity data
            
        Returns:
            Created entity as response schema
        """
        data = schema.model_dump(exclude_unset=True)
        result = await self._adapter.create(self._collection_name, data)
        return self._to_response(result)
    
    async def get_by_id(self, id: Any) -> ResponseSchemaType:
        """
        Retrieve entity by ID.
        
        Args:
            id: Entity primary key
            
        Returns:
            Entity as response schema
            
        Raises:
            NotFoundError: If entity not found
        """
        result = await self._adapter.get_by_id(self._collection_name, id)
        if not result:
            raise NotFoundError(
                message=f"{self._collection_name} not found",
                resource_type=self._collection_name,
                resource_id=id,
            )
        return self._to_response(result)
    
    async def get_all(
        self,
        skip: int = 0,
        limit: int = 100,
        filters: Optional[Dict[str, Any]] = None,
        sort_by: Optional[str] = None,
        sort_order: str = "asc",
    ) -> List[ResponseSchemaType]:
        """
        Retrieve multiple entities with pagination.
        
        Args:
            skip: Records to skip
            limit: Maximum records
            filters: Filter criteria
            sort_by: Sort field
            sort_order: Sort direction
            
        Returns:
            List of entities as response schemas
        """
        results = await self._adapter.get_all(
            self._collection_name,
            skip=skip,
            limit=limit,
            filters=filters,
            sort_by=sort_by,
            sort_order=sort_order,
        )
        return [self._to_response(r) for r in results]
    
    async def update(
        self,
        id: Any,
        schema: UpdateSchemaType,
    ) -> ResponseSchemaType:
        """
        Update an existing entity.
        
        Args:
            id: Entity primary key
            schema: Update schema with changes
            
        Returns:
            Updated entity as response schema
            
        Raises:
            NotFoundError: If entity not found
        """
        data = schema.model_dump(exclude_unset=True)
        result = await self._adapter.update(self._collection_name, id, data)
        if not result:
            raise NotFoundError(
                message=f"{self._collection_name} not found",
                resource_type=self._collection_name,
                resource_id=id,
            )
        return self._to_response(result)
    
    async def delete(self, id: Any) -> bool:
        """
        Delete an entity by ID.
        
        Args:
            id: Entity primary key
            
        Returns:
            True if deleted
            
        Raises:
            NotFoundError: If entity not found
        """
        deleted = await self._adapter.delete(self._collection_name, id)
        if not deleted:
            raise NotFoundError(
                message=f"{self._collection_name} not found",
                resource_type=self._collection_name,
                resource_id=id,
            )
        return True
    
    # ==========================================================================
    # QUERY OPERATIONS
    # ==========================================================================
    
    async def count(
        self,
        filters: Optional[Dict[str, Any]] = None,
    ) -> int:
        """Count entities matching filters."""
        return await self._adapter.count(self._collection_name, filters)
    
    async def exists(self, id: Any) -> bool:
        """Check if entity exists."""
        result = await self._adapter.get_by_id(self._collection_name, id)
        return result is not None
    
    async def find_one(
        self,
        filters: Dict[str, Any],
    ) -> Optional[ResponseSchemaType]:
        """Find single entity by filters."""
        result = await self._adapter.find_one(self._collection_name, filters)
        return self._to_response(result) if result else None
    
    # ==========================================================================
    # PAGINATION HELPERS
    # ==========================================================================
    
    async def get_paginated(
        self,
        page: int = 1,
        page_size: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        sort_by: Optional[str] = None,
        sort_order: str = "asc",
    ) -> Dict[str, Any]:
        """
        Get paginated results with metadata.
        
        Args:
            page: Page number (1-indexed)
            page_size: Items per page
            filters: Filter criteria
            sort_by: Sort field
            sort_order: Sort direction
            
        Returns:
            Dict with items, total, page, page_size, pages
        """
        skip = (page - 1) * page_size
        
        items = await self.get_all(
            skip=skip,
            limit=page_size,
            filters=filters,
            sort_by=sort_by,
            sort_order=sort_order,
        )
        total = await self.count(filters)
        pages = (total + page_size - 1) // page_size
        
        return {
            "items": items,
            "total": total,
            "page": page,
            "page_size": page_size,
            "pages": pages,
        }
