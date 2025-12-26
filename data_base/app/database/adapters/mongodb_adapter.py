# ==============================================================================
# MONGODB ADAPTER - Motor Async Driver Implementation
# ==============================================================================
# Document-oriented database adapter with full async support
# Uses Motor for non-blocking MongoDB operations
# ==============================================================================

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Dict, List, Optional

from bson import ObjectId
from motor.motor_asyncio import (
    AsyncIOMotorClient,
    AsyncIOMotorClientSession,
    AsyncIOMotorDatabase,
)
from pymongo import ASCENDING, DESCENDING

from app.core.settings import settings
from app.core.exceptions import DatabaseError
from app.database.adapters.base_adapter import BaseDatabaseAdapter

logger = logging.getLogger(__name__)


class MongoDBAdapter(BaseDatabaseAdapter[Dict[str, Any]]):
    """
    MongoDB database adapter using Motor async driver.
    
    Provides full async support for document-oriented operations
    with automatic ObjectId serialization and transaction support.
    
    Features:
        - Async MongoDB operations using Motor
        - Automatic ObjectId <-> string conversion
        - Transaction support with session context
        - Aggregation pipeline support for complex queries
        
    Attributes:
        _connection_url: MongoDB connection string
        _database_name: Target database name
        _client: Motor async client
        _database: Target database instance
        
    Example:
        >>> adapter = MongoDBAdapter()
        >>> await adapter.connect()
        >>> doc = await adapter.create("users", {"email": "test@example.com"})
        >>> print(doc["id"])  # String ID
    """
    
    def __init__(
        self,
        connection_url: Optional[str] = None,
        database_name: Optional[str] = None,
    ) -> None:
        """
        Initialize MongoDB adapter.
        
        Args:
            connection_url: MongoDB connection URI (defaults to settings)
            database_name: Database name (defaults to settings)
        """
        self._connection_url = connection_url or settings.MONGODB_URL
        self._database_name = database_name or settings.MONGODB_DB
        self._client: Optional[AsyncIOMotorClient] = None
        self._database: Optional[AsyncIOMotorDatabase] = None
    
    # ==========================================================================
    # ID SERIALIZATION HELPERS
    # ==========================================================================
    
    @staticmethod
    def _serialize_id(document: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert MongoDB ObjectId to string for serialization.
        
        Transforms _id to id and converts ObjectId to string.
        
        Args:
            document: MongoDB document with _id
            
        Returns:
            Document with string id field
        """
        if document and "_id" in document:
            document["id"] = str(document.pop("_id"))
        return document
    
    @staticmethod
    def _deserialize_id(id_value: Any) -> ObjectId:
        """
        Convert string ID to MongoDB ObjectId.
        
        Args:
            id_value: String or ObjectId
            
        Returns:
            ObjectId instance
        """
        if isinstance(id_value, str):
            return ObjectId(id_value)
        return id_value
    
    def _build_query(
        self,
        filters: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Build MongoDB query from filter dictionary.
        
        Handles special cases like id -> _id conversion and
        operator dictionaries ($gt, $lt, $in, etc.).
        
        Args:
            filters: Filter dictionary
            
        Returns:
            MongoDB query dictionary
        """
        if not filters:
            return {}
        
        query = {}
        for key, value in filters.items():
            if key == "id":
                query["_id"] = self._deserialize_id(value)
            elif isinstance(value, dict):
                # Handle operators like $gt, $lt, $in, etc.
                query[key] = value
            else:
                query[key] = value
        
        return query
    
    # ==========================================================================
    # LIFECYCLE METHODS
    # ==========================================================================
    
    async def connect(self) -> None:
        """
        Initialize MongoDB connection.
        
        Creates Motor client and selects target database.
        """
        try:
            self._client = AsyncIOMotorClient(
                self._connection_url,
                maxPoolSize=settings.DB_POOL_SIZE,
                minPoolSize=1,
                maxIdleTimeMS=settings.DB_POOL_TIMEOUT * 1000,
            )
            self._database = self._client[self._database_name]
            
            # Verify connection
            await self._client.admin.command("ping")
            
            logger.info(
                f"MongoDB adapter connected to {self._database_name}"
            )
            
        except Exception as e:
            logger.error(f"Failed to connect to MongoDB: {e}")
            raise DatabaseError(f"MongoDB connection failed: {e}")
    
    async def disconnect(self) -> None:
        """Close MongoDB connection."""
        if self._client:
            self._client.close()
            logger.info("MongoDB adapter disconnected")
    
    async def health_check(self) -> bool:
        """
        Verify database connectivity.
        
        Returns:
            True if connection is healthy
        """
        try:
            if self._client:
                await self._client.admin.command("ping")
                return True
            return False
        except Exception as e:
            logger.warning(f"MongoDB health check failed: {e}")
            return False
    
    # ==========================================================================
    # SESSION MANAGEMENT
    # ==========================================================================
    
    @asynccontextmanager
    async def session(self) -> AsyncIterator[AsyncIOMotorClientSession]:
        """
        Provide transactional session scope.
        
        MongoDB transactions are supported on replica sets.
        Falls back gracefully for standalone instances.
        
        Yields:
            AsyncIOMotorClientSession instance
        """
        if not self._client:
            raise RuntimeError(
                "Database not connected. Call connect() first."
            )
        
        async with await self._client.start_session() as session:
            try:
                async with session.start_transaction():
                    yield session
            except Exception:
                # Transaction will be aborted automatically
                raise
    
    # ==========================================================================
    # CRUD OPERATIONS
    # ==========================================================================
    
    async def create(
        self,
        collection: str,
        data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Create a new document."""
        if not self._database:
            raise RuntimeError("Database not connected")
        
        # Remove id if present (MongoDB will generate _id)
        data = {k: v for k, v in data.items() if k != "id"}
        
        result = await self._database[collection].insert_one(data)
        data["id"] = str(result.inserted_id)
        return data
    
    async def get_by_id(
        self,
        collection: str,
        id: Any,
    ) -> Optional[Dict[str, Any]]:
        """Retrieve document by ID."""
        if not self._database:
            raise RuntimeError("Database not connected")
        
        document = await self._database[collection].find_one(
            {"_id": self._deserialize_id(id)}
        )
        return self._serialize_id(document) if document else None
    
    async def get_all(
        self,
        collection: str,
        skip: int = 0,
        limit: int = 100,
        filters: Optional[Dict[str, Any]] = None,
        sort_by: Optional[str] = None,
        sort_order: str = "asc",
    ) -> List[Dict[str, Any]]:
        """Retrieve multiple documents with pagination."""
        if not self._database:
            raise RuntimeError("Database not connected")
        
        query = self._build_query(filters)
        cursor = self._database[collection].find(query)
        
        # Apply sorting
        if sort_by:
            direction = ASCENDING if sort_order.lower() == "asc" else DESCENDING
            cursor = cursor.sort(sort_by, direction)
        
        # Apply pagination
        cursor = cursor.skip(skip).limit(limit)
        
        documents = await cursor.to_list(length=limit)
        return [self._serialize_id(doc) for doc in documents]
    
    async def update(
        self,
        collection: str,
        id: Any,
        data: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """Update an existing document."""
        if not self._database:
            raise RuntimeError("Database not connected")
        
        # Remove id from update data
        data = {k: v for k, v in data.items() if k != "id"}
        
        result = await self._database[collection].find_one_and_update(
            {"_id": self._deserialize_id(id)},
            {"$set": data},
            return_document=True,
        )
        return self._serialize_id(result) if result else None
    
    async def delete(
        self,
        collection: str,
        id: Any,
    ) -> bool:
        """Delete a document by ID."""
        if not self._database:
            raise RuntimeError("Database not connected")
        
        result = await self._database[collection].delete_one(
            {"_id": self._deserialize_id(id)}
        )
        return result.deleted_count > 0
    
    # ==========================================================================
    # QUERY OPERATIONS
    # ==========================================================================
    
    async def count(
        self,
        collection: str,
        filters: Optional[Dict[str, Any]] = None,
    ) -> int:
        """Count documents matching filters."""
        if not self._database:
            raise RuntimeError("Database not connected")
        
        query = self._build_query(filters)
        return await self._database[collection].count_documents(query)
    
    async def exists(
        self,
        collection: str,
        filters: Dict[str, Any],
    ) -> bool:
        """Check if any document matches filters."""
        count = await self.count(collection, filters)
        return count > 0
    
    async def find_one(
        self,
        collection: str,
        filters: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """Find a single document matching filters."""
        if not self._database:
            raise RuntimeError("Database not connected")
        
        query = self._build_query(filters)
        document = await self._database[collection].find_one(query)
        return self._serialize_id(document) if document else None
    
    # ==========================================================================
    # BULK OPERATIONS
    # ==========================================================================
    
    async def bulk_create(
        self,
        collection: str,
        data: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Bulk insert documents."""
        if not self._database:
            raise RuntimeError("Database not connected")
        
        # Clean data (remove id fields)
        clean_data = [
            {k: v for k, v in item.items() if k != "id"}
            for item in data
        ]
        
        result = await self._database[collection].insert_many(clean_data)
        
        # Add IDs to returned documents
        for i, inserted_id in enumerate(result.inserted_ids):
            data[i]["id"] = str(inserted_id)
        
        return data
    
    async def bulk_update(
        self,
        collection: str,
        filters: Dict[str, Any],
        data: Dict[str, Any],
    ) -> int:
        """Bulk update documents matching filters."""
        if not self._database:
            raise RuntimeError("Database not connected")
        
        query = self._build_query(filters)
        result = await self._database[collection].update_many(
            query,
            {"$set": data},
        )
        return result.modified_count
    
    async def bulk_delete(
        self,
        collection: str,
        filters: Dict[str, Any],
    ) -> int:
        """Bulk delete documents matching filters."""
        if not self._database:
            raise RuntimeError("Database not connected")
        
        query = self._build_query(filters)
        result = await self._database[collection].delete_many(query)
        return result.deleted_count
    
    # ==========================================================================
    # RAW QUERY EXECUTION
    # ==========================================================================
    
    async def execute_raw(
        self,
        query: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """
        Execute aggregation pipeline.
        
        For MongoDB, query is the collection name and params
        should contain the aggregation pipeline.
        
        Args:
            query: Collection name
            params: Dict containing "pipeline" key with aggregation stages
            
        Returns:
            List of aggregation results
        """
        if not self._database:
            raise RuntimeError("Database not connected")
        
        pipeline = params.get("pipeline", []) if params else []
        cursor = self._database[query].aggregate(pipeline)
        results = await cursor.to_list(length=None)
        return [self._serialize_id(doc) for doc in results]
