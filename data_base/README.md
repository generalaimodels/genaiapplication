# Generalized Multi-Database Backend System with FastAPI

## Complete End-to-End Architecture Plan (SOTA)

---

## 1. System Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           CLIENT APPLICATIONS                                │
│         (Web App / Mobile App / Chatbot / Admin Dashboard)                  │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              API GATEWAY                                     │
│                    (Rate Limiting / Auth / Load Balancer)                   │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         FASTAPI APPLICATION LAYER                           │
├─────────────────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐    │
│  │   Routers    │  │  Middleware  │  │   Services   │  │  Validators  │    │
│  └──────────────┘  └──────────────┘  └──────────────┘  └──────────────┘    │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    DATABASE ABSTRACTION LAYER (DAL)                         │
├─────────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    Repository Pattern Interface                      │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                      Unit of Work Pattern                            │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                    ┌─────────────────┼─────────────────┐
                    ▼                 ▼                 ▼
            ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
            │   SQLite     │  │  PostgreSQL  │  │   MongoDB    │
            │   Adapter    │  │   Adapter    │  │   Adapter    │
            └──────────────┘  └──────────────┘  └──────────────┘
                    │                 │                 │
                    ▼                 ▼                 ▼
            ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
            │   SQLite     │  │  PostgreSQL  │  │   MongoDB    │
            │   Database   │  │   Database   │  │   Database   │
            └──────────────┘  └──────────────┘  └──────────────┘
```

---

## 2. Project Structure

```
project_root/
│
├── app/
│   ├── __init__.py
│   ├── main.py                          # FastAPI application entry point
│   ├── config.py                        # Configuration management
│   │
│   ├── core/
│   │   ├── __init__.py
│   │   ├── settings.py                  # Environment settings
│   │   ├── security.py                  # Authentication/Authorization
│   │   ├── exceptions.py                # Custom exceptions
│   │   └── constants.py                 # Application constants
│   │
│   ├── database/
│   │   ├── __init__.py
│   │   ├── base.py                      # Base database interface
│   │   ├── connection_manager.py        # Connection pooling & management
│   │   ├── factory.py                   # Database factory pattern
│   │   │
│   │   ├── adapters/
│   │   │   ├── __init__.py
│   │   │   ├── base_adapter.py          # Abstract adapter interface
│   │   │   ├── sqlite_adapter.py        # SQLite implementation
│   │   │   ├── postgresql_adapter.py    # PostgreSQL implementation
│   │   │   └── mongodb_adapter.py       # MongoDB implementation
│   │   │
│   │   ├── repositories/
│   │   │   ├── __init__.py
│   │   │   ├── base_repository.py       # Generic repository interface
│   │   │   ├── sql_repository.py        # SQL-based repository
│   │   │   └── nosql_repository.py      # NoSQL-based repository
│   │   │
│   │   └── unit_of_work/
│   │       ├── __init__.py
│   │       └── uow.py                   # Unit of Work implementation
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── base.py                      # Base model definitions
│   │   ├── sql_models/                  # SQLAlchemy models
│   │   │   ├── __init__.py
│   │   │   ├── user.py
│   │   │   ├── chat.py
│   │   │   ├── transaction.py
│   │   │   ├── product.py
│   │   │   ├── project.py
│   │   │   └── course.py
│   │   │
│   │   └── nosql_models/                # MongoDB document models
│   │       ├── __init__.py
│   │       └── documents.py
│   │
│   ├── schemas/
│   │   ├── __init__.py
│   │   ├── base.py                      # Base Pydantic schemas
│   │   ├── user.py
│   │   ├── chat.py
│   │   ├── transaction.py
│   │   ├── product.py
│   │   ├── project.py
│   │   └── course.py
│   │
│   ├── api/
│   │   ├── __init__.py
│   │   ├── deps.py                      # Dependency injection
│   │   │
│   │   └── v1/
│   │       ├── __init__.py
│   │       ├── router.py                # Main API router
│   │       │
│   │       └── endpoints/
│   │           ├── __init__.py
│   │           ├── auth.py
│   │           ├── users.py
│   │           ├── chat.py              # Chatbot endpoints
│   │           ├── transactions.py      # Banking endpoints
│   │           ├── products.py          # E-commerce endpoints
│   │           ├── projects.py          # Project management endpoints
│   │           └── courses.py           # LMS endpoints
│   │
│   ├── services/
│   │   ├── __init__.py
│   │   ├── base_service.py              # Generic service layer
│   │   ├── user_service.py
│   │   ├── chat_service.py
│   │   ├── transaction_service.py
│   │   ├── product_service.py
│   │   ├── project_service.py
│   │   └── course_service.py
│   │
│   ├── middleware/
│   │   ├── __init__.py
│   │   ├── logging.py
│   │   ├── error_handler.py
│   │   └── rate_limiter.py
│   │
│   └── utils/
│       ├── __init__.py
│       ├── helpers.py
│       └── validators.py
│
├── migrations/
│   └── versions/
│
├── tests/
│   ├── __init__.py
│   ├── conftest.py
│   ├── unit/
│   └── integration/
│
├── alembic.ini
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── .env
```

---

## 3. Core Configuration System

### 3.1 Settings Configuration

```python
# app/core/settings.py

from pydantic_settings import BaseSettings
from typing import Optional, Literal
from functools import lru_cache
from enum import Enum


class DatabaseType(str, Enum):
    SQLITE = "sqlite"
    POSTGRESQL = "postgresql"
    MONGODB = "mongodb"


class Settings(BaseSettings):
    # Application
    APP_NAME: str = "Generalized Database System"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    ENVIRONMENT: Literal["development", "staging", "production"] = "development"
    
    # API
    API_V1_PREFIX: str = "/api/v1"
    
    # Database Configuration
    DATABASE_TYPE: DatabaseType = DatabaseType.POSTGRESQL
    
    # SQLite
    SQLITE_URL: str = "sqlite:///./app.db"
    
    # PostgreSQL
    POSTGRES_HOST: str = "localhost"
    POSTGRES_PORT: int = 5432
    POSTGRES_USER: str = "postgres"
    POSTGRES_PASSWORD: str = "password"
    POSTGRES_DB: str = "app_db"
    
    # MongoDB
    MONGODB_URL: str = "mongodb://localhost:27017"
    MONGODB_DB: str = "app_db"
    
    # Connection Pool
    DB_POOL_SIZE: int = 10
    DB_MAX_OVERFLOW: int = 20
    DB_POOL_TIMEOUT: int = 30
    
    # Security
    SECRET_KEY: str = "your-secret-key"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7
    
    # Rate Limiting
    RATE_LIMIT_REQUESTS: int = 100
    RATE_LIMIT_WINDOW: int = 60
    
    @property
    def postgres_url(self) -> str:
        return (
            f"postgresql+asyncpg://{self.POSTGRES_USER}:"
            f"{self.POSTGRES_PASSWORD}@{self.POSTGRES_HOST}:"
            f"{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
        )
    
    @property
    def database_url(self) -> str:
        if self.DATABASE_TYPE == DatabaseType.SQLITE:
            return self.SQLITE_URL.replace("sqlite://", "sqlite+aiosqlite://")
        elif self.DATABASE_TYPE == DatabaseType.POSTGRESQL:
            return self.postgres_url
        elif self.DATABASE_TYPE == DatabaseType.MONGODB:
            return self.MONGODB_URL
        raise ValueError(f"Unsupported database type: {self.DATABASE_TYPE}")
    
    class Config:
        env_file = ".env"
        case_sensitive = True


@lru_cache()
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
```

---

## 4. Database Abstraction Layer (DAL)

### 4.1 Abstract Base Adapter

```python
# app/database/adapters/base_adapter.py

from abc import ABC, abstractmethod
from typing import TypeVar, Generic, List, Optional, Dict, Any
from contextlib import asynccontextmanager

T = TypeVar('T')


class BaseDatabaseAdapter(ABC, Generic[T]):
    """
    Abstract base class for all database adapters.
    Provides unified interface for CRUD operations across different databases.
    """
    
    @abstractmethod
    async def connect(self) -> None:
        """Establish database connection."""
        pass
    
    @abstractmethod
    async def disconnect(self) -> None:
        """Close database connection."""
        pass
    
    @abstractmethod
    @asynccontextmanager
    async def session(self):
        """Provide a transactional session scope."""
        pass
    
    @abstractmethod
    async def create(self, collection: str, data: Dict[str, Any]) -> T:
        """Create a new record."""
        pass
    
    @abstractmethod
    async def get_by_id(self, collection: str, id: Any) -> Optional[T]:
        """Retrieve a record by ID."""
        pass
    
    @abstractmethod
    async def get_all(
        self,
        collection: str,
        skip: int = 0,
        limit: int = 100,
        filters: Optional[Dict[str, Any]] = None,
        sort_by: Optional[str] = None,
        sort_order: str = "asc"
    ) -> List[T]:
        """Retrieve multiple records with pagination and filtering."""
        pass
    
    @abstractmethod
    async def update(
        self,
        collection: str,
        id: Any,
        data: Dict[str, Any]
    ) -> Optional[T]:
        """Update an existing record."""
        pass
    
    @abstractmethod
    async def delete(self, collection: str, id: Any) -> bool:
        """Delete a record by ID."""
        pass
    
    @abstractmethod
    async def count(
        self,
        collection: str,
        filters: Optional[Dict[str, Any]] = None
    ) -> int:
        """Count records matching filters."""
        pass
    
    @abstractmethod
    async def execute_raw(self, query: str, params: Optional[Dict] = None) -> Any:
        """Execute raw query for complex operations."""
        pass
    
    @abstractmethod
    async def bulk_create(
        self,
        collection: str,
        data: List[Dict[str, Any]]
    ) -> List[T]:
        """Bulk insert records."""
        pass
    
    @abstractmethod
    async def bulk_update(
        self,
        collection: str,
        filters: Dict[str, Any],
        data: Dict[str, Any]
    ) -> int:
        """Bulk update records matching filters."""
        pass
    
    @abstractmethod
    async def bulk_delete(
        self,
        collection: str,
        filters: Dict[str, Any]
    ) -> int:
        """Bulk delete records matching filters."""
        pass
```

### 4.2 PostgreSQL Adapter

```python
# app/database/adapters/postgresql_adapter.py

from typing import List, Optional, Dict, Any, Type
from contextlib import asynccontextmanager
from sqlalchemy.ext.asyncio import (
    create_async_engine,
    AsyncSession,
    async_sessionmaker,
    AsyncEngine
)
from sqlalchemy import select, update, delete, func, text, and_
from sqlalchemy.orm import DeclarativeBase

from .base_adapter import BaseDatabaseAdapter
from app.core.settings import settings


class PostgreSQLAdapter(BaseDatabaseAdapter):
    """PostgreSQL database adapter using SQLAlchemy async."""
    
    def __init__(self, database_url: Optional[str] = None):
        self._database_url = database_url or settings.postgres_url
        self._engine: Optional[AsyncEngine] = None
        self._session_factory: Optional[async_sessionmaker] = None
        self._model_registry: Dict[str, Type[DeclarativeBase]] = {}
    
    def register_model(self, name: str, model: Type[DeclarativeBase]) -> None:
        """Register a model for table mapping."""
        self._model_registry[name] = model
    
    def _get_model(self, collection: str) -> Type[DeclarativeBase]:
        """Get registered model by collection name."""
        if collection not in self._model_registry:
            raise ValueError(f"Model '{collection}' not registered")
        return self._model_registry[collection]
    
    async def connect(self) -> None:
        """Initialize database engine and session factory."""
        self._engine = create_async_engine(
            self._database_url,
            pool_size=settings.DB_POOL_SIZE,
            max_overflow=settings.DB_MAX_OVERFLOW,
            pool_timeout=settings.DB_POOL_TIMEOUT,
            echo=settings.DEBUG
        )
        
        self._session_factory = async_sessionmaker(
            bind=self._engine,
            class_=AsyncSession,
            expire_on_commit=False,
            autoflush=False
        )
    
    async def disconnect(self) -> None:
        """Close database connections."""
        if self._engine:
            await self._engine.dispose()
    
    @asynccontextmanager
    async def session(self):
        """Provide transactional session scope."""
        if not self._session_factory:
            raise RuntimeError("Database not connected. Call connect() first.")
        
        session: AsyncSession = self._session_factory()
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()
    
    async def create(self, collection: str, data: Dict[str, Any]) -> Any:
        """Create a new record."""
        model = self._get_model(collection)
        
        async with self.session() as session:
            instance = model(**data)
            session.add(instance)
            await session.flush()
            await session.refresh(instance)
            return instance
    
    async def get_by_id(self, collection: str, id: Any) -> Optional[Any]:
        """Retrieve record by ID."""
        model = self._get_model(collection)
        
        async with self.session() as session:
            result = await session.get(model, id)
            return result
    
    async def get_all(
        self,
        collection: str,
        skip: int = 0,
        limit: int = 100,
        filters: Optional[Dict[str, Any]] = None,
        sort_by: Optional[str] = None,
        sort_order: str = "asc"
    ) -> List[Any]:
        """Retrieve multiple records with pagination."""
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
        data: Dict[str, Any]
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
    
    async def delete(self, collection: str, id: Any) -> bool:
        """Delete a record by ID."""
        model = self._get_model(collection)
        
        async with self.session() as session:
            instance = await session.get(model, id)
            if not instance:
                return False
            
            await session.delete(instance)
            return True
    
    async def count(
        self,
        collection: str,
        filters: Optional[Dict[str, Any]] = None
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
    
    async def execute_raw(self, query: str, params: Optional[Dict] = None) -> Any:
        """Execute raw SQL query."""
        async with self.session() as session:
            result = await session.execute(text(query), params or {})
            return result.fetchall()
    
    async def bulk_create(
        self,
        collection: str,
        data: List[Dict[str, Any]]
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
        data: Dict[str, Any]
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
        filters: Dict[str, Any]
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
```

### 4.3 MongoDB Adapter

```python
# app/database/adapters/mongodb_adapter.py

from typing import List, Optional, Dict, Any
from contextlib import asynccontextmanager
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from bson import ObjectId
from pymongo import ASCENDING, DESCENDING

from .base_adapter import BaseDatabaseAdapter
from app.core.settings import settings


class MongoDBAdapter(BaseDatabaseAdapter):
    """MongoDB database adapter using Motor async driver."""
    
    def __init__(
        self,
        connection_url: Optional[str] = None,
        database_name: Optional[str] = None
    ):
        self._connection_url = connection_url or settings.MONGODB_URL
        self._database_name = database_name or settings.MONGODB_DB
        self._client: Optional[AsyncIOMotorClient] = None
        self._database: Optional[AsyncIOMotorDatabase] = None
    
    async def connect(self) -> None:
        """Initialize MongoDB connection."""
        self._client = AsyncIOMotorClient(self._connection_url)
        self._database = self._client[self._database_name]
    
    async def disconnect(self) -> None:
        """Close MongoDB connection."""
        if self._client:
            self._client.close()
    
    @asynccontextmanager
    async def session(self):
        """Provide transactional session scope (MongoDB session)."""
        if not self._client:
            raise RuntimeError("Database not connected. Call connect() first.")
        
        async with await self._client.start_session() as session:
            async with session.start_transaction():
                yield session
    
    def _serialize_id(self, document: Dict[str, Any]) -> Dict[str, Any]:
        """Convert ObjectId to string for serialization."""
        if document and "_id" in document:
            document["id"] = str(document.pop("_id"))
        return document
    
    def _deserialize_id(self, id_value: Any) -> ObjectId:
        """Convert string ID to ObjectId."""
        if isinstance(id_value, str):
            return ObjectId(id_value)
        return id_value
    
    def _build_query(self, filters: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Build MongoDB query from filters."""
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
    
    async def create(self, collection: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new document."""
        result = await self._database[collection].insert_one(data)
        data["id"] = str(result.inserted_id)
        return data
    
    async def get_by_id(
        self,
        collection: str,
        id: Any
    ) -> Optional[Dict[str, Any]]:
        """Retrieve document by ID."""
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
        sort_order: str = "asc"
    ) -> List[Dict[str, Any]]:
        """Retrieve multiple documents with pagination."""
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
        data: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Update an existing document."""
        result = await self._database[collection].find_one_and_update(
            {"_id": self._deserialize_id(id)},
            {"$set": data},
            return_document=True
        )
        return self._serialize_id(result) if result else None
    
    async def delete(self, collection: str, id: Any) -> bool:
        """Delete a document by ID."""
        result = await self._database[collection].delete_one(
            {"_id": self._deserialize_id(id)}
        )
        return result.deleted_count > 0
    
    async def count(
        self,
        collection: str,
        filters: Optional[Dict[str, Any]] = None
    ) -> int:
        """Count documents matching filters."""
        query = self._build_query(filters)
        return await self._database[collection].count_documents(query)
    
    async def execute_raw(self, query: str, params: Optional[Dict] = None) -> Any:
        """Execute aggregation pipeline."""
        # For MongoDB, query is treated as collection name
        # params contains the aggregation pipeline
        pipeline = params.get("pipeline", []) if params else []
        cursor = self._database[query].aggregate(pipeline)
        return await cursor.to_list(length=None)
    
    async def bulk_create(
        self,
        collection: str,
        data: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Bulk insert documents."""
        result = await self._database[collection].insert_many(data)
        
        for i, inserted_id in enumerate(result.inserted_ids):
            data[i]["id"] = str(inserted_id)
        
        return data
    
    async def bulk_update(
        self,
        collection: str,
        filters: Dict[str, Any],
        data: Dict[str, Any]
    ) -> int:
        """Bulk update documents matching filters."""
        query = self._build_query(filters)
        result = await self._database[collection].update_many(
            query,
            {"$set": data}
        )
        return result.modified_count
    
    async def bulk_delete(
        self,
        collection: str,
        filters: Dict[str, Any]
    ) -> int:
        """Bulk delete documents matching filters."""
        query = self._build_query(filters)
        result = await self._database[collection].delete_many(query)
        return result.deleted_count
```

### 4.4 SQLite Adapter

```python
# app/database/adapters/sqlite_adapter.py

from typing import List, Optional, Dict, Any, Type
from contextlib import asynccontextmanager
from sqlalchemy.ext.asyncio import (
    create_async_engine,
    AsyncSession,
    async_sessionmaker,
    AsyncEngine
)
from sqlalchemy import select, update, delete, func, text, and_
from sqlalchemy.orm import DeclarativeBase

from .base_adapter import BaseDatabaseAdapter
from app.core.settings import settings


class SQLiteAdapter(BaseDatabaseAdapter):
    """SQLite database adapter using SQLAlchemy async with aiosqlite."""
    
    def __init__(self, database_url: Optional[str] = None):
        self._database_url = database_url or settings.SQLITE_URL.replace(
            "sqlite://", "sqlite+aiosqlite://"
        )
        self._engine: Optional[AsyncEngine] = None
        self._session_factory: Optional[async_sessionmaker] = None
        self._model_registry: Dict[str, Type[DeclarativeBase]] = {}
    
    def register_model(self, name: str, model: Type[DeclarativeBase]) -> None:
        """Register a model for table mapping."""
        self._model_registry[name] = model
    
    def _get_model(self, collection: str) -> Type[DeclarativeBase]:
        """Get registered model by collection name."""
        if collection not in self._model_registry:
            raise ValueError(f"Model '{collection}' not registered")
        return self._model_registry[collection]
    
    async def connect(self) -> None:
        """Initialize database engine."""
        self._engine = create_async_engine(
            self._database_url,
            echo=settings.DEBUG
        )
        
        self._session_factory = async_sessionmaker(
            bind=self._engine,
            class_=AsyncSession,
            expire_on_commit=False,
            autoflush=False
        )
        
        # Create tables
        async with self._engine.begin() as conn:
            from app.models.base import SQLBase
            await conn.run_sync(SQLBase.metadata.create_all)
    
    async def disconnect(self) -> None:
        """Close database connections."""
        if self._engine:
            await self._engine.dispose()
    
    @asynccontextmanager
    async def session(self):
        """Provide transactional session scope."""
        if not self._session_factory:
            raise RuntimeError("Database not connected")
        
        session: AsyncSession = self._session_factory()
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()
    
    # Remaining methods are identical to PostgreSQLAdapter
    # ... (same implementation as PostgreSQL for CRUD operations)
```

### 4.5 Database Factory

```python
# app/database/factory.py

from typing import Optional
from enum import Enum

from app.core.settings import settings, DatabaseType
from app.database.adapters.base_adapter import BaseDatabaseAdapter
from app.database.adapters.postgresql_adapter import PostgreSQLAdapter
from app.database.adapters.mongodb_adapter import MongoDBAdapter
from app.database.adapters.sqlite_adapter import SQLiteAdapter


class DatabaseFactory:
    """
    Factory class for creating database adapters.
    Implements Factory Pattern for database abstraction.
    """
    
    _instances: dict[DatabaseType, BaseDatabaseAdapter] = {}
    
    @classmethod
    def create_adapter(
        cls,
        db_type: Optional[DatabaseType] = None,
        **kwargs
    ) -> BaseDatabaseAdapter:
        """
        Create and return appropriate database adapter.
        
        Args:
            db_type: Type of database (sqlite, postgresql, mongodb)
            **kwargs: Additional configuration parameters
            
        Returns:
            Database adapter instance
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
        
        elif db_type == DatabaseType.POSTGRESQL:
            adapter = PostgreSQLAdapter(
                database_url=kwargs.get("database_url")
            )
        
        elif db_type == DatabaseType.MONGODB:
            adapter = MongoDBAdapter(
                connection_url=kwargs.get("connection_url"),
                database_name=kwargs.get("database_name")
            )
        
        else:
            raise ValueError(f"Unsupported database type: {db_type}")
        
        cls._instances[db_type] = adapter
        return adapter
    
    @classmethod
    async def initialize(cls, db_type: Optional[DatabaseType] = None) -> None:
        """Initialize database connection."""
        adapter = cls.create_adapter(db_type)
        await adapter.connect()
    
    @classmethod
    async def shutdown(cls) -> None:
        """Close all database connections."""
        for adapter in cls._instances.values():
            await adapter.disconnect()
        cls._instances.clear()
    
    @classmethod
    def get_adapter(
        cls,
        db_type: Optional[DatabaseType] = None
    ) -> BaseDatabaseAdapter:
        """Get existing adapter instance."""
        db_type = db_type or settings.DATABASE_TYPE
        
        if db_type not in cls._instances:
            raise RuntimeError(
                f"Database adapter for {db_type} not initialized"
            )
        
        return cls._instances[db_type]
```

---

## 5. Repository Pattern Implementation

### 5.1 Base Repository

```python
# app/database/repositories/base_repository.py

from abc import ABC, abstractmethod
from typing import TypeVar, Generic, List, Optional, Dict, Any
from pydantic import BaseModel

from app.database.adapters.base_adapter import BaseDatabaseAdapter

ModelType = TypeVar("ModelType")
CreateSchemaType = TypeVar("CreateSchemaType", bound=BaseModel)
UpdateSchemaType = TypeVar("UpdateSchemaType", bound=BaseModel)


class BaseRepository(ABC, Generic[ModelType, CreateSchemaType, UpdateSchemaType]):
    """
    Abstract base repository providing standard CRUD operations.
    Implements Repository Pattern for data access abstraction.
    """
    
    def __init__(self, adapter: BaseDatabaseAdapter, collection_name: str):
        self._adapter = adapter
        self._collection_name = collection_name
    
    @abstractmethod
    def _to_entity(self, data: Dict[str, Any]) -> ModelType:
        """Convert database record to domain entity."""
        pass
    
    @abstractmethod
    def _to_dict(self, entity: ModelType) -> Dict[str, Any]:
        """Convert domain entity to dictionary."""
        pass
    
    async def create(self, schema: CreateSchemaType) -> ModelType:
        """Create a new entity."""
        data = schema.model_dump(exclude_unset=True)
        result = await self._adapter.create(self._collection_name, data)
        return self._to_entity(result) if isinstance(result, dict) else result
    
    async def get_by_id(self, id: Any) -> Optional[ModelType]:
        """Retrieve entity by ID."""
        result = await self._adapter.get_by_id(self._collection_name, id)
        return self._to_entity(result) if result else None
    
    async def get_all(
        self,
        skip: int = 0,
        limit: int = 100,
        filters: Optional[Dict[str, Any]] = None,
        sort_by: Optional[str] = None,
        sort_order: str = "asc"
    ) -> List[ModelType]:
        """Retrieve multiple entities with pagination."""
        results = await self._adapter.get_all(
            self._collection_name,
            skip=skip,
            limit=limit,
            filters=filters,
            sort_by=sort_by,
            sort_order=sort_order
        )
        return [self._to_entity(r) if isinstance(r, dict) else r for r in results]
    
    async def update(self, id: Any, schema: UpdateSchemaType) -> Optional[ModelType]:
        """Update an existing entity."""
        data = schema.model_dump(exclude_unset=True)
        result = await self._adapter.update(self._collection_name, id, data)
        return self._to_entity(result) if result else None
    
    async def delete(self, id: Any) -> bool:
        """Delete an entity by ID."""
        return await self._adapter.delete(self._collection_name, id)
    
    async def count(self, filters: Optional[Dict[str, Any]] = None) -> int:
        """Count entities matching filters."""
        return await self._adapter.count(self._collection_name, filters)
    
    async def exists(self, id: Any) -> bool:
        """Check if entity exists."""
        result = await self.get_by_id(id)
        return result is not None
    
    async def bulk_create(
        self,
        schemas: List[CreateSchemaType]
    ) -> List[ModelType]:
        """Bulk create entities."""
        data = [s.model_dump(exclude_unset=True) for s in schemas]
        results = await self._adapter.bulk_create(self._collection_name, data)
        return [self._to_entity(r) if isinstance(r, dict) else r for r in results]
```

### 5.2 Unit of Work Pattern

```python
# app/database/unit_of_work/uow.py

from abc import ABC, abstractmethod
from typing import Optional, Type
from contextlib import asynccontextmanager

from app.database.adapters.base_adapter import BaseDatabaseAdapter
from app.database.factory import DatabaseFactory
from app.database.repositories.base_repository import BaseRepository


class AbstractUnitOfWork(ABC):
    """
    Abstract Unit of Work pattern implementation.
    Manages transactional boundaries and repository access.
    """
    
    @abstractmethod
    async def __aenter__(self):
        pass
    
    @abstractmethod
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass
    
    @abstractmethod
    async def commit(self):
        pass
    
    @abstractmethod
    async def rollback(self):
        pass


class UnitOfWork(AbstractUnitOfWork):
    """
    Concrete Unit of Work implementation.
    Coordinates database transactions across multiple repositories.
    """
    
    def __init__(self, adapter: Optional[BaseDatabaseAdapter] = None):
        self._adapter = adapter or DatabaseFactory.get_adapter()
        self._repositories: dict[str, BaseRepository] = {}
        self._session = None
    
    def register_repository(
        self,
        name: str,
        repository_class: Type[BaseRepository],
        collection_name: str
    ) -> None:
        """Register a repository for use within this unit of work."""
        self._repositories[name] = repository_class(
            adapter=self._adapter,
            collection_name=collection_name
        )
    
    def get_repository(self, name: str) -> BaseRepository:
        """Get a registered repository."""
        if name not in self._repositories:
            raise ValueError(f"Repository '{name}' not registered")
        return self._repositories[name]
    
    async def __aenter__(self):
        """Enter transactional context."""
        self._session = self._adapter.session()
        await self._session.__aenter__()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit transactional context."""
        if exc_type:
            await self.rollback()
        await self._session.__aexit__(exc_type, exc_val, exc_tb)
    
    async def commit(self):
        """Commit the transaction."""
        if self._session:
            # Session auto-commits on successful exit
            pass
    
    async def rollback(self):
        """Rollback the transaction."""
        # Handled by session context manager
        pass


@asynccontextmanager
async def get_unit_of_work():
    """Dependency for getting a Unit of Work instance."""
    uow = UnitOfWork()
    async with uow:
        yield uow
```

---

## 6. Domain Models

### 6.1 SQL Models (SQLAlchemy)

```python
# app/models/sql_models/base.py

from datetime import datetime
from typing import Optional
from sqlalchemy import DateTime, String, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class SQLBase(DeclarativeBase):
    """Base class for all SQL models."""
    pass


class TimestampMixin:
    """Mixin for timestamp fields."""
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False
    )


# app/models/sql_models/user.py

import uuid
from sqlalchemy import String, Boolean
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.dialects.postgresql import UUID

from .base import SQLBase, TimestampMixin


class User(SQLBase, TimestampMixin):
    """User model for authentication and profile."""
    
    __tablename__ = "users"
    
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4
    )
    email: Mapped[str] = mapped_column(
        String(255),
        unique=True,
        nullable=False,
        index=True
    )
    hashed_password: Mapped[str] = mapped_column(String(255), nullable=False)
    full_name: Mapped[str] = mapped_column(String(255), nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    is_superuser: Mapped[bool] = mapped_column(Boolean, default=False)
    
    # Relationships
    chat_sessions = relationship("ChatSession", back_populates="user")
    transactions = relationship("Transaction", back_populates="user")
    orders = relationship("Order", back_populates="user")


# app/models/sql_models/chat.py

import uuid
from sqlalchemy import String, Text, ForeignKey, Enum
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.dialects.postgresql import UUID
import enum

from .base import SQLBase, TimestampMixin


class MessageRole(str, enum.Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class ChatSession(SQLBase, TimestampMixin):
    """Chat session model for conversation management."""
    
    __tablename__ = "chat_sessions"
    
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id"),
        nullable=False
    )
    title: Mapped[str] = mapped_column(String(255), nullable=True)
    is_active: Mapped[bool] = mapped_column(default=True)
    
    # Relationships
    user = relationship("User", back_populates="chat_sessions")
    messages = relationship(
        "ChatMessage",
        back_populates="session",
        order_by="ChatMessage.created_at"
    )


class ChatMessage(SQLBase, TimestampMixin):
    """Individual chat message model."""
    
    __tablename__ = "chat_messages"
    
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4
    )
    session_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("chat_sessions.id"),
        nullable=False
    )
    role: Mapped[MessageRole] = mapped_column(
        Enum(MessageRole),
        nullable=False
    )
    content: Mapped[str] = mapped_column(Text, nullable=False)
    metadata_: Mapped[Optional[str]] = mapped_column(
        "metadata",
        Text,
        nullable=True
    )
    
    # Relationships
    session = relationship("ChatSession", back_populates="messages")


# app/models/sql_models/transaction.py

import uuid
from decimal import Decimal
from sqlalchemy import String, Numeric, ForeignKey, Enum
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.dialects.postgresql import UUID
import enum

from .base import SQLBase, TimestampMixin


class TransactionType(str, enum.Enum):
    CREDIT = "credit"
    DEBIT = "debit"
    TRANSFER = "transfer"


class TransactionStatus(str, enum.Enum):
    PENDING = "pending"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class Transaction(SQLBase, TimestampMixin):
    """Banking transaction model."""
    
    __tablename__ = "transactions"
    
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id"),
        nullable=False
    )
    transaction_type: Mapped[TransactionType] = mapped_column(
        Enum(TransactionType),
        nullable=False
    )
    amount: Mapped[Decimal] = mapped_column(
        Numeric(precision=18, scale=2),
        nullable=False
    )
    currency: Mapped[str] = mapped_column(String(3), default="USD")
    status: Mapped[TransactionStatus] = mapped_column(
        Enum(TransactionStatus),
        default=TransactionStatus.PENDING
    )
    reference_id: Mapped[str] = mapped_column(String(100), unique=True)
    description: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    
    # Relationships
    user = relationship("User", back_populates="transactions")


# app/models/sql_models/product.py

import uuid
from decimal import Decimal
from sqlalchemy import String, Text, Numeric, Integer, Boolean
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.dialects.postgresql import UUID, ARRAY

from .base import SQLBase, TimestampMixin


class Product(SQLBase, TimestampMixin):
    """E-commerce product model."""
    
    __tablename__ = "products"
    
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=True)
    price: Mapped[Decimal] = mapped_column(
        Numeric(precision=10, scale=2),
        nullable=False
    )
    stock_quantity: Mapped[int] = mapped_column(Integer, default=0)
    category: Mapped[str] = mapped_column(String(100), nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    sku: Mapped[str] = mapped_column(String(50), unique=True)
    
    # Relationships
    order_items = relationship("OrderItem", back_populates="product")


class Order(SQLBase, TimestampMixin):
    """E-commerce order model."""
    
    __tablename__ = "orders"
    
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id"),
        nullable=False
    )
    total_amount: Mapped[Decimal] = mapped_column(
        Numeric(precision=12, scale=2),
        nullable=False
    )
    status: Mapped[str] = mapped_column(String(50), default="pending")
    shipping_address: Mapped[str] = mapped_column(Text, nullable=True)
    
    # Relationships
    user = relationship("User", back_populates="orders")
    items = relationship("OrderItem", back_populates="order")


class OrderItem(SQLBase):
    """Order item linking orders and products."""
    
    __tablename__ = "order_items"
    
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4
    )
    order_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("orders.id"),
        nullable=False
    )
    product_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("products.id"),
        nullable=False
    )
    quantity: Mapped[int] = mapped_column(Integer, nullable=False)
    unit_price: Mapped[Decimal] = mapped_column(
        Numeric(precision=10, scale=2),
        nullable=False
    )
    
    # Relationships
    order = relationship("Order", back_populates="items")
    product = relationship("Product", back_populates="order_items")


# app/models/sql_models/project.py

import uuid
from datetime import date
from sqlalchemy import String, Text, Date, ForeignKey, Enum
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.dialects.postgresql import UUID
import enum

from .base import SQLBase, TimestampMixin


class ProjectStatus(str, enum.Enum):
    PLANNING = "planning"
    IN_PROGRESS = "in_progress"
    ON_HOLD = "on_hold"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class TaskPriority(str, enum.Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class Project(SQLBase, TimestampMixin):
    """Project management model."""
    
    __tablename__ = "projects"
    
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=True)
    owner_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id"),
        nullable=False
    )
    status: Mapped[ProjectStatus] = mapped_column(
        Enum(ProjectStatus),
        default=ProjectStatus.PLANNING
    )
    start_date: Mapped[date] = mapped_column(Date, nullable=True)
    end_date: Mapped[date] = mapped_column(Date, nullable=True)
    
    # Relationships
    tasks = relationship("Task", back_populates="project")


class Task(SQLBase, TimestampMixin):
    """Task model for project management."""
    
    __tablename__ = "tasks"
    
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4
    )
    project_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("projects.id"),
        nullable=False
    )
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=True)
    assignee_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id"),
        nullable=True
    )
    priority: Mapped[TaskPriority] = mapped_column(
        Enum(TaskPriority),
        default=TaskPriority.MEDIUM
    )
    due_date: Mapped[Optional[date]] = mapped_column(Date, nullable=True)
    is_completed: Mapped[bool] = mapped_column(default=False)
    
    # Relationships
    project = relationship("Project", back_populates="tasks")


# app/models/sql_models/course.py

import uuid
from sqlalchemy import String, Text, Integer, ForeignKey, Boolean
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.dialects.postgresql import UUID

from .base import SQLBase, TimestampMixin


class Course(SQLBase, TimestampMixin):
    """LMS Course model."""
    
    __tablename__ = "courses"
    
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4
    )
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=True)
    instructor_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id"),
        nullable=False
    )
    is_published: Mapped[bool] = mapped_column(Boolean, default=False)
    duration_hours: Mapped[int] = mapped_column(Integer, default=0)
    
    # Relationships
    sections = relationship("CourseSection", back_populates="course")
    enrollments = relationship("Enrollment", back_populates="course")


class CourseSection(SQLBase, TimestampMixin):
    """Course section/module model."""
    
    __tablename__ = "course_sections"
    
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4
    )
    course_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("courses.id"),
        nullable=False
    )
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=True)
    order: Mapped[int] = mapped_column(Integer, default=0)
    
    # Relationships
    course = relationship("Course", back_populates="sections")


class Enrollment(SQLBase, TimestampMixin):
    """Student enrollment model."""
    
    __tablename__ = "enrollments"
    
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id"),
        nullable=False
    )
    course_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("courses.id"),
        nullable=False
    )
    progress_percentage: Mapped[int] = mapped_column(Integer, default=0)
    is_completed: Mapped[bool] = mapped_column(Boolean, default=False)
    
    # Relationships
    course = relationship("Course", back_populates="enrollments")
```

---

## 7. Pydantic Schemas

```python
# app/schemas/base.py

from datetime import datetime
from typing import Optional, Generic, TypeVar, List
from pydantic import BaseModel, ConfigDict
from uuid import UUID

DataType = TypeVar("DataType")


class BaseSchema(BaseModel):
    """Base schema with common configuration."""
    model_config = ConfigDict(
        from_attributes=True,
        str_strip_whitespace=True,
        validate_assignment=True
    )


class TimestampSchema(BaseSchema):
    """Schema with timestamp fields."""
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class PaginatedResponse(BaseModel, Generic[DataType]):
    """Generic paginated response schema."""
    items: List[DataType]
    total: int
    page: int
    size: int
    pages: int


class APIResponse(BaseModel, Generic[DataType]):
    """Standard API response wrapper."""
    success: bool = True
    message: str = "Success"
    data: Optional[DataType] = None


# app/schemas/user.py

from typing import Optional
from pydantic import EmailStr
from uuid import UUID

from .base import BaseSchema, TimestampSchema


class UserBase(BaseSchema):
    email: EmailStr
    full_name: str


class UserCreate(UserBase):
    password: str


class UserUpdate(BaseSchema):
    email: Optional[EmailStr] = None
    full_name: Optional[str] = None
    password: Optional[str] = None
    is_active: Optional[bool] = None


class UserResponse(UserBase, TimestampSchema):
    id: UUID
    is_active: bool
    is_superuser: bool


# app/schemas/chat.py

from typing import Optional, List
from uuid import UUID
from enum import Enum

from .base import BaseSchema, TimestampSchema


class MessageRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class ChatMessageCreate(BaseSchema):
    role: MessageRole
    content: str
    metadata: Optional[dict] = None


class ChatMessageResponse(ChatMessageCreate, TimestampSchema):
    id: UUID
    session_id: UUID


class ChatSessionCreate(BaseSchema):
    title: Optional[str] = None


class ChatSessionResponse(TimestampSchema):
    id: UUID
    user_id: UUID
    title: Optional[str]
    is_active: bool
    messages: List[ChatMessageResponse] = []


class ChatSessionListResponse(TimestampSchema):
    id: UUID
    title: Optional[str]
    is_active: bool
    message_count: int = 0


# app/schemas/transaction.py

from decimal import Decimal
from typing import Optional
from uuid import UUID
from enum import Enum

from .base import BaseSchema, TimestampSchema


class TransactionType(str, Enum):
    CREDIT = "credit"
    DEBIT = "debit"
    TRANSFER = "transfer"


class TransactionStatus(str, Enum):
    PENDING = "pending"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TransactionCreate(BaseSchema):
    transaction_type: TransactionType
    amount: Decimal
    currency: str = "USD"
    description: Optional[str] = None


class TransactionUpdate(BaseSchema):
    status: Optional[TransactionStatus] = None
    description: Optional[str] = None


class TransactionResponse(TransactionCreate, TimestampSchema):
    id: UUID
    user_id: UUID
    status: TransactionStatus
    reference_id: str


# app/schemas/product.py

from decimal import Decimal
from typing import Optional
from uuid import UUID

from .base import BaseSchema, TimestampSchema


class ProductCreate(BaseSchema):
    name: str
    description: Optional[str] = None
    price: Decimal
    stock_quantity: int = 0
    category: Optional[str] = None
    sku: str


class ProductUpdate(BaseSchema):
    name: Optional[str] = None
    description: Optional[str] = None
    price: Optional[Decimal] = None
    stock_quantity: Optional[int] = None
    category: Optional[str] = None
    is_active: Optional[bool] = None


class ProductResponse(ProductCreate, TimestampSchema):
    id: UUID
    is_active: bool


# app/schemas/project.py

from datetime import date
from typing import Optional, List
from uuid import UUID
from enum import Enum

from .base import BaseSchema, TimestampSchema


class ProjectStatus(str, Enum):
    PLANNING = "planning"
    IN_PROGRESS = "in_progress"
    ON_HOLD = "on_hold"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class TaskPriority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class TaskCreate(BaseSchema):
    title: str
    description: Optional[str] = None
    assignee_id: Optional[UUID] = None
    priority: TaskPriority = TaskPriority.MEDIUM
    due_date: Optional[date] = None


class TaskUpdate(BaseSchema):
    title: Optional[str] = None
    description: Optional[str] = None
    assignee_id: Optional[UUID] = None
    priority: Optional[TaskPriority] = None
    due_date: Optional[date] = None
    is_completed: Optional[bool] = None


class TaskResponse(TaskCreate, TimestampSchema):
    id: UUID
    project_id: UUID
    is_completed: bool


class ProjectCreate(BaseSchema):
    name: str
    description: Optional[str] = None
    start_date: Optional[date] = None
    end_date: Optional[date] = None


class ProjectUpdate(BaseSchema):
    name: Optional[str] = None
    description: Optional[str] = None
    status: Optional[ProjectStatus] = None
    start_date: Optional[date] = None
    end_date: Optional[date] = None


class ProjectResponse(ProjectCreate, TimestampSchema):
    id: UUID
    owner_id: UUID
    status: ProjectStatus
    tasks: List[TaskResponse] = []


# app/schemas/course.py

from typing import Optional, List
from uuid import UUID

from .base import BaseSchema, TimestampSchema


class CourseSectionCreate(BaseSchema):
    title: str
    content: Optional[str] = None
    order: int = 0


class CourseSectionUpdate(BaseSchema):
    title: Optional[str] = None
    content: Optional[str] = None
    order: Optional[int] = None


class CourseSectionResponse(CourseSectionCreate, TimestampSchema):
    id: UUID
    course_id: UUID


class CourseCreate(BaseSchema):
    title: str
    description: Optional[str] = None
    duration_hours: int = 0


class CourseUpdate(BaseSchema):
    title: Optional[str] = None
    description: Optional[str] = None
    is_published: Optional[bool] = None
    duration_hours: Optional[int] = None


class CourseResponse(CourseCreate, TimestampSchema):
    id: UUID
    instructor_id: UUID
    is_published: bool
    sections: List[CourseSectionResponse] = []


class EnrollmentCreate(BaseSchema):
    course_id: UUID


class EnrollmentResponse(TimestampSchema):
    id: UUID
    user_id: UUID
    course_id: UUID
    progress_percentage: int
    is_completed: bool
```

---

## 8. Service Layer

```python
# app/services/base_service.py

from typing import TypeVar, Generic, List, Optional, Dict, Any
from uuid import UUID

from app.database.repositories.base_repository import BaseRepository
from app.schemas.base import PaginatedResponse

ModelType = TypeVar("ModelType")
CreateSchemaType = TypeVar("CreateSchemaType")
UpdateSchemaType = TypeVar("UpdateSchemaType")
ResponseSchemaType = TypeVar("ResponseSchemaType")


class BaseService(Generic[ModelType, CreateSchemaType, UpdateSchemaType, ResponseSchemaType]):
    """
    Generic service layer providing business logic abstraction.
    Implements common CRUD operations with validation and transformation.
    """
    
    def __init__(self, repository: BaseRepository):
        self._repository = repository
    
    async def create(self, data: CreateSchemaType) -> ResponseSchemaType:
        """Create a new entity."""
        result = await self._repository.create(data)
        return self._to_response(result)
    
    async def get_by_id(self, id: UUID) -> Optional[ResponseSchemaType]:
        """Retrieve entity by ID."""
        result = await self._repository.get_by_id(id)
        return self._to_response(result) if result else None
    
    async def get_paginated(
        self,
        page: int = 1,
        size: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        sort_by: Optional[str] = None,
        sort_order: str = "asc"
    ) -> PaginatedResponse[ResponseSchemaType]:
        """Retrieve paginated entities."""
        skip = (page - 1) * size
        
        items = await self._repository.get_all(
            skip=skip,
            limit=size,
            filters=filters,
            sort_by=sort_by,
            sort_order=sort_order
        )
        
        total = await self._repository.count(filters)
        pages = (total + size - 1) // size
        
        return PaginatedResponse(
            items=[self._to_response(item) for item in items],
            total=total,
            page=page,
            size=size,
            pages=pages
        )
    
    async def update(
        self,
        id: UUID,
        data: UpdateSchemaType
    ) -> Optional[ResponseSchemaType]:
        """Update an existing entity."""
        result = await self._repository.update(id, data)
        return self._to_response(result) if result else None
    
    async def delete(self, id: UUID) -> bool:
        """Delete an entity."""
        return await self._repository.delete(id)
    
    def _to_response(self, entity: ModelType) -> ResponseSchemaType:
        """Convert entity to response schema. Override in subclasses."""
        raise NotImplementedError


# app/services/chat_service.py

from typing import Optional, List
from uuid import UUID

from app.services.base_service import BaseService
from app.database.repositories.base_repository import BaseRepository
from app.schemas.chat import (
    ChatSessionCreate,
    ChatSessionResponse,
    ChatMessageCreate,
    ChatMessageResponse
)


class ChatService(BaseService):
    """Service for chatbot functionality with conversation history."""
    
    def __init__(
        self,
        session_repository: BaseRepository,
        message_repository: BaseRepository
    ):
        self._session_repo = session_repository
        self._message_repo = message_repository
    
    async def create_session(
        self,
        user_id: UUID,
        data: ChatSessionCreate
    ) -> ChatSessionResponse:
        """Create a new chat session."""
        session_data = data.model_dump()
        session_data["user_id"] = user_id
        
        result = await self._session_repo.create(
            type("Schema", (), session_data)()
        )
        return ChatSessionResponse.model_validate(result)
    
    async def get_user_sessions(
        self,
        user_id: UUID,
        skip: int = 0,
        limit: int = 20
    ) -> List[ChatSessionResponse]:
        """Get all sessions for a user."""
        sessions = await self._session_repo.get_all(
            skip=skip,
            limit=limit,
            filters={"user_id": user_id},
            sort_by="created_at",
            sort_order="desc"
        )
        return [ChatSessionResponse.model_validate(s) for s in sessions]
    
    async def add_message(
        self,
        session_id: UUID,
        data: ChatMessageCreate
    ) -> ChatMessageResponse:
        """Add a message to a session."""
        message_data = data.model_dump()
        message_data["session_id"] = session_id
        
        result = await self._message_repo.create(
            type("Schema", (), message_data)()
        )
        return ChatMessageResponse.model_validate(result)
    
    async def get_session_history(
        self,
        session_id: UUID,
        limit: int = 50
    ) -> List[ChatMessageResponse]:
        """Get message history for a session."""
        messages = await self._message_repo.get_all(
            filters={"session_id": session_id},
            sort_by="created_at",
            sort_order="asc",
            limit=limit
        )
        return [ChatMessageResponse.model_validate(m) for m in messages]


# app/services/transaction_service.py

from typing import Optional
from uuid import UUID, uuid4
from decimal import Decimal

from app.services.base_service import BaseService
from app.schemas.transaction import (
    TransactionCreate,
    TransactionUpdate,
    TransactionResponse,
    TransactionStatus
)
from app.core.exceptions import InsufficientFundsError, TransactionError


class TransactionService(BaseService):
    """Service for banking transaction operations."""
    
    async def create_transaction(
        self,
        user_id: UUID,
        data: TransactionCreate
    ) -> TransactionResponse:
        """Create a new transaction with validation."""
        # Generate unique reference ID
        reference_id = f"TXN-{uuid4().hex[:12].upper()}"
        
        transaction_data = data.model_dump()
        transaction_data["user_id"] = user_id
        transaction_data["reference_id"] = reference_id
        transaction_data["status"] = TransactionStatus.PENDING
        
        # Validate transaction amount
        if data.amount <= 0:
            raise TransactionError("Transaction amount must be positive")
        
        result = await self._repository.create(
            type("Schema", (), transaction_data)()
        )
        
        # Process transaction (in real scenario, this would be async)
        await self._process_transaction(result)
        
        return TransactionResponse.model_validate(result)
    
    async def _process_transaction(self, transaction) -> None:
        """Process transaction asynchronously."""
        # Implementation would include:
        # - Balance verification
        # - External payment gateway integration
        # - Status update
        pass
    
    async def get_user_transactions(
        self,
        user_id: UUID,
        skip: int = 0,
        limit: int = 20,
        status: Optional[TransactionStatus] = None
    ) -> List[TransactionResponse]:
        """Get transactions for a user."""
        filters = {"user_id": user_id}
        if status:
            filters["status"] = status
        
        transactions = await self._repository.get_all(
            skip=skip,
            limit=limit,
            filters=filters,
            sort_by="created_at",
            sort_order="desc"
        )
        return [TransactionResponse.model_validate(t) for t in transactions]
```

---

## 9. API Endpoints

```python
# app/api/deps.py

from typing import Annotated
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from app.database.factory import DatabaseFactory
from app.database.adapters.base_adapter import BaseDatabaseAdapter
from app.core.security import verify_token
from app.schemas.user import UserResponse

security = HTTPBearer()


async def get_database() -> BaseDatabaseAdapter:
    """Dependency for database adapter."""
    return DatabaseFactory.get_adapter()


async def get_current_user(
    credentials: Annotated[HTTPAuthorizationCredentials, Depends(security)],
    db: Annotated[BaseDatabaseAdapter, Depends(get_database)]
) -> UserResponse:
    """Get current authenticated user."""
    token = credentials.credentials
    payload = verify_token(token)
    
    if not payload:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials"
        )
    
    user_id = payload.get("sub")
    user = await db.get_by_id("users", user_id)
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    return UserResponse.model_validate(user)


CurrentUser = Annotated[UserResponse, Depends(get_current_user)]
Database = Annotated[BaseDatabaseAdapter, Depends(get_database)]


# app/api/v1/endpoints/chat.py

from typing import List
from uuid import UUID
from fastapi import APIRouter, Depends, HTTPException, status

from app.api.deps import CurrentUser, Database
from app.schemas.chat import (
    ChatSessionCreate,
    ChatSessionResponse,
    ChatSessionListResponse,
    ChatMessageCreate,
    ChatMessageResponse
)
from app.schemas.base import APIResponse, PaginatedResponse
from app.services.chat_service import ChatService

router = APIRouter(prefix="/chat", tags=["Chat"])


@router.post("/sessions", response_model=APIResponse[ChatSessionResponse])
async def create_chat_session(
    data: ChatSessionCreate,
    current_user: CurrentUser,
    db: Database
):
    """Create a new chat session."""
    service = ChatService(
        session_repository=db,
        message_repository=db
    )
    session = await service.create_session(current_user.id, data)
    return APIResponse(data=session, message="Session created successfully")


@router.get("/sessions", response_model=APIResponse[List[ChatSessionListResponse]])
async def get_chat_sessions(
    current_user: CurrentUser,
    db: Database,
    skip: int = 0,
    limit: int = 20
):
    """Get all chat sessions for current user."""
    service = ChatService(
        session_repository=db,
        message_repository=db
    )
    sessions = await service.get_user_sessions(
        current_user.id,
        skip=skip,
        limit=limit
    )
    return APIResponse(data=sessions)


@router.get("/sessions/{session_id}", response_model=APIResponse[ChatSessionResponse])
async def get_chat_session(
    session_id: UUID,
    current_user: CurrentUser,
    db: Database
):
    """Get a specific chat session with messages."""
    session = await db.get_by_id("chat_sessions", session_id)
    
    if not session or session.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found"
        )
    
    service = ChatService(
        session_repository=db,
        message_repository=db
    )
    messages = await service.get_session_history(session_id)
    
    response = ChatSessionResponse.model_validate(session)
    response.messages = messages
    
    return APIResponse(data=response)


@router.post(
    "/sessions/{session_id}/messages",
    response_model=APIResponse[ChatMessageResponse]
)
async def add_message(
    session_id: UUID,
    data: ChatMessageCreate,
    current_user: CurrentUser,
    db: Database
):
    """Add a message to a chat session."""
    session = await db.get_by_id("chat_sessions", session_id)
    
    if not session or session.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found"
        )
    
    service = ChatService(
        session_repository=db,
        message_repository=db
    )
    message = await service.add_message(session_id, data)
    
    return APIResponse(data=message, message="Message added successfully")


@router.delete("/sessions/{session_id}", response_model=APIResponse)
async def delete_chat_session(
    session_id: UUID,
    current_user: CurrentUser,
    db: Database
):
    """Delete a chat session."""
    session = await db.get_by_id("chat_sessions", session_id)
    
    if not session or session.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found"
        )
    
    await db.delete("chat_sessions", session_id)
    return APIResponse(message="Session deleted successfully")


# app/api/v1/endpoints/transactions.py

from typing import List, Optional
from uuid import UUID
from fastapi import APIRouter, Depends, Query

from app.api.deps import CurrentUser, Database
from app.schemas.transaction import (
    TransactionCreate,
    TransactionResponse,
    TransactionStatus
)
from app.schemas.base import APIResponse, PaginatedResponse
from app.services.transaction_service import TransactionService

router = APIRouter(prefix="/transactions", tags=["Transactions"])


@router.post("", response_model=APIResponse[TransactionResponse])
async def create_transaction(
    data: TransactionCreate,
    current_user: CurrentUser,
    db: Database
):
    """Create a new transaction."""
    service = TransactionService(repository=db)
    transaction = await service.create_transaction(current_user.id, data)
    return APIResponse(data=transaction, message="Transaction created")


@router.get("", response_model=APIResponse[List[TransactionResponse]])
async def get_transactions(
    current_user: CurrentUser,
    db: Database,
    skip: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100),
    status: Optional[TransactionStatus] = None
):
    """Get user transactions."""
    service = TransactionService(repository=db)
    transactions = await service.get_user_transactions(
        current_user.id,
        skip=skip,
        limit=limit,
        status=status
    )
    return APIResponse(data=transactions)


@router.get("/{transaction_id}", response_model=APIResponse[TransactionResponse])
async def get_transaction(
    transaction_id: UUID,
    current_user: CurrentUser,
    db: Database
):
    """Get a specific transaction."""
    transaction = await db.get_by_id("transactions", transaction_id)
    
    if not transaction or transaction.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Transaction not found"
        )
    
    return APIResponse(data=TransactionResponse.model_validate(transaction))


# app/api/v1/router.py

from fastapi import APIRouter

from app.api.v1.endpoints import (
    auth,
    users,
    chat,
    transactions,
    products,
    projects,
    courses
)

api_router = APIRouter()

api_router.include_router(auth.router)
api_router.include_router(users.router)
api_router.include_router(chat.router)
api_router.include_router(transactions.router)
api_router.include_router(products.router)
api_router.include_router(projects.router)
api_router.include_router(courses.router)
```

---

## 10. Main Application

```python
# app/main.py

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware

from app.core.settings import settings
from app.database.factory import DatabaseFactory
from app.api.v1.router import api_router
from app.middleware.logging import LoggingMiddleware
from app.middleware.error_handler import error_handler_middleware


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management."""
    # Startup
    await DatabaseFactory.initialize()
    
    # Register models for SQL databases
    if settings.DATABASE_TYPE in ["sqlite", "postgresql"]:
        adapter = DatabaseFactory.get_adapter()
        from app.models.sql_models import (
            User, ChatSession, ChatMessage,
            Transaction, Product, Order, OrderItem,
            Project, Task, Course, CourseSection, Enrollment
        )
        
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
    
    yield
    
    # Shutdown
    await DatabaseFactory.shutdown()


def create_application() -> FastAPI:
    """Application factory."""
    app = FastAPI(
        title=settings.APP_NAME,
        version=settings.APP_VERSION,
        debug=settings.DEBUG,
        lifespan=lifespan,
        docs_url="/api/docs",
        redoc_url="/api/redoc",
        openapi_url="/api/openapi.json"
    )
    
    # Middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"]
    )
    app.add_middleware(GZipMiddleware, minimum_size=1000)
    app.add_middleware(LoggingMiddleware)
    
    # Error handlers
    app.middleware("http")(error_handler_middleware)
    
    # Routes
    app.include_router(api_router, prefix=settings.API_V1_PREFIX)
    
    # Health check
    @app.get("/health")
    async def health_check():
        return {
            "status": "healthy",
            "version": settings.APP_VERSION,
            "database": settings.DATABASE_TYPE
        }
    
    return app


app = create_application()
```

---

## 11. Docker Configuration

```yaml
# docker-compose.yml

version: "3.9"

services:
  app:
    build:
      context: .
      dockerfile: Dockerfile
    ports:
      - "8000:8000"
    environment:
      - DATABASE_TYPE=${DATABASE_TYPE:-postgresql}
      - POSTGRES_HOST=postgres
      - POSTGRES_PORT=5432
      - POSTGRES_USER=app_user
      - POSTGRES_PASSWORD=app_password
      - POSTGRES_DB=app_db
      - MONGODB_URL=mongodb://mongodb:27017
      - MONGODB_DB=app_db
    depends_on:
      postgres:
        condition: service_healthy
      mongodb:
        condition: service_started
    volumes:
      - ./app:/app/app
    command: uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

  postgres:
    image: postgres:15-alpine
    environment:
      - POSTGRES_USER=app_user
      - POSTGRES_PASSWORD=app_password
      - POSTGRES_DB=app_db
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U app_user -d app_db"]
      interval: 10s
      timeout: 5s
      retries: 5

  mongodb:
    image: mongo:6.0
    ports:
      - "27017:27017"
    volumes:
      - mongodb_data:/data/db

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data

volumes:
  postgres_data:
  mongodb_data:
  redis_data:
```

```dockerfile
# Dockerfile

FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Create non-root user
RUN adduser --disabled-password --gecos '' appuser
USER appuser

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

## 12. Requirements

```text
# requirements.txt

# FastAPI Core
fastapi==0.109.0
uvicorn[standard]==0.27.0
pydantic==2.5.3
pydantic-settings==2.1.0

# Database - SQL
sqlalchemy[asyncio]==2.0.25
asyncpg==0.29.0
aiosqlite==0.19.0
alembic==1.13.1

# Database - MongoDB
motor==3.3.2
pymongo==4.6.1

# Security
python-jose[cryptography]==3.3.0
passlib[bcrypt]==1.7.4
python-multipart==0.0.6

# Utilities
httpx==0.26.0
orjson==3.9.12
python-dotenv==1.0.0

# Testing
pytest==7.4.4
pytest-asyncio==0.23.3
pytest-cov==4.1.0

# Development
black==24.1.0
isort==5.13.2
mypy==1.8.0
```

---

## 13. Summary Architecture Diagram

```
┌────────────────────────────────────────────────────────────────────────────┐
│                            APPLICATION LAYERS                               │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────┐    │
│  │                        PRESENTATION LAYER                          │    │
│  │  • FastAPI Routers (chat, transactions, products, projects, LMS)  │    │
│  │  • Request/Response Schemas (Pydantic)                             │    │
│  │  • Authentication Middleware                                        │    │
│  └────────────────────────────────────────────────────────────────────┘    │
│                                    │                                         │
│                                    ▼                                         │
│  ┌────────────────────────────────────────────────────────────────────┐    │
│  │                         SERVICE LAYER                               │    │
│  │  • Business Logic Implementation                                    │    │
│  │  • Validation & Transformation                                      │    │
│  │  • Cross-cutting Concerns                                          │    │
│  └────────────────────────────────────────────────────────────────────┘    │
│                                    │                                         │
│                                    ▼                                         │
│  ┌────────────────────────────────────────────────────────────────────┐    │
│  │                       REPOSITORY LAYER                              │    │
│  │  • Generic CRUD Operations                                          │    │
│  │  • Query Building                                                   │    │
│  │  • Unit of Work Pattern                                            │    │
│  └────────────────────────────────────────────────────────────────────┘    │
│                                    │                                         │
│                                    ▼                                         │
│  ┌────────────────────────────────────────────────────────────────────┐    │
│  │                    DATABASE ABSTRACTION LAYER                       │    │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐              │    │
│  │  │   SQLite     │  │  PostgreSQL  │  │   MongoDB    │              │    │
│  │  │   Adapter    │  │   Adapter    │  │   Adapter    │              │    │
│  │  └──────────────┘  └──────────────┘  └──────────────┘              │    │
│  └────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
└────────────────────────────────────────────────────────────────────────────┘
```

---

## Key Benefits of This Architecture

| Feature | Benefit |
|---------|---------|
| **Database Agnostic** | Switch between SQLite, PostgreSQL, MongoDB via configuration |
| **Repository Pattern** | Decoupled data access with testable abstractions |
| **Factory Pattern** | Dynamic adapter instantiation based on configuration |
| **Unit of Work** | Transactional consistency across operations |
| **Service Layer** | Business logic isolation from infrastructure |
| **Async/Await** | High-performance non-blocking I/O |
| **Type Safety** | Full Pydantic validation and type hints |
| **Modular Design** | Easy to extend with new modules (chat, banking, etc.) |

This architecture supports all mentioned use cases (Chatbot, Banking, E-Commerce, Project Management, LMS) with a single unified codebase.