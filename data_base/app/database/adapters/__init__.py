# ==============================================================================
# DATABASE ADAPTERS PACKAGE
# ==============================================================================

"""
Database Adapters
=================

Provides unified interface implementations for different databases:
- BaseDatabaseAdapter: Abstract interface definition
- PostgreSQLAdapter: PostgreSQL using SQLAlchemy async
- MongoDBAdapter: MongoDB using Motor async driver
- SQLiteAdapter: SQLite using aiosqlite
"""

from app.database.adapters.base_adapter import BaseDatabaseAdapter
from app.database.adapters.postgresql_adapter import PostgreSQLAdapter
from app.database.adapters.mongodb_adapter import MongoDBAdapter
from app.database.adapters.sqlite_adapter import SQLiteAdapter

__all__ = [
    "BaseDatabaseAdapter",
    "PostgreSQLAdapter",
    "MongoDBAdapter",
    "SQLiteAdapter",
]
