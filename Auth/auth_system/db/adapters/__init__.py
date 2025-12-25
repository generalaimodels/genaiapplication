# =============================================================================
# DATABASE ADAPTERS INITIALIZATION
# =============================================================================
# File: db/adapters/__init__.py
# Description: Adapters module exports
# =============================================================================

from db.adapters.sqlite_adapter import SQLiteAdapter, get_sqlite_adapter
from db.adapters.postgres_adapter import PostgresAdapter, get_postgres_adapter
from db.adapters.redis_adapter import RedisAdapter, get_redis_adapter

__all__ = [
    "SQLiteAdapter",
    "get_sqlite_adapter",
    "PostgresAdapter",
    "get_postgres_adapter",
    "RedisAdapter",
    "get_redis_adapter",
]
