# =============================================================================
# DATABASE MODULE INITIALIZATION
# =============================================================================
# File: db/__init__.py
# Description: Database module exports
# =============================================================================

from db.base import Base, IDBAdapter, BaseDBAdapter, IRedisAdapter
from db.factory import (
    DBFactory,
    DatabaseType,
    get_db_adapter,
    get_redis_adapter,
    get_db_session,
    get_redis,
)
from db.models import (
    User,
    Session,
    RefreshToken,
    AuditLog,
    AuditAction,
    AuditStatus,
)
from db.adapters import (
    SQLiteAdapter,
    PostgresAdapter,
    RedisAdapter,
)

__all__ = [
    # Base
    "Base",
    "IDBAdapter",
    "BaseDBAdapter",
    "IRedisAdapter",
    
    # Factory
    "DBFactory",
    "DatabaseType",
    "get_db_adapter",
    "get_redis_adapter",
    "get_db_session",
    "get_redis",
    
    # Models
    "User",
    "Session",
    "RefreshToken",
    "AuditLog",
    "AuditAction",
    "AuditStatus",
    
    # Adapters
    "SQLiteAdapter",
    "PostgresAdapter",
    "RedisAdapter",
]
