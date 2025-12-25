# =============================================================================
# SOTA AUTHENTICATION SYSTEM - SQLITE ADAPTER
# =============================================================================
# File: db/adapters/sqlite_adapter.py
# Description: SQLite database adapter for development and testing
#              Uses aiosqlite for async operations with SQLAlchemy
# =============================================================================

from typing import Any, Dict, Optional
import os
from pathlib import Path

from db.base import BaseDBAdapter
from core.config import settings


class SQLiteAdapter(BaseDBAdapter):
    """
    ┌─────────────────────────────────────────────────────────────────────────┐
    │                    SQLITE DATABASE ADAPTER                               │
    │  Async SQLite implementation for development and testing environments   │
    │  Uses aiosqlite driver with SQLAlchemy async ORM                        │
    └─────────────────────────────────────────────────────────────────────────┘
    
    Features:
        - Zero-configuration setup
        - File-based persistent storage
        - In-memory option for testing
        - Auto-creation of database directory
        - SQLite-specific optimizations (WAL mode, etc.)
    
    Usage:
        adapter = SQLiteAdapter()
        await adapter.connect()
        async with adapter.get_session() as session:
            # perform database operations
        await adapter.disconnect()
    """
    
    def __init__(
        self,
        database_url: Optional[str] = None,
        **kwargs: Any
    ):
        """
        Initialize SQLite adapter.
        
        Args:
            database_url: Optional custom database URL
                         Defaults to settings.database_url
            **kwargs: Additional engine options
                     
        Engine Options:
            - echo: bool - Log SQL queries (default: settings.debug)
            - pool_pre_ping: bool - Test connections before use (default: True)
        """
        # Use provided URL or get from settings
        if database_url is None:
            database_url = settings.database_url
        
        # Ensure database directory exists for file-based SQLite
        if "sqlite" in database_url and ":memory:" not in database_url:
            db_path = database_url.replace("sqlite+aiosqlite:///", "")
            db_dir = Path(db_path).parent
            db_dir.mkdir(parents=True, exist_ok=True)
        
        # Default SQLite engine options
        default_options = {
            "echo": settings.debug,
            "pool_pre_ping": True,
            # SQLite doesn't support connection pooling in traditional sense
            # Each connection is isolated, but we configure for async safety
            "connect_args": {
                "check_same_thread": False,
                # Enable foreign key support
                "timeout": 30,
            },
        }
        
        # Merge with provided options
        default_options.update(kwargs)
        
        super().__init__(database_url, **default_options)
    
    async def connect(self) -> None:
        """
        Connect to SQLite database with optimizations.
        
        Applies SQLite-specific PRAGMA settings for performance:
            - WAL mode for better concurrency
            - Foreign keys enabled
            - Synchronous mode for safety
        """
        await super().connect()
        
        # Apply SQLite pragmas for optimization using raw connection
        from sqlalchemy import text
        
        async with self.get_session() as session:
            # Enable Write-Ahead Logging for better concurrency
            await session.execute(text("PRAGMA journal_mode=WAL"))
            # Enable foreign key constraints
            await session.execute(text("PRAGMA foreign_keys=ON"))
            # Synchronous mode: NORMAL offers good balance
            await session.execute(text("PRAGMA synchronous=NORMAL"))
            # Cache size: 64MB (negative = KB)
            await session.execute(text("PRAGMA cache_size=-65536"))
            # Busy timeout: 30 seconds
            await session.execute(text("PRAGMA busy_timeout=30000"))
            await session.commit()
    
    async def execute(
        self,
        query: str,
        params: Optional[Dict[str, Any]] = None
    ) -> Any:
        """
        Execute raw SQL with PRAGMA support.
        
        SQLite PRAGMA statements don't use parameters,
        so we handle them specially.
        """
        from sqlalchemy import text
        
        async with self.get_session() as session:
            if query.strip().upper().startswith("PRAGMA"):
                result = await session.execute(text(query))
            else:
                result = await session.execute(
                    text(query),
                    params or {}
                )
            return result
    
    @classmethod
    def create_for_testing(cls) -> "SQLiteAdapter":
        """
        Create an in-memory SQLite adapter for testing.
        
        Returns:
            SQLiteAdapter: Configured for in-memory testing
            
        Note:
            In-memory databases are ephemeral - data is lost
            when the connection closes. Useful for fast tests.
        """
        return cls(
            database_url="sqlite+aiosqlite:///:memory:",
            echo=False,
        )
    
    async def vacuum(self) -> None:
        """
        Run VACUUM to reclaim space and defragment database.
        
        Should be run periodically for file-based databases
        to maintain performance.
        """
        await self.execute("VACUUM")
    
    async def get_database_size(self) -> int:
        """
        Get the size of the database file in bytes.
        
        Returns:
            int: Database file size in bytes, 0 for in-memory
        """
        if ":memory:" in self._database_url:
            return 0
        
        db_path = self._database_url.replace("sqlite+aiosqlite:///", "")
        if os.path.exists(db_path):
            return os.path.getsize(db_path)
        return 0


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def get_sqlite_adapter(**kwargs: Any) -> SQLiteAdapter:
    """
    Factory function to create SQLite adapter.
    
    Args:
        **kwargs: Options passed to SQLiteAdapter
        
    Returns:
        SQLiteAdapter: Configured adapter instance
    """
    return SQLiteAdapter(**kwargs)
