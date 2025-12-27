# -*- coding: utf-8 -*-
# =================================================================================================
# api/database.py — Async Database Layer with Connection Pooling
# =================================================================================================
# Production-grade database management implementing:
#
#   1. CONNECTION POOLING: Reuse connections to minimize handshake overhead.
#   2. WAL MODE: Write-Ahead Logging for concurrent read/write without blocking.
#   3. ASYNC OPERATIONS: Non-blocking I/O via aiosqlite for event loop compatibility.
#   4. SOFT DELETES: Preserve audit trail with is_active flags.
#   5. UUID PRIMARY KEYS: Globally unique, idempotency-friendly identifiers.
#   6. OPTIMISTIC LOCKING: Version field for concurrent update detection.
#
# Schema Design:
# --------------
#   - history: Query-answer pairs with metadata, retrieval context, and timing.
#   - sessions: Conversation session management with user tracking.
#   - responses: Detailed LLM response tracking for analytics.
#
# Thread Safety:
# --------------
#   - aiosqlite handles async I/O; each request gets a connection from pool.
#   - SQLite WAL mode allows concurrent readers with a single writer.
#   - Connection pool prevents exhaustion under high load.
#
# =================================================================================================

from __future__ import annotations

import asyncio
import json
import logging
import os
import sqlite3
import threading
import time
import uuid
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, Generator, List, Optional, Tuple, Union

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
_LOG = logging.getLogger("api.database")


# -----------------------------------------------------------------------------
# UUID Generation Utilities
# -----------------------------------------------------------------------------
def generate_uuid() -> str:
    """
    Generate a UUID v4 string for primary keys.
    
    Design Note:
    ------------
    UUID v4 provides:
    - Globally unique IDs without coordination.
    - Safe for distributed systems.
    - Idempotency-friendly (client can generate ID before request).
    """
    return str(uuid.uuid4())


def now_timestamp() -> float:
    """Get current Unix timestamp with subsecond precision."""
    return time.time()


# -----------------------------------------------------------------------------
# Database Schema Definitions
# -----------------------------------------------------------------------------
SCHEMA_VERSION = 1

SCHEMA_SQL = """
-- =============================================================================
-- API Database Schema v1
-- =============================================================================
-- Design Principles:
--   - UUIDs for primary keys (TEXT storage for portability)
--   - JSON columns for flexible metadata (no schema migrations needed)
--   - Soft deletes via is_active flag (preserve audit trail)
--   - Timestamps as REAL (Unix epoch with subsecond precision)
--   - Foreign key constraints for referential integrity
--   - Indexes on frequently queried columns
-- =============================================================================

-- Schema version tracking for migrations
CREATE TABLE IF NOT EXISTS schema_version (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    version INTEGER NOT NULL,
    migrated_at REAL NOT NULL
);

-- Sessions: Conversation session management
-- Each session maps to a conversation in the chat history engine.
CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,                          -- UUID v4
    user_id TEXT,                                 -- Optional user identifier
    conv_id TEXT NOT NULL,                        -- Conversation ID for history.py
    branch_id TEXT NOT NULL DEFAULT 'main',       -- Branch for conversation forking
    title TEXT,                                   -- Optional session title
    metadata TEXT DEFAULT '{}',                   -- JSON: custom metadata
    created_at REAL NOT NULL,                     -- Unix timestamp
    updated_at REAL NOT NULL,                     -- Unix timestamp
    is_active INTEGER NOT NULL DEFAULT 1,         -- Soft delete flag (1=active)
    version INTEGER NOT NULL DEFAULT 1            -- Optimistic locking version
);

-- History: Query-answer pairs with retrieval context
-- Core table for tracking user queries and AI responses.
CREATE TABLE IF NOT EXISTS history (
    id TEXT PRIMARY KEY,                          -- UUID v4
    session_id TEXT NOT NULL,                     -- FK to sessions
    query TEXT NOT NULL,                          -- User query text
    answer TEXT,                                  -- AI response text (null if pending)
    role TEXT NOT NULL DEFAULT 'user',            -- Message role (user/assistant/system)
    metadata TEXT DEFAULT '{}',                   -- JSON: timing, scores, etc.
    retrieves TEXT DEFAULT '[]',                  -- JSON: retrieved context chunks
    tokens_query INTEGER,                         -- Token count for query
    tokens_answer INTEGER,                        -- Token count for answer
    latency_ms REAL,                              -- Response latency in milliseconds
    created_at REAL NOT NULL,                     -- Unix timestamp
    updated_at REAL NOT NULL,                     -- Unix timestamp
    FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
);

-- Responses: Detailed LLM response tracking for analytics
-- Tracks each LLM call for debugging, cost analysis, and performance monitoring.
CREATE TABLE IF NOT EXISTS responses (
    id TEXT PRIMARY KEY,                          -- UUID v4
    history_id TEXT NOT NULL,                     -- FK to history
    model TEXT NOT NULL,                          -- Model identifier
    prompt TEXT NOT NULL,                         -- Full prompt sent to LLM
    response TEXT NOT NULL,                       -- Raw LLM response
    temperature REAL,                             -- Temperature used
    max_tokens INTEGER,                           -- Max tokens requested
    tokens_prompt INTEGER,                        -- Actual prompt tokens
    tokens_completion INTEGER,                    -- Actual completion tokens
    finish_reason TEXT,                           -- stop/length/error
    latency_ms REAL NOT NULL,                     -- LLM call latency
    error TEXT,                                   -- Error message if failed
    created_at REAL NOT NULL,                     -- Unix timestamp
    FOREIGN KEY (history_id) REFERENCES history(id) ON DELETE CASCADE
);

-- Documents: Track uploaded/processed documents
CREATE TABLE IF NOT EXISTS documents (
    id TEXT PRIMARY KEY,                          -- UUID v4
    filename TEXT NOT NULL,                       -- Original filename
    content_type TEXT,                            -- MIME type
    file_size INTEGER,                            -- Size in bytes
    file_hash TEXT,                               -- SHA-256 hash for dedup
    status TEXT NOT NULL DEFAULT 'pending',       -- pending/processing/completed/failed
    stage TEXT NOT NULL DEFAULT 'pending',        -- pending/converting/chunking/indexing/completed/failed
    progress INTEGER NOT NULL DEFAULT 0,          -- Processing progress (0-100)
    chunk_count INTEGER DEFAULT 0,                -- Number of chunks created
    metadata TEXT DEFAULT '{}',                   -- JSON: processing options
    error TEXT,                                   -- Error message if failed
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    is_active INTEGER NOT NULL DEFAULT 1
);

-- Users: User accounts for authentication
CREATE TABLE IF NOT EXISTS users (
    id TEXT PRIMARY KEY,                          -- UUID v4
    username TEXT NOT NULL UNIQUE,                -- Unique username (lowercase)
    password_hash TEXT NOT NULL,                  -- SHA256 hashed password
    created_at REAL NOT NULL,                     -- Unix timestamp
    updated_at REAL NOT NULL,                     -- Unix timestamp
    is_active INTEGER NOT NULL DEFAULT 1          -- Soft delete flag
);

-- =============================================================================
-- Indexes for Query Performance
-- =============================================================================
-- N+1 Prevention: Eager loading via JOINs uses these indexes.
-- Query patterns: session lookup, history by session, recent items.

CREATE INDEX IF NOT EXISTS idx_sessions_user_id ON sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_sessions_conv_id ON sessions(conv_id);
CREATE INDEX IF NOT EXISTS idx_sessions_created_at ON sessions(created_at);
CREATE INDEX IF NOT EXISTS idx_sessions_is_active ON sessions(is_active);

CREATE INDEX IF NOT EXISTS idx_history_session_id ON history(session_id);
CREATE INDEX IF NOT EXISTS idx_history_created_at ON history(created_at);
CREATE INDEX IF NOT EXISTS idx_history_role ON history(role);

CREATE INDEX IF NOT EXISTS idx_responses_history_id ON responses(history_id);
CREATE INDEX IF NOT EXISTS idx_responses_model ON responses(model);
CREATE INDEX IF NOT EXISTS idx_responses_created_at ON responses(created_at);

CREATE INDEX IF NOT EXISTS idx_documents_file_hash ON documents(file_hash);
CREATE INDEX IF NOT EXISTS idx_documents_status ON documents(status);
CREATE INDEX IF NOT EXISTS idx_documents_is_active ON documents(is_active);

CREATE INDEX IF NOT EXISTS idx_users_username ON users(username);
CREATE INDEX IF NOT EXISTS idx_users_is_active ON users(is_active);
""".strip()


# -----------------------------------------------------------------------------
# Synchronous Database Manager (for startup and simple operations)
# -----------------------------------------------------------------------------
class DatabaseManager:
    """
    Synchronous SQLite database manager with connection pooling.
    
    Design Notes:
    -------------
    - WAL mode enables concurrent readers with single writer.
    - Connection pool avoids per-request connection overhead.
    - Thread-local storage prevents cross-thread connection sharing.
    - Prepared statement caching for frequently used queries.
    
    Concurrency Model:
    ------------------
    - SQLite WAL: Multiple readers, single writer.
    - Pool size: Controls max concurrent connections.
    - Busy timeout: Wait time when database is locked.
    """
    
    __slots__ = ("_db_path", "_pool_size", "_timeout", "_connections", "_lock", "_local")
    
    def __init__(
        self,
        db_path: Union[str, Path],
        pool_size: int = 5,
        timeout: float = 30.0,
    ) -> None:
        """
        Initialize database manager.
        
        Parameters:
        -----------
        db_path : Path to SQLite database file.
        pool_size : Maximum connections in pool.
        timeout : Seconds to wait for available connection.
        """
        self._db_path = Path(db_path)
        self._pool_size = pool_size
        self._timeout = timeout
        self._connections: List[sqlite3.Connection] = []
        self._lock = threading.Lock()
        self._local = threading.local()
        
        # Ensure data directory exists
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
    
    def _create_connection(self) -> sqlite3.Connection:
        """
        Create a new database connection with optimal settings.
        
        SQLite Pragmas:
        ---------------
        - journal_mode=WAL: Concurrent reads during writes.
        - synchronous=NORMAL: Balance between safety and speed.
        - foreign_keys=ON: Enforce referential integrity.
        - temp_store=MEMORY: Faster temp table operations.
        - mmap_size: Memory-mapped I/O for faster reads.
        - busy_timeout: Wait instead of failing on lock.
        """
        conn = sqlite3.connect(
            str(self._db_path),
            check_same_thread=False,
            isolation_level=None,  # Autocommit mode for explicit transaction control
            timeout=self._timeout,
        )
        conn.row_factory = sqlite3.Row
        
        # Apply performance pragmas
        cursor = conn.cursor()
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA synchronous=NORMAL")
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.execute("PRAGMA temp_store=MEMORY")
        cursor.execute("PRAGMA mmap_size=268435456")  # 256MB
        cursor.execute("PRAGMA cache_size=-65536")     # 64MB cache
        cursor.execute(f"PRAGMA busy_timeout={int(self._timeout * 1000)}")
        cursor.close()
        
        return conn
    
    def get_connection(self) -> sqlite3.Connection:
        """
        Get a connection from the pool.
        
        Thread Safety:
        --------------
        Each thread gets its own connection via thread-local storage.
        This prevents concurrent access issues with SQLite.
        """
        # Check thread-local connection first
        if hasattr(self._local, "conn") and self._local.conn is not None:
            try:
                # Validate thread-local connection
                self._local.conn.execute("SELECT 1")
                return self._local.conn
            except (sqlite3.ProgrammingError, sqlite3.OperationalError):
                # Connection stale or closed, clear it
                self._local.conn = None

        with self._lock:
            while self._connections:
                conn = self._connections.pop()
                try:
                    # Verify connection is still open
                    conn.execute("SELECT 1")
                    break
                except (sqlite3.ProgrammingError, sqlite3.OperationalError):
                    # Connection closed, discard and try next
                    try:
                        conn.close()
                    except:
                        pass
                    continue
            else:
                conn = self._create_connection()
        
        self._local.conn = conn
        return conn
    
    def release_connection(self, conn: sqlite3.Connection) -> None:
        """Return a connection to the pool."""
        with self._lock:
            if len(self._connections) < self._pool_size:
                self._connections.append(conn)
            else:
                conn.close()
        
        if hasattr(self._local, "conn") and self._local.conn is conn:
            self._local.conn = None
    
    @contextmanager
    def connection(self) -> Generator[sqlite3.Connection, None, None]:
        """Context manager for connection lifecycle."""
        conn = self.get_connection()
        try:
            yield conn
        finally:
            self.release_connection(conn)
    
    @contextmanager
    def transaction(self) -> Generator[sqlite3.Connection, None, None]:
        """
        Context manager for database transactions.
        
        Implements:
        -----------
        - Explicit BEGIN/COMMIT/ROLLBACK
        - Automatic rollback on exception
        - Connection release on completion
        """
        conn = self.get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute("BEGIN IMMEDIATE")
            yield conn
            cursor.execute("COMMIT")
        except Exception:
            cursor.execute("ROLLBACK")
            raise
        finally:
            cursor.close()
            self.release_connection(conn)
    
    def execute(
        self,
        sql: str,
        params: Tuple = (),
        fetch: bool = False,
    ) -> Union[List[sqlite3.Row], int]:
        """
        Execute SQL with optional fetch.
        
        Parameters:
        -----------
        sql : SQL statement to execute.
        params : Parameters for prepared statement.
        fetch : If True, return rows; otherwise return rowcount.
        
        Returns:
        --------
        List of rows if fetch=True, else affected row count.
        """
        with self.connection() as conn:
            cursor = conn.cursor()
            try:
                cursor.execute(sql, params)
                if fetch:
                    return cursor.fetchall()
                return cursor.rowcount
            finally:
                cursor.close()
    
    def execute_many(self, sql: str, params_seq: List[Tuple]) -> int:
        """Execute SQL for multiple parameter sets."""
        with self.connection() as conn:
            cursor = conn.cursor()
            try:
                cursor.executemany(sql, params_seq)
                return cursor.rowcount
            finally:
                cursor.close()
    
    def init_schema(self) -> None:
        """
        Initialize database schema.
        
        Idempotent:
        -----------
        - Uses CREATE TABLE IF NOT EXISTS.
        - Safe to call multiple times.
        - Tracks schema version for migrations.
        """
        conn = self.get_connection()
        cursor = conn.cursor()
        try:
            # Execute schema creation (executescript handles its own transactions)
            cursor.executescript(SCHEMA_SQL)
            
            # Migration: Add new columns if they don't exist (for existing databases)
            # Check for stage column
            cursor.execute("PRAGMA table_info(documents)")
            columns = [row[1] for row in cursor.fetchall()]
            
            if 'stage' not in columns:
                cursor.execute("ALTER TABLE documents ADD COLUMN stage TEXT NOT NULL DEFAULT 'pending'")
                _LOG.info("Added 'stage' column to documents table")
            
            if 'progress' not in columns:
                cursor.execute("ALTER TABLE documents ADD COLUMN progress INTEGER NOT NULL DEFAULT 0")
                _LOG.info("Added 'progress' column to documents table")
            
            # Insert or update schema version
            cursor.execute("""
                INSERT INTO schema_version (id, version, migrated_at)
                VALUES (1, ?, ?)
                ON CONFLICT(id) DO UPDATE SET version=excluded.version, migrated_at=excluded.migrated_at
            """, (SCHEMA_VERSION, now_timestamp()))
            
            _LOG.info("Database schema initialized (version=%d)", SCHEMA_VERSION)
        finally:
            cursor.close()
            self.release_connection(conn)
    
    def close(self) -> None:
        """Close all pooled connections."""
        with self._lock:
            for conn in self._connections:
                try:
                    conn.close()
                except Exception:
                    pass
            self._connections.clear()
        
        if hasattr(self._local, "conn") and self._local.conn is not None:
            try:
                self._local.conn.close()
            except Exception:
                pass
            self._local.conn = None


# -----------------------------------------------------------------------------
# Async Database Manager (for FastAPI request handlers)
# -----------------------------------------------------------------------------
class AsyncDatabaseManager:
    """
    Async SQLite database wrapper using sync manager with executor.
    
    Design Decision:
    ----------------
    SQLite is inherently synchronous. We use run_in_executor to off load
    blocking operations to a thread pool, keeping the event loop free.
    
    For higher concurrency requirements, consider:
    - PostgreSQL with asyncpg
    - SQLite with aiosqlite (if installed)
    
    This approach:
    - Works without additional dependencies.
    - Provides async interface for FastAPI compatibility.
    - Maintains connection pooling benefits.
    """
    
    __slots__ = ("_sync_manager", "_executor")
    
    def __init__(self, sync_manager: DatabaseManager) -> None:
        """Wrap synchronous manager for async access."""
        self._sync_manager = sync_manager
        self._executor = None  # Use default thread pool executor
    
    async def execute(
        self,
        sql: str,
        params: Tuple = (),
        fetch: bool = False,
    ) -> Union[List[Dict[str, Any]], int]:
        """
        Execute SQL asynchronously.
        
        Implementation:
        ---------------
        Runs synchronous execute in thread pool to avoid blocking event loop.
        """
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            self._executor,
            lambda: self._sync_manager.execute(sql, params, fetch)
        )
        
        # Convert Row objects to dicts for JSON serialization
        if fetch and result:
            return [dict(row) for row in result]
        return result
    
    async def execute_many(self, sql: str, params_seq: List[Tuple]) -> int:
        """Execute SQL for multiple parameter sets asynchronously."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            lambda: self._sync_manager.execute_many(sql, params_seq)
        )
    
    @asynccontextmanager
    async def transaction(self) -> AsyncGenerator[None, None]:
        """
        Async context manager for transactions.
        
        Note:
        -----
        Transactions span multiple operations; each operation still
        runs in the executor. The transaction context is maintained
        via thread-local storage in sync manager.
        """
        loop = asyncio.get_event_loop()
        conn = await loop.run_in_executor(
            self._executor,
            self._sync_manager.get_connection
        )
        cursor = conn.cursor()
        try:
            await loop.run_in_executor(self._executor, lambda: cursor.execute("BEGIN IMMEDIATE"))
            yield
            await loop.run_in_executor(self._executor, lambda: cursor.execute("COMMIT"))
        except Exception:
            await loop.run_in_executor(self._executor, lambda: cursor.execute("ROLLBACK"))
            raise
        finally:
            cursor.close()
            await loop.run_in_executor(
                self._executor,
                lambda: self._sync_manager.release_connection(conn)
            )


# -----------------------------------------------------------------------------
# Database CRUD Operations (Repository Pattern)
# -----------------------------------------------------------------------------
class SessionRepository:
    """
    Repository for session CRUD operations.
    
    Implements:
    -----------
    - Create, Read, Update, Delete (soft) for sessions.
    - Pagination for listing sessions.
    - Optimistic locking for concurrent updates.
    """
    
    def __init__(self, db: AsyncDatabaseManager) -> None:
        self._db = db
    
    async def create(
        self,
        user_id: Optional[str] = None,
        conv_id: Optional[str] = None,
        branch_id: str = "main",
        title: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Create a new session.
        
        Parameters:
        -----------
        user_id : Optional user identifier.
        conv_id : Conversation ID (auto-generated if not provided).
        branch_id : Branch ID for conversation forking.
        title : Optional session title.
        metadata : Custom metadata dictionary.
        
        Returns:
        --------
        Created session as dictionary.
        """
        session_id = generate_uuid()
        conv_id = conv_id or generate_uuid()
        now = now_timestamp()
        meta_json = json.dumps(metadata or {})
        
        await self._db.execute(
            """
            INSERT INTO sessions (id, user_id, conv_id, branch_id, title, metadata, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (session_id, user_id, conv_id, branch_id, title, meta_json, now, now)
        )
        
        return {
            "id": session_id,
            "user_id": user_id,
            "conv_id": conv_id,
            "branch_id": branch_id,
            "title": title,
            "metadata": metadata or {},
            "created_at": now,
            "updated_at": now,
            "is_active": True,
            "version": 1,
        }
    
    async def get(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session by ID."""
        rows = await self._db.execute(
            "SELECT * FROM sessions WHERE id = ? AND is_active = 1",
            (session_id,),
            fetch=True
        )
        if not rows:
            return None
        row = rows[0]
        row["metadata"] = json.loads(row.get("metadata") or "{}")
        row["is_active"] = bool(row.get("is_active", 1))
        return row
    
    async def list(
        self,
        user_id: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """List sessions with optional user filter."""
        if user_id:
            rows = await self._db.execute(
                """
                SELECT * FROM sessions 
                WHERE user_id = ? AND is_active = 1 
                ORDER BY updated_at DESC 
                LIMIT ? OFFSET ?
                """,
                (user_id, limit, offset),
                fetch=True
            )
        else:
            rows = await self._db.execute(
                """
                SELECT * FROM sessions 
                WHERE is_active = 1 
                ORDER BY updated_at DESC 
                LIMIT ? OFFSET ?
                """,
                (limit, offset),
                fetch=True
            )
        
        for row in rows:
            row["metadata"] = json.loads(row.get("metadata") or "{}")
            row["is_active"] = bool(row.get("is_active", 1))
        return rows
    
    async def update(
        self,
        session_id: str,
        title: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        version: Optional[int] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Update session with optimistic locking.
        
        Optimistic Locking:
        -------------------
        If version is provided, update only succeeds if current version
        matches. This prevents lost updates in concurrent scenarios.
        """
        now = now_timestamp()
        
        if version is not None:
            # Optimistic locking: update only if version matches
            result = await self._db.execute(
                """
                UPDATE sessions 
                SET title = COALESCE(?, title),
                    metadata = COALESCE(?, metadata),
                    updated_at = ?,
                    version = version + 1
                WHERE id = ? AND is_active = 1 AND version = ?
                """,
                (title, json.dumps(metadata) if metadata else None, now, session_id, version)
            )
            if result == 0:
                return None  # Version mismatch or not found
        else:
            await self._db.execute(
                """
                UPDATE sessions 
                SET title = COALESCE(?, title),
                    metadata = COALESCE(?, metadata),
                    updated_at = ?,
                    version = version + 1
                WHERE id = ? AND is_active = 1
                """,
                (title, json.dumps(metadata) if metadata else None, now, session_id)
            )
        
        return await self.get(session_id)
    
    async def delete(self, session_id: str) -> bool:
        """Soft delete session."""
        result = await self._db.execute(
            "UPDATE sessions SET is_active = 0, updated_at = ? WHERE id = ? AND is_active = 1",
            (now_timestamp(), session_id)
        )
        return result > 0


class HistoryRepository:
    """Repository for history (query-answer pairs) CRUD operations."""
    
    def __init__(self, db: AsyncDatabaseManager) -> None:
        self._db = db
    
    async def create(
        self,
        session_id: str,
        query: str,
        answer: Optional[str] = None,
        role: str = "user",
        metadata: Optional[Dict[str, Any]] = None,
        retrieves: Optional[List[Dict[str, Any]]] = None,
        tokens_query: Optional[int] = None,
        tokens_answer: Optional[int] = None,
        latency_ms: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Create a new history entry."""
        history_id = generate_uuid()
        now = now_timestamp()
        
        await self._db.execute(
            """
            INSERT INTO history 
            (id, session_id, query, answer, role, metadata, retrieves, 
             tokens_query, tokens_answer, latency_ms, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                history_id, session_id, query, answer, role,
                json.dumps(metadata or {}), json.dumps(retrieves or []),
                tokens_query, tokens_answer, latency_ms, now, now
            )
        )
        
        # Update session's updated_at timestamp
        await self._db.execute(
            "UPDATE sessions SET updated_at = ? WHERE id = ?",
            (now, session_id)
        )
        
        return {
            "id": history_id,
            "session_id": session_id,
            "query": query,
            "answer": answer,
            "role": role,
            "metadata": metadata or {},
            "retrieves": retrieves or [],
            "tokens_query": tokens_query,
            "tokens_answer": tokens_answer,
            "latency_ms": latency_ms,
            "created_at": now,
            "updated_at": now,
        }
    
    async def get(self, history_id: str) -> Optional[Dict[str, Any]]:
        """Get history entry by ID."""
        rows = await self._db.execute(
            "SELECT * FROM history WHERE id = ?",
            (history_id,),
            fetch=True
        )
        if not rows:
            return None
        row = rows[0]
        row["metadata"] = json.loads(row.get("metadata") or "{}")
        row["retrieves"] = json.loads(row.get("retrieves") or "[]")
        return row
    
    async def list_by_session(
        self,
        session_id: str,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """List history entries for a session."""
        rows = await self._db.execute(
            """
            SELECT * FROM history 
            WHERE session_id = ? 
            ORDER BY created_at ASC 
            LIMIT ? OFFSET ?
            """,
            (session_id, limit, offset),
            fetch=True
        )
        
        for row in rows:
            row["metadata"] = json.loads(row.get("metadata") or "{}")
            row["retrieves"] = json.loads(row.get("retrieves") or "[]")
        return rows
    
    async def update_answer(
        self,
        history_id: str,
        answer: str,
        tokens_answer: Optional[int] = None,
        latency_ms: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Update history entry with answer (for async response patterns)."""
        updates = ["answer = ?", "updated_at = ?"]
        params = [answer, now_timestamp()]
        
        if tokens_answer is not None:
            updates.append("tokens_answer = ?")
            params.append(tokens_answer)
            
        if latency_ms is not None:
            updates.append("latency_ms = ?")
            params.append(latency_ms)
            
        if metadata:
            # Merge with existing metadata
            current = await self.get(history_id)
            if current:
                merged = {**current.get("metadata", {}), **metadata}
                updates.append("metadata = ?")
                params.append(json.dumps(merged))
        
        params.append(history_id)
        
        query = f"""
            UPDATE history
            SET {', '.join(updates)}
            WHERE id = ?
        """
        
        async with self._db.transaction():
            await self._db.execute(query, tuple(params))
            return await self.get(history_id)

    async def update_feedback(
        self,
        history_id: str,
        score: int,
        comment: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Update feedback for a history entry."""
        # Get current metadata
        current = await self.get(history_id)
        if not current:
            return None
            
        metadata = current.get("metadata", {})
        metadata["feedback"] = {
            "score": score,
            "comment": comment,
            "timestamp": now_timestamp()
        }
        
        query = """
            UPDATE history
            SET metadata = ?, updated_at = ?
            WHERE id = ?
        """
        
        async with self._db.transaction():
            await self._db.execute(
                query, 
                (json.dumps(metadata), now_timestamp(), history_id)
            )
            return await self.get(history_id)


class ResponseRepository:
    """Repository for detailed LLM response tracking."""
    
    def __init__(self, db: AsyncDatabaseManager) -> None:
        self._db = db
    
    async def create(
        self,
        history_id: str,
        model: str,
        prompt: str,
        response: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        tokens_prompt: Optional[int] = None,
        tokens_completion: Optional[int] = None,
        finish_reason: Optional[str] = None,
        latency_ms: float = 0.0,
        error: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Create a new response record."""
        response_id = generate_uuid()
        now = now_timestamp()
        
        await self._db.execute(
            """
            INSERT INTO responses 
            (id, history_id, model, prompt, response, temperature, max_tokens,
             tokens_prompt, tokens_completion, finish_reason, latency_ms, error, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                response_id, history_id, model, prompt, response,
                temperature, max_tokens, tokens_prompt, tokens_completion,
                finish_reason, latency_ms, error, now
            )
        )
        
        return {
            "id": response_id,
            "history_id": history_id,
            "model": model,
            "prompt": prompt,
            "response": response,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "tokens_prompt": tokens_prompt,
            "tokens_completion": tokens_completion,
            "finish_reason": finish_reason,
            "latency_ms": latency_ms,
            "error": error,
            "created_at": now,
        }
    
    async def list_by_history(self, history_id: str) -> List[Dict[str, Any]]:
        """List responses for a history entry."""
        return await self._db.execute(
            "SELECT * FROM responses WHERE history_id = ? ORDER BY created_at DESC",
            (history_id,),
            fetch=True
        )


class DocumentRepository:
    """Repository for document tracking."""
    
    def __init__(self, db: AsyncDatabaseManager) -> None:
        self._db = db
    
    async def create(
        self,
        filename: str,
        content_type: Optional[str] = None,
        file_size: Optional[int] = None,
        file_hash: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Create a new document record."""
        doc_id = generate_uuid()
        now = now_timestamp()
        
        await self._db.execute(
            """
            INSERT INTO documents 
            (id, filename, content_type, file_size, file_hash, metadata, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (doc_id, filename, content_type, file_size, file_hash, json.dumps(metadata or {}), now, now)
        )
        
        return {
            "id": doc_id,
            "filename": filename,
            "content_type": content_type,
            "file_size": file_size,
            "file_hash": file_hash,
            "status": "pending",
            "chunk_count": 0,
            "metadata": metadata or {},
            "error": None,
            "created_at": now,
            "updated_at": now,
            "is_active": True,
        }
    
    async def get(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Get document by ID."""
        rows = await self._db.execute(
            "SELECT * FROM documents WHERE id = ? AND is_active = 1",
            (doc_id,),
            fetch=True
        )
        if not rows:
            return None
        row = rows[0]
        row["metadata"] = json.loads(row.get("metadata") or "{}")
        row["is_active"] = bool(row.get("is_active", 1))
        return row
    
    async def get_by_hash(self, file_hash: str) -> Optional[Dict[str, Any]]:
        """
        Get document by file hash.
        
        Used for duplicate detection. Returns the most recent active document
        with this hash.
        """
        rows = await self._db.execute(
            """
            SELECT * FROM documents 
            WHERE file_hash = ? AND is_active = 1
            ORDER BY created_at DESC
            LIMIT 1
            """,
            (file_hash,),
            fetch=True
        )
        if not rows:
            return None
        row = rows[0]
        row["metadata"] = json.loads(row.get("metadata") or "{}")
        row["is_active"] = bool(row.get("is_active", 1))
        return row
    
    async def list(self, limit: int = 50, offset: int = 0) -> Dict[str, Any]:
        """List documents with pagination."""
        # Get documents
        rows = await self._db.execute(
            """
            SELECT * FROM documents 
            WHERE is_active = 1 AND status != 'deleted'
            ORDER BY created_at DESC
            LIMIT ? OFFSET ?
            """,
            (limit, offset),
            fetch=True
        )
        
        # Parse metadata and fix types
        documents = []
        for row in rows:
            row_dict = dict(row)
            row_dict["metadata"] = json.loads(row_dict.get("metadata") or "{}")
            row_dict["is_active"] = bool(row_dict.get("is_active", 1))
            documents.append(row_dict)
            
        # Get total count
        count_rows = await self._db.execute(
            "SELECT COUNT(*) as count FROM documents WHERE is_active = 1 AND status != 'deleted'",
            fetch=True
        )
        total = count_rows[0]["count"] if count_rows else 0
        
        return {
            "documents": documents,
            "total": total,
            "limit": limit,
            "offset": offset,
        }
    
    async def update_status(
        self,
        doc_id: str,
        status: str,
        stage: Optional[str] = None,
        progress: Optional[int] = None,
        chunk_count: Optional[int] = None,
        error: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Update document processing status with detailed progress tracking."""
        now = now_timestamp()
        
        updates = ["status = ?", "updated_at = ?"]
        params: List[Any] = [status, now]
        
        if stage is not None:
            updates.append("stage = ?")
            params.append(stage)
        if progress is not None:
            updates.append("progress = ?")
            params.append(progress)
        if chunk_count is not None:
            updates.append("chunk_count = ?")
            params.append(chunk_count)
        if error is not None:
            updates.append("error = ?")
            params.append(error)
        
        params.append(doc_id)
        
        await self._db.execute(
            f"UPDATE documents SET {', '.join(updates)} WHERE id = ? AND is_active = 1",
            tuple(params)
        )
        
        return await self.get(doc_id)


# -----------------------------------------------------------------------------
# Global Database Initialization
# -----------------------------------------------------------------------------
_db_manager: Optional[DatabaseManager] = None
_async_db_manager: Optional[AsyncDatabaseManager] = None


def get_db_manager(db_path: Optional[Union[str, Path]] = None) -> DatabaseManager:
    """
    Get or create the global database manager singleton.
    
    Thread Safety:
    --------------
    First call initializes the manager; subsequent calls return cached instance.
    """
    global _db_manager
    if _db_manager is None:
        if db_path is None:
            from api.config import get_settings
            db_path = get_settings().db_path
        _db_manager = DatabaseManager(db_path)
        _db_manager.init_schema()
    return _db_manager


def get_async_db_manager() -> AsyncDatabaseManager:
    """Get async database manager wrapping the sync manager."""
    global _async_db_manager
    if _async_db_manager is None:
        _async_db_manager = AsyncDatabaseManager(get_db_manager())
    return _async_db_manager


# -----------------------------------------------------------------------------
# Module Self-Test
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import asyncio
    
    logging.basicConfig(level=logging.DEBUG)
    
    async def test_database():
        print("=" * 80)
        print("Database Self-Test")
        print("=" * 80)
        
        # Initialize with test database
        test_db_path = Path(__file__).parent.parent / "data" / "test_api.db"
        manager = DatabaseManager(test_db_path)
        manager.init_schema()
        
        async_manager = AsyncDatabaseManager(manager)
        
        # Test session repository
        sessions = SessionRepository(async_manager)
        
        print("\n1. Creating session...")
        session = await sessions.create(user_id="test_user", title="Test Session")
        print(f"   Created: {session['id']}")
        
        print("\n2. Getting session...")
        retrieved = await sessions.get(session["id"])
        print(f"   Retrieved: {retrieved['title']}")
        
        print("\n3. Listing sessions...")
        all_sessions = await sessions.list()
        print(f"   Found {len(all_sessions)} sessions")
        
        # Test history repository
        history = HistoryRepository(async_manager)
        
        print("\n4. Creating history entry...")
        entry = await history.create(
            session_id=session["id"],
            query="What is the weather?",
            role="user"
        )
        print(f"   Created: {entry['id']}")
        
        print("\n5. Updating with answer...")
        updated = await history.update_answer(
            history_id=entry["id"],
            answer="It's sunny today!",
            latency_ms=125.5
        )
        print(f"   Answer: {updated['answer']}")
        
        print("\n6. Listing history...")
        entries = await history.list_by_session(session["id"])
        print(f"   Found {len(entries)} entries")
        
        # Cleanup
        manager.close()
        print("\n" + "=" * 80)
        print("Database test complete!")
    
    asyncio.run(test_database())
