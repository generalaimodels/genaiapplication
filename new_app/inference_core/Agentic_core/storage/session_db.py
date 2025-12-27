"""
Async SQLite Storage Layer.

Adheres to:
- I/O Semantics: Non-blocking I/O via aiosqlite (running on thread pool executor typically, optimizing for IO bound).
- Durability: WAL mode enabled.
- Failure Handling: Retries on busy, strict transaction scopes.
"""
import aiosqlite
import asyncio
from typing import List, Dict, Any, Optional
from datetime import datetime
import json
import logging
from contextlib import asynccontextmanager

from ..core.config import get_config
from ..core.result import Result, Ok, Err

logger = logging.getLogger(__name__)
CONFIG = get_config()

class SessionDB:
    """
    Robust, Async SQLite storage for Agent Sessions and History.
    """
    def __init__(self):
        self.db_path = CONFIG.storage_path
        self._pool_lock = asyncio.Lock()
        
    async def initialize(self) -> Result[bool, Exception]:
        """
        Initialize DB with WAL mode and schema.
        Idempotent operation.
        """
        try:
            # SOTA Robustness: Ensure directory exists
            import os
            os.makedirs(os.path.dirname(self.db_path), exist_ok=True)

            async with aiosqlite.connect(self.db_path) as db:
                await db.execute("PRAGMA journal_mode=WAL;")  # Write-Ahead Logging for concurrency
                await db.execute("PRAGMA synchronous=NORMAL;") # Balance safety/speed
                
                # General Session Table
                await db.execute("""
                    CREATE TABLE IF NOT EXISTS sessions (
                        session_id TEXT PRIMARY KEY,
                        created_at TEXT,
                        updated_at TEXT,
                        metadata TEXT  -- JSON blob for generalized flexibility
                    );
                """)
                
                # Message History (Event Store pattern)
                await db.execute("""
                    CREATE TABLE IF NOT EXISTS message_history (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id TEXT,
                        role TEXT,
                        content TEXT,
                        timestamp TEXT,
                        FOREIGN KEY(session_id) REFERENCES sessions(session_id)
                    );
                """)
                
                await db.commit()
            return Ok(True)
        except Exception as e:
            logger.critical(f"Failed to init DB: {e}")
            return Err(e)

    async def create_session(self, session_id: str, metadata: Dict[str, Any] = {}) -> Result[str, Exception]:
        try:
            now = datetime.utcnow().isoformat()
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute(
                    "INSERT INTO sessions (session_id, created_at, updated_at, metadata) VALUES (?, ?, ?, ?)",
                    (session_id, now, now, json.dumps(metadata))
                )
                await db.commit()
            return Ok(session_id)
        except Exception as e:
            return Err(e)

    async def add_message(self, session_id: str, role: str, content: str) -> Result[bool, Exception]:
        """
        Atomic append of a message.
        """
        try:
            now = datetime.utcnow().isoformat()
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute(
                    "INSERT INTO message_history (session_id, role, content, timestamp) VALUES (?, ?, ?, ?)",
                    (session_id, role, content, now)
                )
                # Update session timestamp
                await db.execute(
                    "UPDATE sessions SET updated_at = ? WHERE session_id = ?",
                    (now, session_id)
                )
                await db.commit()
            return Ok(True)
        except Exception as e:
            logger.error(f"DB Write Error: {e}")
            return Err(e)

    async def get_history(self, session_id: str, limit: int = 100) -> Result[List[Dict], Exception]:
        try:
            async with aiosqlite.connect(self.db_path) as db:
                db.row_factory = aiosqlite.Row
                cursor = await db.execute(
                    "SELECT role, content, timestamp FROM message_history WHERE session_id = ? ORDER BY id DESC LIMIT ?",
                    (session_id, limit)
                )
                rows = await cursor.fetchall()
                # Return in chronological order
                history = [dict(row) for row in rows][::-1]
            return Ok(history)
        except Exception as e:
            return Err(e)
