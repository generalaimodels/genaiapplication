"""
Memory Archiver.

Adheres to:
- Robustness: Runs as background task.
- Efficiency: Batches writes to SQLite to reduce I/O pressure.
"""
import asyncio
import logging
from typing import List, Dict
from ..storage.session_db import SessionDB
from ..core.config import get_config

logger = logging.getLogger(__name__)

class MemoryArchiver:
    def __init__(self, session_id: str, db: SessionDB):
        self.session_id = session_id
        self.db = db
        self.queue = asyncio.Queue()
        self._shutdown = False
        self._worker_task = None

    async def start(self):
        self._worker_task = asyncio.create_task(self._worker())

    async def stop(self):
        self._shutdown = True
        await self.queue.put(None) # Sentinel
        if self._worker_task:
            await self._worker_task

    async def archive_message(self, role: str, content: str):
        await self.queue.put({"role": role, "content": content})

    async def _worker(self):
        buffer = []
        while not self._shutdown or not self.queue.empty():
            try:
                # Batch processing
                item = await asyncio.wait_for(self.queue.get(), timeout=2.0)
                if item is None:
                    break
                
                buffer.append(item)
                
                # Flush if buffer gets big
                if len(buffer) >= 10:
                    await self._flush(buffer)
                    buffer = []
                    
            except asyncio.TimeoutError:
                # Idle flush
                if buffer:
                    await self._flush(buffer)
                    buffer = []
            except Exception as e:
                logger.error(f"Archiver error: {e}")
        
        # Final flush
        if buffer:
            await self._flush(buffer)

    async def _flush(self, items: List[Dict]):
        # In SOTA, we would do a batch insert. 
        # Our SessionDB currently has single add_message, but SQLite handles concurrency well.
        # We could optimize SessionDB to accept batch.
        for item in items:
            await self.db.add_message(self.session_id, item['role'], item['content'])
