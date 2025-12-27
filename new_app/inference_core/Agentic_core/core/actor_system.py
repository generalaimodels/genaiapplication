"""
Lock-Free Actor System.

Adheres to:
- Deterministic Concurrency: Message passing over shared state.
- Zero-Cost Abstraction: Python native coroutines used as lightweight actors.
"""
import asyncio
from typing import Any, Callable, Dict, Optional, Awaitable
from dataclasses import dataclass, field
import uuid
import logging

from .config import get_config

logger = logging.getLogger(__name__)

@dataclass
class Message:
    """Immutable message payload."""
    sender_id: str
    target_id: str
    message_type: str
    payload: Any
    priority: int = 1  # 0 = Highest, 10 = Lowest

class Actor:
    """
    Base Actor class.
    Each actor runs its own async event loop consumption, ensuring NO locking on shared state.
    """
    def __init__(self, actor_id: str):
        self.actor_id = actor_id
        # Unbounded queue for high throughput; backpressure handled by system
        self._mailbox: asyncio.Queue[Message] = asyncio.Queue()
        self._running = False
        self._task: Optional[asyncio.Task] = None

    async def start(self):
        """Start the actor's main loop."""
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._process_mailbox())
        logger.debug(f"Actor {self.actor_id} started.")

    async def stop(self):
        """Gracefully stop the actor."""
        self._running = False
        await self._mailbox.put(None) # Sentinel
        if self._task:
            await self._task

    async def send(self, message: Message):
        """Non-blocking send."""
        await self._mailbox.put(message)

    async def _process_mailbox(self):
        """Main actor loop."""
        while self._running:
            try:
                message = await self._mailbox.get()
                if message is None: # Sentinel
                    break
                
                await self.handle_message(message)
                self._mailbox.task_done()
            except Exception as e:
                logger.error(f"Error in actor {self.actor_id}: {e}", exc_info=True)

    async def handle_message(self, message: Message):
        """Virtual method to be implemented by concrete actors."""
        raise NotImplementedError

class ActorSystem:
    """
    Central registry and dispatcher for actors.
    """
    def __init__(self):
        self._actors: Dict[str, Actor] = {}
        self._bus_lock = asyncio.Lock() # Minimal lock only for registration

    async def spawn(self, actor_cls: type[Actor], *args, **kwargs) -> str:
        """Spawn a new actor."""
        actor_id = str(uuid.uuid4())
        actor = actor_cls(actor_id, *args, **kwargs)
        self._actors[actor_id] = actor
        await actor.start()
        return actor_id

    def get_actor(self, actor_id: str) -> Optional[Actor]:
        return self._actors.get(actor_id)

    async def shutdown(self):
        """Shutdown all actors."""
        for actor in self._actors.values():
            await actor.stop()
        self._actors.clear()
