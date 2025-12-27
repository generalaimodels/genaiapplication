"""
Robust Retry Logic (Exponential Backoff).

Adheres to:
- Failure Domain Analysis: Handle transient failures gracefully.
"""
import asyncio
import logging
import random
from typing import TypeVar, Callable, Awaitable, Union

logger = logging.getLogger(__name__)

T = TypeVar("T")

async def retry_with_backoff(
    func: Callable[[], Awaitable[T]],
    retries: int = 3,
    base_delay: float = 0.5,
    max_delay: float = 10.0,
    exceptions: tuple = (Exception,)
) -> T:
    """
    Executes an async function with exponential backoff and jitter.
    """
    attempt = 0
    while True:
        try:
            return await func()
        except exceptions as e:
            attempt += 1
            if attempt > retries:
                logger.error(f"Max retries reached for {func.__name__}")
                raise e
            
            # Exponential backoff + Jitter
            delay = min(base_delay * (2 ** (attempt - 1)), max_delay)
            jitter = delay * 0.1 * random.random()
            sleep_time = delay + jitter
            
            logger.warning(f"Retry {attempt}/{retries} for {func.__name__} due to {e}. Sleeping {sleep_time:.2f}s")
            await asyncio.sleep(sleep_time)
