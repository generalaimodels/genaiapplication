"""
High-Performance Async Utilities.

Adheres to:
- Zero-Cost Abstraction: Minimal overhead wrappers around native asyncio.
- Deterministic Concurrency: Explicit concurrency limits and timeout controls.
- Failure Domain: Result types and exception boundaries.
- Cache Locality: LRU cache with bounded memory footprint.
"""
import asyncio
import functools
import time
import logging
from typing import TypeVar, Callable, Any, Awaitable, Optional, Dict, Tuple
from collections import OrderedDict
from ..core.result import Result, Ok, Err

logger = logging.getLogger(__name__)

T = TypeVar('T')

# ============================================================================
# ASYNC PRIMITIVES DESIGN
# ============================================================================
# Goals:
# 1. Bounded parallelism: Prevent resource exhaustion
# 2. Retry with backoff: Handle transient failures
# 3. Timeout enforcement: Circuit breaker for hanging operations
# 4. Async caching: Reduce redundant async operations
#
# Performance Targets:
# - gather_with_concurrency: O(n/k) where n=tasks, k=concurrency
# - retry_with_backoff: O(attempts) with exponential delay
# - async_lru_cache: O(1) hit, O(log n) eviction (OrderedDict)
# ============================================================================

async def gather_with_concurrency(
    limit: int,
    *coros: Awaitable[T],
    return_exceptions: bool = False
) -> list[T]:
    """
    Execute coroutines with bounded concurrency.
    
    Complexity: O(n/k) execution time where n=total, k=limit
    
    Args:
        limit: Maximum concurrent executions
        *coros: Coroutines to execute
        return_exceptions: If True, return exceptions instead of raising
        
    Returns:
        List of results in input order
        
    Example:
        results = await gather_with_concurrency(5, *[fetch(url) for url in urls])
    """
    semaphore = asyncio.Semaphore(limit)
    
    async def bounded_coro(coro: Awaitable[T]) -> T:
        async with semaphore:
            return await coro
    
    return await asyncio.gather(
        *[bounded_coro(c) for c in coros],
        return_exceptions=return_exceptions
    )


async def retry_with_backoff(
    coro_fn: Callable[[], Awaitable[T]],
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    jitter: bool = True
) -> Result[T, Exception]:
    """
    Execute coroutine with exponential backoff retry.
    
    Retry delay formula:
    delay = min(base_delay * (exponential_base ** attempt), max_delay)
    With jitter: delay *= random(0.5, 1.5)
    
    Complexity: O(max_retries) attempts
    
    Args:
        coro_fn: Callable that returns awaitable (called fresh each retry)
        max_retries: Maximum retry attempts
        base_delay: Initial delay in seconds
        max_delay: Maximum delay cap
        exponential_base: Base for exponential growth (default: 2.0)
        jitter: Add randomness to prevent thundering herd
        
    Returns:
        Ok(result) on success, Err(last_exception) on failure
        
    Example:
        result = await retry_with_backoff(
            lambda: client.fetch(url),
            max_retries=5,
            base_delay=1.0
        )
    """
    import random
    
    last_exception = None
    
    for attempt in range(max_retries + 1):
        try:
            result = await coro_fn()
            if attempt > 0:
                logger.info(f"Retry succeeded on attempt {attempt + 1}")
            return Ok(result)
            
        except Exception as e:
            last_exception = e
            
            if attempt < max_retries:
                # Calculate delay with exponential backoff
                delay = min(base_delay * (exponential_base ** attempt), max_delay)
                
                # Add jitter to prevent thundering herd
                if jitter:
                    delay *= random.uniform(0.5, 1.5)
                
                logger.warning(
                    f"Attempt {attempt + 1}/{max_retries + 1} failed: {e}. "
                    f"Retrying in {delay:.2f}s..."
                )
                
                await asyncio.sleep(delay)
            else:
                logger.error(f"All {max_retries + 1} attempts failed")
    
    return Err(last_exception or Exception("Retry failed with unknown error"))


def timeout_decorator(seconds: float):
    """
    Decorator to enforce timeout on async functions.
    
    Args:
        seconds: Timeout duration
        
    Example:
        @timeout_decorator(10.0)
        async def slow_operation():
            await asyncio.sleep(20)  # Will raise TimeoutError
    """
    def decorator(func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            try:
                return await asyncio.wait_for(
                    func(*args, **kwargs),
                    timeout=seconds
                )
            except asyncio.TimeoutError:
                logger.error(f"{func.__name__} timed out after {seconds}s")
                raise
        
        return wrapper
    return decorator


class AsyncLRUCache:
    """
    Thread-safe async LRU cache.
    
    Performance Characteristics:
    - Get: O(1) average (dict lookup)
    - Put: O(1) average
    - Eviction: O(1) (remove oldest from OrderedDict)
    - Memory: O(max_size * avg_value_size)
    
    Thread Safety: Protected by asyncio.Lock
    """
    
    def __init__(self, max_size: int = 1024, ttl: Optional[float] = None):
        """
        Initialize LRU cache.
        
        Args:
            max_size: Maximum cache entries
            ttl: Time-to-live in seconds (None = no expiration)
        """
        assert max_size > 0, "max_size must be positive"
        
        self.max_size = max_size
        self.ttl = ttl
        self._cache: OrderedDict[Any, Tuple[Any, float]] = OrderedDict()
        self._lock = asyncio.Lock()
        self._hits = 0
        self._misses = 0
    
    async def get(self, key: Any) -> Optional[Any]:
        """
        Retrieve value from cache.
        
        Complexity: O(1) average case
        
        Returns: Cached value or None if not found/expired
        """
        async with self._lock:
            if key not in self._cache:
                self._misses += 1
                return None
            
            value, timestamp = self._cache[key]
            
            # Check TTL
            if self.ttl is not None:
                if time.time() - timestamp > self.ttl:
                    # Expired, remove
                    del self._cache[key]
                    self._misses += 1
                    return None
            
            # Move to end (mark as recently used)
            self._cache.move_to_end(key)
            self._hits += 1
            return value
    
    async def put(self, key: Any, value: Any) -> None:
        """
        Store value in cache.
        
        Complexity: O(1) average case
        """
        async with self._lock:
            # Update existing key
            if key in self._cache:
                self._cache.move_to_end(key)
                self._cache[key] = (value, time.time())
                return
            
            # Add new key
            self._cache[key] = (value, time.time())
            
            # Evict oldest if over capacity
            if len(self._cache) > self.max_size:
                # FIFO eviction (popitem(last=False) removes oldest)
                evicted_key, _ = self._cache.popitem(last=False)
                logger.debug(f"LRU evicted key: {evicted_key}")
    
    async def invalidate(self, key: Any) -> bool:
        """
        Remove specific key from cache.
        
        Returns: True if key existed, False otherwise
        """
        async with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False
    
    async def clear(self) -> None:
        """
        Clear entire cache.
        """
        async with self._lock:
            self._cache.clear()
            self._hits = 0
            self._misses = 0
    
    async def stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        """
        async with self._lock:
            total_requests = self._hits + self._misses
            hit_rate = self._hits / total_requests if total_requests > 0 else 0.0
            
            return {
                "size": len(self._cache),
                "max_size": self.max_size,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": hit_rate,
                "ttl": self.ttl
            }


def async_lru_cache(max_size: int = 128, ttl: Optional[float] = None):
    """
    Decorator for async function memoization with LRU eviction.
    
    Args:
        max_size: Maximum cached results
        ttl: Time-to-live for cached results (seconds)
        
    Example:
        @async_lru_cache(max_size=256, ttl=60.0)
        async def expensive_computation(x: int) -> int:
            await asyncio.sleep(1)
            return x * x
    """
    cache = AsyncLRUCache(max_size=max_size, ttl=ttl)
    
    def decorator(func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            # Create cache key from args and kwargs
            cache_key = (args, tuple(sorted(kwargs.items())))
            
            # Try cache first
            cached = await cache.get(cache_key)
            if cached is not None:
                logger.debug(f"Cache hit: {func.__name__}{args}")
                return cached
            
            # Execute and cache
            logger.debug(f"Cache miss: {func.__name__}{args}")
            result = await func(*args, **kwargs)
            await cache.put(cache_key, result)
            
            return result
        
        # Attach cache for introspection
        wrapper.cache = cache  # type: ignore
        return wrapper
    
    return decorator


async def run_with_timeout(
    coro: Awaitable[T],
    timeout: float,
    default: Optional[T] = None
) -> T:
    """
    Execute coroutine with timeout, returning default on timeout.
    
    Args:
        coro: Coroutine to execute
        timeout: Timeout in seconds
        default: Default value to return on timeout
        
    Returns:
        Result or default
        
    Raises:
        asyncio.TimeoutError if default is None
    """
    try:
        return await asyncio.wait_for(coro, timeout=timeout)
    except asyncio.TimeoutError:
        if default is not None:
            logger.warning(f"Operation timed out after {timeout}s, returning default")
            return default
        raise


async def batch_execute(
    items: list[T],
    async_fn: Callable[[T], Awaitable[Any]],
    batch_size: int = 10,
    delay_between_batches: float = 0.0
) -> list[Any]:
    """
    Execute async function on items in batches.
    
    Useful for rate-limited APIs or preventing resource exhaustion.
    
    Complexity: O(n/b * d) where n=items, b=batch_size, d=delay
    
    Args:
        items: Items to process
        async_fn: Async function to apply to each item
        batch_size: Items per batch
        delay_between_batches: Delay in seconds between batches
        
    Returns:
        List of results in input order
    """
    results = []
    
    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]
        logger.debug(f"Processing batch {i//batch_size + 1} ({len(batch)} items)")
        
        batch_results = await asyncio.gather(*[async_fn(item) for item in batch])
        results.extend(batch_results)
        
        # Delay between batches (rate limiting)
        if delay_between_batches > 0 and i + batch_size < len(items):
            await asyncio.sleep(delay_between_batches)
    
    return results
