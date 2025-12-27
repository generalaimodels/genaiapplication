# -*- coding: utf-8 -*-
# =================================================================================================
# api/cache.py — RAG Caching Infrastructure
# =================================================================================================
# Async-safe caching layer for RAG pipeline optimization:
#
#   1. EMBEDDING CACHE: Cache query embeddings to avoid re-computation
#   2. RETRIEVAL CACHE: Cache search results for repeated queries
#   3. RESPONSE CACHE: Cache full responses for frequent queries
#
# Features:
# ---------
#   - LRU eviction strategy
#   - TTL-based expiration
#   - Async-safe with asyncio.Lock
#   - Hit/miss statistics
#   - Memory-efficient ordered dict storage
#
# Usage:
# ------
#   from api.cache import get_retrieval_cache, hash_query
#   
#   cache = get_retrieval_cache()
#   key = hash_query("my query", top_k=5)
#   
#   cached = await cache.get(key)
#   if cached:
#       return cached
#   
#   result = await expensive_retrieval()
#   await cache.set(key, result)
#
# =================================================================================================

from __future__ import annotations

import asyncio
import hashlib
import logging
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict, Generic, List, Optional, Tuple, TypeVar

_LOG = logging.getLogger("api.cache")

T = TypeVar("T")


# =============================================================================
# Cache Entry
# =============================================================================

@dataclass
class CacheEntry(Generic[T]):
    """
    Cache entry with TTL and access tracking.
    
    Attributes:
        value: Cached value
        created_at: Unix timestamp when entry was created
        ttl_seconds: Time-to-live in seconds
        hits: Number of times this entry was accessed
        last_accessed: Unix timestamp of last access
    """
    value: T
    created_at: float
    ttl_seconds: float
    hits: int = 0
    last_accessed: float = field(default_factory=time.time)
    
    @property
    def is_expired(self) -> bool:
        """Check if entry has exceeded its TTL."""
        return time.time() - self.created_at > self.ttl_seconds
    
    @property
    def age_seconds(self) -> float:
        """Get age of entry in seconds."""
        return time.time() - self.created_at
    
    def touch(self) -> None:
        """Update access tracking."""
        self.hits += 1
        self.last_accessed = time.time()


# =============================================================================
# Async LRU Cache
# =============================================================================

class AsyncLRUCache(Generic[T]):
    """
    Async-safe LRU cache with TTL support.
    
    Features:
    ---------
    - LRU eviction when max size reached
    - TTL expiration for stale entries
    - Thread-safe with asyncio.Lock
    - Hit/miss statistics
    - Batch operations for efficiency
    
    Algorithm:
    ----------
    Uses OrderedDict for O(1) access and LRU ordering.
    Items are moved to end on access (most recently used).
    Eviction removes from front (least recently used).
    
    Memory:
    -------
    Each entry stores value + metadata (~100 bytes overhead).
    Total memory ≈ maxsize × (avg_value_size + 100 bytes)
    """
    
    def __init__(
        self,
        maxsize: int = 1000,
        ttl_seconds: float = 300.0,
        name: str = "cache",
    ):
        """
        Initialize cache.
        
        Args:
            maxsize: Maximum number of entries (default 1000)
            ttl_seconds: Default TTL for entries (default 5 minutes)
            name: Cache name for logging
        """
        self.maxsize = maxsize
        self.ttl_seconds = ttl_seconds
        self.name = name
        
        self._cache: OrderedDict[str, CacheEntry[T]] = OrderedDict()
        self._lock = asyncio.Lock()
        
        # Statistics
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        self._expirations = 0
    
    async def get(self, key: str) -> Optional[T]:
        """
        Get value from cache.
        
        Returns None if key not found or entry expired.
        Updates LRU ordering on hit.
        """
        async with self._lock:
            entry = self._cache.get(key)
            
            if entry is None:
                self._misses += 1
                return None
            
            if entry.is_expired:
                del self._cache[key]
                self._misses += 1
                self._expirations += 1
                return None
            
            # Update LRU ordering (move to end)
            self._cache.move_to_end(key)
            entry.touch()
            self._hits += 1
            
            return entry.value
    
    async def set(
        self,
        key: str,
        value: T,
        ttl: Optional[float] = None,
    ) -> None:
        """
        Set value in cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Optional TTL override (uses default if None)
        """
        async with self._lock:
            # Evict if at capacity (before adding new entry)
            while len(self._cache) >= self.maxsize:
                # Remove least recently used (first item)
                evicted_key, _ = self._cache.popitem(last=False)
                self._evictions += 1
                _LOG.debug("%s: Evicted %s", self.name, evicted_key[:16])
            
            # Add new entry
            self._cache[key] = CacheEntry(
                value=value,
                created_at=time.time(),
                ttl_seconds=ttl if ttl is not None else self.ttl_seconds,
            )
    
    async def delete(self, key: str) -> bool:
        """Delete entry from cache. Returns True if key was found."""
        async with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False
    
    async def clear(self) -> int:
        """Clear all entries. Returns count of cleared entries."""
        async with self._lock:
            count = len(self._cache)
            self._cache.clear()
            return count
    
    async def get_many(self, keys: List[str]) -> Dict[str, T]:
        """Get multiple values at once (more efficient than individual gets)."""
        results = {}
        async with self._lock:
            for key in keys:
                entry = self._cache.get(key)
                if entry and not entry.is_expired:
                    self._cache.move_to_end(key)
                    entry.touch()
                    self._hits += 1
                    results[key] = entry.value
                else:
                    self._misses += 1
                    if entry and entry.is_expired:
                        del self._cache[key]
                        self._expirations += 1
        return results
    
    async def set_many(
        self,
        items: Dict[str, T],
        ttl: Optional[float] = None,
    ) -> None:
        """Set multiple values at once."""
        async with self._lock:
            for key, value in items.items():
                # Evict if needed
                while len(self._cache) >= self.maxsize:
                    self._cache.popitem(last=False)
                    self._evictions += 1
                
                self._cache[key] = CacheEntry(
                    value=value,
                    created_at=time.time(),
                    ttl_seconds=ttl if ttl is not None else self.ttl_seconds,
                )
    
    async def cleanup_expired(self) -> int:
        """Remove all expired entries. Returns count removed."""
        async with self._lock:
            expired_keys = [
                key for key, entry in self._cache.items()
                if entry.is_expired
            ]
            for key in expired_keys:
                del self._cache[key]
                self._expirations += 1
            return len(expired_keys)
    
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = self._hits + self._misses
        hit_rate = self._hits / total_requests if total_requests > 0 else 0.0
        
        return {
            "name": self.name,
            "size": len(self._cache),
            "maxsize": self.maxsize,
            "ttl_seconds": self.ttl_seconds,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": round(hit_rate, 4),
            "evictions": self._evictions,
            "expirations": self._expirations,
        }
    
    async def info(self) -> Dict[str, Any]:
        """Get detailed cache info including entry ages."""
        async with self._lock:
            if not self._cache:
                return {
                    **self.stats(),
                    "oldest_entry_age_s": 0,
                    "newest_entry_age_s": 0,
                }
            
            ages = [entry.age_seconds for entry in self._cache.values()]
            
            return {
                **self.stats(),
                "oldest_entry_age_s": max(ages),
                "newest_entry_age_s": min(ages),
                "avg_entry_age_s": sum(ages) / len(ages),
            }


# =============================================================================
# Cache Key Helpers
# =============================================================================

def hash_query(query: str, **kwargs) -> str:
    """
    Generate cache key from query and options.
    
    Creates a stable hash from query text and any additional
    parameters that affect the result.
    
    Args:
        query: Query string
        **kwargs: Additional parameters (top_k, collection, etc.)
    
    Returns:
        32-character hex hash
    """
    key_parts = [query.strip().lower()]
    
    # Sort kwargs for consistent ordering
    for k, v in sorted(kwargs.items()):
        if v is not None:
            key_parts.append(f"{k}={v}")
    
    key_str = "|".join(key_parts)
    return hashlib.sha256(key_str.encode("utf-8")).hexdigest()[:32]


def hash_embedding_input(text: str, model: str = "default") -> str:
    """
    Generate cache key for embedding input.
    
    Args:
        text: Text to embed
        model: Embedding model name
    
    Returns:
        32-character hex hash
    """
    key_str = f"{model}|{text.strip()}"
    return hashlib.sha256(key_str.encode("utf-8")).hexdigest()[:32]


# =============================================================================
# Global Cache Instances
# =============================================================================

# Cache singletons
_embedding_cache: Optional[AsyncLRUCache] = None
_retrieval_cache: Optional[AsyncLRUCache] = None
_response_cache: Optional[AsyncLRUCache] = None


def get_embedding_cache() -> AsyncLRUCache:
    """
    Get embedding cache (long TTL - embeddings don't change).
    
    Caches query embeddings to avoid re-computing expensive
    embedding operations for repeated queries.
    
    Config:
    - maxsize: 5000 entries
    - ttl: 1 hour
    """
    global _embedding_cache
    if _embedding_cache is None:
        _embedding_cache = AsyncLRUCache(
            maxsize=5000,
            ttl_seconds=3600.0,  # 1 hour
            name="embedding_cache",
        )
        _LOG.info("Embedding cache initialized: maxsize=5000, ttl=3600s")
    return _embedding_cache


def get_retrieval_cache() -> AsyncLRUCache:
    """
    Get retrieval results cache (medium TTL).
    
    Caches search/retrieval results for repeated queries.
    Invalidated when documents are added/updated.
    
    Config:
    - maxsize: 1000 entries  
    - ttl: 5 minutes
    """
    global _retrieval_cache
    if _retrieval_cache is None:
        _retrieval_cache = AsyncLRUCache(
            maxsize=1000,
            ttl_seconds=300.0,  # 5 minutes
            name="retrieval_cache",
        )
        _LOG.info("Retrieval cache initialized: maxsize=1000, ttl=300s")
    return _retrieval_cache


def get_response_cache() -> AsyncLRUCache:
    """
    Get full response cache (short TTL for freshness).
    
    Caches complete RAG responses for very frequent queries.
    Short TTL ensures users get fresh content.
    
    Config:
    - maxsize: 500 entries
    - ttl: 1 minute
    """
    global _response_cache
    if _response_cache is None:
        _response_cache = AsyncLRUCache(
            maxsize=500,
            ttl_seconds=60.0,  # 1 minute
            name="response_cache",
        )
        _LOG.info("Response cache initialized: maxsize=500, ttl=60s")
    return _response_cache


async def clear_all_caches() -> Dict[str, int]:
    """Clear all cache instances. Returns count of cleared entries per cache."""
    results = {}
    
    if _embedding_cache:
        results["embedding"] = await _embedding_cache.clear()
    if _retrieval_cache:
        results["retrieval"] = await _retrieval_cache.clear()
    if _response_cache:
        results["response"] = await _response_cache.clear()
    
    _LOG.info("All caches cleared: %s", results)
    return results


def get_all_cache_stats() -> Dict[str, Dict[str, Any]]:
    """Get statistics for all cache instances."""
    stats = {}
    
    if _embedding_cache:
        stats["embedding"] = _embedding_cache.stats()
    if _retrieval_cache:
        stats["retrieval"] = _retrieval_cache.stats()
    if _response_cache:
        stats["response"] = _response_cache.stats()
    
    return stats


async def invalidate_retrieval_cache() -> int:
    """
    Invalidate retrieval cache.
    
    Call this when documents are added/updated/deleted
    to ensure search results reflect latest content.
    """
    if _retrieval_cache:
        count = await _retrieval_cache.clear()
        _LOG.info("Retrieval cache invalidated: %d entries cleared", count)
        return count
    return 0
