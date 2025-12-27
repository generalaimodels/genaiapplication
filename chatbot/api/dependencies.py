# -*- coding: utf-8 -*-
# =================================================================================================
# api/dependencies.py — FastAPI Dependency Injection
# =================================================================================================
# Production-grade dependency injection implementing:
#
#   1. SERVICE SINGLETONS: Heavy services instantiated once, reused across requests.
#   2. LAZY LOADING: Services loaded on first use to reduce startup time.
#   3. REQUEST SCOPE: Per-request dependencies (db sessions, auth context).
#   4. LIFECYCLE MANAGEMENT: Proper cleanup on shutdown.
#   5. TYPE HINTS: Full typing for IDE support and documentation.
#
# Dependency Categories:
# ----------------------
#   - Configuration: Settings access.
#   - Database: Async database manager and repositories.
#   - Services: AI services (history, vector, LLM).
#   - Authentication: API key validation.
#   - Rate Limiting: Request quotas.
#
# Usage in Routers:
# -----------------
#   from api.dependencies import get_session_repo, get_llm_client
#   
#   @router.post("/chat")
#   async def chat(
#       request: ChatRequest,
#       sessions: SessionRepository = Depends(get_session_repo),
#       llm: LLMClient = Depends(get_llm_client),
#   ):
#       ...
#
# =================================================================================================

from __future__ import annotations

import asyncio
import logging
from functools import lru_cache
from typing import Any, AsyncGenerator, Generator, Optional, TYPE_CHECKING

from fastapi import Depends, Header, HTTPException, Request, status

# -----------------------------------------------------------------------------
# Type Checking Imports (avoid circular imports)
# -----------------------------------------------------------------------------
if TYPE_CHECKING:
    from api.config import Settings
    from api.database import (
        AsyncDatabaseManager,
        DatabaseManager,
        DocumentRepository,
        HistoryRepository,
        ResponseRepository,
        SessionRepository,
    )

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
_LOG = logging.getLogger("api.dependencies")


# =============================================================================
# Configuration Dependencies
# =============================================================================

@lru_cache(maxsize=1)
def get_settings() -> "Settings":
    """
    Get cached application settings.
    
    Singleton Pattern:
    ------------------
    Settings loaded once on first call, cached permanently.
    Use clear_settings_cache() in tests to reload.
    """
    from api.config import get_settings as load_settings
    return load_settings()


# =============================================================================
# Database Dependencies
# =============================================================================

_db_manager_instance: Optional["DatabaseManager"] = None
_async_db_manager_instance: Optional["AsyncDatabaseManager"] = None


def get_db_manager() -> "DatabaseManager":
    """
    Get synchronous database manager singleton.
    
    Thread Safety:
    --------------
    First call initializes manager; subsequent calls return cached instance.
    """
    global _db_manager_instance
    if _db_manager_instance is None:
        from api.database import DatabaseManager
        settings = get_settings()
        _db_manager_instance = DatabaseManager(settings.db_path)
        _db_manager_instance.init_schema()
        _LOG.info("Database manager initialized: %s", settings.db_path)
    return _db_manager_instance


def get_async_db() -> "AsyncDatabaseManager":
    """
    Get async database manager singleton.
    
    Wraps synchronous manager for async/await compatibility.
    """
    global _async_db_manager_instance
    if _async_db_manager_instance is None:
        from api.database import AsyncDatabaseManager
        _async_db_manager_instance = AsyncDatabaseManager(get_db_manager())
        _LOG.info("Async database manager initialized")
    return _async_db_manager_instance


# Repository factory functions
def get_session_repo() -> "SessionRepository":
    """Get session repository instance."""
    from api.database import SessionRepository
    return SessionRepository(get_async_db())


def get_history_repo() -> "HistoryRepository":
    """Get history repository instance."""
    from api.database import HistoryRepository
    return HistoryRepository(get_async_db())


def get_response_repo() -> "ResponseRepository":
    """Get response repository instance."""
    from api.database import ResponseRepository
    return ResponseRepository(get_async_db())


def get_document_repo() -> "DocumentRepository":
    """Get document repository instance."""
    from api.database import DocumentRepository
    return DocumentRepository(get_async_db())


# =============================================================================
# AI Service Dependencies (Lazy Loaded)
# =============================================================================

_chat_history_instance: Optional[Any] = None
_vector_base_instance: Optional[Any] = None
_llm_client_instance: Optional[Any] = None
_embedding_adapter_instance: Optional[Any] = None


def get_chat_history() -> Any:
    """
    Get GeneralizedChatHistory singleton.
    
    Lazy Loading:
    -------------
    Heavy AI services are loaded on first use to:
    - Reduce cold start time.
    - Allow health checks before full initialization.
    - Avoid loading if endpoints not used.
    """
    global _chat_history_instance
    if _chat_history_instance is None:
        try:
            from history import GeneralizedChatHistory, EMBED_DIM_DEFAULT
            settings = get_settings()
            _chat_history_instance = GeneralizedChatHistory(
                db_folder=str(settings.data_dir),
                d=settings.embed_dim,
            )
            _LOG.info("Chat history engine initialized")
        except ImportError as e:
            _LOG.warning("Chat history module not available: %s", e)
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Chat history service not available",
            )
    return _chat_history_instance


def get_embedding_adapter() -> Any:
    """
    Get embedding adapter singleton.
    
    Loads HuggingFace embedding model on first use.
    """
    global _embedding_adapter_instance
    if _embedding_adapter_instance is None:
        try:
            import torch
            from langchain_huggingface import HuggingFaceEmbeddings
            from torchvectorbase import EmbeddingAdapter
            
            settings = get_settings()
            # Force CPU to avoid "No HIP GPUs are available" in docker container without passthrough
            device = "cpu" 
            
            hf_embeddings = HuggingFaceEmbeddings(
                model_name=settings.embed_model,
                model_kwargs={"device": device, "trust_remote_code": True},
            )
            
            _embedding_adapter_instance = EmbeddingAdapter(
                hf_embeddings,
                device=device,
                normalize=settings.embed_normalize,
                batch_size=settings.embed_batch_size,
            )
            _LOG.info("Embedding adapter initialized: model=%s, device=%s", settings.embed_model, device)
        except ImportError as e:
            _LOG.warning("Embedding adapter not available: %s", e)
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Embedding service not available",
            )
    return _embedding_adapter_instance


def get_vector_base() -> Any:
    """
    Get VectorBase singleton with loaded document chunks.
    
    Initializes vector index for semantic search and loads
    any existing chunks from the processed directory.
    """
    global _vector_base_instance
    if _vector_base_instance is None:
        try:
            import torch
            import json
            from pathlib import Path
            from torchvectorbase import VectorBase
            
            settings = get_settings()
            emb = get_embedding_adapter()
            # Force CPU for vector operations as well
            device = "cpu"
            
            _vector_base_instance = VectorBase(
                emb,
                dim=settings.embed_dim,
                metric="cosine",
                device=device,
            )
            _LOG.info("Vector base initialized: dim=%d, device=%s", settings.embed_dim, device)
            
            # Persistence Path
            store_path = settings.data_dir / "vector_store.pt"
            
            # Try loading existing index first
            loaded = False
            if store_path.exists():
                try:
                    _vector_base_instance.load(str(store_path))
                    loaded = True
                except Exception as e:
                    _LOG.warning("Failed to load vector store from disk: %s", e)
            
            if not loaded:
                # -------------------------------------------------------------------------
                # Load existing chunks from JSONL files into vector base
                # -------------------------------------------------------------------------
                processed_dir = settings.data_dir / "uploads" / "processed"
                if processed_dir.exists():
                    jsonl_files = list(processed_dir.glob("*_chunks.jsonl"))
                    if jsonl_files:
                        _LOG.info("Loading %d chunk files into vector base...", len(jsonl_files))
                        
                        all_records = []
                        for jsonl_file in jsonl_files:
                            try:
                                with open(jsonl_file, "r", encoding="utf-8") as f:
                                    for line in f:
                                        line = line.strip()
                                        if line:
                                            record = json.loads(line)
                                            # Ensure required fields exist
                                            if "text" in record and record["text"]:
                                                all_records.append({
                                                    "doc_id": record.get("doc_id", str(jsonl_file.stem)),
                                                    "index": record.get("index", 0),
                                                    "text": record["text"],
                                                    "start": record.get("start", 0),
                                                    "end": record.get("end", 0),
                                                    "hash64": record.get("hash64", 0),
                                                })
                            except Exception as e:
                                _LOG.warning("Failed to load chunks from %s: %s", jsonl_file, e)
                        
                        if all_records:
                            _LOG.info("Loaded %d chunks, creating collection and indexing...", len(all_records))
                            
                            # Create collection and insert records
                            try:
                                _vector_base_instance.create_collection(
                                    settings.default_collection,
                                    dim=settings.embed_dim,
                                    metric="cosine",
                                )
                                
                                # Insert records (this embeds and indexes them)
                                inserted_ids, stats = _vector_base_instance.insert(
                                    all_records,
                                    return_stats=True,
                                )
                                _LOG.info(
                                    "Vector base loaded: total=%d, embedded=%d, inserted=%d",
                                    stats.get("total", 0),
                                    stats.get("embedded", 0),
                                    stats.get("inserted", 0),
                                )
                                
                                # Build IVF index for fast search if enough records
                                if len(inserted_ids) >= 10:
                                    try:
                                        from torchvectorbase import IVFBuildParams
                                        nlist = min(max(1, len(inserted_ids) // 4), 4096)
                                        _vector_base_instance.build_index(
                                            kind="IVF_PQ",
                                            params=IVFBuildParams(nlist=nlist, pq_m=16),
                                        )
                                        _LOG.info("Vector index built: nlist=%d", nlist)
                                    except Exception as idx_err:
                                        _LOG.warning("Failed to build IVF index: %s", idx_err)
                                
                                # Save to disk for future fast startup
                                try:
                                    _vector_base_instance.save(str(store_path))
                                except Exception as save_err:
                                    _LOG.error("Failed to save vector store: %s", save_err)
                                        
                            except Exception as insert_err:
                                _LOG.error("Failed to insert records: %s", insert_err)
                        else:
                            _LOG.info("No valid chunks found in JSONL files")
                    else:
                        _LOG.info("No chunk files found in processed directory")
                else:
                    _LOG.debug("Processed directory does not exist: %s", processed_dir)
                
        except ImportError as e:
            _LOG.warning("Vector base not available: %s", e)
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Vector search service not available",
            )
    return _vector_base_instance


def get_llm_client() -> Any:
    """
    Get LLMClient singleton.
    
    High-performance async LLM client with:
    - Connection pooling.
    - Retry with exponential backoff.
    - Bounded concurrency.
    """
    global _llm_client_instance
    if _llm_client_instance is None:
        try:
            from vllm_generation import LLMClient, LLMConfig
            
            settings = get_settings()
            config = LLMConfig(
                api_key=settings.llm_api_key,
                base_url=settings.llm_base_url,
                model=settings.llm_model,
                temperature=settings.llm_temperature,
                max_tokens=settings.llm_max_tokens,
                request_timeout_s=settings.llm_timeout_s,
                max_retries=settings.llm_max_retries,
                max_concurrency=settings.llm_max_concurrency,
                backoff_initial_s=settings.llm_backoff_initial_s,
                backoff_max_s=settings.llm_backoff_max_s,
                jitter_s=settings.llm_jitter_s,
            )
            
            _llm_client_instance = LLMClient(config)
            _LOG.info("LLM client initialized: model=%s, base_url=%s", settings.llm_model, settings.llm_base_url)
        except ImportError as e:
            _LOG.warning("LLM client not available: %s", e)
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="LLM service not available",
            )
    return _llm_client_instance


# =============================================================================
# Authentication Dependencies
# =============================================================================

async def verify_api_key(
    request: Request,
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
) -> Optional[str]:
    """
    Verify API key from request header.
    
    Authentication Flow:
    --------------------
    1. If auth disabled in settings, allow all requests.
    2. If no API key provided, return 401.
    3. If API key invalid, return 401.
    4. If valid, return the API key for logging/tracking.
    
    Security Notes:
    ---------------
    - Uses constant-time comparison to prevent timing attacks.
    - API keys should be stored hashed in production.
    """
    settings = get_settings()
    
    if not settings.auth_enabled:
        return None
    
    if not x_api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key required",
            headers={"WWW-Authenticate": "ApiKey"},
        )
    
    # Constant-time comparison
    import hmac
    valid = False
    for valid_key in settings.auth_api_keys:
        if hmac.compare_digest(x_api_key, valid_key):
            valid = True
            break
    
    if not valid:
        _LOG.warning("Invalid API key attempt: %s...", x_api_key[:8])
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "ApiKey"},
        )
    
    return x_api_key


def optional_api_key(
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
) -> Optional[str]:
    """
    Optional API key extraction (no validation).
    
    Use for endpoints that support both authenticated and anonymous access.
    """
    return x_api_key


# =============================================================================
# Request Context Dependencies
# =============================================================================

def get_request_id(request: Request) -> str:
    """
    Get request ID from request state.
    
    Requires RequestIDMiddleware to be active.
    """
    return getattr(request.state, "request_id", "unknown")


def get_client_ip(request: Request) -> str:
    """
    Get client IP address from request.
    
    Handles proxy headers (X-Forwarded-For, X-Real-IP).
    """
    # X-Forwarded-For can contain multiple IPs
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    
    real_ip = request.headers.get("X-Real-IP")
    if real_ip:
        return real_ip.strip()
    
    if request.client:
        return request.client.host
    
    return "unknown"


# =============================================================================
# Background Task Dependencies
# =============================================================================

_background_tasks_lock = asyncio.Lock()
_background_tasks: set = set()


async def schedule_background_task(coro) -> None:
    """
    Schedule a background task with tracking.
    
    Ensures tasks are properly awaited on shutdown.
    """
    task = asyncio.create_task(coro)
    
    async with _background_tasks_lock:
        _background_tasks.add(task)
    
    def on_complete(t):
        asyncio.create_task(remove_task(t))
    
    task.add_done_callback(on_complete)


async def remove_task(task) -> None:
    """Remove completed task from tracking set."""
    async with _background_tasks_lock:
        _background_tasks.discard(task)


async def wait_background_tasks(timeout: float = 30.0) -> None:
    """
    Wait for all background tasks to complete.
    
    Called during graceful shutdown.
    """
    async with _background_tasks_lock:
        tasks = list(_background_tasks)
    
    if tasks:
        _LOG.info("Waiting for %d background tasks...", len(tasks))
        done, pending = await asyncio.wait(tasks, timeout=timeout)
        
        if pending:
            _LOG.warning("Cancelling %d pending background tasks", len(pending))
            for task in pending:
                task.cancel()


# =============================================================================
# Cleanup Functions
# =============================================================================

async def cleanup_services() -> None:
    """
    Clean up all service instances on shutdown.
    
    Called during FastAPI lifespan shutdown.
    """
    global _chat_history_instance, _vector_base_instance, _llm_client_instance
    global _embedding_adapter_instance, _db_manager_instance, _async_db_manager_instance
    
    _LOG.info("Cleaning up services...")
    
    # Wait for background tasks
    await wait_background_tasks()
    
    # Close chat history
    if _chat_history_instance is not None:
        try:
            _chat_history_instance.close()
        except Exception as e:
            _LOG.warning("Error closing chat history: %s", e)
        _chat_history_instance = None
    
    # Close database manager
    if _db_manager_instance is not None:
        try:
            _db_manager_instance.close()
        except Exception as e:
            _LOG.warning("Error closing database: %s", e)
        _db_manager_instance = None
    
    _async_db_manager_instance = None
    _vector_base_instance = None
    _llm_client_instance = None
    _embedding_adapter_instance = None
    
    _LOG.info("Service cleanup complete")


# =============================================================================
# Health Check Dependencies
# =============================================================================

async def check_database_health() -> bool:
    """Check if database is accessible."""
    try:
        db = get_async_db()
        await db.execute("SELECT 1", fetch=True)
        return True
    except Exception as e:
        _LOG.warning("Database health check failed: %s", e)
        return False


async def check_llm_health() -> bool:
    """Check if LLM service is accessible."""
    try:
        # Just check if client can be instantiated
        # Actual LLM call would be too slow for health check
        get_llm_client()
        return True
    except Exception as e:
        _LOG.warning("LLM health check failed: %s", e)
        return False


async def get_service_health() -> dict:
    """
    Get health status of all services.
    
    Returns:
    --------
    Dict with service names and their status ("healthy"/"unhealthy").
    """
    results = {}
    
    # Database check
    results["database"] = "healthy" if await check_database_health() else "unhealthy"
    
    # LLM check (just instantiation, not actual call)
    try:
        get_llm_client()
        results["llm"] = "healthy"
    except Exception:
        results["llm"] = "unhealthy"
    
    # Vector base check
    try:
        get_vector_base()
        results["vector_search"] = "healthy"
    except Exception:
        results["vector_search"] = "unavailable"
    
    # Chat history check
    try:
        get_chat_history()
        results["chat_history"] = "healthy"
    except Exception:
        results["chat_history"] = "unavailable"
    
    return results
