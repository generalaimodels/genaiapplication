# -*- coding: utf-8 -*-
# =================================================================================================
# api/routers/search.py — Vector Search & Retrieval Endpoints
# =================================================================================================
# Production-grade semantic search implementing:
#
#   1. VECTOR SEARCH: Semantic similarity search via embeddings.
#   2. RERANKING: Rethinker-based structure-aware reranking.
#   3. SIMILAR DOCS: Find documents similar to a given document.
#   4. INDEX MANAGEMENT: Build/rebuild vector index.
#
# =================================================================================================

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status

from api.schemas import (
    SearchRequest,
    SearchResponse,
    SearchResult,
)
from api.exceptions import ServiceError, ServiceUnavailableError
from api.dependencies import (
    get_vector_base,
    get_embedding_adapter,
    get_settings,
)
from api.database import generate_uuid

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
_LOG = logging.getLogger("api.routers.search")

# -----------------------------------------------------------------------------
# Router Configuration
# -----------------------------------------------------------------------------
router = APIRouter(
    prefix="/search",
    tags=["Search"],
    responses={
        503: {"description": "Search service unavailable"},
    },
)


# =============================================================================
# Semantic Search Endpoint
# =============================================================================

@router.post(
    "",
    response_model=SearchResponse,
    summary="Semantic Search",
    description="""
    Perform semantic vector search over indexed documents.
    
    **Features**:
    - Embedding-based similarity search
    - Configurable top-k results
    - Optional metadata filtering
    - Optional Rethinker reranking
    
    **Request Example**:
    ```json
    {
        "query": "What are the compliance requirements?",
        "top_k": 10,
        "collection": "docs",
        "rerank": true
    }
    ```
    """,
)
async def semantic_search(
    request: SearchRequest,
) -> SearchResponse:
    """Perform semantic vector search."""
    start_time = time.perf_counter()
    settings = get_settings()
    
    try:
        vector_base = get_vector_base()
        
        # Perform search
        collection = request.collection or settings.default_collection
        
        # Check if collection exists
        if not hasattr(vector_base, 'collection') or vector_base.collection is None:
            # Create collection if it doesn't exist
            try:
                vector_base.create_collection(collection, settings.embed_dim, "cosine")
            except Exception:
                pass  # Collection may already exist
        
        # Search
        scores, ids = vector_base.search(
            request.query,
            request.top_k,
        )
        
        # Convert results
        results: List[SearchResult] = []
        
        if hasattr(vector_base, 'collection') and vector_base.collection is not None:
            for i, (score, doc_id) in enumerate(zip(scores.tolist() if hasattr(scores, 'tolist') else scores, 
                                                     ids.tolist() if hasattr(ids, 'tolist') else ids)):
                # Get text from collection
                text = ""
                metadata = {}
                
                try:
                    if hasattr(vector_base.collection, 'texts') and doc_id < len(vector_base.collection.texts):
                        text = vector_base.collection.texts[doc_id]
                    if hasattr(vector_base.collection, 'metas') and doc_id < len(vector_base.collection.metas):
                        metadata = vector_base.collection.metas[doc_id]
                except Exception:
                    pass
                
                results.append(SearchResult(
                    id=str(doc_id),
                    text=text[:2000] if request.include_text else "",
                    score=float(score) if isinstance(score, (int, float)) else 0.0,
                    chunk_index=i,
                    metadata=metadata,
                ))
        
        latency_ms = (time.perf_counter() - start_time) * 1000
        
        return SearchResponse(
            query=request.query,
            results=results,
            total=len(results),
            latency_ms=latency_ms,
        )
        
    except Exception as e:
        _LOG.error("Search error: %s", e, exc_info=True)
        raise ServiceUnavailableError(f"Search failed: {str(e)}")


# =============================================================================
# Rerank Endpoint
# =============================================================================

@router.post(
    "/rerank",
    response_model=SearchResponse,
    summary="Rerank Results",
    description="""
    Apply Rethinker reranking to search results.
    
    **Features**:
    - Structure-aware graph-augmented retrieval
    - Semantic + lexical scoring
    - Context window assembly
    """,
)
async def rerank_search(
    request: SearchRequest,
) -> SearchResponse:
    """Perform search with Rethinker reranking."""
    start_time = time.perf_counter()
    settings = get_settings()
    
    try:
        vector_base = get_vector_base()
        
        # Import Rethinker
        from rethinker_retrieval import Rethinker, RethinkerParams
        
        # Create Rethinker
        params = RethinkerParams(
            top_nodes_final=request.top_k,
        )
        rethinker = Rethinker(vector_base, params)
        
        # Perform search
        result = rethinker.search(request.query)
        contexts = result.get("contexts", [])
        
        # Convert results
        results: List[SearchResult] = []
        for i, ctx in enumerate(contexts):
            results.append(SearchResult(
                id=str(i),
                text=ctx.get("text", "")[:2000] if request.include_text else "",
                score=ctx.get("score", 0.0),
                doc_id=ctx.get("doc_id"),
                chunk_index=ctx.get("chunk_index"),
                metadata=ctx.get("metadata", {}),
            ))
        
        latency_ms = (time.perf_counter() - start_time) * 1000
        
        return SearchResponse(
            query=request.query,
            results=results,
            total=len(results),
            latency_ms=latency_ms,
        )
        
    except ImportError:
        _LOG.warning("Rethinker not available, falling back to basic search")
        return await semantic_search(request)
    except Exception as e:
        _LOG.error("Rerank error: %s", e, exc_info=True)
        raise ServiceUnavailableError(f"Reranking failed: {str(e)}")


# =============================================================================
# Index Stats Endpoint
# =============================================================================

@router.get(
    "/stats",
    summary="Index Statistics",
    description="Get vector index statistics.",
)
async def index_stats() -> Dict[str, Any]:
    """Get vector index statistics."""
    try:
        vector_base = get_vector_base()
        
        stats = {
            "status": "available",
            "collection": None,
            "document_count": 0,
            "embedding_dim": 0,
            "metric": "cosine",
            "index_built": False,
        }
        
        if hasattr(vector_base, 'collection') and vector_base.collection is not None:
            stats["collection"] = getattr(vector_base.collection, 'name', 'default')
            stats["document_count"] = getattr(vector_base.collection, 'size', 0)
        
        if hasattr(vector_base, 'dim'):
            stats["embedding_dim"] = vector_base.dim
        
        if hasattr(vector_base, 'index') and vector_base.index is not None:
            stats["index_built"] = True
        
        return stats
        
    except Exception as e:
        return {
            "status": "unavailable",
            "error": str(e),
        }


# =============================================================================
# Embed Text Endpoint (Utility)
# =============================================================================

@router.post(
    "/embed",
    summary="Embed Text",
    description="Get embedding vector for text (utility endpoint).",
)
async def embed_text(
    text: str = Query(..., min_length=1, max_length=8192),
) -> Dict[str, Any]:
    """Get embedding for text."""
    try:
        embedder = get_embedding_adapter()
        
        import torch
        embedding = embedder.embed_query(text)
        
        # Convert to list
        if isinstance(embedding, torch.Tensor):
            embedding = embedding.cpu().numpy().tolist()
            if isinstance(embedding[0], list):
                embedding = embedding[0]
        
        return {
            "text": text[:100] + "..." if len(text) > 100 else text,
            "embedding": embedding,
            "dimension": len(embedding),
        }
        
    except Exception as e:
        _LOG.error("Embedding error: %s", e)
        raise ServiceUnavailableError(f"Embedding failed: {str(e)}")
