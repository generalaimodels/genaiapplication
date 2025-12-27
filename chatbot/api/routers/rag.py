# -*- coding: utf-8 -*-
# =================================================================================================
# api/routers/rag.py — RAG (Retrieval-Augmented Generation) Query Endpoints
# =================================================================================================
# End-to-end RAG pipeline implementing:
#
#   1. QUERY: Combined retrieve + generate in a single POST request.
#   2. STREAMING QUERY: Same as above but with SSE streaming response.
#   3. HISTORY INTEGRATION: Optionally store query/answer in session history.
#
# This router provides the "post with query" functionality that automatically:
#   - Retrieves relevant context from the vector store (via Rethinker or VectorBase)
#   - Builds a prompt with the retrieved context
#   - Generates an answer using the LLM
#   - Optionally stores the interaction in conversation history
#
# =================================================================================================

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any, AsyncGenerator, Dict, List, Optional

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from api.schemas import (
    ChatResponse,
    ContextChunk,
    SearchResult,
)
from api.exceptions import ServiceError, ServiceUnavailableError, TimeoutError
from api.dependencies import (
    get_session_repo,
    get_history_repo,
    get_response_repo,
    get_llm_client,
    get_vector_base,
    get_settings,
)
from api.database import SessionRepository, HistoryRepository, ResponseRepository, generate_uuid

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
_LOG = logging.getLogger("api.routers.rag")

# -----------------------------------------------------------------------------
# Router Configuration
# -----------------------------------------------------------------------------
router = APIRouter(
    prefix="/rag",
    tags=["RAG"],
    responses={
        503: {"description": "Service unavailable"},
        504: {"description": "Request timeout"},
    },
)


# =============================================================================
# Request/Response Schemas
# =============================================================================

class RAGQueryRequest(BaseModel):
    """
    Request model for RAG query (retrieve + generate).
    
    This is the main "post with query" endpoint that:
    1. Retrieves relevant documents
    2. Generates an answer using the LLM
    3. Optionally stores in history
    """
    query: str = Field(
        min_length=1,
        max_length=32768,
        description="The question or query to answer.",
    )
    session_id: Optional[str] = Field(
        default=None,
        description="Session ID for history storage. Creates new session if not provided.",
    )
    collection: Optional[str] = Field(
        default=None,
        max_length=128,
        description="Collection to search (uses default if not provided).",
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=50,
        description="Number of context chunks to retrieve.",
    )
    temperature: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=2.0,
        description="LLM sampling temperature.",
    )
    max_tokens: Optional[int] = Field(
        default=None,
        ge=1,
        le=16384,
        description="Maximum tokens to generate.",
    )
    system_prompt: Optional[str] = Field(
        default=None,
        max_length=8192,
        description="Custom system prompt. Uses RAG default if not provided.",
    )
    model: Optional[str] = Field(
        default=None,
        max_length=256,
        description="Model override.",
    )
    use_reranker: bool = Field(
        default=True,
        description="Whether to use Rethinker reranking for better context.",
    )
    store_history: bool = Field(
        default=True,
        description="Whether to store the query and answer in session history.",
    )

    model_config = {"str_strip_whitespace": True}


class RAGQueryResponse(BaseModel):
    """Response model for RAG query."""
    id: str = Field(description="Response ID (UUID).")
    session_id: Optional[str] = Field(default=None, description="Session ID if stored.")
    query: str = Field(description="Original query.")
    answer: str = Field(description="Generated answer.")
    context: List[ContextChunk] = Field(
        default_factory=list,
        description="Retrieved context chunks used for the answer.",
    )
    model: str = Field(description="Model used for generation.")
    tokens_prompt: Optional[int] = Field(default=None, description="Prompt tokens used.")
    tokens_completion: Optional[int] = Field(default=None, description="Completion tokens.")
    latency_ms: float = Field(description="Total response latency.")
    retrieval_latency_ms: float = Field(description="Time spent on retrieval.")
    generation_latency_ms: float = Field(description="Time spent on generation.")
    metadata: Dict[str, Any] = Field(default_factory=dict)


# =============================================================================
# SOTA Retrieval Engine — Hybrid Search with Reciprocal Rank Fusion
# =============================================================================
# Implements state-of-the-art retrieval mechanisms:
#   1. Hybrid Search: Combines semantic (dense) and lexical (BM25) retrieval
#   2. Reciprocal Rank Fusion (RRF): Merges multiple ranked lists robustly
#   3. Rethinker Integration: Graph-augmented retrieval with beam search
#   4. Adaptive Fallback: Graceful degradation when components unavailable
# =============================================================================

from collections import defaultdict
from dataclasses import dataclass
import re
import math


@dataclass
class RetrievalConfig:
    """
    Configuration for SOTA retrieval pipeline.
    
    Attributes:
        semantic_weight: Weight for semantic search in hybrid fusion (0-1).
        lexical_weight: Weight for lexical BM25 search in hybrid fusion (0-1).
        rrf_k: RRF constant k (higher = more uniform weighting, default 60).
        semantic_oversample: Oversample factor for semantic search before fusion.
        lexical_oversample: Oversample factor for lexical search before fusion.
        use_query_expansion: Enable query expansion for better recall.
        max_context_chars: Maximum characters per context chunk.
    """
    semantic_weight: float = 0.65
    lexical_weight: float = 0.35
    rrf_k: int = 60
    semantic_oversample: int = 3
    lexical_oversample: int = 3
    use_query_expansion: bool = False
    max_context_chars: int = 2000


def _tokenize_for_bm25(text: str) -> List[str]:
    """
    Tokenize text for BM25 scoring.
    
    Simple alphanumeric tokenizer with lowercase normalization.
    Production systems may use more sophisticated tokenizers.
    
    Args:
        text: Input text to tokenize.
    
    Returns:
        List of lowercase tokens.
    """
    return re.findall(r"[A-Za-z0-9_]+", str(text).lower())


def _compute_bm25_score(
    query_tokens: List[str],
    doc_tokens: List[str],
    avg_doc_len: float,
    doc_freq: Dict[str, int],
    total_docs: int,
    k1: float = 0.9,
    b: float = 0.4,
) -> float:
    """
    Compute BM25 score for a single document.
    
    BM25 Formula:
        score = Σ IDF(qi) * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * |D|/avgdl))
    
    Where:
        IDF(qi) = log((N - df(qi) + 0.5) / (df(qi) + 0.5) + 1)
        tf = term frequency in document
        |D| = document length
        avgdl = average document length
    
    Args:
        query_tokens: Tokenized query.
        doc_tokens: Tokenized document.
        avg_doc_len: Average document length in corpus.
        doc_freq: Document frequency for each term.
        total_docs: Total number of documents.
        k1: Term frequency saturation parameter.
        b: Length normalization parameter.
    
    Returns:
        BM25 score (higher is better).
    """
    if not query_tokens or not doc_tokens:
        return 0.0
    
    doc_len = len(doc_tokens)
    tf_map: Dict[str, int] = defaultdict(int)
    for token in doc_tokens:
        tf_map[token] += 1
    
    score = 0.0
    query_terms = set(query_tokens)
    
    for term in query_terms:
        tf = tf_map.get(term, 0)
        if tf == 0:
            continue
        
        df = doc_freq.get(term, 0)
        # IDF with BM25 log-smoothing
        idf = math.log((total_docs - df + 0.5) / (df + 0.5) + 1.0)
        
        # BM25 term score
        numerator = tf * (k1 + 1.0)
        denominator = tf + k1 * (1.0 - b + b * (doc_len / max(1.0, avg_doc_len)))
        score += idf * numerator / max(1e-9, denominator)
    
    return score


def _reciprocal_rank_fusion(
    rankings: List[List[int]],
    scores_per_ranking: Optional[List[List[float]]] = None,
    k: int = 60,
) -> List[tuple[int, float]]:
    """
    Merge multiple ranked lists using Reciprocal Rank Fusion (RRF).
    
    RRF is a robust rank aggregation method that:
    - Handles different score scales naturally
    - Is resistant to outliers
    - Proven effective for hybrid search
    
    Formula:
        RRF_score(d) = Σ 1 / (k + rank_i(d))
    
    Where rank_i(d) is the rank of document d in ranking list i (1-indexed).
    
    Args:
        rankings: List of ranked document ID lists (each in decreasing relevance).
        scores_per_ranking: Optional original scores for tie-breaking.
        k: RRF constant (default 60, standard in literature).
    
    Returns:
        Merged list of (doc_id, rrf_score) sorted by score descending.
    
    References:
        Cormack et al. "Reciprocal Rank Fusion outperforms Condorcet and 
        individual Rank Learning Methods" (SIGIR 2009)
    """
    rrf_scores: Dict[int, float] = defaultdict(float)
    original_scores: Dict[int, float] = {}
    
    for list_idx, ranking in enumerate(rankings):
        for rank, doc_id in enumerate(ranking, start=1):
            rrf_scores[doc_id] += 1.0 / (k + rank)
            
            # Store best original score for tie-breaking
            if scores_per_ranking and list_idx < len(scores_per_ranking):
                if rank - 1 < len(scores_per_ranking[list_idx]):
                    orig_score = scores_per_ranking[list_idx][rank - 1]
                    if doc_id not in original_scores or orig_score > original_scores[doc_id]:
                        original_scores[doc_id] = orig_score
    
    # Sort by RRF score (descending), then by original score for ties
    merged = sorted(
        rrf_scores.items(),
        key=lambda x: (x[1], original_scores.get(x[0], 0.0)),
        reverse=True,
    )
    
    return merged


def _expand_query(query: str) -> str:
    """
    Expand query for better recall (optional enhancement).
    
    Simple query expansion via synonym-like additions.
    Production systems would use more sophisticated methods:
    - Word embeddings for semantic expansion
    - LLM-based query reformulation
    - Pseudo-relevance feedback
    
    Args:
        query: Original query string.
    
    Returns:
        Expanded query string.
    """
    # For now, return query unchanged
    # Future: Integrate LLM-based expansion
    return query


def _distance_to_similarity(dist: float, metric: str) -> float:
    """
    Convert distance (smaller is better) to similarity (larger is better).
    
    Args:
        dist: Distance value from vector search.
        metric: Distance metric used ("cosine", "l2", "ip").
    
    Returns:
        Similarity score in [0, 1] range.
    """
    m = metric.lower() if metric else "cosine"
    
    if m == "cosine":
        # cosine distance = 1 - cosine_similarity
        # cosine_similarity in [-1, 1], map to [0, 1]
        cos_sim = 1.0 - dist
        return max(0.0, min(1.0, 0.5 * (cos_sim + 1.0)))
    
    if m == "l2":
        # L2 distance >= 0, use inverse mapping
        return 1.0 / (1.0 + max(0.0, dist))
    
    if m == "ip":
        # Inner product returned as negative distance
        # Use logistic squash for bounded output
        sim = -dist
        return 1.0 / (1.0 + math.exp(-sim))
    
    return 0.0


def _get_rag_system_prompt() -> str:
    """
    Get default RAG system prompt.
    
    Designed for grounded, context-aware responses.
    """
    return """You are a helpful assistant. Answer questions based on the provided context. If the context is insufficient, say so."""


async def _retrieve_context(
    query: str,
    top_k: int,
    collection: Optional[str],
    use_reranker: bool,
) -> tuple[List[ContextChunk], float]:
    """
    SOTA Context Retrieval Pipeline.
    
    Implements a multi-stage retrieval pipeline:
    
    Stage 1: Rethinker Graph-Augmented Retrieval (if enabled and available)
        - DFS traversal over document graph
        - Combines semantic and structural signals
        - Best quality, higher latency
    
    Stage 2: Hybrid Search Fallback
        - Semantic search via VectorBase
        - Lexical BM25 search over texts
        - Reciprocal Rank Fusion for merging
    
    Stage 3: Pure Semantic Fallback
        - Basic vector search
        - Used when BM25 index unavailable
    
    Args:
        query: User query string.
        top_k: Number of context chunks to retrieve.
        collection: Optional collection name (uses default if None).
        use_reranker: Whether to use Rethinker for graph-augmented retrieval.
    
    Returns:
        Tuple of (context_chunks, latency_ms).
    
    Algorithm Complexity:
        - Rethinker: O(beam_width * max_depth * semantic_k) 
        - Hybrid: O(top_k * oversample * (semantic_search + bm25_score))
        - RRF: O(n log n) where n = total candidates
    """
    start_time = time.perf_counter()
    config = RetrievalConfig()
    context_chunks: List[ContextChunk] = []
    
    # =========================================================================
    # Cache Lookup: Check for cached retrieval results
    # =========================================================================
    try:
        from api.cache import get_retrieval_cache, hash_query
        
        cache_key = hash_query(
            query, 
            top_k=top_k, 
            collection=collection or "default",
            use_reranker=use_reranker,
        )
        retrieval_cache = get_retrieval_cache()
        
        cached_result = await retrieval_cache.get(cache_key)
        if cached_result is not None:
            cached_chunks, cached_latency = cached_result
            latency_ms = (time.perf_counter() - start_time) * 1000
            _LOG.debug(
                "Retrieval cache HIT: key=%s, chunks=%d, latency=%.2fms", 
                cache_key[:16], len(cached_chunks), latency_ms
            )
            # Annotate with cache hit metadata
            for chunk in cached_chunks:
                if hasattr(chunk, 'metadata') and chunk.metadata:
                    chunk.metadata['cache_hit'] = True
            return cached_chunks, latency_ms
            
    except ImportError:
        _LOG.debug("Cache module not available, proceeding without cache")
        cache_key = None  # Disable caching if module unavailable
    except Exception as cache_err:
        _LOG.warning("Cache lookup failed: %s", cache_err)
        cache_key = None
    
    try:
        vector_base = get_vector_base()
        
        # =====================================================================
        # Stage 1: Rethinker Graph-Augmented Retrieval
        # =====================================================================
        if use_reranker:
            try:
                from rethinker_retrieval import Rethinker, RethinkerParams
                
                if hasattr(vector_base, 'collection') and vector_base.collection is not None:
                    coll = vector_base.collection
                    
                    # Verify collection has data
                    if hasattr(coll, 'size') and coll.size > 0:
                        # Configure Rethinker for optimal retrieval
                        params = RethinkerParams(
                            top_nodes_final=top_k,
                            seed_sem_topk=min(64, top_k * 6),
                            seed_lex_topk=min(64, top_k * 6),
                            max_depth=3,
                            beam_per_depth=8,
                            semantic_k_per_node=8,
                            w_sem_query=0.60,
                            w_lex=0.30,
                            w_adjacent=0.10,
                            decay_per_depth=0.85,
                            draw_above=2,
                            draw_below=2,
                            max_chars_per_context=config.max_context_chars,
                        )
                        
                        rethinker = Rethinker(vector_base, params)
                        result = rethinker.search(query)
                        contexts = result.get("contexts", [])
                        
                        if contexts:
                            for i, ctx in enumerate(contexts[:top_k]):
                                context_chunks.append(ContextChunk(
                                    text=str(ctx.get("text", ""))[:config.max_context_chars],
                                    score=float(ctx.get("score", 0.0)),
                                    doc_id=str(ctx.get("doc_id", "")),
                                    chunk_index=int(ctx.get("start_index", i)),
                                    metadata={
                                        "start_char": ctx.get("start_char", 0),
                                        "end_char": ctx.get("end_char", 0),
                                        "node_ids": ctx.get("node_ids", []),
                                        "retrieval_method": "rethinker",
                                    },
                                ))
                            
                            latency_ms = (time.perf_counter() - start_time) * 1000
                            _LOG.debug("Rethinker retrieved %d contexts in %.2fms", 
                                      len(context_chunks), latency_ms)
                            return context_chunks, latency_ms
                        
                        _LOG.debug("Rethinker returned empty contexts, falling back")
                    else:
                        _LOG.debug("Collection empty, skipping Rethinker")
                        
            except ImportError:
                _LOG.debug("Rethinker not available, falling back to hybrid search")
            except Exception as e:
                _LOG.warning("Rethinker search failed: %s, falling back to hybrid search", e)
        
        # =====================================================================
        # Stage 2: Hybrid Search with Reciprocal Rank Fusion
        # =====================================================================
        if hasattr(vector_base, 'collection') and vector_base.collection is not None:
            coll = vector_base.collection
            
            # Check if collection has data
            coll_size = getattr(coll, 'size', 0)
            if coll_size == 0:
                _LOG.debug("Collection is empty, no results to retrieve")
                latency_ms = (time.perf_counter() - start_time) * 1000
                return [], latency_ms
            
            # Get collection texts for BM25
            texts: List[str] = []
            records: List[Any] = []
            
            if hasattr(coll, 'texts'):
                texts = list(coll.texts) if coll.texts else []
            if hasattr(coll, 'records'):
                records = list(coll.records) if coll.records else []
            
            # -----------------------------------------------------------------
            # Semantic Search via VectorBase
            # -----------------------------------------------------------------
            semantic_k = min(top_k * config.semantic_oversample, coll_size)
            semantic_ranking: List[int] = []
            semantic_scores: List[float] = []
            
            try:
                # Use VectorBase.search() API properly
                if hasattr(vector_base, 'search'):
                    results, contexts_raw = vector_base.search(query, k=semantic_k)
                    
                    if results and len(results) > 0:
                        # results[0] contains list of (id, distance) tuples
                        for doc_id, distance in results[0]:
                            semantic_ranking.append(int(doc_id))
                            # Convert distance to similarity
                            metric = getattr(coll, 'metric', 'cosine')
                            sim = _distance_to_similarity(float(distance), metric)
                            semantic_scores.append(sim)
                
                _LOG.debug("Semantic search returned %d results", len(semantic_ranking))
                
            except Exception as e:
                _LOG.warning("Semantic search failed: %s", e)
            
            # -----------------------------------------------------------------
            # Lexical BM25 Search
            # -----------------------------------------------------------------
            lexical_ranking: List[int] = []
            lexical_scores: List[float] = []
            
            if texts:
                try:
                    # Tokenize all documents
                    doc_tokens_list = [_tokenize_for_bm25(t) for t in texts]
                    
                    # Compute document frequencies
                    doc_freq: Dict[str, int] = defaultdict(int)
                    for doc_tokens in doc_tokens_list:
                        seen = set(doc_tokens)
                        for token in seen:
                            doc_freq[token] += 1
                    
                    # Compute average document length
                    total_tokens = sum(len(dt) for dt in doc_tokens_list)
                    avg_doc_len = total_tokens / max(1, len(doc_tokens_list))
                    
                    # Tokenize query
                    query_tokens = _tokenize_for_bm25(query)
                    
                    # Score all documents
                    bm25_scores: List[tuple[int, float]] = []
                    for doc_id, doc_tokens in enumerate(doc_tokens_list):
                        score = _compute_bm25_score(
                            query_tokens=query_tokens,
                            doc_tokens=doc_tokens,
                            avg_doc_len=avg_doc_len,
                            doc_freq=doc_freq,
                            total_docs=len(texts),
                        )
                        if score > 0:
                            bm25_scores.append((doc_id, score))
                    
                    # Sort by BM25 score descending
                    bm25_scores.sort(key=lambda x: x[1], reverse=True)
                    
                    # Take top-k for lexical ranking
                    lexical_k = min(top_k * config.lexical_oversample, len(bm25_scores))
                    for doc_id, score in bm25_scores[:lexical_k]:
                        lexical_ranking.append(doc_id)
                        lexical_scores.append(score)
                    
                    _LOG.debug("BM25 search returned %d results", len(lexical_ranking))
                    
                except Exception as e:
                    _LOG.warning("BM25 search failed: %s", e)
            
            # -----------------------------------------------------------------
            # Reciprocal Rank Fusion
            # -----------------------------------------------------------------
            if semantic_ranking or lexical_ranking:
                rankings = []
                scores_per_ranking = []
                
                if semantic_ranking:
                    rankings.append(semantic_ranking)
                    scores_per_ranking.append(semantic_scores)
                
                if lexical_ranking:
                    rankings.append(lexical_ranking)
                    # Normalize lexical scores to [0, 1]
                    if lexical_scores:
                        max_lex = max(lexical_scores)
                        normalized_lex = [s / max_lex if max_lex > 0 else 0 for s in lexical_scores]
                        scores_per_ranking.append(normalized_lex)
                
                # Merge with RRF
                merged = _reciprocal_rank_fusion(
                    rankings=rankings,
                    scores_per_ranking=scores_per_ranking,
                    k=config.rrf_k,
                )
                
                _LOG.debug("RRF merged %d unique documents", len(merged))
                
                # Extract top-k results
                for doc_id, rrf_score in merged[:top_k]:
                    text = ""
                    metadata: Dict[str, Any] = {}
                    doc_id_str = ""
                    chunk_index = 0
                    
                    # Get text from Collection.texts
                    if texts and 0 <= doc_id < len(texts):
                        text = texts[doc_id][:config.max_context_chars]
                    
                    # Get metadata from Collection.records
                    if records and 0 <= doc_id < len(records):
                        rec = records[doc_id]
                        if hasattr(rec, 'doc_id'):
                            doc_id_str = str(rec.doc_id)
                        if hasattr(rec, 'index'):
                            chunk_index = int(rec.index)
                        if hasattr(rec, 'meta'):
                            metadata = dict(rec.meta) if rec.meta else {}
                        if hasattr(rec, 'start'):
                            metadata['start_char'] = rec.start
                        if hasattr(rec, 'end'):
                            metadata['end_char'] = rec.end
                    
                    metadata["retrieval_method"] = "hybrid_rrf"
                    metadata["rrf_score"] = rrf_score
                    
                    context_chunks.append(ContextChunk(
                        text=text,
                        score=rrf_score,
                        doc_id=doc_id_str,
                        chunk_index=chunk_index,
                        metadata=metadata,
                    ))
            
            # -----------------------------------------------------------------
            # Fallback: Pure Semantic Results (if RRF produced nothing)
            # -----------------------------------------------------------------
            if not context_chunks and semantic_ranking:
                _LOG.debug("RRF produced no results, using pure semantic ranking")
                
                for i, doc_id in enumerate(semantic_ranking[:top_k]):
                    text = ""
                    metadata: Dict[str, Any] = {}
                    doc_id_str = ""
                    chunk_index = 0
                    
                    if texts and 0 <= doc_id < len(texts):
                        text = texts[doc_id][:config.max_context_chars]
                    
                    if records and 0 <= doc_id < len(records):
                        rec = records[doc_id]
                        if hasattr(rec, 'doc_id'):
                            doc_id_str = str(rec.doc_id)
                        if hasattr(rec, 'index'):
                            chunk_index = int(rec.index)
                        if hasattr(rec, 'meta'):
                            metadata = dict(rec.meta) if rec.meta else {}
                    
                    metadata["retrieval_method"] = "semantic_fallback"
                    
                    score = semantic_scores[i] if i < len(semantic_scores) else 0.0
                    
                    context_chunks.append(ContextChunk(
                        text=text,
                        score=score,
                        doc_id=doc_id_str,
                        chunk_index=chunk_index,
                        metadata=metadata,
                    ))
        
    except Exception as e:
        _LOG.error("Retrieval pipeline failed: %s", e, exc_info=True)
    
    latency_ms = (time.perf_counter() - start_time) * 1000
    
    # =========================================================================
    # Cache Store: Save retrieval results for future queries
    # =========================================================================
    if context_chunks and cache_key:
        try:
            await retrieval_cache.set(cache_key, (context_chunks, latency_ms))
            _LOG.debug(
                "Retrieval cache STORE: key=%s, chunks=%d",
                cache_key[:16], len(context_chunks)
            )
        except Exception as store_err:
            _LOG.warning("Cache store failed: %s", store_err)
    
    _LOG.debug("Retrieved %d contexts in %.2fms", len(context_chunks), latency_ms)
    return context_chunks, latency_ms


def _build_context_prompt(query: str, context_chunks: List[ContextChunk]) -> str:
    """Build the user prompt with context."""
    if not context_chunks:
        return f"Question: {query}"
    
    context_text = "\n\n".join([
        f"[Context {i+1}] (score: {chunk.score:.3f})\n{chunk.text}"
        for i, chunk in enumerate(context_chunks)
    ])
    
    return f"""Context:
{context_text}

Question: {query}

Please answer the question based on the context provided above."""


# =============================================================================
# RAG Query Endpoint (Non-Streaming)
# =============================================================================

@router.post(
    "/query",
    response_model=RAGQueryResponse,
    summary="RAG Query",
    description="""
    End-to-end RAG query: retrieve context and generate answer.
    
    **Flow**:
    1. Search vector store for relevant documents
    2. Build prompt with retrieved context
    3. Generate answer using LLM
    4. Optionally store in session history
    
    **Request Example**:
    ```json
    {
        "query": "What are the compliance requirements for data storage?",
        "top_k": 5,
        "temperature": 0.7,
        "store_history": true
    }
    ```
    """,
)
async def rag_query(
    request: RAGQueryRequest,
    background_tasks: BackgroundTasks,
    sessions: SessionRepository = Depends(get_session_repo),
    history_repo: HistoryRepository = Depends(get_history_repo),
    response_repo: ResponseRepository = Depends(get_response_repo),
) -> RAGQueryResponse:
    """Execute end-to-end RAG query."""
    start_time = time.perf_counter()
    settings = get_settings()
    
    # Step 1: Retrieve context
    context_chunks, retrieval_latency_ms = await _retrieve_context(
        query=request.query,
        top_k=request.top_k,
        collection=request.collection,
        use_reranker=request.use_reranker,
    )
    
    # Step 2: Build prompt
    system_prompt = request.system_prompt or _get_rag_system_prompt()
    user_prompt = _build_context_prompt(request.query, context_chunks)
    
    # Step 3: Generate answer
    generation_start = time.perf_counter()
    try:
        llm = get_llm_client()
        
        result = await llm.chat(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            model=request.model,
        )
        
        answer = result.content
        
    except asyncio.TimeoutError:
        raise TimeoutError(operation="RAG generation")
    except Exception as e:
        _LOG.error("RAG generation error: %s", e, exc_info=True)
        raise ServiceUnavailableError(f"Generation failed: {str(e)}")
    
    generation_latency_ms = (time.perf_counter() - generation_start) * 1000
    total_latency_ms = (time.perf_counter() - start_time) * 1000
    
    # Calculate tokens (approximate)
    prompt_tokens = max(1, len(user_prompt) // 4)
    completion_tokens = max(1, len(answer) // 4)
    
    # Step 4: Optionally store in history
    session_id = None
    history_id = None
    
    if request.store_history:
        try:
            # Get or create session
            if request.session_id:
                session = await sessions.get(request.session_id)
                if session:
                    session_id = request.session_id
            
            if not session_id:
                session = await sessions.create()
                session_id = session["id"]
            
            # Create history entry
            history_entry = await history_repo.create(
                session_id=session_id,
                query=request.query,
                role="user",
                tokens_query=prompt_tokens,
            )
            history_id = history_entry["id"]
            
            # Update with answer
            await history_repo.update_answer(
                history_id=history_id,
                answer=answer,
                tokens_answer=completion_tokens,
                latency_ms=total_latency_ms,
            )
        except Exception as e:
            _LOG.warning("Failed to store in history: %s", e)
    
    response_id = generate_uuid()
    
    # Background: track response
    if history_id:
        background_tasks.add_task(
            _track_rag_response,
            response_repo,
            response_id,
            history_id,
            request,
            answer,
            prompt_tokens,
            completion_tokens,
            total_latency_ms,
        )
    
    return RAGQueryResponse(
        id=response_id,
        session_id=session_id,
        query=request.query,
        answer=answer,
        context=context_chunks,
        model=request.model or settings.llm_model,
        tokens_prompt=prompt_tokens,
        tokens_completion=completion_tokens,
        latency_ms=total_latency_ms,
        retrieval_latency_ms=retrieval_latency_ms,
        generation_latency_ms=generation_latency_ms,
        metadata={
            "history_id": history_id,
            "context_count": len(context_chunks),
            "used_reranker": request.use_reranker and len(context_chunks) > 0,
        },
    )


# =============================================================================
# RAG Query Streaming Endpoint
# =============================================================================

@router.post(
    "/query/stream",
    summary="Streaming RAG Query",
    description="""
    End-to-end RAG query with streaming response (Server-Sent Events).
    
    **SSE Format**:
    ```
    data: {"type": "context", "chunks": [...]}
    
    data: {"type": "token", "content": "Hello"}
    
    data: {"type": "done", "response": {...}}
    ```
    
    **Client Example**:
    ```javascript
    const response = await fetch('/api/v1/rag/query/stream', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({query: "What is...?"})
    });
    const reader = response.body.getReader();
    // Process stream...
    ```
    """,
    response_class=StreamingResponse,
)
async def rag_query_stream(
    request: RAGQueryRequest,
    sessions: SessionRepository = Depends(get_session_repo),
    history_repo: HistoryRepository = Depends(get_history_repo),
) -> StreamingResponse:
    """Execute end-to-end RAG query with streaming response."""
    
    async def generate_sse() -> AsyncGenerator[str, None]:
        """Generate SSE events for streaming RAG response."""
        start_time = time.perf_counter()
        settings = get_settings()
        full_response = ""
        
        try:
            # Step 1: Retrieve context
            context_chunks, retrieval_latency_ms = await _retrieve_context(
                query=request.query,
                top_k=request.top_k,
                collection=request.collection,
                use_reranker=request.use_reranker,
            )
            
            # Send context to client
            context_data = [
                {
                    "text": chunk.text[:500],  # Truncate for SSE
                    "score": chunk.score,
                    "doc_id": chunk.doc_id,
                }
                for chunk in context_chunks
            ]
            yield f"data: {json.dumps({'type': 'context', 'chunks': context_data, 'retrieval_latency_ms': retrieval_latency_ms})}\n\n"
            
            # Step 2: Build prompt
            system_prompt = request.system_prompt or _get_rag_system_prompt()
            user_prompt = _build_context_prompt(request.query, context_chunks)
            
            # Step 3: Stream generation
            llm = get_llm_client()
            
            async for token in llm.chat_stream(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                model=request.model,
            ):
                full_response += token
                yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"
            
            # Calculate final stats
            total_latency_ms = (time.perf_counter() - start_time) * 1000
            prompt_tokens = max(1, len(user_prompt) // 4)
            completion_tokens = max(1, len(full_response) // 4)
            
            # Optionally store in history
            session_id = None
            if request.store_history:
                try:
                    if request.session_id:
                        session = await sessions.get(request.session_id)
                        if session:
                            session_id = request.session_id
                    
                    if not session_id:
                        session = await sessions.create()
                        session_id = session["id"]
                    
                    history_entry = await history_repo.create(
                        session_id=session_id,
                        query=request.query,
                        role="user",
                        tokens_query=prompt_tokens,
                    )
                    await history_repo.update_answer(
                        history_id=history_entry["id"],
                        answer=full_response,
                        tokens_answer=completion_tokens,
                        latency_ms=total_latency_ms,
                    )
                except Exception as e:
                    _LOG.warning("Failed to store streaming result in history: %s", e)
            
            # Send completion event
            yield f"data: {json.dumps({'type': 'done', 'response': {'answer': full_response, 'session_id': session_id, 'latency_ms': total_latency_ms, 'model': request.model or settings.llm_model, 'context_count': len(context_chunks)}})}\n\n"
            
        except Exception as e:
            _LOG.error("Streaming RAG error: %s", e)
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
    
    return StreamingResponse(
        generate_sse(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# =============================================================================
# Simple GET Endpoint for Quick Queries
# =============================================================================

@router.get(
    "/ask",
    response_model=RAGQueryResponse,
    summary="Quick RAG Query (GET)",
    description="""
    Simple GET endpoint for quick RAG queries.
    
    **Usage**: 
    ```
    GET /api/v1/rag/ask?q=What is the capital of France?&top_k=3
    ```
    
    For full options, use POST `/rag/query` instead.
    """,
)
async def rag_ask(
    q: str,
    top_k: int = 5,
    background_tasks: BackgroundTasks = BackgroundTasks(),
    sessions: SessionRepository = Depends(get_session_repo),
    history_repo: HistoryRepository = Depends(get_history_repo),
    response_repo: ResponseRepository = Depends(get_response_repo),
) -> RAGQueryResponse:
    """Quick GET-based RAG query."""
    request = RAGQueryRequest(
        query=q,
        top_k=top_k,
        store_history=False,  # GET requests don't store history by default
    )
    return await rag_query(
        request=request,
        background_tasks=background_tasks,
        sessions=sessions,
        history_repo=history_repo,
        response_repo=response_repo,
    )


# =============================================================================
# Background Task Helper
# =============================================================================

async def _track_rag_response(
    response_repo: ResponseRepository,
    response_id: str,
    history_id: str,
    request: RAGQueryRequest,
    answer: str,
    tokens_prompt: int,
    tokens_completion: int,
    latency_ms: float,
) -> None:
    """Track RAG response in database (background task)."""
    try:
        settings = get_settings()
        await response_repo.create(
            history_id=history_id,
            model=request.model or settings.llm_model,
            prompt=request.query,
            response=answer,
            temperature=request.temperature or settings.llm_temperature,
            max_tokens=request.max_tokens or settings.llm_max_tokens,
            tokens_prompt=tokens_prompt,
            tokens_completion=tokens_completion,
            finish_reason="stop",
            latency_ms=latency_ms,
        )
    except Exception as e:
        _LOG.warning("Failed to track RAG response: %s", e)
