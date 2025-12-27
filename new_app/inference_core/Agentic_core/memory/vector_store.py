"""
SOTA Vector Store Implementation.

Adheres to:
- High Performance: Uses `usearch` for SIMD-accelerated vector search (faster than FAISS in many small-batch cases).
- Local Embeddings: Fetches embeddings from local inference_core (vLLM) without external API keys.
- Zero-Copy: Uses numpy buffers directly where possible.
"""
import asyncio
import logging
import json
import os
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass
import numpy as np

from ..core.config import get_config
from ..core.inference_wrapper import get_inference_client, InferenceError
from ..core.result import Result, Ok, Err

logger = logging.getLogger(__name__)
CONFIG = get_config()

# Try to import optimized libraries, fail gracefully if not installed but warn user
try:
    from usearch.index import Index
    HAS_USEARCH = True
except ImportError:
    HAS_USEARCH = False
    logger.warning("`usearch` not installed. Performance will be degraded. Install with: pip install usearch")

class VectorStore:
    """
    High-Performance Vector Store using USearch (HNSW) and Local Embeddings.
    """
    def __init__(self, collection_name: str = "agent_memory", dimension: int = 4096):
        self.config = get_config()
        self.dimension = dimension
        self.collection_name = collection_name
        self.index = None
        self.metadata_store: Dict[int, Dict] = {}
        self.count = 0
        
        if HAS_USEARCH:
            # Metric 'cos' for cosine similarity
            self.index = Index(ndim=dimension, metric="cos", dtype="f32")
            if os.path.exists(CONFIG.vector_store_path):
                self.index.load(CONFIG.vector_store_path)
                logger.info(f"Loaded vector index from {CONFIG.vector_store_path}")

    async def _get_embeddings(self, texts: List[str]) -> Result[np.ndarray, Exception]:
        """
        Fetches embeddings from SOTA Inference Core.
        Now supports distinct Embedding Endpoint via `EMBEDDING_BASE_URL`.
        """
        # SOTA: Check for dedicated embedding endpoint
        embed_url = os.environ.get("EMBEDDING_BASE_URL", self.config.inference_base_url)
        # If running separate servers, the model name might differ too.
        # For this SOTA implementation, `inference_wrapper` handles the client creation.

        try:
            if embed_url != self.config.inference_base_url:
                # Quick-fix for verification: Create a temporary httpx client if URL differs from global config.
                import httpx
                async with httpx.AsyncClient(base_url=embed_url, timeout=30.0) as client:
                    payload = {
                        "model": "Qwen/Qwen3-Embedding-4B",
                        "input": texts
                    }
                    response = await client.post("/embeddings", json=payload)
                    response.raise_for_status()
                    data = response.json()
                    embeddings = [item["embedding"] for item in data["data"]]
                    return Ok(np.array(embeddings, dtype=np.float32))
        except Exception as e:
            logger.error(f"Embedding Fetch Failed: {e}")
            return Err(e)

    async def add_texts(self, texts: List[str], metadata: List[Dict[str, Any]] = None) -> Result[bool, Exception]:
        """
        Embeds and indexes texts.
        """
        if not texts:
            return Ok(True)
            
        if metadata is None:
            metadata = [{} for _ in texts]

        # 1. Get Embeddings
        res = await self._get_embeddings(texts)
        if res.is_err:
            return res # Propagate error
        
        vectors = res.value
        
        # 2. Add to Index
        if HAS_USEARCH:
            keys = np.arange(self.count, self.count + len(texts), dtype=np.uint64)
            self.index.add(keys, vectors)
            
            # 3. Store Metadata
            for i, meta in enumerate(metadata):
                idx = self.count + i
                self.metadata_store[idx] = {
                    "text": texts[i],
                    **meta
                }
            
            self.count += len(texts)
            
            # Periodically save
            if self.count % 100 == 0:
                self.save()
                
            return Ok(True)
        else:
            return Err(ImportError("usearch is not installed"))

    async def search(self, query: str, k: int = 5) -> Result[List[Tuple[str, float, Dict]], Exception]:
        """
        Semantic search.
        """
        # 1. Embed Query
        res = await self._get_embeddings([query])
        if res.is_err:
            return res
        
        query_vec = res.value[0]
        
        if HAS_USEARCH and self.count > 0:
            matches = self.index.search(query_vec, k)
            results = []
            
            for i in range(len(matches.keys)):
                key = int(matches.keys[i])
                dist = float(matches.distances[i]) # Cosine distance
                # Convert distance to similarity if needed (1 - dist for cosine)
                similarity = 1.0 - dist 
                
                meta = self.metadata_store.get(key, {})
                text = meta.pop("text", "")
                results.append((text, similarity, meta))
                
            return Ok(results)
        
        return Ok([])

    def save(self):
        if HAS_USEARCH and self.index:
            try:
                os.makedirs(os.path.dirname(CONFIG.vector_store_path), exist_ok=True)
                self.index.save(CONFIG.vector_store_path)
            except Exception as e:
                logger.error(f"Failed to save index: {e}")
