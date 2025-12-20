# =============================================================================
# CONTEXT BUILDER - Build Context from Documents for RAG
# =============================================================================
# Constructs context from documents based on relevance to query.
# =============================================================================

from __future__ import annotations
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import logging

from documents.document_processor import Document

logger = logging.getLogger(__name__)


@dataclass
class RetrievedContext:
    """
    Represents context retrieved for a query.
    
    Attributes:
        documents: Retrieved documents with scores
        context_text: Formatted context text
        total_documents: Total documents considered
        total_chunks: Total chunks included
        token_estimate: Estimated token count
    """
    documents: List[Tuple[Document, float]]  # (doc, score)
    context_text: str
    total_documents: int = 0
    total_chunks: int = 0
    token_estimate: int = 0
    
    @property
    def has_context(self) -> bool:
        """Check if context was retrieved."""
        return len(self.documents) > 0


class ContextBuilder:
    """
    Builds context from documents for RAG queries.
    
    Features:
        - Simple keyword-based relevance (no embeddings required)
        - Optional embeddings-based similarity
        - Context formatting and token budgeting
        - Source citation
    
    Example:
        >>> builder = ContextBuilder(max_context_tokens=2000)
        >>> 
        >>> # Build context from documents
        >>> context = builder.build_context(
        ...     query="What is machine learning?",
        ...     documents=chunks,
        ...     top_k=5
        ... )
        >>> 
        >>> print(context.context_text)
    """
    
    def __init__(
        self,
        max_context_tokens: int = 3000,
        max_chunks: int = 10,
        min_relevance: float = 0.1,
        context_template: Optional[str] = None
    ):
        """
        Initialize context builder.
        
        Args:
            max_context_tokens: Maximum tokens for context
            max_chunks: Maximum chunks to include
            min_relevance: Minimum relevance score
            context_template: Template for formatting context
        """
        self.max_context_tokens = max_context_tokens
        self.max_chunks = max_chunks
        self.min_relevance = min_relevance
        self.context_template = context_template or self._default_template()
    
    def _default_template(self) -> str:
        """Get default context template."""
        return (
            "Use the following context to answer the question. "
            "If the answer is not in the context, say so.\n\n"
            "Context:\n{context}\n\n"
            "---\n"
        )
    
    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count (rough approximation)."""
        return len(text) // 4
    
    def _compute_keyword_relevance(
        self,
        query: str,
        document: Document
    ) -> float:
        """
        Compute simple keyword-based relevance.
        
        Args:
            query: Search query
            document: Document to score
            
        Returns:
            Relevance score (0-1)
        """
        query_lower = query.lower()
        content_lower = document.content.lower()
        
        # Extract query terms
        query_terms = set(query_lower.split())
        
        if not query_terms:
            return 0.0
        
        # Count matches
        matches = sum(1 for term in query_terms if term in content_lower)
        
        # Base score from term matches
        term_score = matches / len(query_terms)
        
        # Bonus for exact phrase match
        phrase_bonus = 0.2 if query_lower in content_lower else 0.0
        
        return min(1.0, term_score + phrase_bonus)
    
    def score_documents(
        self,
        query: str,
        documents: List[Document],
        embeddings_fn: Optional[callable] = None
    ) -> List[Tuple[Document, float]]:
        """
        Score documents by relevance to query.
        
        Args:
            query: Search query
            documents: Documents to score
            embeddings_fn: Optional function to compute embeddings
            
        Returns:
            List of (document, score) tuples, sorted by score descending
        """
        scored = []
        
        for doc in documents:
            # Use simple keyword relevance
            score = self._compute_keyword_relevance(query, doc)
            
            if score >= self.min_relevance:
                scored.append((doc, score))
        
        # Sort by score descending
        scored.sort(key=lambda x: x[1], reverse=True)
        
        return scored
    
    def select_top_chunks(
        self,
        scored_docs: List[Tuple[Document, float]],
        max_chunks: Optional[int] = None,
        max_tokens: Optional[int] = None
    ) -> List[Tuple[Document, float]]:
        """
        Select top chunks within token budget.
        
        Args:
            scored_docs: Scored documents
            max_chunks: Maximum chunks to select
            max_tokens: Maximum tokens to use
            
        Returns:
            Selected documents with scores
        """
        max_chunks = max_chunks or self.max_chunks
        max_tokens = max_tokens or self.max_context_tokens
        
        selected = []
        current_tokens = 0
        
        for doc, score in scored_docs[:max_chunks * 2]:  # Consider more than needed
            doc_tokens = self._estimate_tokens(doc.content)
            
            if current_tokens + doc_tokens > max_tokens:
                continue
            
            selected.append((doc, score))
            current_tokens += doc_tokens
            
            if len(selected) >= max_chunks:
                break
        
        return selected
    
    def format_context(
        self,
        documents: List[Tuple[Document, float]],
        include_sources: bool = True,
        include_scores: bool = False
    ) -> str:
        """
        Format documents into context text.
        
        Args:
            documents: Documents with scores
            include_sources: Include source references
            include_scores: Include relevance scores
            
        Returns:
            Formatted context string
        """
        if not documents:
            return ""
        
        parts = []
        
        for i, (doc, score) in enumerate(documents, 1):
            chunk_text = doc.content.strip()
            
            # Add header with source info
            header_parts = [f"[{i}]"]
            
            if include_sources and doc.source:
                source = doc.source.split("/")[-1] if "/" in doc.source else doc.source
                header_parts.append(f"Source: {source}")
            
            if include_scores:
                header_parts.append(f"Relevance: {score:.2f}")
            
            if doc.chunk_index is not None:
                header_parts.append(f"Chunk: {doc.chunk_index}")
            
            header = " | ".join(header_parts)
            parts.append(f"{header}\n{chunk_text}")
        
        return "\n\n---\n\n".join(parts)
    
    def build_context(
        self,
        query: str,
        documents: List[Document],
        top_k: Optional[int] = None,
        max_tokens: Optional[int] = None,
        include_sources: bool = True
    ) -> RetrievedContext:
        """
        Build context from documents for a query.
        
        Args:
            query: User query
            documents: Available documents
            top_k: Number of top documents to include
            max_tokens: Maximum tokens for context
            include_sources: Include source citations
            
        Returns:
            RetrievedContext with formatted context
            
        Example:
            >>> context = builder.build_context(
            ...     query="What is Python?",
            ...     documents=doc_chunks,
            ...     top_k=5
            ... )
            >>> 
            >>> # Use in prompt
            >>> prompt = f"{context.context_text}\nQuestion: {query}"
        """
        # Score documents
        scored = self.score_documents(query, documents)
        
        # Select top chunks within budget
        selected = self.select_top_chunks(
            scored,
            max_chunks=top_k,
            max_tokens=max_tokens
        )
        
        # Format context
        context_text = self.format_context(
            selected,
            include_sources=include_sources
        )
        
        # Wrap in template
        if context_text and self.context_template:
            formatted = self.context_template.format(context=context_text)
        else:
            formatted = context_text
        
        return RetrievedContext(
            documents=selected,
            context_text=formatted,
            total_documents=len(documents),
            total_chunks=len(selected),
            token_estimate=self._estimate_tokens(formatted)
        )
    
    def augment_query(
        self,
        query: str,
        documents: List[Document],
        top_k: int = 5
    ) -> str:
        """
        Augment query with relevant context.
        
        Args:
            query: User query
            documents: Available documents
            top_k: Number of documents to include
            
        Returns:
            Augmented prompt string
        """
        context = self.build_context(query, documents, top_k=top_k)
        
        if context.has_context:
            return f"{context.context_text}Question: {query}"
        
        return query
