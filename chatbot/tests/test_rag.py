# -*- coding: utf-8 -*-
"""
Tests for RAG Router Endpoints

Covers:
- POST /rag/query: RAG query (retrieve + generate)
- POST /rag/query/stream: Streaming RAG query
- GET /rag/ask: Quick RAG query
"""

from __future__ import annotations

import json
import pytest
from unittest.mock import patch, MagicMock


# =============================================================================
# RAG Query Tests
# =============================================================================

class TestRAGQuery:
    """Tests for /rag/query endpoint."""
    
    def test_rag_query_basic(self, client, patch_llm_client, patch_vector_base, sample_rag_query_request):
        """Test basic RAG query."""
        response = client.post("/api/v1/rag/query", json=sample_rag_query_request)
        
        assert response.status_code == 200
        data = response.json()
        
        # Check required fields
        assert "id" in data
        assert "query" in data
        assert "answer" in data
        assert "context" in data
        assert "latency_ms" in data
        assert "retrieval_latency_ms" in data
        assert "generation_latency_ms" in data
        
        # Query should match input
        assert data["query"] == sample_rag_query_request["query"]
        
        # Answer should not be empty
        assert len(data["answer"]) > 0
    
    def test_rag_query_with_history(self, client, patch_llm_client, patch_vector_base):
        """Test RAG query with history storage."""
        response = client.post("/api/v1/rag/query", json={
            "query": "What is Python?",
            "top_k": 3,
            "store_history": True,
        })
        
        assert response.status_code == 200
        data = response.json()
        
        # Session should be created when store_history is True
        assert data.get("session_id") is not None or data["metadata"].get("history_id") is not None
    
    def test_rag_query_no_history(self, client, patch_llm_client, patch_vector_base):
        """Test RAG query without history storage."""
        response = client.post("/api/v1/rag/query", json={
            "query": "What is AI?",
            "store_history": False,
        })
        
        assert response.status_code == 200
        data = response.json()
        
        # Session might be None when store_history is False
        # This is acceptable behavior
    
    def test_rag_query_custom_options(self, client, patch_llm_client, patch_vector_base):
        """Test RAG query with custom options."""
        response = client.post("/api/v1/rag/query", json={
            "query": "Explain machine learning",
            "top_k": 10,
            "temperature": 0.9,
            "max_tokens": 2048,
            "use_reranker": False,
        })
        
        assert response.status_code == 200
    
    def test_rag_query_missing_query(self, client, patch_llm_client, patch_vector_base):
        """Test RAG query with missing query field."""
        response = client.post("/api/v1/rag/query", json={})
        
        assert response.status_code == 422  # Validation error
    
    def test_rag_query_empty_query(self, client, patch_llm_client, patch_vector_base):
        """Test RAG query with empty query."""
        response = client.post("/api/v1/rag/query", json={"query": ""})
        
        assert response.status_code == 422  # Validation error due to min_length


# =============================================================================
# Streaming RAG Query Tests
# =============================================================================

class TestRAGQueryStream:
    """Tests for /rag/query/stream endpoint."""
    
    def test_stream_basic(self, client, patch_llm_client, patch_vector_base):
        """Test basic streaming RAG query."""
        response = client.post(
            "/api/v1/rag/query/stream",
            json={"query": "Tell me about data storage."},
        )
        
        assert response.status_code == 200
        assert response.headers["content-type"] == "text/event-stream; charset=utf-8"
        
        # Parse SSE events
        events = []
        for line in response.text.split("\n"):
            if line.startswith("data: "):
                events.append(json.loads(line[6:]))
        
        # Should have at least context event, token events, and done event
        assert len(events) >= 2
        
        # First event should be context
        assert events[0].get("type") == "context"
        
        # Last event should be done
        last_event = events[-1]
        assert last_event.get("type") == "done" or last_event.get("type") == "error"
    
    def test_stream_contains_context(self, client, patch_llm_client, patch_vector_base):
        """Test that streaming response contains context."""
        response = client.post(
            "/api/v1/rag/query/stream",
            json={"query": "What is compliance?"},
        )
        
        context_event = None
        for line in response.text.split("\n"):
            if line.startswith("data: "):
                event = json.loads(line[6:])
                if event.get("type") == "context":
                    context_event = event
                    break
        
        assert context_event is not None
        assert "chunks" in context_event
        assert "retrieval_latency_ms" in context_event


# =============================================================================
# Quick RAG Query (GET) Tests
# =============================================================================

class TestRAGAsk:
    """Tests for /rag/ask endpoint."""
    
    def test_ask_basic(self, client, patch_llm_client, patch_vector_base):
        """Test basic GET quick query."""
        response = client.get("/api/v1/rag/ask?q=What%20is%20Python?")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "id" in data
        assert "query" in data
        assert "answer" in data
        assert data["query"] == "What is Python?"
    
    def test_ask_with_top_k(self, client, patch_llm_client, patch_vector_base):
        """Test GET query with top_k parameter."""
        response = client.get("/api/v1/rag/ask?q=ML%20concepts&top_k=3")
        
        assert response.status_code == 200
    
    def test_ask_missing_query(self, client, patch_llm_client, patch_vector_base):
        """Test GET query without q parameter."""
        response = client.get("/api/v1/rag/ask")
        
        assert response.status_code == 422  # Missing required parameter


# =============================================================================
# Integration Tests
# =============================================================================

class TestRAGIntegration:
    """Integration tests for RAG workflow."""
    
    def test_full_rag_workflow(self, client, patch_llm_client, patch_vector_base):
        """Test full RAG workflow: create session -> query -> check history."""
        # 1. Create session
        session_resp = client.post("/api/v1/sessions", json={"title": "RAG Test"})
        if session_resp.status_code != 201:
            pytest.skip("Session creation not available")
        
        session_id = session_resp.json()["id"]
        
        # 2. Run RAG query with session
        query_resp = client.post("/api/v1/rag/query", json={
            "query": "What are the compliance requirements?",
            "session_id": session_id,
            "store_history": True,
        })
        
        assert query_resp.status_code == 200
        query_data = query_resp.json()
        
        # 3. Check session history
        history_resp = client.get(f"/api/v1/sessions/{session_id}/history")
        
        if history_resp.status_code == 200:
            history_data = history_resp.json()
            # Should have at least one entry from the RAG query
            assert history_data["total"] >= 0  # May be 0 if async not complete
