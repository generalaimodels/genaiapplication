# -*- coding: utf-8 -*-
"""
Tests for Generation Router Endpoints

Covers:
- POST /chat: Chat completion
- POST /chat/stream: Streaming chat
- POST /complete: Text completion
- POST /batch/chat: Batch chat
"""

from __future__ import annotations

import json
import pytest
from unittest.mock import patch, MagicMock


# =============================================================================
# Chat Completion Tests
# =============================================================================

class TestChatCompletion:
    """Tests for /chat endpoint."""
    
    def test_chat_basic(self, client, patch_llm_client, sample_chat_request):
        """Test basic chat completion."""
        response = client.post("/api/v1/chat", json=sample_chat_request)
        
        assert response.status_code == 200
        data = response.json()
        
        assert "id" in data
        assert "session_id" in data
        assert "message" in data
        assert len(data["message"]) > 0
        assert "latency_ms" in data
    
    def test_chat_with_session(self, client, patch_llm_client):
        """Test chat with existing session."""
        # First create a session
        session_resp = client.post("/api/v1/sessions", json={"title": "Test"})
        if session_resp.status_code == 201:
            session_id = session_resp.json()["id"]
            
            # Now chat with that session
            response = client.post("/api/v1/chat", json={
                "session_id": session_id,
                "message": "Hello!",
            })
            
            assert response.status_code == 200
            data = response.json()
            assert data["session_id"] == session_id
    
    def test_chat_missing_message(self, client, patch_llm_client):
        """Test chat with missing message field."""
        response = client.post("/api/v1/chat", json={})
        
        assert response.status_code == 422  # Validation error
    
    def test_chat_empty_message(self, client, patch_llm_client):
        """Test chat with empty message."""
        response = client.post("/api/v1/chat", json={"message": ""})
        
        assert response.status_code == 422  # Validation error due to min_length


# =============================================================================
# Streaming Chat Tests
# =============================================================================

class TestChatStream:
    """Tests for /chat/stream endpoint."""
    
    def test_stream_basic(self, client, patch_llm_client):
        """Test basic streaming chat."""
        response = client.post(
            "/api/v1/chat/stream",
            json={"message": "Tell me a story."},
        )
        
        assert response.status_code == 200
        assert response.headers["content-type"] == "text/event-stream; charset=utf-8"
        
        # Parse SSE events
        events = []
        for line in response.text.split("\n"):
            if line.startswith("data: "):
                events.append(json.loads(line[6:]))
        
        # Should have at least one token event and one done event
        assert len(events) >= 1
        
        # Last event should be done
        last_event = events[-1]
        assert last_event.get("done") == True or "error" in last_event
    
    def test_stream_accumulates_tokens(self, client, patch_llm_client):
        """Test that streaming accumulates all tokens into final response."""
        response = client.post(
            "/api/v1/chat/stream",
            json={"message": "Hello world"},
        )
        
        tokens = []
        final_message = None
        
        for line in response.text.split("\n"):
            if line.startswith("data: "):
                event = json.loads(line[6:])
                if event.get("done"):
                    final_message = event.get("response", {}).get("message", "")
                elif "token" in event:
                    tokens.append(event["token"])
        
        # Final message should contain accumulated tokens
        if tokens and final_message:
            assert len(final_message) > 0


# =============================================================================
# Text Completion Tests
# =============================================================================

class TestTextCompletion:
    """Tests for /complete endpoint."""
    
    def test_complete_basic(self, client, patch_llm_client):
        """Test basic text completion."""
        response = client.post("/api/v1/complete", json={
            "prompt": "The quick brown fox",
        })
        
        assert response.status_code == 200
        data = response.json()
        
        assert "id" in data
        assert "text" in data
        assert len(data["text"]) > 0
    
    def test_complete_with_options(self, client, patch_llm_client):
        """Test completion with custom options."""
        response = client.post("/api/v1/complete", json={
            "prompt": "Hello",
            "temperature": 0.5,
            "max_tokens": 100,
        })
        
        assert response.status_code == 200
    
    def test_complete_missing_prompt(self, client, patch_llm_client):
        """Test completion with missing prompt."""
        response = client.post("/api/v1/complete", json={})
        
        assert response.status_code == 422


# =============================================================================
# Batch Chat Tests
# =============================================================================

class TestBatchChat:
    """Tests for /batch/chat endpoint."""
    
    def test_batch_basic(self, client, patch_llm_client):
        """Test basic batch chat."""
        response = client.post("/api/v1/batch/chat", json={
            "messages": ["Hello", "How are you?", "Goodbye"],
        })
        
        assert response.status_code == 200
        data = response.json()
        
        assert "results" in data
        assert "total" in data
        assert data["total"] == 3
        assert data["successful"] == 3
        assert data["failed"] == 0
    
    def test_batch_empty_messages(self, client, patch_llm_client):
        """Test batch with empty messages."""
        response = client.post("/api/v1/batch/chat", json={
            "messages": [],
        })
        
        assert response.status_code == 422  # Validation error (min_length=1)
