# -*- coding: utf-8 -*-
"""
Pytest Fixtures for CCA Chatbot API Tests

Provides:
- TestClient: FastAPI test client with app instance
- Mock fixtures: Mock LLM, VectorBase, and other services
- Sample data: Test requests and expected responses
"""

from __future__ import annotations

import asyncio
import pytest
from typing import Any, Dict, Generator, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi.testclient import TestClient

# -----------------------------------------------------------------------------
# App Fixture
# -----------------------------------------------------------------------------

@pytest.fixture(scope="session")
def app():
    """Create FastAPI app instance for testing."""
    from api.main import create_app
    return create_app()


@pytest.fixture(scope="session")
def client(app) -> Generator[TestClient, None, None]:
    """Create TestClient for making requests."""
    with TestClient(app) as c:
        yield c


# -----------------------------------------------------------------------------
# Mock LLM Client
# -----------------------------------------------------------------------------

class MockChatResult:
    """Mock chat result from LLM."""
    def __init__(self, content: str):
        self.content = content
        self.raw = {"mock": True}


class MockCompletionResult:
    """Mock completion result from LLM."""
    def __init__(self, text: str):
        self.text = text
        self.raw = {"mock": True}


class MockLLMClient:
    """Mock LLM client for testing without real API calls."""
    
    async def chat(
        self,
        *,
        system_prompt: Optional[str] = None,
        user_prompt: Optional[str] = None,
        **kwargs
    ) -> MockChatResult:
        """Return mock chat response."""
        return MockChatResult(f"Mock response to: {user_prompt[:50] if user_prompt else 'empty'}")
    
    async def complete(
        self,
        *,
        prompt: Optional[str] = None,
        **kwargs
    ) -> MockCompletionResult:
        """Return mock completion response."""
        return MockCompletionResult(f"Mock completion for: {prompt[:50] if prompt else 'empty'}")
    
    async def chat_stream(
        self,
        *,
        system_prompt: Optional[str] = None,
        user_prompt: Optional[str] = None,
        **kwargs
    ):
        """Yield mock streaming tokens."""
        mock_response = f"Mock streaming response to: {user_prompt[:30] if user_prompt else 'empty'}"
        for word in mock_response.split():
            yield word + " "
            await asyncio.sleep(0.01)
    
    async def run_batch_chat(
        self,
        user_prompts: List[str],
        **kwargs
    ) -> List[MockChatResult]:
        """Return mock batch responses."""
        return [MockChatResult(f"Mock batch response {i}") for i in range(len(user_prompts))]


@pytest.fixture
def mock_llm_client():
    """Provide mock LLM client."""
    return MockLLMClient()


@pytest.fixture
def patch_llm_client(mock_llm_client):
    """Patch get_llm_client to return mock."""
    with patch("api.dependencies.get_llm_client", return_value=mock_llm_client):
        with patch("api.routers.generation.get_llm_client", return_value=mock_llm_client):
            with patch("api.routers.rag.get_llm_client", return_value=mock_llm_client):
                yield mock_llm_client


# -----------------------------------------------------------------------------
# Mock VectorBase
# -----------------------------------------------------------------------------

class MockDocChunk:
    """Mock DocChunk record for testing."""
    def __init__(self, doc_id: str, index: int, text: str, start: int = 0, end: int = 0):
        self.doc_id = doc_id
        self.index = index
        self.text = text
        self.start = start
        self.end = end
        self.meta = {"source": "test"}


class MockCollection:
    """Mock collection for vector store with full attribute support."""
    def __init__(self):
        self.texts = [
            "Document about machine learning concepts and algorithms.",
            "Guide to Python programming best practices and patterns.",
            "Information about data storage and compliance requirements.",
        ]
        # Create proper DocChunk-like records
        self.records = [
            MockDocChunk("doc1", 0, self.texts[0], 0, len(self.texts[0])),
            MockDocChunk("doc2", 0, self.texts[1], 0, len(self.texts[1])),
            MockDocChunk("doc3", 0, self.texts[2], 0, len(self.texts[2])),
        ]
        self.metadata = [
            {"doc_id": "doc1", "title": "ML Basics"},
            {"doc_id": "doc2", "title": "Python Guide"},
            {"doc_id": "doc3", "title": "Data Compliance"},
        ]
        self.name = "test_collection"
        self.metric = "cosine"
    
    @property
    def size(self) -> int:
        return len(self.texts)


class MockVectorBase:
    """
    Mock vector base for testing without real embeddings.
    
    Simulates VectorBase.search() API returning:
        - results: List[List[Tuple[int, float]]] - per-query list of (doc_id, distance)
        - contexts: List[Dict] - per-query context metadata
    """
    
    def __init__(self):
        self.collection = MockCollection()
        self.dim = 768
        self.index = True
        self.device = "cpu"
    
    def search(self, query: str, k: int = 5, **kwargs):
        """
        Return mock search results in VectorBase.search() format.
        
        Returns:
            Tuple of (results, contexts) where:
            - results: List[List[Tuple[int, float]]] - ranked (id, distance) pairs
            - contexts: List[Dict] - context metadata
        """
        # Mock results: lower distance = more relevant
        mock_results = [
            [(0, 0.15), (1, 0.25), (2, 0.35)][:k]
        ]
        
        mock_contexts = [{
            "topk": [
                {"id": 0, "doc_id": "doc1", "distance": 0.15, "text": self.collection.texts[0]},
                {"id": 1, "doc_id": "doc2", "distance": 0.25, "text": self.collection.texts[1]},
                {"id": 2, "doc_id": "doc3", "distance": 0.35, "text": self.collection.texts[2]},
            ][:k]
        }]
        
        return mock_results, mock_contexts
    
    def create_collection(self, name: str, dim: int, metric: str):
        """Mock create collection."""
        pass


@pytest.fixture
def mock_vector_base():
    """Provide mock vector base."""
    return MockVectorBase()


@pytest.fixture
def patch_vector_base(mock_vector_base):
    """Patch get_vector_base to return mock."""
    with patch("api.dependencies.get_vector_base", return_value=mock_vector_base):
        with patch("api.routers.search.get_vector_base", return_value=mock_vector_base):
            with patch("api.routers.generation.get_vector_base", return_value=mock_vector_base):
                with patch("api.routers.rag.get_vector_base", return_value=mock_vector_base):
                    yield mock_vector_base


# -----------------------------------------------------------------------------
# Sample Test Data
# -----------------------------------------------------------------------------

@pytest.fixture
def sample_chat_request() -> Dict[str, Any]:
    """Sample chat request data."""
    return {
        "message": "What is machine learning?",
        "temperature": 0.7,
        "max_tokens": 1024,
        "include_context": True,
    }


@pytest.fixture
def sample_rag_query_request() -> Dict[str, Any]:
    """Sample RAG query request data."""
    return {
        "query": "What are the compliance requirements for data storage?",
        "top_k": 5,
        "temperature": 0.7,
        "use_reranker": False,  # Disable reranker for simpler testing
        "store_history": False,
    }


@pytest.fixture
def sample_search_request() -> Dict[str, Any]:
    """Sample search request data."""
    return {
        "query": "machine learning",
        "top_k": 10,
        "include_text": True,
    }


@pytest.fixture
def sample_session_create() -> Dict[str, Any]:
    """Sample session creation data."""
    return {
        "user_id": "test_user",
        "title": "Test Session",
        "metadata": {"source": "pytest"},
    }
