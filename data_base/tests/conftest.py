# ==============================================================================
# CONFTEST - Pytest Fixtures and Configuration
# ==============================================================================
# Shared fixtures for all tests
# ==============================================================================

from __future__ import annotations

import asyncio
import os
import sys
from typing import AsyncGenerator
from uuid import uuid4

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

# Set test environment before importing app
os.environ["ENVIRONMENT"] = "development"
os.environ["DEBUG"] = "false"
os.environ["DATABASE_TYPE"] = "sqlite"
os.environ["SQLITE_URL"] = "sqlite+aiosqlite:///./test_app.db"
os.environ["SECRET_KEY"] = "test-secret-key-for-testing-only-32chars!"
os.environ["RATE_LIMIT_ENABLED"] = "false"

# Add app to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ==============================================================================
# HTTP CLIENT FIXTURES
# ==============================================================================

@pytest_asyncio.fixture
async def client() -> AsyncGenerator[AsyncClient, None]:
    """Create async HTTP client for API testing."""
    # Reset factory to ensure clean state
    from app.database.factory import DatabaseFactory
    DatabaseFactory.reset()
    
    # Remove old test db
    if os.path.exists("./test_app.db"):
        try:
            os.remove("./test_app.db")
        except (PermissionError, OSError):
            pass
    
    # Import app after environment is set
    from app.main import app
    
    # Initialize database with model registration
    try:
        await DatabaseFactory.initialize()
    except Exception as e:
        print(f"DB Init error: {e}")
    
    transport = ASGITransport(app=app)
    async with AsyncClient(
        transport=transport,
        base_url="http://test",
        timeout=30.0,
    ) as async_client:
        yield async_client
    
    # Cleanup
    try:
        await DatabaseFactory.shutdown()
    except Exception:
        pass
    
    DatabaseFactory.reset()
    
    # Remove test database file
    if os.path.exists("./test_app.db"):
        try:
            os.remove("./test_app.db")
        except (PermissionError, OSError):
            pass


@pytest_asyncio.fixture
async def auth_client(client: AsyncClient) -> AsyncGenerator[tuple[AsyncClient, str, dict], None]:
    """
    Create authenticated client with test user.
    
    Returns:
        Tuple of (client, user_id, user_data)
    """
    from app.core.security import create_access_token
    
    # Create test user
    user_data = {
        "email": f"testuser_{uuid4().hex[:8]}@example.com",
        "password": "TestPassword123!",
        "full_name": "Test User",
    }
    
    # Register user
    response = await client.post("/api/v1/auth/register", json=user_data)
    assert response.status_code == 201, f"Failed to register: {response.text}"
    
    result = response.json()
    user_id = result["data"]["id"]
    
    # Create auth token
    token = create_access_token(subject=user_id)
    
    # Set auth header
    client.headers["Authorization"] = f"Bearer {token}"
    
    yield client, user_id, user_data
    
    # Cleanup - remove auth header
    if "Authorization" in client.headers:
        del client.headers["Authorization"]


# ==============================================================================
# HELPER FIXTURES
# ==============================================================================

@pytest.fixture
def sample_user_data() -> dict:
    """Generate sample user registration data."""
    return {
        "email": f"user_{uuid4().hex[:8]}@example.com",
        "password": "SecurePass123!",
        "full_name": "Sample User",
    }


@pytest.fixture
def sample_chat_session_data() -> dict:
    """Generate sample chat session data."""
    return {
        "title": f"Test Session {uuid4().hex[:6]}",
        "description": "A test chat session",
    }


@pytest.fixture
def sample_chat_message_data() -> dict:
    """Generate sample chat message data."""
    return {
        "role": "user",
        "content": "Hello, this is a test message!",
    }


@pytest.fixture
def sample_transaction_data() -> dict:
    """Generate sample transaction data."""
    return {
        "transaction_type": "credit",
        "amount": "100.00",
        "currency": "USD",
        "description": "Test deposit",
    }
