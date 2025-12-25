# =============================================================================
# SOTA AUTHENTICATION SYSTEM - TEST CONFIGURATION
# =============================================================================
# File: tests/conftest.py
# Description: Pytest fixtures for testing with in-memory SQLite and fakeredis
# =============================================================================

import asyncio
from typing import AsyncGenerator
import pytest
import pytest_asyncio

from fastapi.testclient import TestClient
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker

from db.base import Base
from db.adapters.sqlite_adapter import SQLiteAdapter
from db.factory import DBFactory
from main import app


# =============================================================================
# EVENT LOOP CONFIGURATION
# =============================================================================

@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


# =============================================================================
# DATABASE FIXTURES
# =============================================================================

@pytest_asyncio.fixture(scope="function")
async def db_adapter() -> AsyncGenerator[SQLiteAdapter, None]:
    """
    Create in-memory SQLite adapter for testing.
    
    Yields fresh database for each test.
    """
    adapter = SQLiteAdapter.create_for_testing()
    await adapter.connect()
    await adapter.create_tables()
    
    yield adapter
    
    await adapter.disconnect()


@pytest_asyncio.fixture(scope="function")
async def db_session(db_adapter: SQLiteAdapter) -> AsyncGenerator[AsyncSession, None]:
    """
    Create database session for testing.
    
    Each test gets its own session with automatic cleanup.
    """
    async with db_adapter.get_session() as session:
        yield session


# =============================================================================
# REDIS FIXTURES
# =============================================================================

@pytest_asyncio.fixture(scope="function")
async def redis_mock():
    """
    Create fake Redis for testing.
    
    Uses fakeredis to simulate Redis operations.
    """
    try:
        import fakeredis.aioredis
        
        fake_redis = fakeredis.aioredis.FakeRedis()
        yield fake_redis
        await fake_redis.close()
    except ImportError:
        # If fakeredis not available, skip Redis tests
        pytest.skip("fakeredis not installed")


# =============================================================================
# APPLICATION FIXTURES
# =============================================================================

@pytest.fixture(scope="function")
def test_client() -> TestClient:
    """
    Create synchronous test client.
    
    For simple API tests without async context.
    """
    return TestClient(app)


@pytest_asyncio.fixture(scope="function")
async def async_client() -> AsyncGenerator[AsyncClient, None]:
    """
    Create async test client.
    
    For testing async endpoints and WebSockets.
    """
    async with AsyncClient(app=app, base_url="http://test") as client:
        yield client


# =============================================================================
# USER FIXTURES
# =============================================================================

@pytest.fixture
def sample_user_data():
    """Sample user registration data."""
    return {
        "email": "test@example.com",
        "username": "testuser",
        "password": "SecurePass123!",
    }


@pytest.fixture
def sample_login_data():
    """Sample login credentials."""
    return {
        "email": "test@example.com",
        "password": "SecurePass123!",
    }


# =============================================================================
# TOKEN FIXTURES
# =============================================================================

@pytest.fixture
def sample_tokens():
    """Sample token data for testing."""
    return {
        "access_token": "eyJ...",
        "refresh_token": "eyJ...",
        "token_type": "bearer",
        "expires_in": 900,
        "session_id": "test-session-id",
    }
