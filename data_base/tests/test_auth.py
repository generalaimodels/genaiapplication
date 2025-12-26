# ==============================================================================
# AUTH ENDPOINT TESTS
# ==============================================================================
# Tests for authentication endpoints: register, login
# ==============================================================================

import pytest
from httpx import AsyncClient


class TestAuthRegister:
    """Tests for user registration endpoint."""
    
    @pytest.mark.asyncio
    async def test_register_success(self, client: AsyncClient, sample_user_data: dict):
        """Test successful user registration."""
        response = await client.post("/api/v1/auth/register", json=sample_user_data)
        
        assert response.status_code == 201
        data = response.json()
        
        assert data["success"] is True
        assert data["data"]["email"] == sample_user_data["email"]
        assert data["data"]["full_name"] == sample_user_data["full_name"]
        assert "id" in data["data"]
        assert "hashed_password" not in data["data"]
    
    @pytest.mark.asyncio
    async def test_register_duplicate_email(self, client: AsyncClient, sample_user_data: dict):
        """Test registration with duplicate email fails."""
        # Register first user
        response1 = await client.post("/api/v1/auth/register", json=sample_user_data)
        assert response1.status_code == 201
        
        # Try registering with same email
        response2 = await client.post("/api/v1/auth/register", json=sample_user_data)
        assert response2.status_code == 409
    
    @pytest.mark.asyncio
    async def test_register_invalid_email(self, client: AsyncClient):
        """Test registration with invalid email fails."""
        data = {
            "email": "invalid-email",
            "password": "SecurePass123!",
            "full_name": "Test User",
        }
        response = await client.post("/api/v1/auth/register", json=data)
        assert response.status_code == 422
    
    @pytest.mark.asyncio
    async def test_register_weak_password(self, client: AsyncClient):
        """Test registration with weak password fails."""
        data = {
            "email": "test@example.com",
            "password": "weak",
            "full_name": "Test User",
        }
        response = await client.post("/api/v1/auth/register", json=data)
        assert response.status_code == 422


class TestAuthLogin:
    """Tests for user login endpoint."""
    
    @pytest.mark.asyncio
    async def test_login_json_success(self, client: AsyncClient, sample_user_data: dict):
        """Test successful login with JSON."""
        # Register user first
        await client.post("/api/v1/auth/register", json=sample_user_data)
        
        # Login
        login_data = {
            "email": sample_user_data["email"],
            "password": sample_user_data["password"],
        }
        response = await client.post("/api/v1/auth/login/json", json=login_data)
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] is True
        assert "tokens" in data["data"]
        assert "access_token" in data["data"]["tokens"]
        assert "refresh_token" in data["data"]["tokens"]
        assert data["data"]["tokens"]["token_type"] == "bearer"
    
    @pytest.mark.asyncio
    async def test_login_wrong_password(self, client: AsyncClient, sample_user_data: dict):
        """Test login with wrong password fails."""
        # Register user first
        await client.post("/api/v1/auth/register", json=sample_user_data)
        
        # Login with wrong password
        login_data = {
            "email": sample_user_data["email"],
            "password": "WrongPassword123!",
        }
        response = await client.post("/api/v1/auth/login/json", json=login_data)
        
        assert response.status_code == 401
    
    @pytest.mark.asyncio
    async def test_login_nonexistent_user(self, client: AsyncClient):
        """Test login with nonexistent user fails."""
        login_data = {
            "email": "nonexistent@example.com",
            "password": "SomePassword123!",
        }
        response = await client.post("/api/v1/auth/login/json", json=login_data)
        
        assert response.status_code == 401
