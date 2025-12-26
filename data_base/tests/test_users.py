# ==============================================================================
# USER ENDPOINT TESTS
# ==============================================================================
# Tests for user profile endpoints
# ==============================================================================

import pytest
from httpx import AsyncClient


class TestUserMe:
    """Tests for /users/me endpoint."""
    
    @pytest.mark.asyncio
    async def test_get_current_user(self, auth_client):
        """Test getting current user profile."""
        client, user_id, user_data = auth_client
        
        response = await client.get("/api/v1/users/me")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] is True
        assert data["data"]["id"] == user_id
        assert data["data"]["email"] == user_data["email"]
    
    @pytest.mark.asyncio
    async def test_get_current_user_unauthenticated(self, client: AsyncClient):
        """Test getting current user without auth fails."""
        response = await client.get("/api/v1/users/me")
        assert response.status_code == 401
    
    @pytest.mark.asyncio
    async def test_update_current_user(self, auth_client):
        """Test updating current user profile."""
        client, user_id, _ = auth_client
        
        update_data = {
            "full_name": "Updated Name",
            "bio": "This is my updated bio",
        }
        
        response = await client.patch("/api/v1/users/me", json=update_data)
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] is True
        assert data["data"]["full_name"] == "Updated Name"
        assert data["data"]["bio"] == "This is my updated bio"


class TestUserById:
    """Tests for /users/{user_id} endpoint."""
    
    @pytest.mark.asyncio
    async def test_get_user_by_id(self, auth_client):
        """Test getting user by ID."""
        client, user_id, user_data = auth_client
        
        response = await client.get(f"/api/v1/users/{user_id}")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] is True
        assert data["data"]["id"] == user_id
        assert data["data"]["email"] == user_data["email"]
    
    @pytest.mark.asyncio
    async def test_get_nonexistent_user(self, auth_client):
        """Test getting nonexistent user returns 404."""
        client, _, _ = auth_client
        
        fake_id = "00000000-0000-0000-0000-000000000000"
        response = await client.get(f"/api/v1/users/{fake_id}")
        
        assert response.status_code == 404


class TestPasswordChange:
    """Tests for password change endpoint."""
    
    @pytest.mark.asyncio
    async def test_change_password_success(self, auth_client):
        """Test successful password change."""
        client, _, user_data = auth_client
        
        password_data = {
            "current_password": user_data["password"],
            "new_password": "NewSecurePass456!",
        }
        
        response = await client.post("/api/v1/users/me/password", json=password_data)
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
    
    @pytest.mark.asyncio
    async def test_change_password_wrong_current(self, auth_client):
        """Test password change with wrong current password fails."""
        client, _, _ = auth_client
        
        password_data = {
            "current_password": "WrongCurrentPass123!",
            "new_password": "NewSecurePass456!",
        }
        
        response = await client.post("/api/v1/users/me/password", json=password_data)
        
        assert response.status_code == 400
