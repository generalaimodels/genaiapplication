# ==============================================================================
# CHAT ENDPOINT TESTS
# ==============================================================================
# Tests for chat session and message endpoints
# ==============================================================================

import pytest
from httpx import AsyncClient


class TestChatSessions:
    """Tests for chat session CRUD endpoints."""
    
    @pytest.mark.asyncio
    async def test_create_session(self, auth_client, sample_chat_session_data: dict):
        """Test creating a new chat session."""
        client, user_id, _ = auth_client
        
        response = await client.post(
            "/api/v1/chat/sessions",
            json=sample_chat_session_data,
        )
        
        assert response.status_code == 201
        data = response.json()
        
        assert data["success"] is True
        assert data["data"]["title"] == sample_chat_session_data["title"]
        assert data["data"]["user_id"] == user_id
        assert data["data"]["is_active"] is True
        assert "id" in data["data"]
    
    @pytest.mark.asyncio
    async def test_create_session_unauthenticated(
        self, client: AsyncClient, sample_chat_session_data: dict
    ):
        """Test creating session without auth fails."""
        response = await client.post(
            "/api/v1/chat/sessions",
            json=sample_chat_session_data,
        )
        assert response.status_code == 401
    
    @pytest.mark.asyncio
    async def test_list_sessions(self, auth_client, sample_chat_session_data: dict):
        """Test listing user's chat sessions."""
        client, _, _ = auth_client
        
        # Create a few sessions
        for i in range(3):
            session_data = {**sample_chat_session_data, "title": f"Session {i}"}
            await client.post("/api/v1/chat/sessions", json=session_data)
        
        # List sessions
        response = await client.get("/api/v1/chat/sessions")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] is True
        assert len(data["data"]) >= 3
    
    @pytest.mark.asyncio
    async def test_get_session(self, auth_client, sample_chat_session_data: dict):
        """Test getting a specific chat session."""
        client, _, _ = auth_client
        
        # Create session
        create_response = await client.post(
            "/api/v1/chat/sessions",
            json=sample_chat_session_data,
        )
        session_id = create_response.json()["data"]["id"]
        
        # Get session
        response = await client.get(f"/api/v1/chat/sessions/{session_id}")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] is True
        assert data["data"]["id"] == session_id
    
    @pytest.mark.asyncio
    async def test_get_nonexistent_session(self, auth_client):
        """Test getting nonexistent session returns 404."""
        client, _, _ = auth_client
        
        fake_id = "00000000-0000-0000-0000-000000000000"
        response = await client.get(f"/api/v1/chat/sessions/{fake_id}")
        
        assert response.status_code == 404
    
    @pytest.mark.asyncio
    async def test_update_session(self, auth_client, sample_chat_session_data: dict):
        """Test updating a chat session."""
        client, _, _ = auth_client
        
        # Create session
        create_response = await client.post(
            "/api/v1/chat/sessions",
            json=sample_chat_session_data,
        )
        session_id = create_response.json()["data"]["id"]
        
        # Update session
        update_data = {"title": "Updated Title"}
        response = await client.patch(
            f"/api/v1/chat/sessions/{session_id}",
            json=update_data,
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] is True
        assert data["data"]["title"] == "Updated Title"
    
    @pytest.mark.asyncio
    async def test_delete_session(self, auth_client, sample_chat_session_data: dict):
        """Test deleting a chat session."""
        client, _, _ = auth_client
        
        # Create session
        create_response = await client.post(
            "/api/v1/chat/sessions",
            json=sample_chat_session_data,
        )
        session_id = create_response.json()["data"]["id"]
        
        # Delete session
        response = await client.delete(f"/api/v1/chat/sessions/{session_id}")
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        
        # Verify deletion
        get_response = await client.get(f"/api/v1/chat/sessions/{session_id}")
        assert get_response.status_code == 404


class TestChatMessages:
    """Tests for chat message endpoints."""
    
    @pytest.mark.asyncio
    async def test_add_message(
        self, auth_client, sample_chat_session_data: dict, sample_chat_message_data: dict
    ):
        """Test adding a message to a session."""
        client, _, _ = auth_client
        
        # Create session
        create_response = await client.post(
            "/api/v1/chat/sessions",
            json=sample_chat_session_data,
        )
        session_id = create_response.json()["data"]["id"]
        
        # Add message
        response = await client.post(
            f"/api/v1/chat/sessions/{session_id}/messages",
            json=sample_chat_message_data,
        )
        
        assert response.status_code == 201
        data = response.json()
        
        assert data["success"] is True
        assert data["data"]["role"] == sample_chat_message_data["role"]
        assert data["data"]["content"] == sample_chat_message_data["content"]
        assert data["data"]["session_id"] == session_id
    
    @pytest.mark.asyncio
    async def test_get_messages(
        self, auth_client, sample_chat_session_data: dict, sample_chat_message_data: dict
    ):
        """Test getting messages from a session."""
        client, _, _ = auth_client
        
        # Create session
        create_response = await client.post(
            "/api/v1/chat/sessions",
            json=sample_chat_session_data,
        )
        session_id = create_response.json()["data"]["id"]
        
        # Add messages
        for i in range(3):
            msg = {**sample_chat_message_data, "content": f"Message {i}"}
            await client.post(f"/api/v1/chat/sessions/{session_id}/messages", json=msg)
        
        # Get messages
        response = await client.get(f"/api/v1/chat/sessions/{session_id}/messages")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] is True
        assert len(data["data"]) == 3
    
    @pytest.mark.asyncio
    async def test_get_session_with_messages(
        self, auth_client, sample_chat_session_data: dict, sample_chat_message_data: dict
    ):
        """Test getting session with messages included."""
        client, _, _ = auth_client
        
        # Create session and add message
        create_response = await client.post(
            "/api/v1/chat/sessions",
            json=sample_chat_session_data,
        )
        session_id = create_response.json()["data"]["id"]
        
        await client.post(
            f"/api/v1/chat/sessions/{session_id}/messages",
            json=sample_chat_message_data,
        )
        
        # Get session with messages
        response = await client.get(
            f"/api/v1/chat/sessions/{session_id}",
            params={"include_messages": "true"},
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] is True
        assert "messages" in data["data"]
        assert len(data["data"]["messages"]) >= 1
