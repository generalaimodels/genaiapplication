# ==============================================================================
# TRANSACTION ENDPOINT TESTS
# ==============================================================================
# Tests for transaction endpoints
# ==============================================================================

import pytest
from httpx import AsyncClient


class TestTransactions:
    """Tests for transaction CRUD endpoints."""
    
    @pytest.mark.asyncio
    async def test_create_transaction(self, auth_client, sample_transaction_data: dict):
        """Test creating a new transaction."""
        client, user_id, _ = auth_client
        
        response = await client.post("/api/v1/transactions", json=sample_transaction_data)
        
        assert response.status_code == 201
        data = response.json()
        
        assert data["success"] is True
        assert data["data"]["transaction_type"] == sample_transaction_data["transaction_type"]
        assert float(data["data"]["amount"]) == float(sample_transaction_data["amount"])
        assert data["data"]["currency"] == sample_transaction_data["currency"]
        assert data["data"]["user_id"] == user_id
        assert data["data"]["status"] == "pending"
        assert "reference_id" in data["data"]
    
    @pytest.mark.asyncio
    async def test_create_transaction_unauthenticated(
        self, client: AsyncClient, sample_transaction_data: dict
    ):
        """Test creating transaction without auth fails."""
        response = await client.post("/api/v1/transactions", json=sample_transaction_data)
        assert response.status_code == 401
    
    @pytest.mark.asyncio
    async def test_list_transactions(self, auth_client, sample_transaction_data: dict):
        """Test listing user's transactions."""
        client, _, _ = auth_client
        
        # Create a few transactions
        for i in range(3):
            txn_data = {**sample_transaction_data, "description": f"Transaction {i}"}
            await client.post("/api/v1/transactions", json=txn_data)
        
        # List transactions
        response = await client.get("/api/v1/transactions")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] is True
        assert len(data["data"]) >= 3
    
    @pytest.mark.asyncio
    async def test_list_transactions_with_filter(self, auth_client):
        """Test listing transactions with type filter."""
        client, _, _ = auth_client
        
        # Create credit and debit transactions
        await client.post("/api/v1/transactions", json={
            "transaction_type": "credit",
            "amount": "50.00",
            "currency": "USD",
        })
        await client.post("/api/v1/transactions", json={
            "transaction_type": "debit",
            "amount": "30.00",
            "currency": "USD",
        })
        
        # Filter by type
        response = await client.get(
            "/api/v1/transactions",
            params={"transaction_type": "credit"},
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] is True
        # All returned should be credits
        for txn in data["data"]:
            assert txn["transaction_type"] == "credit"
    
    @pytest.mark.asyncio
    async def test_get_transaction(self, auth_client, sample_transaction_data: dict):
        """Test getting a specific transaction."""
        client, _, _ = auth_client
        
        # Create transaction
        create_response = await client.post(
            "/api/v1/transactions",
            json=sample_transaction_data,
        )
        txn_id = create_response.json()["data"]["id"]
        
        # Get transaction
        response = await client.get(f"/api/v1/transactions/{txn_id}")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] is True
        assert data["data"]["id"] == txn_id
    
    @pytest.mark.asyncio
    async def test_get_nonexistent_transaction(self, auth_client):
        """Test getting nonexistent transaction returns 404."""
        client, _, _ = auth_client
        
        fake_id = "00000000-0000-0000-0000-000000000000"
        response = await client.get(f"/api/v1/transactions/{fake_id}")
        
        assert response.status_code == 404
    
    @pytest.mark.asyncio
    async def test_get_transaction_by_reference(
        self, auth_client, sample_transaction_data: dict
    ):
        """Test getting transaction by reference ID."""
        client, _, _ = auth_client
        
        # Create transaction
        create_response = await client.post(
            "/api/v1/transactions",
            json=sample_transaction_data,
        )
        reference_id = create_response.json()["data"]["reference_id"]
        
        # Get by reference
        response = await client.get(f"/api/v1/transactions/reference/{reference_id}")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] is True
        assert data["data"]["reference_id"] == reference_id
    
    @pytest.mark.asyncio
    async def test_cancel_transaction(self, auth_client, sample_transaction_data: dict):
        """Test cancelling a pending transaction."""
        client, _, _ = auth_client
        
        # Create transaction
        create_response = await client.post(
            "/api/v1/transactions",
            json=sample_transaction_data,
        )
        txn_id = create_response.json()["data"]["id"]
        
        # Cancel transaction
        response = await client.post(f"/api/v1/transactions/{txn_id}/cancel")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] is True
        assert data["data"]["status"] == "cancelled"


class TestTransactionSummary:
    """Tests for transaction summary endpoint."""
    
    @pytest.mark.asyncio
    async def test_get_summary(self, auth_client):
        """Test getting transaction summary."""
        client, _, _ = auth_client
        
        # Get summary (even with no transactions)
        response = await client.get("/api/v1/transactions/summary")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] is True
        assert "total_credits" in data["data"]
        assert "total_debits" in data["data"]
        assert "net_balance" in data["data"]
        assert "transaction_count" in data["data"]
