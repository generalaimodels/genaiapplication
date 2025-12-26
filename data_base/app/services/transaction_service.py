# ==============================================================================
# TRANSACTION SERVICE - Banking/Financial Operations
# ==============================================================================
# Business logic for financial transactions
# ==============================================================================

from __future__ import annotations

from decimal import Decimal
from typing import Any, Dict, List, Optional
from uuid import uuid4

from app.database.adapters.base_adapter import BaseDatabaseAdapter
from app.core.exceptions import (
    NotFoundError,
    InsufficientFundsError,
    ValidationError,
)
from app.schemas.transaction import (
    TransactionCreate,
    TransactionUpdate,
    TransactionResponse,
    TransactionSummary,
)
from app.services.base_service import BaseService


class TransactionService(
    BaseService[Any, TransactionCreate, TransactionUpdate, TransactionResponse]
):
    """
    Transaction service for financial operations.
    
    Provides transaction creation, status management,
    and financial summaries.
    """
    
    def __init__(self, adapter: BaseDatabaseAdapter) -> None:
        """Initialize transaction service."""
        super().__init__(adapter, "transactions")
    
    def _to_response(self, entity: Any) -> TransactionResponse:
        """Convert transaction entity to response."""
        if isinstance(entity, dict):
            return TransactionResponse.model_validate(entity)
        return TransactionResponse.model_validate(entity, from_attributes=True)
    
    def _generate_reference_id(self) -> str:
        """Generate unique transaction reference ID."""
        return f"TXN-{str(uuid4())[:12].upper()}"
    
    # ==========================================================================
    # TRANSACTION OPERATIONS
    # ==========================================================================
    
    async def create_transaction(
        self,
        user_id: str,
        schema: TransactionCreate,
        balance: Optional[Decimal] = None,
    ) -> TransactionResponse:
        """
        Create a new transaction.
        
        Args:
            user_id: User ID for transaction
            schema: Transaction creation data
            balance: Optional balance after transaction
            
        Returns:
            Created transaction response
        """
        data = schema.model_dump(exclude_unset=True)
        data["user_id"] = user_id
        data["reference_id"] = self._generate_reference_id()
        data["status"] = "pending"
        
        if balance is not None:
            data["balance_after"] = balance
        
        result = await self._adapter.create(self._collection_name, data)
        return self._to_response(result)
    
    async def get_user_transactions(
        self,
        user_id: str,
        skip: int = 0,
        limit: int = 50,
        transaction_type: Optional[str] = None,
        status: Optional[str] = None,
    ) -> List[TransactionResponse]:
        """
        Get transactions for a user with optional filters.
        
        Args:
            user_id: User ID
            skip: Records to skip
            limit: Maximum records
            transaction_type: Filter by type
            status: Filter by status
            
        Returns:
            List of transaction responses
        """
        filters: Dict[str, Any] = {"user_id": user_id}
        
        if transaction_type:
            filters["transaction_type"] = transaction_type
        if status:
            filters["status"] = status
        
        results = await self._adapter.get_all(
            self._collection_name,
            skip=skip,
            limit=limit,
            filters=filters,
            sort_by="created_at",
            sort_order="desc",
        )
        return [self._to_response(r) for r in results]
    
    async def get_by_reference(
        self,
        reference_id: str,
    ) -> TransactionResponse:
        """Get transaction by reference ID."""
        result = await self._adapter.find_one(
            self._collection_name,
            {"reference_id": reference_id},
        )
        if not result:
            raise NotFoundError(
                message="Transaction not found",
                resource_type="transaction",
            )
        return self._to_response(result)
    
    async def complete_transaction(
        self,
        transaction_id: str,
        balance_after: Optional[Decimal] = None,
    ) -> TransactionResponse:
        """Mark a transaction as completed."""
        data: Dict[str, Any] = {"status": "completed"}
        if balance_after is not None:
            data["balance_after"] = balance_after
        
        result = await self._adapter.update(
            self._collection_name,
            transaction_id,
            data,
        )
        if not result:
            raise NotFoundError(
                message="Transaction not found",
                resource_type="transaction",
                resource_id=transaction_id,
            )
        return self._to_response(result)
    
    async def fail_transaction(
        self,
        transaction_id: str,
        reason: Optional[str] = None,
    ) -> TransactionResponse:
        """Mark a transaction as failed."""
        data: Dict[str, Any] = {"status": "failed"}
        if reason:
            data["description"] = reason
        
        result = await self._adapter.update(
            self._collection_name,
            transaction_id,
            data,
        )
        if not result:
            raise NotFoundError(
                message="Transaction not found",
                resource_type="transaction",
                resource_id=transaction_id,
            )
        return self._to_response(result)
    
    async def cancel_transaction(
        self,
        transaction_id: str,
        user_id: str,
    ) -> TransactionResponse:
        """Cancel a pending transaction."""
        # Get transaction
        txn = await self._adapter.get_by_id(self._collection_name, transaction_id)
        if not txn:
            raise NotFoundError(
                message="Transaction not found",
                resource_type="transaction",
                resource_id=transaction_id,
            )
        
        # Verify ownership
        owner_id = (
            txn.get("user_id")
            if isinstance(txn, dict)
            else getattr(txn, "user_id")
        )
        if owner_id != user_id:
            raise ValidationError(
                message="Cannot cancel another user's transaction"
            )
        
        # Check if cancellable
        status = (
            txn.get("status")
            if isinstance(txn, dict)
            else getattr(txn, "status")
        )
        if status != "pending":
            raise ValidationError(
                message=f"Cannot cancel transaction with status: {status}"
            )
        
        result = await self._adapter.update(
            self._collection_name,
            transaction_id,
            {"status": "cancelled"},
        )
        return self._to_response(result)
    
    # ==========================================================================
    # ANALYTICS
    # ==========================================================================
    
    async def get_user_summary(
        self,
        user_id: str,
    ) -> TransactionSummary:
        """
        Get transaction summary for a user.
        
        Args:
            user_id: User ID
            
        Returns:
            Transaction summary with totals
        """
        # Get all completed transactions
        transactions = await self._adapter.get_all(
            self._collection_name,
            limit=10000,
            filters={"user_id": user_id, "status": "completed"},
        )
        
        total_credits = Decimal("0.00")
        total_debits = Decimal("0.00")
        
        for txn in transactions:
            txn_type = (
                txn.get("transaction_type")
                if isinstance(txn, dict)
                else getattr(txn, "transaction_type")
            )
            amount = (
                Decimal(str(txn.get("amount", 0)))
                if isinstance(txn, dict)
                else Decimal(str(getattr(txn, "amount", 0)))
            )
            
            if txn_type == "credit":
                total_credits += amount
            elif txn_type == "debit":
                total_debits += amount
        
        return TransactionSummary(
            total_credits=total_credits,
            total_debits=total_debits,
            net_balance=total_credits - total_debits,
            transaction_count=len(transactions),
        )
