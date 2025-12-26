# ==============================================================================
# TRANSACTION SCHEMAS - Banking/Financial Operations
# ==============================================================================
# Request/Response schemas for financial transactions
# ==============================================================================

from __future__ import annotations

from decimal import Decimal
from typing import Optional

from pydantic import Field, field_validator

from app.schemas.base import BaseSchema, TimestampSchema


class TransactionCreate(BaseSchema):
    """Schema for creating a transaction."""
    
    transaction_type: str = Field(
        ...,
        pattern="^(credit|debit|transfer)$",
        description="Transaction type (credit/debit/transfer)",
    )
    amount: Decimal = Field(
        ...,
        gt=0,
        description="Transaction amount",
    )
    currency: str = Field(
        "USD",
        pattern="^[A-Z]{3}$",
        description="Currency code (ISO 4217)",
    )
    description: Optional[str] = Field(
        None,
        max_length=500,
        description="Transaction description",
    )
    recipient_id: Optional[str] = Field(
        None,
        description="Recipient user ID (for transfers)",
    )
    
    @field_validator("amount")
    @classmethod
    def validate_amount(cls, v: Decimal) -> Decimal:
        """Ensure amount is positive and reasonable."""
        if v <= 0:
            raise ValueError("Amount must be positive")
        if v > Decimal("1000000000"):
            raise ValueError("Amount exceeds maximum limit")
        return v


class TransactionUpdate(BaseSchema):
    """Schema for updating a transaction (limited fields)."""
    
    status: Optional[str] = Field(
        None,
        pattern="^(pending|completed|failed|cancelled)$",
        description="Transaction status",
    )
    description: Optional[str] = Field(
        None,
        max_length=500,
        description="Transaction description",
    )


class TransactionResponse(TimestampSchema):
    """Schema for transaction response."""
    
    id: str = Field(
        ...,
        description="Transaction unique identifier",
    )
    user_id: str = Field(
        ...,
        description="Transaction owner ID",
    )
    transaction_type: str = Field(
        ...,
        description="Transaction type",
    )
    amount: Decimal = Field(
        ...,
        description="Transaction amount",
    )
    currency: str = Field(
        ...,
        description="Currency code",
    )
    status: str = Field(
        ...,
        description="Transaction status",
    )
    description: Optional[str] = Field(
        None,
        description="Transaction description",
    )
    reference_id: str = Field(
        ...,
        description="External reference ID",
    )
    recipient_id: Optional[str] = Field(
        None,
        description="Recipient user ID",
    )
    balance_after: Optional[Decimal] = Field(
        None,
        description="Account balance after transaction",
    )


class TransactionSummary(BaseSchema):
    """Schema for transaction summary/statistics."""
    
    total_credits: Decimal = Field(
        ...,
        description="Total credit amount",
    )
    total_debits: Decimal = Field(
        ...,
        description="Total debit amount",
    )
    net_balance: Decimal = Field(
        ...,
        description="Net balance (credits - debits)",
    )
    transaction_count: int = Field(
        ...,
        description="Total number of transactions",
    )
    period_start: Optional[str] = Field(
        None,
        description="Summary period start date",
    )
    period_end: Optional[str] = Field(
        None,
        description="Summary period end date",
    )
