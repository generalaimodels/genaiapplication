# ==============================================================================
# TRANSACTION MODEL - Banking/Financial Operations
# ==============================================================================
# Financial transaction tracking for banking use case
# ==============================================================================

from __future__ import annotations

from decimal import Decimal
from typing import TYPE_CHECKING, Optional

from sqlalchemy import ForeignKey, String, Text, Numeric, Enum as SQLEnum
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.domain_models.base import SQLBase, TimestampMixin

if TYPE_CHECKING:
    from app.domain_models.user import User

import enum


class TransactionType(str, enum.Enum):
    """Types of financial transactions."""
    CREDIT = "credit"
    DEBIT = "debit"
    TRANSFER = "transfer"


class TransactionStatus(str, enum.Enum):
    """Status states for transactions."""
    PENDING = "pending"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class Transaction(SQLBase, TimestampMixin):
    """
    Financial transaction model for banking operations.
    
    Tracks all financial movements including deposits, withdrawals,
    and transfers between accounts.
    
    Attributes:
        user_id: Account holder
        transaction_type: Type (credit/debit/transfer)
        amount: Transaction value
        currency: Currency code (ISO 4217)
        status: Current transaction status
        description: Transaction description
        reference_id: External reference identifier
        
    Relationships:
        user: Transaction owner
    """
    
    __tablename__ = "transactions"
    
    # Foreign keys
    user_id: Mapped[str] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"),
        index=True,
        nullable=False,
    )
    
    # Transaction details
    transaction_type: Mapped[TransactionType] = mapped_column(
        SQLEnum(TransactionType),
        nullable=False,
    )
    amount: Mapped[Decimal] = mapped_column(
        Numeric(precision=18, scale=2),
        nullable=False,
    )
    currency: Mapped[str] = mapped_column(
        String(3),
        default="USD",
        nullable=False,
    )
    
    # Status tracking
    status: Mapped[TransactionStatus] = mapped_column(
        SQLEnum(TransactionStatus),
        default=TransactionStatus.PENDING,
        nullable=False,
    )
    
    # Metadata
    description: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
    )
    reference_id: Mapped[str] = mapped_column(
        String(50),
        unique=True,
        index=True,
        nullable=False,
    )
    
    # For transfers
    recipient_id: Mapped[Optional[str]] = mapped_column(
        String(36),
        nullable=True,
    )
    
    # Balance after transaction
    balance_after: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(precision=18, scale=2),
        nullable=True,
    )
    
    # Relationships
    user: Mapped["User"] = relationship(
        "User",
        back_populates="transactions",
    )
    
    def __repr__(self) -> str:
        return f"<Transaction(id={self.id}, type={self.transaction_type}, amount={self.amount})>"
