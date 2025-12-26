# ==============================================================================
# BASE MODEL - SQLAlchemy Foundation
# ==============================================================================
# Base declarative class and common mixins for all SQL models
# ==============================================================================

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import uuid4

from sqlalchemy import DateTime, func, String
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class SQLBase(DeclarativeBase):
    """
    Base class for all SQLAlchemy models.
    
    Provides a common foundation with:
    - Automatic UUID primary key generation
    - Dictionary serialization method
    - Type annotations for mapped columns
    
    All domain models should inherit from this class.
    
    Example:
        >>> class User(SQLBase):
        ...     __tablename__ = "users"
        ...     email: Mapped[str] = mapped_column(String(255))
    """
    
    # Default primary key for all models
    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=lambda: str(uuid4()),
    )
    
    def to_dict(self) -> dict[str, Any]:
        """
        Convert model instance to dictionary.
        
        Returns:
            Dictionary with all column values
        """
        return {
            column.name: getattr(self, column.name)
            for column in self.__table__.columns
        }
    
    def __repr__(self) -> str:
        """Generate readable representation."""
        class_name = self.__class__.__name__
        return f"<{class_name}(id={self.id})>"


class TimestampMixin:
    """
    Mixin providing automatic timestamp tracking.
    
    Adds created_at and updated_at columns with automatic
    value population and update on modification.
    
    Attributes:
        created_at: Timestamp of record creation (auto-set)
        updated_at: Timestamp of last update (auto-updated)
        
    Example:
        >>> class User(SQLBase, TimestampMixin):
        ...     __tablename__ = "users"
        ...     email: Mapped[str] = mapped_column(String(255))
    """
    
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )


class SoftDeleteMixin:
    """
    Mixin for soft delete functionality.
    
    Instead of physical deletion, records are marked as deleted
    with a timestamp, allowing recovery and audit trails.
    
    Attributes:
        deleted_at: Timestamp when record was deleted (None if active)
        is_deleted: Computed property for deletion status
    """
    
    deleted_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        default=None,
    )
    
    @property
    def is_deleted(self) -> bool:
        """Check if record is soft-deleted."""
        return self.deleted_at is not None
    
    def soft_delete(self) -> None:
        """Mark record as deleted."""
        self.deleted_at = datetime.utcnow()
    
    def restore(self) -> None:
        """Restore a soft-deleted record."""
        self.deleted_at = None
