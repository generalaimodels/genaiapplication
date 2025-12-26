# ==============================================================================
# USER MODEL - Authentication and Authorization
# ==============================================================================
# User entity for authentication, profiles, and access control
# ==============================================================================

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional

from sqlalchemy import Boolean, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.domain_models.base import SQLBase, TimestampMixin

if TYPE_CHECKING:
    from app.domain_models.chat import ChatSession
    from app.domain_models.transaction import Transaction
    from app.domain_models.order import Order
    from app.domain_models.project import Project, Task
    from app.domain_models.course import Enrollment


class User(SQLBase, TimestampMixin):
    """
    User model for authentication and authorization.
    
    Stores user credentials, profile information, and manages
    relationships to user-owned entities.
    
    Attributes:
        email: Unique email address (login identifier)
        hashed_password: Bcrypt-hashed password
        full_name: User's display name
        is_active: Account activation status
        is_superuser: Admin privileges flag
        
    Relationships:
        chat_sessions: User's chat conversations
        transactions: User's financial transactions
        orders: User's e-commerce orders
        projects: Projects owned by user
        assigned_tasks: Tasks assigned to user
        enrollments: Course enrollments
    """
    
    __tablename__ = "users"
    
    # Authentication fields
    email: Mapped[str] = mapped_column(
        String(255),
        unique=True,
        index=True,
        nullable=False,
    )
    hashed_password: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )
    
    # Profile fields
    full_name: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
    )
    bio: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
    )
    avatar_url: Mapped[Optional[str]] = mapped_column(
        String(500),
        nullable=True,
    )
    
    # Status flags
    is_active: Mapped[bool] = mapped_column(
        Boolean,
        default=True,
        nullable=False,
    )
    is_superuser: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
    )
    is_verified: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
    )
    
    # Relationships
    chat_sessions: Mapped[List["ChatSession"]] = relationship(
        "ChatSession",
        back_populates="user",
        cascade="all, delete-orphan",
    )
    transactions: Mapped[List["Transaction"]] = relationship(
        "Transaction",
        back_populates="user",
        cascade="all, delete-orphan",
    )
    orders: Mapped[List["Order"]] = relationship(
        "Order",
        back_populates="user",
        cascade="all, delete-orphan",
    )
    projects: Mapped[List["Project"]] = relationship(
        "Project",
        back_populates="owner",
        foreign_keys="Project.owner_id",
        cascade="all, delete-orphan",
    )
    assigned_tasks: Mapped[List["Task"]] = relationship(
        "Task",
        back_populates="assignee",
        foreign_keys="Task.assignee_id",
    )
    enrollments: Mapped[List["Enrollment"]] = relationship(
        "Enrollment",
        back_populates="user",
        cascade="all, delete-orphan",
    )
    
    def __repr__(self) -> str:
        return f"<User(id={self.id}, email={self.email})>"
