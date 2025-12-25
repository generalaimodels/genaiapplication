# =============================================================================
# SOTA AUTHENTICATION SYSTEM - DATABASE MODELS
# =============================================================================
# File: db/models.py
# Description: SQLAlchemy ORM models for Users, Sessions, Tokens, and Audit Logs
#              Full-featured schema with relationships, indexes, and constraints
# =============================================================================

from typing import Optional, List
from datetime import datetime, timezone
from uuid import uuid4
import json

from sqlalchemy import (
    Column,
    String,
    Boolean,
    Integer,
    DateTime,
    Text,
    ForeignKey,
    Index,
    JSON,
)
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.orm import relationship, Mapped, mapped_column
from sqlalchemy.sql import func

from db.base import Base


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def generate_uuid() -> str:
    """Generate a new UUID string."""
    return str(uuid4())


def utc_now() -> datetime:
    """Get current UTC timestamp."""
    return datetime.now(timezone.utc)


# =============================================================================
# USER MODEL
# =============================================================================

class User(Base):
    """
    ┌─────────────────────────────────────────────────────────────────────────┐
    │                    USER MODEL                                            │
    │  Core user entity with authentication and account management fields    │
    └─────────────────────────────────────────────────────────────────────────┘
    
    Fields:
        - id:             UUID primary key (auto-generated)
        - email:          Unique email address (indexed)
        - username:       Unique username (indexed)
        - password_hash:  Argon2id/Bcrypt hashed password
        - is_active:      Account activation status
        - is_verified:    Email verification status
        - is_superuser:   Admin privileges flag
        - created_at:     Account creation timestamp
        - updated_at:     Last update timestamp
        - last_login:     Last successful login
        - failed_attempts: Consecutive failed login attempts
        - locked_until:   Account lockout expiration
    
    Relationships:
        - sessions:       One-to-many with Session
        - refresh_tokens: One-to-many with RefreshToken
        - audit_logs:     One-to-many with AuditLog
    """
    
    __tablename__ = "users"
    
    # Primary Key
    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=generate_uuid,
        index=True
    )
    
    # Authentication Fields
    email: Mapped[str] = mapped_column(
        String(255),
        unique=True,
        nullable=False,
        index=True
    )
    username: Mapped[str] = mapped_column(
        String(100),
        unique=True,
        nullable=False,
        index=True
    )
    password_hash: Mapped[str] = mapped_column(
        String(255),
        nullable=False
    )
    
    # Account Status
    is_active: Mapped[bool] = mapped_column(
        Boolean,
        default=True,
        nullable=False
    )
    is_verified: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False
    )
    is_superuser: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False
    )
    
    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        onupdate=utc_now,
        nullable=False
    )
    last_login: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True
    )
    
    # Security: Brute Force Protection
    failed_attempts: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False
    )
    locked_until: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True
    )
    
    # Relationships
    sessions: Mapped[List["Session"]] = relationship(
        "Session",
        back_populates="user",
        cascade="all, delete-orphan",
        lazy="dynamic"
    )
    refresh_tokens: Mapped[List["RefreshToken"]] = relationship(
        "RefreshToken",
        back_populates="user",
        cascade="all, delete-orphan",
        lazy="dynamic"
    )
    audit_logs: Mapped[List["AuditLog"]] = relationship(
        "AuditLog",
        back_populates="user",
        cascade="all, delete-orphan",
        lazy="dynamic"
    )
    
    # Composite Index for common queries
    __table_args__ = (
        Index("ix_users_email_active", "email", "is_active"),
    )
    
    def __repr__(self) -> str:
        return f"<User(id={self.id}, email={self.email}, username={self.username})>"
    
    @property
    def is_locked(self) -> bool:
        """Check if account is currently locked."""
        if self.locked_until is None:
            return False
        return datetime.now(timezone.utc) < self.locked_until
    
    @property
    def roles(self) -> List[str]:
        """Get user roles list."""
        roles = ["user"]
        if self.is_superuser:
            roles.append("admin")
        return roles


# =============================================================================
# SESSION MODEL
# =============================================================================

class Session(Base):
    """
    ┌─────────────────────────────────────────────────────────────────────────┐
    │                    SESSION MODEL                                         │
    │  Tracks active user sessions across devices                             │
    └─────────────────────────────────────────────────────────────────────────┘
    
    Fields:
        - session_id:    UUID primary key (indexed for fast lookup)
        - user_id:       Foreign key to users table
        - token_hash:    Hash of the session token
        - device_info:   JSON with user agent, device type, etc.
        - ip_address:    Client IP (IPv4 or IPv6)
        - created_at:    Session creation time
        - expires_at:    Session expiration (indexed for cleanup)
        - last_activity: Last request timestamp
        - is_active:     Session validity flag
    """
    
    __tablename__ = "sessions"
    
    # Primary Key
    session_id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=generate_uuid,
        index=True
    )
    
    # Foreign Key
    user_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    
    # Session Data
    token_hash: Mapped[str] = mapped_column(
        String(255),
        nullable=False
    )
    device_info: Mapped[Optional[dict]] = mapped_column(
        JSON,
        nullable=True,
        default=dict
    )
    ip_address: Mapped[str] = mapped_column(
        String(45),  # IPv6 max length
        nullable=False
    )
    
    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        nullable=False
    )
    expires_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        index=True
    )
    last_activity: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        nullable=False
    )
    
    # Status
    is_active: Mapped[bool] = mapped_column(
        Boolean,
        default=True,
        nullable=False
    )
    
    # Relationships
    user: Mapped["User"] = relationship(
        "User",
        back_populates="sessions"
    )
    refresh_tokens: Mapped[List["RefreshToken"]] = relationship(
        "RefreshToken",
        back_populates="session",
        cascade="all, delete-orphan",
        lazy="dynamic"
    )
    
    # Indexes
    __table_args__ = (
        Index("ix_sessions_user_active", "user_id", "is_active"),
        Index("ix_sessions_expires", "expires_at"),
    )
    
    def __repr__(self) -> str:
        return f"<Session(id={self.session_id}, user_id={self.user_id})>"
    
    @property
    def is_expired(self) -> bool:
        """Check if session has expired."""
        return datetime.now(timezone.utc) > self.expires_at


# =============================================================================
# REFRESH TOKEN MODEL
# =============================================================================

class RefreshToken(Base):
    """
    ┌─────────────────────────────────────────────────────────────────────────┐
    │                    REFRESH TOKEN MODEL                                   │
    │  Manages refresh tokens with rotation and revocation support            │
    └─────────────────────────────────────────────────────────────────────────┘
    
    Fields:
        - id:           UUID primary key
        - user_id:      Foreign key to users
        - session_id:   Foreign key to sessions
        - token_hash:   SHA-256 hash of token (indexed for lookup)
        - expires_at:   Token expiration (indexed for cleanup)
        - revoked:      Revocation flag
        - revoked_at:   Revocation timestamp
    """
    
    __tablename__ = "refresh_tokens"
    
    # Primary Key
    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=generate_uuid
    )
    
    # Foreign Keys
    user_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    session_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("sessions.session_id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    
    # Token Data
    token_hash: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        index=True
    )
    
    # Timestamps
    expires_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        index=True
    )
    
    # Revocation
    revoked: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False
    )
    revoked_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True
    )
    
    # Relationships
    user: Mapped["User"] = relationship(
        "User",
        back_populates="refresh_tokens"
    )
    session: Mapped["Session"] = relationship(
        "Session",
        back_populates="refresh_tokens"
    )
    
    def __repr__(self) -> str:
        return f"<RefreshToken(id={self.id}, user_id={self.user_id})>"
    
    @property
    def is_valid(self) -> bool:
        """Check if token is valid (not expired and not revoked)."""
        if self.revoked:
            return False
        return datetime.now(timezone.utc) < self.expires_at


# =============================================================================
# AUDIT LOG MODEL
# =============================================================================

class AuditLog(Base):
    """
    ┌─────────────────────────────────────────────────────────────────────────┐
    │                    AUDIT LOG MODEL                                       │
    │  Security audit trail for authentication events                         │
    └─────────────────────────────────────────────────────────────────────────┘
    
    Actions Tracked:
        - LOGIN_SUCCESS, LOGIN_FAILED
        - LOGOUT
        - REGISTER
        - PASSWORD_CHANGE, PASSWORD_RESET
        - EMAIL_VERIFY
        - ACCOUNT_LOCK, ACCOUNT_UNLOCK
        - TOKEN_REFRESH
        - SESSION_REVOKE
    """
    
    __tablename__ = "audit_logs"
    
    # Primary Key
    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=generate_uuid
    )
    
    # Foreign Key (nullable for failed login attempts)
    user_id: Mapped[Optional[str]] = mapped_column(
        String(36),
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
        index=True
    )
    
    # Event Data
    action: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        index=True
    )
    status: Mapped[str] = mapped_column(
        String(20),
        nullable=False,  # SUCCESS, FAILED
        index=True
    )
    
    # Context
    ip_address: Mapped[str] = mapped_column(
        String(45),
        nullable=False
    )
    user_agent: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True
    )
    details: Mapped[Optional[dict]] = mapped_column(
        JSON,
        nullable=True,
        default=dict
    )
    
    # Timestamp
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        nullable=False,
        index=True
    )
    
    # Relationships
    user: Mapped[Optional["User"]] = relationship(
        "User",
        back_populates="audit_logs"
    )
    
    # Indexes for common queries
    __table_args__ = (
        Index("ix_audit_action_status", "action", "status"),
        Index("ix_audit_created", "created_at"),
    )
    
    def __repr__(self) -> str:
        return f"<AuditLog(id={self.id}, action={self.action}, status={self.status})>"


# =============================================================================
# AUDIT LOG ACTION TYPES
# =============================================================================

class AuditAction:
    """Constants for audit log action types."""
    
    # Authentication
    LOGIN_SUCCESS = "LOGIN_SUCCESS"
    LOGIN_FAILED = "LOGIN_FAILED"
    LOGOUT = "LOGOUT"
    LOGOUT_ALL = "LOGOUT_ALL"
    
    # Registration
    REGISTER = "REGISTER"
    EMAIL_VERIFY = "EMAIL_VERIFY"
    
    # Password
    PASSWORD_CHANGE = "PASSWORD_CHANGE"
    PASSWORD_RESET_REQUEST = "PASSWORD_RESET_REQUEST"
    PASSWORD_RESET_COMPLETE = "PASSWORD_RESET_COMPLETE"
    
    # Account
    ACCOUNT_LOCK = "ACCOUNT_LOCK"
    ACCOUNT_UNLOCK = "ACCOUNT_UNLOCK"
    ACCOUNT_DELETE = "ACCOUNT_DELETE"
    
    # Session
    SESSION_REVOKE = "SESSION_REVOKE"
    TOKEN_REFRESH = "TOKEN_REFRESH"
    
    # Admin
    ADMIN_USER_UPDATE = "ADMIN_USER_UPDATE"
    ADMIN_USER_DELETE = "ADMIN_USER_DELETE"


class AuditStatus:
    """Constants for audit log status types."""
    
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"
