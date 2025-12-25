# =============================================================================
# SOTA AUTHENTICATION SYSTEM - AUTH REPOSITORY
# =============================================================================
# File: auth/repository.py
# Description: Data access layer for user operations
#              Implements repository pattern with SQLAlchemy async
# =============================================================================

from typing import Optional, List
from datetime import datetime, timezone, timedelta

from sqlalchemy import select, update, delete, func
from sqlalchemy.ext.asyncio import AsyncSession

from db.models import User, Session, RefreshToken, AuditLog, AuditAction, AuditStatus
from core.config import settings


class UserRepository:
    """
    ┌─────────────────────────────────────────────────────────────────────────┐
    │                    USER REPOSITORY                                       │
    │  Data access layer for User entity operations                           │
    │  Provides clean separation between business logic and data access       │
    └─────────────────────────────────────────────────────────────────────────┘
    
    All methods are async and work with SQLAlchemy AsyncSession.
    The repository does not handle transactions - that's the caller's
    responsibility.
    """
    
    def __init__(self, session: AsyncSession):
        """
        Initialize repository with database session.
        
        Args:
            session: SQLAlchemy async session
        """
        self._session = session
    
    # =========================================================================
    # CREATE OPERATIONS
    # =========================================================================
    
    async def create(
        self,
        email: str,
        username: str,
        password_hash: str,
        is_verified: bool = False,
        is_superuser: bool = False,
    ) -> User:
        """
        Create a new user.
        
        Args:
            email: User's email address
            username: User's username
            password_hash: Hashed password
            is_verified: Email verification status
            is_superuser: Admin flag
            
        Returns:
            User: Created user entity
        """
        user = User(
            email=email.lower(),
            username=username.lower(),
            password_hash=password_hash,
            is_verified=is_verified,
            is_superuser=is_superuser,
        )
        
        self._session.add(user)
        await self._session.flush()
        await self._session.refresh(user)
        
        return user
    
    # =========================================================================
    # READ OPERATIONS
    # =========================================================================
    
    async def get_by_id(self, user_id: str) -> Optional[User]:
        """
        Get user by ID.
        
        Args:
            user_id: User UUID
            
        Returns:
            User if found, None otherwise
        """
        result = await self._session.execute(
            select(User).where(User.id == user_id)
        )
        return result.scalar_one_or_none()
    
    async def get_by_email(self, email: str) -> Optional[User]:
        """
        Get user by email address.
        
        Args:
            email: User's email
            
        Returns:
            User if found, None otherwise
        """
        result = await self._session.execute(
            select(User).where(User.email == email.lower())
        )
        return result.scalar_one_or_none()
    
    async def get_by_username(self, username: str) -> Optional[User]:
        """
        Get user by username.
        
        Args:
            username: User's username
            
        Returns:
            User if found, None otherwise
        """
        result = await self._session.execute(
            select(User).where(User.username == username.lower())
        )
        return result.scalar_one_or_none()
    
    async def get_by_email_or_username(
        self,
        identifier: str
    ) -> Optional[User]:
        """
        Get user by email or username.
        
        Args:
            identifier: Email or username
            
        Returns:
            User if found, None otherwise
        """
        identifier_lower = identifier.lower()
        result = await self._session.execute(
            select(User).where(
                (User.email == identifier_lower) |
                (User.username == identifier_lower)
            )
        )
        return result.scalar_one_or_none()
    
    async def exists_email(self, email: str) -> bool:
        """
        Check if email already exists.
        
        Args:
            email: Email to check
            
        Returns:
            True if email exists
        """
        result = await self._session.execute(
            select(func.count()).select_from(User).where(
                User.email == email.lower()
            )
        )
        return result.scalar() > 0
    
    async def exists_username(self, username: str) -> bool:
        """
        Check if username already exists.
        
        Args:
            username: Username to check
            
        Returns:
            True if username exists
        """
        result = await self._session.execute(
            select(func.count()).select_from(User).where(
                User.username == username.lower()
            )
        )
        return result.scalar() > 0
    
    async def list_users(
        self,
        skip: int = 0,
        limit: int = 50,
        is_active: Optional[bool] = None,
    ) -> List[User]:
        """
        List users with pagination.
        
        Args:
            skip: Number of records to skip
            limit: Maximum records to return
            is_active: Optional filter by active status
            
        Returns:
            List of users
        """
        query = select(User)
        
        if is_active is not None:
            query = query.where(User.is_active == is_active)
        
        query = query.offset(skip).limit(limit).order_by(User.created_at.desc())
        
        result = await self._session.execute(query)
        return list(result.scalars().all())
    
    async def count_users(self, is_active: Optional[bool] = None) -> int:
        """
        Count total users.
        
        Args:
            is_active: Optional filter by active status
            
        Returns:
            Total count
        """
        query = select(func.count()).select_from(User)
        
        if is_active is not None:
            query = query.where(User.is_active == is_active)
        
        result = await self._session.execute(query)
        return result.scalar() or 0
    
    # =========================================================================
    # UPDATE OPERATIONS
    # =========================================================================
    
    async def update(
        self,
        user: User,
        **kwargs
    ) -> User:
        """
        Update user fields.
        
        Args:
            user: User entity to update
            **kwargs: Fields to update
            
        Returns:
            Updated user
        """
        for key, value in kwargs.items():
            if hasattr(user, key):
                setattr(user, key, value)
        
        await self._session.flush()
        await self._session.refresh(user)
        
        return user
    
    async def update_password(
        self,
        user: User,
        password_hash: str
    ) -> User:
        """
        Update user's password hash.
        
        Args:
            user: User entity
            password_hash: New password hash
            
        Returns:
            Updated user
        """
        user.password_hash = password_hash
        user.failed_attempts = 0  # Reset on password change
        user.locked_until = None
        
        await self._session.flush()
        
        return user
    
    async def update_last_login(self, user: User) -> User:
        """
        Update last login timestamp.
        
        Args:
            user: User entity
            
        Returns:
            Updated user
        """
        user.last_login = datetime.now(timezone.utc)
        user.failed_attempts = 0
        
        await self._session.flush()
        
        return user
    
    async def increment_failed_attempts(self, user: User) -> User:
        """
        Increment failed login attempts.
        
        If max attempts exceeded, locks the account.
        
        Args:
            user: User entity
            
        Returns:
            Updated user
        """
        user.failed_attempts += 1
        
        # Lock account if max attempts exceeded
        if user.failed_attempts >= settings.max_login_attempts:
            user.locked_until = (
                datetime.now(timezone.utc) +
                timedelta(minutes=settings.lockout_duration_minutes)
            )
        
        await self._session.flush()
        
        return user
    
    async def reset_failed_attempts(self, user: User) -> User:
        """
        Reset failed login attempts counter.
        
        Args:
            user: User entity
            
        Returns:
            Updated user
        """
        user.failed_attempts = 0
        user.locked_until = None
        
        await self._session.flush()
        
        return user
    
    async def lock_account(
        self,
        user: User,
        duration_minutes: Optional[int] = None
    ) -> User:
        """
        Lock user account.
        
        Args:
            user: User entity
            duration_minutes: Lock duration (None for permanent)
            
        Returns:
            Updated user
        """
        if duration_minutes:
            user.locked_until = (
                datetime.now(timezone.utc) +
                timedelta(minutes=duration_minutes)
            )
        else:
            user.is_active = False
        
        await self._session.flush()
        
        return user
    
    async def unlock_account(self, user: User) -> User:
        """
        Unlock user account.
        
        Args:
            user: User entity
            
        Returns:
            Updated user
        """
        user.locked_until = None
        user.failed_attempts = 0
        user.is_active = True
        
        await self._session.flush()
        
        return user
    
    async def verify_email(self, user: User) -> User:
        """
        Mark user email as verified.
        
        Args:
            user: User entity
            
        Returns:
            Updated user
        """
        user.is_verified = True
        
        await self._session.flush()
        
        return user
    
    # =========================================================================
    # DELETE OPERATIONS
    # =========================================================================
    
    async def delete(self, user: User) -> None:
        """
        Delete user (cascade deletes sessions, tokens, logs).
        
        Args:
            user: User entity to delete
        """
        await self._session.delete(user)
        await self._session.flush()
    
    async def soft_delete(self, user: User) -> User:
        """
        Soft delete user by deactivating account.
        
        Args:
            user: User entity
            
        Returns:
            Updated user
        """
        user.is_active = False
        
        await self._session.flush()
        
        return user


class SessionRepository:
    """
    ┌─────────────────────────────────────────────────────────────────────────┐
    │                    SESSION REPOSITORY                                    │
    │  Data access layer for Session entity operations                        │
    └─────────────────────────────────────────────────────────────────────────┘
    """
    
    def __init__(self, session: AsyncSession):
        """Initialize repository with database session."""
        self._session = session
    
    async def create(
        self,
        user_id: str,
        session_id: str,
        token_hash: str,
        ip_address: str,
        expires_at: datetime,
        device_info: Optional[dict] = None,
    ) -> Session:
        """Create a new session."""
        session_obj = Session(
            session_id=session_id,
            user_id=user_id,
            token_hash=token_hash,
            ip_address=ip_address,
            expires_at=expires_at,
            device_info=device_info or {},
        )
        
        self._session.add(session_obj)
        await self._session.flush()
        await self._session.refresh(session_obj)
        
        return session_obj
    
    async def get_by_id(self, session_id: str) -> Optional[Session]:
        """Get session by ID."""
        result = await self._session.execute(
            select(Session).where(Session.session_id == session_id)
        )
        return result.scalar_one_or_none()
    
    async def get_active_by_user(self, user_id: str) -> List[Session]:
        """Get all active sessions for a user."""
        result = await self._session.execute(
            select(Session).where(
                (Session.user_id == user_id) &
                (Session.is_active == True) &
                (Session.expires_at > datetime.now(timezone.utc))
            ).order_by(Session.created_at.desc())
        )
        return list(result.scalars().all())
    
    async def update_activity(self, session_obj: Session) -> Session:
        """Update session last activity."""
        session_obj.last_activity = datetime.now(timezone.utc)
        await self._session.flush()
        return session_obj
    
    async def revoke(self, session_obj: Session) -> Session:
        """Revoke a session."""
        session_obj.is_active = False
        await self._session.flush()
        return session_obj
    
    async def revoke_all_for_user(self, user_id: str) -> int:
        """Revoke all sessions for a user."""
        result = await self._session.execute(
            update(Session)
            .where(Session.user_id == user_id)
            .values(is_active=False)
        )
        await self._session.flush()
        return result.rowcount
    
    async def delete_expired(self) -> int:
        """Delete expired sessions."""
        result = await self._session.execute(
            delete(Session).where(
                Session.expires_at < datetime.now(timezone.utc)
            )
        )
        await self._session.flush()
        return result.rowcount


class RefreshTokenRepository:
    """Repository for refresh token operations."""
    
    def __init__(self, session: AsyncSession):
        """Initialize repository with database session."""
        self._session = session
    
    async def create(
        self,
        user_id: str,
        session_id: str,
        token_hash: str,
        expires_at: datetime,
    ) -> RefreshToken:
        """Create a new refresh token."""
        token = RefreshToken(
            user_id=user_id,
            session_id=session_id,
            token_hash=token_hash,
            expires_at=expires_at,
        )
        
        self._session.add(token)
        await self._session.flush()
        await self._session.refresh(token)
        
        return token
    
    async def get_by_hash(self, token_hash: str) -> Optional[RefreshToken]:
        """Get refresh token by hash."""
        result = await self._session.execute(
            select(RefreshToken).where(
                RefreshToken.token_hash == token_hash
            )
        )
        return result.scalar_one_or_none()
    
    async def revoke(self, token: RefreshToken) -> RefreshToken:
        """Revoke a refresh token."""
        token.revoked = True
        token.revoked_at = datetime.now(timezone.utc)
        await self._session.flush()
        return token
    
    async def revoke_all_for_user(self, user_id: str) -> int:
        """Revoke all refresh tokens for a user."""
        result = await self._session.execute(
            update(RefreshToken)
            .where(RefreshToken.user_id == user_id)
            .values(revoked=True, revoked_at=datetime.now(timezone.utc))
        )
        await self._session.flush()
        return result.rowcount


class AuditLogRepository:
    """Repository for audit log operations."""
    
    def __init__(self, session: AsyncSession):
        """Initialize repository with database session."""
        self._session = session
    
    async def create(
        self,
        action: str,
        status: str,
        ip_address: str,
        user_id: Optional[str] = None,
        user_agent: Optional[str] = None,
        details: Optional[dict] = None,
    ) -> AuditLog:
        """Create a new audit log entry."""
        log = AuditLog(
            user_id=user_id,
            action=action,
            status=status,
            ip_address=ip_address,
            user_agent=user_agent,
            details=details or {},
        )
        
        self._session.add(log)
        await self._session.flush()
        
        return log
    
    async def get_by_user(
        self,
        user_id: str,
        skip: int = 0,
        limit: int = 50,
    ) -> List[AuditLog]:
        """Get audit logs for a user."""
        result = await self._session.execute(
            select(AuditLog)
            .where(AuditLog.user_id == user_id)
            .order_by(AuditLog.created_at.desc())
            .offset(skip)
            .limit(limit)
        )
        return list(result.scalars().all())
