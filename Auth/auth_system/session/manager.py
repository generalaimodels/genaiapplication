# =============================================================================
# SOTA AUTHENTICATION SYSTEM - SESSION MANAGER
# =============================================================================
# File: session/manager.py
# Description: Session lifecycle management coordinating DB and Redis
# =============================================================================

from typing import Optional, List
from datetime import datetime, timezone, timedelta

from sqlalchemy.ext.asyncio import AsyncSession

from session.storage import SessionStorage
from session.models import SessionData, SessionInfo, SessionList
from auth.repository import SessionRepository
from db.adapters.redis_adapter import RedisAdapter
from db.models import User
from core.config import settings
from core.security import generate_session_id


class SessionManager:
    """
    ┌─────────────────────────────────────────────────────────────────────────┐
    │                    SESSION MANAGER                                       │
    │  Coordinates session lifecycle between database and Redis               │
    │  Provides unified interface for all session operations                  │
    └─────────────────────────────────────────────────────────────────────────┘
    
    Session Flow:
        1. Create: Store in DB (persistent) and Redis (cache)
        2. Validate: Check Redis first (fast), fallback to DB
        3. Update: Update activity in both stores
        4. Revoke: Mark inactive in DB, delete from Redis
    """
    
    def __init__(
        self,
        db_session: AsyncSession,
        redis: RedisAdapter,
    ):
        """
        Initialize session manager.
        
        Args:
            db_session: Database session for persistent storage
            redis: Redis client for caching
        """
        self._db_session = db_session
        self._session_repo = SessionRepository(db_session)
        self._storage = SessionStorage(redis)
    
    async def create_session(
        self,
        user: User,
        ip_address: str,
        device_info: Optional[dict] = None,
        remember_me: bool = False,
    ) -> str:
        """
        Create a new session for user.
        
        Args:
            user: User entity
            ip_address: Client IP address
            device_info: Device information
            remember_me: Extend session duration
            
        Returns:
            str: New session ID
        """
        session_id = generate_session_id()
        now = datetime.now(timezone.utc)
        
        # Calculate expiration
        if remember_me:
            ttl = 30 * 24 * 60 * 60  # 30 days
            expires_at = now + timedelta(days=30)
        else:
            ttl = settings.session_ttl
            expires_at = now + timedelta(seconds=ttl)
        
        # Create in database
        import hashlib
        token_hash = hashlib.sha256(session_id.encode()).hexdigest()
        
        await self._session_repo.create(
            user_id=user.id,
            session_id=session_id,
            token_hash=token_hash,
            ip_address=ip_address,
            expires_at=expires_at,
            device_info=device_info,
        )
        
        # Cache in Redis
        session_data = SessionData(
            user_id=user.id,
            email=user.email,
            username=user.username,
            roles=user.roles,
            created_at=now,
            last_activity=now,
            device_info=device_info,
            ip_address=ip_address,
        )
        
        await self._storage.create_session(session_id, session_data, ttl)
        
        return session_id
    
    async def get_session(self, session_id: str) -> Optional[SessionData]:
        """
        Get session data, checking cache first.
        
        Args:
            session_id: Session identifier
            
        Returns:
            SessionData if valid, None otherwise
        """
        # Try cache first
        session_data = await self._storage.get_session(session_id)
        
        if session_data:
            return session_data
        
        # Fallback to database
        session_obj = await self._session_repo.get_by_id(session_id)
        
        if not session_obj or not session_obj.is_active or session_obj.is_expired:
            return None
        
        # Rebuild cache from database
        # This would need the user data, which we'd need to fetch
        # For simplicity, return None and force re-authentication
        return None
    
    async def validate_session(self, session_id: str) -> bool:
        """
        Validate if session is active and not expired.
        
        Args:
            session_id: Session identifier
            
        Returns:
            bool: True if valid
        """
        # Check cache
        if await self._storage.session_exists(session_id):
            return True
        
        # Check database
        session_obj = await self._session_repo.get_by_id(session_id)
        
        return (
            session_obj is not None and
            session_obj.is_active and
            not session_obj.is_expired
        )
    
    async def update_activity(self, session_id: str, user_id: str) -> None:
        """
        Update session last activity.
        
        Args:
            session_id: Session identifier
            user_id: User ID for verification
        """
        # Update cache
        await self._storage.update_activity(session_id)
        
        # Update database
        session_obj = await self._session_repo.get_by_id(session_id)
        if session_obj and session_obj.user_id == user_id:
            await self._session_repo.update_activity(session_obj)
    
    async def revoke_session(
        self,
        session_id: str,
        user_id: str,
    ) -> bool:
        """
        Revoke a specific session.
        
        Args:
            session_id: Session identifier
            user_id: User ID for verification
            
        Returns:
            bool: True if successful
        """
        # Delete from cache
        await self._storage.delete_session(session_id, user_id)
        
        # Mark as inactive in database
        session_obj = await self._session_repo.get_by_id(session_id)
        if session_obj and session_obj.user_id == user_id:
            await self._session_repo.revoke(session_obj)
            return True
        
        return False
    
    async def revoke_all_sessions(
        self,
        user_id: str,
        except_current: Optional[str] = None,
    ) -> int:
        """
        Revoke all sessions for a user.
        
        Args:
            user_id: User ID
            except_current: Session ID to keep active
            
        Returns:
            int: Number of sessions revoked
        """
        # Get all session IDs from cache
        session_ids = await self._storage.get_user_session_ids(user_id)
        
        count = 0
        for session_id in session_ids:
            if except_current and session_id == except_current:
                continue
            await self._storage.delete_session(session_id, user_id)
            count += 1
        
        # Also revoke in database
        await self._session_repo.revoke_all_for_user(user_id)
        
        # If keeping current, re-activate it
        if except_current:
            session_obj = await self._session_repo.get_by_id(except_current)
            if session_obj:
                session_obj.is_active = True
                await self._db_session.flush()
        
        return count
    
    async def get_user_sessions(
        self,
        user_id: str,
        current_session_id: Optional[str] = None,
    ) -> SessionList:
        """
        Get all active sessions for a user.
        
        Args:
            user_id: User ID
            current_session_id: Current session to mark
            
        Returns:
            SessionList: List of active sessions
        """
        # Get from database (source of truth for session list)
        sessions = await self._session_repo.get_active_by_user(user_id)
        
        session_infos = [
            SessionInfo(
                session_id=s.session_id,
                user_id=s.user_id,
                created_at=s.created_at,
                last_activity=s.last_activity,
                expires_at=s.expires_at,
                device_info=s.device_info,
                ip_address=s.ip_address,
                is_current=s.session_id == current_session_id,
            )
            for s in sessions
        ]
        
        return SessionList(
            sessions=session_infos,
            total=len(session_infos),
        )
    
    async def blacklist_token(
        self,
        token_jti: str,
        expires_in: int,
    ) -> None:
        """
        Add token to blacklist.
        
        Args:
            token_jti: Token unique identifier
            expires_in: Seconds until token expires
        """
        await self._storage.blacklist_token(token_jti, expires_in)
    
    async def is_token_blacklisted(self, token_jti: str) -> bool:
        """
        Check if token is blacklisted.
        
        Args:
            token_jti: Token unique identifier
            
        Returns:
            bool: True if blacklisted
        """
        return await self._storage.is_token_blacklisted(token_jti)
    
    async def cleanup_expired(self) -> int:
        """
        Clean up expired sessions from database.
        
        Returns:
            int: Number of sessions cleaned up
        """
        return await self._session_repo.delete_expired()
