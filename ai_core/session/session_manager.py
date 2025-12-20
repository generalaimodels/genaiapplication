# =============================================================================
# SESSION MANAGER - Session Lifecycle Management
# =============================================================================
# Manages user sessions with automatic expiration, persistence, and cleanup.
# =============================================================================

from __future__ import annotations
import uuid
import time
import threading
from typing import Dict, Optional, Any, List
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


@dataclass
class Session:
    """
    Represents a user session.
    
    Attributes:
        session_id: Unique session identifier
        user_id: Optional user identifier
        created_at: Session creation timestamp
        last_accessed: Last access timestamp
        expires_at: Session expiration timestamp
        metadata: Additional session data
        is_active: Whether session is active
    """
    session_id: str
    user_id: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    last_accessed: datetime = field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    is_active: bool = True
    
    # Session state
    model: Optional[str] = None
    system_prompt: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 4096
    
    @property
    def is_expired(self) -> bool:
        """Check if session is expired."""
        if self.expires_at is None:
            return False
        return datetime.now() > self.expires_at
    
    @property
    def age_seconds(self) -> float:
        """Get session age in seconds."""
        return (datetime.now() - self.created_at).total_seconds()
    
    def touch(self) -> None:
        """Update last accessed timestamp."""
        self.last_accessed = datetime.now()
    
    def extend(self, seconds: int) -> None:
        """Extend session expiration."""
        if self.expires_at:
            self.expires_at = self.expires_at + timedelta(seconds=seconds)
        else:
            self.expires_at = datetime.now() + timedelta(seconds=seconds)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert session to dictionary."""
        return {
            "session_id": self.session_id,
            "user_id": self.user_id,
            "created_at": self.created_at.isoformat(),
            "last_accessed": self.last_accessed.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "is_active": self.is_active,
            "model": self.model,
            "system_prompt": self.system_prompt,
            "metadata": self.metadata,
        }


class SessionManager:
    """
    Manages session lifecycle with automatic cleanup.
    
    Features:
        - Create, retrieve, update, delete sessions
        - Automatic session expiration
        - Thread-safe operations
        - Session persistence hooks
    
    Example:
        >>> manager = SessionManager(default_ttl=3600)
        >>> session = manager.create_session(user_id="user-123")
        >>> 
        >>> # Later
        >>> session = manager.get_session(session.session_id)
        >>> if session:
        ...     session.touch()
    """
    
    def __init__(
        self,
        default_ttl: int = 3600,
        cleanup_interval: int = 300,
        auto_cleanup: bool = True,
        max_sessions: int = 10000
    ):
        """
        Initialize session manager.
        
        Args:
            default_ttl: Default session TTL in seconds
            cleanup_interval: Cleanup check interval in seconds
            auto_cleanup: Enable automatic cleanup thread
            max_sessions: Maximum concurrent sessions
        """
        self.default_ttl = default_ttl
        self.cleanup_interval = cleanup_interval
        self.max_sessions = max_sessions
        
        self._sessions: Dict[str, Session] = {}
        self._lock = threading.RLock()
        self._cleanup_thread: Optional[threading.Thread] = None
        self._shutdown = threading.Event()
        
        if auto_cleanup:
            self._start_cleanup_thread()
    
    def _start_cleanup_thread(self) -> None:
        """Start background cleanup thread."""
        self._cleanup_thread = threading.Thread(
            target=self._cleanup_loop,
            daemon=True,
            name="SessionCleanup"
        )
        self._cleanup_thread.start()
    
    def _cleanup_loop(self) -> None:
        """Background cleanup loop."""
        while not self._shutdown.wait(self.cleanup_interval):
            self.cleanup_expired()
    
    def create_session(
        self,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        ttl: Optional[int] = None,
        model: Optional[str] = None,
        system_prompt: Optional[str] = None,
        **metadata
    ) -> Session:
        """
        Create a new session.
        
        Args:
            user_id: Optional user identifier
            session_id: Custom session ID (auto-generated if not provided)
            ttl: Session TTL in seconds (uses default if not specified)
            model: Model to use for this session
            system_prompt: System prompt for this session
            **metadata: Additional session metadata
            
        Returns:
            Created Session object
        """
        with self._lock:
            # Check capacity
            if len(self._sessions) >= self.max_sessions:
                # Remove oldest expired sessions
                self.cleanup_expired()
                if len(self._sessions) >= self.max_sessions:
                    raise RuntimeError("Maximum session limit reached")
            
            # Generate session ID
            sid = session_id or str(uuid.uuid4())
            
            # Calculate expiration
            ttl_seconds = ttl or self.default_ttl
            expires_at = datetime.now() + timedelta(seconds=ttl_seconds)
            
            session = Session(
                session_id=sid,
                user_id=user_id,
                expires_at=expires_at,
                model=model,
                system_prompt=system_prompt,
                metadata=metadata
            )
            
            self._sessions[sid] = session
            logger.debug(f"Created session: {sid}")
            
            return session
    
    def get_session(self, session_id: str) -> Optional[Session]:
        """
        Get session by ID.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Session object or None if not found/expired
        """
        with self._lock:
            session = self._sessions.get(session_id)
            
            if session is None:
                return None
            
            if session.is_expired or not session.is_active:
                self._sessions.pop(session_id, None)
                return None
            
            session.touch()
            return session
    
    def get_or_create_session(
        self,
        session_id: Optional[str] = None,
        **kwargs
    ) -> Session:
        """
        Get existing session or create new one.
        
        Args:
            session_id: Session ID to look up
            **kwargs: Arguments for create_session if creating new
            
        Returns:
            Session object
        """
        if session_id:
            session = self.get_session(session_id)
            if session:
                return session
        
        return self.create_session(session_id=session_id, **kwargs)
    
    def update_session(
        self,
        session_id: str,
        **updates
    ) -> Optional[Session]:
        """
        Update session attributes.
        
        Args:
            session_id: Session identifier
            **updates: Attributes to update
            
        Returns:
            Updated Session or None if not found
        """
        with self._lock:
            session = self.get_session(session_id)
            if session:
                for key, value in updates.items():
                    if hasattr(session, key):
                        setattr(session, key, value)
                    else:
                        session.metadata[key] = value
                session.touch()
            return session
    
    def delete_session(self, session_id: str) -> bool:
        """
        Delete a session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            True if deleted, False if not found
        """
        with self._lock:
            if session_id in self._sessions:
                del self._sessions[session_id]
                logger.debug(f"Deleted session: {session_id}")
                return True
            return False
    
    def deactivate_session(self, session_id: str) -> bool:
        """
        Deactivate a session without deleting.
        
        Args:
            session_id: Session identifier
            
        Returns:
            True if deactivated, False if not found
        """
        with self._lock:
            session = self._sessions.get(session_id)
            if session:
                session.is_active = False
                return True
            return False
    
    def cleanup_expired(self) -> int:
        """
        Remove all expired sessions.
        
        Returns:
            Number of sessions removed
        """
        with self._lock:
            expired = [
                sid for sid, session in self._sessions.items()
                if session.is_expired or not session.is_active
            ]
            
            for sid in expired:
                del self._sessions[sid]
            
            if expired:
                logger.debug(f"Cleaned up {len(expired)} expired sessions")
            
            return len(expired)
    
    def list_sessions(
        self,
        user_id: Optional[str] = None,
        active_only: bool = True
    ) -> List[Session]:
        """
        List sessions with optional filtering.
        
        Args:
            user_id: Filter by user ID
            active_only: Only return active sessions
            
        Returns:
            List of matching sessions
        """
        with self._lock:
            sessions = list(self._sessions.values())
            
            if user_id:
                sessions = [s for s in sessions if s.user_id == user_id]
            
            if active_only:
                sessions = [
                    s for s in sessions 
                    if s.is_active and not s.is_expired
                ]
            
            return sessions
    
    def get_session_count(self) -> int:
        """Get current session count."""
        with self._lock:
            return len(self._sessions)
    
    def shutdown(self) -> None:
        """Shutdown session manager."""
        self._shutdown.set()
        if self._cleanup_thread:
            self._cleanup_thread.join(timeout=5)
    
    def __enter__(self) -> "SessionManager":
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.shutdown()
