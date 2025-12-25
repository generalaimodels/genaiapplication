# =============================================================================
# SOTA AUTHENTICATION SYSTEM - AUTH SERVICE
# =============================================================================
# File: auth/service.py
# Description: Business logic layer for authentication operations
#              Orchestrates repository, security, and session components
# =============================================================================

from typing import Optional, Tuple
from datetime import datetime, timezone, timedelta
import hashlib

from sqlalchemy.ext.asyncio import AsyncSession

from auth.repository import (
    UserRepository,
    SessionRepository,
    RefreshTokenRepository,
    AuditLogRepository,
)
from auth.schemas import (
    UserCreate,
    UserResponse,
    TokenResponse,
    RefreshResponse,
    SessionResponse,
    SessionListResponse,
)
from db.models import User, Session as SessionModel, AuditAction, AuditStatus
from core.security import (
    password_manager,
    jwt_manager,
    password_validator,
    generate_session_id,
)
from core.config import settings
from core.exceptions import (
    InvalidCredentialsError,
    UserExistsError,
    UserNotFoundError,
    UserInactiveError,
    AccountLockedError,
    TokenInvalidError,
    TokenExpiredError,
    SessionNotFoundError,
)


class AuthService:
    """
    ┌─────────────────────────────────────────────────────────────────────────┐
    │                    AUTHENTICATION SERVICE                                │
    │  Business logic layer handling all authentication operations            │
    │  Coordinates between repositories, security, and session management     │
    └─────────────────────────────────────────────────────────────────────────┘
    
    Responsibilities:
        - User registration with validation
        - Login with brute-force protection
        - Token generation and refresh
        - Session management
        - Password operations
        - Audit logging
    """
    
    def __init__(self, session: AsyncSession):
        """
        Initialize service with database session.
        
        Args:
            session: SQLAlchemy async session
        """
        self._session = session
        self._user_repo = UserRepository(session)
        self._session_repo = SessionRepository(session)
        self._token_repo = RefreshTokenRepository(session)
        self._audit_repo = AuditLogRepository(session)
    
    # =========================================================================
    # REGISTRATION
    # =========================================================================
    
    async def register(
        self,
        user_data: UserCreate,
        ip_address: str,
        user_agent: Optional[str] = None,
    ) -> UserResponse:
        """
        Register a new user.
        
        Args:
            user_data: Registration data
            ip_address: Client IP address
            user_agent: Client user agent
            
        Returns:
            UserResponse: Created user data
            
        Raises:
            UserExistsError: If email or username already exists
            PasswordValidationError: If password doesn't meet requirements
        """
        # Check if email exists
        if await self._user_repo.exists_email(user_data.email):
            raise UserExistsError(field="email")
        
        # Check if username exists
        if await self._user_repo.exists_username(user_data.username):
            raise UserExistsError(field="username")
        
        # Validate password strength
        password_validator.ensure_valid(user_data.password)
        
        # Hash password
        password_hash = password_manager.hash_password(user_data.password)
        
        # Create user
        user = await self._user_repo.create(
            email=user_data.email,
            username=user_data.username,
            password_hash=password_hash,
        )
        
        # Log registration
        await self._audit_repo.create(
            action=AuditAction.REGISTER,
            status=AuditStatus.SUCCESS,
            ip_address=ip_address,
            user_id=user.id,
            user_agent=user_agent,
        )
        
        return UserResponse.model_validate(user)
    
    # =========================================================================
    # LOGIN
    # =========================================================================
    
    async def login(
        self,
        email: str,
        password: str,
        ip_address: str,
        user_agent: Optional[str] = None,
        device_info: Optional[dict] = None,
        remember_me: bool = False,
    ) -> TokenResponse:
        """
        Authenticate user and create session.
        
        Flow:
            1. Validate user exists and is active
            2. Check account lock status
            3. Verify password
            4. Create session
            5. Generate tokens
            6. Log event
        
        Args:
            email: User's email
            password: User's password
            ip_address: Client IP
            user_agent: Client user agent
            device_info: Device information
            remember_me: Extend session duration
            
        Returns:
            TokenResponse: Access and refresh tokens
            
        Raises:
            InvalidCredentialsError: Invalid email/password
            UserInactiveError: Account is inactive
            AccountLockedError: Account is locked
        """
        # Get user by email
        user = await self._user_repo.get_by_email(email)
        
        if not user:
            # Log failed attempt (no user)
            await self._audit_repo.create(
                action=AuditAction.LOGIN_FAILED,
                status=AuditStatus.FAILED,
                ip_address=ip_address,
                user_agent=user_agent,
                details={"reason": "user_not_found", "email": email},
            )
            raise InvalidCredentialsError()
        
        # Check if account is active
        if not user.is_active:
            await self._audit_repo.create(
                action=AuditAction.LOGIN_FAILED,
                status=AuditStatus.FAILED,
                ip_address=ip_address,
                user_id=user.id,
                user_agent=user_agent,
                details={"reason": "account_inactive"},
            )
            raise UserInactiveError()
        
        # Check if account is locked
        if user.is_locked:
            await self._audit_repo.create(
                action=AuditAction.LOGIN_FAILED,
                status=AuditStatus.FAILED,
                ip_address=ip_address,
                user_id=user.id,
                user_agent=user_agent,
                details={"reason": "account_locked"},
            )
            raise AccountLockedError(
                locked_until=user.locked_until.isoformat() if user.locked_until else None
            )
        
        # Verify password
        is_valid, needs_rehash = password_manager.verify_password(
            password, user.password_hash
        )
        
        if not is_valid:
            # Increment failed attempts
            await self._user_repo.increment_failed_attempts(user)
            
            await self._audit_repo.create(
                action=AuditAction.LOGIN_FAILED,
                status=AuditStatus.FAILED,
                ip_address=ip_address,
                user_id=user.id,
                user_agent=user_agent,
                details={"reason": "invalid_password"},
            )
            raise InvalidCredentialsError()
        
        # Rehash password if needed (algorithm upgrade)
        if needs_rehash:
            new_hash = password_manager.hash_password(password)
            await self._user_repo.update_password(user, new_hash)
        
        # Update last login and reset failed attempts
        await self._user_repo.update_last_login(user)
        
        # Create session
        session_id = generate_session_id()
        
        # Calculate session expiration
        if remember_me:
            expires_at = datetime.now(timezone.utc) + timedelta(days=30)
        else:
            expires_at = datetime.now(timezone.utc) + timedelta(
                hours=settings.jwt_session_token_expire_hours
            )
        
        # Create session record
        session_obj = await self._session_repo.create(
            user_id=user.id,
            session_id=session_id,
            token_hash=self._hash_token(session_id),
            ip_address=ip_address,
            expires_at=expires_at,
            device_info=device_info or {"user_agent": user_agent},
        )
        
        # Generate tokens
        token_pair = jwt_manager.create_token_pair(
            user_id=user.id,
            session_id=session_id,
            roles=user.roles,
        )
        
        # Store refresh token hash in database
        refresh_expires = datetime.now(timezone.utc) + timedelta(
            days=settings.jwt_refresh_token_expire_days
        )
        await self._token_repo.create(
            user_id=user.id,
            session_id=session_id,
            token_hash=self._hash_token(token_pair.refresh_token),
            expires_at=refresh_expires,
        )
        
        # Log successful login
        await self._audit_repo.create(
            action=AuditAction.LOGIN_SUCCESS,
            status=AuditStatus.SUCCESS,
            ip_address=ip_address,
            user_id=user.id,
            user_agent=user_agent,
            details={"session_id": session_id},
        )
        
        return TokenResponse(
            access_token=token_pair.access_token,
            refresh_token=token_pair.refresh_token,
            token_type="bearer",
            expires_in=token_pair.expires_in,
            session_id=session_id,
        )
    
    # =========================================================================
    # LOGOUT
    # =========================================================================
    
    async def logout(
        self,
        session_id: str,
        user_id: str,
        ip_address: str,
        user_agent: Optional[str] = None,
    ) -> bool:
        """
        End a user session.
        
        Args:
            session_id: Session to revoke
            user_id: User ID for verification
            ip_address: Client IP
            user_agent: Client user agent
            
        Returns:
            bool: True if successful
        """
        session_obj = await self._session_repo.get_by_id(session_id)
        
        if session_obj and session_obj.user_id == user_id:
            await self._session_repo.revoke(session_obj)
        
        # Log logout
        await self._audit_repo.create(
            action=AuditAction.LOGOUT,
            status=AuditStatus.SUCCESS,
            ip_address=ip_address,
            user_id=user_id,
            user_agent=user_agent,
            details={"session_id": session_id},
        )
        
        return True
    
    async def logout_all(
        self,
        user_id: str,
        ip_address: str,
        user_agent: Optional[str] = None,
        except_current: Optional[str] = None,
    ) -> int:
        """
        End all user sessions.
        
        Args:
            user_id: User ID
            ip_address: Client IP
            user_agent: Client user agent
            except_current: Session ID to keep active
            
        Returns:
            int: Number of sessions revoked
        """
        # Get all active sessions
        sessions = await self._session_repo.get_active_by_user(user_id)
        
        count = 0
        for session_obj in sessions:
            if except_current and session_obj.session_id == except_current:
                continue
            await self._session_repo.revoke(session_obj)
            count += 1
        
        # Revoke all refresh tokens
        await self._token_repo.revoke_all_for_user(user_id)
        
        # Log
        await self._audit_repo.create(
            action=AuditAction.LOGOUT_ALL,
            status=AuditStatus.SUCCESS,
            ip_address=ip_address,
            user_id=user_id,
            user_agent=user_agent,
            details={"sessions_revoked": count},
        )
        
        return count
    
    # =========================================================================
    # TOKEN REFRESH
    # =========================================================================
    
    async def refresh_token(
        self,
        refresh_token: str,
        ip_address: str,
        user_agent: Optional[str] = None,
    ) -> RefreshResponse:
        """
        Refresh access token using refresh token.
        
        Args:
            refresh_token: Valid refresh token
            ip_address: Client IP
            user_agent: Client user agent
            
        Returns:
            RefreshResponse: New access token
            
        Raises:
            TokenInvalidError: Invalid refresh token
            TokenExpiredError: Expired refresh token
        """
        # Verify refresh token
        try:
            payload = jwt_manager.verify_token(refresh_token, expected_type="refresh")
        except TokenExpiredError:
            raise
        except Exception:
            raise TokenInvalidError()
        
        # Check if token is blacklisted (by looking up hash in DB)
        token_hash = self._hash_token(refresh_token)
        stored_token = await self._token_repo.get_by_hash(token_hash)
        
        if not stored_token or stored_token.revoked:
            raise TokenInvalidError()
        
        # Verify session is still active
        session_obj = await self._session_repo.get_by_id(payload.session_id)
        
        if not session_obj or not session_obj.is_active:
            raise TokenInvalidError()
        
        # Get user
        user = await self._user_repo.get_by_id(payload.sub)
        
        if not user or not user.is_active:
            raise TokenInvalidError()
        
        # Update session activity
        await self._session_repo.update_activity(session_obj)
        
        # Generate new access token
        access_token = jwt_manager.create_token(
            user_id=user.id,
            token_type="access",
            session_id=payload.session_id,
            roles=user.roles,
        )
        
        # Log
        await self._audit_repo.create(
            action=AuditAction.TOKEN_REFRESH,
            status=AuditStatus.SUCCESS,
            ip_address=ip_address,
            user_id=user.id,
            user_agent=user_agent,
        )
        
        return RefreshResponse(
            access_token=access_token,
            token_type="bearer",
            expires_in=settings.jwt_access_token_expire_minutes * 60,
        )
    
    # =========================================================================
    # PASSWORD OPERATIONS
    # =========================================================================
    
    async def change_password(
        self,
        user_id: str,
        current_password: str,
        new_password: str,
        ip_address: str,
        user_agent: Optional[str] = None,
    ) -> bool:
        """
        Change user's password.
        
        Args:
            user_id: User ID
            current_password: Current password
            new_password: New password
            ip_address: Client IP
            user_agent: Client user agent
            
        Returns:
            bool: True if successful
            
        Raises:
            InvalidCredentialsError: Current password incorrect
        """
        user = await self._user_repo.get_by_id(user_id)
        
        if not user:
            raise UserNotFoundError(user_id=user_id)
        
        # Verify current password
        is_valid, _ = password_manager.verify_password(
            current_password, user.password_hash
        )
        
        if not is_valid:
            raise InvalidCredentialsError()
        
        # Validate new password
        password_validator.ensure_valid(new_password)
        
        # Hash and update
        new_hash = password_manager.hash_password(new_password)
        await self._user_repo.update_password(user, new_hash)
        
        # Log
        await self._audit_repo.create(
            action=AuditAction.PASSWORD_CHANGE,
            status=AuditStatus.SUCCESS,
            ip_address=ip_address,
            user_id=user.id,
            user_agent=user_agent,
        )
        
        return True
    
    # =========================================================================
    # SESSION MANAGEMENT
    # =========================================================================
    
    async def get_user_sessions(
        self,
        user_id: str,
        current_session_id: Optional[str] = None,
    ) -> SessionListResponse:
        """
        Get all active sessions for a user.
        
        Args:
            user_id: User ID
            current_session_id: Current session to mark
            
        Returns:
            SessionListResponse: List of sessions
        """
        sessions = await self._session_repo.get_active_by_user(user_id)
        
        session_responses = [
            SessionResponse(
                session_id=s.session_id,
                device_info=s.device_info,
                ip_address=s.ip_address,
                created_at=s.created_at,
                last_activity=s.last_activity,
                is_current=s.session_id == current_session_id,
            )
            for s in sessions
        ]
        
        return SessionListResponse(
            sessions=session_responses,
            total=len(session_responses),
        )
    
    async def revoke_session(
        self,
        user_id: str,
        session_id: str,
        ip_address: str,
        user_agent: Optional[str] = None,
    ) -> bool:
        """
        Revoke a specific session.
        
        Args:
            user_id: User ID for verification
            session_id: Session to revoke
            ip_address: Client IP
            user_agent: Client user agent
            
        Returns:
            bool: True if successful
        """
        session_obj = await self._session_repo.get_by_id(session_id)
        
        if not session_obj:
            raise SessionNotFoundError(session_id=session_id)
        
        if session_obj.user_id != user_id:
            raise SessionNotFoundError(session_id=session_id)
        
        await self._session_repo.revoke(session_obj)
        
        # Log
        await self._audit_repo.create(
            action=AuditAction.SESSION_REVOKE,
            status=AuditStatus.SUCCESS,
            ip_address=ip_address,
            user_id=user_id,
            user_agent=user_agent,
            details={"revoked_session_id": session_id},
        )
        
        return True
    
    # =========================================================================
    # USER PROFILE
    # =========================================================================
    
    async def get_user(self, user_id: str) -> Optional[UserResponse]:
        """Get user by ID."""
        user = await self._user_repo.get_by_id(user_id)
        if user:
            return UserResponse.model_validate(user)
        return None
    
    async def update_user(
        self,
        user_id: str,
        email: Optional[str] = None,
        username: Optional[str] = None,
    ) -> UserResponse:
        """Update user profile."""
        user = await self._user_repo.get_by_id(user_id)
        
        if not user:
            raise UserNotFoundError(user_id=user_id)
        
        updates = {}
        
        if email and email.lower() != user.email:
            if await self._user_repo.exists_email(email):
                raise UserExistsError(field="email")
            updates["email"] = email.lower()
            updates["is_verified"] = False
        
        if username and username.lower() != user.username:
            if await self._user_repo.exists_username(username):
                raise UserExistsError(field="username")
            updates["username"] = username.lower()
        
        if updates:
            user = await self._user_repo.update(user, **updates)
        
        return UserResponse.model_validate(user)
    
    async def delete_user(
        self,
        user_id: str,
        ip_address: str,
        user_agent: Optional[str] = None,
    ) -> bool:
        """Delete user account."""
        user = await self._user_repo.get_by_id(user_id)
        
        if not user:
            raise UserNotFoundError(user_id=user_id)
        
        # Log before deletion
        await self._audit_repo.create(
            action=AuditAction.ACCOUNT_DELETE,
            status=AuditStatus.SUCCESS,
            ip_address=ip_address,
            user_id=user_id,
            user_agent=user_agent,
        )
        
        await self._user_repo.delete(user)
        
        return True
    
    # =========================================================================
    # HELPER METHODS
    # =========================================================================
    
    def _hash_token(self, token: str) -> str:
        """Create SHA-256 hash of token for storage."""
        return hashlib.sha256(token.encode()).hexdigest()
