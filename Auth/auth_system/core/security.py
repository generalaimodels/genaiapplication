# =============================================================================
# SOTA AUTHENTICATION SYSTEM - CORE SECURITY MODULE
# =============================================================================
# File: core/security.py
# Description: Security utilities including password hashing and JWT management
#              Implements OWASP best practices with Argon2id and RS256/HS256
# =============================================================================

from typing import Optional, Dict, Any, Literal, Union
from datetime import datetime, timedelta, timezone
from uuid import uuid4
import secrets

from passlib.context import CryptContext
from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError, InvalidHash
from jose import jwt, JWTError, ExpiredSignatureError
from pydantic import BaseModel

from core.config import settings
from core.exceptions import (
    TokenExpiredError,
    TokenInvalidError,
    TokenBlacklistedError,
    PasswordValidationError,
)


# =============================================================================
# PASSWORD HASHER CONFIGURATION
# =============================================================================

class PasswordManager:
    """
    ┌─────────────────────────────────────────────────────────────────────────┐
    │                    PASSWORD HASHING MANAGER                              │
    │  Implements OWASP recommended Argon2id with Bcrypt fallback            │
    │  Supports automatic algorithm upgrade on password verification         │
    └─────────────────────────────────────────────────────────────────────────┘
    
    Algorithm Selection:
        - Primary:  Argon2id (OWASP recommended for new passwords)
        - Fallback: Bcrypt (for legacy password verification)
    
    Argon2id Parameters (OWASP recommended):
        - Memory:      64 MB (65536 KB)
        - Iterations:  3
        - Parallelism: 4
        - Salt:        16 bytes (auto-generated)
    """
    
    def __init__(self):
        """Initialize password manager with configured algorithms."""
        
        # Argon2id hasher with OWASP recommended parameters
        self._argon2_hasher = PasswordHasher(
            time_cost=settings.argon2_time_cost,
            memory_cost=settings.argon2_memory_cost,
            parallelism=settings.argon2_parallelism,
            hash_len=32,
            salt_len=16,
        )
        
        # Bcrypt context for legacy password support
        self._bcrypt_context = CryptContext(
            schemes=["bcrypt"],
            deprecated="auto",
            bcrypt__rounds=settings.bcrypt_rounds,
        )
        
        # Current preferred algorithm
        self._preferred_algorithm = settings.password_hash_algorithm
    
    def hash_password(self, password: str) -> str:
        """
        Hash a password using the configured algorithm.
        
        Args:
            password: Plain text password to hash
            
        Returns:
            str: Hashed password string
            
        Example:
            >>> pm = PasswordManager()
            >>> hashed = pm.hash_password("SecurePassword123!")
            >>> hashed.startswith("$argon2id$")
            True
        """
        if self._preferred_algorithm == "argon2":
            return self._argon2_hasher.hash(password)
        else:
            return self._bcrypt_context.hash(password)
    
    def verify_password(
        self,
        plain_password: str,
        hashed_password: str
    ) -> tuple[bool, bool]:
        """
        Verify a password against its hash with algorithm detection.
        
        Automatically detects the algorithm used and verifies accordingly.
        Returns both verification status and whether rehash is needed.
        
        Args:
            plain_password: Plain text password to verify
            hashed_password: Stored password hash
            
        Returns:
            tuple[bool, bool]: (is_valid, needs_rehash)
                - is_valid: True if password matches
                - needs_rehash: True if password should be rehashed with current algorithm
                
        Example:
            >>> pm = PasswordManager()
            >>> hashed = pm.hash_password("test123")
            >>> is_valid, needs_rehash = pm.verify_password("test123", hashed)
            >>> is_valid
            True
        """
        needs_rehash = False
        is_valid = False
        
        try:
            # Detect algorithm from hash prefix
            if hashed_password.startswith("$argon2"):
                # Verify with Argon2
                try:
                    self._argon2_hasher.verify(hashed_password, plain_password)
                    is_valid = True
                    
                    # Check if rehash is needed (parameters changed)
                    if self._argon2_hasher.check_needs_rehash(hashed_password):
                        needs_rehash = True
                        
                except VerifyMismatchError:
                    is_valid = False
                except InvalidHash:
                    is_valid = False
                    
            elif hashed_password.startswith("$2"):
                # Verify with Bcrypt
                is_valid = self._bcrypt_context.verify(plain_password, hashed_password)
                
                # Upgrade to Argon2 if that's the preferred algorithm
                if is_valid and self._preferred_algorithm == "argon2":
                    needs_rehash = True
                    
            else:
                # Unknown hash format
                is_valid = False
                
        except Exception:
            is_valid = False
        
        return is_valid, needs_rehash
    
    def needs_upgrade(self, hashed_password: str) -> bool:
        """
        Check if a password hash needs to be upgraded to current algorithm.
        
        Args:
            hashed_password: Stored password hash
            
        Returns:
            bool: True if rehash is recommended
        """
        if self._preferred_algorithm == "argon2":
            if not hashed_password.startswith("$argon2"):
                return True
            try:
                return self._argon2_hasher.check_needs_rehash(hashed_password)
            except InvalidHash:
                return True
        return False


# =============================================================================
# JWT TOKEN PAYLOAD MODELS
# =============================================================================

class TokenPayload(BaseModel):
    """
    JWT Token payload structure for type-safe token handling.
    
    Attributes:
        sub: Subject (user ID)
        jti: Unique token identifier
        type: Token type (access, refresh, session)
        session_id: Associated session identifier
        iat: Issued at timestamp
        exp: Expiration timestamp
        roles: User roles list
    """
    sub: str  # User ID
    jti: str  # Token ID
    type: Literal["access", "refresh", "session"]
    session_id: Optional[str] = None
    iat: datetime
    exp: datetime
    roles: list[str] = []
    
    class Config:
        from_attributes = True


class TokenPair(BaseModel):
    """Token pair response model."""
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int
    session_id: str


# =============================================================================
# JWT TOKEN MANAGER
# =============================================================================

class JWTManager:
    """
    ┌─────────────────────────────────────────────────────────────────────────┐
    │                    JWT TOKEN MANAGER                                     │
    │  Handles creation, validation, and lifecycle of JWT tokens             │
    │  Supports both symmetric (HS256) and asymmetric (RS256) algorithms     │
    └─────────────────────────────────────────────────────────────────────────┘
    
    Token Types:
        - Access Token:  Short-lived (15 min), for API access
        - Refresh Token: Long-lived (7 days), for token renewal
        - Session Token: Medium-lived (24 hours), for session tracking
    
    Security Features:
        - Unique token ID (jti) for blacklisting
        - Session binding for multi-device support
        - Configurable expiration times
        - Algorithm flexibility (HS256/RS256)
    """
    
    def __init__(self):
        """Initialize JWT manager with configured settings."""
        self._secret_key = settings.jwt_secret_key
        self._algorithm = settings.jwt_algorithm
        self._access_token_expire = timedelta(minutes=settings.jwt_access_token_expire_minutes)
        self._refresh_token_expire = timedelta(days=settings.jwt_refresh_token_expire_days)
        self._session_token_expire = timedelta(hours=settings.jwt_session_token_expire_hours)
        
        # Load RSA keys if using asymmetric algorithm
        self._private_key: Optional[str] = None
        self._public_key: Optional[str] = None
        
        if self._algorithm.startswith("RS"):
            self._load_rsa_keys()
    
    def _load_rsa_keys(self) -> None:
        """Load RSA keys from configured paths."""
        if settings.jwt_private_key_path:
            with open(settings.jwt_private_key_path, "r") as f:
                self._private_key = f.read()
        
        if settings.jwt_public_key_path:
            with open(settings.jwt_public_key_path, "r") as f:
                self._public_key = f.read()
    
    def _get_signing_key(self) -> str:
        """Get the appropriate signing key based on algorithm."""
        if self._algorithm.startswith("RS") and self._private_key:
            return self._private_key
        return self._secret_key
    
    def _get_verification_key(self) -> str:
        """Get the appropriate verification key based on algorithm."""
        if self._algorithm.startswith("RS") and self._public_key:
            return self._public_key
        return self._secret_key
    
    def create_token(
        self,
        user_id: str,
        token_type: Literal["access", "refresh", "session"],
        session_id: Optional[str] = None,
        roles: Optional[list[str]] = None,
        additional_claims: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Create a JWT token with specified claims.
        
        Args:
            user_id: User identifier (subject)
            token_type: Type of token to create
            session_id: Optional session identifier
            roles: User roles for authorization
            additional_claims: Additional custom claims
            
        Returns:
            str: Encoded JWT token
            
        Example:
            >>> jwt_mgr = JWTManager()
            >>> token = jwt_mgr.create_token(
            ...     user_id="uuid-123",
            ...     token_type="access",
            ...     session_id="session-456",
            ...     roles=["user", "admin"]
            ... )
        """
        now = datetime.now(timezone.utc)
        
        # Determine expiration based on token type
        if token_type == "access":
            expires_delta = self._access_token_expire
        elif token_type == "refresh":
            expires_delta = self._refresh_token_expire
        else:  # session
            expires_delta = self._session_token_expire
        
        expire = now + expires_delta
        
        # Build claims
        claims = {
            "sub": user_id,
            "jti": str(uuid4()),
            "type": token_type,
            "iat": now,
            "exp": expire,
            "roles": roles or [],
        }
        
        if session_id:
            claims["session_id"] = session_id
        
        if additional_claims:
            claims.update(additional_claims)
        
        # Encode and sign token
        return jwt.encode(
            claims,
            self._get_signing_key(),
            algorithm=self._algorithm
        )
    
    def create_token_pair(
        self,
        user_id: str,
        session_id: str,
        roles: Optional[list[str]] = None,
    ) -> TokenPair:
        """
        Create a complete token pair (access + refresh).
        
        Args:
            user_id: User identifier
            session_id: Session identifier
            roles: User roles
            
        Returns:
            TokenPair: Access and refresh tokens with metadata
        """
        access_token = self.create_token(
            user_id=user_id,
            token_type="access",
            session_id=session_id,
            roles=roles,
        )
        
        refresh_token = self.create_token(
            user_id=user_id,
            token_type="refresh",
            session_id=session_id,
            roles=roles,
        )
        
        return TokenPair(
            access_token=access_token,
            refresh_token=refresh_token,
            token_type="bearer",
            expires_in=int(self._access_token_expire.total_seconds()),
            session_id=session_id,
        )
    
    def decode_token(
        self,
        token: str,
        verify_exp: bool = True,
    ) -> TokenPayload:
        """
        Decode and validate a JWT token.
        
        Args:
            token: Encoded JWT token
            verify_exp: Whether to verify expiration
            
        Returns:
            TokenPayload: Decoded and validated token payload
            
        Raises:
            TokenExpiredError: If token has expired
            TokenInvalidError: If token is malformed or signature is invalid
        """
        try:
            payload = jwt.decode(
                token,
                self._get_verification_key(),
                algorithms=[self._algorithm],
                options={"verify_exp": verify_exp}
            )
            
            return TokenPayload(
                sub=payload.get("sub", ""),
                jti=payload.get("jti", ""),
                type=payload.get("type", "access"),
                session_id=payload.get("session_id"),
                iat=datetime.fromtimestamp(payload.get("iat", 0), tz=timezone.utc),
                exp=datetime.fromtimestamp(payload.get("exp", 0), tz=timezone.utc),
                roles=payload.get("roles", []),
            )
            
        except ExpiredSignatureError:
            raise TokenExpiredError()
        except JWTError as e:
            raise TokenInvalidError(details={"error": str(e)})
    
    def verify_token(
        self,
        token: str,
        expected_type: Optional[Literal["access", "refresh", "session"]] = None,
    ) -> TokenPayload:
        """
        Verify a token and optionally check its type.
        
        Args:
            token: JWT token to verify
            expected_type: Expected token type (optional)
            
        Returns:
            TokenPayload: Verified token payload
            
        Raises:
            TokenExpiredError: If token has expired
            TokenInvalidError: If token is invalid or wrong type
        """
        payload = self.decode_token(token)
        
        if expected_type and payload.type != expected_type:
            raise TokenInvalidError(
                details={"expected_type": expected_type, "actual_type": payload.type}
            )
        
        return payload
    
    def get_token_expiry(self, token: str) -> Optional[datetime]:
        """
        Get the expiration time of a token without full validation.
        
        Args:
            token: JWT token
            
        Returns:
            datetime: Token expiration time, or None if invalid
        """
        try:
            payload = self.decode_token(token, verify_exp=False)
            return payload.exp
        except Exception:
            return None
    
    def get_remaining_ttl(self, token: str) -> int:
        """
        Get remaining time-to-live in seconds.
        
        Args:
            token: JWT token
            
        Returns:
            int: Seconds until expiration, 0 if expired
        """
        expiry = self.get_token_expiry(token)
        if expiry is None:
            return 0
        
        remaining = (expiry - datetime.now(timezone.utc)).total_seconds()
        return max(0, int(remaining))


# =============================================================================
# PASSWORD VALIDATION
# =============================================================================

class PasswordValidator:
    """
    Password strength validation following OWASP guidelines.
    
    Rules:
        - Minimum 8 characters
        - Maximum 128 characters
        - At least one uppercase letter
        - At least one lowercase letter
        - At least one digit
        - At least one special character
        - Not in common passwords list
    """
    
    COMMON_PASSWORDS = {
        "password", "123456", "12345678", "qwerty", "abc123",
        "monkey", "1234567", "letmein", "trustno1", "dragon",
        "baseball", "iloveyou", "master", "sunshine", "ashley",
        "bailey", "shadow", "123123", "654321", "superman",
        "qazwsx", "michael", "football", "password1", "password123"
    }
    
    @classmethod
    def validate(cls, password: str) -> tuple[bool, list[str]]:
        """
        Validate password strength.
        
        Args:
            password: Password to validate
            
        Returns:
            tuple[bool, list[str]]: (is_valid, list of error messages)
        """
        errors = []
        
        if len(password) < 8:
            errors.append("Password must be at least 8 characters long")
        
        if len(password) > 128:
            errors.append("Password must not exceed 128 characters")
        
        if not any(c.isupper() for c in password):
            errors.append("Password must contain at least one uppercase letter")
        
        if not any(c.islower() for c in password):
            errors.append("Password must contain at least one lowercase letter")
        
        if not any(c.isdigit() for c in password):
            errors.append("Password must contain at least one digit")
        
        special_chars = set("!@#$%^&*()_+-=[]{}|;':\",./<>?`~")
        if not any(c in special_chars for c in password):
            errors.append("Password must contain at least one special character")
        
        if password.lower() in cls.COMMON_PASSWORDS:
            errors.append("Password is too common")
        
        return len(errors) == 0, errors
    
    @classmethod
    def ensure_valid(cls, password: str) -> None:
        """
        Validate password and raise exception if invalid.
        
        Args:
            password: Password to validate
            
        Raises:
            PasswordValidationError: If password doesn't meet requirements
        """
        is_valid, errors = cls.validate(password)
        if not is_valid:
            raise PasswordValidationError(
                message="; ".join(errors),
                details={"validation_errors": errors}
            )


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def generate_secure_token(length: int = 32) -> str:
    """
    Generate a cryptographically secure random token.
    
    Args:
        length: Number of bytes (result will be 2x in hex)
        
    Returns:
        str: Hex-encoded random token
    """
    return secrets.token_hex(length)


def generate_session_id() -> str:
    """Generate a unique session identifier."""
    return str(uuid4())


def generate_verification_code(length: int = 6) -> str:
    """Generate a numeric verification code."""
    return "".join(secrets.choice("0123456789") for _ in range(length))


# =============================================================================
# SINGLETON INSTANCES
# =============================================================================

password_manager = PasswordManager()
jwt_manager = JWTManager()
password_validator = PasswordValidator()
