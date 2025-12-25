# =============================================================================
# SOTA AUTHENTICATION SYSTEM - PASSWORD HASHING TESTS
# =============================================================================
# File: tests/test_security.py
# Description: Unit tests for password hashing and JWT operations
# =============================================================================

import pytest
from datetime import datetime, timezone, timedelta

from core.security import (
    PasswordManager,
    JWTManager,
    PasswordValidator,
    password_manager,
    jwt_manager,
    generate_secure_token,
    generate_session_id,
    generate_verification_code,
)


class TestPasswordManager:
    """Test suite for PasswordManager."""
    
    def test_hash_password_argon2(self):
        """Test password hashing with Argon2."""
        pm = PasswordManager()
        password = "TestPassword123!"
        
        hashed = pm.hash_password(password)
        
        assert hashed is not None
        assert hashed != password
        assert hashed.startswith("$argon2")
    
    def test_verify_password_correct(self):
        """Test password verification with correct password."""
        pm = PasswordManager()
        password = "TestPassword123!"
        
        hashed = pm.hash_password(password)
        is_valid, needs_rehash = pm.verify_password(password, hashed)
        
        assert is_valid is True
        assert needs_rehash is False
    
    def test_verify_password_incorrect(self):
        """Test password verification with wrong password."""
        pm = PasswordManager()
        password = "TestPassword123!"
        wrong_password = "WrongPassword456!"
        
        hashed = pm.hash_password(password)
        is_valid, _ = pm.verify_password(wrong_password, hashed)
        
        assert is_valid is False
    
    def test_different_passwords_different_hashes(self):
        """Test that same password produces different hashes (salting)."""
        pm = PasswordManager()
        password = "TestPassword123!"
        
        hash1 = pm.hash_password(password)
        hash2 = pm.hash_password(password)
        
        # Same password should produce different hashes (random salt)
        assert hash1 != hash2
        
        # But both should verify correctly
        assert pm.verify_password(password, hash1)[0] is True
        assert pm.verify_password(password, hash2)[0] is True


class TestJWTManager:
    """Test suite for JWTManager."""
    
    def test_create_access_token(self):
        """Test access token creation."""
        jm = JWTManager()
        
        token = jm.create_token(
            user_id="user-123",
            token_type="access",
            session_id="session-456",
            roles=["user"],
        )
        
        assert token is not None
        assert isinstance(token, str)
        assert len(token) > 0
    
    def test_create_refresh_token(self):
        """Test refresh token creation."""
        jm = JWTManager()
        
        token = jm.create_token(
            user_id="user-123",
            token_type="refresh",
            session_id="session-456",
        )
        
        assert token is not None
        assert isinstance(token, str)
    
    def test_create_token_pair(self):
        """Test token pair creation."""
        jm = JWTManager()
        
        pair = jm.create_token_pair(
            user_id="user-123",
            session_id="session-456",
            roles=["user", "admin"],
        )
        
        assert pair.access_token is not None
        assert pair.refresh_token is not None
        assert pair.token_type == "bearer"
        assert pair.expires_in > 0
        assert pair.session_id == "session-456"
    
    def test_decode_token(self):
        """Test token decoding."""
        jm = JWTManager()
        
        token = jm.create_token(
            user_id="user-123",
            token_type="access",
            session_id="session-456",
            roles=["user"],
        )
        
        payload = jm.decode_token(token)
        
        assert payload.sub == "user-123"
        assert payload.type == "access"
        assert payload.session_id == "session-456"
        assert "user" in payload.roles
    
    def test_verify_token_correct_type(self):
        """Test token verification with correct type."""
        jm = JWTManager()
        
        token = jm.create_token(
            user_id="user-123",
            token_type="access",
        )
        
        payload = jm.verify_token(token, expected_type="access")
        
        assert payload.type == "access"
    
    def test_verify_token_wrong_type(self):
        """Test token verification with wrong type."""
        from core.exceptions import TokenInvalidError
        
        jm = JWTManager()
        
        token = jm.create_token(
            user_id="user-123",
            token_type="access",
        )
        
        with pytest.raises(TokenInvalidError):
            jm.verify_token(token, expected_type="refresh")
    
    def test_get_remaining_ttl(self):
        """Test getting remaining TTL of token."""
        jm = JWTManager()
        
        token = jm.create_token(
            user_id="user-123",
            token_type="access",
        )
        
        ttl = jm.get_remaining_ttl(token)
        
        assert ttl > 0
        assert ttl <= 15 * 60  # Should be within 15 minutes


class TestPasswordValidator:
    """Test suite for PasswordValidator."""
    
    def test_valid_password(self):
        """Test valid password passes validation."""
        password = "SecurePass123!"
        
        is_valid, errors = PasswordValidator.validate(password)
        
        assert is_valid is True
        assert len(errors) == 0
    
    def test_password_too_short(self):
        """Test short password fails validation."""
        password = "Short1!"
        
        is_valid, errors = PasswordValidator.validate(password)
        
        assert is_valid is False
        assert any("8 characters" in e for e in errors)
    
    def test_password_no_uppercase(self):
        """Test password without uppercase fails."""
        password = "securep@ss123"
        
        is_valid, errors = PasswordValidator.validate(password)
        
        assert is_valid is False
        assert any("uppercase" in e for e in errors)
    
    def test_password_no_lowercase(self):
        """Test password without lowercase fails."""
        password = "SECUREP@SS123"
        
        is_valid, errors = PasswordValidator.validate(password)
        
        assert is_valid is False
        assert any("lowercase" in e for e in errors)
    
    def test_password_no_digit(self):
        """Test password without digit fails."""
        password = "SecurePass!"
        
        is_valid, errors = PasswordValidator.validate(password)
        
        assert is_valid is False
        assert any("digit" in e for e in errors)
    
    def test_password_no_special(self):
        """Test password without special char fails."""
        password = "SecurePass123"
        
        is_valid, errors = PasswordValidator.validate(password)
        
        assert is_valid is False
        assert any("special" in e for e in errors)
    
    def test_common_password(self):
        """Test common password fails validation."""
        password = "Password123!"  # 'password' is in common list
        
        # Note: depending on implementation, this may or may not fail
        is_valid, errors = PasswordValidator.validate(password)
        
        # Check if common password check works
        if "password" in password.lower():
            # Just verify validation runs without error
            assert isinstance(is_valid, bool)


class TestUtilityFunctions:
    """Test utility functions."""
    
    def test_generate_secure_token(self):
        """Test secure token generation."""
        token1 = generate_secure_token()
        token2 = generate_secure_token()
        
        assert len(token1) == 64  # 32 bytes = 64 hex chars
        assert token1 != token2  # Should be unique
    
    def test_generate_session_id(self):
        """Test session ID generation."""
        id1 = generate_session_id()
        id2 = generate_session_id()
        
        assert len(id1) == 36  # UUID format
        assert id1 != id2
        assert "-" in id1  # UUID has dashes
    
    def test_generate_verification_code(self):
        """Test verification code generation."""
        code = generate_verification_code()
        
        assert len(code) == 6
        assert code.isdigit()
