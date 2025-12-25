# =============================================================================
# SOTA AUTHENTICATION SYSTEM - CORE CONFIGURATION MODULE
# =============================================================================
# File: core/config.py
# Description: Centralized configuration management using Pydantic Settings
#              Supports multi-environment with type-safe validation
# =============================================================================

from typing import Literal, Optional, List
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import computed_field, field_validator


class Settings(BaseSettings):
    """
    ┌─────────────────────────────────────────────────────────────────────────┐
    │                    APPLICATION SETTINGS                                  │
    │  Type-safe configuration with automatic environment variable loading    │
    │  Supports: development, staging, production environments                │
    └─────────────────────────────────────────────────────────────────────────┘
    """
    
    # -------------------------------------------------------------------------
    # APPLICATION CORE
    # -------------------------------------------------------------------------
    app_name: str = "AuthSystem"
    app_env: Literal["development", "staging", "production"] = "development"
    debug: bool = True
    secret_key: str = "change-me-in-production-minimum-32-characters-long"
    
    # -------------------------------------------------------------------------
    # DATABASE CONFIGURATION
    # -------------------------------------------------------------------------
    db_type: Literal["sqlite", "postgresql"] = "sqlite"
    
    # SQLite (Development/Testing)
    sqlite_path: str = "./data/auth.db"
    
    # PostgreSQL (Production)
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_user: str = "auth_user"
    postgres_password: str = "secure-password"
    postgres_db: str = "auth_db"
    
    # Connection Pool
    db_pool_size: int = 5
    db_max_overflow: int = 10
    db_pool_timeout: int = 30
    
    # -------------------------------------------------------------------------
    # REDIS CONFIGURATION
    # -------------------------------------------------------------------------
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_password: Optional[str] = None
    redis_db: int = 0
    redis_ssl: bool = False
    session_ttl: int = 86400  # 24 hours in seconds
    
    # -------------------------------------------------------------------------
    # JWT CONFIGURATION
    # -------------------------------------------------------------------------
    jwt_secret_key: str = "jwt-secret-key-change-in-production-minimum-64-chars"
    jwt_algorithm: Literal["HS256", "HS384", "HS512", "RS256", "RS384", "RS512"] = "HS256"
    jwt_access_token_expire_minutes: int = 15
    jwt_refresh_token_expire_days: int = 7
    jwt_session_token_expire_hours: int = 24
    
    # RSA Keys (for RS256/RS384/RS512)
    jwt_private_key_path: Optional[str] = None
    jwt_public_key_path: Optional[str] = None
    
    # -------------------------------------------------------------------------
    # SECURITY SETTINGS
    # -------------------------------------------------------------------------
    password_hash_algorithm: Literal["argon2", "bcrypt"] = "argon2"
    bcrypt_rounds: int = 12
    
    # Argon2 Parameters (OWASP recommended)
    argon2_memory_cost: int = 65536  # 64 MB
    argon2_time_cost: int = 3
    argon2_parallelism: int = 4
    
    # Rate Limiting
    rate_limit_per_minute: int = 60
    rate_limit_burst: int = 10
    
    # Brute Force Protection
    max_login_attempts: int = 5
    lockout_duration_minutes: int = 15
    
    # -------------------------------------------------------------------------
    # CORS CONFIGURATION
    # -------------------------------------------------------------------------
    cors_origins: str = "http://localhost:3000,http://localhost:8080"
    cors_allow_credentials: bool = True
    
    # -------------------------------------------------------------------------
    # LOGGING
    # -------------------------------------------------------------------------
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    log_format: Literal["json", "text"] = "json"
    
    # -------------------------------------------------------------------------
    # COMPUTED PROPERTIES
    # -------------------------------------------------------------------------
    
    @computed_field
    @property
    def database_url(self) -> str:
        """
        Dynamically construct the database URL based on db_type selection.
        
        Returns:
            str: Async-compatible database connection URL
            
        Example:
            - SQLite:     "sqlite+aiosqlite:///./data/auth.db"
            - PostgreSQL: "postgresql+asyncpg://user:pass@host:port/db"
        """
        if self.db_type == "sqlite":
            # Ensure data directory exists for SQLite
            db_path = Path(self.sqlite_path)
            db_path.parent.mkdir(parents=True, exist_ok=True)
            return f"sqlite+aiosqlite:///{self.sqlite_path}"
        else:
            return (
                f"postgresql+asyncpg://{self.postgres_user}:"
                f"{self.postgres_password}@{self.postgres_host}:"
                f"{self.postgres_port}/{self.postgres_db}"
            )
    
    @computed_field
    @property
    def sync_database_url(self) -> str:
        """
        Synchronous database URL for Alembic migrations.
        """
        if self.db_type == "sqlite":
            return f"sqlite:///{self.sqlite_path}"
        else:
            return (
                f"postgresql://{self.postgres_user}:"
                f"{self.postgres_password}@{self.postgres_host}:"
                f"{self.postgres_port}/{self.postgres_db}"
            )
    
    @computed_field
    @property
    def redis_url(self) -> str:
        """
        Construct Redis connection URL with optional authentication.
        """
        protocol = "rediss" if self.redis_ssl else "redis"
        auth = f":{self.redis_password}@" if self.redis_password else ""
        return f"{protocol}://{auth}{self.redis_host}:{self.redis_port}/{self.redis_db}"
    
    @computed_field
    @property
    def cors_origins_list(self) -> List[str]:
        """
        Parse CORS origins from comma-separated string to list.
        """
        return [origin.strip() for origin in self.cors_origins.split(",")]
    
    @computed_field
    @property
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.app_env == "production"
    
    @computed_field
    @property
    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.app_env == "development"
    
    # -------------------------------------------------------------------------
    # VALIDATORS
    # -------------------------------------------------------------------------
    
    @field_validator("secret_key")
    @classmethod
    def validate_secret_key(cls, v: str) -> str:
        """Ensure secret key has minimum length for security."""
        if len(v) < 32:
            raise ValueError("SECRET_KEY must be at least 32 characters long")
        return v
    
    @field_validator("jwt_secret_key")
    @classmethod
    def validate_jwt_secret_key(cls, v: str) -> str:
        """Ensure JWT secret key has minimum length for security."""
        if len(v) < 32:
            raise ValueError("JWT_SECRET_KEY must be at least 32 characters long")
        return v
    
    # -------------------------------------------------------------------------
    # PYDANTIC SETTINGS CONFIG
    # -------------------------------------------------------------------------
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )


@lru_cache()
def get_settings() -> Settings:
    """
    Retrieve cached settings instance for application-wide configuration access.
    
    Uses LRU cache to ensure single instance throughout application lifecycle.
    This pattern provides both performance and singleton guarantee.
    
    Returns:
        Settings: Validated configuration instance
        
    Usage:
        from core.config import get_settings
        settings = get_settings()
        print(settings.database_url)
    """
    return Settings()


# =============================================================================
# MODULE EXPORTS
# =============================================================================
settings = get_settings()
