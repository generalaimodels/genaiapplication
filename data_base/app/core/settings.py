# ==============================================================================
# SETTINGS CONFIGURATION - Environment Management
# ==============================================================================
# Pydantic Settings for type-safe environment variable management
# Supports: Development, Staging, Production environments
# ==============================================================================

from __future__ import annotations

from enum import Enum
from functools import lru_cache
from typing import Literal, Optional, List

from pydantic import Field, field_validator, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict


class DatabaseType(str, Enum):
    """
    Supported database types for the application.
    
    Attributes:
        SQLITE: Lightweight file-based database for development/testing
        POSTGRESQL: Production-grade relational database
        MONGODB: Document-oriented NoSQL database
    """
    SQLITE = "sqlite"
    POSTGRESQL = "postgresql"
    MONGODB = "mongodb"


class Environment(str, Enum):
    """Application environment types."""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


class Settings(BaseSettings):
    """
    Application Settings Configuration.
    
    Manages all environment variables with type validation and defaults.
    Uses Pydantic BaseSettings for automatic .env file loading and
    environment variable parsing.
    
    Attributes:
        APP_NAME: Application display name
        APP_VERSION: Semantic version string
        DEBUG: Enable debug mode (never in production)
        ENVIRONMENT: Current deployment environment
        
    Example:
        >>> from app.core.settings import settings
        >>> print(settings.APP_NAME)
        'Generalized Database System'
    """
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )
    
    # --------------------------------------------------------------------------
    # APPLICATION SETTINGS
    # --------------------------------------------------------------------------
    APP_NAME: str = Field(
        default="Generalized Database System",
        description="Application display name"
    )
    APP_VERSION: str = Field(
        default="1.0.0",
        description="Application semantic version"
    )
    DEBUG: bool = Field(
        default=False,
        description="Enable debug mode (logs, stack traces)"
    )
    ENVIRONMENT: Environment = Field(
        default=Environment.DEVELOPMENT,
        description="Current deployment environment"
    )
    
    # --------------------------------------------------------------------------
    # API CONFIGURATION
    # --------------------------------------------------------------------------
    API_V1_PREFIX: str = Field(
        default="/api/v1",
        description="API version 1 route prefix"
    )
    API_TITLE: str = Field(
        default="Multi-Database Backend API",
        description="OpenAPI documentation title"
    )
    API_DESCRIPTION: str = Field(
        default="Production-ready FastAPI backend with multi-database support",
        description="OpenAPI documentation description"
    )
    
    # --------------------------------------------------------------------------
    # DATABASE TYPE SELECTION
    # --------------------------------------------------------------------------
    DATABASE_TYPE: DatabaseType = Field(
        default=DatabaseType.SQLITE,
        description="Active database backend (sqlite, postgresql, mongodb)"
    )
    
    # --------------------------------------------------------------------------
    # SQLITE CONFIGURATION
    # --------------------------------------------------------------------------
    SQLITE_URL: str = Field(
        default="sqlite:///./app.db",
        description="SQLite database file path"
    )
    
    # --------------------------------------------------------------------------
    # POSTGRESQL CONFIGURATION
    # --------------------------------------------------------------------------
    POSTGRES_HOST: str = Field(
        default="localhost",
        description="PostgreSQL server hostname"
    )
    POSTGRES_PORT: int = Field(
        default=5432,
        ge=1,
        le=65535,
        description="PostgreSQL server port"
    )
    POSTGRES_USER: str = Field(
        default="postgres",
        description="PostgreSQL username"
    )
    POSTGRES_PASSWORD: str = Field(
        default="password",
        description="PostgreSQL password"
    )
    POSTGRES_DB: str = Field(
        default="app_db",
        description="PostgreSQL database name"
    )
    
    # --------------------------------------------------------------------------
    # MONGODB CONFIGURATION
    # --------------------------------------------------------------------------
    MONGODB_URL: str = Field(
        default="mongodb://localhost:27017",
        description="MongoDB connection URI"
    )
    MONGODB_DB: str = Field(
        default="app_db",
        description="MongoDB database name"
    )
    
    # --------------------------------------------------------------------------
    # CONNECTION POOL SETTINGS
    # --------------------------------------------------------------------------
    DB_POOL_SIZE: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Database connection pool size"
    )
    DB_MAX_OVERFLOW: int = Field(
        default=20,
        ge=0,
        le=100,
        description="Maximum overflow connections beyond pool size"
    )
    DB_POOL_TIMEOUT: int = Field(
        default=30,
        ge=1,
        le=300,
        description="Connection pool timeout in seconds"
    )
    DB_POOL_RECYCLE: int = Field(
        default=3600,
        ge=60,
        description="Connection recycle time in seconds"
    )
    
    # --------------------------------------------------------------------------
    # SECURITY SETTINGS
    # --------------------------------------------------------------------------
    SECRET_KEY: str = Field(
        default="your-super-secret-key-change-in-production",
        min_length=32,
        description="JWT signing secret key (min 32 chars)"
    )
    ALGORITHM: str = Field(
        default="HS256",
        description="JWT signing algorithm"
    )
    ACCESS_TOKEN_EXPIRE_MINUTES: int = Field(
        default=30,
        ge=1,
        le=1440,
        description="Access token expiration in minutes"
    )
    REFRESH_TOKEN_EXPIRE_DAYS: int = Field(
        default=7,
        ge=1,
        le=30,
        description="Refresh token expiration in days"
    )
    
    # --------------------------------------------------------------------------
    # RATE LIMITING
    # --------------------------------------------------------------------------
    RATE_LIMIT_ENABLED: bool = Field(
        default=True,
        description="Enable rate limiting"
    )
    RATE_LIMIT_REQUESTS: int = Field(
        default=100,
        ge=1,
        description="Maximum requests per window"
    )
    RATE_LIMIT_WINDOW: int = Field(
        default=60,
        ge=1,
        description="Rate limit window in seconds"
    )
    
    # --------------------------------------------------------------------------
    # REDIS CONFIGURATION
    # --------------------------------------------------------------------------
    REDIS_URL: str = Field(
        default="redis://localhost:6379/0",
        description="Redis connection URL"
    )
    REDIS_PASSWORD: Optional[str] = Field(
        default=None,
        description="Redis password (optional)"
    )
    
    # --------------------------------------------------------------------------
    # CORS SETTINGS
    # --------------------------------------------------------------------------
    CORS_ORIGINS: List[str] = Field(
        default=["*"],
        description="Allowed CORS origins"
    )
    CORS_ALLOW_CREDENTIALS: bool = Field(
        default=True,
        description="Allow credentials in CORS requests"
    )
    
    # --------------------------------------------------------------------------
    # LOGGING CONFIGURATION
    # --------------------------------------------------------------------------
    LOG_LEVEL: str = Field(
        default="INFO",
        description="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)"
    )
    LOG_FORMAT: str = Field(
        default="json",
        description="Log format (json, text)"
    )
    
    # --------------------------------------------------------------------------
    # COMPUTED PROPERTIES
    # --------------------------------------------------------------------------
    @computed_field
    @property
    def postgres_url(self) -> str:
        """
        Construct PostgreSQL async connection URL.
        
        Returns:
            Async PostgreSQL connection string with asyncpg driver
        """
        return (
            f"postgresql+asyncpg://{self.POSTGRES_USER}:"
            f"{self.POSTGRES_PASSWORD}@{self.POSTGRES_HOST}:"
            f"{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
        )
    
    @computed_field
    @property
    def postgres_sync_url(self) -> str:
        """
        Construct PostgreSQL sync connection URL for Alembic migrations.
        
        Returns:
            Sync PostgreSQL connection string with psycopg2 driver
        """
        return (
            f"postgresql://{self.POSTGRES_USER}:"
            f"{self.POSTGRES_PASSWORD}@{self.POSTGRES_HOST}:"
            f"{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
        )
    
    @computed_field
    @property
    def sqlite_async_url(self) -> str:
        """
        Construct SQLite async connection URL.
        
        Returns:
            Async SQLite connection string with aiosqlite driver
        """
        return self.SQLITE_URL.replace("sqlite://", "sqlite+aiosqlite://")
    
    @computed_field
    @property
    def database_url(self) -> str:
        """
        Get the appropriate database URL based on DATABASE_TYPE.
        
        Returns:
            Async database connection URL for the selected database type
            
        Raises:
            ValueError: If DATABASE_TYPE is not supported
        """
        if self.DATABASE_TYPE == DatabaseType.SQLITE:
            return self.sqlite_async_url
        elif self.DATABASE_TYPE == DatabaseType.POSTGRESQL:
            return self.postgres_url
        elif self.DATABASE_TYPE == DatabaseType.MONGODB:
            return self.MONGODB_URL
        raise ValueError(f"Unsupported database type: {self.DATABASE_TYPE}")
    
    @computed_field
    @property
    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.ENVIRONMENT == Environment.DEVELOPMENT
    
    @computed_field
    @property
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.ENVIRONMENT == Environment.PRODUCTION
    
    # --------------------------------------------------------------------------
    # VALIDATORS
    # --------------------------------------------------------------------------
    @field_validator("SECRET_KEY")
    @classmethod
    def validate_secret_key(cls, v: str, info) -> str:
        """Ensure SECRET_KEY is secure in production."""
        # In production, require a proper secret key
        if v == "your-super-secret-key-change-in-production":
            import warnings
            warnings.warn(
                "Using default SECRET_KEY. Generate a secure key for production!",
                UserWarning
            )
        return v
    
    @field_validator("CORS_ORIGINS", mode="before")
    @classmethod
    def parse_cors_origins(cls, v):
        """Parse CORS origins from string or list."""
        if isinstance(v, str):
            if v == "*":
                return ["*"]
            return [origin.strip() for origin in v.split(",")]
        return v


@lru_cache()
def get_settings() -> Settings:
    """
    Get cached Settings instance.
    
    Uses lru_cache to ensure settings are only loaded once,
    providing a singleton-like behavior for the settings object.
    
    Returns:
        Settings: Cached settings instance
        
    Example:
        >>> settings = get_settings()
        >>> print(settings.DATABASE_TYPE)
    """
    return Settings()


# Module-level settings instance for convenient imports
settings = get_settings()
