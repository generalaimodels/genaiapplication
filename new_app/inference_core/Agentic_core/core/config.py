"""
Core configuration module for Agentic Framework 3.0.

Adheres to:
- Zero-Cost Abstraction: Pydantic settings are computed at startup.
- Environment Variable support for containerized deployment.
"""
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, SecretStr
from typing import Optional, Literal
from functools import lru_cache

class AgenticConfig(BaseSettings):
    """
    Immutable configuration for the Agentic Framework.
    """
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        frozen=True  # Immutable after creation
    )

    # Inference endpoints
    inference_base_url: str = "http://localhost:8007"
    embedding_base_url: str = "http://localhost:8009"
    
    # Model configuration
    model_name: str = "openai/gpt-oss-20b"
    embedding_model: str = "Qwen/Qwen3-Embedding-4B"
    inference_api_key: SecretStr = Field(default="EMPTY", description="API Key for the inference backend")

    # Performance Settings
    max_concurrent_agents: int = Field(default=1000, description="Maximum number of persistent agents")
    max_context_window: int = Field(default=128000, description="Max tokens per context window")
    enable_zero_copy: bool = Field(default=True, description="Enable zero-copy data paths where possible")
    
    # Storage Settings
    storage_path: str = Field(default="./data/agentic.db", description="Path to SQLite session database")
    vector_store_path: str = Field(default="./data/vectors", description="Path to vector store persistence")
    
    # Logging & Observability
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    enable_telemetry: bool = False

@lru_cache
def get_config() -> AgenticConfig:
    """
    Returns the singleton configuration instance.
    Cached for nanosecond access times in hot paths.
    """
    return AgenticConfig()
