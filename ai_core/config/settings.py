# =============================================================================
# SETTINGS - Environment-Based Configuration Management
# =============================================================================
# Comprehensive configuration system supporting all LLM providers.
# Loads from environment variables with sensible defaults.
# 
# ENVIRONMENT VARIABLES:
#   - LLM_BASE_URL: Custom endpoint for VLLM/self-hosted models
#   - LLM_API_KEY: API key for custom endpoints
#   - OPENAI_API_KEY: OpenAI API key
#   - ANTHROPIC_API_KEY: Anthropic/Claude API key
#   - GEMINI_API_KEY: Google Gemini API key
#   - AZURE_API_KEY: Azure OpenAI API key
#   - And 100+ more provider keys supported via LiteLLM
# =============================================================================

from __future__ import annotations
import os
from typing import Optional, Dict, Any, List, Literal
from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    ==========================================================================
    SETTINGS - Centralized Configuration for AI Core
    ==========================================================================
    
    This class manages all configuration for the chatbot core AI system.
    Configuration can be provided via:
        1. Environment variables (highest priority)
        2. .env file in project root
        3. Programmatic overrides
    
    Attributes:
        Provider Settings:
            provider: LLM provider name (openai, anthropic, gemini, vllm, etc.)
            model: Model name/identifier
            api_key: Provider API key
            base_url: Custom API endpoint (for VLLM/self-hosted)
        
        Generation Parameters:
            temperature: Sampling temperature (0.0-2.0)
            max_tokens: Maximum tokens in response
            top_p: Nucleus sampling parameter
            frequency_penalty: Frequency penalty (-2.0 to 2.0)
            presence_penalty: Presence penalty (-2.0 to 2.0)
        
        Session Settings:
            session_ttl: Session time-to-live in seconds
            max_history_messages: Maximum conversation history length
            enable_history: Whether to maintain conversation history
        
        System Settings:
            timeout: Request timeout in seconds
            retry_count: Number of retry attempts
            retry_delay: Base delay between retries
            log_level: Logging level
    
    Example:
        >>> settings = Settings(
        ...     provider="openai",
        ...     model="gpt-4o",
        ...     api_key="sk-..."
        ... )
        
        >>> # Or from environment variables
        >>> # export OPENAI_API_KEY="sk-..."
        >>> settings = Settings()
    ==========================================================================
    """
    
    # =========================================================================
    # MODEL CONFIGURATION
    # =========================================================================
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        populate_by_name=True,  # Allow both field names AND aliases as input
    )
    
    # =========================================================================
    # PROVIDER SETTINGS
    # =========================================================================
    
    provider: str = Field(
        default="openai",
        description=(
            "LLM provider identifier. Supported values include:\n"
            "  - 'openai': OpenAI API (GPT-4, GPT-3.5)\n"
            "  - 'anthropic': Anthropic Claude\n"
            "  - 'gemini': Google Gemini\n"
            "  - 'azure': Azure OpenAI\n"
            "  - 'vllm': Self-hosted VLLM server\n"
            "  - 'ollama': Local Ollama instance\n"
            "  - 'bedrock': AWS Bedrock\n"
            "  - 'vertex_ai': Google Vertex AI\n"
            "  - And 100+ more via LiteLLM"
        )
    )
    
    model: str = Field(
        default="gpt-4o-mini",
        description="Model identifier (e.g., 'gpt-4o', 'claude-3-5-sonnet-20241022')"
    )
    
    # =========================================================================
    # API CREDENTIALS - Supporting All Major Providers
    # =========================================================================
    
    # Primary API Key (for custom endpoints or single-provider setup)
    api_key: Optional[str] = Field(
        default=None,
        alias="LLM_API_KEY",
        description="Primary API key for LLM provider"
    )
    
    # Custom endpoint for VLLM/self-hosted models
    base_url: Optional[str] = Field(
        default=None,
        alias="LLM_BASE_URL",
        description="Custom API endpoint URL (e.g., 'http://10.180.93.12:8007/v1')"
    )
    
    # Provider-specific API keys
    openai_api_key: Optional[str] = Field(
        default=None,
        alias="OPENAI_API_KEY",
        description="OpenAI API key"
    )
    
    anthropic_api_key: Optional[str] = Field(
        default=None,
        alias="ANTHROPIC_API_KEY",
        description="Anthropic/Claude API key"
    )
    
    gemini_api_key: Optional[str] = Field(
        default=None,
        alias="GEMINI_API_KEY",
        description="Google Gemini API key"
    )
    
    azure_api_key: Optional[str] = Field(
        default=None,
        alias="AZURE_API_KEY",
        description="Azure OpenAI API key"
    )
    
    azure_api_base: Optional[str] = Field(
        default=None,
        alias="AZURE_API_BASE",
        description="Azure OpenAI endpoint base URL"
    )
    
    azure_api_version: str = Field(
        default="2024-02-15-preview",
        alias="AZURE_API_VERSION",
        description="Azure OpenAI API version"
    )
    
    aws_access_key_id: Optional[str] = Field(
        default=None,
        alias="AWS_ACCESS_KEY_ID",
        description="AWS access key for Bedrock"
    )
    
    aws_secret_access_key: Optional[str] = Field(
        default=None,
        alias="AWS_SECRET_ACCESS_KEY",
        description="AWS secret key for Bedrock"
    )
    
    aws_region_name: str = Field(
        default="us-east-1",
        alias="AWS_REGION_NAME",
        description="AWS region for Bedrock"
    )
    
    together_api_key: Optional[str] = Field(
        default=None,
        alias="TOGETHER_API_KEY",
        description="Together AI API key"
    )
    
    mistral_api_key: Optional[str] = Field(
        default=None,
        alias="MISTRAL_API_KEY",
        description="Mistral AI API key"
    )
    
    cohere_api_key: Optional[str] = Field(
        default=None,
        alias="COHERE_API_KEY",
        description="Cohere API key"
    )
    
    replicate_api_key: Optional[str] = Field(
        default=None,
        alias="REPLICATE_API_KEY",
        description="Replicate API key"
    )
    
    huggingface_api_key: Optional[str] = Field(
        default=None,
        alias="HUGGINGFACE_API_KEY",
        description="HuggingFace API key"
    )
    
    # =========================================================================
    # GENERATION PARAMETERS
    # =========================================================================
    
    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Sampling temperature (0.0-2.0). Higher = more creative."
    )
    
    max_tokens: int = Field(
        default=4096,
        ge=1,
        le=128000,
        description="Maximum tokens in response"
    )
    
    top_p: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Nucleus sampling parameter (0.0-1.0)"
    )
    
    frequency_penalty: float = Field(
        default=0.0,
        ge=-2.0,
        le=2.0,
        description="Frequency penalty (-2.0 to 2.0)"
    )
    
    presence_penalty: float = Field(
        default=0.0,
        ge=-2.0,
        le=2.0,
        description="Presence penalty (-2.0 to 2.0)"
    )
    
    stop_sequences: Optional[List[str]] = Field(
        default=None,
        description="List of stop sequences"
    )
    
    # =========================================================================
    # SESSION SETTINGS
    # =========================================================================
    
    session_ttl: int = Field(
        default=3600,
        ge=60,
        description="Session time-to-live in seconds (default: 1 hour)"
    )
    
    max_history_messages: int = Field(
        default=50,
        ge=1,
        le=1000,
        description="Maximum messages to retain in conversation history"
    )
    
    enable_history: bool = Field(
        default=True,
        description="Whether to maintain conversation history"
    )
    
    history_summarization_threshold: int = Field(
        default=30,
        ge=5,
        description="Summarize history when exceeding this message count"
    )
    
    # =========================================================================
    # DOCUMENT PROCESSING SETTINGS
    # =========================================================================
    
    chunk_size: int = Field(
        default=1000,
        ge=100,
        le=10000,
        description="Document chunk size in characters"
    )
    
    chunk_overlap: int = Field(
        default=200,
        ge=0,
        le=1000,
        description="Overlap between document chunks"
    )
    
    max_context_chunks: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Maximum chunks to include in context"
    )
    
    relevance_threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Minimum relevance score for document chunks"
    )
    
    embedding_model: str = Field(
        default="text-embedding-3-small",
        description="Model for generating embeddings"
    )
    
    # =========================================================================
    # SYSTEM SETTINGS
    # =========================================================================
    
    timeout: int = Field(
        default=60,
        ge=5,
        le=600,
        description="Request timeout in seconds"
    )
    
    retry_count: int = Field(
        default=3,
        ge=0,
        le=10,
        description="Number of retry attempts on failure"
    )
    
    retry_delay: float = Field(
        default=1.0,
        ge=0.1,
        le=60.0,
        description="Base delay between retries in seconds"
    )
    
    retry_multiplier: float = Field(
        default=2.0,
        ge=1.0,
        le=5.0,
        description="Exponential backoff multiplier"
    )
    
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO",
        description="Logging level"
    )
    
    enable_cost_tracking: bool = Field(
        default=True,
        description="Track and log API costs"
    )
    
    # =========================================================================
    # VALIDATORS
    # =========================================================================
    
    @field_validator("provider")
    @classmethod
    def validate_provider(cls, v: str) -> str:
        """Normalize provider name to lowercase."""
        return v.lower().strip()
    
    @field_validator("model")
    @classmethod
    def validate_model(cls, v: str) -> str:
        """Ensure model name is not empty."""
        if not v or not v.strip():
            raise ValueError("Model name cannot be empty")
        return v.strip()
    
    @model_validator(mode="after")
    def validate_api_credentials(self) -> "Settings":
        """
        Validate that appropriate API credentials are provided.
        
        For custom endpoints (VLLM), base_url is required.
        For cloud providers, corresponding API key must be set.
        """
        # Check for VLLM/custom endpoint
        if self.provider in ("vllm", "custom", "openai-compatible"):
            if self.base_url:
                # Custom endpoint with optional API key (some accept "EMPTY")
                return self
        
        # For standard providers, check for API key
        api_key = self.get_provider_api_key()
        
        # Allow None for runtime configuration
        return self
    
    # =========================================================================
    # HELPER METHODS
    # =========================================================================
    
    def get_provider_api_key(self) -> Optional[str]:
        """
        Get the API key for the configured provider.
        
        Returns:
            API key string or None if not configured.
            
        Priority:
            1. Provider-specific key (e.g., openai_api_key for OpenAI)
            2. Generic api_key
            3. Environment variable
        """
        provider_key_map = {
            "openai": self.openai_api_key,
            "anthropic": self.anthropic_api_key,
            "gemini": self.gemini_api_key,
            "azure": self.azure_api_key,
            "together": self.together_api_key,
            "together_ai": self.together_api_key,
            "mistral": self.mistral_api_key,
            "cohere": self.cohere_api_key,
            "replicate": self.replicate_api_key,
            "huggingface": self.huggingface_api_key,
        }
        
        # Check provider-specific key first
        provider_key = provider_key_map.get(self.provider)
        if provider_key:
            return provider_key
        
        # Fall back to generic api_key
        if self.api_key:
            return self.api_key
        
        return None
    
    def get_litellm_model_string(self) -> str:
        """
        Get the LiteLLM-compatible model string.
        
        LiteLLM uses a prefix convention for routing:
            - openai/gpt-4o
            - anthropic/claude-3-5-sonnet-20241022
            - gemini/gemini-1.5-pro
            - together_ai/meta-llama/Llama-3.2-90B
        
        Returns:
            Properly formatted model string for LiteLLM.
        """
        # If model already has provider prefix, return as-is
        if "/" in self.model and not self.model.startswith("http"):
            return self.model
        
        # Provider prefixes for LiteLLM
        prefix_map = {
            "openai": "",  # OpenAI is default, no prefix needed
            "anthropic": "anthropic/",
            "gemini": "gemini/",
            "azure": "azure/",
            "bedrock": "bedrock/",
            "vertex_ai": "vertex_ai/",
            "together": "together_ai/",
            "together_ai": "together_ai/",
            "mistral": "mistral/",
            "cohere": "cohere/",
            "replicate": "replicate/",
            "huggingface": "huggingface/",
            "ollama": "ollama/",
            "vllm": "openai/",  # VLLM uses OpenAI-compatible API
            "custom": "openai/",  # Custom endpoints typically OpenAI-compatible
        }
        
        prefix = prefix_map.get(self.provider, "")
        return f"{prefix}{self.model}"
    
    def to_generation_params(self) -> Dict[str, Any]:
        """
        Convert settings to generation parameters dict.
        
        Returns:
            Dictionary of generation parameters for API calls.
        """
        params = {
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
        }
        
        # Add optional parameters if non-default
        if self.frequency_penalty != 0.0:
            params["frequency_penalty"] = self.frequency_penalty
        
        if self.presence_penalty != 0.0:
            params["presence_penalty"] = self.presence_penalty
        
        if self.stop_sequences:
            params["stop"] = self.stop_sequences
        
        return params
    
    def to_client_config(self) -> Dict[str, Any]:
        """
        Convert settings to client configuration dict.
        
        Returns:
            Dictionary suitable for initializing LLM clients.
        """
        config = {
            "model": self.get_litellm_model_string(),
            "api_key": self.get_provider_api_key(),
            "timeout": self.timeout,
        }
        
        # Add base_url for custom endpoints
        if self.base_url:
            config["base_url"] = self.base_url
            config["api_base"] = self.base_url  # LiteLLM compatibility
        
        # Azure-specific configuration
        if self.provider == "azure":
            config["api_version"] = self.azure_api_version
            if self.azure_api_base:
                config["api_base"] = self.azure_api_base
        
        return config


# =============================================================================
# FACTORY FUNCTION FOR QUICK INITIALIZATION
# =============================================================================

def get_settings(**overrides) -> Settings:
    """
    Factory function to create Settings with optional overrides.
    
    Args:
        **overrides: Key-value pairs to override default settings.
        
    Returns:
        Configured Settings instance.
        
    Example:
        >>> settings = get_settings(
        ...     provider="vllm",
        ...     base_url="http://10.180.93.12:8007/v1",
        ...     model="my-model"
        ... )
    """
    return Settings(**overrides)
