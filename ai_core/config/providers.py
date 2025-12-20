# =============================================================================
# PROVIDERS - Provider Registry and Configuration Presets
# =============================================================================
# Centralized registry of all supported LLM providers with their configurations,
# model mappings, default parameters, and rate limit information.
# =============================================================================

from __future__ import annotations
from typing import Dict, Any, List, Optional, Literal
from dataclasses import dataclass, field
from enum import Enum


# =============================================================================
# PROVIDER TYPES
# =============================================================================

class ProviderType(str, Enum):
    """
    Enumeration of supported LLM provider categories.
    """
    COMMERCIAL_API = "commercial_api"      # OpenAI, Anthropic, Google, etc.
    CLOUD_PLATFORM = "cloud_platform"      # Azure, AWS Bedrock, Vertex AI
    OPEN_SOURCE = "open_source"            # Together, Replicate, HuggingFace
    SELF_HOSTED = "self_hosted"            # VLLM, Ollama, LocalAI


# =============================================================================
# PROVIDER CONFIGURATION
# =============================================================================

@dataclass
class ProviderConfig:
    """
    Configuration for a single LLM provider.
    
    Attributes:
        name: Provider identifier
        display_name: Human-readable provider name
        provider_type: Category of provider
        api_key_env: Environment variable name for API key
        base_url: Default API endpoint (if applicable)
        default_model: Recommended default model
        supported_models: List of known supported models
        supports_streaming: Whether streaming is supported
        supports_embeddings: Whether embeddings are supported
        supports_function_calling: Whether function calling is supported
        rate_limits: Rate limit configuration
        default_params: Default generation parameters
    """
    name: str
    display_name: str
    provider_type: ProviderType
    api_key_env: str
    base_url: Optional[str] = None
    default_model: str = ""
    supported_models: List[str] = field(default_factory=list)
    supports_streaming: bool = True
    supports_embeddings: bool = False
    supports_function_calling: bool = False
    supports_vision: bool = False
    rate_limits: Dict[str, int] = field(default_factory=dict)
    default_params: Dict[str, Any] = field(default_factory=dict)
    litellm_prefix: str = ""


# =============================================================================
# PROVIDER CONFIGURATIONS REGISTRY
# =============================================================================

PROVIDER_CONFIGS: Dict[str, ProviderConfig] = {
    
    # =========================================================================
    # COMMERCIAL API PROVIDERS
    # =========================================================================
    
    "openai": ProviderConfig(
        name="openai",
        display_name="OpenAI",
        provider_type=ProviderType.COMMERCIAL_API,
        api_key_env="OPENAI_API_KEY",
        base_url="https://api.openai.com/v1",
        default_model="gpt-4o-mini",
        supported_models=[
            "gpt-4o",
            "gpt-4o-mini", 
            "gpt-4-turbo",
            "gpt-4",
            "gpt-3.5-turbo",
            "o1-preview",
            "o1-mini",
            "gpt-4-vision-preview",
        ],
        supports_streaming=True,
        supports_embeddings=True,
        supports_function_calling=True,
        supports_vision=True,
        rate_limits={"rpm": 10000, "tpm": 2000000},
        default_params={"temperature": 0.7, "max_tokens": 4096},
        litellm_prefix="",
    ),
    
    "anthropic": ProviderConfig(
        name="anthropic",
        display_name="Anthropic Claude",
        provider_type=ProviderType.COMMERCIAL_API,
        api_key_env="ANTHROPIC_API_KEY",
        base_url="https://api.anthropic.com",
        default_model="claude-3-5-sonnet-20241022",
        supported_models=[
            "claude-3-5-sonnet-20241022",
            "claude-3-5-haiku-20241022",
            "claude-3-opus-20240229",
            "claude-3-sonnet-20240229",
            "claude-3-haiku-20240307",
        ],
        supports_streaming=True,
        supports_embeddings=False,
        supports_function_calling=True,
        supports_vision=True,
        rate_limits={"rpm": 4000, "tpm": 400000},
        default_params={"temperature": 0.7, "max_tokens": 4096},
        litellm_prefix="anthropic/",
    ),
    
    "gemini": ProviderConfig(
        name="gemini",
        display_name="Google Gemini",
        provider_type=ProviderType.COMMERCIAL_API,
        api_key_env="GEMINI_API_KEY",
        base_url="https://generativelanguage.googleapis.com/v1beta",
        default_model="gemini-1.5-flash",
        supported_models=[
            "gemini-1.5-pro",
            "gemini-1.5-flash",
            "gemini-1.5-flash-8b",
            "gemini-pro",
            "gemini-pro-vision",
        ],
        supports_streaming=True,
        supports_embeddings=True,
        supports_function_calling=True,
        supports_vision=True,
        rate_limits={"rpm": 1500, "tpm": 1000000},
        default_params={"temperature": 0.7, "max_tokens": 8192},
        litellm_prefix="gemini/",
    ),
    
    "mistral": ProviderConfig(
        name="mistral",
        display_name="Mistral AI",
        provider_type=ProviderType.COMMERCIAL_API,
        api_key_env="MISTRAL_API_KEY",
        base_url="https://api.mistral.ai/v1",
        default_model="mistral-large-latest",
        supported_models=[
            "mistral-large-latest",
            "mistral-medium-latest",
            "mistral-small-latest",
            "open-mixtral-8x22b",
            "open-mixtral-8x7b",
            "codestral-latest",
        ],
        supports_streaming=True,
        supports_embeddings=True,
        supports_function_calling=True,
        supports_vision=False,
        rate_limits={"rpm": 1000, "tpm": 500000},
        default_params={"temperature": 0.7, "max_tokens": 4096},
        litellm_prefix="mistral/",
    ),
    
    "cohere": ProviderConfig(
        name="cohere",
        display_name="Cohere",
        provider_type=ProviderType.COMMERCIAL_API,
        api_key_env="COHERE_API_KEY",
        base_url="https://api.cohere.ai/v1",
        default_model="command-r-plus",
        supported_models=[
            "command-r-plus",
            "command-r",
            "command",
            "command-light",
        ],
        supports_streaming=True,
        supports_embeddings=True,
        supports_function_calling=True,
        supports_vision=False,
        rate_limits={"rpm": 10000, "tpm": 100000},
        default_params={"temperature": 0.7, "max_tokens": 4096},
        litellm_prefix="cohere/",
    ),
    
    # =========================================================================
    # CLOUD PLATFORM PROVIDERS
    # =========================================================================
    
    "azure": ProviderConfig(
        name="azure",
        display_name="Azure OpenAI",
        provider_type=ProviderType.CLOUD_PLATFORM,
        api_key_env="AZURE_API_KEY",
        base_url=None,  # User must provide
        default_model="gpt-4o",
        supported_models=[
            "gpt-4o",
            "gpt-4",
            "gpt-35-turbo",
        ],
        supports_streaming=True,
        supports_embeddings=True,
        supports_function_calling=True,
        supports_vision=True,
        rate_limits={"rpm": 10000, "tpm": 1000000},
        default_params={"temperature": 0.7, "max_tokens": 4096},
        litellm_prefix="azure/",
    ),
    
    "bedrock": ProviderConfig(
        name="bedrock",
        display_name="AWS Bedrock",
        provider_type=ProviderType.CLOUD_PLATFORM,
        api_key_env="AWS_ACCESS_KEY_ID",
        base_url=None,
        default_model="anthropic.claude-3-sonnet-20240229-v1:0",
        supported_models=[
            "anthropic.claude-3-5-sonnet-20241022-v2:0",
            "anthropic.claude-3-sonnet-20240229-v1:0",
            "anthropic.claude-3-haiku-20240307-v1:0",
            "amazon.titan-text-express-v1",
            "amazon.titan-text-lite-v1",
            "meta.llama3-70b-instruct-v1:0",
            "mistral.mixtral-8x7b-instruct-v0:1",
        ],
        supports_streaming=True,
        supports_embeddings=True,
        supports_function_calling=True,
        supports_vision=True,
        rate_limits={"rpm": 1000, "tpm": 100000},
        default_params={"temperature": 0.7, "max_tokens": 4096},
        litellm_prefix="bedrock/",
    ),
    
    "vertex_ai": ProviderConfig(
        name="vertex_ai",
        display_name="Google Vertex AI",
        provider_type=ProviderType.CLOUD_PLATFORM,
        api_key_env="GOOGLE_APPLICATION_CREDENTIALS",
        base_url=None,
        default_model="gemini-1.5-pro",
        supported_models=[
            "gemini-1.5-pro",
            "gemini-1.5-flash",
            "claude-3-5-sonnet@20241022",
            "claude-3-sonnet@20240229",
        ],
        supports_streaming=True,
        supports_embeddings=True,
        supports_function_calling=True,
        supports_vision=True,
        rate_limits={"rpm": 5000, "tpm": 500000},
        default_params={"temperature": 0.7, "max_tokens": 8192},
        litellm_prefix="vertex_ai/",
    ),
    
    # =========================================================================
    # OPEN SOURCE / INFERENCE PROVIDERS
    # =========================================================================
    
    "together": ProviderConfig(
        name="together",
        display_name="Together AI",
        provider_type=ProviderType.OPEN_SOURCE,
        api_key_env="TOGETHER_API_KEY",
        base_url="https://api.together.xyz/v1",
        default_model="meta-llama/Llama-3.2-90B-Vision-Instruct-Turbo",
        supported_models=[
            "meta-llama/Llama-3.2-90B-Vision-Instruct-Turbo",
            "meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo",
            "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
            "mistralai/Mixtral-8x22B-Instruct-v0.1",
            "Qwen/Qwen2.5-72B-Instruct-Turbo",
        ],
        supports_streaming=True,
        supports_embeddings=True,
        supports_function_calling=True,
        supports_vision=True,
        rate_limits={"rpm": 600, "tpm": 100000},
        default_params={"temperature": 0.7, "max_tokens": 4096},
        litellm_prefix="together_ai/",
    ),
    
    "replicate": ProviderConfig(
        name="replicate",
        display_name="Replicate",
        provider_type=ProviderType.OPEN_SOURCE,
        api_key_env="REPLICATE_API_KEY",
        base_url="https://api.replicate.com/v1",
        default_model="meta/llama-2-70b-chat",
        supported_models=[
            "meta/llama-2-70b-chat",
            "meta/llama-2-13b-chat",
            "mistralai/mixtral-8x7b-instruct-v0.1",
        ],
        supports_streaming=True,
        supports_embeddings=False,
        supports_function_calling=False,
        supports_vision=True,
        rate_limits={"rpm": 600, "tpm": 100000},
        default_params={"temperature": 0.7, "max_tokens": 4096},
        litellm_prefix="replicate/",
    ),
    
    "huggingface": ProviderConfig(
        name="huggingface",
        display_name="HuggingFace",
        provider_type=ProviderType.OPEN_SOURCE,
        api_key_env="HUGGINGFACE_API_KEY",
        base_url="https://api-inference.huggingface.co/models",
        default_model="meta-llama/Llama-3.2-3B-Instruct",
        supported_models=[
            "meta-llama/Llama-3.2-3B-Instruct",
            "mistralai/Mistral-7B-Instruct-v0.3",
            "microsoft/Phi-3-mini-4k-instruct",
        ],
        supports_streaming=True,
        supports_embeddings=True,
        supports_function_calling=False,
        supports_vision=False,
        rate_limits={"rpm": 300, "tpm": 50000},
        default_params={"temperature": 0.7, "max_tokens": 2048},
        litellm_prefix="huggingface/",
    ),
    
    "groq": ProviderConfig(
        name="groq",
        display_name="Groq",
        provider_type=ProviderType.OPEN_SOURCE,
        api_key_env="GROQ_API_KEY",
        base_url="https://api.groq.com/openai/v1",
        default_model="llama-3.1-70b-versatile",
        supported_models=[
            "llama-3.3-70b-versatile",
            "llama-3.1-70b-versatile",
            "llama-3.1-8b-instant",
            "mixtral-8x7b-32768",
            "gemma2-9b-it",
        ],
        supports_streaming=True,
        supports_embeddings=False,
        supports_function_calling=True,
        supports_vision=False,
        rate_limits={"rpm": 30, "tpm": 14400},
        default_params={"temperature": 0.7, "max_tokens": 8192},
        litellm_prefix="groq/",
    ),
    
    "deepseek": ProviderConfig(
        name="deepseek",
        display_name="DeepSeek",
        provider_type=ProviderType.OPEN_SOURCE,
        api_key_env="DEEPSEEK_API_KEY",
        base_url="https://api.deepseek.com/v1",
        default_model="deepseek-chat",
        supported_models=[
            "deepseek-chat",
            "deepseek-coder",
        ],
        supports_streaming=True,
        supports_embeddings=False,
        supports_function_calling=True,
        supports_vision=False,
        rate_limits={"rpm": 300, "tpm": 100000},
        default_params={"temperature": 0.7, "max_tokens": 4096},
        litellm_prefix="deepseek/",
    ),
    
    # =========================================================================
    # SELF-HOSTED PROVIDERS
    # =========================================================================
    
    "vllm": ProviderConfig(
        name="vllm",
        display_name="VLLM (Self-Hosted)",
        provider_type=ProviderType.SELF_HOSTED,
        api_key_env="LLM_API_KEY",
        base_url=None,  # User must provide
        default_model="",
        supported_models=[],  # Depends on deployment
        supports_streaming=True,
        supports_embeddings=True,
        supports_function_calling=True,
        supports_vision=True,
        rate_limits={},  # No external limits
        default_params={"temperature": 0.7, "max_tokens": 4096},
        litellm_prefix="openai/",  # VLLM uses OpenAI-compatible API
    ),
    
    "ollama": ProviderConfig(
        name="ollama",
        display_name="Ollama (Local)",
        provider_type=ProviderType.SELF_HOSTED,
        api_key_env="",
        base_url="http://localhost:11434",
        default_model="llama3.2",
        supported_models=[
            "llama3.2",
            "llama3.1",
            "mistral",
            "codellama",
            "phi3",
            "gemma2",
            "qwen2.5",
        ],
        supports_streaming=True,
        supports_embeddings=True,
        supports_function_calling=False,
        supports_vision=True,
        rate_limits={},
        default_params={"temperature": 0.7, "max_tokens": 4096},
        litellm_prefix="ollama/",
    ),
    
    "lmstudio": ProviderConfig(
        name="lmstudio",
        display_name="LM Studio (Local)",
        provider_type=ProviderType.SELF_HOSTED,
        api_key_env="",
        base_url="http://localhost:1234/v1",
        default_model="local-model",
        supported_models=[],
        supports_streaming=True,
        supports_embeddings=True,
        supports_function_calling=False,
        supports_vision=False,
        rate_limits={},
        default_params={"temperature": 0.7, "max_tokens": 4096},
        litellm_prefix="openai/",
    ),
    
    "openrouter": ProviderConfig(
        name="openrouter",
        display_name="OpenRouter",
        provider_type=ProviderType.OPEN_SOURCE,
        api_key_env="OPENROUTER_API_KEY",
        base_url="https://openrouter.ai/api/v1",
        default_model="anthropic/claude-3.5-sonnet",
        supported_models=[
            "anthropic/claude-3.5-sonnet",
            "openai/gpt-4o",
            "google/gemini-pro-1.5",
            "meta-llama/llama-3.1-405b-instruct",
        ],
        supports_streaming=True,
        supports_embeddings=False,
        supports_function_calling=True,
        supports_vision=True,
        rate_limits={"rpm": 200, "tpm": 100000},
        default_params={"temperature": 0.7, "max_tokens": 4096},
        litellm_prefix="openrouter/",
    ),
}


# =============================================================================
# PROVIDER REGISTRY CLASS
# =============================================================================

class ProviderRegistry:
    """
    Registry for managing LLM provider configurations.
    
    Provides methods to:
        - Get provider configuration by name
        - List all available providers
        - Filter providers by type
        - Check feature support
        - Add custom provider configurations
    
    Example:
        >>> registry = ProviderRegistry()
        >>> config = registry.get_provider("openai")
        >>> print(config.display_name)  # "OpenAI"
        
        >>> # List all self-hosted providers
        >>> self_hosted = registry.get_providers_by_type(ProviderType.SELF_HOSTED)
    """
    
    def __init__(self):
        """Initialize registry with default provider configurations."""
        self._providers: Dict[str, ProviderConfig] = PROVIDER_CONFIGS.copy()
    
    def get_provider(self, name: str) -> Optional[ProviderConfig]:
        """
        Get provider configuration by name.
        
        Args:
            name: Provider identifier (case-insensitive)
            
        Returns:
            ProviderConfig or None if not found
        """
        return self._providers.get(name.lower())
    
    def list_providers(self) -> List[str]:
        """
        List all available provider names.
        
        Returns:
            List of provider identifiers
        """
        return list(self._providers.keys())
    
    def get_providers_by_type(
        self, 
        provider_type: ProviderType
    ) -> List[ProviderConfig]:
        """
        Get all providers of a specific type.
        
        Args:
            provider_type: Type of provider to filter by
            
        Returns:
            List of matching ProviderConfig objects
        """
        return [
            config for config in self._providers.values()
            if config.provider_type == provider_type
        ]
    
    def get_streaming_providers(self) -> List[ProviderConfig]:
        """Get all providers that support streaming."""
        return [
            config for config in self._providers.values()
            if config.supports_streaming
        ]
    
    def get_embedding_providers(self) -> List[ProviderConfig]:
        """Get all providers that support embeddings."""
        return [
            config for config in self._providers.values()
            if config.supports_embeddings
        ]
    
    def get_vision_providers(self) -> List[ProviderConfig]:
        """Get all providers that support vision/image input."""
        return [
            config for config in self._providers.values()
            if config.supports_vision
        ]
    
    def register_provider(self, config: ProviderConfig) -> None:
        """
        Register a custom provider configuration.
        
        Args:
            config: ProviderConfig to register
            
        Example:
            >>> custom = ProviderConfig(
            ...     name="custom-llm",
            ...     display_name="My Custom LLM",
            ...     provider_type=ProviderType.SELF_HOSTED,
            ...     api_key_env="CUSTOM_API_KEY",
            ...     base_url="http://my-server:8000/v1",
            ...     default_model="my-model"
            ... )
            >>> registry.register_provider(custom)
        """
        self._providers[config.name.lower()] = config
    
    def get_model_provider(self, model_string: str) -> Optional[str]:
        """
        Detect provider from model string.
        
        Args:
            model_string: Model identifier (may include provider prefix)
            
        Returns:
            Provider name or None if cannot be determined
            
        Example:
            >>> registry.get_model_provider("anthropic/claude-3-5-sonnet")
            "anthropic"
            >>> registry.get_model_provider("gpt-4o")
            "openai"
        """
        # Check for explicit provider prefix
        if "/" in model_string:
            potential_provider = model_string.split("/")[0]
            if potential_provider in self._providers:
                return potential_provider
        
        # Check if model belongs to known provider
        model_lower = model_string.lower()
        
        # OpenAI models
        if any(m in model_lower for m in ["gpt-", "o1-", "davinci", "curie"]):
            return "openai"
        
        # Anthropic models
        if "claude" in model_lower:
            return "anthropic"
        
        # Gemini models
        if "gemini" in model_lower:
            return "gemini"
        
        # Mistral models
        if "mistral" in model_lower or "mixtral" in model_lower:
            return "mistral"
        
        # Llama models (could be various providers)
        if "llama" in model_lower:
            return "together"  # Default to Together for Llama
        
        return None
