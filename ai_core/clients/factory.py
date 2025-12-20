# =============================================================================
# CLIENT FACTORY - Auto-Detection and Creation of LLM Clients
# =============================================================================
# Factory pattern for creating appropriate LLM client based on provider/model.
# =============================================================================

from __future__ import annotations
from typing import Optional, Dict, Any, Type
import logging

from clients.base_client import BaseLLMClient
from clients.litellm_client import LiteLLMClient
from config.providers import ProviderRegistry, PROVIDER_CONFIGS

logger = logging.getLogger(__name__)


class LLMClientFactory:
    """
    Factory for creating LLM clients with automatic provider detection.
    
    Features:
        - Auto-detects provider from model name
        - Configures client with appropriate settings
        - Supports custom client registration
        - Handles VLLM/self-hosted endpoints
    
    Example:
        >>> # Auto-detect from model
        >>> client = LLMClientFactory.create("gpt-4o", api_key="sk-...")
        
        >>> # VLLM self-hosted
        >>> client = LLMClientFactory.create(
        ...     model="my-model",
        ...     base_url="http://10.180.93.12:8007/v1",
        ...     api_key="EMPTY"
        ... )
        
        >>> # Explicit provider
        >>> client = LLMClientFactory.create(
        ...     model="claude-3-5-sonnet-20241022",
        ...     provider="anthropic",
        ...     api_key="sk-ant-..."
        ... )
    """
    
    # Registry of client implementations
    _client_registry: Dict[str, Type[BaseLLMClient]] = {
        "default": LiteLLMClient,
        "litellm": LiteLLMClient,
    }
    
    # Provider registry instance
    _provider_registry = ProviderRegistry()
    
    @classmethod
    def create(
        cls,
        model: str,
        provider: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        client_type: str = "litellm",
        **kwargs
    ) -> BaseLLMClient:
        """
        Create an LLM client with automatic configuration.
        
        Args:
            model: Model identifier (e.g., "gpt-4o", "Claude-3-5-sonnet")
            provider: Provider name (auto-detected if not specified)
            api_key: API key for authentication
            base_url: Custom API endpoint (for VLLM/self-hosted)
            client_type: Client implementation to use
            **kwargs: Additional client options
            
        Returns:
            Configured LLM client instance
            
        Raises:
            ValueError: If client type is unknown
        """
        # Get client class
        client_cls = cls._client_registry.get(client_type)
        if not client_cls:
            raise ValueError(f"Unknown client type: {client_type}")
        
        # Auto-detect provider if not specified
        if not provider and not base_url:
            provider = cls._provider_registry.get_model_provider(model)
        
        # Get provider config for additional settings
        if provider:
            provider_config = cls._provider_registry.get_provider(provider)
            if provider_config:
                # =========================================================================
                # IMPORTANT: Only set base_url for SELF-HOSTED providers (VLLM, Ollama, etc.)
                # For cloud providers (Gemini, Anthropic, etc.), LiteLLM handles routing
                # internally based on the model prefix - no base_url needed.
                # =========================================================================
                from config.providers import ProviderType
                if provider_config.provider_type == ProviderType.SELF_HOSTED:
                    if not base_url and provider_config.base_url:
                        base_url = provider_config.base_url
                
                # Format model with provider prefix if needed
                if provider_config.litellm_prefix:
                    if not model.startswith(provider_config.litellm_prefix):
                        model = f"{provider_config.litellm_prefix}{model}"
        
        # Handle VLLM/self-hosted endpoints
        if base_url and not provider:
            provider = "vllm"
            kwargs.setdefault("custom_llm_provider", "openai")
        
        logger.info(f"Creating {client_type} client for model: {model}")
        
        return client_cls(
            model=model,
            api_key=api_key,
            base_url=base_url,
            **kwargs
        )
    
    @classmethod
    def register_client(
        cls,
        name: str,
        client_class: Type[BaseLLMClient]
    ) -> None:
        """
        Register a custom client implementation.
        
        Args:
            name: Client type name
            client_class: Client class implementing BaseLLMClient
        """
        cls._client_registry[name] = client_class
        logger.info(f"Registered client type: {name}")
    
    @classmethod
    def list_client_types(cls) -> list:
        """List available client types."""
        return list(cls._client_registry.keys())
    
    @classmethod
    def list_providers(cls) -> list:
        """List available providers."""
        return cls._provider_registry.list_providers()


def create_client(
    model: str,
    provider: Optional[str] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    **kwargs
) -> BaseLLMClient:
    """
    Convenience function to create an LLM client.
    
    Args:
        model: Model identifier
        provider: Provider name (optional)
        api_key: API key
        base_url: Custom endpoint (for VLLM)
        **kwargs: Additional options
        
    Returns:
        Configured LLM client
        
    Example:
        >>> from AI_core.clients import create_client
        >>> 
        >>> # OpenAI
        >>> client = create_client("gpt-4o", api_key="sk-...")
        >>> 
        >>> # VLLM
        >>> client = create_client(
        ...     model="my-model",
        ...     base_url="http://10.180.93.12:8007/v1",
        ...     api_key="EMPTY"
        ... )
        >>> 
        >>> # Gemini
        >>> client = create_client(
        ...     model="gemini-1.5-pro",
        ...     provider="gemini",
        ...     api_key="AIza..."
        ... )
    """
    return LLMClientFactory.create(
        model=model,
        provider=provider,
        api_key=api_key,
        base_url=base_url,
        **kwargs
    )
