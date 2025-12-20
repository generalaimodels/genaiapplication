# =============================================================================
# LITELLM CLIENT - Unified Interface for 100+ LLM Providers
# =============================================================================
# Production-grade LLM client using LiteLLM library.
# Supports OpenAI, Anthropic, Gemini, Azure, VLLM, Ollama, and 100+ more.
# =============================================================================

from __future__ import annotations
import os
import asyncio
import logging
from typing import List, Dict, Any, Optional, AsyncIterator, Iterator
from datetime import datetime

import litellm
from litellm import completion, acompletion, embedding, aembedding, token_counter
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from clients.base_client import (
    BaseLLMClient, ChatMessage, ChatResponse, ChatChunk,
    CompletionResponse, EmbeddingResponse, GenerationParams,
)

logger = logging.getLogger(__name__)


class LiteLLMClient(BaseLLMClient):
    """
    Unified LLM Client supporting 100+ providers via LiteLLM.
    
    Providers: OpenAI, Anthropic, Gemini, Azure, VLLM, Ollama, Together AI, etc.
    
    Example:
        >>> client = LiteLLMClient(model="gpt-4o", api_key="sk-...")
        >>> response = client.chat([ChatMessage(role="user", content="Hello!")])
        
        >>> # VLLM self-hosted
        >>> client = LiteLLMClient(
        ...     model="my-model",
        ...     base_url="http://10.180.93.12:8007/v1",
        ...     api_key="EMPTY"
        ... )
    """
    
    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: int = 60,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        enable_cost_tracking: bool = True,
        fallback_models: Optional[List[str]] = None,
        custom_llm_provider: Optional[str] = None,
        **kwargs
    ):
        super().__init__(model, api_key, base_url, timeout, max_retries, **kwargs)
        
        self.retry_delay = retry_delay
        self.enable_cost_tracking = enable_cost_tracking
        self.fallback_models = fallback_models or []
        self.custom_llm_provider = custom_llm_provider
        self._usage_stats = {"total_requests": 0, "total_tokens": 0, "total_cost": 0.0, "errors": 0}
        
        self._configure_litellm()
    
    def _configure_litellm(self) -> None:
        """Configure LiteLLM global settings."""
        if self.api_key:
            # =========================================================================
            # PROVIDER DETECTION: Priority order for determining the provider
            # 1. Explicit custom_llm_provider (set via kwargs)
            # 2. Model prefix (e.g., "gemini/model-name", "anthropic/claude-3")
            # 3. Model name patterns (e.g., "gpt-4", "claude-3")
            # =========================================================================
            model_lower = self.model.lower()
            provider = self.custom_llm_provider.lower() if self.custom_llm_provider else None
            
            # Extract provider from model prefix if present (e.g., "gemini/model-name")
            if not provider and "/" in model_lower:
                prefix = model_lower.split("/")[0]
                if prefix in ("gemini", "anthropic", "mistral", "cohere", "groq", 
                              "together_ai", "deepseek", "huggingface", "bedrock",
                              "vertex_ai", "azure", "openrouter"):
                    provider = prefix
            
            # Set appropriate environment variable based on detected provider
            if provider == "gemini":
                os.environ["GEMINI_API_KEY"] = self.api_key
            elif provider == "anthropic":
                os.environ["ANTHROPIC_API_KEY"] = self.api_key
            elif provider == "mistral":
                os.environ["MISTRAL_API_KEY"] = self.api_key
            elif provider == "cohere":
                os.environ["COHERE_API_KEY"] = self.api_key
            elif provider == "groq":
                os.environ["GROQ_API_KEY"] = self.api_key
            elif provider == "together_ai":
                os.environ["TOGETHER_API_KEY"] = self.api_key
            elif provider == "deepseek":
                os.environ["DEEPSEEK_API_KEY"] = self.api_key
            elif provider == "huggingface":
                os.environ["HUGGINGFACE_API_KEY"] = self.api_key
            elif provider == "openrouter":
                os.environ["OPENROUTER_API_KEY"] = self.api_key
            # Fallback: detect from model name patterns
            elif "claude" in model_lower or "anthropic" in model_lower:
                os.environ["ANTHROPIC_API_KEY"] = self.api_key
            elif "gemini" in model_lower:
                os.environ["GEMINI_API_KEY"] = self.api_key
            elif "mistral" in model_lower or "mixtral" in model_lower:
                os.environ["MISTRAL_API_KEY"] = self.api_key
            elif "command" in model_lower:  # Cohere models start with "command"
                os.environ["COHERE_API_KEY"] = self.api_key
            else:
                # Default to OpenAI for unknown providers
                os.environ["OPENAI_API_KEY"] = self.api_key
        
        litellm.set_verbose = False
        litellm.drop_params = True
    
    def _get_model_string(self) -> str:
        """
        Get properly formatted model string for LiteLLM.
        
        LiteLLM uses provider prefixes to route requests:
            - gemini/gemini-1.5-flash -> Google Gemini API
            - anthropic/claude-3 -> Anthropic API
            - openai/model -> OpenAI-compatible endpoints (VLLM, Ollama, etc.)
        """
        model = self.model
        model_lower = model.lower()
        
        # =========================================================================
        # CLOUD PROVIDER PREFIXES - These should NOT be overridden
        # =========================================================================
        cloud_prefixes = (
            "gemini/", "anthropic/", "mistral/", "cohere/", "groq/",
            "together_ai/", "deepseek/", "huggingface/", "bedrock/",
            "vertex_ai/", "azure/", "openrouter/"
        )
        
        # If model already has a cloud provider prefix, return as-is
        if any(model_lower.startswith(prefix) for prefix in cloud_prefixes):
            return model
        
        # If model already has provider prefix (including openai/), return as-is
        if "/" in model:
            return model
        
        # For custom base_url (VLLM/self-hosted), use openai/ prefix
        if self.base_url:
            return f"openai/{model}"
        
        # Auto-detect and add provider prefix
        model_lower = model.lower()
        
        # Gemini models
        if "gemini" in model_lower:
            return f"gemini/{model}"
        
        # Claude models
        if "claude" in model_lower:
            return f"anthropic/{model}"
        
        # Mistral models
        if "mistral" in model_lower or "mixtral" in model_lower:
            return f"mistral/{model}"
        
        # Cohere models
        if "command" in model_lower:
            return f"cohere/{model}"
        
        # OpenAI models (default) - no prefix needed
        return model
    
    def _prepare_messages(self, messages: List[ChatMessage]) -> List[Dict[str, Any]]:
        """Convert ChatMessage objects to API format."""
        return [msg.to_dict() for msg in messages]
    
    def _build_completion_kwargs(
        self, messages: List[Dict], params: Optional[GenerationParams] = None,
        stream: bool = False, **kwargs
    ) -> Dict[str, Any]:
        """Build kwargs for LiteLLM completion call."""
        completion_kwargs = {
            "model": self._get_model_string(),
            "messages": messages,
            "stream": stream,
            "timeout": self.timeout,
        }
        
        # For custom base_url (VLLM/self-hosted)
        if self.base_url:
            completion_kwargs["api_base"] = self.base_url
            completion_kwargs["custom_llm_provider"] = self.custom_llm_provider or "openai"
        
        # Always pass api_key for the call
        if self.api_key:
            completion_kwargs["api_key"] = self.api_key
        
        if params:
            completion_kwargs.update(params.to_dict())
        
        completion_kwargs.update(kwargs)
        return completion_kwargs

    
    def _parse_response(self, response: Any) -> ChatResponse:
        """Parse LiteLLM response to ChatResponse."""
        choice = response.choices[0]
        message = choice.message
        
        return ChatResponse(
            content=message.content or "",
            role=message.role,
            model=response.model,
            finish_reason=choice.finish_reason,
            usage={
                "prompt_tokens": getattr(response.usage, "prompt_tokens", 0),
                "completion_tokens": getattr(response.usage, "completion_tokens", 0),
                "total_tokens": getattr(response.usage, "total_tokens", 0),
            } if response.usage else None,
            function_call=getattr(message, "function_call", None),
            tool_calls=getattr(message, "tool_calls", None),
            response_id=response.id,
        )
    
    def _parse_stream_chunk(self, chunk: Any, index: int) -> ChatChunk:
        """Parse streaming chunk to ChatChunk."""
        delta = chunk.choices[0].delta
        return ChatChunk(
            content=delta.content or "",
            role=getattr(delta, "role", None),
            finish_reason=chunk.choices[0].finish_reason,
            chunk_index=index,
        )
    
    def chat(
        self, messages: List[ChatMessage],
        params: Optional[GenerationParams] = None, **kwargs
    ) -> ChatResponse:
        """Generate a chat completion."""
        @retry(
            stop=stop_after_attempt(self.max_retries),
            wait=wait_exponential(multiplier=self.retry_delay, max=60)
        )
        def _chat_with_retry():
            formatted = self._prepare_messages(messages)
            completion_kwargs = self._build_completion_kwargs(formatted, params, False, **kwargs)
            response = completion(**completion_kwargs)
            return self._parse_response(response)
        
        try:
            return _chat_with_retry()
        except Exception as e:
            self._usage_stats["errors"] += 1
            for fallback in self.fallback_models:
                try:
                    original = self.model
                    self.model = fallback
                    result = _chat_with_retry()
                    self.model = original
                    return result
                except Exception:
                    continue
            raise
    
    async def achat(
        self, messages: List[ChatMessage],
        params: Optional[GenerationParams] = None, **kwargs
    ) -> ChatResponse:
        """Async chat completion."""
        formatted = self._prepare_messages(messages)
        completion_kwargs = self._build_completion_kwargs(formatted, params, False, **kwargs)
        response = await acompletion(**completion_kwargs)
        return self._parse_response(response)
    
    def stream_chat(
        self, messages: List[ChatMessage],
        params: Optional[GenerationParams] = None, **kwargs
    ) -> Iterator[ChatChunk]:
        """Stream a chat completion."""
        formatted = self._prepare_messages(messages)
        completion_kwargs = self._build_completion_kwargs(formatted, params, True, **kwargs)
        response = completion(**completion_kwargs)
        
        for i, chunk in enumerate(response):
            if chunk.choices and chunk.choices[0].delta:
                yield self._parse_stream_chunk(chunk, i)
    
    async def astream_chat(
        self, messages: List[ChatMessage],
        params: Optional[GenerationParams] = None, **kwargs
    ) -> AsyncIterator[ChatChunk]:
        """Async stream a chat completion."""
        formatted = self._prepare_messages(messages)
        completion_kwargs = self._build_completion_kwargs(formatted, params, True, **kwargs)
        response = await acompletion(**completion_kwargs)
        
        i = 0
        async for chunk in response:
            if chunk.choices and chunk.choices[0].delta:
                yield self._parse_stream_chunk(chunk, i)
                i += 1
    
    def complete(
        self, prompt: str,
        params: Optional[GenerationParams] = None, **kwargs
    ) -> CompletionResponse:
        """Generate a text completion."""
        messages = [ChatMessage(role="user", content=prompt)]
        response = self.chat(messages, params, **kwargs)
        return CompletionResponse(
            text=response.content, model=response.model,
            finish_reason=response.finish_reason, usage=response.usage
        )
    
    async def acomplete(
        self, prompt: str,
        params: Optional[GenerationParams] = None, **kwargs
    ) -> CompletionResponse:
        """Async text completion."""
        messages = [ChatMessage(role="user", content=prompt)]
        response = await self.achat(messages, params, **kwargs)
        return CompletionResponse(
            text=response.content, model=response.model,
            finish_reason=response.finish_reason, usage=response.usage
        )
    
    def get_embeddings(
        self, texts: List[str],
        model: Optional[str] = None, **kwargs
    ) -> EmbeddingResponse:
        """Generate embeddings for texts."""
        embed_model = model or "text-embedding-3-small"
        embed_kwargs = {"model": embed_model, "input": texts}
        
        if self.api_key:
            embed_kwargs["api_key"] = self.api_key
        if self.base_url:
            embed_kwargs["api_base"] = self.base_url
        
        response = embedding(**embed_kwargs)
        vectors = [item["embedding"] for item in response.data]
        
        return EmbeddingResponse(
            embeddings=vectors,
            model=getattr(response, "model", embed_model),
            usage={"prompt_tokens": response.usage.prompt_tokens,
                   "total_tokens": response.usage.total_tokens} if hasattr(response, "usage") else None
        )
    
    async def aget_embeddings(
        self, texts: List[str],
        model: Optional[str] = None, **kwargs
    ) -> EmbeddingResponse:
        """Async embeddings generation."""
        embed_model = model or "text-embedding-3-small"
        embed_kwargs = {"model": embed_model, "input": texts}
        
        if self.api_key:
            embed_kwargs["api_key"] = self.api_key
        if self.base_url:
            embed_kwargs["api_base"] = self.base_url
        
        response = await aembedding(**embed_kwargs)
        vectors = [item["embedding"] for item in response.data]
        
        return EmbeddingResponse(embeddings=vectors, model=embed_model)
    
    async def abatch_chat(
        self, message_batches: List[List[ChatMessage]],
        params: Optional[GenerationParams] = None,
        max_concurrent: int = 5, **kwargs
    ) -> List[ChatResponse]:
        """Process multiple chat requests concurrently."""
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_single(messages: List[ChatMessage]) -> ChatResponse:
            async with semaphore:
                return await self.achat(messages, params, **kwargs)
        
        tasks = [process_single(msgs) for msgs in message_batches]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        responses = []
        for result in results:
            if isinstance(result, Exception):
                responses.append(ChatResponse(content=f"Error: {result}", metadata={"error": True}))
            else:
                responses.append(result)
        return responses
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        try:
            return token_counter(model=self.model, text=text)
        except Exception:
            return len(text) // 4
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """Get usage statistics."""
        return self._usage_stats.copy()
