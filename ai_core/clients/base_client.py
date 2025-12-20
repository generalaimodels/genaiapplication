# =============================================================================
# BASE LLM CLIENT - Abstract Interface for All LLM Providers
# =============================================================================
# Defines the contract that all LLM client implementations must follow.
# Supports synchronous and asynchronous operations, streaming, batching,
# embeddings, and comprehensive error handling.
# =============================================================================

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import (
    List, 
    Dict, 
    Any, 
    Optional, 
    AsyncIterator, 
    Iterator,
    Union,
    Callable,
    TypeVar,
    Generic
)
from dataclasses import dataclass, field
from datetime import datetime
import asyncio


# =============================================================================
# TYPE DEFINITIONS
# =============================================================================

MessageRole = str  # "system" | "user" | "assistant" | "function" | "tool"
T = TypeVar("T")


# =============================================================================
# DATA CLASSES FOR CLIENT OPERATIONS
# =============================================================================

@dataclass
class ChatMessage:
    """
    Represents a single message in a conversation.
    
    Attributes:
        role: Message role (system, user, assistant, function, tool)
        content: Message content (text or structured)
        name: Optional name for function/tool messages
        function_call: Optional function call data
        tool_calls: Optional tool call data
        metadata: Additional message metadata
    """
    role: MessageRole
    content: Union[str, List[Dict[str, Any]], None]
    name: Optional[str] = None
    function_call: Optional[Dict[str, Any]] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary for API calls."""
        message = {"role": self.role, "content": self.content}
        if self.name:
            message["name"] = self.name
        if self.function_call:
            message["function_call"] = self.function_call
        if self.tool_calls:
            message["tool_calls"] = self.tool_calls
        return message


@dataclass
class ChatResponse:
    """
    Response from a chat completion request.
    
    Attributes:
        content: Generated text content
        role: Response role (usually "assistant")
        model: Model that generated the response
        finish_reason: Why generation stopped (stop, length, etc.)
        usage: Token usage statistics
        function_call: Function call if generated
        tool_calls: Tool calls if generated
        response_id: Unique response identifier
        created_at: Response timestamp
        metadata: Additional response metadata
    """
    content: str
    role: MessageRole = "assistant"
    model: Optional[str] = None
    finish_reason: Optional[str] = None
    usage: Optional[Dict[str, int]] = None
    function_call: Optional[Dict[str, Any]] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    response_id: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def total_tokens(self) -> int:
        """Get total tokens used."""
        if self.usage:
            return self.usage.get("total_tokens", 0)
        return 0
    
    @property
    def prompt_tokens(self) -> int:
        """Get prompt tokens used."""
        if self.usage:
            return self.usage.get("prompt_tokens", 0)
        return 0
    
    @property
    def completion_tokens(self) -> int:
        """Get completion tokens used."""
        if self.usage:
            return self.usage.get("completion_tokens", 0)
        return 0


@dataclass
class ChatChunk:
    """
    Streaming chunk from a chat completion.
    
    Attributes:
        content: Chunk text content (delta)
        role: Role (usually only in first chunk)
        finish_reason: Set in final chunk
        chunk_index: Position in stream
        metadata: Additional chunk metadata
    """
    content: str
    role: Optional[MessageRole] = None
    finish_reason: Optional[str] = None
    chunk_index: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def is_final(self) -> bool:
        """Check if this is the final chunk."""
        return self.finish_reason is not None


@dataclass
class EmbeddingResponse:
    """
    Response from an embedding request.
    
    Attributes:
        embeddings: List of embedding vectors
        model: Model used for embeddings
        usage: Token usage statistics
    """
    embeddings: List[List[float]]
    model: Optional[str] = None
    usage: Optional[Dict[str, int]] = None


@dataclass  
class CompletionResponse:
    """
    Response from a text completion request.
    
    Attributes:
        text: Generated text
        model: Model that generated the response
        finish_reason: Why generation stopped
        usage: Token usage statistics
        logprobs: Log probabilities if requested
    """
    text: str
    model: Optional[str] = None
    finish_reason: Optional[str] = None
    usage: Optional[Dict[str, int]] = None
    logprobs: Optional[Dict[str, Any]] = None


# =============================================================================
# GENERATION PARAMETERS
# =============================================================================

@dataclass
class GenerationParams:
    """
    Parameters controlling text generation.
    
    Attributes:
        temperature: Sampling temperature (0.0-2.0)
        max_tokens: Maximum tokens to generate
        top_p: Nucleus sampling parameter
        top_k: Top-k sampling parameter
        frequency_penalty: Frequency penalty
        presence_penalty: Presence penalty
        stop: Stop sequences
        n: Number of completions to generate
        logprobs: Whether to return log probabilities
        echo: Whether to echo the prompt
        seed: Random seed for reproducibility
        response_format: Desired response format
    """
    temperature: float = 0.7
    max_tokens: int = 4096
    top_p: float = 1.0
    top_k: Optional[int] = None
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    stop: Optional[List[str]] = None
    n: int = 1
    logprobs: Optional[bool] = None
    echo: bool = False
    seed: Optional[int] = None
    response_format: Optional[Dict[str, str]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary, excluding None values."""
        params = {}
        for key, value in self.__dict__.items():
            if value is not None:
                params[key] = value
        return params


# =============================================================================
# ABSTRACT BASE CLIENT
# =============================================================================

class BaseLLMClient(ABC):
    """
    ==========================================================================
    BASE LLM CLIENT - Abstract Interface for All Providers
    ==========================================================================
    
    This abstract class defines the contract for all LLM client implementations.
    Every provider (OpenAI, Anthropic, VLLM, etc.) must implement these methods.
    
    Features:
        - Synchronous and asynchronous chat/completion
        - Streaming support for real-time responses
        - Batch processing for multiple queries
        - Embeddings generation
        - Token counting
        - Comprehensive error handling
    
    Implementation Notes:
        - All methods should handle provider-specific exceptions
        - Implementations should support retry logic
        - Streaming methods must yield proper chunk objects
        - Batch methods should use concurrent processing where possible
    
    Example Implementation:
        >>> class MyClient(BaseLLMClient):
        ...     def chat(self, messages, **kwargs):
        ...         # Implementation here
        ...         return ChatResponse(content="Hello!")
    ==========================================================================
    """
    
    # =========================================================================
    # INITIALIZATION
    # =========================================================================
    
    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: int = 60,
        max_retries: int = 3,
        **kwargs
    ):
        """
        Initialize the LLM client.
        
        Args:
            model: Model identifier
            api_key: API key for authentication
            base_url: Custom API endpoint URL
            timeout: Request timeout in seconds
            max_retries: Maximum retry attempts
            **kwargs: Additional provider-specific options
        """
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.timeout = timeout
        self.max_retries = max_retries
        self.extra_options = kwargs
    
    # =========================================================================
    # ABSTRACT METHODS - Must be implemented by subclasses
    # =========================================================================
    
    @abstractmethod
    def chat(
        self,
        messages: List[ChatMessage],
        params: Optional[GenerationParams] = None,
        **kwargs
    ) -> ChatResponse:
        """
        Generate a chat completion.
        
        Args:
            messages: Conversation messages
            params: Generation parameters
            **kwargs: Additional options
            
        Returns:
            ChatResponse with generated content
            
        Raises:
            LLMError: On API errors
            AuthenticationError: On auth failures
            RateLimitError: On rate limit exceeded
        """
        pass
    
    @abstractmethod
    async def achat(
        self,
        messages: List[ChatMessage],
        params: Optional[GenerationParams] = None,
        **kwargs
    ) -> ChatResponse:
        """
        Async version of chat completion.
        
        Args:
            messages: Conversation messages
            params: Generation parameters
            **kwargs: Additional options
            
        Returns:
            ChatResponse with generated content
        """
        pass
    
    @abstractmethod
    def stream_chat(
        self,
        messages: List[ChatMessage],
        params: Optional[GenerationParams] = None,
        **kwargs
    ) -> Iterator[ChatChunk]:
        """
        Stream a chat completion.
        
        Args:
            messages: Conversation messages
            params: Generation parameters
            **kwargs: Additional options
            
        Yields:
            ChatChunk objects with streaming content
        """
        pass
    
    @abstractmethod
    async def astream_chat(
        self,
        messages: List[ChatMessage],
        params: Optional[GenerationParams] = None,
        **kwargs
    ) -> AsyncIterator[ChatChunk]:
        """
        Async stream a chat completion.
        
        Args:
            messages: Conversation messages
            params: Generation parameters
            **kwargs: Additional options
            
        Yields:
            ChatChunk objects with streaming content
        """
        pass
    
    @abstractmethod
    def complete(
        self,
        prompt: str,
        params: Optional[GenerationParams] = None,
        **kwargs
    ) -> CompletionResponse:
        """
        Generate a text completion.
        
        Args:
            prompt: Input prompt
            params: Generation parameters
            **kwargs: Additional options
            
        Returns:
            CompletionResponse with generated text
        """
        pass
    
    @abstractmethod
    async def acomplete(
        self,
        prompt: str,
        params: Optional[GenerationParams] = None,
        **kwargs
    ) -> CompletionResponse:
        """
        Async version of text completion.
        
        Args:
            prompt: Input prompt
            params: Generation parameters
            **kwargs: Additional options
            
        Returns:
            CompletionResponse with generated text
        """
        pass
    
    @abstractmethod
    def get_embeddings(
        self,
        texts: List[str],
        model: Optional[str] = None,
        **kwargs
    ) -> EmbeddingResponse:
        """
        Generate embeddings for texts.
        
        Args:
            texts: List of texts to embed
            model: Embedding model (if different from chat model)
            **kwargs: Additional options
            
        Returns:
            EmbeddingResponse with embedding vectors
        """
        pass
    
    @abstractmethod
    async def aget_embeddings(
        self,
        texts: List[str],
        model: Optional[str] = None,
        **kwargs
    ) -> EmbeddingResponse:
        """
        Async version of embeddings generation.
        
        Args:
            texts: List of texts to embed
            model: Embedding model (if different from chat model)
            **kwargs: Additional options
            
        Returns:
            EmbeddingResponse with embedding vectors
        """
        pass
    
    # =========================================================================
    # BATCH PROCESSING - Default implementations using concurrent execution
    # =========================================================================
    
    def batch_chat(
        self,
        message_batches: List[List[ChatMessage]],
        params: Optional[GenerationParams] = None,
        max_concurrent: int = 5,
        **kwargs
    ) -> List[ChatResponse]:
        """
        Process multiple chat requests in batch.
        
        Default implementation runs sequentially.
        Subclasses may override for parallel processing.
        
        Args:
            message_batches: List of message lists
            params: Generation parameters (shared)
            max_concurrent: Maximum concurrent requests
            **kwargs: Additional options
            
        Returns:
            List of ChatResponse objects
        """
        responses = []
        for messages in message_batches:
            response = self.chat(messages, params, **kwargs)
            responses.append(response)
        return responses
    
    async def abatch_chat(
        self,
        message_batches: List[List[ChatMessage]],
        params: Optional[GenerationParams] = None,
        max_concurrent: int = 5,
        **kwargs
    ) -> List[ChatResponse]:
        """
        Async batch chat processing with concurrency control.
        
        Args:
            message_batches: List of message lists
            params: Generation parameters (shared)
            max_concurrent: Maximum concurrent requests
            **kwargs: Additional options
            
        Returns:
            List of ChatResponse objects
        """
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_with_semaphore(messages: List[ChatMessage]) -> ChatResponse:
            async with semaphore:
                return await self.achat(messages, params, **kwargs)
        
        tasks = [process_with_semaphore(msgs) for msgs in message_batches]
        return await asyncio.gather(*tasks)
    
    # =========================================================================
    # UTILITY METHODS
    # =========================================================================
    
    def count_tokens(self, text: str) -> int:
        """
        Count tokens in text.
        
        Default implementation provides rough estimate.
        Subclasses should override with provider-specific tokenizer.
        
        Args:
            text: Text to count tokens for
            
        Returns:
            Estimated token count
        """
        # Rough estimate: ~4 characters per token
        return len(text) // 4
    
    def count_message_tokens(self, messages: List[ChatMessage]) -> int:
        """
        Count tokens in a list of messages.
        
        Args:
            messages: List of chat messages
            
        Returns:
            Total estimated token count
        """
        total = 0
        for msg in messages:
            if isinstance(msg.content, str):
                total += self.count_tokens(msg.content)
            elif isinstance(msg.content, list):
                for part in msg.content:
                    if isinstance(part, dict) and "text" in part:
                        total += self.count_tokens(part["text"])
        return total
    
    def is_available(self) -> bool:
        """
        Check if the client is available and properly configured.
        
        Returns:
            True if client can make API calls
        """
        return self.api_key is not None or self.base_url is not None
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the configured model.
        
        Returns:
            Dictionary with model information
        """
        return {
            "model": self.model,
            "base_url": self.base_url,
            "timeout": self.timeout,
            "max_retries": self.max_retries,
        }
    
    # =========================================================================
    # CONTEXT MANAGER SUPPORT
    # =========================================================================
    
    def __enter__(self) -> "BaseLLMClient":
        """Enter context manager."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit context manager."""
        pass
    
    async def __aenter__(self) -> "BaseLLMClient":
        """Enter async context manager."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit async context manager."""
        pass
