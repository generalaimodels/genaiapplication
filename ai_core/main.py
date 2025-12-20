# =============================================================================
# CHATBOT CORE AI - Unified API Entry Point
# =============================================================================
# The main public interface for the Advanced Chatbot Core AI system.
# Provides a simple, unified API for interacting with 100+ LLM providers.
# =============================================================================
#
# SUPPORTED PROVIDERS:
#   - OpenAI (GPT-4, GPT-3.5, o1)
#   - Anthropic (Claude 3.5, Claude 3)
#   - Google (Gemini 1.5 Pro, Gemini Flash)
#   - Azure OpenAI
#   - AWS Bedrock
#   - Google Vertex AI
#   - VLLM (self-hosted)
#   - Ollama (local)
#   - Together AI
#   - Mistral AI
#   - Cohere
#   - Groq
#   - DeepSeek
#   - And 100+ more via LiteLLM
#
# USAGE EXAMPLES:
#
#   # 1. OpenAI
#   chatbot = ChatbotCoreAI(model="gpt-4o", api_key="sk-...")
#   response = chatbot.chat("What is Python?")
#
#   # 2. VLLM Self-Hosted
#   chatbot = ChatbotCoreAI(
#       model="my-model",
#       base_url="http://10.180.93.12:8007/v1",
#       api_key="EMPTY"
#   )
#
#   # 3. Gemini
#   chatbot = ChatbotCoreAI(
#       model="gemini-1.5-pro",
#       provider="gemini",
#       api_key="AIza..."
#   )
#
#   # 4. Streaming
#   for chunk in chatbot.stream_chat("Explain quantum physics"):
#       print(chunk.content, end="", flush=True)
#
#   # 5. Document-Augmented (RAG)
#   response = chatbot.chat_with_documents(
#       query="Summarize the key points",
#       documents=["report.pdf", "data.txt"]
#   )
#
# =============================================================================

from __future__ import annotations
import asyncio
from typing import (
    List, Dict, Any, Optional, Union, AsyncIterator, Iterator, Callable
)
from pathlib import Path
import logging

from core.chatbot_engine import ChatbotEngine
from clients.base_client import ChatMessage, ChatResponse, ChatChunk
from session.session_manager import Session
from config.settings import Settings
from config.providers import ProviderRegistry

logger = logging.getLogger(__name__)


class ChatbotCoreAI:
    """
    ==========================================================================
    CHATBOT CORE AI - Advanced AI Chatbot Interface
    ==========================================================================
    
    The main entry point for the Advanced Chatbot Core AI system.
    Provides a simple, unified API for 100+ LLM providers.
    
    Features:
        ✓ Multi-Provider Support (OpenAI, Anthropic, Gemini, VLLM, Ollama, etc.)
        ✓ Session Management with History
        ✓ Document-Augmented Responses (RAG)
        ✓ Streaming Responses
        ✓ Batch Processing
        ✓ Feedback Collection
        ✓ Production-Ready Error Handling
    
    Quick Start:
        >>> from AI_core import ChatbotCoreAI
        >>> 
        >>> # Initialize with any provider
        >>> chatbot = ChatbotCoreAI(model="gpt-4o", api_key="sk-...")
        >>> 
        >>> # Chat
        >>> response = chatbot.chat("What is machine learning?")
        >>> print(response.content)
    
    Provider Examples:
        >>> # OpenAI
        >>> chatbot = ChatbotCoreAI(model="gpt-4o", api_key="sk-...")
        
        >>> # VLLM (self-hosted)
        >>> chatbot = ChatbotCoreAI(
        ...     model="my-model",
        ...     base_url="http://10.180.93.12:8007/v1",
        ...     api_key="EMPTY"
        ... )
        
        >>> # Gemini
        >>> chatbot = ChatbotCoreAI(
        ...     model="gemini-1.5-pro",
        ...     provider="gemini",
        ...     api_key="AIzaSyB..."
        ... )
        
        >>> # Anthropic Claude
        >>> chatbot = ChatbotCoreAI(
        ...     model="claude-3-5-sonnet-20241022",
        ...     provider="anthropic",
        ...     api_key="sk-ant-..."
        ... )
        
        >>> # Ollama (local)
        >>> chatbot = ChatbotCoreAI(
        ...     model="llama3.2",
        ...     provider="ollama",
        ...     base_url="http://localhost:11434"
        ... )
    ==========================================================================
    """
    
    def __init__(
        self,
        model: Optional[str] = None,
        provider: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        timeout: int = 60,
        max_retries: int = 3,
        enable_history: bool = True,
        max_history_messages: int = 50,
        **kwargs
    ):
        """
        Initialize ChatbotCoreAI.
        
        Args:
            model: Model identifier (e.g., "gpt-4o", "claude-3-5-sonnet-20241022")
            provider: Provider name ("openai", "anthropic", "gemini", "vllm", etc.)
            api_key: API key for the provider
            base_url: Custom API endpoint (required for VLLM/self-hosted)
            system_prompt: Default system prompt
            temperature: Default sampling temperature (0.0-2.0)
            max_tokens: Default maximum response tokens
            timeout: Request timeout in seconds
            max_retries: Maximum retry attempts
            enable_history: Enable conversation history
            max_history_messages: Maximum history messages per session
            **kwargs: Additional provider-specific options
        
        Environment Variables:
            - LLM_BASE_URL: Custom endpoint URL
            - LLM_API_KEY: Generic API key
            - OPENAI_API_KEY: OpenAI API key
            - ANTHROPIC_API_KEY: Anthropic API key
            - GEMINI_API_KEY: Gemini API key
            - And more...
        
        Example:
            >>> # Minimal - uses environment variables
            >>> chatbot = ChatbotCoreAI()
            
            >>> # OpenAI
            >>> chatbot = ChatbotCoreAI(model="gpt-4o", api_key="sk-...")
            
            >>> # VLLM
            >>> chatbot = ChatbotCoreAI(
            ...     model="my-model",
            ...     base_url="http://10.180.93.12:8007/v1",
            ...     api_key="EMPTY"
            ... )
        """
        # Build settings
        settings = Settings(
            model=model or "gpt-4o-mini",
            provider=provider or "openai",
            api_key=api_key,
            base_url=base_url,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            retry_count=max_retries,
            enable_history=enable_history,
            max_history_messages=max_history_messages,
        )
        
        # Initialize engine
        self._engine = ChatbotEngine(
            settings=settings,
            system_prompt=system_prompt,
            **kwargs
        )
        
        self._provider_registry = ProviderRegistry()
        
        logger.info(
            f"ChatbotCoreAI initialized: model={model}, provider={provider}"
        )
    
    # =========================================================================
    # CORE CHAT METHODS
    # =========================================================================
    
    def chat(
        self,
        query: str,
        *,
        session_id: Optional[str] = None,
        system_prompt: Optional[str] = None,
        include_history: bool = True,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        stop: Optional[List[str]] = None,
        **kwargs
    ) -> ChatResponse:
        """
        Generate a chat response.
        
        This is the primary method for interacting with the AI.
        
        Args:
            query: User message/question
            session_id: Session ID for conversation history (optional)
            system_prompt: System prompt override
            include_history: Include previous messages from session
            temperature: Sampling temperature (0.0-2.0)
            max_tokens: Maximum response tokens
            top_p: Nucleus sampling parameter
            stop: Stop sequences
            **kwargs: Additional options
            
        Returns:
            ChatResponse with content, usage stats, and metadata
            
        Example:
            >>> # Simple query
            >>> response = chatbot.chat("What is Python?")
            >>> print(response.content)
            
            >>> # With session for multi-turn conversation
            >>> r1 = chatbot.chat("I'm learning programming", session_id="user-123")
            >>> r2 = chatbot.chat("What language should I start with?", session_id="user-123")
            
            >>> # Custom parameters
            >>> response = chatbot.chat(
            ...     "Write a creative story",
            ...     temperature=0.9,
            ...     max_tokens=2000
            ... )
        """
        return self._engine.chat(
            query=query,
            session_id=session_id,
            system_prompt=system_prompt,
            include_history=include_history,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            stop=stop,
            **kwargs
        )
    
    async def achat(
        self,
        query: str,
        *,
        session_id: Optional[str] = None,
        system_prompt: Optional[str] = None,
        include_history: bool = True,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> ChatResponse:
        """
        Async chat completion.
        
        Args:
            query: User message
            session_id: Session ID
            system_prompt: System prompt
            include_history: Include history
            temperature: Temperature
            max_tokens: Max tokens
            **kwargs: Additional options
            
        Returns:
            ChatResponse
            
        Example:
            >>> async def main():
            ...     response = await chatbot.achat("Hello!")
            ...     print(response.content)
        """
        return await self._engine.achat(
            query=query,
            session_id=session_id,
            system_prompt=system_prompt,
            include_history=include_history,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )
    
    # =========================================================================
    # DOCUMENT-AUGMENTED CHAT (RAG)
    # =========================================================================
    
    def chat_with_documents(
        self,
        query: str,
        documents: Union[str, Path, List[str], List[Path]],
        *,
        session_id: Optional[str] = None,
        system_prompt: Optional[str] = None,
        top_k: int = 5,
        include_sources: bool = True,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> ChatResponse:
        """
        Chat with document context (RAG - Retrieval Augmented Generation).
        
        Processes documents and includes relevant content in the context
        for more informed responses.
        
        Args:
            query: User question
            documents: Document(s) - file paths or text content
            session_id: Session ID
            system_prompt: System prompt
            top_k: Number of relevant chunks to include
            include_sources: Include source citations
            temperature: Temperature
            max_tokens: Max tokens
            **kwargs: Additional options
            
        Returns:
            ChatResponse with document-informed answer
            
        Example:
            >>> # From files
            >>> response = chatbot.chat_with_documents(
            ...     query="What are the main findings?",
            ...     documents=["report.txt", "analysis.md"]
            ... )
            
            >>> # From text
            >>> response = chatbot.chat_with_documents(
            ...     query="Summarize this",
            ...     documents=["Long document text here..."]
            ... )
        """
        # Normalize documents to list
        if isinstance(documents, (str, Path)):
            documents = [documents]
        
        return self._engine.chat_with_documents(
            query=query,
            documents=documents,
            session_id=session_id,
            system_prompt=system_prompt,
            top_k=top_k,
            include_sources=include_sources,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )
    
    # =========================================================================
    # STREAMING
    # =========================================================================
    
    def stream_chat(
        self,
        query: str,
        *,
        session_id: Optional[str] = None,
        system_prompt: Optional[str] = None,
        include_history: bool = True,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        on_chunk: Optional[Callable[[str], None]] = None,
        **kwargs
    ) -> Iterator[ChatChunk]:
        """
        Stream chat response in real-time.
        
        Yields response chunks as they're generated for
        real-time display.
        
        Args:
            query: User message
            session_id: Session ID
            system_prompt: System prompt
            include_history: Include history
            temperature: Temperature
            max_tokens: Max tokens
            on_chunk: Callback for each chunk
            **kwargs: Additional options
            
        Yields:
            ChatChunk objects with content
            
        Example:
            >>> # Basic streaming
            >>> for chunk in chatbot.stream_chat("Explain AI"):
            ...     print(chunk.content, end="", flush=True)
            >>> print()
            
            >>> # With callback
            >>> def print_chunk(text):
            ...     print(text, end="", flush=True)
            >>> 
            >>> list(chatbot.stream_chat("Hello", on_chunk=print_chunk))
        """
        return self._engine.stream_chat(
            query=query,
            session_id=session_id,
            system_prompt=system_prompt,
            include_history=include_history,
            temperature=temperature,
            max_tokens=max_tokens,
            on_chunk=on_chunk,
            **kwargs
        )
    
    async def astream_chat(
        self,
        query: str,
        *,
        session_id: Optional[str] = None,
        system_prompt: Optional[str] = None,
        include_history: bool = True,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> AsyncIterator[ChatChunk]:
        """
        Async streaming chat.
        
        Args:
            query: User message
            session_id: Session ID
            system_prompt: System prompt
            include_history: Include history
            temperature: Temperature
            max_tokens: Max tokens
            **kwargs: Additional options
            
        Yields:
            ChatChunk objects
            
        Example:
            >>> async for chunk in chatbot.astream_chat("Hello"):
            ...     print(chunk.content, end="", flush=True)
        """
        async for chunk in self._engine.astream_chat(
            query=query,
            session_id=session_id,
            system_prompt=system_prompt,
            include_history=include_history,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        ):
            yield chunk
    
    # =========================================================================
    # BATCH PROCESSING
    # =========================================================================
    
    def batch_chat(
        self,
        queries: List[str],
        *,
        session_id: Optional[str] = None,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> List[ChatResponse]:
        """
        Process multiple queries sequentially.
        
        Args:
            queries: List of queries
            session_id: Shared session ID
            system_prompt: System prompt
            temperature: Temperature
            max_tokens: Max tokens
            **kwargs: Additional options
            
        Returns:
            List of ChatResponse objects
            
        Example:
            >>> queries = ["What is Python?", "What is Java?", "What is Rust?"]
            >>> responses = chatbot.batch_chat(queries)
            >>> for r in responses:
            ...     print(r.content[:100])
        """
        return self._engine.batch_chat(
            queries=queries,
            session_id=session_id,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )
    
    async def abatch_chat(
        self,
        queries: List[str],
        *,
        max_concurrent: int = 5,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> List[ChatResponse]:
        """
        Process multiple queries concurrently.
        
        Args:
            queries: List of queries
            max_concurrent: Maximum concurrent requests
            temperature: Temperature
            max_tokens: Max tokens
            **kwargs: Additional options
            
        Returns:
            List of ChatResponse objects
            
        Example:
            >>> responses = await chatbot.abatch_chat(
            ...     ["Q1", "Q2", "Q3"],
            ...     max_concurrent=3
            ... )
        """
        return await self._engine.abatch_chat(
            queries=queries,
            max_concurrent=max_concurrent,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )
    
    # =========================================================================
    # SESSION MANAGEMENT
    # =========================================================================
    
    def create_session(
        self,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        system_prompt: Optional[str] = None,
        **metadata
    ) -> Session:
        """
        Create a new conversation session.
        
        Args:
            user_id: User identifier
            session_id: Custom session ID (auto-generated if not provided)
            system_prompt: Session-specific system prompt
            **metadata: Additional session data
            
        Returns:
            Session object
            
        Example:
            >>> session = chatbot.create_session(
            ...     user_id="user-123",
            ...     system_prompt="You are a helpful coding assistant"
            ... )
            >>> print(session.session_id)
        """
        return self._engine.create_session(
            user_id=user_id,
            session_id=session_id,
            system_prompt=system_prompt,
            **metadata
        )
    
    def get_session(self, session_id: str) -> Optional[Session]:
        """Get session by ID."""
        return self._engine.get_session(session_id)
    
    def clear_session(
        self,
        session_id: str,
        keep_system: bool = True
    ) -> None:
        """
        Clear session history.
        
        Args:
            session_id: Session ID
            keep_system: Keep system prompt
        """
        self._engine.clear_session(session_id, keep_system)
    
    def delete_session(self, session_id: str) -> bool:
        """Delete a session."""
        return self._engine.delete_session(session_id)
    
    def get_history(
        self,
        session_id: str,
        n: Optional[int] = None
    ) -> List[ChatMessage]:
        """
        Get conversation history.
        
        Args:
            session_id: Session ID
            n: Number of recent messages (None = all)
            
        Returns:
            List of ChatMessage objects
        """
        return self._engine.get_history(session_id, n)
    
    # =========================================================================
    # FEEDBACK
    # =========================================================================
    
    def submit_feedback(
        self,
        session_id: str,
        feedback_type: str,
        value: Any = None,
        message_id: Optional[str] = None,
        comment: Optional[str] = None
    ) -> bool:
        """
        Submit user feedback.
        
        Args:
            session_id: Session ID
            feedback_type: "thumbs_up", "thumbs_down", "rating", "text"
            value: Feedback value (bool, int 1-5, or text)
            message_id: Specific message ID
            comment: Optional comment
            
        Returns:
            True if submitted successfully
            
        Example:
            >>> chatbot.submit_feedback(
            ...     session_id="session-123",
            ...     feedback_type="thumbs_up"
            ... )
            
            >>> chatbot.submit_feedback(
            ...     session_id="session-123",
            ...     feedback_type="rating",
            ...     value=5,
            ...     comment="Great response!"
            ... )
        """
        return self._engine.submit_feedback(
            session_id=session_id,
            feedback_type=feedback_type,
            value=value,
            message_id=message_id,
            comment=comment
        )
    
    # =========================================================================
    # UTILITIES
    # =========================================================================
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        return self._engine.count_tokens(text)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        return self._engine.get_model_info()
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """Get usage statistics."""
        return self._engine.get_usage_stats()
    
    @staticmethod
    def list_providers() -> List[str]:
        """List all available providers."""
        return ProviderRegistry().list_providers()
    
    @staticmethod
    def get_provider_models(provider: str) -> List[str]:
        """Get recommended models for a provider."""
        registry = ProviderRegistry()
        config = registry.get_provider(provider)
        return config.supported_models if config else []


# =============================================================================
# CONVENIENCE FUNCTION
# =============================================================================

def create_chatbot(
    model: str,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    provider: Optional[str] = None,
    **kwargs
) -> ChatbotCoreAI:
    """
    Convenience function to create a chatbot instance.
    
    Args:
        model: Model identifier
        api_key: API key
        base_url: Custom endpoint
        provider: Provider name
        **kwargs: Additional options
        
    Returns:
        ChatbotCoreAI instance
        
    Example:
        >>> from AI_core.main import create_chatbot
        >>> 
        >>> chatbot = create_chatbot("gpt-4o", api_key="sk-...")
        >>> response = chatbot.chat("Hello!")
    """
    return ChatbotCoreAI(
        model=model,
        api_key=api_key,
        base_url=base_url,
        provider=provider,
        **kwargs
    )


# =============================================================================
# EXAMPLE USAGE (For Testing Only - Remove in Production)
# =============================================================================
# NOTE: Never hardcode API keys in production code. Use environment variables.
# Example: export GEMINI_API_KEY="your-api-key-here"
# =============================================================================

if __name__ == "__main__":
    # Example: Gemini Provider
    # Supported models: gemini-1.5-pro, gemini-1.5-flash, gemini-1.5-flash-8b
    chat_reponse = ChatbotCoreAI(
        model="gemini-1.5-flash",  # Use a valid model name
        provider="gemini",
        api_key="AIzaSyAKzcuZ_0XkWQySV8XsM1hZHdMqaoTf4Vs",  # Replace with your key or use env var
        temperature=0.7,
    )
    
    response = chat_reponse.chat("What is Python programming language? Answer in 2 sentences.")
    print(response.content)
