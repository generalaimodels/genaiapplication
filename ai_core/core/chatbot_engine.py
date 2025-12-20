# =============================================================================
# CHATBOT ENGINE - Main Orchestration Engine
# =============================================================================
# The core engine that orchestrates all chatbot operations including
# query processing, session management, document augmentation, and responses.
# =============================================================================

from __future__ import annotations
import asyncio
import uuid
from typing import (
    List, Dict, Any, Optional, Union, AsyncIterator, Iterator, Callable
)
from pathlib import Path
import logging

from clients.base_client import (
    ChatMessage, ChatResponse, ChatChunk, GenerationParams
)
from clients.litellm_client import LiteLLMClient
from clients.factory import create_client
from session.session_manager import SessionManager, Session
from session.history_manager import HistoryManager
from session.feedback_collector import FeedbackCollector, FeedbackType
from documents.document_processor import DocumentProcessor, Document
from documents.context_builder import ContextBuilder
from core.response_handler import ResponseHandler, ProcessedResponse
from core.stream_handler import StreamHandler, StreamResult
from config.settings import Settings

logger = logging.getLogger(__name__)


class ChatbotEngine:
    """
    ==========================================================================
    CHATBOT ENGINE - Core Orchestration for AI Chatbot
    ==========================================================================
    
    The main engine that coordinates all chatbot functionality:
        - Query processing with session management
        - Document-augmented responses (RAG)
        - Streaming and batch operations
        - History tracking and feedback collection
    
    Supports 100+ LLM providers via LiteLLM including:
        - OpenAI, Anthropic, Google Gemini
        - Azure OpenAI, AWS Bedrock
        - VLLM (self-hosted), Ollama (local)
        - Together AI, Mistral, Cohere, and more
    
    Example:
        >>> # Initialize with OpenAI
        >>> engine = ChatbotEngine(
        ...     model="gpt-4o",
        ...     api_key="sk-..."
        ... )
        >>> 
        >>> # Simple query
        >>> response = engine.chat("What is Python?")
        >>> print(response.content)
        
        >>> # With session
        >>> response = engine.chat(
        ...     "What is Python?",
        ...     session_id="user-123"
        ... )
        
        >>> # VLLM self-hosted
        >>> engine = ChatbotEngine(
        ...     model="my-model",
        ...     base_url="http://10.180.93.12:8007/v1",
        ...     api_key="EMPTY"
        ... )
    ==========================================================================
    """
    
    def __init__(
        self,
        model: Optional[str] = None,
        provider: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        settings: Optional[Settings] = None,
        system_prompt: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize the chatbot engine.
        
        Args:
            model: Model identifier (e.g., "gpt-4o", "claude-3-5-sonnet")
            provider: Provider name (auto-detected if not specified)
            api_key: API key for the provider
            base_url: Custom API endpoint (for VLLM/self-hosted)
            settings: Optional Settings object for full configuration
            system_prompt: Default system prompt
            **kwargs: Additional options passed to LLM client
        
        Example:
            >>> # OpenAI
            >>> engine = ChatbotEngine(model="gpt-4o", api_key="sk-...")
            
            >>> # VLLM
            >>> engine = ChatbotEngine(
            ...     model="my-model",
            ...     base_url="http://10.180.93.12:8007/v1",
            ...     api_key="EMPTY"
            ... )
            
            >>> # Gemini
            >>> engine = ChatbotEngine(
            ...     model="gemini-1.5-pro",
            ...     provider="gemini",
            ...     api_key="AIza..."
            ... )
            
            >>> # From settings
            >>> settings = Settings(provider="anthropic", model="claude-3-5-sonnet")
            >>> engine = ChatbotEngine(settings=settings)
        """
        # Load from settings or parameters
        if settings:
            self.settings = settings
        else:
            self.settings = Settings(
                model=model or "gpt-4o-mini",
                provider=provider or "openai",
                api_key=api_key,
                base_url=base_url,
                **{k: v for k, v in kwargs.items() if hasattr(Settings, k)}
            )
        
        # Override with direct parameters if provided
        if model:
            self.settings.model = model
        if api_key:
            self.settings.api_key = api_key
        if base_url:
            self.settings.base_url = base_url
        
        # Initialize LLM client
        self.client = create_client(
            model=self.settings.model,
            provider=provider or self.settings.provider,
            api_key=self.settings.get_provider_api_key(),
            base_url=self.settings.base_url,
            timeout=self.settings.timeout,
            max_retries=self.settings.retry_count,
            **{k: v for k, v in kwargs.items() if not hasattr(Settings, k)}
        )
        
        # Initialize components
        self.session_manager = SessionManager(
            default_ttl=self.settings.session_ttl,
            auto_cleanup=True
        )
        
        self.history_manager = HistoryManager(
            max_messages_per_session=self.settings.max_history_messages,
            default_system_prompt=system_prompt
        )
        
        self.feedback_collector = FeedbackCollector()
        
        self.document_processor = DocumentProcessor(
            chunk_size=self.settings.chunk_size,
            chunk_overlap=self.settings.chunk_overlap
        )
        
        self.context_builder = ContextBuilder(
            max_context_tokens=self.settings.max_tokens // 2,
            max_chunks=self.settings.max_context_chunks,
            min_relevance=self.settings.relevance_threshold
        )
        
        self.response_handler = ResponseHandler()
        self.stream_handler = StreamHandler()
        
        # Default system prompt
        self.default_system_prompt = system_prompt
        
        logger.info(f"ChatbotEngine initialized with model: {self.settings.model}")
    
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
        max_history_messages: Optional[int] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        stop: Optional[List[str]] = None,
        response_format: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> ChatResponse:
        """
        Generate a chat response.
        
        Args:
            query: User query/message
            session_id: Session identifier for history (optional)
            system_prompt: System prompt override
            include_history: Include conversation history
            max_history_messages: Limit history messages
            temperature: Sampling temperature (0.0-2.0)
            max_tokens: Maximum response tokens
            top_p: Nucleus sampling parameter
            stop: Stop sequences
            response_format: Response format ("text", "json")
            metadata: Additional metadata
            **kwargs: Extra options passed to LLM
            
        Returns:
            ChatResponse with generated content
            
        Example:
            >>> response = engine.chat("What is machine learning?")
            >>> print(response.content)
            
            >>> # With session
            >>> response = engine.chat(
            ...     "Tell me more",
            ...     session_id="user-123",
            ...     temperature=0.8
            ... )
        """
        # Build messages
        messages = self._build_messages(
            query=query,
            session_id=session_id,
            system_prompt=system_prompt or self.default_system_prompt,
            include_history=include_history,
            max_history=max_history_messages
        )
        
        # Build generation params
        params = self._build_params(
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            stop=stop
        )
        
        # Call LLM
        response = self.client.chat(messages, params, **kwargs)
        
        # Update history if session exists
        if session_id:
            self.history_manager.add_user_message(session_id, query)
            self.history_manager.add_assistant_message(session_id, response.content)
        
        return response
    
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
            query: User query
            session_id: Session identifier
            system_prompt: System prompt override
            include_history: Include history
            temperature: Sampling temperature
            max_tokens: Max tokens
            **kwargs: Additional options
            
        Returns:
            ChatResponse
        """
        messages = self._build_messages(
            query=query,
            session_id=session_id,
            system_prompt=system_prompt or self.default_system_prompt,
            include_history=include_history
        )
        
        params = self._build_params(
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        response = await self.client.achat(messages, params, **kwargs)
        
        if session_id:
            self.history_manager.add_user_message(session_id, query)
            self.history_manager.add_assistant_message(session_id, response.content)
        
        return response
    
    # =========================================================================
    # DOCUMENT-AUGMENTED CHAT (RAG)
    # =========================================================================
    
    def chat_with_documents(
        self,
        query: str,
        documents: Union[List[str], List[Path], List[Document]],
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
        Chat with document context (RAG).
        
        Args:
            query: User query
            documents: Documents (file paths, text, or Document objects)
            session_id: Session identifier
            system_prompt: System prompt override
            top_k: Number of relevant chunks to include
            include_sources: Include source citations
            temperature: Sampling temperature
            max_tokens: Max tokens
            **kwargs: Additional options
            
        Returns:
            ChatResponse with document-informed answer
            
        Example:
            >>> response = engine.chat_with_documents(
            ...     query="Summarize the main points",
            ...     documents=["report.txt", "data.csv"],
            ...     top_k=5
            ... )
        """
        # Process documents
        doc_objects = self._process_documents(documents)
        
        # Build context
        context = self.context_builder.build_context(
            query=query,
            documents=doc_objects,
            top_k=top_k,
            include_sources=include_sources
        )
        
        # Build augmented query
        if context.has_context:
            augmented_query = f"{context.context_text}\nQuestion: {query}"
        else:
            augmented_query = query
        
        # Build messages
        messages = self._build_messages(
            query=augmented_query,
            session_id=session_id,
            system_prompt=system_prompt or self._get_rag_system_prompt(),
            include_history=True
        )
        
        params = self._build_params(
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        response = self.client.chat(messages, params, **kwargs)
        
        # Add metadata about context
        response.metadata["context_chunks"] = context.total_chunks
        response.metadata["context_tokens"] = context.token_estimate
        
        if session_id:
            self.history_manager.add_user_message(session_id, query)
            self.history_manager.add_assistant_message(session_id, response.content)
        
        return response
    
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
        Stream chat response.
        
        Args:
            query: User query
            session_id: Session identifier
            system_prompt: System prompt override
            include_history: Include history
            temperature: Sampling temperature
            max_tokens: Max tokens
            on_chunk: Callback for each chunk
            **kwargs: Additional options
            
        Yields:
            ChatChunk objects
            
        Example:
            >>> for chunk in engine.stream_chat("Explain quantum physics"):
            ...     print(chunk.content, end="", flush=True)
        """
        messages = self._build_messages(
            query=query,
            session_id=session_id,
            system_prompt=system_prompt or self.default_system_prompt,
            include_history=include_history
        )
        
        params = self._build_params(
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        content_parts = []
        
        for chunk in self.client.stream_chat(messages, params, **kwargs):
            if chunk.content:
                content_parts.append(chunk.content)
                if on_chunk:
                    on_chunk(chunk.content)
            yield chunk
        
        # Update history after stream completes
        if session_id and content_parts:
            full_content = "".join(content_parts)
            self.history_manager.add_user_message(session_id, query)
            self.history_manager.add_assistant_message(session_id, full_content)
    
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
        Async stream chat response.
        
        Args:
            query: User query
            session_id: Session identifier
            system_prompt: System prompt
            include_history: Include history
            temperature: Temperature
            max_tokens: Max tokens
            **kwargs: Additional options
            
        Yields:
            ChatChunk objects
        """
        messages = self._build_messages(
            query=query,
            session_id=session_id,
            system_prompt=system_prompt or self.default_system_prompt,
            include_history=include_history
        )
        
        params = self._build_params(
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        content_parts = []
        
        async for chunk in self.client.astream_chat(messages, params, **kwargs):
            if chunk.content:
                content_parts.append(chunk.content)
            yield chunk
        
        if session_id and content_parts:
            full_content = "".join(content_parts)
            self.history_manager.add_user_message(session_id, query)
            self.history_manager.add_assistant_message(session_id, full_content)
    
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
        """
        responses = []
        for query in queries:
            response = self.chat(
                query,
                session_id=session_id,
                system_prompt=system_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )
            responses.append(response)
        return responses
    
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
            max_concurrent: Max concurrent requests
            temperature: Temperature
            max_tokens: Max tokens
            **kwargs: Additional options
            
        Returns:
            List of ChatResponse objects
        """
        params = self._build_params(
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        message_batches = [
            [ChatMessage(role="user", content=q)]
            for q in queries
        ]
        
        return await self.client.abatch_chat(
            message_batches,
            params=params,
            max_concurrent=max_concurrent,
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
        Create a new session.
        
        Args:
            user_id: User identifier
            session_id: Custom session ID
            system_prompt: Session system prompt
            **metadata: Additional metadata
            
        Returns:
            Created Session object
        """
        session = self.session_manager.create_session(
            user_id=user_id,
            session_id=session_id,
            model=self.settings.model,
            system_prompt=system_prompt,
            **metadata
        )
        
        # Set system prompt in history
        if system_prompt:
            self.history_manager.set_system_prompt(session.session_id, system_prompt)
        
        return session
    
    def get_session(self, session_id: str) -> Optional[Session]:
        """Get session by ID."""
        return self.session_manager.get_session(session_id)
    
    def clear_session(self, session_id: str, keep_system: bool = True) -> None:
        """Clear session history."""
        self.history_manager.clear_history(session_id, keep_system)
    
    def delete_session(self, session_id: str) -> bool:
        """Delete a session."""
        self.history_manager.delete_history(session_id)
        return self.session_manager.delete_session(session_id)
    
    def get_history(
        self,
        session_id: str,
        n: Optional[int] = None
    ) -> List[ChatMessage]:
        """Get session history."""
        return self.history_manager.get_messages(session_id, n)
    
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
        Submit feedback for a session/message.
        
        Args:
            session_id: Session identifier
            feedback_type: "thumbs_up", "thumbs_down", "rating", "text"
            value: Feedback value
            message_id: Specific message ID
            comment: Optional comment
            
        Returns:
            True if submitted successfully
        """
        try:
            fb_type = FeedbackType(feedback_type)
            self.feedback_collector.submit_feedback(
                session_id=session_id,
                feedback_type=fb_type,
                value=value,
                message_id=message_id,
                comment=comment
            )
            return True
        except Exception as e:
            logger.error(f"Feedback error: {e}")
            return False
    
    # =========================================================================
    # HELPER METHODS
    # =========================================================================
    
    def _build_messages(
        self,
        query: str,
        session_id: Optional[str],
        system_prompt: Optional[str],
        include_history: bool,
        max_history: Optional[int] = None
    ) -> List[ChatMessage]:
        """Build message list for API call."""
        messages = []
        
        # Add system prompt
        if system_prompt:
            messages.append(ChatMessage(role="system", content=system_prompt))
        
        # Add history if session exists
        if session_id and include_history:
            history = self.history_manager.get_messages(
                session_id, 
                n=max_history,
                include_system=False  # Already added above
            )
            messages.extend(history)
        
        # Add current query
        messages.append(ChatMessage(role="user", content=query))
        
        return messages
    
    def _build_params(
        self,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        stop: Optional[List[str]] = None
    ) -> GenerationParams:
        """Build generation parameters."""
        return GenerationParams(
            temperature=temperature or self.settings.temperature,
            max_tokens=max_tokens or self.settings.max_tokens,
            top_p=top_p or self.settings.top_p,
            stop=stop or self.settings.stop_sequences
        )
    
    def _process_documents(
        self,
        documents: Union[List[str], List[Path], List[Document]]
    ) -> List[Document]:
        """Process documents into chunks."""
        if not documents:
            return []
        
        # Check if already Document objects
        if isinstance(documents[0], Document):
            return documents
        
        # Process from files/text
        return self.document_processor.process_content(documents, auto_chunk=True)
    
    def _get_rag_system_prompt(self) -> str:
        """Get system prompt for RAG."""
        return (
            "You are a helpful assistant. Answer questions based on the provided context. "
            "If the answer is not in the context, say you don't have enough information. "
            "Always cite your sources when possible."
        )
    
    # =========================================================================
    # UTILITIES
    # =========================================================================
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        return self.client.count_tokens(text)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        return self.client.get_model_info()
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """Get usage statistics."""
        return self.client.get_usage_stats()
