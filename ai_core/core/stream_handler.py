# =============================================================================
# STREAM HANDLER - Streaming Response Management
# =============================================================================
# Handles streaming responses with buffering and callbacks.
# =============================================================================

from __future__ import annotations
import asyncio
from typing import AsyncIterator, Iterator, Callable, Optional, List, Any
from dataclasses import dataclass, field
import logging

from clients.base_client import ChatChunk

logger = logging.getLogger(__name__)


@dataclass
class StreamResult:
    """
    Result from completed stream.
    
    Attributes:
        content: Full accumulated content
        chunks: All chunks received
        chunk_count: Total chunks
        finish_reason: Final finish reason
    """
    content: str
    chunks: List[ChatChunk] = field(default_factory=list)
    chunk_count: int = 0
    finish_reason: Optional[str] = None
    
    @property
    def is_complete(self) -> bool:
        """Check if stream completed normally."""
        return self.finish_reason == "stop"


class StreamHandler:
    """
    Handles streaming responses with callbacks.
    
    Features:
        - Content accumulation
        - Chunk callbacks
        - Buffered output
        - Stream state tracking
    
    Example:
        >>> handler = StreamHandler(on_chunk=print)
        >>> 
        >>> # Process stream
        >>> result = handler.process_stream(chunk_iterator)
        >>> print(f"Total: {result.content}")
        
        >>> # Async version
        >>> async for text in handler.stream_text(async_iterator):
        ...     print(text, end="", flush=True)
    """
    
    def __init__(
        self,
        on_chunk: Optional[Callable[[str], None]] = None,
        on_complete: Optional[Callable[[StreamResult], None]] = None,
        buffer_size: int = 0
    ):
        """
        Initialize stream handler.
        
        Args:
            on_chunk: Callback for each chunk
            on_complete: Callback on stream completion
            buffer_size: Buffer chunks before callback (0 = immediate)
        """
        self.on_chunk = on_chunk
        self.on_complete = on_complete
        self.buffer_size = buffer_size
    
    def process_stream(
        self,
        stream: Iterator[ChatChunk]
    ) -> StreamResult:
        """
        Process a synchronous stream.
        
        Args:
            stream: Iterator of ChatChunk objects
            
        Returns:
            StreamResult with accumulated content
        """
        content_parts = []
        chunks = []
        buffer = []
        finish_reason = None
        
        for chunk in stream:
            chunks.append(chunk)
            
            if chunk.content:
                content_parts.append(chunk.content)
                buffer.append(chunk.content)
                
                # Flush buffer if needed
                if self.buffer_size == 0 or len(buffer) >= self.buffer_size:
                    if self.on_chunk:
                        self.on_chunk("".join(buffer))
                    buffer = []
            
            if chunk.finish_reason:
                finish_reason = chunk.finish_reason
        
        # Flush remaining buffer
        if buffer and self.on_chunk:
            self.on_chunk("".join(buffer))
        
        result = StreamResult(
            content="".join(content_parts),
            chunks=chunks,
            chunk_count=len(chunks),
            finish_reason=finish_reason
        )
        
        if self.on_complete:
            self.on_complete(result)
        
        return result
    
    async def aprocess_stream(
        self,
        stream: AsyncIterator[ChatChunk]
    ) -> StreamResult:
        """
        Process an async stream.
        
        Args:
            stream: Async iterator of ChatChunk objects
            
        Returns:
            StreamResult with accumulated content
        """
        content_parts = []
        chunks = []
        buffer = []
        finish_reason = None
        
        async for chunk in stream:
            chunks.append(chunk)
            
            if chunk.content:
                content_parts.append(chunk.content)
                buffer.append(chunk.content)
                
                if self.buffer_size == 0 or len(buffer) >= self.buffer_size:
                    if self.on_chunk:
                        self.on_chunk("".join(buffer))
                    buffer = []
            
            if chunk.finish_reason:
                finish_reason = chunk.finish_reason
        
        if buffer and self.on_chunk:
            self.on_chunk("".join(buffer))
        
        result = StreamResult(
            content="".join(content_parts),
            chunks=chunks,
            chunk_count=len(chunks),
            finish_reason=finish_reason
        )
        
        if self.on_complete:
            self.on_complete(result)
        
        return result
    
    def stream_text(
        self,
        stream: Iterator[ChatChunk]
    ) -> Iterator[str]:
        """
        Yield text content from stream.
        
        Args:
            stream: Iterator of ChatChunk objects
            
        Yields:
            Text content strings
        """
        for chunk in stream:
            if chunk.content:
                yield chunk.content
    
    async def astream_text(
        self,
        stream: AsyncIterator[ChatChunk]
    ) -> AsyncIterator[str]:
        """
        Async yield text content from stream.
        
        Args:
            stream: Async iterator of ChatChunk objects
            
        Yields:
            Text content strings
        """
        async for chunk in stream:
            if chunk.content:
                yield chunk.content
    
    def collect_stream(
        self,
        stream: Iterator[ChatChunk]
    ) -> str:
        """
        Collect stream into single string.
        
        Args:
            stream: Iterator of ChatChunk objects
            
        Returns:
            Complete accumulated content
        """
        return "".join(self.stream_text(stream))
    
    async def acollect_stream(
        self,
        stream: AsyncIterator[ChatChunk]
    ) -> str:
        """
        Async collect stream into single string.
        
        Args:
            stream: Async iterator of ChatChunk objects
            
        Returns:
            Complete accumulated content
        """
        parts = []
        async for text in self.astream_text(stream):
            parts.append(text)
        return "".join(parts)


def stream_to_print(stream: Iterator[ChatChunk]) -> str:
    """
    Stream content to stdout and return final text.
    
    Args:
        stream: Iterator of ChatChunk objects
        
    Returns:
        Complete content
    """
    handler = StreamHandler(on_chunk=lambda x: print(x, end="", flush=True))
    result = handler.process_stream(stream)
    print()  # Final newline
    return result.content


async def astream_to_print(stream: AsyncIterator[ChatChunk]) -> str:
    """
    Async stream content to stdout and return final text.
    
    Args:
        stream: Async iterator of ChatChunk objects
        
    Returns:
        Complete content
    """
    handler = StreamHandler(on_chunk=lambda x: print(x, end="", flush=True))
    result = await handler.aprocess_stream(stream)
    print()
    return result.content
