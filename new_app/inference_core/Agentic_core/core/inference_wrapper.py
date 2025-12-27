"""
Zero-Allocation Inference Wrapper.

Adheres to:
- Zero-Copy I/O: Uses orjson for fast JSON handling.
- Deterministic Concurrency: Explicit timeout management.
- Failure Domain Analysis: Returns Result types.
"""
import httpx
import orjson
from typing import Any, Dict, Optional, AsyncGenerator, Union, List
from contextlib import asynccontextmanager
from dataclasses import dataclass

from .config import get_config

# Initialize config once
CONFIG = get_config()

class InferenceError(Exception):
    """Base error for inference failures."""
    pass

@dataclass
class CompletionResult:
    """Zero-overhead result wrapper."""
    content: str
    usage: Dict[str, int]
    raw_response: Dict[str, Any]
    
    @property
    def tokens_used(self) -> int:
        """Get total tokens used."""
        return self.usage.get('total_tokens', 0)

class InferenceWrapper:
    """
    High-performance wrapper around the Inference Core API.
    Designed for 10k+ concurrent requests.
    """
    
    def __init__(self):
        config = get_config()
        
        # Model configuration
        self.model = config.model_name
        
        # Construct base URL with /v1 prefix if not present
        base_url = config.inference_base_url
        if not base_url.endswith('/v1'):
            base_url = f"{base_url}/v1"
        
        # Connection pool with optimized keep-alive limits
        self._client = httpx.AsyncClient(
            base_url=base_url,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {config.inference_api_key.get_secret_value()}"
            },
            limits=httpx.Limits(max_keepalive_connections=config.max_concurrent_agents, max_connections=config.max_concurrent_agents + 100),
            timeout=httpx.Timeout(60.0, connect=5.0),
            transport=httpx.AsyncHTTPTransport(retries=3)
        )

    async def close(self):
        """Explicitly close the connection pool."""
        await self._client.aclose()

    async def chat_completion(
        self, 
        messages: list, 
        temperature: float = 0.7, 
        max_tokens: int = 1000,
        stream: bool = False,
        response_format: Optional[Dict] = None,
        tools: Optional[List[Dict]] = None,
        tool_choice: Optional[Union[str, Dict]] = None
    ) -> Union[CompletionResult, AsyncGenerator[str, None]]:
        """
        Executes a chat completion request.
        Uses orjson for serialization speed (3-10x faster than stdlib).
        """
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": stream
        }
        
        if response_format:
            payload["response_format"] = response_format
        if tools:
            payload["tools"] = tools
        if tool_choice:
            payload["tool_choice"] = tool_choice

        # Serialize directly to bytes
        content = orjson.dumps(payload)

        try:
            if stream:
                return self._stream_response(content)
            else:
                return await self._unary_response(content)
        except httpx.HTTPError as e:
            # Wrap HTTP errors in domain-specific error
            raise InferenceError(f"Inference request failed: {str(e)}") from e

    async def _unary_response(self, content: bytes) -> CompletionResult:
        response = await self._client.post("/chat/completions", content=content)
        response.raise_for_status()
        
        # Zero-copy decode (avoid creating intermediate strings if possible, though needed for dict)
        data = orjson.loads(response.content)
        
        # Extract content (handle both standard and reasoning models)
        message = data["choices"][0]["message"]
        content_text = message.get("content") or message.get("reasoning_content") or ""
        
        return CompletionResult(
            content=content_text,
            usage=data.get("usage", {}),
            raw_response=data
        )

    async def _stream_response(self, content: bytes) -> AsyncGenerator[str, None]:
        async with self._client.stream("POST", "/chat/completions", content=content) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    chunk_str = line[6:]
                    if chunk_str == "[DONE]":
                        break
                    try:
                        chunk = orjson.loads(chunk_str)
                        delta = chunk["choices"][0]["delta"].get("content")
                        if delta:
                            yield delta
                    except orjson.JSONDecodeError:
                        continue

# Singleton instance
_GLOBAL_WRAPPER: Optional[InferenceWrapper] = None

@asynccontextmanager
async def get_inference_client():
    """
    Context manager for RAII-style access to the global inference client.
    """
    global _GLOBAL_WRAPPER
    if _GLOBAL_WRAPPER is None:
        _GLOBAL_WRAPPER = InferenceWrapper()
    
    try:
        yield _GLOBAL_WRAPPER
    finally:
        # In a real app, we might keep this open. 
        # For strict resource determinism, we could refcount or rely on app shutdown.
        pass

# Cleanup hook
async def shutdown_inference_client():
    global _GLOBAL_WRAPPER
    if _GLOBAL_WRAPPER:
        await _GLOBAL_WRAPPER.close()
        _GLOBAL_WRAPPER = None
