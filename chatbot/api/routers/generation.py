# -*- coding: utf-8 -*-
# =================================================================================================
# api/routers/generation.py — LLM Generation Endpoints
# =================================================================================================
# High-performance LLM generation implementing:
#
#   1. CHAT COMPLETION: Single-turn chat with context retrieval.
#   2. STREAMING: Server-Sent Events for real-time token streaming.
#   3. TEXT COMPLETION: Raw text completion (non-chat).
#   4. BATCH PROCESSING: Parallel processing of multiple prompts.
#
# Concurrency Model:
# ------------------
#   - Bounded semaphore limits concurrent LLM calls.
#   - Async client with connection pooling.
#   - Retry with exponential backoff + jitter.
#
# Rate Limiting:
# --------------
#   - Per-IP rate limiting via middleware.
#   - LLM concurrency limit in LLMClient.
#
# =================================================================================================

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any, AsyncGenerator, Dict, List, Optional

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request, status
from fastapi.responses import StreamingResponse

from api.schemas import (
    BatchChatRequest,
    BatchChatResponse,
    ChatRequest,
    ChatResponse,
    CompletionRequest,
    CompletionResponse,
    ContextChunk,
)
from api.exceptions import ServiceError, ServiceUnavailableError, TimeoutError, assert_found
from api.dependencies import (
    get_session_repo,
    get_history_repo,
    get_response_repo,
    get_llm_client,
    get_chat_history,
    get_vector_base,
    get_request_id,
    get_settings,
)
from api.database import SessionRepository, HistoryRepository, ResponseRepository, generate_uuid

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
_LOG = logging.getLogger("api.routers.generation")

# -----------------------------------------------------------------------------
# Router Configuration
# -----------------------------------------------------------------------------
router = APIRouter(
    tags=["Generation"],
    responses={
        503: {"description": "LLM service unavailable"},
        504: {"description": "Request timeout"},
    },
)


# =============================================================================
# Chat Completion Endpoint
# =============================================================================

@router.post(
    "/chat",
    response_model=ChatResponse,
    summary="Chat Completion",
    description="""
    Send a message and receive an AI response.
    
    **Features**:
    - Automatic session creation if not provided
    - Context retrieval from conversation history
    - Semantic search for relevant information
    - Response tracked in history
    
    **Request Example**:
    ```json
    {
        "session_id": "abc-123",
        "message": "What is the capital of France?",
        "temperature": 0.7,
        "max_tokens": 1024,
        "include_context": true
    }
    ```
    
    **Performance**:
    - Typical latency: 500-3000ms depending on model and context size.
    - Timeout: 60 seconds (configurable).
    - Retries: 3 with exponential backoff.
    """,
)
async def chat_completion(
    request: ChatRequest,
    background_tasks: BackgroundTasks,
    sessions: SessionRepository = Depends(get_session_repo),
    history_repo: HistoryRepository = Depends(get_history_repo),
    response_repo: ResponseRepository = Depends(get_response_repo),
) -> ChatResponse:
    """Process chat message and return AI response."""
    start_time = time.perf_counter()
    settings = get_settings()
    
    # Get or create session
    session_id = request.session_id
    if session_id:
        session = await sessions.get(session_id)
        if session is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Session {session_id} not found",
            )
    else:
        # Auto-create session
        session = await sessions.create()
        session_id = session["id"]
        _LOG.info("Auto-created session: %s", session_id)
    
    # Add user message to history
    query_tokens = max(1, len(request.message) // 4)
    history_entry = await history_repo.create(
        session_id=session_id,
        query=request.message,
        role="user",
        tokens_query=query_tokens,
    )
    
    # Build context if enabled
    context_chunks: List[ContextChunk] = []
    context_text = ""
    
    if request.include_context:
        try:
            context_budget = request.context_token_budget or settings.context_token_budget
            
            # Try chat history context
            chat_history = get_chat_history()
            context_messages = chat_history.build_context(
                conv_id=session["conv_id"],
                branch_id=session["branch_id"],
                query_text=request.message,
                budget_ctx=context_budget,
            )
            
            for conv_id, branch_id, msg_id, content in context_messages:
                context_chunks.append(ContextChunk(
                    text=content[:500],  # Truncate for response
                    score=0.9,
                    metadata={"source": "history", "msg_id": msg_id},
                ))
                context_text += content + "\n\n"
            
        except Exception as e:
            _LOG.warning("Context retrieval failed: %s", e)
    
    # Build prompt
    system_prompt = request.system_prompt or _get_default_system_prompt()
    user_prompt = request.message
    
    if context_text:
        user_prompt = f"Context:\n{context_text}\n\nQuestion: {request.message}"
    
    # Get LLM response
    try:
        llm = get_llm_client()
        
        result = await llm.chat(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            model=request.model,
        )
        
        answer = result.content
        finish_reason = "stop"
        
    except asyncio.TimeoutError:
        raise TimeoutError(operation="LLM chat completion")
    except Exception as e:
        _LOG.error("LLM error: %s", e, exc_info=True)
        raise ServiceUnavailableError(f"LLM service error: {str(e)}")
    
    # Calculate latency
    latency_ms = (time.perf_counter() - start_time) * 1000
    answer_tokens = max(1, len(answer) // 4)
    
    # Update history with answer
    await history_repo.update_answer(
        history_id=history_entry["id"],
        answer=answer,
        tokens_answer=answer_tokens,
        latency_ms=latency_ms,
    )
    
    # Track response in background
    response_id = generate_uuid()
    background_tasks.add_task(
        _track_response,
        response_repo,
        response_id,
        history_entry["id"],
        request,
        answer,
        query_tokens,
        answer_tokens,
        latency_ms,
        finish_reason,
    )
    
    return ChatResponse(
        id=response_id,
        session_id=session_id,
        message=answer,
        context=context_chunks,
        model=request.model or settings.llm_model,
        tokens_prompt=query_tokens,
        tokens_completion=answer_tokens,
        latency_ms=latency_ms,
        finish_reason=finish_reason,
        metadata={"history_id": history_entry["id"]},
    )


# =============================================================================
# Streaming Chat Endpoint (Server-Sent Events)
# =============================================================================

@router.post(
    "/chat/stream",
    summary="Streaming Chat",
    description="""
    Stream chat response tokens as Server-Sent Events.
    
    **SSE Format**:
    ```
    data: {"token": "Hello", "done": false}
    
    data: {"token": " world", "done": false}
    
    data: {"token": "", "done": true, "response": {...}}
    ```
    
    **Client Integration**:
    ```javascript
    const eventSource = new EventSource('/api/v1/chat/stream');
    eventSource.onmessage = (event) => {
        const data = JSON.parse(event.data);
        if (data.done) {
            console.log('Complete:', data.response);
        } else {
            console.log('Token:', data.token);
        }
    };
    ```
    """,
    response_class=StreamingResponse,
)
async def chat_stream(
    request: ChatRequest,
    sessions: SessionRepository = Depends(get_session_repo),
    history_repo: HistoryRepository = Depends(get_history_repo),
) -> StreamingResponse:
    """Stream chat response as SSE with real LLM token streaming."""
    
    async def generate_sse() -> AsyncGenerator[str, None]:
        """Generate SSE events for streaming response."""
        start_time = time.perf_counter()
        settings = get_settings()
        full_response = ""
        history_id = None
        
        try:
            # Get or create session
            session_id = request.session_id
            if session_id:
                session = await sessions.get(session_id)
                if session is None:
                    yield f"data: {json.dumps({'error': 'Session not found', 'done': True})}\n\n"
                    return
            else:
                session = await sessions.create()
                session_id = session["id"]
            
            # Fetch history context (last 10 turns)
            # We fetch BEFORE creating the new entry to avoid self-inclusion or duplicates
            past_history = []
            if session_id:
                entries = await history_repo.list_by_session(session_id, limit=10)
                # Entries are typically recent-first, so we reverse
                for entry in reversed(entries):
                    past_history.append({"role": "user", "content": entry["query"]})
                    if entry["answer"]:
                        past_history.append({"role": "assistant", "content": entry["answer"]})

            # Save user message to history
            query_tokens = max(1, len(request.message) // 4)
            history_entry = await history_repo.create(
                session_id=session_id,
                query=request.message,
                role="user",
                tokens_query=query_tokens,
            )
            history_id = history_entry["id"]
            
            # Get LLM client
            llm = get_llm_client()
            
            # Build prompt
            system_prompt = request.system_prompt or _get_default_system_prompt()
            
            # Build context if enabled
            context_text = ""
            if request.include_context:
                try:
                    vector_base = get_vector_base()
                    if hasattr(vector_base, 'collection') and vector_base.collection is not None:
                        coll = vector_base.collection
                        try:
                            from rethinker_retrieval import Rethinker, RethinkerParams
                            params = RethinkerParams(top_nodes_final=5)
                            rethinker = Rethinker(vector_base, params)
                            result = rethinker.search(request.message)
                            contexts = result.get("contexts", [])
                            for ctx in contexts[:5]:
                                context_text += ctx.get("text", "") + "\n\n"
                        except ImportError:
                            if hasattr(vector_base, 'search'):
                                results, ctx_raw = vector_base.search(request.message, k=5)
                                if results and len(results) > 0:
                                    for doc_id, _ in results[0]:
                                        if hasattr(coll, 'texts') and 0 <= doc_id < len(coll.texts):
                                            context_text += coll.texts[doc_id] + "\n\n"
                except Exception as e:
                    _LOG.debug("Context retrieval failed for streaming: %s", e)
            
            user_prompt = request.message
            if context_text:
                user_prompt = f"Context:\n{context_text}\n\nQuestion: {request.message}"
            
            # Prepare conversation messages
            # [System, History..., User(Latest)]
            # We use extra_messages for all content to preserve order if supported, 
            # or rely on llm_client implementation.
            # vllm_generation.py appends extra_messages AFTER system/user prompts.
            # To get [History, Latest], we pass Latest IN extra_messages.
            # So user_prompt=None, extra_messages=past_history + [{"role": "user", "content": user_prompt}]
            
            full_messages = past_history + [{"role": "user", "content": user_prompt}]

            # Stream tokens from LLM
            effective_max_tokens = request.max_tokens or settings.llm_max_tokens or 4096
            
            async for token in llm.chat_stream(
                system_prompt=system_prompt,
                user_prompt=None, # Set to None to allow extra_messages to handle full flow
                extra_messages=full_messages,
                temperature=request.temperature,
                max_tokens=effective_max_tokens,
                model=request.model,
            ):
                full_response += token
                yield f"data: {json.dumps({'token': token, 'done': False})}\n\n"
            
            # Send completion event
            latency_ms = (time.perf_counter() - start_time) * 1000
            
            # Save answer to history
            if history_id and full_response:
                answer_tokens = max(1, len(full_response) // 4)
                await history_repo.update_answer(
                    history_id=history_id,
                    answer=full_response,
                    tokens_answer=answer_tokens,
                    latency_ms=latency_ms,
                )
            
            yield f"data: {json.dumps({'token': '', 'done': True, 'response': {'message': full_response, 'latency_ms': latency_ms, 'model': request.model or settings.llm_model, 'session_id': session_id}})}\n\n"
            
        except Exception as e:
            _LOG.error("Streaming error: %s", e)
            yield f"data: {json.dumps({'error': str(e), 'done': True})}\n\n"
    
    return StreamingResponse(
        generate_sse(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# =============================================================================
# Text Completion Endpoint
# =============================================================================

@router.post(
    "/complete",
    response_model=CompletionResponse,
    summary="Text Completion",
    description="""
    Raw text completion (non-chat format).
    
    **Use Case**: Code completion, text continuation, few-shot prompting.
    
    **Note**: For conversational AI, use `/chat` endpoint instead.
    """,
)
async def text_completion(
    request: CompletionRequest,
) -> CompletionResponse:
    """Process text completion request."""
    start_time = time.perf_counter()
    settings = get_settings()
    
    try:
        llm = get_llm_client()
        
        result = await llm.complete(
            prompt=request.prompt,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            model=request.model,
        )
        
        latency_ms = (time.perf_counter() - start_time) * 1000
        prompt_tokens = max(1, len(request.prompt) // 4)
        completion_tokens = max(1, len(result.text) // 4)
        
        return CompletionResponse(
            id=generate_uuid(),
            text=result.text,
            model=request.model or settings.llm_model,
            tokens_prompt=prompt_tokens,
            tokens_completion=completion_tokens,
            latency_ms=latency_ms,
            finish_reason="stop",
        )
        
    except asyncio.TimeoutError:
        raise TimeoutError(operation="text completion")
    except Exception as e:
        _LOG.error("Completion error: %s", e, exc_info=True)
        raise ServiceUnavailableError(f"LLM service error: {str(e)}")


# =============================================================================
# Batch Chat Endpoint
# =============================================================================

@router.post(
    "/batch/chat",
    response_model=BatchChatResponse,
    summary="Batch Chat",
    description="""
    Process multiple chat messages in parallel.
    
    **Use Case**: Bulk processing, benchmarking, parallel queries.
    
    **Concurrency**: Limited by LLM client's max_concurrency setting (default: 8).
    
    **Error Handling**: Failed items don't block successful ones; results include
    error information for failed items.
    """,
)
async def batch_chat(
    request: BatchChatRequest,
    background_tasks: BackgroundTasks,
    sessions: SessionRepository = Depends(get_session_repo),
) -> BatchChatResponse:
    """Process batch of chat messages."""
    start_time = time.perf_counter()
    settings = get_settings()
    
    # Get or create session
    session_id = request.session_id
    if session_id:
        session = await sessions.get(session_id)
        if session is None:
            session = await sessions.create()
            session_id = session["id"]
    else:
        session = await sessions.create()
        session_id = session["id"]
    
    results: List[ChatResponse] = []
    successful = 0
    failed = 0
    
    try:
        llm = get_llm_client()
        system_prompt = request.system_prompt or _get_default_system_prompt()
        
        # Run batch chat
        batch_results = await llm.run_batch_chat(
            user_prompts=request.messages,
            system_prompt=system_prompt,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            return_exceptions=True,
        )
        
        for i, result in enumerate(batch_results):
            if isinstance(result, Exception):
                failed += 1
                results.append(ChatResponse(
                    id=generate_uuid(),
                    session_id=session_id,
                    message=f"Error: {str(result)}",
                    context=[],
                    model=settings.llm_model,
                    latency_ms=0,
                    finish_reason="error",
                    metadata={"error": str(result), "index": i},
                ))
            else:
                successful += 1
                results.append(ChatResponse(
                    id=generate_uuid(),
                    session_id=session_id,
                    message=result.content,
                    context=[],
                    model=settings.llm_model,
                    latency_ms=0,  # Individual latencies not tracked
                    finish_reason="stop",
                    metadata={"index": i},
                ))
        
    except Exception as e:
        _LOG.error("Batch error: %s", e, exc_info=True)
        raise ServiceUnavailableError(f"Batch processing failed: {str(e)}")
    
    latency_ms = (time.perf_counter() - start_time) * 1000
    
    return BatchChatResponse(
        results=results,
        total=len(request.messages),
        successful=successful,
        failed=failed,
        latency_ms=latency_ms,
    )


# =============================================================================
# Helper Functions
# =============================================================================

def _get_default_system_prompt() -> str:
    """Get default system prompt for chat."""
    return """You are a helpful AI assistant. Answer questions accurately and concisely.
If you don't know the answer, say so. Use the provided context when available."""


async def _track_response(
    response_repo: ResponseRepository,
    response_id: str,
    history_id: str,
    request: ChatRequest,
    answer: str,
    tokens_prompt: int,
    tokens_completion: int,
    latency_ms: float,
    finish_reason: str,
) -> None:
    """Track LLM response in database (background task)."""
    try:
        settings = get_settings()
        await response_repo.create(
            history_id=history_id,
            model=request.model or settings.llm_model,
            prompt=request.message,
            response=answer,
            temperature=request.temperature or settings.llm_temperature,
            max_tokens=request.max_tokens or settings.llm_max_tokens,
            tokens_prompt=tokens_prompt,
            tokens_completion=tokens_completion,
            finish_reason=finish_reason,
            latency_ms=latency_ms,
        )
    except Exception as e:
        _LOG.warning("Failed to track response: %s", e)
