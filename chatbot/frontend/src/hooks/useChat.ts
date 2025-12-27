// =============================================================================
// useChat — Chat Hook with Streaming & RAG Support (Debug Version)
// =============================================================================

import { useState, useCallback, useRef, useEffect } from 'react';
import type { Message, ContextChunk } from '../types/api';
import { sessionsApi } from '../api/client';

const API_BASE = '/api/v1';

function generateId(): string {
    return `msg_${Date.now()}_${Math.random().toString(36).slice(2, 9)}`;
}

const historyCache = new Map<string, Message[]>();

interface UseChatState {
    messages: Message[];
    isStreaming: boolean;
    isLoading: boolean;
    error: string | null;
    useRag: boolean;
}

interface UseChatActions {
    sendMessage: (content: string) => Promise<void>;
    cancelStream: () => void;
    clearMessages: () => void;
    loadHistory: (sessionId: string) => Promise<void>;
    toggleRag: () => void;
}

export function useChat(sessionId: string | null): UseChatState & UseChatActions {
    const [messages, setMessages] = useState<Message[]>([]);
    const [isStreaming, setIsStreaming] = useState(false);
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [useRag, setUseRag] = useState(false); // Start with chat mode

    const abortControllerRef = useRef<AbortController | null>(null);
    const currentSessionRef = useRef<string | null>(null);

    const toggleRag = useCallback(() => {
        setUseRag(prev => !prev);
    }, []);

    // Load history when session changes
    useEffect(() => {
        if (!sessionId) {
            currentSessionRef.current = null;
            setMessages([]);
            return;
        }

        if (sessionId === currentSessionRef.current) {
            return;
        }

        currentSessionRef.current = sessionId;
        console.log('[useChat] Session changed to:', sessionId);

        if (historyCache.has(sessionId)) {
            console.log('[useChat] Loading from cache');
            setMessages(historyCache.get(sessionId)!);
            return;
        }

        setIsLoading(true);
        sessionsApi.getHistory(sessionId, { limit: 100 })
            .then(response => {
                console.log('[useChat] History loaded:', response);
                if (currentSessionRef.current !== sessionId) return;

                const loadedMessages: Message[] = [];
                for (const entry of response.entries || []) {
                    loadedMessages.push({
                        id: `${entry.id}_q`,
                        role: 'user',
                        content: entry.query,
                        timestamp: entry.created_at * 1000,
                    });
                    if (entry.answer) {
                        loadedMessages.push({
                            id: `${entry.id}_a`,
                            role: 'assistant',
                            content: entry.answer,
                            timestamp: entry.updated_at * 1000,
                            context: entry.retrieves as ContextChunk[],
                        });
                    }
                }
                loadedMessages.sort((a, b) => a.timestamp - b.timestamp);
                historyCache.set(sessionId, loadedMessages);
                setMessages(loadedMessages);
            })
            .catch(err => {
                console.error('[useChat] History error:', err);
                setError(err.message);
            })
            .finally(() => setIsLoading(false));

        return () => {
            abortControllerRef.current?.abort();
        };
    }, [sessionId]);

    // Send message with streaming
    const sendMessage = useCallback(async (content: string) => {
        console.log('[useChat] sendMessage called:', { sessionId, content: content?.slice(0, 20), isStreaming });

        if (!sessionId) {
            console.error('[useChat] No session ID!');
            return;
        }
        if (!content.trim()) {
            console.error('[useChat] Empty content!');
            return;
        }
        if (isStreaming) {
            console.error('[useChat] Already streaming!');
            return;
        }

        const userMsg: Message = {
            id: generateId(),
            role: 'user',
            content: content.trim(),
            timestamp: Date.now(),
        };

        const assistantMsgId = generateId();
        const assistantMsg: Message = {
            id: assistantMsgId,
            role: 'assistant',
            content: '',
            timestamp: Date.now(),
            isStreaming: true,
        };

        console.log('[useChat] Adding messages to state');
        setMessages(prev => [...prev, userMsg, assistantMsg]);
        setIsStreaming(true);
        setError(null);

        abortControllerRef.current = new AbortController();

        const endpoint = useRag ? `${API_BASE}/rag/query/stream` : `${API_BASE}/chat/stream`;
        const body = useRag
            ? { query: content.trim(), session_id: sessionId, top_k: 5 }
            : { message: content.trim(), session_id: sessionId };

        console.log('[useChat] Fetching:', endpoint, body);

        try {
            const response = await fetch(endpoint, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(body),
                signal: abortControllerRef.current.signal,
            });

            console.log('[useChat] Response status:', response.status);

            if (!response.ok) {
                throw new Error(`HTTP ${response.status}`);
            }

            const reader = response.body?.getReader();
            if (!reader) {
                console.error('[useChat] No response body reader!');
                throw new Error('No response body');
            }

            console.log('[useChat] Starting to read stream...');

            const decoder = new TextDecoder();
            let buffer = '';
            let fullContent = '';
            let context: ContextChunk[] = [];
            let tokenCount = 0;

            while (true) {
                const { done, value } = await reader.read();

                if (done) {
                    console.log('[useChat] Stream done, total tokens:', tokenCount);
                    break;
                }

                const chunk = decoder.decode(value, { stream: true });
                buffer += chunk;

                const lines = buffer.split('\n');
                buffer = lines.pop() || '';

                for (const line of lines) {
                    if (!line.startsWith('data: ')) continue;

                    const dataStr = line.slice(6).trim();
                    if (!dataStr || dataStr === '[DONE]') continue;

                    try {
                        const parsed = JSON.parse(dataStr);

                        if (parsed.error) {
                            console.error('[useChat] Stream error:', parsed.error);
                            throw new Error(parsed.error);
                        }

                        // Handle RAG context
                        if (parsed.type === 'context' && parsed.chunks) {
                            context = parsed.chunks.map((c: any) => ({
                                text: c.text,
                                score: c.score,
                                doc_id: c.doc_id,
                            }));
                            setMessages(prev => prev.map(m =>
                                m.id === assistantMsgId ? { ...m, context } : m
                            ));
                        }

                        // Handle tokens - check both formats
                        const token = parsed.token ?? (parsed.type === 'token' ? parsed.content : null);
                        const reason = parsed.finish_reason;

                        if (token !== null && token !== undefined && token !== '') {
                            tokenCount++;
                            fullContent += token;

                            // Log first few tokens
                            if (tokenCount <= 3) {
                                console.log(`[useChat] Token ${tokenCount}:`, token);
                            }

                            // Update UI
                            setMessages(prev => prev.map(m =>
                                m.id === assistantMsgId ? { ...m, content: fullContent, finish_reason: reason || m.finish_reason } : m
                            ));
                        } else if (reason) {
                            setMessages(prev => prev.map(m =>
                                m.id === assistantMsgId ? { ...m, finish_reason: reason } : m
                            ));
                        }

                        // Handle completion
                        if (parsed.done === true || parsed.type === 'done') {
                            console.log('[useChat] Received done signal:', parsed);

                            // Extract answer from RAG done response if present
                            // RAG format: {type: 'done', response: {answer: '...', ...}}
                            if (parsed.response?.answer) {
                                const ragAnswer = parsed.response.answer;
                                // Only use RAG answer if we didn't accumulate tokens
                                if (!fullContent || fullContent.length === 0) {
                                    fullContent = ragAnswer;
                                    console.log('[useChat] Using RAG response.answer, length:', ragAnswer.length);
                                }
                                // Update context from response if available
                                if (parsed.response.context_count !== undefined) {
                                    console.log('[useChat] RAG context count:', parsed.response.context_count);
                                }
                            }

                            // Update message content  
                            setMessages(prev => prev.map(m =>
                                m.id === assistantMsgId ? { ...m, content: fullContent } : m
                            ));
                        }
                    } catch (e) {
                        console.warn('[useChat] Parse error:', e, 'line:', line);
                    }
                }
            }

            // Finalize message
            console.log('[useChat] Finalizing message, content length:', fullContent.length);
            setMessages(prev => prev.map(m =>
                m.id === assistantMsgId
                    ? { ...m, isStreaming: false, content: fullContent || 'No response received', context, finish_reason: prev.find(pm => pm.id === assistantMsgId)?.finish_reason }
                    : m
            ));

            // Update cache
            historyCache.set(sessionId, [
                ...(historyCache.get(sessionId) || []),
                userMsg,
                {
                    ...assistantMsg,
                    content: fullContent,
                    isStreaming: false,
                    context,
                    finish_reason: messages.find(m => m.id === assistantMsgId)?.finish_reason // Use state value as truth
                }
            ]);

        } catch (err: any) {
            console.error('[useChat] Error:', err);

            if (err.name === 'AbortError') {
                setMessages(prev => prev.map(m =>
                    m.isStreaming ? { ...m, isStreaming: false } : m
                ));
            } else {
                setError(err.message || 'Failed to send message');
                setMessages(prev => prev.filter(m => !m.isStreaming));
            }
        } finally {
            console.log('[useChat] Cleanup');
            setIsStreaming(false);
            abortControllerRef.current = null;
        }
    }, [sessionId, isStreaming, useRag]);

    const cancelStream = useCallback(() => {
        console.log('[useChat] Cancelling stream');
        abortControllerRef.current?.abort();
    }, []);

    const clearMessages = useCallback(() => {
        setMessages([]);
        if (sessionId) historyCache.delete(sessionId);
    }, [sessionId]);

    const loadHistory = useCallback(async (sid: string) => {
        historyCache.delete(sid);
        currentSessionRef.current = null;
    }, []);

    return {
        messages,
        isStreaming,
        isLoading,
        error,
        useRag,
        sendMessage,
        cancelStream,
        clearMessages,
        loadHistory,
        toggleRag,
    };
}

export default useChat;
