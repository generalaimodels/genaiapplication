// =============================================================================
//  Chatbot — API Client with SSE Streaming Support
// =============================================================================
// Production-grade API client featuring:
//   • Type-safe request/response handling
//   • Server-Sent Events (SSE) for streaming responses
//   • Automatic retry with exponential backoff
//   • AbortController support for request cancellation
//   • Comprehensive error handling
// =============================================================================

import type {
    Session,
    SessionCreate,
    SessionUpdate,
    SessionListResponse,
    HistoryListResponse,
    ChatRequest,
    ChatResponse,
    RAGQueryRequest,
    RAGQueryResponse,
    SearchRequest,
    SearchResponse,
    DocumentUploadResponse,
    DocumentStatus,
    DocumentListResponse,
    HealthResponse,
    ErrorResponse,
    StreamChunk,
    ContextChunk,
} from '../types/api';

// -----------------------------------------------------------------------------
// Configuration
// -----------------------------------------------------------------------------

/** Base URL for API requests (uses Vite proxy in development) */
const API_BASE_URL = '/api/v1';

/** Default request timeout in milliseconds */
const DEFAULT_TIMEOUT = 60000;

/** Maximum retry attempts for failed requests */
const MAX_RETRIES = 3;

/** Initial backoff delay for retries (milliseconds) */
const INITIAL_BACKOFF = 500;

// -----------------------------------------------------------------------------
// Error Classes
// -----------------------------------------------------------------------------

/** Custom API error with parsed response details */
export class ApiError extends Error {
    constructor(
        message: string,
        public status: number,
        public code?: string,
        public details?: ErrorResponse
    ) {
        super(message);
        this.name = 'ApiError';
    }
}

// -----------------------------------------------------------------------------
// Request Helpers
// -----------------------------------------------------------------------------

/**
 * Create headers for API requests.
 * Includes Content-Type and any auth headers.
 */
function createHeaders(contentType = 'application/json'): Headers {
    const headers = new Headers();
    if (contentType) {
        headers.set('Content-Type', contentType);
    }
    headers.set('Accept', 'application/json');
    return headers;
}

/**
 * Sleep for specified milliseconds (used for retry backoff).
 */
function sleep(ms: number): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, ms));
}

/**
 * Calculate exponential backoff with jitter.
 */
function calculateBackoff(attempt: number): number {
    const exponential = INITIAL_BACKOFF * Math.pow(2, attempt);
    const jitter = Math.random() * 250;
    return Math.min(exponential + jitter, 8000);
}

/**
 * Generic fetch wrapper with retry logic and error handling.
 */
async function fetchWithRetry<T>(
    url: string,
    options: RequestInit,
    retries = MAX_RETRIES
): Promise<T> {
    let lastError: Error | null = null;

    for (let attempt = 0; attempt <= retries; attempt++) {
        try {
            const controller = new AbortController();
            const timeoutId = setTimeout(() => controller.abort(), DEFAULT_TIMEOUT);

            const response = await fetch(url, {
                ...options,
                signal: controller.signal,
            });

            clearTimeout(timeoutId);

            if (!response.ok) {
                let errorDetails: ErrorResponse | undefined;
                let errorMessage = `HTTP ${response.status}`;

                try {
                    const jsonResponse = await response.json();
                    // FastAPI HTTPException uses 'detail', our custom errors use 'message'
                    errorMessage = jsonResponse.detail || jsonResponse.message || errorMessage;
                    errorDetails = jsonResponse;
                } catch {
                    // Response may not be JSON
                }

                const error = new ApiError(
                    errorMessage,
                    response.status,
                    errorDetails?.error,
                    errorDetails
                );

                // Don't retry client errors (4xx)
                if (response.status >= 400 && response.status < 500) {
                    throw error;
                }

                lastError = error;
            } else {
                return await response.json();
            }
        } catch (error) {
            if (error instanceof ApiError) {
                throw error;
            }

            lastError = error instanceof Error ? error : new Error(String(error));

            // Don't retry if aborted
            if (lastError.name === 'AbortError') {
                throw new ApiError('Request timeout', 408);
            }
        }

        // Wait before retry
        if (attempt < retries) {
            await sleep(calculateBackoff(attempt));
        }
    }

    throw lastError || new ApiError('Request failed', 500);
}

// =============================================================================
// Session API
// =============================================================================

export const sessionsApi = {
    /**
     * Create a new chat session.
     */
    async create(data: SessionCreate = {}): Promise<Session> {
        return fetchWithRetry(`${API_BASE_URL}/sessions`, {
            method: 'POST',
            headers: createHeaders(),
            body: JSON.stringify(data),
        });
    },

    /**
     * List all sessions with optional pagination.
     */
    async list(
        params: { user_id?: string; limit?: number; offset?: number } = {}
    ): Promise<SessionListResponse> {
        const searchParams = new URLSearchParams();
        if (params.user_id) searchParams.set('user_id', params.user_id);
        if (params.limit) searchParams.set('limit', String(params.limit));
        if (params.offset) searchParams.set('offset', String(params.offset));

        const query = searchParams.toString();
        const url = `${API_BASE_URL}/sessions${query ? `?${query}` : ''}`;

        return fetchWithRetry(url, {
            method: 'GET',
            headers: createHeaders(),
        });
    },

    /**
     * Get session by ID.
     */
    async get(sessionId: string): Promise<Session> {
        return fetchWithRetry(`${API_BASE_URL}/sessions/${sessionId}`, {
            method: 'GET',
            headers: createHeaders(),
        });
    },

    /**
     * Update session.
     */
    async update(sessionId: string, data: SessionUpdate): Promise<Session> {
        return fetchWithRetry(`${API_BASE_URL}/sessions/${sessionId}`, {
            method: 'PATCH',
            headers: createHeaders(),
            body: JSON.stringify(data),
        });
    },

    /**
     * Delete session (soft delete).
     */
    async delete(sessionId: string): Promise<void> {
        await fetchWithRetry(`${API_BASE_URL}/sessions/${sessionId}`, {
            method: 'DELETE',
            headers: createHeaders(),
        });
    },

    /**
     * Get session history.
     */
    async getHistory(
        sessionId: string,
        params: { limit?: number; offset?: number } = {}
    ): Promise<HistoryListResponse> {
        const searchParams = new URLSearchParams();
        if (params.limit) searchParams.set('limit', String(params.limit));
        if (params.offset) searchParams.set('offset', String(params.offset));

        const query = searchParams.toString();
        const url = `${API_BASE_URL}/sessions/${sessionId}/history${query ? `?${query}` : ''}`;

        return fetchWithRetry(url, {
            method: 'GET',
            headers: createHeaders(),
        });
    },

    /**
     * Submit feedback for a message.
     */
    async submitFeedback(
        sessionId: string,
        messageId: string,
        score: number,
        comment?: string
    ): Promise<any> {
        return fetchWithRetry(`${API_BASE_URL}/sessions/${sessionId}/messages/${messageId}/feedback`, {
            method: 'POST',
            headers: createHeaders(),
            body: JSON.stringify({ score, comment }),
        });
    },
};

// =============================================================================
// Chat API
// =============================================================================

export const chatApi = {
    /**
     * Send a chat message and get a complete response.
     */
    async send(data: ChatRequest): Promise<ChatResponse> {
        return fetchWithRetry(`${API_BASE_URL}/chat`, {
            method: 'POST',
            headers: createHeaders(),
            body: JSON.stringify(data),
        });
    },

    /**
     * Stream chat response using Server-Sent Events.
     * Yields chunks as they arrive from the server.
     */
    async *stream(
        data: ChatRequest,
        signal?: AbortSignal
    ): AsyncGenerator<StreamChunk, void, unknown> {
        const response = await fetch(`${API_BASE_URL}/chat/stream`, {
            method: 'POST',
            headers: createHeaders(),
            body: JSON.stringify(data),
            signal,
        });

        if (!response.ok) {
            let errorDetails: ErrorResponse | undefined;
            try {
                errorDetails = await response.json();
            } catch {
                // Ignore parse errors
            }
            throw new ApiError(
                errorDetails?.message || `HTTP ${response.status}`,
                response.status,
                errorDetails?.error,
                errorDetails
            );
        }

        const reader = response.body?.getReader();
        if (!reader) {
            throw new ApiError('No response body', 500);
        }

        const decoder = new TextDecoder();
        let buffer = '';

        try {
            while (true) {
                const { done, value } = await reader.read();

                if (done) {
                    break;
                }

                buffer += decoder.decode(value, { stream: true });
                const lines = buffer.split('\n');
                buffer = lines.pop() || '';

                for (const line of lines) {
                    if (line.startsWith('data: ')) {
                        const dataStr = line.slice(6).trim();

                        if (!dataStr) continue;

                        if (dataStr === '[DONE]') {
                            yield { type: 'done' };
                            return;
                        }

                        try {
                            const parsed = JSON.parse(dataStr);

                            // Handle error responses
                            if (parsed.error) {
                                yield { type: 'error', error: parsed.error };
                                continue;
                            }

                            // Handle backend's streaming format: {token: string, done: boolean, response?: {...}}
                            if (parsed.done === true && parsed.response) {
                                // Final message with complete response
                                yield {
                                    type: 'done',
                                    finish_reason: 'stop',
                                };
                            } else if (parsed.token !== undefined) {
                                // Token-by-token streaming
                                yield { type: 'content', content: parsed.token };
                            } else if (parsed.content !== undefined) {
                                // Fallback for content field format
                                yield { type: 'content', content: parsed.content };
                            } else if (parsed.context) {
                                yield { type: 'context', context: parsed.context };
                            } else if (parsed.finish_reason) {
                                yield { type: 'done', finish_reason: parsed.finish_reason };
                            }
                        } catch {
                            // If not JSON, treat as raw content
                            yield { type: 'content', content: dataStr };
                        }
                    }
                }
            }
        } finally {
            reader.releaseLock();
        }
    },
};

// =============================================================================
// RAG API
// =============================================================================

export const ragApi = {
    /**
     * Perform RAG query (retrieve + generate).
     */
    async query(data: RAGQueryRequest): Promise<RAGQueryResponse> {
        return fetchWithRetry(`${API_BASE_URL}/rag/query`, {
            method: 'POST',
            headers: createHeaders(),
            body: JSON.stringify(data),
        });
    },

    /**
     * Stream RAG query response.
     */
    async *stream(
        data: RAGQueryRequest,
        signal?: AbortSignal
    ): AsyncGenerator<StreamChunk, void, unknown> {
        const response = await fetch(`${API_BASE_URL}/rag/query/stream`, {
            method: 'POST',
            headers: createHeaders(),
            body: JSON.stringify(data),
            signal,
        });

        if (!response.ok) {
            let errorDetails: ErrorResponse | undefined;
            try {
                errorDetails = await response.json();
            } catch {
                // Ignore parse errors
            }
            throw new ApiError(
                errorDetails?.message || `HTTP ${response.status}`,
                response.status,
                errorDetails?.error,
                errorDetails
            );
        }

        const reader = response.body?.getReader();
        if (!reader) {
            throw new ApiError('No response body', 500);
        }

        const decoder = new TextDecoder();
        let buffer = '';

        try {
            while (true) {
                const { done, value } = await reader.read();

                if (done) break;

                buffer += decoder.decode(value, { stream: true });
                const lines = buffer.split('\n');
                buffer = lines.pop() || '';

                for (const line of lines) {
                    if (line.startsWith('data: ')) {
                        const dataStr = line.slice(6).trim();

                        if (!dataStr) continue;

                        if (dataStr === '[DONE]') {
                            yield { type: 'done' };
                            return;
                        }

                        try {
                            const parsed = JSON.parse(dataStr);

                            // Handle error
                            if (parsed.error) {
                                yield { type: 'error', error: parsed.error };
                                continue;
                            }

                            // RAG uses type-based events
                            switch (parsed.type) {
                                case 'context':
                                    // Context comes with chunks array
                                    if (parsed.chunks) {
                                        yield {
                                            type: 'context',
                                            context: parsed.chunks.map((c: any) => ({
                                                text: c.text,
                                                score: c.score,
                                                doc_id: c.doc_id,
                                            }))
                                        };
                                    }
                                    break;

                                case 'token':
                                    // Token streaming
                                    if (parsed.content !== undefined) {
                                        yield { type: 'content', content: parsed.content };
                                    }
                                    break;

                                case 'done':
                                case 'complete':
                                    yield { type: 'done', finish_reason: 'stop' };
                                    break;

                                default:
                                    // Fallback for other formats
                                    if (parsed.content !== undefined) {
                                        yield { type: 'content', content: parsed.content };
                                    } else if (parsed.token !== undefined) {
                                        yield { type: 'content', content: parsed.token };
                                    }
                            }
                        } catch {
                            yield { type: 'content', content: dataStr };
                        }
                    }
                }
            }
        } finally {
            reader.releaseLock();
        }
    },
};

// =============================================================================
// Search API
// =============================================================================

export const searchApi = {
    /**
     * Perform semantic search.
     */
    async search(data: SearchRequest): Promise<SearchResponse> {
        return fetchWithRetry(`${API_BASE_URL}/search`, {
            method: 'POST',
            headers: createHeaders(),
            body: JSON.stringify(data),
        });
    },

    /**
     * Perform search with reranking.
     */
    async rerank(data: SearchRequest): Promise<SearchResponse> {
        return fetchWithRetry(`${API_BASE_URL}/search/rerank`, {
            method: 'POST',
            headers: createHeaders(),
            body: JSON.stringify({ ...data, rerank: true }),
        });
    },
};

// =============================================================================
// Documents API
// =============================================================================

export const documentsApi = {
    /**
     * List documents with pagination.
     */
    async list(limit = 50, offset = 0): Promise<DocumentListResponse> {
        return fetchWithRetry(`${API_BASE_URL}/documents/?limit=${limit}&offset=${offset}`, {
            method: 'GET',
            headers: createHeaders(),
        });
    },

    /**
     * Upload a document for processing.
     */
    async upload(file: File): Promise<DocumentUploadResponse> {
        const formData = new FormData();
        formData.append('file', file);

        const headers = new Headers();
        headers.set('Accept', 'application/json');

        return fetchWithRetry(`${API_BASE_URL}/documents/upload`, {
            method: 'POST',
            headers,
            body: formData,
        }, 0); // Disable retries for uploads to prevent duplicate conflicts
    },

    /**
     * Get document processing status.
     */
    async getStatus(docId: string): Promise<DocumentStatus> {
        return fetchWithRetry(`${API_BASE_URL}/documents/${docId}/status`, {
            method: 'GET',
            headers: createHeaders(),
        });
    },

    /**
     * Delete a document.
     */
    async delete(docId: string): Promise<void> {
        await fetchWithRetry(`${API_BASE_URL}/documents/${docId}`, {
            method: 'DELETE',
            headers: createHeaders(),
        });
    },
};

// =============================================================================
// Health API
// =============================================================================

export const healthApi = {
    /**
     * Check API health.
     */
    async check(): Promise<HealthResponse> {
        return fetchWithRetry(`${API_BASE_URL}/health`, {
            method: 'GET',
            headers: createHeaders(),
        });
    },
};

// =============================================================================
// Authentication API
// =============================================================================

/** User information from auth response */
export interface AuthUser {
    id: string;
    username: string;
    created_at: number;
}

/** Token response from login/register */
export interface AuthTokenResponse {
    access_token: string;
    token_type: string;
    expires_in: number;
    user: AuthUser;
}

/** Auth request (login/register) */
export interface AuthRequest {
    username: string;
    password: string;
}

/** Storage key for auth token */
const AUTH_TOKEN_KEY = 'cca_auth_token';
const AUTH_USER_KEY = 'cca_auth_user';

export const authApi = {
    /**
     * Register a new user account.
     */
    async register(data: AuthRequest): Promise<AuthTokenResponse> {
        const response = await fetchWithRetry<AuthTokenResponse>(`${API_BASE_URL}/auth/register`, {
            method: 'POST',
            headers: createHeaders(),
            body: JSON.stringify(data),
        });

        // Store token and user info
        this.storeAuth(response);
        return response;
    },

    /**
     * Login with username and password.
     */
    async login(data: AuthRequest): Promise<AuthTokenResponse> {
        const response = await fetchWithRetry<AuthTokenResponse>(`${API_BASE_URL}/auth/login`, {
            method: 'POST',
            headers: createHeaders(),
            body: JSON.stringify(data),
        });

        // Store token and user info
        this.storeAuth(response);
        return response;
    },

    /**
     * Verify stored token is still valid.
     */
    async verify(): Promise<AuthUser | null> {
        const token = this.getToken();
        if (!token) return null;

        try {
            const headers = createHeaders();
            headers.set('Authorization', `Bearer ${token}`);

            const response = await fetch(`${API_BASE_URL}/auth/verify`, {
                method: 'POST',
                headers,
            });

            if (!response.ok) {
                // Token invalid, clear storage
                this.logout();
                return null;
            }

            return await response.json();
        } catch {
            return null;
        }
    },

    /**
     * Store auth token and user info.
     */
    storeAuth(response: AuthTokenResponse): void {
        localStorage.setItem(AUTH_TOKEN_KEY, response.access_token);
        localStorage.setItem(AUTH_USER_KEY, JSON.stringify(response.user));
    },

    /**
     * Get stored auth token.
     */
    getToken(): string | null {
        return localStorage.getItem(AUTH_TOKEN_KEY);
    },

    /**
     * Get stored user info.
     */
    getUser(): AuthUser | null {
        try {
            const stored = localStorage.getItem(AUTH_USER_KEY);
            return stored ? JSON.parse(stored) : null;
        } catch {
            return null;
        }
    },

    /**
     * Check if user is authenticated.
     */
    isAuthenticated(): boolean {
        return !!this.getToken();
    },

    /**
     * Logout - clear stored auth.
     */
    logout(): void {
        localStorage.removeItem(AUTH_TOKEN_KEY);
        localStorage.removeItem(AUTH_USER_KEY);
    },
};

// =============================================================================
// Unified API Client
// =============================================================================

export const api = {
    sessions: sessionsApi,
    chat: chatApi,
    rag: ragApi,
    search: searchApi,
    documents: documentsApi,
    health: healthApi,
    auth: authApi,
};

export default api;

