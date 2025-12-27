// =============================================================================
// Chatbot — TypeScript API Type Definitions
// =============================================================================
// Production-grade type definitions matching the FastAPI backend schemas.
// Provides complete type safety for all API interactions.
// =============================================================================

// -----------------------------------------------------------------------------
// Session Types
// -----------------------------------------------------------------------------

/** Request to create a new chat session */
export interface SessionCreate {
    user_id?: string;
    title?: string;
    conv_id?: string;
    branch_id?: string;
    metadata?: Record<string, unknown>;
}

/** Request to update an existing session */
export interface SessionUpdate {
    title?: string;
    metadata?: Record<string, unknown>;
    version?: number;
}

/** Session data returned from API */
export interface Session {
    id: string;
    user_id: string | null;
    conv_id: string;
    branch_id: string;
    title: string | null;
    metadata: Record<string, unknown>;
    created_at: number;
    updated_at: number;
    is_active: boolean;
    version: number;
}

/** Paginated session list response */
export interface SessionListResponse {
    sessions: Session[];
    total: number;
    limit: number;
    offset: number;
}

// -----------------------------------------------------------------------------
// Message & History Types
// -----------------------------------------------------------------------------

/** Message role in conversation */
export type MessageRole = 'user' | 'assistant' | 'system';

/** Request to add a message to history */
export interface MessageCreate {
    query: string;
    role?: MessageRole;
    metadata?: Record<string, unknown>;
}

/** A single history entry (query-answer pair) */
export interface HistoryEntry {
    id: string;
    session_id: string;
    query: string;
    answer: string | null;
    role: string;
    metadata: Record<string, unknown>;
    retrieves: ContextChunk[];
    tokens_query: number | null;
    tokens_answer: number | null;
    latency_ms: number | null;
    created_at: number;
    updated_at: number;
}

/** Paginated history list response */
export interface HistoryListResponse {
    entries: HistoryEntry[];
    total: number;
    session_id: string;
}

// -----------------------------------------------------------------------------
// Chat & Generation Types
// -----------------------------------------------------------------------------

/** Request for chat completion */
export interface ChatRequest {
    session_id?: string;
    message: string;
    system_prompt?: string;
    temperature?: number;
    max_tokens?: number;
    include_context?: boolean;
    context_token_budget?: number;
    model?: string;
}

/** A chunk of context retrieved for the response */
export interface ContextChunk {
    text: string;
    score: number;
    doc_id: string | null;
    chunk_index: number | null;
    metadata: Record<string, unknown>;
}

/** Chat completion response */
export interface ChatResponse {
    id: string;
    session_id: string;
    message: string;
    context: ContextChunk[];
    model: string;
    tokens_prompt: number | null;
    tokens_completion: number | null;
    latency_ms: number;
    finish_reason: string | null;
    metadata: Record<string, unknown>;
}

// -----------------------------------------------------------------------------
// RAG Query Types
// -----------------------------------------------------------------------------

/** Request for RAG-enhanced query */
export interface RAGQueryRequest {
    query: string;
    session_id?: string;
    top_k?: number;
    collection?: string;
    use_reranker?: boolean;
    temperature?: number;
    max_tokens?: number;
    system_prompt?: string;
    include_retrieval?: boolean;
    store_history?: boolean;
}

/** RAG query response */
export interface RAGQueryResponse {
    id: string;
    session_id: string | null;
    answer: string;
    sources: ContextChunk[];
    model: string;
    tokens_prompt: number | null;
    tokens_completion: number | null;
    total_latency_ms: number;
    retrieval_latency_ms: number;
    generation_latency_ms: number;
    metadata: Record<string, unknown>;
}

// -----------------------------------------------------------------------------
// Search Types
// -----------------------------------------------------------------------------

/** Request for semantic search */
export interface SearchRequest {
    query: string;
    top_k?: number;
    collection?: string;
    filters?: Record<string, unknown>;
    include_text?: boolean;
    rerank?: boolean;
}

/** A single search result */
export interface SearchResult {
    id: string;
    text: string;
    score: number;
    doc_id: string | null;
    chunk_index: number | null;
    metadata: Record<string, unknown>;
}

/** Search response */
export interface SearchResponse {
    query: string;
    results: SearchResult[];
    total: number;
    latency_ms: number;
}

// -----------------------------------------------------------------------------
// Document Types
// -----------------------------------------------------------------------------

/** Document upload response */
export interface DocumentUploadResponse {
    id: string;
    filename: string;
    status: 'pending' | 'processing' | 'completed' | 'failed';
    message: string;
}

/** Document processing status with detailed progress */
export interface DocumentStatus {
    id: string;
    filename: string;
    status: 'pending' | 'processing' | 'completed' | 'failed';
    stage: 'pending' | 'converting' | 'chunking' | 'indexing' | 'completed' | 'failed';
    progress: number;
    chunk_count: number;
    file_size: number | null;
    error: string | null;
    created_at: number;
    updated_at: number;
}

/** Document list response */
export interface DocumentListResponse {
    documents: DocumentStatus[];
    total: number;
    limit: number;
    offset: number;
}

// -----------------------------------------------------------------------------
// Health & System Types
// -----------------------------------------------------------------------------

/** Health check response */
export interface HealthResponse {
    status: 'healthy' | 'degraded' | 'unhealthy';
    timestamp: number;
    version: string;
    components: Record<string, string>;
}

// -----------------------------------------------------------------------------
// Error Types
// -----------------------------------------------------------------------------

/** API error detail */
export interface ErrorDetail {
    code: string;
    message: string;
    field: string | null;
}

/** Standardized API error response */
export interface ErrorResponse {
    error: string;
    message: string;
    details: ErrorDetail[];
    request_id: string | null;
    timestamp: number;
}

// -----------------------------------------------------------------------------
// SSE Stream Types
// -----------------------------------------------------------------------------

/** Server-Sent Event for streaming chat */
export interface SSEEvent {
    event: 'message' | 'done' | 'error' | 'context';
    data: string;
}

/** Parsed streaming message chunk */
export interface StreamChunk {
    type: 'content' | 'context' | 'done' | 'error';
    content?: string;
    context?: ContextChunk[];
    error?: string;
    finish_reason?: string;
}

// -----------------------------------------------------------------------------
// UI State Types
// -----------------------------------------------------------------------------

/** Message for UI display */
export interface Message {
    id: string;
    role: MessageRole;
    content: string;
    timestamp: number;
    isStreaming?: boolean;
    context?: ContextChunk[];
    latency_ms?: number;
    finish_reason?: string;
}

/** Active theme */
export type Theme = 'dark' | 'light';

/** Toast notification type */
export type ToastType = 'success' | 'error' | 'warning' | 'info';

/** Toast notification */
export interface Toast {
    id: string;
    type: ToastType;
    message: string;
    duration?: number;
}
