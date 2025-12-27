// =============================================================================
// App.tsx — Premium Chatbot with Authentication & AI Session Naming
// =============================================================================

import React, { useState, useCallback, useRef, useEffect } from 'react';
import { useTheme } from './hooks/useTheme';
import { useSessions } from './hooks/useSessions';
import { useChat } from './hooks/useChat';
import { useDocuments } from './hooks/useDocuments';
import { authApi } from './api/client';
import Header from './components/Header';
import SessionSidebar from './components/SessionSidebar';
import ChatMessage from './components/ChatMessage';
import ChatInput from './components/ChatInput';
import DocumentUpload from './components/DocumentUpload';
import ProcessingStatus from './components/ProcessingStatus';
import DocumentLibrary from './components/DocumentLibrary';
import Login from './components/Login';
import './App.css';

// -----------------------------------------------------------------------------
// Helper Functions
// -----------------------------------------------------------------------------

/**
 * Generate a smart title from the first message
 */
function generateSmartTitle(userMessage: string): string {
    const firstSentence = userMessage.split(/[.!?\n]/)[0].trim();
    if (firstSentence.length <= 40) {
        return firstSentence || 'New Chat';
    }
    const truncated = firstSentence.slice(0, 40);
    const lastSpace = truncated.lastIndexOf(' ');
    return (lastSpace > 20 ? truncated.slice(0, lastSpace) : truncated) + '...';
}

// -----------------------------------------------------------------------------
// Main App Component
// -----------------------------------------------------------------------------

const App: React.FC = () => {
    // Authentication state - check if user is already logged in
    const [currentUser, setCurrentUser] = useState<string | null>(() => {
        const user = authApi.getUser();
        return user?.username || null;
    });

    const { theme, toggleTheme } = useTheme();

    const {
        sessions,
        activeSessionId,
        isLoading: sessionsLoading,
        createSession,
        selectSession,
        deleteSession,
        updateSessionTitle,
    } = useSessions();

    const {
        messages,
        isStreaming,
        isLoading: chatLoading,
        error: chatError,
        useRag,
        sendMessage,
        cancelStream,
        toggleRag,
    } = useChat(activeSessionId);

    const {
        processingDocs,
        uploadDocument,
        dismissDocument,
        dismissAllCompleted,
        isProcessing
    } = useDocuments();

    const [isUploadOpen, setIsUploadOpen] = useState(false);
    const [isLibraryOpen, setIsLibraryOpen] = useState(false);
    const chatContainerRef = useRef<HTMLDivElement>(null);
    const titleUpdatedRef = useRef<Set<string>>(new Set());
    const activeSession = sessions.find(s => s.id === activeSessionId);

    // Auto-scroll on new messages
    useEffect(() => {
        if (chatContainerRef.current) {
            chatContainerRef.current.scrollTo({
                top: chatContainerRef.current.scrollHeight,
                behavior: 'smooth',
            });
        }
    }, [messages]);

    // Auto-update session title after first exchange
    useEffect(() => {
        if (!activeSessionId) return;
        if (titleUpdatedRef.current.has(activeSessionId)) return;

        // Find first user message and first assistant response
        const userMsg = messages.find(m => m.role === 'user');
        const assistantMsg = messages.find(m => m.role === 'assistant' && !m.isStreaming && m.content);

        if (userMsg && assistantMsg) {
            titleUpdatedRef.current.add(activeSessionId);
            const smartTitle = generateSmartTitle(userMsg.content);
            updateSessionTitle(activeSessionId, smartTitle);
        }
    }, [activeSessionId, messages, updateSessionTitle]);

    const handleNewSession = useCallback(async () => {
        try {
            await createSession();
        } catch (err) {
            console.error('Failed to create session:', err);
        }
    }, [createSession]);

    const handleSendMessage = useCallback(async (content: string) => {
        if (!activeSessionId) {
            try {
                await createSession({ title: 'New Chat' });
                setTimeout(() => sendMessage(content), 100);
            } catch (err) {
                console.error('Failed to create session:', err);
            }
            return;
        }
        await sendMessage(content);
    }, [activeSessionId, createSession, sendMessage]);


    /**
     * Handle successful login
     */
    const handleLoginSuccess = useCallback((username: string) => {
        setCurrentUser(username);
    }, []);

    /**
     * Handle logout
     */
    const handleLogout = useCallback(() => {
        authApi.logout();
        setCurrentUser(null);
    }, []);

    // If not authenticated, show Login page
    if (!currentUser) {
        return (
            <div data-theme={theme}>
                <Login onLoginSuccess={handleLoginSuccess} />
            </div>
        );
    }

    // Authenticated - show Chat interface
    return (
        <div className="app" data-theme={theme}>
            {/* Sidebar */}
            <SessionSidebar
                sessions={sessions}
                activeSessionId={activeSessionId}
                isLoading={sessionsLoading}
                onSelectSession={selectSession}
                onCreateSession={handleNewSession}
                onDeleteSession={deleteSession}
            />

            {/* Main Content */}
            <main className="main-content">
                <Header
                    theme={theme}
                    onToggleTheme={toggleTheme}
                    onUploadClick={() => setIsUploadOpen(true)}
                    onLibraryClick={() => setIsLibraryOpen(true)}
                    sessionTitle={activeSession?.title}
                    isProcessing={isProcessing}
                    username={currentUser}
                    onLogout={handleLogout}
                />

                <div className="chat-container">
                    {/* Error */}
                    {chatError && (
                        <div className="error-banner">
                            ⚠️ {chatError}
                        </div>
                    )}

                    {/* Chat Messages */}
                    <div className="chat-scroll" ref={chatContainerRef}>
                        {messages.length === 0 && !chatLoading ? (
                            <div className="empty-state">
                                <div className="empty-icon">💬</div>
                                <h2>Hello, {currentUser}!</h2>
                                <p>How can I help you today?</p>
                                <div className="quick-actions">
                                    <button onClick={() => handleSendMessage("Explain transformers in AI")}>
                                        🤖 Explain transformers
                                    </button>
                                    <button onClick={() => handleSendMessage("What is attention mechanism?")}>
                                        🧠 Attention mechanism
                                    </button>
                                    <button onClick={() => setIsUploadOpen(true)}>
                                        📄 Upload document
                                    </button>
                                </div>
                            </div>
                        ) : (
                            <>
                                {chatLoading && messages.length === 0 && (
                                    <div className="loading-skeleton">
                                        <div className="skeleton-line"></div>
                                        <div className="skeleton-line short"></div>
                                    </div>
                                )}
                                {messages.map((message, index) => (
                                    <ChatMessage
                                        key={message.id}
                                        message={message}
                                        isLatest={index === messages.length - 1}
                                        sessionId={activeSessionId || undefined}
                                        onContinue={() => handleSendMessage("Continue generation from where you left off")}
                                    />
                                ))}
                            </>
                        )}
                    </div>

                    {/* Input */}
                    <ChatInput
                        onSend={handleSendMessage}
                        isLoading={isStreaming}
                        onCancel={cancelStream}
                        useRag={useRag}
                        onToggleRag={toggleRag}
                    />
                </div>
            </main>

            {/* Upload Modal */}
            <DocumentUpload
                isOpen={isUploadOpen}
                onClose={() => setIsUploadOpen(false)}
                onUpload={uploadDocument}
            />

            {/* Library Modal */}
            <DocumentLibrary
                isOpen={isLibraryOpen}
                onClose={() => setIsLibraryOpen(false)}
            />

            {/* Processing Status Panel */}
            <ProcessingStatus
                documents={processingDocs}
                onDismiss={dismissDocument}
                onDismissAll={dismissAllCompleted}
            />
        </div>
    );
};

export default App;

