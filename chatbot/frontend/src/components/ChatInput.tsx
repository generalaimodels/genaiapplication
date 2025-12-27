// =============================================================================
// ChatInput — Premium Input Component
// =============================================================================
// Auto-resizing textarea with:
//   • Send button with loading state
//   • Keyboard shortcuts (Enter to send)
//   • Character counter
//   • Glass morphism styling
//   • Smooth height transitions
// =============================================================================

import React, { useState, useRef, useCallback, useEffect, memo } from 'react';
import './ChatInput.css';

/* Inline styles for new toggle (ideally move to .css file, but injecting here for speed/consistency with component) */
const styles = `
.btn-rag-toggle {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 32px;
    height: 32px;
    border-radius: 8px;
    border: none;
    background: transparent;
    color: var(--color-text-secondary);
    cursor: pointer;
    transition: all 0.2s ease;
    margin-right: 8px;
}
.btn-rag-toggle:hover {
    background: rgba(255, 255, 255, 0.1);
    color: var(--color-text-primary);
}
.btn-rag-active {
    background: rgba(139, 92, 246, 0.2);
    color: var(--color-accent-primary);
}
`;
// Inject styles
const styleSheet = document.createElement("style");
styleSheet.innerText = styles;
document.head.appendChild(styleSheet);

// -----------------------------------------------------------------------------
// Props Interface
// -----------------------------------------------------------------------------

interface ChatInputProps {
    onSend: (message: string) => void;
    isLoading?: boolean;
    onCancel?: () => void;
    placeholder?: string;
    maxLength?: number;
    useRag?: boolean;
    onToggleRag?: () => void;
}

// -----------------------------------------------------------------------------
// Component
// -----------------------------------------------------------------------------

const ChatInput: React.FC<ChatInputProps> = memo(({
    onSend,
    isLoading = false,
    onCancel,
    placeholder = 'Message AI Assistant...',
    maxLength = 32768,
    useRag = false,
    onToggleRag,
}) => {
    const [value, setValue] = useState('');
    const textareaRef = useRef<HTMLTextAreaElement>(null);

    // Dynamic placeholder
    const currentPlaceholder = useRag ? 'Ask questions about your documents...' : placeholder;
    const adjustHeight = useCallback(() => {
        const textarea = textareaRef.current;
        if (!textarea) return;

        textarea.style.height = 'auto';
        const newHeight = Math.min(textarea.scrollHeight, 200);
        textarea.style.height = `${newHeight}px`;
    }, []);

    // Adjust on value change
    useEffect(() => {
        adjustHeight();
    }, [value, adjustHeight]);

    // Handle input change
    const handleChange = useCallback((e: React.ChangeEvent<HTMLTextAreaElement>) => {
        const newValue = e.target.value;
        if (newValue.length <= maxLength) {
            setValue(newValue);
        }
    }, [maxLength]);

    // Handle send
    const handleSend = useCallback(() => {
        const trimmed = value.trim();
        if (!trimmed || isLoading) return;

        onSend(trimmed);
        setValue('');

        // Reset height
        if (textareaRef.current) {
            textareaRef.current.style.height = 'auto';
        }
    }, [value, isLoading, onSend]);

    // Handle keyboard shortcuts
    const handleKeyDown = useCallback((e: React.KeyboardEvent<HTMLTextAreaElement>) => {
        // Enter without Shift to send
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            handleSend();
        }

        // Escape to cancel streaming
        if (e.key === 'Escape' && isLoading && onCancel) {
            e.preventDefault();
            onCancel();
        }
    }, [handleSend, isLoading, onCancel]);

    // Focus input on mount
    useEffect(() => {
        textareaRef.current?.focus();
    }, []);

    const isEmpty = !value.trim();
    const charCount = value.length;
    const showCharCount = charCount > maxLength * 0.8;

    return (
        <div className="chat-input-container">
            <div className="chat-input-wrapper glass">
                {/* Textarea */}
                <textarea
                    ref={textareaRef}
                    className="chat-input"
                    value={value}
                    onChange={handleChange}
                    onKeyDown={handleKeyDown}
                    placeholder={currentPlaceholder}
                    rows={1}
                    disabled={isLoading}
                    aria-label="Message input"
                />

                {/* Actions */}
                <div className="chat-input-actions">
                    {/* RAG Toggle */}
                    {onToggleRag && (
                        <button
                            className={`btn-rag-toggle ${useRag ? 'btn-rag-active' : ''}`}
                            onClick={onToggleRag}
                            title={useRag ? "Switch to Chat Mode" : "Switch to RAG Mode"}
                        >
                            {useRag ? '📚' : '💬'}
                        </button>
                    )}

                    {/* Character counter */}
                    {showCharCount && (
                        <span
                            className={`char-count ${charCount >= maxLength ? 'char-count-limit' : ''}`}
                        >
                            {charCount.toLocaleString()} / {maxLength.toLocaleString()}
                        </span>
                    )}

                    {/* Send or Cancel button */}
                    {isLoading ? (
                        <button
                            className="btn-cancel"
                            onClick={onCancel}
                            title="Cancel (Esc)"
                            aria-label="Cancel generation"
                        >
                            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                                <rect x="3" y="3" width="18" height="18" rx="2" />
                            </svg>
                        </button>
                    ) : (
                        <button
                            className="btn-send"
                            onClick={handleSend}
                            disabled={isEmpty}
                            title="Send (Enter)"
                            aria-label="Send message"
                        >
                            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                                <path d="M22 2L11 13" />
                                <path d="M22 2L15 22L11 13L2 9L22 2Z" />
                            </svg>
                        </button>
                    )}
                </div>
            </div>

            {/* Hint */}
            <div className="chat-input-hint">
                <span>Press <kbd>Enter</kbd> to send, <kbd>Shift + Enter</kbd> for new line</span>
            </div>
        </div>
    );
});

ChatInput.displayName = 'ChatInput';

export default ChatInput;
