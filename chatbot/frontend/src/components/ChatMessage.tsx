// =============================================================================
// ChatMessage — Simple, Robust Markdown & Math Rendering
// =============================================================================

import React, { useState, useCallback } from 'react';
import ReactMarkdown from 'react-markdown';
// @ts-ignore
import remarkGfm from 'remark-gfm';
// @ts-ignore
import remarkMath from 'remark-math';
// @ts-ignore
import rehypeKatex from 'rehype-katex';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { oneDark } from 'react-syntax-highlighter/dist/esm/styles/prism';
import type { Message, ContextChunk } from '../types/api';
import { sessionsApi } from '../api/client';
import 'katex/dist/katex.min.css';
import './ChatMessage.css';

interface ChatMessageProps {
    message: Message;
    isLatest?: boolean;
    sessionId?: string;
    onContinue?: () => void;
}

// Normalize LaTeX delimiters for KaTeX
function processContent(content: string): string {
    if (!content) return '';

    let text = content;

    // 1. Convert \[ ... \] to $$ ... $$ (display math)
    text = text.replace(/\\\[([\s\S]*?)\\\]/g, (match, inner) => `\n$$${inner}$$\n`);

    // 2. Convert \( ... \) to $ ... $ (inline math)
    text = text.replace(/\\\(([\s\S]*?)\\\)/g, (match, inner) => `$${inner}$`);

    // 3. Fix backticked math: `$...$` -> $...$ 
    text = text.replace(/`\$([^`]+)\$`/g, (match, inner) => `$${inner}$`);

    // 4. Fix backticked LaTeX: `\text{...}` -> $\text{...}$
    text = text.replace(/`(\\[a-zA-Z]+[^`]*)`/g, (match, inner) => {
        if (/\\[a-zA-Z]/.test(inner)) {
            return `$${inner}$`;
        }
        return match;
    });

    // 5. Convert parentheses containing LaTeX: (\text{...}) -> $\text{...}$
    // CAUTION: Must NOT match \left( or function calls like \sin(x)
    // We strictly look for ( \command ... ) where \command is usually text-like
    text = text.replace(/([^\\]|^)\((\\text\{[^}]+\}|\\operatorname\{[^}]+\}[^)]*)\)/g, (match, prefix, inner) => {
        return `${prefix}$${inner}$`;
    });

    // 6. Fix "naked" block math that starts on a new line
    // CRITICAL: Strip indentation to avoid it being parsed as a code block!
    // CRITICAL: Match the ENTIRE line, not just the prefix match
    text = text.replace(/(^|\n)[ \t]*((?:\\text\{.|\\begin\{.|[a-zA-Z0-9_]+\s*=\s*\\[a-zA-Z])[^\n]*)/g, (match, prefix, inner) => {
        // Don't wrap if already wrapped
        if (inner.includes('$$')) return match;
        // Don't wrap if it looks like a list item
        if (inner.match(/^[\d-]+\./)) return match;

        return `${prefix}\n$$${inner.trim()}$$\n`;
    });

    return text;
}

// Copy button
const CopyButton: React.FC<{ text: string }> = ({ text }) => {
    const [copied, setCopied] = useState(false);

    const handleCopy = useCallback(async () => {
        await navigator.clipboard.writeText(text);
        setCopied(true);
        setTimeout(() => setCopied(false), 2000);
    }, [text]);

    return (
        <button className="code-copy-btn" onClick={handleCopy}>
            {copied ? '✓ Copied' : 'Copy'}
        </button>
    );
};

// Thinking indicator
const ThinkingIndicator: React.FC = () => (
    <div className="thinking">
        <span className="dot"></span>
        <span className="dot"></span>
        <span className="dot"></span>
    </div>
);

// Sources panel
const Sources: React.FC<{ context: ContextChunk[] }> = ({ context }) => {
    const [open, setOpen] = useState(false);

    if (!context?.length) return null;

    return (
        <div className="sources">
            <button className="sources-btn" onClick={() => setOpen(!open)}>
                📚 {context.length} source{context.length > 1 ? 's' : ''} {open ? '▲' : '▼'}
            </button>
            {open && (
                <div className="sources-content">
                    {context.slice(0, 3).map((c, i) => (
                        <div key={i} className="source-card">
                            <span className="source-num">{i + 1}</span>
                            <p>{c.text?.slice(0, 150)}...</p>
                        </div>
                    ))}
                </div>
            )}
        </div>
    );
};

const ChatMessage: React.FC<ChatMessageProps> = ({ message, sessionId, isLatest, onContinue }) => {
    const isUser = message.role === 'user';
    const isStreaming = message.isStreaming;
    const content = processContent(message.content || '');

    return (
        <div className={`msg ${isUser ? 'msg-user' : 'msg-assistant'} ${isStreaming ? 'msg-streaming' : ''}`}>
            {/* Avatar */}
            <div className={`msg-avatar ${isUser ? 'avatar-user' : 'avatar-bot'}`}>
                {isUser ? 'U' : 'AI'}
            </div>

            {/* Content */}
            <div className="msg-content">
                <div className="msg-role">{isUser ? 'You' : 'Assistant'}</div>

                <div className="msg-body">
                    {isStreaming && !content ? (
                        <ThinkingIndicator />
                    ) : isUser ? (
                        <p className="msg-text">{message.content}</p>
                    ) : (
                        <div className="markdown-body">
                            <ReactMarkdown
                                remarkPlugins={[remarkGfm, remarkMath]}
                                rehypePlugins={[rehypeKatex]}
                                components={{
                                    code({ node, inline, className, children, ...props }: any) {
                                        const match = /language-(\w+)/.exec(className || '');
                                        const code = String(children).replace(/\n$/, '');

                                        if (inline) {
                                            return <code className="code-inline" {...props}>{children}</code>;
                                        }

                                        return (
                                            <div className="code-block">
                                                <div className="code-header">
                                                    <span>{match?.[1] || 'code'}</span>
                                                    <CopyButton text={code} />
                                                </div>
                                                <SyntaxHighlighter
                                                    style={oneDark}
                                                    language={match?.[1] || 'text'}
                                                    PreTag="div"
                                                >
                                                    {code}
                                                </SyntaxHighlighter>
                                            </div>
                                        );
                                    },
                                    table({ children }: any) {
                                        return (
                                            <div className="table-container">
                                                <table>{children}</table>
                                            </div>
                                        );
                                    },
                                }}
                            >
                                {content}
                            </ReactMarkdown>
                            {isStreaming && <span className="cursor">▌</span>}
                        </div>
                    )}
                </div>

                {/* Sources */}
                {!isUser && !isStreaming && message.context && (
                    <Sources context={message.context} />
                )}

                {/* Feedback & Continue */}
                {!isUser && !isStreaming && (
                    <div className="msg-actions">
                        <FeedbackButtons
                            sessionId={sessionId!}
                            messageId={message.id}
                        />
                        {isLatest && message.finish_reason === 'length' && onContinue && (
                            <button className="continue-btn" onClick={onContinue}>
                                ⟳ Continue Generating
                            </button>
                        )}
                    </div>
                )}
            </div>
        </div>
    );
};

// Feedback Component
const FeedbackButtons: React.FC<{ sessionId: string; messageId: string }> = ({ sessionId, messageId }) => {
    const [status, setStatus] = React.useState<'idle' | 'submitting' | 'success' | 'error'>('idle');

    const handleFeedback = async (score: number) => {
        if (!sessionId) return;
        setStatus('submitting');
        try {
            await sessionsApi.submitFeedback(sessionId, messageId, score);
            setStatus('success');
        } catch (e) {
            console.error('Feedback failed:', e);
            setStatus('error');
        }
    };

    if (status === 'success') {
        return <div className="msg-feedback text-green-500 text-xs mt-2">Thanks for your feedback!</div>;
    }

    return (
        <div className="msg-feedback flex gap-2 mt-2 opacity-50 hover:opacity-100 transition-opacity">
            <button
                onClick={() => handleFeedback(1)}
                className="hover:text-green-500 transition-colors p-1"
                title="Helpful"
                disabled={status === 'submitting'}
            >
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M14 10h4.764a2 2 0 011.789 2.894l-3.5 7A2 2 0 0115.263 21h-4.017c-.163 0-.326-.02-.485-.06L7 20m7-10V5a2 2 0 00-2-2h-.095c-.5 0-.905.405-.905.905 0 .714-.211 1.412-.608 2.006L7 11v9m7-10h-2M7 20H5a2 2 0 01-2-2v-6a2 2 0 012-2h2.5" /></svg>
            </button>
            <button
                onClick={() => handleFeedback(-1)}
                className="hover:text-red-500 transition-colors p-1"
                title="Not Helpful"
                disabled={status === 'submitting'}
            >
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M10 14H5.236a2 2 0 01-1.789-2.894l3.5-7A2 2 0 018.736 3h4.018a2 2 0 01.485.06l3.76.94m-7 10v5a2 2 0 002 2h.095c.5 0 .905-.405.905-.905 0-.714.211-1.412.608-2.006L17 13V4m-7 10h2m5-10h2a2 2 0 012 2v6a2 2 0 01-2 2h-2.5" /></svg>
            </button>
            {status === 'error' && (
                <span className="text-red-500 text-xs ml-2">Failed to submit</span>
            )}
        </div>
    );
};

export default ChatMessage;
