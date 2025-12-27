// =============================================================================
// SessionSidebar — Premium Session List Component
// =============================================================================
// Collapsible sidebar featuring:
//   • New session button with gradient
//   • Session list with active highlighting
//   • Delete with hover reveal
//   • Smooth slide animations
//   • Mobile-friendly hamburger toggle
// =============================================================================

import React, { useState, useCallback, memo } from 'react';
import type { Session } from '../types/api';
import './SessionSidebar.css';

// -----------------------------------------------------------------------------
// Props Interface
// -----------------------------------------------------------------------------

interface SessionSidebarProps {
    sessions: Session[];
    activeSessionId: string | null;
    isLoading: boolean;
    onSelectSession: (sessionId: string) => void;
    onCreateSession: () => void;
    onDeleteSession: (sessionId: string) => void;
}

// -----------------------------------------------------------------------------
// Sub-Components
// -----------------------------------------------------------------------------

/** Session list item */
const SessionItem: React.FC<{
    session: Session;
    isActive: boolean;
    onSelect: () => void;
    onDelete: () => void;
}> = memo(({ session, isActive, onSelect, onDelete }) => {
    const [showDelete, setShowDelete] = useState(false);

    // Format relative time
    const timeAgo = useCallback((timestamp: number) => {
        const seconds = Math.floor((Date.now() / 1000) - timestamp);

        if (seconds < 60) return 'Just now';
        if (seconds < 3600) return `${Math.floor(seconds / 60)}m ago`;
        if (seconds < 86400) return `${Math.floor(seconds / 3600)}h ago`;
        if (seconds < 604800) return `${Math.floor(seconds / 86400)}d ago`;

        return new Date(timestamp * 1000).toLocaleDateString();
    }, []);

    const handleDelete = useCallback((e: React.MouseEvent) => {
        e.stopPropagation();
        onDelete();
    }, [onDelete]);

    return (
        <div
            className={`session-item ${isActive ? 'session-item-active' : ''}`}
            onClick={onSelect}
            onMouseEnter={() => setShowDelete(true)}
            onMouseLeave={() => setShowDelete(false)}
            role="button"
            tabIndex={0}
            onKeyDown={(e) => e.key === 'Enter' && onSelect()}
        >
            {/* Icon */}
            <div className="session-icon">
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z" />
                </svg>
            </div>

            {/* Content */}
            <div className="session-content">
                <div className="session-title">
                    {session.title || 'New Chat'}
                </div>
                <div className="session-time">
                    {timeAgo(session.updated_at)}
                </div>
            </div>

            {/* Delete button */}
            <button
                className={`session-delete ${showDelete ? 'session-delete-visible' : ''}`}
                onClick={handleDelete}
                title="Delete session"
                aria-label="Delete session"
            >
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <path d="M3 6h18" />
                    <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6" />
                    <path d="M8 6V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2" />
                </svg>
            </button>
        </div>
    );
});

SessionItem.displayName = 'SessionItem';

/** Loading skeleton */
const SessionSkeleton: React.FC = () => (
    <div className="session-item session-skeleton">
        <div className="skeleton" style={{ width: 32, height: 32, borderRadius: 8 }} />
        <div className="session-content">
            <div className="skeleton" style={{ width: '70%', height: 14 }} />
            <div className="skeleton" style={{ width: '40%', height: 10, marginTop: 6 }} />
        </div>
    </div>
);

// -----------------------------------------------------------------------------
// Main Component
// -----------------------------------------------------------------------------

const SessionSidebar: React.FC<SessionSidebarProps> = memo(({
    sessions,
    activeSessionId,
    isLoading,
    onSelectSession,
    onCreateSession,
    onDeleteSession,
}) => {
    const [isCollapsed, setIsCollapsed] = useState(false);
    const [isMobileOpen, setIsMobileOpen] = useState(false);

    const toggleMobile = useCallback(() => {
        setIsMobileOpen(prev => !prev);
    }, []);

    const handleSelectSession = useCallback((sessionId: string) => {
        onSelectSession(sessionId);
        setIsMobileOpen(false); // Close on mobile after selection
    }, [onSelectSession]);

    return (
        <>
            {/* Mobile Toggle Button */}
            <button
                className="sidebar-mobile-toggle"
                onClick={toggleMobile}
                aria-label="Toggle sidebar"
            >
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    {isMobileOpen ? (
                        <path d="M18 6L6 18M6 6l12 12" />
                    ) : (
                        <>
                            <path d="M3 12h18" />
                            <path d="M3 6h18" />
                            <path d="M3 18h18" />
                        </>
                    )}
                </svg>
            </button>

            {/* Overlay for mobile */}
            {isMobileOpen && (
                <div
                    className="sidebar-overlay"
                    onClick={() => setIsMobileOpen(false)}
                />
            )}

            {/* Sidebar */}
            <aside className={`
        sidebar 
        glass-heavy
        ${isCollapsed ? 'sidebar-collapsed' : ''} 
        ${isMobileOpen ? 'sidebar-mobile-open' : ''}
      `}>
                {/* Logo & New Chat — ChatGPT Style */}
                <div className="sidebar-header">
                    {/* Logo */}
                    <div className="sidebar-logo">
                        <div className="sidebar-logo-icon">
                            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                                <circle cx="12" cy="12" r="10" />
                                <path d="M8 12l2 2 4-4" />
                            </svg>
                        </div>
                        {!isCollapsed && <span className="sidebar-logo-text">AI Assistant</span>}
                    </div>

                    {/* New chat button */}
                    <button
                        className="btn-new-chat"
                        onClick={onCreateSession}
                        title="New chat"
                        aria-label="Start new chat"
                    >
                        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                            <path d="M12 5v14M5 12h14" />
                        </svg>
                        {!isCollapsed && <span>New Chat</span>}
                    </button>
                </div>

                {/* Session List */}
                <div className="session-list">
                    {isLoading ? (
                        // Loading skeletons
                        <>
                            <SessionSkeleton />
                            <SessionSkeleton />
                            <SessionSkeleton />
                        </>
                    ) : sessions.length === 0 ? (
                        // Empty state
                        <div className="session-empty">
                            <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                                <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z" />
                            </svg>
                            <p>No conversations yet</p>
                            <span>Start a new chat to begin</span>
                        </div>
                    ) : (
                        // Session items
                        sessions.map((session, index) => (
                            <div
                                key={session.id}
                                className="animate-slide-in-left"
                                style={{ animationDelay: `${index * 50}ms` }}
                            >
                                <SessionItem
                                    session={session}
                                    isActive={session.id === activeSessionId}
                                    onSelect={() => handleSelectSession(session.id)}
                                    onDelete={() => onDeleteSession(session.id)}
                                />
                            </div>
                        ))
                    )}
                </div>

                {/* Footer */}
                <div className="sidebar-footer">
                    <button
                        className="sidebar-collapse-btn btn-ghost"
                        onClick={() => setIsCollapsed(!isCollapsed)}
                        title={isCollapsed ? 'Expand sidebar' : 'Collapse sidebar'}
                    >
                        <svg
                            width="18"
                            height="18"
                            viewBox="0 0 24 24"
                            fill="none"
                            stroke="currentColor"
                            strokeWidth="2"
                            style={{ transform: isCollapsed ? 'rotate(180deg)' : 'rotate(0deg)' }}
                        >
                            <polyline points="15,18 9,12 15,6" />
                        </svg>
                    </button>
                </div>
            </aside>
        </>
    );
});

SessionSidebar.displayName = 'SessionSidebar';

export default SessionSidebar;
