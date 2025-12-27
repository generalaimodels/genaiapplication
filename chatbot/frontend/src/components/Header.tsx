// =============================================================================
// Header — Premium Top Navigation Component
// =============================================================================
// App header featuring:
//   • Logo with gradient text
//   • Theme toggle (sun/moon icons)
//   • Document upload button
//   • User info and logout button
//   • Smooth transitions
// =============================================================================

import React, { memo } from 'react';
import type { Theme } from '../types/api';
import './Header.css';

// -----------------------------------------------------------------------------
// Props Interface
// -----------------------------------------------------------------------------

interface HeaderProps {
    theme: Theme;
    onToggleTheme: () => void;
    onUploadClick: () => void;
    onLibraryClick: () => void;  // [NEW]
    sessionTitle?: string | null;
    isProcessing?: boolean;
    username?: string;
    onLogout?: () => void;
}

// -----------------------------------------------------------------------------
// Component
// -----------------------------------------------------------------------------

const Header: React.FC<HeaderProps> = memo(({
    theme,
    onToggleTheme,
    onUploadClick,
    onLibraryClick, // [NEW]
    sessionTitle,
    isProcessing = false,
    username,
    onLogout,
}) => {
    return (
        <header className="header glass">
            {/* Session Title — Clean, Minimal */}
            <div className="header-brand">
                <div className="header-title-wrapper">
                    {sessionTitle ? (
                        <h1 className="header-session-title">{sessionTitle}</h1>
                    ) : (
                        <h1 className="header-session-title header-session-default">New Chat</h1>
                    )}
                </div>
            </div>

            {/* Actions */}
            <div className="header-actions">
                {/* Processing Status */}
                {isProcessing && (
                    <div className="header-processing animate-fade-in" title="Processing documents in background">
                        <div className="animate-spin h-4 w-4 border-2 border-blue-500 border-t-transparent rounded-full" />
                        <span className="text-xs font-medium text-blue-500 hidden sm:inline">Indexing...</span>
                    </div>
                )}

                {/* Library Button */}
                <button
                    className="btn-icon header-action"
                    onClick={onLibraryClick}
                    title="Document Library"
                    aria-label="Document Library"
                >
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                        <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
                        <polyline points="14 2 14 8 20 8" />
                        <line x1="16" y1="13" x2="8" y2="13" />
                        <line x1="16" y1="17" x2="8" y2="17" />
                        <polyline points="10 9 9 9 8 9" />
                    </svg>
                </button>

                {/* Upload Button */}
                <button
                    className="btn-icon header-action"
                    onClick={onUploadClick}
                    title="Upload document"
                    aria-label="Upload document"
                >
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                        <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                        <polyline points="17,8 12,3 7,8" />
                        <line x1="12" y1="3" x2="12" y2="15" />
                    </svg>
                </button>

                {/* Theme Toggle */}
                <button
                    className="btn-icon header-action theme-toggle"
                    onClick={onToggleTheme}
                    title={`Switch to ${theme === 'dark' ? 'light' : 'dark'} mode`}
                    aria-label={`Switch to ${theme === 'dark' ? 'light' : 'dark'} mode`}
                >
                    {theme === 'dark' ? (
                        // Sun icon for light mode
                        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                            <circle cx="12" cy="12" r="5" />
                            <line x1="12" y1="1" x2="12" y2="3" />
                            <line x1="12" y1="21" x2="12" y2="23" />
                            <line x1="4.22" y1="4.22" x2="5.64" y2="5.64" />
                            <line x1="18.36" y1="18.36" x2="19.78" y2="19.78" />
                            <line x1="1" y1="12" x2="3" y2="12" />
                            <line x1="21" y1="12" x2="23" y2="12" />
                            <line x1="4.22" y1="19.78" x2="5.64" y2="18.36" />
                            <line x1="18.36" y1="5.64" x2="19.78" y2="4.22" />
                        </svg>
                    ) : (
                        // Moon icon for dark mode
                        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                            <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z" />
                        </svg>
                    )}
                </button>

                {/* User Info & Logout */}
                {username && (
                    <div className="header-user">
                        <span className="header-username">{username}</span>
                        {onLogout && (
                            <button
                                className="btn-icon header-action header-logout"
                                onClick={onLogout}
                                title="Logout"
                                aria-label="Logout"
                            >
                                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                                    <path d="M9 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h4" />
                                    <polyline points="16,17 21,12 16,7" />
                                    <line x1="21" y1="12" x2="9" y2="12" />
                                </svg>
                            </button>
                        )}
                    </div>
                )}
            </div>
        </header>
    );
});

Header.displayName = 'Header';

export default Header;
