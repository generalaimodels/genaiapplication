// =============================================================================
// Login — Premium Authentication Component
// =============================================================================
// Features:
//   • Login and Registration modes with smooth toggle
//   • Unique user validation via backend API
//   • Cross-browser compatible (backend storage)
//   • Form validation with visual feedback
//   • Smooth animations and glassmorphic styling
// =============================================================================

import React, { useState, useCallback, useEffect, memo } from 'react';
import { authApi, ApiError } from '../api/client';
import './Login.css';

// -----------------------------------------------------------------------------
// Types & Interfaces
// -----------------------------------------------------------------------------

interface LoginProps {
    onLoginSuccess: (username: string) => void;
}

// -----------------------------------------------------------------------------
// Main Component
// -----------------------------------------------------------------------------

const Login: React.FC<LoginProps> = memo(({ onLoginSuccess }) => {
    // State management
    const [isRegisterMode, setIsRegisterMode] = useState(false);
    const [username, setUsername] = useState('');
    const [password, setPassword] = useState('');
    const [confirmPassword, setConfirmPassword] = useState('');
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [success, setSuccess] = useState<string | null>(null);

    // Form validation errors
    const [usernameError, setUsernameError] = useState<string | null>(null);
    const [passwordError, setPasswordError] = useState<string | null>(null);

    // Clear messages when switching modes
    useEffect(() => {
        setError(null);
        setSuccess(null);
        setUsernameError(null);
        setPasswordError(null);
    }, [isRegisterMode]);

    /**
     * Validate form inputs
     */
    const validateForm = useCallback((): boolean => {
        let isValid = true;
        setUsernameError(null);
        setPasswordError(null);

        // Username validation
        if (!username.trim()) {
            setUsernameError('Username is required');
            isValid = false;
        } else if (username.length < 3) {
            setUsernameError('Username must be at least 3 characters');
            isValid = false;
        } else if (!/^[a-zA-Z0-9_]+$/.test(username)) {
            setUsernameError('Username can only contain letters, numbers, and underscores');
            isValid = false;
        }

        // Password validation
        if (!password) {
            setPasswordError('Password is required');
            isValid = false;
        } else if (password.length < 4) {
            setPasswordError('Password must be at least 4 characters');
            isValid = false;
        }

        // Confirm password (register mode only)
        if (isRegisterMode && password !== confirmPassword) {
            setPasswordError('Passwords do not match');
            isValid = false;
        }

        return isValid;
    }, [username, password, confirmPassword, isRegisterMode]);

    /**
     * Handle form submission
     */
    const handleSubmit = useCallback(async (e: React.FormEvent) => {
        e.preventDefault();
        setError(null);
        setSuccess(null);

        if (!validateForm()) return;

        setIsLoading(true);

        try {
            if (isRegisterMode) {
                // Registration flow via backend API
                const response = await authApi.register({ username, password });
                setSuccess('Account created successfully! Logging you in...');
                // Auto-login after registration
                setTimeout(() => {
                    onLoginSuccess(response.user.username);
                }, 800);
            } else {
                // Login flow via backend API
                const response = await authApi.login({ username, password });
                onLoginSuccess(response.user.username);
            }
        } catch (err) {
            if (err instanceof ApiError) {
                // Handle specific API errors
                if (err.status === 409) {
                    setError('This username is already taken. Please choose a different one.');
                } else if (err.status === 401) {
                    setError('Invalid username or password');
                } else {
                    setError(err.details?.message || err.message || 'Authentication failed');
                }
            } else {
                setError('Connection error. Please check your network and try again.');
            }
        } finally {
            setIsLoading(false);
        }
    }, [username, password, isRegisterMode, validateForm, onLoginSuccess]);

    /**
     * Toggle between login and register modes
     */
    const toggleMode = useCallback(() => {
        setIsRegisterMode(prev => !prev);
        setPassword('');
        setConfirmPassword('');
    }, []);

    return (
        <div className="login-page">
            <div className="login-card">
                {/* Header */}
                <div className="login-header">
                    <div className="login-logo">
                        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                            <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z" />
                        </svg>
                    </div>
                    <h1 className="login-title">
                        {isRegisterMode ? 'Create Account' : 'Welcome Back'}
                    </h1>
                    <p className="login-subtitle">
                        {isRegisterMode
                            ? 'Create an account to get started'
                            : 'Sign in to continue'}
                    </p>
                </div>

                {/* Alert Messages */}
                {error && (
                    <div className="login-alert alert-error">
                        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                            <circle cx="12" cy="12" r="10" />
                            <line x1="12" y1="8" x2="12" y2="12" />
                            <line x1="12" y1="16" x2="12.01" y2="16" />
                        </svg>
                        {error}
                    </div>
                )}

                {success && (
                    <div className="login-alert alert-success">
                        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                            <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14" />
                            <polyline points="22 4 12 14.01 9 11.01" />
                        </svg>
                        {success}
                    </div>
                )}

                {/* Form */}
                <form className="login-form" onSubmit={handleSubmit}>
                    {/* Username */}
                    <div className="form-group">
                        <label className="form-label" htmlFor="username">
                            Username
                        </label>
                        <input
                            id="username"
                            type="text"
                            className={`form-input ${usernameError ? 'input-error' : ''}`}
                            placeholder="Enter your username"
                            value={username}
                            onChange={(e) => setUsername(e.target.value)}
                            autoComplete="username"
                            disabled={isLoading}
                        />
                        {usernameError && (
                            <span className="form-error">{usernameError}</span>
                        )}
                    </div>

                    {/* Password */}
                    <div className="form-group">
                        <label className="form-label" htmlFor="password">
                            Password
                        </label>
                        <input
                            id="password"
                            type="password"
                            className={`form-input ${passwordError ? 'input-error' : ''}`}
                            placeholder="Enter your password"
                            value={password}
                            onChange={(e) => setPassword(e.target.value)}
                            autoComplete={isRegisterMode ? 'new-password' : 'current-password'}
                            disabled={isLoading}
                        />
                        {!isRegisterMode && passwordError && (
                            <span className="form-error">{passwordError}</span>
                        )}
                    </div>

                    {/* Confirm Password (Register mode only) */}
                    {isRegisterMode && (
                        <div className="form-group">
                            <label className="form-label" htmlFor="confirmPassword">
                                Confirm Password
                            </label>
                            <input
                                id="confirmPassword"
                                type="password"
                                className={`form-input ${passwordError ? 'input-error' : ''}`}
                                placeholder="Confirm your password"
                                value={confirmPassword}
                                onChange={(e) => setConfirmPassword(e.target.value)}
                                autoComplete="new-password"
                                disabled={isLoading}
                            />
                            {passwordError && (
                                <span className="form-error">{passwordError}</span>
                            )}
                        </div>
                    )}

                    {/* Submit Button */}
                    <button
                        type="submit"
                        className="login-button"
                        disabled={isLoading}
                    >
                        {isLoading && <span className="btn-spinner" />}
                        {isLoading
                            ? (isRegisterMode ? 'Creating Account...' : 'Signing In...')
                            : (isRegisterMode ? 'Create Account' : 'Sign In')}
                    </button>
                </form>

                {/* Toggle Mode */}
                <div className="login-toggle">
                    {isRegisterMode ? (
                        <>
                            Already have an account?{' '}
                            <span className="login-toggle-link" onClick={toggleMode}>
                                Sign In
                            </span>
                        </>
                    ) : (
                        <>
                            Don't have an account?{' '}
                            <span className="login-toggle-link" onClick={toggleMode}>
                                Register
                            </span>
                        </>
                    )}
                </div>
            </div>
        </div>
    );
});

Login.displayName = 'Login';

export default Login;
