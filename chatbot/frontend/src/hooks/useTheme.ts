// =============================================================================
// useTheme — Theme Toggle Hook
// =============================================================================
// Manages dark/light theme state with localStorage persistence.
// Syncs with system preference by default.
// =============================================================================

import { useState, useEffect, useCallback } from 'react';
import type { Theme } from '../types/api';

/** Local storage key for theme preference */
const THEME_STORAGE_KEY = 'cca-theme';

/**
 * Detect system color scheme preference.
 */
function getSystemTheme(): Theme {
    if (typeof window === 'undefined') return 'dark';
    return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
}

/**
 * Get stored theme or fall back to system preference.
 */
function getInitialTheme(): Theme {
    if (typeof window === 'undefined') return 'dark';

    const stored = localStorage.getItem(THEME_STORAGE_KEY);
    if (stored === 'dark' || stored === 'light') {
        return stored;
    }

    return getSystemTheme();
}

/**
 * Apply theme to document.
 */
function applyTheme(theme: Theme): void {
    document.documentElement.setAttribute('data-theme', theme);
}

/**
 * Custom hook for theme management.
 * 
 * @returns Current theme and toggle function
 * 
 * @example
 * const { theme, toggleTheme, setTheme } = useTheme();
 */
export function useTheme() {
    const [theme, setThemeState] = useState<Theme>(getInitialTheme);

    // Apply theme on mount and changes
    useEffect(() => {
        applyTheme(theme);
    }, [theme]);

    // Listen for system theme changes
    useEffect(() => {
        const mediaQuery = window.matchMedia('(prefers-color-scheme: dark)');

        const handleChange = (e: MediaQueryListEvent) => {
            const stored = localStorage.getItem(THEME_STORAGE_KEY);
            // Only auto-switch if user hasn't set a preference
            if (!stored) {
                setThemeState(e.matches ? 'dark' : 'light');
            }
        };

        mediaQuery.addEventListener('change', handleChange);
        return () => mediaQuery.removeEventListener('change', handleChange);
    }, []);

    // Set specific theme
    const setTheme = useCallback((newTheme: Theme) => {
        setThemeState(newTheme);
        localStorage.setItem(THEME_STORAGE_KEY, newTheme);
    }, []);

    // Toggle between dark and light
    const toggleTheme = useCallback(() => {
        setTheme(theme === 'dark' ? 'light' : 'dark');
    }, [theme, setTheme]);

    return { theme, setTheme, toggleTheme };
}

export default useTheme;
