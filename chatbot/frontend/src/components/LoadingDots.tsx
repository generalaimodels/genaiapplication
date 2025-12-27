// =============================================================================
// LoadingDots — Animated Loading Indicator
// =============================================================================
// Simple typing indicator with animated dots
// =============================================================================

import React, { memo } from 'react';
import './LoadingDots.css';

interface LoadingDotsProps {
    size?: 'sm' | 'md' | 'lg';
    color?: 'primary' | 'secondary' | 'muted';
}

const LoadingDots: React.FC<LoadingDotsProps> = memo(({
    size = 'md',
    color = 'primary',
}) => {
    return (
        <div className={`loading-dots loading-dots-${size} loading-dots-${color}`}>
            <span className="loading-dot" />
            <span className="loading-dot" />
            <span className="loading-dot" />
        </div>
    );
});

LoadingDots.displayName = 'LoadingDots';

export default LoadingDots;
