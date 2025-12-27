// =============================================================================
// ProcessingStatus — Document Processing Progress Panel
// =============================================================================
// Features:
//   • Real-time progress bar with stages
//   • Estimated time remaining
//   • Status indicators (converting, chunking, indexing)
//   • Smooth animations and premium styling
// =============================================================================

import React, { memo, useCallback, useMemo } from 'react';
import './ProcessingStatus.css';

// -----------------------------------------------------------------------------
// Types
// -----------------------------------------------------------------------------

export interface ProcessingDocument {
    id: string;
    filename: string;
    status: 'pending' | 'processing' | 'completed' | 'failed';
    stage?: 'pending' | 'converting' | 'chunking' | 'indexing' | 'completed' | 'failed';
    progress?: number;
    fileSize?: number;
    startedAt?: number;
    chunkCount?: number;
    error?: string;
}

interface ProcessingStatusProps {
    documents: ProcessingDocument[];
    onDismiss?: (docId: string) => void;
    onDismissAll?: () => void;
}

// -----------------------------------------------------------------------------
// Stage Configuration
// -----------------------------------------------------------------------------

const STAGES = [
    { key: 'converting', label: 'Converting', icon: '📄' },
    { key: 'chunking', label: 'Chunking', icon: '✂️' },
    { key: 'indexing', label: 'Indexing', icon: '🔍' },
];

const STAGE_ORDER = ['pending', 'converting', 'chunking', 'indexing', 'completed'];

// -----------------------------------------------------------------------------
// Utility Functions
// -----------------------------------------------------------------------------

function formatTimeRemaining(seconds: number): string {
    if (seconds < 60) return `~${Math.ceil(seconds)} sec`;
    if (seconds < 3600) return `~${Math.ceil(seconds / 60)} min`;
    return `~${Math.ceil(seconds / 3600)} hr`;
}

function estimateTimeRemaining(
    progress: number,
    startedAt: number,
    _fileSize?: number
): string {
    if (progress >= 100) return 'Complete';
    if (progress <= 0) return 'Starting...';

    const elapsed = (Date.now() - startedAt) / 1000;
    const rate = progress / elapsed;
    const remaining = (100 - progress) / rate;

    if (remaining > 300) return formatTimeRemaining(remaining);
    return formatTimeRemaining(Math.max(remaining, 5));
}

function getStageStatus(currentStage: string, checkStage: string): 'done' | 'active' | 'pending' {
    const currentIdx = STAGE_ORDER.indexOf(currentStage);
    const checkIdx = STAGE_ORDER.indexOf(checkStage);

    if (checkIdx < currentIdx) return 'done';
    if (checkIdx === currentIdx) return 'active';
    return 'pending';
}

// -----------------------------------------------------------------------------
// Progress Item Component
// -----------------------------------------------------------------------------

interface ProgressItemProps {
    doc: ProcessingDocument;
    onDismiss?: (id: string) => void;
}

const ProgressItem: React.FC<ProgressItemProps> = memo(({ doc, onDismiss }) => {
    const progress = doc.progress ?? 0;
    const stage = doc.stage ?? 'pending';
    const startedAt = doc.startedAt ?? Date.now();

    const timeRemaining = useMemo(() => {
        if (doc.status === 'completed') return 'Complete';
        if (doc.status === 'failed') return 'Failed';
        return estimateTimeRemaining(progress, startedAt, doc.fileSize);
    }, [progress, startedAt, doc.fileSize, doc.status]);

    const handleDismiss = useCallback(() => {
        onDismiss?.(doc.id);
    }, [doc.id, onDismiss]);

    const isComplete = doc.status === 'completed';
    const isFailed = doc.status === 'failed';

    return (
        <div className={`progress-item ${isComplete ? 'progress-complete' : ''} ${isFailed ? 'progress-failed' : ''}`}>
            {/* Header */}
            <div className="progress-item-header">
                <div className="progress-item-icon">
                    {isComplete ? (
                        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                            <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14" />
                            <polyline points="22 4 12 14.01 9 11.01" />
                        </svg>
                    ) : isFailed ? (
                        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                            <circle cx="12" cy="12" r="10" />
                            <line x1="15" y1="9" x2="9" y2="15" />
                            <line x1="9" y1="9" x2="15" y2="15" />
                        </svg>
                    ) : (
                        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                            <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
                            <polyline points="14,2 14,8 20,8" />
                        </svg>
                    )}
                </div>
                <span className="progress-item-filename">{doc.filename}</span>
                {(isComplete || isFailed) && (
                    <button className="progress-item-dismiss" onClick={handleDismiss} aria-label="Dismiss">
                        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                            <path d="M18 6L6 18M6 6l12 12" />
                        </svg>
                    </button>
                )}
            </div>

            {/* Progress Bar */}
            <div className="progress-bar-container">
                <div
                    className={`progress-bar-fill ${isFailed ? 'progress-bar-error' : ''}`}
                    style={{ width: `${isFailed ? 100 : progress}%` }}
                />
            </div>

            {/* Progress Info */}
            <div className="progress-info">
                <span className="progress-percent">
                    {isFailed ? 'Error' : `${progress}%`}
                </span>
                <span className="progress-time">
                    {isFailed ? (doc.error || 'Processing failed') : timeRemaining}
                </span>
            </div>

            {/* Stages */}
            {!isComplete && !isFailed && (
                <div className="progress-stages">
                    {STAGES.map(({ key, label, icon }) => {
                        const stageStatus = getStageStatus(stage, key);
                        return (
                            <div key={key} className={`progress-stage progress-stage-${stageStatus}`}>
                                <span className="progress-stage-icon">
                                    {stageStatus === 'done' ? '✓' : stageStatus === 'active' ? icon : '○'}
                                </span>
                                <span className="progress-stage-label">{label}</span>
                            </div>
                        );
                    })}
                </div>
            )}

            {/* Completed Info */}
            {isComplete && doc.chunkCount !== undefined && (
                <div className="progress-complete-info">
                    <span>✓ {doc.chunkCount} chunks indexed</span>
                </div>
            )}
        </div>
    );
});

ProgressItem.displayName = 'ProgressItem';

// -----------------------------------------------------------------------------
// Main Component
// -----------------------------------------------------------------------------

const ProcessingStatus: React.FC<ProcessingStatusProps> = memo(({
    documents,
    onDismiss,
    onDismissAll,
}) => {
    const [isMinimized, setIsMinimized] = React.useState(false);

    if (!documents.length) return null;

    const activeCount = documents.filter(d => d.status === 'processing' || d.status === 'pending').length;
    const hasActive = activeCount > 0;

    const toggleMinimize = () => setIsMinimized(prev => !prev);

    // Minimized view - just a small floating badge
    if (isMinimized) {
        return (
            <div className="processing-minimized" onClick={toggleMinimize}>
                <span className="processing-minimized-icon">📄</span>
                {hasActive && (
                    <span className="processing-minimized-badge">{activeCount}</span>
                )}
                <span className="processing-minimized-text">
                    {hasActive ? 'Processing...' : 'Done'}
                </span>
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <polyline points="18 15 12 9 6 15" />
                </svg>
            </div>
        );
    }

    return (
        <div className="processing-panel glass">
            {/* Header */}
            <div className="processing-header">
                <div className="processing-header-title">
                    <span className="processing-header-icon">📄</span>
                    <span>Processing Documents</span>
                    {hasActive && (
                        <span className="processing-badge">{activeCount}</span>
                    )}
                </div>
                <div className="processing-header-actions">
                    {onDismissAll && !hasActive && (
                        <button className="processing-dismiss-all" onClick={onDismissAll}>
                            Clear
                        </button>
                    )}
                    <button
                        className="processing-minimize-btn"
                        onClick={toggleMinimize}
                        aria-label="Minimize"
                        title="Minimize"
                    >
                        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                            <polyline points="6 9 12 15 18 9" />
                        </svg>
                    </button>
                </div>
            </div>

            {/* Document List */}
            <div className="processing-list">
                {documents.map(doc => (
                    <ProgressItem key={doc.id} doc={doc} onDismiss={onDismiss} />
                ))}
            </div>
        </div>
    );
});

ProcessingStatus.displayName = 'ProcessingStatus';

export default ProcessingStatus;

