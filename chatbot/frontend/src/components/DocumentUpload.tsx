// =============================================================================
// DocumentUpload — Modal Upload Component
// =============================================================================
// Document upload modal featuring:
//   • Drag and drop support
//   • File type validation
//   • Upload progress indicator
//   • Smooth scale-in animation
// =============================================================================

import React, { useState, useRef, useCallback, memo } from 'react';
import { ApiError } from '../api/client';
import './DocumentUpload.css';

// -----------------------------------------------------------------------------
// Props Interface
// -----------------------------------------------------------------------------

interface DocumentUploadProps {
    isOpen: boolean;
    onClose: () => void;
    onUpload: (file: File) => Promise<any>;
}

// -----------------------------------------------------------------------------
// Constants
// -----------------------------------------------------------------------------


const ALLOWED_EXTENSIONS = ['.pdf', '.txt', '.md', '.json'];
const MAX_SIZE_MB = 50;

// -----------------------------------------------------------------------------
// Component
// -----------------------------------------------------------------------------

const DocumentUpload: React.FC<DocumentUploadProps> = memo(({
    isOpen,
    onClose,
    onUpload,
}) => {
    const [isDragging, setIsDragging] = useState(false);
    const [file, setFile] = useState<File | null>(null);
    const [isUploading, setIsUploading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [progress, setProgress] = useState(0);

    const fileInputRef = useRef<HTMLInputElement>(null);

    // Validate file
    const validateFile = useCallback((file: File): string | null => {
        // Check type
        const extension = '.' + file.name.split('.').pop()?.toLowerCase();
        if (!ALLOWED_EXTENSIONS.includes(extension)) {
            return `Invalid file type. Allowed: ${ALLOWED_EXTENSIONS.join(', ')}`;
        }

        // Check size
        const sizeMB = file.size / (1024 * 1024);
        if (sizeMB > MAX_SIZE_MB) {
            return `File too large. Maximum size: ${MAX_SIZE_MB}MB`;
        }

        return null;
    }, []);

    // Handle file selection
    const handleFileSelect = useCallback((selectedFile: File) => {
        setError(null);

        const validationError = validateFile(selectedFile);
        if (validationError) {
            setError(validationError);
            return;
        }

        setFile(selectedFile);
    }, [validateFile]);

    // Handle drag events
    const handleDragOver = useCallback((e: React.DragEvent) => {
        e.preventDefault();
        e.stopPropagation();
        setIsDragging(true);
    }, []);

    const handleDragLeave = useCallback((e: React.DragEvent) => {
        e.preventDefault();
        e.stopPropagation();
        setIsDragging(false);
    }, []);

    const handleDrop = useCallback((e: React.DragEvent) => {
        e.preventDefault();
        e.stopPropagation();
        setIsDragging(false);

        const droppedFile = e.dataTransfer.files[0];
        if (droppedFile) {
            handleFileSelect(droppedFile);
        }
    }, [handleFileSelect]);

    // Handle input change
    const handleInputChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
        const selectedFile = e.target.files?.[0];
        if (selectedFile) {
            handleFileSelect(selectedFile);
        }
    }, [handleFileSelect]);

    // Handle upload
    const handleUpload = useCallback(async () => {
        if (!file) return;

        try {
            setIsUploading(true);
            setError(null);
            setProgress(0);

            // Use external upload handler (async but we don't wait for processing)
            await onUpload(file);

            setProgress(100);

            // Close immediately to let user continue working while processing happens in background
            setTimeout(() => {
                setFile(null);
                setProgress(0);
                setIsUploading(false);
                onClose();
            }, 500);

        } catch (err) {
            // Extract user-friendly error message
            let message = 'Upload failed. Please try again.';

            if (err instanceof ApiError) {
                // Use the error message from API (now includes 'detail' from FastAPI)
                message = err.message;

                // Make duplicate error more friendly
                if (err.status === 409) {
                    message = 'This document has already been uploaded.';
                }
            } else if (err instanceof Error) {
                message = err.message;
            }

            setError(message);
            setIsUploading(false);
        }
    }, [file, onUpload, onClose]);

    // Handle close
    const handleClose = useCallback(() => {
        if (isUploading) return;

        setFile(null);
        setError(null);
        setProgress(0);
        // Reset file input so the same file can be selected again
        if (fileInputRef.current) {
            fileInputRef.current.value = '';
        }
        onClose();
    }, [isUploading, onClose]);

    // Format file size
    const formatSize = (bytes: number): string => {
        if (bytes < 1024) return `${bytes} B`;
        if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
        return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
    };

    if (!isOpen) return null;

    return (
        <div className="upload-overlay" onClick={handleClose}>
            <div
                className="upload-modal glass animate-scale-in"
                onClick={e => e.stopPropagation()}
            >
                {/* Header */}
                <div className="upload-header">
                    <h2>Upload Document</h2>
                    <button
                        className="btn-icon upload-close"
                        onClick={handleClose}
                        disabled={isUploading}
                        aria-label="Close"
                    >
                        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                            <path d="M18 6L6 18M6 6l12 12" />
                        </svg>
                    </button>
                </div>

                {/* Drop Zone */}
                <div
                    className={`upload-dropzone ${isDragging ? 'upload-dropzone-active' : ''} ${file ? 'upload-dropzone-has-file' : ''}`}
                    onDragOver={handleDragOver}
                    onDragLeave={handleDragLeave}
                    onDrop={handleDrop}
                    onClick={() => {
                        // Only open file browser if no file is selected
                        if (!file) {
                            fileInputRef.current?.click();
                        }
                    }}
                >
                    <input
                        ref={fileInputRef}
                        type="file"
                        accept={ALLOWED_EXTENSIONS.join(',')}
                        onChange={handleInputChange}
                        className="upload-input"
                    />

                    {file ? (
                        <div className="upload-file-preview">
                            <div className="upload-file-icon">
                                <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                                    <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
                                    <polyline points="14,2 14,8 20,8" />
                                </svg>
                            </div>
                            <div className="upload-file-info">
                                <span className="upload-file-name">{file.name}</span>
                                <span className="upload-file-size">{formatSize(file.size)}</span>
                            </div>
                            <button
                                className="btn-icon upload-file-remove"
                                onClick={(e) => {
                                    e.stopPropagation();
                                    setFile(null);
                                }}
                                aria-label="Remove file"
                            >
                                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                                    <path d="M18 6L6 18M6 6l12 12" />
                                </svg>
                            </button>
                        </div>
                    ) : (
                        <div className="upload-placeholder">
                            <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                                <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                                <polyline points="17,8 12,3 7,8" />
                                <line x1="12" y1="3" x2="12" y2="15" />
                            </svg>
                            <p>Drop files here or click to browse</p>
                            <span>PDF, TXT, MD, JSON up to {MAX_SIZE_MB}MB</span>
                        </div>
                    )}
                </div>

                {/* Progress Bar */}
                {isUploading && (
                    <div className="upload-progress">
                        <div
                            className="upload-progress-bar"
                            style={{ width: `${progress}%` }}
                        />
                    </div>
                )}

                {/* Status List (Placeholder for now, or simple polling feedback) */}
                {/* For real status list, we'd need a separate component or state management */}
                {/* User requested "document processing status symbol". 
                    Since this is an upload modal, maybe just show success clearly?
                    Or a mini-list of "Recent Uploads"? */}

                {/* Status Feedback */}
                {isUploading && progress > 30 && progress < 100 && (
                    <div className="upload-status-message animate-fade-in text-blue-400 flex items-center gap-2 mt-2 justify-center">
                        <div className="animate-spin h-4 w-4 border-2 border-current border-t-transparent rounded-full" />
                        <span>Processing document segments...</span>
                    </div>
                )}

                {!isUploading && !error && progress === 100 && (
                    <div className="upload-success-message animate-fade-in">
                        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="text-green-500">
                            <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14" />
                            <polyline points="22 4 12 14.01 9 11.01" />
                        </svg>
                        <span>Processing Complete! Ready for RAG.</span>
                    </div>
                )}

                {/* Error */}
                {error && (
                    <div className="upload-error">
                        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                            <circle cx="12" cy="12" r="10" />
                            <line x1="12" y1="8" x2="12" y2="12" />
                            <line x1="12" y1="16" x2="12.01" y2="16" />
                        </svg>
                        <span>{error}</span>
                    </div>
                )}

                {/* Actions */}
                <div className="upload-actions">
                    <button
                        className="btn-ghost"
                        onClick={handleClose}
                        disabled={isUploading}
                    >
                        Cancel
                    </button>
                    <button
                        className="btn-primary"
                        onClick={handleUpload}
                        disabled={!file || isUploading}
                    >
                        {isUploading ? (progress > 30 ? 'Processing...' : 'Uploading...') : 'Upload'}
                    </button>
                </div>
            </div>
        </div>
    );
});

DocumentUpload.displayName = 'DocumentUpload';

export default DocumentUpload;
