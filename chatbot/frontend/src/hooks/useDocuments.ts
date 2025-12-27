// =============================================================================
// useDocuments — Enhanced Document Processing Hook
// =============================================================================
// Features:
//   • Real-time polling with progress updates
//   • Stage tracking (converting, chunking, indexing)
//   • Estimated time calculation
//   • Multiple document support
// =============================================================================

import { useState, useCallback, useRef, useEffect } from 'react';
import { documentsApi } from '../api/client';

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

// -----------------------------------------------------------------------------
// Hook
// -----------------------------------------------------------------------------

export function useDocuments() {
    const [processingDocs, setProcessingDocs] = useState<ProcessingDocument[]>([]);
    const activePolls = useRef<Set<string>>(new Set());

    // Poll for status updates
    const checkStatus = useCallback(async (docId: string, filename: string, fileSize?: number) => {
        if (activePolls.current.has(docId)) return;

        activePolls.current.add(docId);
        const startedAt = Date.now();

        // Add to processing state
        setProcessingDocs(prev => {
            if (prev.find(d => d.id === docId)) return prev;
            return [...prev, {
                id: docId,
                filename,
                status: 'pending',
                stage: 'pending',
                progress: 0,
                fileSize,
                startedAt,
            }];
        });

        // Poll every 1.5 seconds for more responsive updates
        const poll = setInterval(async () => {
            try {
                const status = await documentsApi.getStatus(docId);

                setProcessingDocs(prev => prev.map(d =>
                    d.id === docId ? {
                        ...d,
                        status: status.status as ProcessingDocument['status'],
                        stage: (status as any).stage || d.stage,
                        progress: (status as any).progress ?? d.progress,
                        chunkCount: status.chunk_count,
                        error: status.error || undefined,
                    } : d
                ));

                // Stop polling when done
                if (status.status === 'completed' || status.status === 'failed') {
                    clearInterval(poll);
                    activePolls.current.delete(docId);
                }
            } catch (err) {
                console.error(`Polling failed for ${docId}:`, err);
            }
        }, 1500);

        // Cleanup on unmount
        return () => {
            clearInterval(poll);
            activePolls.current.delete(docId);
        };
    }, []);

    // Upload document
    const uploadDocument = useCallback(async (file: File) => {
        const response = await documentsApi.upload(file);
        checkStatus(response.id, response.filename, file.size);
        return response;
    }, [checkStatus]);

    // Dismiss a completed/failed document
    const dismissDocument = useCallback((docId: string) => {
        setProcessingDocs(prev => prev.filter(d => d.id !== docId));
    }, []);

    // Dismiss all completed/failed documents
    const dismissAllCompleted = useCallback(() => {
        setProcessingDocs(prev => prev.filter(d =>
            d.status !== 'completed' && d.status !== 'failed'
        ));
    }, []);

    // Auto-remove completed docs after 10 seconds
    useEffect(() => {
        const completed = processingDocs.filter(d => d.status === 'completed');
        if (completed.length === 0) return;

        const timeouts = completed.map(doc => {
            return setTimeout(() => {
                setProcessingDocs(prev => prev.filter(d => d.id !== doc.id));
            }, 10000);
        });

        return () => timeouts.forEach(t => clearTimeout(t));
    }, [processingDocs]);

    return {
        processingDocs,
        uploadDocument,
        dismissDocument,
        dismissAllCompleted,
        isProcessing: processingDocs.some(d => d.status === 'processing' || d.status === 'pending'),
    };
}
