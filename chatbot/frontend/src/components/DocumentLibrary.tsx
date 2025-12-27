import React, { useEffect, useState, useCallback } from 'react';
import { documentsApi } from '../api/client';
import { DocumentStatus } from '../types/api';
import './DocumentLibrary.css';

interface DocumentLibraryProps {
    isOpen: boolean;
    onClose: () => void;
}

const DocumentLibrary: React.FC<DocumentLibraryProps> = ({ isOpen, onClose }) => {
    const [documents, setDocuments] = useState<DocumentStatus[]>([]);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [deletingId, setDeletingId] = useState<string | null>(null);
    const [confirmDeleteId, setConfirmDeleteId] = useState<string | null>(null);

    const fetchDocuments = useCallback(async () => {
        try {
            setLoading(true);
            setError(null);
            const response = await documentsApi.list(100, 0); // Fetch up to 100 docs
            setDocuments(response.documents);
        } catch (err) {
            console.error('Failed to fetch documents:', err);
            setError('Failed to load documents');
        } finally {
            setLoading(false);
        }
    }, []);

    useEffect(() => {
        if (isOpen) {
            fetchDocuments();
        }
    }, [isOpen, fetchDocuments]);

    const requestDelete = (docId: string, e: React.MouseEvent) => {
        e.stopPropagation();
        setConfirmDeleteId(docId);
    };

    const cancelDelete = (e: React.MouseEvent) => {
        e.stopPropagation();
        setConfirmDeleteId(null);
    };

    const confirmDelete = async (docId: string, e: React.MouseEvent) => {
        e.stopPropagation();
        try {
            setDeletingId(docId);
            setConfirmDeleteId(null); // Close confirm
            await documentsApi.delete(docId);
            // Refresh list
            setDocuments(prev => prev.filter(d => d.id !== docId));
        } catch (err) {
            console.error('Failed to delete document:', err);
            setError('Failed to delete document');
        } finally {
            setDeletingId(null);
        }
    };

    const formatSize = (bytes: number | null) => {
        if (bytes === null) return '-';
        if (bytes < 1024) return `${bytes} B`;
        if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
        return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
    };

    const formatDate = (timestamp: number) => {
        return new Date(timestamp * 1000).toLocaleDateString(undefined, {
            year: 'numeric',
            month: 'short',
            day: 'numeric',
            hour: '2-digit',
            minute: '2-digit'
        });
    };

    if (!isOpen) return null;

    return (
        <div className="doc-lib-overlay" onClick={onClose}>
            <div className="doc-lib-modal" onClick={e => e.stopPropagation()}>
                {/* Header */}
                <div className="doc-lib-header">
                    <div className="doc-lib-title">
                        <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                            <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
                            <polyline points="14 2 14 8 20 8" />
                            <line x1="16" y1="13" x2="8" y2="13" />
                            <line x1="16" y1="17" x2="8" y2="17" />
                            <polyline points="10 9 9 9 8 9" />
                        </svg>
                        Document Library
                    </div>
                    <button className="doc-lib-close" onClick={onClose} aria-label="Close">
                        <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                            <line x1="18" y1="6" x2="6" y2="18" />
                            <line x1="6" y1="6" x2="18" y2="18" />
                        </svg>
                    </button>
                </div>

                {/* Content */}
                <div className="doc-lib-content">
                    {error && (
                        <div style={{
                            margin: '16px 24px 0',
                            padding: '12px',
                            background: 'rgba(239, 68, 68, 0.1)',
                            border: '1px solid rgba(239, 68, 68, 0.2)',
                            borderRadius: '8px',
                            color: '#f87171',
                            fontSize: '13px',
                            display: 'flex',
                            alignItems: 'center',
                            gap: '8px'
                        }}>
                            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                                <circle cx="12" cy="12" r="10" />
                                <line x1="12" y1="8" x2="12" y2="12" />
                                <line x1="12" y1="16" x2="12.01" y2="16" />
                            </svg>
                            {error}
                        </div>
                    )}

                    {loading && documents.length === 0 ? (
                        <div className="doc-lib-empty">Loading documents...</div>
                    ) : documents.length === 0 ? (
                        <div className="doc-lib-empty">
                            <div className="empty-icon">📂</div>
                            <div>No documents uploaded yet</div>
                        </div>
                    ) : (
                        <table className="doc-lib-table">
                            <thead>
                                <tr>
                                    <th className="col-name">Name</th>
                                    <th className="col-status">Status</th>
                                    <th className="col-date">Uploaded</th>
                                    <th className="col-actions">Actions</th>
                                </tr>
                            </thead>
                            <tbody>
                                {documents.map(doc => (
                                    <tr key={doc.id} className="doc-lib-row">
                                        <td>{doc.filename}
                                            <div style={{ fontSize: '12px', color: 'var(--text-secondary)', marginTop: '4px' }}>
                                                {formatSize(doc.file_size)}
                                            </div>
                                        </td>
                                        <td>
                                            <div className={`status-badge status-${doc.status}`}>
                                                {doc.status === 'completed' ? (
                                                    <>
                                                        <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3">
                                                            <polyline points="20 6 9 17 4 12" />
                                                        </svg>
                                                        Ready for RAG
                                                    </>
                                                ) : doc.status === 'processing' ? (
                                                    <>
                                                        <svg className="animate-spin" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3">
                                                            <path d="M21 12a9 9 0 1 1-6.219-8.56" />
                                                        </svg>
                                                        Processing...
                                                    </>
                                                ) : doc.status === 'failed' ? (
                                                    <>
                                                        <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3">
                                                            <circle cx="12" cy="12" r="10" />
                                                            <line x1="12" y1="8" x2="12" y2="12" />
                                                            <line x1="12" y1="16" x2="12.01" y2="16" />
                                                        </svg>
                                                        Failed
                                                    </>
                                                ) : (
                                                    'Pending'
                                                )}
                                            </div>
                                            {doc.error && (
                                                <div style={{ fontSize: '11px', color: '#ef4444', marginTop: '4px' }}>
                                                    {doc.error}
                                                </div>
                                            )}
                                        </td>
                                        <td>{formatDate(doc.created_at)}</td>
                                        <td style={{ textAlign: 'right' }}>
                                            {confirmDeleteId === doc.id ? (
                                                <div style={{ display: 'flex', gap: '8px', justifyContent: 'flex-end' }}>
                                                    <button
                                                        className="doc-delete-btn"
                                                        onClick={(e) => confirmDelete(doc.id, e)}
                                                        title="Confirm Delete"
                                                        style={{ color: '#ef4444' }}
                                                    >
                                                        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                                                            <polyline points="20 6 9 17 4 12" />
                                                        </svg>
                                                    </button>
                                                    <button
                                                        className="doc-delete-btn"
                                                        onClick={cancelDelete}
                                                        title="Cancel"
                                                    >
                                                        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                                                            <line x1="18" y1="6" x2="6" y2="18" />
                                                            <line x1="6" y1="6" x2="18" y2="18" />
                                                        </svg>
                                                    </button>
                                                </div>
                                            ) : (
                                                <button
                                                    className="doc-delete-btn"
                                                    onClick={(e) => requestDelete(doc.id, e)}
                                                    disabled={deletingId === doc.id}
                                                    title="Delete Document"
                                                >
                                                    {deletingId === doc.id ? (
                                                        <div className="animate-spin" style={{ width: '18px', height: '18px', border: '2px solid currentColor', borderTopColor: 'transparent', borderRadius: '50%' }} />
                                                    ) : (
                                                        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                                                            <polyline points="3 6 5 6 21 6" />
                                                            <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2" />
                                                        </svg>
                                                    )}
                                                </button>
                                            )}
                                        </td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    )}
                </div>
            </div>
        </div>
    );
};

export default DocumentLibrary;
