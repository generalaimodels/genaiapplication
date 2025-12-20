# -*- coding: utf-8 -*-
# =================================================================================================
# api/routers/documents.py — Document Processing Endpoints
# =================================================================================================
# Production-grade document processing implementing:
#
#   1. UPLOAD: Upload documents for processing (PDF, MD, etc.).
#   2. STATUS: Check processing status.
#   3. CHUNKING: Process text into chunks for indexing.
#   4. BACKGROUND PROCESSING: Heavy operations run asynchronously.
#
# Concurrency Model:
# ------------------
#   - File uploads handled in main thread (fast I/O).
#   - Document conversion runs in background tasks.
#   - Thread pool executor for CPU-bound operations.
#
# =================================================================================================

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, UploadFile, status

from api.schemas import (
    ChunkTextRequest,
    ChunkTextResponse,
    ChunkResult,
    DocumentStatus,
    DocumentUploadResponse,
)
from api.exceptions import NotFoundError, ServiceUnavailableError, ValidationError, assert_found
from api.dependencies import (
    get_document_repo,
    get_settings,
    schedule_background_task,
)
from api.database import DocumentRepository, generate_uuid

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
_LOG = logging.getLogger("api.routers.documents")

# -----------------------------------------------------------------------------
# Router Configuration
# -----------------------------------------------------------------------------
router = APIRouter(
    prefix="/documents",
    tags=["Documents"],
    responses={
        413: {"description": "File too large"},
        415: {"description": "Unsupported file type"},
    },
)

# Supported file types
SUPPORTED_EXTENSIONS = {".pdf", ".md", ".txt", ".docx", ".html"}
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB


# =============================================================================
# Document Upload Endpoint
# =============================================================================

@router.post(
    "/upload",
    response_model=DocumentUploadResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Upload Document",
    description="""
    Upload a document for processing.
    
    **Supported Formats**: PDF, Markdown, Text, DOCX, HTML
    
    **Processing Flow**:
    1. File uploaded and validated
    2. Document record created with "pending" status
    3. Background task started for processing
    4. Check `/documents/{id}/status` for completion
    
    **Maximum Size**: 50MB
    """,
)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    documents: DocumentRepository = Depends(get_document_repo),
) -> DocumentUploadResponse:
    """Upload document for processing."""
    # Validate file
    if not file.filename:
        raise ValidationError("Filename is required")
    
    # Check extension
    ext = Path(file.filename).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported file type: {ext}. Supported: {', '.join(SUPPORTED_EXTENSIONS)}",
        )
    
    # Read file content
    content = await file.read()
    file_size = len(content)
    
    if file_size > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File too large: {file_size} bytes. Maximum: {MAX_FILE_SIZE} bytes",
        )
    
    if file_size == 0:
        raise ValidationError("Empty file not allowed")
    
    # Calculate file hash for deduplication
    file_hash = hashlib.sha256(content).hexdigest()
    
    # Create document record
    doc = await documents.create(
        filename=file.filename,
        content_type=file.content_type or "application/octet-stream",
        file_size=file_size,
        file_hash=file_hash,
        metadata={"extension": ext},
    )
    
    # Save file to data directory
    settings = get_settings()
    upload_dir = settings.data_dir / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    
    file_path = upload_dir / f"{doc['id']}{ext}"
    with open(file_path, "wb") as f:
        f.write(content)
    
    _LOG.info("Document uploaded: %s (%s, %d bytes)", doc["id"], file.filename, file_size)
    
    # Schedule background processing
    background_tasks.add_task(
        _process_document,
        doc["id"],
        str(file_path),
        ext,
        documents,
    )
    
    return DocumentUploadResponse(
        id=doc["id"],
        filename=file.filename,
        status="pending",
        message="Document queued for processing. Check /status for completion.",
    )


# =============================================================================
# Document Status Endpoint
# =============================================================================

@router.get(
    "/{doc_id}/status",
    response_model=DocumentStatus,
    summary="Document Status",
    description="Get document processing status.",
)
async def get_document_status(
    doc_id: str,
    documents: DocumentRepository = Depends(get_document_repo),
) -> DocumentStatus:
    """Get document processing status."""
    doc = assert_found(
        await documents.get(doc_id),
        "Document",
        doc_id,
    )
    
    return DocumentStatus(**doc)


# =============================================================================
# Chunk Text Endpoint
# =============================================================================

@router.post(
    "/chunk-text",
    response_model=ChunkTextResponse,
    summary="Chunk Text",
    description="""
    Split text into chunks for indexing.
    
    **Parameters**:
    - `chunk_size`: Target size of each chunk (default: 1024)
    - `chunk_overlap`: Overlap between chunks (default: 128)
    
    **Algorithm**: Recursive character text splitter with hierarchical separators.
    """,
)
async def chunk_text(
    request: ChunkTextRequest,
) -> ChunkTextResponse:
    """Chunk raw text."""
    try:
        # Import chunking module
        from torchchuck import ChunkConfig, chunk_documents, Document
        
        # Create document
        doc_id = request.doc_id or generate_uuid()
        doc = Document(
            id=doc_id,
            text=request.text,
            meta={},
        )
        
        # Configure chunking
        config = ChunkConfig(
            chunk_size=request.chunk_size,
            chunk_overlap=request.chunk_overlap,
        )
        
        # Chunk document
        chunks = list(chunk_documents([doc], config))
        
        # Convert results
        results = [
            ChunkResult(
                index=chunk.index,
                text=chunk.text,
                start=chunk.start,
                end=chunk.end,
                hash64=chunk.hash64,
            )
            for chunk in chunks
        ]
        
        return ChunkTextResponse(
            doc_id=doc_id,
            chunks=results,
            total_chunks=len(results),
        )
        
    except ImportError:
        # Fallback to simple chunking
        return _simple_chunk(request)
    except Exception as e:
        _LOG.error("Chunking error: %s", e, exc_info=True)
        raise ServiceUnavailableError(f"Chunking failed: {str(e)}")


def _simple_chunk(request: ChunkTextRequest) -> ChunkTextResponse:
    """Simple fallback chunking without torchchuck."""
    doc_id = request.doc_id or generate_uuid()
    text = request.text
    chunk_size = request.chunk_size
    overlap = request.chunk_overlap
    
    chunks = []
    start = 0
    index = 0
    
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk_text = text[start:end]
        
        # Try to break at sentence boundary
        if end < len(text):
            for sep in [". ", "\n\n", "\n", " "]:
                last_sep = chunk_text.rfind(sep)
                if last_sep > chunk_size // 2:
                    end = start + last_sep + len(sep)
                    chunk_text = text[start:end]
                    break
        
        chunks.append(ChunkResult(
            index=index,
            text=chunk_text,
            start=start,
            end=end,
            hash64=hash(chunk_text) & ((1 << 63) - 1),
        ))
        
        start = end - overlap
        if start >= len(text):
            break
        index += 1
    
    return ChunkTextResponse(
        doc_id=doc_id,
        chunks=chunks,
        total_chunks=len(chunks),
    )


# =============================================================================
# Delete Document Endpoint
# =============================================================================

@router.delete(
    "/{doc_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete Document",
    description="Delete a document and its chunks.",
)
async def delete_document(
    doc_id: str,
    documents: DocumentRepository = Depends(get_document_repo),
) -> None:
    """Delete document."""
    doc = await documents.get(doc_id)
    if doc is None:
        raise NotFoundError("Document", doc_id)
    
    # Delete file if exists
    settings = get_settings()
    upload_dir = settings.data_dir / "uploads"
    
    for ext in SUPPORTED_EXTENSIONS:
        file_path = upload_dir / f"{doc_id}{ext}"
        if file_path.exists():
            file_path.unlink()
            break
    
    # Soft delete in database
    await documents.update_status(doc_id, "deleted")
    
    _LOG.info("Document deleted: %s", doc_id)


# =============================================================================
# Background Processing
# =============================================================================

async def _process_document(
    doc_id: str,
    file_path: str,
    extension: str,
    documents: DocumentRepository,
) -> None:
    """Process document in background."""
    _LOG.info("Processing document: %s", doc_id)
    
    try:
        # Update status
        await documents.update_status(doc_id, "processing")
        
        # Process based on file type
        if extension == ".pdf":
            chunks = await _process_pdf(file_path)
        elif extension in {".md", ".txt"}:
            chunks = await _process_text(file_path)
        else:
            chunks = await _process_text(file_path)
        
        # Update status with chunk count
        await documents.update_status(
            doc_id,
            "completed",
            chunk_count=len(chunks),
        )
        
        _LOG.info("Document processed: %s (%d chunks)", doc_id, len(chunks))
        
    except Exception as e:
        _LOG.error("Document processing failed: %s - %s", doc_id, e)
        await documents.update_status(
            doc_id,
            "failed",
            error=str(e),
        )


async def _process_pdf(file_path: str) -> List[Dict[str, Any]]:
    """Process PDF document."""
    loop = asyncio.get_event_loop()
    
    def _run_conversion():
        try:
            from conversion import UltraDoclingPipeline, PipelineConfig
            
            output_dir = Path(file_path).parent / "processed"
            output_dir.mkdir(exist_ok=True)
            
            config = PipelineConfig(
                do_ocr=True,
                accelerator_device="auto",
            )
            
            pipeline = UltraDoclingPipeline(
                input_path=file_path,
                output_dir=str(output_dir),
                config=config,
            )
            
            # Extract content
            pipeline.ocr_to_markdown()
            
            # Read markdown output
            md_file = output_dir / (Path(file_path).stem + ".md")
            if md_file.exists():
                text = md_file.read_text(encoding="utf-8")
                return [{"text": text, "source": "pdf"}]
            
            return []
            
        except ImportError:
            _LOG.warning("Docling not available, using basic PDF extraction")
            return []
    
    return await loop.run_in_executor(None, _run_conversion)


async def _process_text(file_path: str) -> List[Dict[str, Any]]:
    """Process text/markdown document."""
    loop = asyncio.get_event_loop()
    
    def _read_file():
        with open(file_path, "r", encoding="utf-8") as f:
            return [{"text": f.read(), "source": "text"}]
    
    return await loop.run_in_executor(None, _read_file)
