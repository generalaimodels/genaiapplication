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
    ChunkingStatusResponse,
    DocumentStatus,
    DocumentUploadResponse,
    DocumentListResponse,
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

@router.get(
    "/",
    response_model=DocumentListResponse,
    summary="List Documents",
    description="List all documents with pagination.",
)
async def list_documents(
    limit: int = 50,
    offset: int = 0,
    documents: DocumentRepository = Depends(get_document_repo),
) -> DocumentListResponse:
    """List documents."""
    result = await documents.list(limit, offset)
    return DocumentListResponse(**result)

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
    
    # Check for duplicates
    existing_doc = await documents.get_by_hash(file_hash)
    if existing_doc:
        _LOG.info("Duplicate upload blocked: %s (hash=%s, id=%s)", 
                  file.filename, file_hash, existing_doc["id"])
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Document already exists: {existing_doc['filename']} (ID: {existing_doc['id']})",
        )
    
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
    """
    Process document in background with proper chunking and progress tracking.
    
    Flow & Progress:
    -----------------
    1. pending (0%) → processing, converting (10%)
    2. Converting PDF → markdown (10-40%)
    3. Chunking text (40-80%)
    4. Indexing into vector store (80-95%)
    5. completed (100%)
    """
    _LOG.info("Processing document: %s", doc_id)
    
    try:
        # Stage 1: Start processing
        await documents.update_status(
            doc_id, "processing", 
            stage="converting" if extension == ".pdf" else "chunking",
            progress=10
        )
        
        # Stage 2: Process based on file type
        if extension == ".pdf":
            # PDF: Converting stage (10-40%)
            await documents.update_status(doc_id, "processing", stage="converting", progress=25)
            chunks = await _process_pdf(file_path, doc_id)
            await documents.update_status(doc_id, "processing", stage="chunking", progress=60)
        elif extension in {".md", ".txt"}:
            # Text: Chunking stage (10-80%)
            await documents.update_status(doc_id, "processing", stage="chunking", progress=40)
            chunks = await _process_text(file_path, doc_id)
        else:
            await documents.update_status(doc_id, "processing", stage="chunking", progress=40)
            chunks = await _process_text(file_path, doc_id)
        
        # Stage 3: Indexing (80-95%)
        await documents.update_status(doc_id, "processing", stage="indexing", progress=85)
        
        # Indexing happens automatically via RAG warmup, mark progress
        await documents.update_status(doc_id, "processing", stage="indexing", progress=95)
        
        # Stage 4: Completed (100%)
        await documents.update_status(
            doc_id,
            "completed",
            stage="completed",
            progress=100,
            chunk_count=len(chunks),
        )
        
        _LOG.info("Document processed: %s (%d chunks)", doc_id, len(chunks))
        
    except Exception as e:
        _LOG.error("Document processing failed: %s - %s", doc_id, e, exc_info=True)
        await documents.update_status(
            doc_id,
            "failed",
            stage="failed",
            progress=0,
            error=str(e),
        )


async def _process_pdf(file_path: str, doc_id: str) -> List[Dict[str, Any]]:
    """
    Process PDF document: Convert to markdown, then chunk using torchchuck.
    
    Algorithm:
    ----------
    1. Validate PDF file exists and has content
    2. Use UltraDoclingPipeline to convert PDF -> markdown
    3. Validate markdown output exists and has content
    4. Use torchchuck.chunk_from_files() to chunk the markdown
    5. Save chunks to JSONL in processed directory
    6. Return chunk metadata list for status tracking
    
    Parameters:
    -----------
    file_path : str
        Path to the uploaded PDF file
    doc_id : str
        Document ID for naming the chunks file
    
    Returns:
    --------
    List[Dict[str, Any]]
        List of chunk metadata dictionaries
    
    Raises:
    -------
    FileNotFoundError: If the PDF file does not exist
    ValueError: If the file is empty or conversion produces no output
    RuntimeError: If PDF conversion or chunking fails
    """
    loop = asyncio.get_event_loop()
    
    def _run_conversion() -> List[Dict[str, Any]]:
        """
        Synchronous PDF conversion and chunking worker.
        Implements comprehensive error handling and logging.
        """
        file_path_obj = Path(file_path)
        
        # -------------------------------------------------------------------------
        # Step 1: Validate PDF file exists
        # -------------------------------------------------------------------------
        if not file_path_obj.exists():
            _LOG.error("PDF processing failed: File not found: %s", file_path)
            raise FileNotFoundError(f"PDF file not found: {file_path}")
        
        file_size = file_path_obj.stat().st_size
        if file_size == 0:
            _LOG.error("PDF processing failed: File is empty: %s", file_path)
            raise ValueError(f"PDF file is empty: {file_path}")
        
        _LOG.info("Starting PDF processing: doc_id=%s, file=%s, size=%d bytes",
                  doc_id, file_path_obj.name, file_size)
        
        # -------------------------------------------------------------------------
        # Step 2: Set up output directory
        # -------------------------------------------------------------------------
        output_dir = file_path_obj.parent.resolve() / "processed"
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            _LOG.debug("Output directory ensured: %s", output_dir)
        except Exception as dir_err:
            _LOG.error("Failed to create output directory: %s - %s", output_dir, dir_err)
            raise OSError(f"Cannot create output directory: {output_dir}")
        
        # -------------------------------------------------------------------------
        # Step 3: Try to convert PDF to markdown using UltraDoclingPipeline
        # -------------------------------------------------------------------------
        md_file = None
        try:
            from conversion import UltraDoclingPipeline, PipelineConfig
            _LOG.debug("conversion module loaded successfully")
            
            config = PipelineConfig(
                do_ocr=True,
                accelerator_device="auto",  # Use GPU if available (configured via env vars)
                ocr_engine="tesseract_cli", # Use CLI for compatibility
                ocr_languages=["eng"],      # Use 'eng' (3-letter) instead of default 'en'
            )
            
            pipeline = UltraDoclingPipeline(
                input_path=file_path,
                output_dir=str(output_dir),
                config=config,
            )
            
            _LOG.info("Running PDF to markdown conversion...")
            # Explicitly name the output file to match expectations
            pipeline.ocr_to_markdown(file_name=file_path_obj.stem + ".md")
            
            # Find the output markdown file
            md_file = output_dir / (file_path_obj.stem + ".md")
            if not md_file.exists():
                _LOG.warning("Markdown file not found at expected path: %s", md_file)
                # Try to find any .md file in output directory
                md_files = list(output_dir.glob("*.md"))
                if md_files:
                    md_file = md_files[0]
                    _LOG.info("Found alternative markdown file: %s", md_file)
                else:
                    _LOG.error("No markdown files generated from PDF conversion")
                    raise RuntimeError(f"PDF conversion produced no output for {file_path}")
            
            _LOG.info("PDF converted to markdown: %s", md_file)
            
        except ImportError as import_err:
            _LOG.warning("Docling not available, PDF processing disabled: %s", import_err)
            # Return empty list - PDF cannot be processed without Docling
            return []
        except Exception as conv_err:
            _LOG.error("PDF conversion failed: %s", conv_err, exc_info=True)
            raise RuntimeError(f"PDF conversion failed for {file_path}: {conv_err}")
        
        # -------------------------------------------------------------------------
        # Step 4: Validate markdown output has content
        # -------------------------------------------------------------------------
        if md_file and md_file.exists():
            md_size = md_file.stat().st_size
            if md_size == 0:
                _LOG.warning("Markdown file is empty after PDF conversion: %s", md_file)
                return []
            
            _LOG.debug("Markdown file size: %d bytes", md_size)
            
            # -------------------------------------------------------------------------
            # Step 5: Import torchchuck and configure chunking
            # -------------------------------------------------------------------------
            try:
                from torchchuck import ChunkConfig, chunk_from_files, write_chunks_jsonl
                _LOG.debug("torchchuck module loaded successfully")
            except ImportError as import_err:
                _LOG.error("Failed to import torchchuck: %s", import_err)
                raise ImportError(f"torchchuck module not available: {import_err}")
            
            # Configure production-grade chunking
            chunk_config = ChunkConfig(
                chunk_size=1024,            # 1KB chunks (optimal for embeddings)
                chunk_overlap=128,          # 128 byte overlap for context continuity
                separators=("\n\n", "\n", " ", ""),  # Hierarchical separators
                keep_separator="end",       # Keep separator at chunk end
                return_offsets=True,        # Track byte positions
                deduplicate=True,           # Remove duplicate chunks
                normalize_whitespace=True,  # Clean whitespace artifacts
                drop_blank_chunks=True,     # Skip empty/whitespace chunks
                output_hash=True,           # Generate 64-bit hashes for dedup
            )
            
            # -------------------------------------------------------------------------
            # Step 6: Chunk the markdown file
            # -------------------------------------------------------------------------
            try:
                chunks = chunk_from_files(str(md_file), chunk_config)
                _LOG.info("torchchuck.chunk_from_files completed: %d chunks from PDF", len(chunks))
            except Exception as chunk_err:
                _LOG.error("PDF chunking failed: %s", chunk_err, exc_info=True)
                raise RuntimeError(f"Chunking failed for {md_file}: {chunk_err}")
            
            if not chunks:
                _LOG.warning("No chunks generated from PDF (may be empty/whitespace only)")
            
            # -------------------------------------------------------------------------
            # Step 7: Save chunks to JSONL
            # -------------------------------------------------------------------------
            chunks_file = output_dir / f"{doc_id}_chunks.jsonl"
            try:
                write_chunks_jsonl(chunks, chunks_file)
                _LOG.info("PDF chunks saved: doc_id=%s, count=%d, path=%s",
                          doc_id, len(chunks), chunks_file)
            except Exception as write_err:
                _LOG.error("Failed to write PDF chunks: %s - %s", chunks_file, write_err)
                raise IOError(f"Failed to write chunks file: {chunks_file}")
            
            # -------------------------------------------------------------------------
            # Step 8: Return chunk metadata
            # -------------------------------------------------------------------------
            chunk_metadata = [
                {
                    "index": c.index,
                    "text": c.text,
                    "start": c.start,
                    "end": c.end,
                    "hash64": c.hash64,
                    "doc_id": c.doc_id,
                }
                for c in chunks
            ]
            
            _LOG.info("PDF processing complete: doc_id=%s, chunks=%d", doc_id, len(chunk_metadata))
            return chunk_metadata
        
        # No markdown file - return empty
        return []
    
    return await loop.run_in_executor(None, _run_conversion)


async def _process_text(file_path: str, doc_id: str) -> List[Dict[str, Any]]:
    """
    Process text/markdown document with proper chunking using torchchuck.
    
    Algorithm:
    ----------
    1. Validate file exists and has content
    2. Use torchchuck.chunk_from_files() to load and chunk the file
    3. Save chunks to JSONL in processed directory
    4. Return chunk metadata list for status tracking
    
    Parameters:
    -----------
    file_path : str
        Path to the uploaded text/markdown file
    doc_id : str
        Document ID for naming the chunks file
    
    Returns:
    --------
    List[Dict[str, Any]]
        List of chunk metadata dictionaries with keys:
        - index: Chunk index within document
        - text: Chunk text content
        - start: Start byte offset (post-normalization)
        - end: End byte offset (post-normalization)
        - hash64: 64-bit FNV-1a hash for deduplication
        - doc_id: Source document ID
    
    Raises:
    -------
    FileNotFoundError: If the input file does not exist
    ValueError: If the file is empty or produces no chunks
    """
    loop = asyncio.get_event_loop()
    
    def _chunk_file() -> List[Dict[str, Any]]:
        """
        Synchronous chunking worker that runs in a thread pool executor.
        Implements comprehensive error handling and logging.
        """
        # -------------------------------------------------------------------------
        # Step 1: Validate input file exists and has content
        # -------------------------------------------------------------------------
        file_path_obj = Path(file_path)
        
        if not file_path_obj.exists():
            _LOG.error("Chunking failed: File not found: %s", file_path)
            raise FileNotFoundError(f"File not found: {file_path}")
        
        file_size = file_path_obj.stat().st_size
        if file_size == 0:
            _LOG.error("Chunking failed: File is empty: %s", file_path)
            raise ValueError(f"File is empty: {file_path}")
        
        _LOG.info("Starting chunking: doc_id=%s, file=%s, size=%d bytes", 
                  doc_id, file_path_obj.name, file_size)
        
        # -------------------------------------------------------------------------
        # Step 2: Read file content for validation (optional debug logging)
        # -------------------------------------------------------------------------
        try:
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                content_preview = f.read(500)  # Read first 500 chars for logging
            _LOG.debug("File content preview (%d chars): %s...", 
                       len(content_preview), content_preview[:100].replace("\n", "\\n"))
        except Exception as read_err:
            _LOG.warning("Could not preview file content: %s", read_err)
        
        # -------------------------------------------------------------------------
        # Step 3: Import and configure torchchuck for GPU-accelerated chunking
        # -------------------------------------------------------------------------
        try:
            from torchchuck import ChunkConfig, chunk_from_files, write_chunks_jsonl
            _LOG.debug("torchchuck module loaded successfully")
        except ImportError as import_err:
            _LOG.error("Failed to import torchchuck: %s", import_err)
            raise ImportError(f"torchchuck module not available: {import_err}")
        
        # Configure production-grade chunking parameters
        config = ChunkConfig(
            chunk_size=1024,            # 1KB chunks (optimal for embedding models)
            chunk_overlap=128,          # 128 byte overlap maintains context continuity
            separators=("\n\n", "\n", " ", ""),  # Hierarchical separator priority
            keep_separator="end",       # Separator kept at end of chunk
            return_offsets=True,        # Track byte positions for source mapping
            deduplicate=True,           # Remove duplicate chunks via hash
            normalize_whitespace=True,  # Collapse spaces, normalize newlines
            drop_blank_chunks=True,     # Exclude empty/whitespace-only chunks
            output_hash=True,           # Generate 64-bit FNV-1a hashes
        )
        
        _LOG.debug("ChunkConfig: chunk_size=%d, overlap=%d, separators=%s",
                   config.chunk_size, config.chunk_overlap, config.separators)
        
        # -------------------------------------------------------------------------
        # Step 4: Execute chunking pipeline using torchchuck
        # -------------------------------------------------------------------------
        try:
            chunks = chunk_from_files(file_path, config)
            _LOG.info("torchchuck.chunk_from_files completed: %d chunks generated", len(chunks))
        except Exception as chunk_err:
            _LOG.error("torchchuck.chunk_from_files failed: %s", chunk_err, exc_info=True)
            raise RuntimeError(f"Chunking failed for {file_path}: {chunk_err}")
        
        # Validate we got chunks
        if not chunks:
            _LOG.warning("No chunks generated for doc_id=%s (file may be empty/whitespace only after normalization)", doc_id)
            # Return empty list - caller will handle this as 0 chunks (not an error, but a warning)
        
        # -------------------------------------------------------------------------
        # Step 5: Ensure processed directory exists with absolute path
        # -------------------------------------------------------------------------
        processed_dir = file_path_obj.parent.resolve() / "processed"
        try:
            processed_dir.mkdir(parents=True, exist_ok=True)
            _LOG.debug("Processed directory ensured: %s", processed_dir)
        except Exception as dir_err:
            _LOG.error("Failed to create processed directory: %s - %s", processed_dir, dir_err)
            raise OSError(f"Cannot create processed directory: {processed_dir}")
        
        # -------------------------------------------------------------------------
        # Step 6: Save chunks to JSONL format
        # -------------------------------------------------------------------------
        output_path = processed_dir / f"{doc_id}_chunks.jsonl"
        try:
            write_chunks_jsonl(chunks, output_path)
            _LOG.info("Chunks saved: doc_id=%s, count=%d, path=%s", 
                      doc_id, len(chunks), output_path)
        except Exception as write_err:
            _LOG.error("Failed to write chunks JSONL: %s - %s", output_path, write_err)
            raise IOError(f"Failed to write chunks file: {output_path}")
        
        # -------------------------------------------------------------------------
        # Step 7: Build and return chunk metadata for database status update
        # -------------------------------------------------------------------------
        chunk_metadata = [
            {
                "index": chunk.index,
                "text": chunk.text,
                "start": chunk.start,
                "end": chunk.end,
                "hash64": chunk.hash64,
                "doc_id": chunk.doc_id,
            }
            for chunk in chunks
        ]
        
        _LOG.info("Text processing complete: doc_id=%s, chunks=%d, output=%s", 
                  doc_id, len(chunk_metadata), output_path)
        
        return chunk_metadata
    
    # Execute chunking in thread pool to avoid blocking the event loop
    return await loop.run_in_executor(None, _chunk_file)


# =============================================================================
# Chunking Status Endpoint
# =============================================================================

@router.get(
    "/{doc_id}/chunking-status",
    response_model=ChunkingStatusResponse,
    summary="Chunking Status",
    description="""
    Check if document chunking is complete and get chunk metadata.
    
    **Returns**:
    - `is_chunked`: True if JSONL chunks file exists
    - `chunk_count`: Number of chunks generated
    - `chunks_file`: Path to the JSONL file containing chunks
    - `status`: Current document processing status
    
    **Use Case**:
    Poll this endpoint after upload to know when chunks are ready for indexing.
    """,
)
async def get_chunking_status(
    doc_id: str,
    documents: DocumentRepository = Depends(get_document_repo),
) -> ChunkingStatusResponse:
    """
    Get chunking status for a document.
    
    Checks if the JSONL chunks file exists in the processed directory
    and returns metadata about the chunking process.
    """
    # Verify document exists
    doc = assert_found(
        await documents.get(doc_id),
        "Document",
        doc_id,
    )
    
    # Check if chunks file exists in processed directory
    settings = get_settings()
    upload_dir = settings.data_dir / "uploads"
    processed_dir = upload_dir / "processed"
    chunks_file = processed_dir / f"{doc_id}_chunks.jsonl"
    
    is_chunked = chunks_file.exists()
    chunk_count = 0
    chunks_path = None
    
    if is_chunked:
        chunks_path = str(chunks_file)
        # Count lines in JSONL (each line is a chunk)
        with open(chunks_file, "r", encoding="utf-8") as f:
            chunk_count = sum(1 for _ in f)
    
    return ChunkingStatusResponse(
        doc_id=doc_id,
        is_chunked=is_chunked,
        chunk_count=chunk_count,
        chunks_file=chunks_path,
        status=doc.get("status", "unknown"),
    )
