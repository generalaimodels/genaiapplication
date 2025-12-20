# =============================================================================
# DOCUMENT PROCESSOR - Document Ingestion and Processing
# =============================================================================
# Handles loading, parsing, and chunking of documents for RAG.
# =============================================================================

from __future__ import annotations
import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class Document:
    """
    Represents a processed document or chunk.
    
    Attributes:
        content: Text content
        metadata: Document metadata
        doc_id: Unique document identifier
        source: Source file path or URL
        chunk_index: Index if this is a chunk
        embeddings: Pre-computed embeddings
    """
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    doc_id: Optional[str] = None
    source: Optional[str] = None
    chunk_index: Optional[int] = None
    embeddings: Optional[List[float]] = None
    
    @property
    def is_chunk(self) -> bool:
        """Check if this is a chunk."""
        return self.chunk_index is not None
    
    @property
    def word_count(self) -> int:
        """Get approximate word count."""
        return len(self.content.split())
    
    @property
    def char_count(self) -> int:
        """Get character count."""
        return len(self.content)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "content": self.content,
            "metadata": self.metadata,
            "doc_id": self.doc_id,
            "source": self.source,
            "chunk_index": self.chunk_index,
        }


class DocumentProcessor:
    """
    Processes documents for RAG pipelines.
    
    Features:
        - Load documents from files or text
        - Split into chunks with overlap
        - Extract metadata
        - Support for multiple formats
    
    Example:
        >>> processor = DocumentProcessor(chunk_size=1000, chunk_overlap=200)
        >>> 
        >>> # From file
        >>> docs = processor.load_file("report.txt")
        >>> 
        >>> # From text
        >>> docs = processor.load_text("Long document text here...")
        >>> 
        >>> # Chunk documents
        >>> chunks = processor.chunk_documents(docs)
    """
    
    SUPPORTED_EXTENSIONS = {".txt", ".md", ".py", ".json", ".csv", ".html"}
    
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        min_chunk_size: int = 100,
        separator: str = "\n\n"
    ):
        """
        Initialize document processor.
        
        Args:
            chunk_size: Target chunk size in characters
            chunk_overlap: Overlap between chunks
            min_chunk_size: Minimum chunk size
            separator: Text separator for splitting
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
        self.separator = separator
        self._doc_counter = 0
    
    def _generate_doc_id(self) -> str:
        """Generate unique document ID."""
        self._doc_counter += 1
        return f"doc-{self._doc_counter}"
    
    def load_text(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
        source: Optional[str] = None
    ) -> Document:
        """
        Load document from text.
        
        Args:
            text: Document text
            metadata: Additional metadata
            source: Source identifier
            
        Returns:
            Document object
        """
        return Document(
            content=text,
            metadata=metadata or {},
            doc_id=self._generate_doc_id(),
            source=source or "text_input",
        )
    
    def load_file(
        self,
        file_path: Union[str, Path],
        encoding: str = "utf-8"
    ) -> Document:
        """
        Load document from file.
        
        Args:
            file_path: Path to file
            encoding: File encoding
            
        Returns:
            Document object
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file type not supported
        """
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        ext = path.suffix.lower()
        
        # Read file content
        try:
            with open(path, "r", encoding=encoding) as f:
                content = f.read()
        except Exception as e:
            logger.error(f"Error reading file {file_path}: {e}")
            raise
        
        # Build metadata
        metadata = {
            "filename": path.name,
            "extension": ext,
            "size_bytes": path.stat().st_size,
            "modified_at": datetime.fromtimestamp(path.stat().st_mtime).isoformat(),
        }
        
        return Document(
            content=content,
            metadata=metadata,
            doc_id=self._generate_doc_id(),
            source=str(path.absolute()),
        )
    
    def load_files(
        self,
        file_paths: List[Union[str, Path]],
        encoding: str = "utf-8"
    ) -> List[Document]:
        """
        Load multiple documents from files.
        
        Args:
            file_paths: List of file paths
            encoding: File encoding
            
        Returns:
            List of Document objects
        """
        documents = []
        for path in file_paths:
            try:
                doc = self.load_file(path, encoding)
                documents.append(doc)
            except Exception as e:
                logger.warning(f"Skipping file {path}: {e}")
        return documents
    
    def chunk_text(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """
        Split text into chunks.
        
        Args:
            text: Text to chunk
            metadata: Metadata to attach to chunks
            
        Returns:
            List of Document chunks
        """
        if len(text) <= self.chunk_size:
            return [Document(
                content=text,
                metadata=metadata or {},
                doc_id=self._generate_doc_id(),
                chunk_index=0
            )]
        
        chunks = []
        start = 0
        chunk_idx = 0
        
        while start < len(text):
            # Calculate end position
            end = start + self.chunk_size
            
            # Try to find a natural break point
            if end < len(text):
                # Look for separator within last portion
                break_zone_start = max(start, end - 200)
                break_point = text.rfind(self.separator, break_zone_start, end)
                
                if break_point > start:
                    end = break_point + len(self.separator)
                else:
                    # Fall back to space
                    space_point = text.rfind(" ", break_zone_start, end)
                    if space_point > start:
                        end = space_point + 1
            
            chunk_text = text[start:end].strip()
            
            if len(chunk_text) >= self.min_chunk_size:
                chunks.append(Document(
                    content=chunk_text,
                    metadata={**(metadata or {}), "chunk_of": len(text)},
                    doc_id=self._generate_doc_id(),
                    chunk_index=chunk_idx
                ))
                chunk_idx += 1
            
            # Move start with overlap
            start = end - self.chunk_overlap
            if start <= 0 or start >= len(text):
                break
        
        return chunks
    
    def chunk_document(self, document: Document) -> List[Document]:
        """
        Split a document into chunks.
        
        Args:
            document: Document to chunk
            
        Returns:
            List of document chunks
        """
        chunks = self.chunk_text(
            document.content,
            metadata={
                **document.metadata,
                "source": document.source,
                "original_doc_id": document.doc_id
            }
        )
        
        # Update source reference
        for chunk in chunks:
            chunk.source = document.source
        
        return chunks
    
    def chunk_documents(
        self,
        documents: List[Document]
    ) -> List[Document]:
        """
        Split multiple documents into chunks.
        
        Args:
            documents: Documents to chunk
            
        Returns:
            List of all chunks
        """
        all_chunks = []
        for doc in documents:
            chunks = self.chunk_document(doc)
            all_chunks.extend(chunks)
        return all_chunks
    
    def process_content(
        self,
        content: Union[str, List[str], Path, List[Path]],
        auto_chunk: bool = True
    ) -> List[Document]:
        """
        Process content from various sources.
        
        Args:
            content: Text, file path(s), or list of either
            auto_chunk: Whether to automatically chunk
            
        Returns:
            List of processed documents/chunks
        """
        documents = []
        
        # Normalize to list
        if isinstance(content, (str, Path)):
            content = [content]
        
        for item in content:
            if isinstance(item, Path) or (isinstance(item, str) and os.path.exists(item)):
                doc = self.load_file(item)
            else:
                doc = self.load_text(str(item))
            documents.append(doc)
        
        if auto_chunk:
            return self.chunk_documents(documents)
        
        return documents
