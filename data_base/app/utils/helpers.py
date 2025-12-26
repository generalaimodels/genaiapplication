# ==============================================================================
# HELPER UTILITIES
# ==============================================================================
# Common utility functions used across the application
# ==============================================================================

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, TypeVar
from uuid import uuid4
import math

T = TypeVar("T")


def generate_uuid() -> str:
    """Generate a new UUID string."""
    return str(uuid4())


def generate_reference_id(prefix: str = "REF", length: int = 8) -> str:
    """
    Generate a unique reference ID.
    
    Args:
        prefix: ID prefix (e.g., TXN, ORD)
        length: Length of random part
        
    Returns:
        Formatted reference ID (e.g., TXN-A1B2C3D4)
    """
    random_part = str(uuid4()).replace("-", "")[:length].upper()
    return f"{prefix}-{random_part}"


def utc_now() -> datetime:
    """Get current UTC datetime."""
    return datetime.now(timezone.utc)


def paginate_results(
    items: List[T],
    page: int,
    page_size: int,
    total: int,
) -> Dict[str, Any]:
    """
    Create a pagination response dict.
    
    Args:
        items: List of items for current page
        page: Current page number (1-indexed)
        page_size: Items per page
        total: Total item count
        
    Returns:
        Pagination metadata dict
    """
    pages = math.ceil(total / page_size) if total > 0 else 0
    
    return {
        "items": items,
        "total": total,
        "page": page,
        "page_size": page_size,
        "pages": pages,
        "has_next": page < pages,
        "has_prev": page > 1,
    }


def calculate_offset(page: int, page_size: int) -> int:
    """Calculate database offset from page number."""
    return (page - 1) * page_size


def sanitize_string(value: str, max_length: int = 255) -> str:
    """
    Sanitize a string by trimming and truncating.
    
    Args:
        value: Input string
        max_length: Maximum allowed length
        
    Returns:
        Sanitized string
    """
    if not value:
        return ""
    return value.strip()[:max_length]
