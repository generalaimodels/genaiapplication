# ==============================================================================
# REPOSITORIES PACKAGE INITIALIZATION
# ==============================================================================

"""
Repository Pattern Implementation
=================================

Provides data access abstraction through the Repository Pattern:
- BaseRepository: Generic repository interface
- Domain-specific repositories for each entity
"""

from app.database.repositories.base_repository import BaseRepository

__all__ = [
    "BaseRepository",
]
