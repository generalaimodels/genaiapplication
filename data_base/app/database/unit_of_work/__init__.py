# ==============================================================================
# UNIT OF WORK PACKAGE INITIALIZATION
# ==============================================================================

"""
Unit of Work Pattern Implementation
===================================

Provides transactional consistency across repository operations:
- UnitOfWork: Coordinates transactions across repositories
- get_unit_of_work: Dependency injection helper
"""

from app.database.unit_of_work.uow import UnitOfWork, get_unit_of_work

__all__ = [
    "UnitOfWork",
    "get_unit_of_work",
]
