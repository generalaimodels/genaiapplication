# ==============================================================================
# DATABASE PACKAGE INITIALIZATION
# ==============================================================================
# Database Abstraction Layer with multi-database support
# ==============================================================================

"""
Database Module
===============

Provides a unified database abstraction layer supporting:
- SQLite (development/testing)
- PostgreSQL (production)
- MongoDB (document store)

Key Components:
- Adapters: Database-specific implementations
- Factory: Dynamic adapter instantiation
- Repositories: Data access abstraction
- Unit of Work: Transaction management
"""

from app.database.factory import DatabaseFactory
from app.database.adapters.base_adapter import BaseDatabaseAdapter

__all__ = [
    "DatabaseFactory",
    "BaseDatabaseAdapter",
]
