# ==============================================================================
# APP PACKAGE INITIALIZATION
# ==============================================================================
# Generalized Multi-Database Backend System with FastAPI
# Supports: SQLite, PostgreSQL, MongoDB
# Architecture: Repository Pattern, Unit of Work, Factory Pattern
# ==============================================================================

"""
Generalized Multi-Database Backend System
==========================================

A production-ready FastAPI backend supporting multiple database backends
through a unified abstraction layer.

Features:
---------
- Multi-database support (SQLite, PostgreSQL, MongoDB)
- Repository Pattern for data access abstraction
- Unit of Work Pattern for transactional consistency
- Factory Pattern for dynamic adapter instantiation
- JWT-based authentication
- Rate limiting and request throttling
- Comprehensive logging and error handling

Usage:
------
    from app.main import app
    
    # Run with uvicorn
    uvicorn app.main:app --reload

Author: AI-Generated SOTA Implementation
Version: 1.0.0
"""

__version__ = "1.0.0"
__author__ = "AI-Generated SOTA Implementation"
__all__ = ["__version__", "__author__"]
