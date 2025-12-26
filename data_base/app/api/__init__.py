# ==============================================================================
# API PACKAGE INITIALIZATION
# ==============================================================================

"""
API Module
==========

FastAPI routers and endpoint definitions:
- Dependencies: Authentication, database access
- Routers: User, Chat, Transaction, Product, Order, Project, Course
"""

from app.api.router import api_router

__all__ = ["api_router"]
