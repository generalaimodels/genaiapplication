# ==============================================================================
# SERVICES PACKAGE INITIALIZATION
# ==============================================================================

"""
Service Layer
=============

Business logic implementation for all domain entities:
- BaseService: Generic service with common operations
- UserService: Authentication and user management
- ChatService: Chat session and message handling
- TransactionService: Financial operations
- ProductService: Product catalog management
- OrderService: Order processing
- ProjectService: Project management
- CourseService: LMS functionality
"""

from app.services.base_service import BaseService
from app.services.user_service import UserService
from app.services.chat_service import ChatService
from app.services.transaction_service import TransactionService

__all__ = [
    "BaseService",
    "UserService",
    "ChatService",
    "TransactionService",
]
