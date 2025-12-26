# ==============================================================================
# DOMAIN MODELS PACKAGE INITIALIZATION
# ==============================================================================

"""
Domain Models
=============

SQLAlchemy ORM models for database entities:
- User: Authentication and user management
- Chat: Chat sessions and messages
- Transaction: Financial transactions
- Product/Order: E-commerce entities
- Project/Task: Project management
- Course: Learning management
"""

from app.domain_models.base import SQLBase, TimestampMixin
from app.domain_models.user import User
from app.domain_models.chat import ChatSession, ChatMessage
from app.domain_models.transaction import Transaction
from app.domain_models.product import Product
from app.domain_models.order import Order, OrderItem
from app.domain_models.project import Project, Task
from app.domain_models.course import Course, CourseSection, Enrollment

__all__ = [
    "SQLBase",
    "TimestampMixin",
    "User",
    "ChatSession",
    "ChatMessage",
    "Transaction",
    "Product",
    "Order",
    "OrderItem",
    "Project",
    "Task",
    "Course",
    "CourseSection",
    "Enrollment",
]
