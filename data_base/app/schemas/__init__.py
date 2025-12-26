# ==============================================================================
# SCHEMAS PACKAGE INITIALIZATION
# ==============================================================================

"""
Pydantic Schemas
================

Request/Response validation schemas for API endpoints:
- Base: Common schemas and pagination
- User: Authentication and profile schemas
- Chat: Message and session schemas
- Transaction: Financial operation schemas
- Product/Order: E-commerce schemas
- Project/Task: Project management schemas
- Course: LMS schemas
"""

from app.schemas.base import (
    BaseSchema,
    TimestampSchema,
    PaginatedResponse,
    APIResponse,
)
from app.schemas.user import (
    UserCreate,
    UserUpdate,
    UserResponse,
    UserLogin,
    TokenResponse,
)
from app.schemas.chat import (
    ChatSessionCreate,
    ChatSessionUpdate,
    ChatSessionResponse,
    ChatMessageCreate,
    ChatMessageResponse,
)
from app.schemas.transaction import (
    TransactionCreate,
    TransactionUpdate,
    TransactionResponse,
)
from app.schemas.product import (
    ProductCreate,
    ProductUpdate,
    ProductResponse,
)
from app.schemas.order import (
    OrderCreate,
    OrderItemCreate,
    OrderResponse,
    OrderItemResponse,
)
from app.schemas.project import (
    ProjectCreate,
    ProjectUpdate,
    ProjectResponse,
    TaskCreate,
    TaskUpdate,
    TaskResponse,
)
from app.schemas.course import (
    CourseCreate,
    CourseUpdate,
    CourseResponse,
    CourseSectionCreate,
    CourseSectionResponse,
    EnrollmentCreate,
    EnrollmentResponse,
)

__all__ = [
    # Base
    "BaseSchema",
    "TimestampSchema",
    "PaginatedResponse",
    "APIResponse",
    # User
    "UserCreate",
    "UserUpdate",
    "UserResponse",
    "UserLogin",
    "TokenResponse",
    # Chat
    "ChatSessionCreate",
    "ChatSessionUpdate",
    "ChatSessionResponse",
    "ChatMessageCreate",
    "ChatMessageResponse",
    # Transaction
    "TransactionCreate",
    "TransactionUpdate",
    "TransactionResponse",
    # Product
    "ProductCreate",
    "ProductUpdate",
    "ProductResponse",
    # Order
    "OrderCreate",
    "OrderItemCreate",
    "OrderResponse",
    "OrderItemResponse",
    # Project
    "ProjectCreate",
    "ProjectUpdate",
    "ProjectResponse",
    "TaskCreate",
    "TaskUpdate",
    "TaskResponse",
    # Course
    "CourseCreate",
    "CourseUpdate",
    "CourseResponse",
    "CourseSectionCreate",
    "CourseSectionResponse",
    "EnrollmentCreate",
    "EnrollmentResponse",
]
