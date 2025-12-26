# ==============================================================================
# APPLICATION CONSTANTS - Centralized Configuration Values
# ==============================================================================
# Immutable constants used throughout the application
# Organized by category for easy maintenance
# ==============================================================================

from __future__ import annotations

from typing import Final


# ==============================================================================
# API CONSTANTS
# ==============================================================================

class APIConstants:
    """API-related constants."""
    
    # Pagination defaults
    DEFAULT_PAGE_SIZE: Final[int] = 10
    MAX_PAGE_SIZE: Final[int] = 100
    MIN_PAGE_SIZE: Final[int] = 1
    
    # Request timeouts (seconds)
    REQUEST_TIMEOUT: Final[int] = 30
    LONG_REQUEST_TIMEOUT: Final[int] = 120
    
    # Response headers
    RATE_LIMIT_HEADER: Final[str] = "X-RateLimit-Limit"
    RATE_LIMIT_REMAINING_HEADER: Final[str] = "X-RateLimit-Remaining"
    RATE_LIMIT_RESET_HEADER: Final[str] = "X-RateLimit-Reset"
    REQUEST_ID_HEADER: Final[str] = "X-Request-ID"
    
    # Content types
    JSON_CONTENT_TYPE: Final[str] = "application/json"
    MULTIPART_CONTENT_TYPE: Final[str] = "multipart/form-data"


# ==============================================================================
# DATABASE CONSTANTS
# ==============================================================================

class DatabaseConstants:
    """Database-related constants."""
    
    # Collection/Table names
    USERS_COLLECTION: Final[str] = "users"
    CHAT_SESSIONS_COLLECTION: Final[str] = "chat_sessions"
    CHAT_MESSAGES_COLLECTION: Final[str] = "chat_messages"
    TRANSACTIONS_COLLECTION: Final[str] = "transactions"
    PRODUCTS_COLLECTION: Final[str] = "products"
    ORDERS_COLLECTION: Final[str] = "orders"
    ORDER_ITEMS_COLLECTION: Final[str] = "order_items"
    PROJECTS_COLLECTION: Final[str] = "projects"
    TASKS_COLLECTION: Final[str] = "tasks"
    COURSES_COLLECTION: Final[str] = "courses"
    COURSE_SECTIONS_COLLECTION: Final[str] = "course_sections"
    ENROLLMENTS_COLLECTION: Final[str] = "enrollments"
    
    # Query limits
    MAX_BATCH_SIZE: Final[int] = 1000
    DEFAULT_QUERY_LIMIT: Final[int] = 100
    
    # Connection settings
    CONNECTION_RETRY_ATTEMPTS: Final[int] = 3
    CONNECTION_RETRY_DELAY: Final[float] = 1.0  # seconds


# ==============================================================================
# SECURITY CONSTANTS
# ==============================================================================

class SecurityConstants:
    """Security-related constants."""
    
    # Password requirements
    MIN_PASSWORD_LENGTH: Final[int] = 8
    MAX_PASSWORD_LENGTH: Final[int] = 128
    
    # Token settings
    TOKEN_TYPE_BEARER: Final[str] = "bearer"
    AUTH_HEADER_PREFIX: Final[str] = "Bearer"
    
    # Rate limiting
    DEFAULT_RATE_LIMIT: Final[int] = 100
    DEFAULT_RATE_WINDOW: Final[int] = 60  # seconds
    
    # Session settings
    SESSION_COOKIE_NAME: Final[str] = "session_id"
    CSRF_TOKEN_NAME: Final[str] = "csrf_token"


# ==============================================================================
# MESSAGE ROLE CONSTANTS
# ==============================================================================

class MessageRoles:
    """Chat message role constants."""
    
    USER: Final[str] = "user"
    ASSISTANT: Final[str] = "assistant"
    SYSTEM: Final[str] = "system"
    
    @classmethod
    def all_roles(cls) -> list[str]:
        """Get all valid message roles."""
        return [cls.USER, cls.ASSISTANT, cls.SYSTEM]


# ==============================================================================
# TRANSACTION CONSTANTS
# ==============================================================================

class TransactionConstants:
    """Transaction-related constants."""
    
    # Transaction types
    TYPE_CREDIT: Final[str] = "credit"
    TYPE_DEBIT: Final[str] = "debit"
    TYPE_TRANSFER: Final[str] = "transfer"
    
    # Transaction statuses
    STATUS_PENDING: Final[str] = "pending"
    STATUS_COMPLETED: Final[str] = "completed"
    STATUS_FAILED: Final[str] = "failed"
    STATUS_CANCELLED: Final[str] = "cancelled"
    
    # Currency
    DEFAULT_CURRENCY: Final[str] = "USD"
    
    # Reference ID format
    REFERENCE_PREFIX: Final[str] = "TXN"
    REFERENCE_LENGTH: Final[int] = 12


# ==============================================================================
# PROJECT CONSTANTS
# ==============================================================================

class ProjectConstants:
    """Project management constants."""
    
    # Project statuses
    STATUS_PLANNING: Final[str] = "planning"
    STATUS_IN_PROGRESS: Final[str] = "in_progress"
    STATUS_ON_HOLD: Final[str] = "on_hold"
    STATUS_COMPLETED: Final[str] = "completed"
    STATUS_CANCELLED: Final[str] = "cancelled"
    
    # Task priorities
    PRIORITY_LOW: Final[str] = "low"
    PRIORITY_MEDIUM: Final[str] = "medium"
    PRIORITY_HIGH: Final[str] = "high"
    PRIORITY_CRITICAL: Final[str] = "critical"


# ==============================================================================
# ORDER CONSTANTS
# ==============================================================================

class OrderConstants:
    """E-commerce order constants."""
    
    # Order statuses
    STATUS_PENDING: Final[str] = "pending"
    STATUS_CONFIRMED: Final[str] = "confirmed"
    STATUS_PROCESSING: Final[str] = "processing"
    STATUS_SHIPPED: Final[str] = "shipped"
    STATUS_DELIVERED: Final[str] = "delivered"
    STATUS_CANCELLED: Final[str] = "cancelled"
    STATUS_REFUNDED: Final[str] = "refunded"


# ==============================================================================
# ERROR MESSAGES
# ==============================================================================

class ErrorMessages:
    """Standardized error messages."""
    
    # Authentication
    INVALID_CREDENTIALS: Final[str] = "Invalid email or password"
    TOKEN_EXPIRED: Final[str] = "Authentication token has expired"
    TOKEN_INVALID: Final[str] = "Invalid authentication token"
    UNAUTHORIZED: Final[str] = "Authentication required"
    
    # Authorization
    PERMISSION_DENIED: Final[str] = "You don't have permission to perform this action"
    RESOURCE_FORBIDDEN: Final[str] = "Access to this resource is forbidden"
    
    # Resources
    USER_NOT_FOUND: Final[str] = "User not found"
    SESSION_NOT_FOUND: Final[str] = "Session not found"
    TRANSACTION_NOT_FOUND: Final[str] = "Transaction not found"
    PRODUCT_NOT_FOUND: Final[str] = "Product not found"
    ORDER_NOT_FOUND: Final[str] = "Order not found"
    PROJECT_NOT_FOUND: Final[str] = "Project not found"
    TASK_NOT_FOUND: Final[str] = "Task not found"
    COURSE_NOT_FOUND: Final[str] = "Course not found"
    
    # Validation
    INVALID_INPUT: Final[str] = "Invalid input provided"
    INVALID_EMAIL: Final[str] = "Invalid email address"
    WEAK_PASSWORD: Final[str] = "Password does not meet security requirements"
    
    # Business rules
    INSUFFICIENT_FUNDS: Final[str] = "Insufficient funds for this transaction"
    INSUFFICIENT_STOCK: Final[str] = "Insufficient stock available"
    
    # Rate limiting
    RATE_LIMIT_EXCEEDED: Final[str] = "Rate limit exceeded. Please try again later"


# ==============================================================================
# SUCCESS MESSAGES
# ==============================================================================

class SuccessMessages:
    """Standardized success messages."""
    
    # Generic
    CREATED: Final[str] = "Resource created successfully"
    UPDATED: Final[str] = "Resource updated successfully"
    DELETED: Final[str] = "Resource deleted successfully"
    
    # Authentication
    LOGIN_SUCCESS: Final[str] = "Login successful"
    LOGOUT_SUCCESS: Final[str] = "Logout successful"
    PASSWORD_CHANGED: Final[str] = "Password changed successfully"
    
    # Registration
    USER_REGISTERED: Final[str] = "User registered successfully"
    EMAIL_VERIFIED: Final[str] = "Email verified successfully"
    
    # Transactions
    TRANSACTION_CREATED: Final[str] = "Transaction created successfully"
    TRANSACTION_COMPLETED: Final[str] = "Transaction completed successfully"
    
    # Orders
    ORDER_PLACED: Final[str] = "Order placed successfully"
    ORDER_CANCELLED: Final[str] = "Order cancelled successfully"
