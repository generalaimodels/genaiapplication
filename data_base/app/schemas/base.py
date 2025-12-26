# ==============================================================================
# BASE SCHEMAS - Common Schema Patterns
# ==============================================================================
# Foundation schemas for API responses and pagination
# ==============================================================================

from __future__ import annotations

from datetime import datetime
from typing import Any, Generic, List, Optional, TypeVar

from pydantic import BaseModel, ConfigDict, Field


# Type variable for generic response types
T = TypeVar("T")


class BaseSchema(BaseModel):
    """
    Base schema with common configuration.
    
    All API schemas should inherit from this class
    to ensure consistent serialization behavior.
    """
    
    model_config = ConfigDict(
        from_attributes=True,  # Enable ORM mode
        populate_by_name=True,
        use_enum_values=True,
        arbitrary_types_allowed=True,
        json_schema_extra={
            "example": {}
        }
    )


class TimestampSchema(BaseSchema):
    """Schema with automatic timestamp fields."""
    
    created_at: Optional[datetime] = Field(
        None,
        description="Record creation timestamp"
    )
    updated_at: Optional[datetime] = Field(
        None,
        description="Last update timestamp"
    )


class PaginatedResponse(BaseModel, Generic[T]):
    """
    Generic paginated response wrapper.
    
    Provides consistent pagination metadata for list endpoints.
    
    Attributes:
        items: List of result items
        total: Total number of matching items
        page: Current page number (1-indexed)
        page_size: Items per page
        pages: Total number of pages
    """
    
    items: List[T] = Field(
        default_factory=list,
        description="List of items"
    )
    total: int = Field(
        ...,
        description="Total number of items"
    )
    page: int = Field(
        ...,
        ge=1,
        description="Current page number"
    )
    page_size: int = Field(
        ...,
        ge=1,
        le=100,
        description="Items per page"
    )
    pages: int = Field(
        ...,
        ge=0,
        description="Total number of pages"
    )
    
    @property
    def has_next(self) -> bool:
        """Check if there's a next page."""
        return self.page < self.pages
    
    @property
    def has_prev(self) -> bool:
        """Check if there's a previous page."""
        return self.page > 1


class APIResponse(BaseModel, Generic[T]):
    """
    Standard API response wrapper.
    
    Provides consistent response structure for all API endpoints.
    
    Attributes:
        success: Whether the request was successful
        message: Optional status message
        data: Response payload
        errors: Optional error details
    """
    
    success: bool = Field(
        True,
        description="Whether the request was successful"
    )
    message: Optional[str] = Field(
        None,
        description="Status message"
    )
    data: Optional[T] = Field(
        None,
        description="Response data"
    )
    errors: Optional[List[dict[str, Any]]] = Field(
        None,
        description="Error details if any"
    )
    
    @classmethod
    def ok(
        cls,
        data: T,
        message: Optional[str] = None,
    ) -> "APIResponse[T]":
        """Create a successful response."""
        return cls(success=True, data=data, message=message)
    
    @classmethod
    def error(
        cls,
        message: str,
        errors: Optional[List[dict[str, Any]]] = None,
    ) -> "APIResponse[T]":
        """Create an error response."""
        return cls(success=False, message=message, errors=errors)


class HealthResponse(BaseSchema):
    """Health check response schema."""
    
    status: str = Field(
        ...,
        description="Health status"
    )
    version: str = Field(
        ...,
        description="Application version"
    )
    database: str = Field(
        ...,
        description="Database connection status"
    )


class QueryParams(BaseModel):
    """Common query parameters for list endpoints."""
    
    skip: int = Field(
        0,
        ge=0,
        description="Number of records to skip"
    )
    limit: int = Field(
        10,
        ge=1,
        le=100,
        description="Maximum records to return"
    )
    sort_by: Optional[str] = Field(
        None,
        description="Field to sort by"
    )
    sort_order: str = Field(
        "asc",
        pattern="^(asc|desc)$",
        description="Sort direction"
    )
