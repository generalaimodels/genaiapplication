# ==============================================================================
# PRODUCT SCHEMAS - E-commerce Catalog
# ==============================================================================
# Request/Response schemas for product management
# ==============================================================================

from __future__ import annotations

from decimal import Decimal
from typing import Optional

from pydantic import Field, field_validator

from app.schemas.base import BaseSchema, TimestampSchema


class ProductCreate(BaseSchema):
    """Schema for creating a product."""
    
    name: str = Field(
        ...,
        min_length=1,
        max_length=255,
        description="Product name",
    )
    sku: str = Field(
        ...,
        min_length=1,
        max_length=50,
        description="Stock Keeping Unit",
    )
    description: Optional[str] = Field(
        None,
        max_length=5000,
        description="Product description",
    )
    price: Decimal = Field(
        ...,
        ge=0,
        description="Product price",
    )
    cost: Optional[Decimal] = Field(
        None,
        ge=0,
        description="Product cost",
    )
    quantity_available: int = Field(
        0,
        ge=0,
        description="Available inventory",
    )
    category: Optional[str] = Field(
        None,
        max_length=100,
        description="Product category",
    )
    image_url: Optional[str] = Field(
        None,
        max_length=500,
        description="Product image URL",
    )
    
    @field_validator("sku")
    @classmethod
    def validate_sku(cls, v: str) -> str:
        """Normalize SKU to uppercase."""
        return v.upper().strip()


class ProductUpdate(BaseSchema):
    """Schema for updating a product."""
    
    name: Optional[str] = Field(
        None,
        min_length=1,
        max_length=255,
        description="Product name",
    )
    description: Optional[str] = Field(
        None,
        max_length=5000,
        description="Product description",
    )
    price: Optional[Decimal] = Field(
        None,
        ge=0,
        description="Product price",
    )
    cost: Optional[Decimal] = Field(
        None,
        ge=0,
        description="Product cost",
    )
    quantity_available: Optional[int] = Field(
        None,
        ge=0,
        description="Available inventory",
    )
    category: Optional[str] = Field(
        None,
        max_length=100,
        description="Product category",
    )
    image_url: Optional[str] = Field(
        None,
        max_length=500,
        description="Product image URL",
    )
    is_active: Optional[bool] = Field(
        None,
        description="Product active status",
    )
    is_featured: Optional[bool] = Field(
        None,
        description="Featured product flag",
    )


class ProductResponse(TimestampSchema):
    """Schema for product response."""
    
    id: str = Field(
        ...,
        description="Product unique identifier",
    )
    name: str = Field(
        ...,
        description="Product name",
    )
    sku: str = Field(
        ...,
        description="Stock Keeping Unit",
    )
    description: Optional[str] = Field(
        None,
        description="Product description",
    )
    price: Decimal = Field(
        ...,
        description="Product price",
    )
    cost: Optional[Decimal] = Field(
        None,
        description="Product cost",
    )
    quantity_available: int = Field(
        ...,
        description="Available inventory",
    )
    category: Optional[str] = Field(
        None,
        description="Product category",
    )
    image_url: Optional[str] = Field(
        None,
        description="Product image URL",
    )
    is_active: bool = Field(
        ...,
        description="Product active status",
    )
    is_featured: bool = Field(
        ...,
        description="Featured product flag",
    )
    in_stock: bool = Field(
        ...,
        description="Whether product is in stock",
    )
