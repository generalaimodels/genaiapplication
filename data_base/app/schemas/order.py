# ==============================================================================
# ORDER SCHEMAS - E-commerce Orders
# ==============================================================================
# Request/Response schemas for order management
# ==============================================================================

from __future__ import annotations

from decimal import Decimal
from typing import List, Optional

from pydantic import Field, field_validator

from app.schemas.base import BaseSchema, TimestampSchema


class OrderItemCreate(BaseSchema):
    """Schema for creating an order item."""
    
    product_id: str = Field(
        ...,
        description="Product ID to order",
    )
    quantity: int = Field(
        ...,
        ge=1,
        description="Quantity to order",
    )


class OrderItemResponse(TimestampSchema):
    """Schema for order item response."""
    
    id: str = Field(
        ...,
        description="Order item unique identifier",
    )
    order_id: str = Field(
        ...,
        description="Parent order ID",
    )
    product_id: str = Field(
        ...,
        description="Product ID",
    )
    product_name: str = Field(
        ...,
        description="Product name at order time",
    )
    quantity: int = Field(
        ...,
        description="Ordered quantity",
    )
    unit_price: Decimal = Field(
        ...,
        description="Unit price at order time",
    )
    discount: Decimal = Field(
        ...,
        description="Item discount",
    )
    total_price: Decimal = Field(
        ...,
        description="Line item total",
    )


class OrderCreate(BaseSchema):
    """Schema for creating an order."""
    
    items: List[OrderItemCreate] = Field(
        ...,
        min_length=1,
        description="Order items",
    )
    shipping_address: Optional[str] = Field(
        None,
        max_length=500,
        description="Shipping address",
    )
    shipping_city: Optional[str] = Field(
        None,
        max_length=100,
        description="Shipping city",
    )
    shipping_country: Optional[str] = Field(
        None,
        max_length=100,
        description="Shipping country",
    )
    shipping_postal_code: Optional[str] = Field(
        None,
        max_length=20,
        description="Shipping postal code",
    )
    notes: Optional[str] = Field(
        None,
        max_length=1000,
        description="Order notes",
    )
    
    @field_validator("items")
    @classmethod
    def validate_items(cls, v: List[OrderItemCreate]) -> List[OrderItemCreate]:
        """Ensure at least one item in order."""
        if not v:
            raise ValueError("Order must contain at least one item")
        return v


class OrderUpdate(BaseSchema):
    """Schema for updating an order."""
    
    status: Optional[str] = Field(
        None,
        pattern="^(pending|confirmed|processing|shipped|delivered|cancelled|refunded)$",
        description="Order status",
    )
    shipping_address: Optional[str] = Field(
        None,
        max_length=500,
        description="Shipping address",
    )
    notes: Optional[str] = Field(
        None,
        max_length=1000,
        description="Order notes",
    )


class OrderResponse(TimestampSchema):
    """Schema for order response."""
    
    id: str = Field(
        ...,
        description="Order unique identifier",
    )
    user_id: str = Field(
        ...,
        description="Customer ID",
    )
    order_number: str = Field(
        ...,
        description="Human-readable order number",
    )
    status: str = Field(
        ...,
        description="Order status",
    )
    subtotal: Decimal = Field(
        ...,
        description="Items subtotal",
    )
    tax_amount: Decimal = Field(
        ...,
        description="Tax amount",
    )
    shipping_cost: Decimal = Field(
        ...,
        description="Shipping cost",
    )
    discount_amount: Decimal = Field(
        ...,
        description="Total discount",
    )
    total_amount: Decimal = Field(
        ...,
        description="Order total",
    )
    shipping_address: Optional[str] = Field(
        None,
        description="Shipping address",
    )
    shipping_city: Optional[str] = Field(
        None,
        description="Shipping city",
    )
    shipping_country: Optional[str] = Field(
        None,
        description="Shipping country",
    )
    items: List[OrderItemResponse] = Field(
        default_factory=list,
        description="Order line items",
    )
    item_count: int = Field(
        ...,
        description="Total items in order",
    )
