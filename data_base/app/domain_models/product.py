# ==============================================================================
# PRODUCT MODEL - E-commerce Catalog
# ==============================================================================
# Product entity for e-commerce functionality
# ==============================================================================

from __future__ import annotations

from decimal import Decimal
from typing import TYPE_CHECKING, List, Optional

from sqlalchemy import Boolean, Integer, String, Text, Numeric
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.domain_models.base import SQLBase, TimestampMixin

if TYPE_CHECKING:
    from app.domain_models.order import OrderItem


class Product(SQLBase, TimestampMixin):
    """
    Product model for e-commerce catalog.
    
    Represents items available for purchase with inventory
    management and pricing information.
    
    Attributes:
        name: Product display name
        sku: Stock Keeping Unit (unique identifier)
        description: Detailed product description
        price: Current selling price
        cost: Product cost (for profit calculation)
        quantity_available: Current inventory level
        is_active: Whether product is available for sale
        
    Relationships:
        order_items: Order line items containing this product
    """
    
    __tablename__ = "products"
    
    # Basic info
    name: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        index=True,
    )
    sku: Mapped[str] = mapped_column(
        String(50),
        unique=True,
        index=True,
        nullable=False,
    )
    description: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
    )
    
    # Pricing
    price: Mapped[Decimal] = mapped_column(
        Numeric(precision=18, scale=2),
        nullable=False,
    )
    cost: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(precision=18, scale=2),
        nullable=True,
    )
    
    # Inventory
    quantity_available: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False,
    )
    reorder_point: Mapped[int] = mapped_column(
        Integer,
        default=10,
        nullable=False,
    )
    
    # Categorization
    category: Mapped[Optional[str]] = mapped_column(
        String(100),
        nullable=True,
        index=True,
    )
    tags: Mapped[Optional[str]] = mapped_column(
        String(500),  # Comma-separated tags
        nullable=True,
    )
    
    # Media
    image_url: Mapped[Optional[str]] = mapped_column(
        String(500),
        nullable=True,
    )
    
    # Status
    is_active: Mapped[bool] = mapped_column(
        Boolean,
        default=True,
        nullable=False,
    )
    is_featured: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
    )
    
    # Relationships
    order_items: Mapped[List["OrderItem"]] = relationship(
        "OrderItem",
        back_populates="product",
    )
    
    @property
    def in_stock(self) -> bool:
        """Check if product is in stock."""
        return self.quantity_available > 0
    
    @property
    def low_stock(self) -> bool:
        """Check if product is below reorder point."""
        return self.quantity_available <= self.reorder_point
    
    def __repr__(self) -> str:
        return f"<Product(id={self.id}, name={self.name}, sku={self.sku})>"
