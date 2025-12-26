# ==============================================================================
# ORDER MODELS - E-commerce Orders
# ==============================================================================
# Order and OrderItem entities for e-commerce functionality
# ==============================================================================

from __future__ import annotations

from decimal import Decimal
from typing import TYPE_CHECKING, List, Optional
import enum

from sqlalchemy import ForeignKey, Integer, String, Text, Numeric, Enum as SQLEnum
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.domain_models.base import SQLBase, TimestampMixin

if TYPE_CHECKING:
    from app.domain_models.user import User
    from app.domain_models.product import Product


class OrderStatus(str, enum.Enum):
    """Order lifecycle status states."""
    PENDING = "pending"
    CONFIRMED = "confirmed"
    PROCESSING = "processing"
    SHIPPED = "shipped"
    DELIVERED = "delivered"
    CANCELLED = "cancelled"
    REFUNDED = "refunded"


class Order(SQLBase, TimestampMixin):
    """
    Order model representing a customer purchase.
    
    Contains order metadata, shipping information, and
    totals. Individual items are stored in OrderItem.
    
    Attributes:
        user_id: Customer who placed the order
        order_number: Human-readable order identifier
        status: Current order status
        subtotal: Sum of item prices
        tax_amount: Calculated tax
        shipping_cost: Shipping charges
        total_amount: Final order total
        
    Relationships:
        user: Customer who placed order
        items: Line items in the order
    """
    
    __tablename__ = "orders"
    
    # Foreign keys
    user_id: Mapped[str] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"),
        index=True,
        nullable=False,
    )
    
    # Order identification
    order_number: Mapped[str] = mapped_column(
        String(50),
        unique=True,
        index=True,
        nullable=False,
    )
    
    # Status
    status: Mapped[OrderStatus] = mapped_column(
        SQLEnum(OrderStatus),
        default=OrderStatus.PENDING,
        nullable=False,
    )
    
    # Pricing
    subtotal: Mapped[Decimal] = mapped_column(
        Numeric(precision=18, scale=2),
        nullable=False,
    )
    tax_amount: Mapped[Decimal] = mapped_column(
        Numeric(precision=18, scale=2),
        default=Decimal("0.00"),
        nullable=False,
    )
    shipping_cost: Mapped[Decimal] = mapped_column(
        Numeric(precision=18, scale=2),
        default=Decimal("0.00"),
        nullable=False,
    )
    discount_amount: Mapped[Decimal] = mapped_column(
        Numeric(precision=18, scale=2),
        default=Decimal("0.00"),
        nullable=False,
    )
    total_amount: Mapped[Decimal] = mapped_column(
        Numeric(precision=18, scale=2),
        nullable=False,
    )
    
    # Shipping address
    shipping_address: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
    )
    shipping_city: Mapped[Optional[str]] = mapped_column(
        String(100),
        nullable=True,
    )
    shipping_country: Mapped[Optional[str]] = mapped_column(
        String(100),
        nullable=True,
    )
    shipping_postal_code: Mapped[Optional[str]] = mapped_column(
        String(20),
        nullable=True,
    )
    
    # Notes
    notes: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
    )
    
    # Relationships
    user: Mapped["User"] = relationship(
        "User",
        back_populates="orders",
    )
    items: Mapped[List["OrderItem"]] = relationship(
        "OrderItem",
        back_populates="order",
        cascade="all, delete-orphan",
    )
    
    @property
    def item_count(self) -> int:
        """Get total number of items in order."""
        return sum(item.quantity for item in self.items)
    
    def __repr__(self) -> str:
        return f"<Order(id={self.id}, order_number={self.order_number}, status={self.status})>"


class OrderItem(SQLBase, TimestampMixin):
    """
    Order line item linking orders to products.
    
    Stores quantity, price at time of order, and any
    item-specific discounts.
    
    Attributes:
        order_id: Parent order
        product_id: Ordered product
        quantity: Number of units
        unit_price: Price per unit at time of order
        total_price: Line item total
        
    Relationships:
        order: Parent order
        product: Ordered product
    """
    
    __tablename__ = "order_items"
    
    # Foreign keys
    order_id: Mapped[str] = mapped_column(
        ForeignKey("orders.id", ondelete="CASCADE"),
        index=True,
        nullable=False,
    )
    product_id: Mapped[str] = mapped_column(
        ForeignKey("products.id", ondelete="RESTRICT"),
        index=True,
        nullable=False,
    )
    
    # Item details
    quantity: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )
    unit_price: Mapped[Decimal] = mapped_column(
        Numeric(precision=18, scale=2),
        nullable=False,
    )
    discount: Mapped[Decimal] = mapped_column(
        Numeric(precision=18, scale=2),
        default=Decimal("0.00"),
        nullable=False,
    )
    total_price: Mapped[Decimal] = mapped_column(
        Numeric(precision=18, scale=2),
        nullable=False,
    )
    
    # Snapshot of product name at order time
    product_name: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )
    
    # Relationships
    order: Mapped["Order"] = relationship(
        "Order",
        back_populates="items",
    )
    product: Mapped["Product"] = relationship(
        "Product",
        back_populates="order_items",
    )
    
    def __repr__(self) -> str:
        return f"<OrderItem(id={self.id}, product={self.product_name}, qty={self.quantity})>"
