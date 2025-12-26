# ==============================================================================
# COURSE MODELS - Learning Management System
# ==============================================================================
# Course, Section, and Enrollment entities for LMS functionality
# ==============================================================================

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, List, Optional
import enum

from sqlalchemy import ForeignKey, String, Text, Integer, Boolean, Numeric, Date, Enum as SQLEnum
from sqlalchemy.orm import Mapped, mapped_column, relationship
from decimal import Decimal

from app.domain_models.base import SQLBase, TimestampMixin

if TYPE_CHECKING:
    from app.domain_models.user import User


class CourseLevel(str, enum.Enum):
    """Course difficulty level."""
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"


class EnrollmentStatus(str, enum.Enum):
    """Enrollment status states."""
    ACTIVE = "active"
    COMPLETED = "completed"
    DROPPED = "dropped"
    EXPIRED = "expired"


class Course(SQLBase, TimestampMixin):
    """
    Course model for learning management.
    
    Represents a course with sections, enrollments, and progress tracking.
    
    Attributes:
        title: Course title
        description: Course overview
        instructor_name: Course instructor
        level: Difficulty level
        price: Course price
        duration_hours: Estimated completion time
        
    Relationships:
        sections: Course content sections
        enrollments: Student enrollments
    """
    
    __tablename__ = "courses"
    
    # Course info
    title: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        index=True,
    )
    description: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
    )
    short_description: Mapped[Optional[str]] = mapped_column(
        String(500),
        nullable=True,
    )
    
    # Instructor
    instructor_name: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )
    instructor_id: Mapped[Optional[str]] = mapped_column(
        String(36),
        nullable=True,
    )
    
    # Level and category
    level: Mapped[CourseLevel] = mapped_column(
        SQLEnum(CourseLevel),
        default=CourseLevel.BEGINNER,
        nullable=False,
    )
    category: Mapped[Optional[str]] = mapped_column(
        String(100),
        nullable=True,
        index=True,
    )
    tags: Mapped[Optional[str]] = mapped_column(
        String(500),  # Comma-separated
        nullable=True,
    )
    
    # Pricing
    price: Mapped[Decimal] = mapped_column(
        Numeric(precision=10, scale=2),
        default=Decimal("0.00"),
        nullable=False,
    )
    is_free: Mapped[bool] = mapped_column(
        Boolean,
        default=True,
        nullable=False,
    )
    
    # Duration
    duration_hours: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False,
    )
    
    # Media
    thumbnail_url: Mapped[Optional[str]] = mapped_column(
        String(500),
        nullable=True,
    )
    preview_video_url: Mapped[Optional[str]] = mapped_column(
        String(500),
        nullable=True,
    )
    
    # Status
    is_published: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
    )
    published_at: Mapped[Optional[datetime]] = mapped_column(
        nullable=True,
    )
    
    # Relationships
    sections: Mapped[List["CourseSection"]] = relationship(
        "CourseSection",
        back_populates="course",
        cascade="all, delete-orphan",
        order_by="CourseSection.order",
    )
    enrollments: Mapped[List["Enrollment"]] = relationship(
        "Enrollment",
        back_populates="course",
        cascade="all, delete-orphan",
    )
    
    @property
    def section_count(self) -> int:
        """Get number of sections."""
        return len(self.sections)
    
    @property
    def enrollment_count(self) -> int:
        """Get number of enrollments."""
        return len(self.enrollments)
    
    def __repr__(self) -> str:
        return f"<Course(id={self.id}, title={self.title})>"


class CourseSection(SQLBase, TimestampMixin):
    """
    Course section/module containing lessons.
    
    Represents a logical grouping of content within a course.
    
    Attributes:
        course_id: Parent course
        title: Section title
        order: Display order
        content: Section content/lessons
        duration_minutes: Section duration
    """
    
    __tablename__ = "course_sections"
    
    # Foreign keys
    course_id: Mapped[str] = mapped_column(
        ForeignKey("courses.id", ondelete="CASCADE"),
        index=True,
        nullable=False,
    )
    
    # Section info
    title: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )
    description: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
    )
    
    # Content
    content: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
    )
    video_url: Mapped[Optional[str]] = mapped_column(
        String(500),
        nullable=True,
    )
    
    # Order and duration
    order: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False,
    )
    duration_minutes: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False,
    )
    
    # Relationships
    course: Mapped["Course"] = relationship(
        "Course",
        back_populates="sections",
    )
    
    def __repr__(self) -> str:
        return f"<CourseSection(id={self.id}, title={self.title}, order={self.order})>"


class Enrollment(SQLBase, TimestampMixin):
    """
    Enrollment linking users to courses.
    
    Tracks student enrollment status and progress.
    
    Attributes:
        user_id: Enrolled student
        course_id: Enrolled course
        status: Enrollment status
        progress_percent: Completion percentage
        completed_sections: JSON of completed section IDs
        
    Relationships:
        user: Enrolled student
        course: Enrolled course
    """
    
    __tablename__ = "enrollments"
    
    # Foreign keys
    user_id: Mapped[str] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"),
        index=True,
        nullable=False,
    )
    course_id: Mapped[str] = mapped_column(
        ForeignKey("courses.id", ondelete="CASCADE"),
        index=True,
        nullable=False,
    )
    
    # Status
    status: Mapped[EnrollmentStatus] = mapped_column(
        SQLEnum(EnrollmentStatus),
        default=EnrollmentStatus.ACTIVE,
        nullable=False,
    )
    
    # Progress
    progress_percent: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False,
    )
    last_section_id: Mapped[Optional[str]] = mapped_column(
        String(36),
        nullable=True,
    )
    
    # Timestamps
    enrolled_at: Mapped[datetime] = mapped_column(
        default=datetime.utcnow,
        nullable=False,
    )
    completed_at: Mapped[Optional[datetime]] = mapped_column(
        nullable=True,
    )
    expires_at: Mapped[Optional[datetime]] = mapped_column(
        Date,
        nullable=True,
    )
    
    # Relationships
    user: Mapped["User"] = relationship(
        "User",
        back_populates="enrollments",
    )
    course: Mapped["Course"] = relationship(
        "Course",
        back_populates="enrollments",
    )
    
    @property
    def is_completed(self) -> bool:
        """Check if enrollment is completed."""
        return self.status == EnrollmentStatus.COMPLETED
    
    def __repr__(self) -> str:
        return f"<Enrollment(user_id={self.user_id}, course_id={self.course_id}, status={self.status})>"
