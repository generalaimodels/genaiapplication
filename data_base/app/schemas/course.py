# ==============================================================================
# COURSE SCHEMAS - Learning Management System
# ==============================================================================
# Request/Response schemas for courses and enrollments
# ==============================================================================

from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal
from typing import List, Optional

from pydantic import Field

from app.schemas.base import BaseSchema, TimestampSchema


class CourseSectionCreate(BaseSchema):
    """Schema for creating a course section."""
    
    title: str = Field(
        ...,
        min_length=1,
        max_length=255,
        description="Section title",
    )
    description: Optional[str] = Field(
        None,
        max_length=2000,
        description="Section description",
    )
    content: Optional[str] = Field(
        None,
        description="Section content (markdown/HTML)",
    )
    video_url: Optional[str] = Field(
        None,
        max_length=500,
        description="Section video URL",
    )
    order: int = Field(
        0,
        ge=0,
        description="Display order",
    )
    duration_minutes: int = Field(
        0,
        ge=0,
        description="Section duration in minutes",
    )


class CourseSectionResponse(TimestampSchema):
    """Schema for course section response."""
    
    id: str = Field(
        ...,
        description="Section unique identifier",
    )
    course_id: str = Field(
        ...,
        description="Parent course ID",
    )
    title: str = Field(
        ...,
        description="Section title",
    )
    description: Optional[str] = Field(
        None,
        description="Section description",
    )
    content: Optional[str] = Field(
        None,
        description="Section content",
    )
    video_url: Optional[str] = Field(
        None,
        description="Section video URL",
    )
    order: int = Field(
        ...,
        description="Display order",
    )
    duration_minutes: int = Field(
        ...,
        description="Section duration",
    )


class CourseCreate(BaseSchema):
    """Schema for creating a course."""
    
    title: str = Field(
        ...,
        min_length=1,
        max_length=255,
        description="Course title",
    )
    description: Optional[str] = Field(
        None,
        max_length=10000,
        description="Course description",
    )
    short_description: Optional[str] = Field(
        None,
        max_length=500,
        description="Short course description",
    )
    instructor_name: str = Field(
        ...,
        max_length=255,
        description="Instructor name",
    )
    level: str = Field(
        "beginner",
        pattern="^(beginner|intermediate|advanced|expert)$",
        description="Course difficulty level",
    )
    category: Optional[str] = Field(
        None,
        max_length=100,
        description="Course category",
    )
    price: Decimal = Field(
        Decimal("0.00"),
        ge=0,
        description="Course price",
    )
    duration_hours: int = Field(
        0,
        ge=0,
        description="Total course duration in hours",
    )
    thumbnail_url: Optional[str] = Field(
        None,
        max_length=500,
        description="Course thumbnail URL",
    )


class CourseUpdate(BaseSchema):
    """Schema for updating a course."""
    
    title: Optional[str] = Field(
        None,
        min_length=1,
        max_length=255,
        description="Course title",
    )
    description: Optional[str] = Field(
        None,
        max_length=10000,
        description="Course description",
    )
    short_description: Optional[str] = Field(
        None,
        max_length=500,
        description="Short course description",
    )
    instructor_name: Optional[str] = Field(
        None,
        max_length=255,
        description="Instructor name",
    )
    level: Optional[str] = Field(
        None,
        pattern="^(beginner|intermediate|advanced|expert)$",
        description="Course difficulty level",
    )
    category: Optional[str] = Field(
        None,
        max_length=100,
        description="Course category",
    )
    price: Optional[Decimal] = Field(
        None,
        ge=0,
        description="Course price",
    )
    is_published: Optional[bool] = Field(
        None,
        description="Publication status",
    )


class CourseResponse(TimestampSchema):
    """Schema for course response."""
    
    id: str = Field(
        ...,
        description="Course unique identifier",
    )
    title: str = Field(
        ...,
        description="Course title",
    )
    description: Optional[str] = Field(
        None,
        description="Course description",
    )
    short_description: Optional[str] = Field(
        None,
        description="Short description",
    )
    instructor_name: str = Field(
        ...,
        description="Instructor name",
    )
    level: str = Field(
        ...,
        description="Difficulty level",
    )
    category: Optional[str] = Field(
        None,
        description="Course category",
    )
    price: Decimal = Field(
        ...,
        description="Course price",
    )
    is_free: bool = Field(
        ...,
        description="Whether course is free",
    )
    duration_hours: int = Field(
        ...,
        description="Total duration in hours",
    )
    thumbnail_url: Optional[str] = Field(
        None,
        description="Thumbnail URL",
    )
    is_published: bool = Field(
        ...,
        description="Publication status",
    )
    section_count: int = Field(
        0,
        description="Number of sections",
    )
    enrollment_count: int = Field(
        0,
        description="Number of enrollments",
    )
    sections: Optional[List[CourseSectionResponse]] = Field(
        None,
        description="Course sections (when requested)",
    )


class EnrollmentCreate(BaseSchema):
    """Schema for creating an enrollment."""
    
    course_id: str = Field(
        ...,
        description="Course ID to enroll in",
    )


class EnrollmentUpdate(BaseSchema):
    """Schema for updating an enrollment."""
    
    progress_percent: Optional[int] = Field(
        None,
        ge=0,
        le=100,
        description="Progress percentage",
    )
    last_section_id: Optional[str] = Field(
        None,
        description="Last viewed section ID",
    )
    status: Optional[str] = Field(
        None,
        pattern="^(active|completed|dropped|expired)$",
        description="Enrollment status",
    )


class EnrollmentResponse(TimestampSchema):
    """Schema for enrollment response."""
    
    id: str = Field(
        ...,
        description="Enrollment unique identifier",
    )
    user_id: str = Field(
        ...,
        description="Enrolled user ID",
    )
    course_id: str = Field(
        ...,
        description="Enrolled course ID",
    )
    status: str = Field(
        ...,
        description="Enrollment status",
    )
    progress_percent: int = Field(
        ...,
        description="Progress percentage",
    )
    last_section_id: Optional[str] = Field(
        None,
        description="Last viewed section",
    )
    enrolled_at: datetime = Field(
        ...,
        description="Enrollment date",
    )
    completed_at: Optional[datetime] = Field(
        None,
        description="Completion date",
    )
    expires_at: Optional[date] = Field(
        None,
        description="Expiration date",
    )
    is_completed: bool = Field(
        False,
        description="Whether enrollment is completed",
    )
