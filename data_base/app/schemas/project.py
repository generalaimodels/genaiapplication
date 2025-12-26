# ==============================================================================
# PROJECT SCHEMAS - Project Management
# ==============================================================================
# Request/Response schemas for projects and tasks
# ==============================================================================

from __future__ import annotations

from datetime import date
from typing import List, Optional

from pydantic import Field

from app.schemas.base import BaseSchema, TimestampSchema


class TaskCreate(BaseSchema):
    """Schema for creating a task."""
    
    title: str = Field(
        ...,
        min_length=1,
        max_length=255,
        description="Task title",
    )
    description: Optional[str] = Field(
        None,
        max_length=5000,
        description="Task description",
    )
    priority: str = Field(
        "medium",
        pattern="^(low|medium|high|critical)$",
        description="Task priority",
    )
    assignee_id: Optional[str] = Field(
        None,
        description="Assigned user ID",
    )
    due_date: Optional[date] = Field(
        None,
        description="Task due date",
    )
    estimated_hours: Optional[int] = Field(
        None,
        ge=0,
        description="Estimated effort in hours",
    )


class TaskUpdate(BaseSchema):
    """Schema for updating a task."""
    
    title: Optional[str] = Field(
        None,
        min_length=1,
        max_length=255,
        description="Task title",
    )
    description: Optional[str] = Field(
        None,
        max_length=5000,
        description="Task description",
    )
    priority: Optional[str] = Field(
        None,
        pattern="^(low|medium|high|critical)$",
        description="Task priority",
    )
    status: Optional[str] = Field(
        None,
        pattern="^(todo|in_progress|in_review|done|blocked)$",
        description="Task status",
    )
    assignee_id: Optional[str] = Field(
        None,
        description="Assigned user ID",
    )
    due_date: Optional[date] = Field(
        None,
        description="Task due date",
    )
    estimated_hours: Optional[int] = Field(
        None,
        ge=0,
        description="Estimated effort in hours",
    )
    actual_hours: Optional[int] = Field(
        None,
        ge=0,
        description="Actual effort in hours",
    )


class TaskResponse(TimestampSchema):
    """Schema for task response."""
    
    id: str = Field(
        ...,
        description="Task unique identifier",
    )
    project_id: str = Field(
        ...,
        description="Parent project ID",
    )
    assignee_id: Optional[str] = Field(
        None,
        description="Assigned user ID",
    )
    title: str = Field(
        ...,
        description="Task title",
    )
    description: Optional[str] = Field(
        None,
        description="Task description",
    )
    priority: str = Field(
        ...,
        description="Task priority",
    )
    status: str = Field(
        ...,
        description="Task status",
    )
    due_date: Optional[date] = Field(
        None,
        description="Task due date",
    )
    completed_at: Optional[date] = Field(
        None,
        description="Completion date",
    )
    estimated_hours: Optional[int] = Field(
        None,
        description="Estimated hours",
    )
    actual_hours: Optional[int] = Field(
        None,
        description="Actual hours",
    )
    is_overdue: bool = Field(
        False,
        description="Whether task is overdue",
    )


class ProjectCreate(BaseSchema):
    """Schema for creating a project."""
    
    name: str = Field(
        ...,
        min_length=1,
        max_length=255,
        description="Project name",
    )
    description: Optional[str] = Field(
        None,
        max_length=5000,
        description="Project description",
    )
    start_date: Optional[date] = Field(
        None,
        description="Project start date",
    )
    end_date: Optional[date] = Field(
        None,
        description="Project end date",
    )


class ProjectUpdate(BaseSchema):
    """Schema for updating a project."""
    
    name: Optional[str] = Field(
        None,
        min_length=1,
        max_length=255,
        description="Project name",
    )
    description: Optional[str] = Field(
        None,
        max_length=5000,
        description="Project description",
    )
    status: Optional[str] = Field(
        None,
        pattern="^(planning|in_progress|on_hold|completed|cancelled)$",
        description="Project status",
    )
    start_date: Optional[date] = Field(
        None,
        description="Project start date",
    )
    end_date: Optional[date] = Field(
        None,
        description="Project end date",
    )


class ProjectResponse(TimestampSchema):
    """Schema for project response."""
    
    id: str = Field(
        ...,
        description="Project unique identifier",
    )
    owner_id: str = Field(
        ...,
        description="Project owner ID",
    )
    name: str = Field(
        ...,
        description="Project name",
    )
    description: Optional[str] = Field(
        None,
        description="Project description",
    )
    status: str = Field(
        ...,
        description="Project status",
    )
    start_date: Optional[date] = Field(
        None,
        description="Project start date",
    )
    end_date: Optional[date] = Field(
        None,
        description="Project end date",
    )
    task_count: int = Field(
        0,
        description="Total number of tasks",
    )
    completed_task_count: int = Field(
        0,
        description="Completed tasks count",
    )
    tasks: Optional[List[TaskResponse]] = Field(
        None,
        description="Project tasks (when requested)",
    )
