# ==============================================================================
# PROJECT MODELS - Project Management
# ==============================================================================
# Project and Task entities for project management functionality
# ==============================================================================

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, List, Optional
import enum

from sqlalchemy import ForeignKey, String, Text, Date, Enum as SQLEnum
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.domain_models.base import SQLBase, TimestampMixin

if TYPE_CHECKING:
    from app.domain_models.user import User


class ProjectStatus(str, enum.Enum):
    """Project lifecycle status states."""
    PLANNING = "planning"
    IN_PROGRESS = "in_progress"
    ON_HOLD = "on_hold"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class TaskPriority(str, enum.Enum):
    """Task priority levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class TaskStatus(str, enum.Enum):
    """Task status states."""
    TODO = "todo"
    IN_PROGRESS = "in_progress"
    IN_REVIEW = "in_review"
    DONE = "done"
    BLOCKED = "blocked"


class Project(SQLBase, TimestampMixin):
    """
    Project model for project management.
    
    Represents a project with tasks, deadlines, and team members.
    
    Attributes:
        owner_id: Project owner/creator
        name: Project name
        description: Detailed description
        status: Current project status
        start_date: Planned start date
        end_date: Planned end date
        
    Relationships:
        owner: Project owner
        tasks: Tasks within project
    """
    
    __tablename__ = "projects"
    
    # Foreign keys
    owner_id: Mapped[str] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"),
        index=True,
        nullable=False,
    )
    
    # Project info
    name: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        index=True,
    )
    description: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
    )
    
    # Status
    status: Mapped[ProjectStatus] = mapped_column(
        SQLEnum(ProjectStatus),
        default=ProjectStatus.PLANNING,
        nullable=False,
    )
    
    # Timeline
    start_date: Mapped[Optional[datetime]] = mapped_column(
        Date,
        nullable=True,
    )
    end_date: Mapped[Optional[datetime]] = mapped_column(
        Date,
        nullable=True,
    )
    
    # Relationships
    owner: Mapped["User"] = relationship(
        "User",
        back_populates="projects",
        foreign_keys=[owner_id],
    )
    tasks: Mapped[List["Task"]] = relationship(
        "Task",
        back_populates="project",
        cascade="all, delete-orphan",
    )
    
    @property
    def task_count(self) -> int:
        """Get total number of tasks."""
        return len(self.tasks)
    
    @property
    def completed_task_count(self) -> int:
        """Get number of completed tasks."""
        return sum(1 for t in self.tasks if t.status == TaskStatus.DONE)
    
    def __repr__(self) -> str:
        return f"<Project(id={self.id}, name={self.name}, status={self.status})>"


class Task(SQLBase, TimestampMixin):
    """
    Task model within a project.
    
    Represents an individual task with assignment, priority,
    and deadline tracking.
    
    Attributes:
        project_id: Parent project
        assignee_id: Assigned team member
        title: Task title
        description: Task details
        priority: Priority level
        status: Current status
        due_date: Task deadline
        
    Relationships:
        project: Parent project
        assignee: Assigned user
    """
    
    __tablename__ = "tasks"
    
    # Foreign keys
    project_id: Mapped[str] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"),
        index=True,
        nullable=False,
    )
    assignee_id: Mapped[Optional[str]] = mapped_column(
        ForeignKey("users.id", ondelete="SET NULL"),
        index=True,
        nullable=True,
    )
    
    # Task info
    title: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )
    description: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
    )
    
    # Priority and status
    priority: Mapped[TaskPriority] = mapped_column(
        SQLEnum(TaskPriority),
        default=TaskPriority.MEDIUM,
        nullable=False,
    )
    status: Mapped[TaskStatus] = mapped_column(
        SQLEnum(TaskStatus),
        default=TaskStatus.TODO,
        nullable=False,
    )
    
    # Timeline
    due_date: Mapped[Optional[datetime]] = mapped_column(
        Date,
        nullable=True,
    )
    completed_at: Mapped[Optional[datetime]] = mapped_column(
        Date,
        nullable=True,
    )
    
    # Estimated effort (hours)
    estimated_hours: Mapped[Optional[int]] = mapped_column(
        nullable=True,
    )
    actual_hours: Mapped[Optional[int]] = mapped_column(
        nullable=True,
    )
    
    # Relationships
    project: Mapped["Project"] = relationship(
        "Project",
        back_populates="tasks",
    )
    assignee: Mapped[Optional["User"]] = relationship(
        "User",
        back_populates="assigned_tasks",
        foreign_keys=[assignee_id],
    )
    
    @property
    def is_overdue(self) -> bool:
        """Check if task is past due date."""
        if self.due_date and self.status not in [TaskStatus.DONE]:
            return datetime.now().date() > self.due_date
        return False
    
    def __repr__(self) -> str:
        return f"<Task(id={self.id}, title={self.title}, status={self.status})>"
