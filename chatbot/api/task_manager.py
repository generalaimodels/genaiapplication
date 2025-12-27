# -*- coding: utf-8 -*-
# =================================================================================================
# api/task_manager.py — Advanced Async Task Manager
# =================================================================================================
# SOTA implementation using asyncio event loop patterns:
#
#   1. PRIORITY QUEUE: Tasks ordered by priority and submission time
#   2. WORKER POOL: Configurable concurrent workers with semaphore control  
#   3. LIFECYCLE: Full task lifecycle management (pending → running → complete/failed)
#   4. CANCELLATION: Structured cancellation with cleanup
#   5. GRACEFUL SHUTDOWN: Wait for running tasks, cancel pending
#
# Usage:
# ------
#   from api.task_manager import get_task_manager, TaskPriority
#   
#   manager = get_task_manager()
#   task_id = await manager.submit(
#       my_async_function(arg1, arg2),
#       name="process_document",
#       priority=TaskPriority.HIGH,
#   )
#   result = await manager.wait(task_id)
#
# =================================================================================================

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Coroutine, Dict, List, Optional, Set, Tuple
from contextlib import asynccontextmanager

_LOG = logging.getLogger("api.task_manager")


# =============================================================================
# Enums
# =============================================================================

class TaskStatus(str, Enum):
    """Task lifecycle status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskPriority(Enum):
    """Task priority levels (higher value = higher priority)."""
    LOW = 0
    NORMAL = 5
    HIGH = 10
    CRITICAL = 20


# =============================================================================
# Data Classes
# =============================================================================

@dataclass(order=True)
class TaskEntry:
    """
    Task queue entry with priority ordering.
    
    Uses negative priority for max-heap behavior in PriorityQueue.
    """
    priority: int  # Negative of TaskPriority.value for max-heap
    timestamp: float = field(compare=True)
    task_id: str = field(compare=False)
    coroutine: Coroutine = field(compare=False, repr=False)
    name: str = field(compare=False, default="")
    metadata: Dict[str, Any] = field(compare=False, default_factory=dict)
    
    def __post_init__(self):
        if not self.name:
            self.name = f"task_{self.task_id[:8]}"


@dataclass
class TaskResult:
    """Result of a completed task with timing information."""
    task_id: str
    status: TaskStatus
    result: Any = None
    error: Optional[str] = None
    started_at: float = 0.0
    completed_at: float = 0.0
    
    @property
    def duration_ms(self) -> float:
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at) * 1000
        return 0.0


@dataclass
class TaskInfo:
    """Public task information for status queries."""
    task_id: str
    name: str
    status: TaskStatus
    priority: str
    submitted_at: float
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# Progress Callback
# =============================================================================

class ProgressTracker:
    """
    Progress tracker for long-running tasks.
    
    Usage in task:
        async def my_task(progress: ProgressTracker):
            for i in range(100):
                await progress.update(i / 100, f"Step {i}")
                ...
    """
    
    def __init__(self, task_id: str, callback: Optional[Callable] = None):
        self.task_id = task_id
        self._callback = callback
        self._progress = 0.0
        self._message = ""
        self._updated_at = time.time()
    
    @property
    def progress(self) -> float:
        return self._progress
    
    @property
    def message(self) -> str:
        return self._message
    
    async def update(self, progress: float, message: str = "") -> None:
        """Update progress (0.0 to 1.0) with optional message."""
        self._progress = max(0.0, min(1.0, progress))
        self._message = message
        self._updated_at = time.time()
        
        if self._callback:
            try:
                await self._callback(self.task_id, self._progress, message)
            except Exception as e:
                _LOG.warning("Progress callback error: %s", e)


# =============================================================================
# Task Manager
# =============================================================================

class AsyncTaskManager:
    """
    Advanced async task manager with event loop integration.
    
    Features:
    ---------
    - Priority-based async task queue  
    - Configurable max concurrent workers
    - Task lifecycle tracking with events
    - Graceful shutdown with configurable timeout
    - Progress tracking support
    - Task cancellation
    
    Thread Safety:
    --------------
    All operations are async-safe. The underlying asyncio.PriorityQueue
    handles concurrent access. Task results are protected by task-specific
    events for wait operations.
    """
    
    def __init__(
        self,
        max_workers: int = 10,
        shutdown_timeout: float = 30.0,
        enable_progress: bool = True,
    ):
        """
        Initialize task manager.
        
        Args:
            max_workers: Maximum concurrent worker tasks
            shutdown_timeout: Seconds to wait for running tasks during shutdown
            enable_progress: Whether to enable progress tracking
        """
        self.max_workers = max_workers
        self.shutdown_timeout = shutdown_timeout
        self.enable_progress = enable_progress
        
        # Task queue (priority queue for ordering)
        self._queue: asyncio.PriorityQueue[TaskEntry] = asyncio.PriorityQueue()
        
        # Task tracking
        self._pending_tasks: Dict[str, TaskEntry] = {}
        self._running_tasks: Dict[str, asyncio.Task] = {}
        self._results: Dict[str, TaskResult] = {}
        self._task_info: Dict[str, TaskInfo] = {}
        
        # Completion events (for wait() operation)
        self._task_events: Dict[str, asyncio.Event] = {}
        
        # Progress tracking
        self._progress: Dict[str, ProgressTracker] = {}
        
        # Concurrency control
        self._semaphore = asyncio.Semaphore(max_workers)
        self._workers: Set[asyncio.Task] = set()
        
        # Lifecycle
        self._running = False
        self._shutdown_event = asyncio.Event()
        self._started_at: Optional[float] = None
        
    async def start(self) -> None:
        """Start the task manager worker pool."""
        if self._running:
            _LOG.warning("Task manager already running")
            return
        
        self._running = True
        self._shutdown_event.clear()
        self._started_at = time.time()
        
        # Start worker tasks
        for i in range(self.max_workers):
            worker = asyncio.create_task(
                self._worker(i),
                name=f"task_worker_{i}"
            )
            self._workers.add(worker)
            worker.add_done_callback(self._workers.discard)
        
        _LOG.info(
            "Task manager started: workers=%d, shutdown_timeout=%.1fs",
            self.max_workers, self.shutdown_timeout
        )
    
    async def shutdown(self) -> None:
        """Gracefully shutdown the task manager."""
        if not self._running:
            return
        
        _LOG.info("Task manager shutting down...")
        self._running = False
        self._shutdown_event.set()
        
        # Cancel pending tasks in queue
        cancelled_count = 0
        while not self._queue.empty():
            try:
                entry = self._queue.get_nowait()
                self._mark_cancelled(entry.task_id, "Shutdown requested")
                cancelled_count += 1
            except asyncio.QueueEmpty:
                break
        
        if cancelled_count:
            _LOG.info("Cancelled %d pending tasks", cancelled_count)
        
        # Wait for running tasks
        if self._running_tasks:
            running_ids = list(self._running_tasks.keys())
            _LOG.info("Waiting for %d running tasks...", len(running_ids))
            
            pending_tasks = list(self._running_tasks.values())
            done, still_running = await asyncio.wait(
                pending_tasks,
                timeout=self.shutdown_timeout,
            )
            
            # Force cancel remaining
            if still_running:
                _LOG.warning("Force cancelling %d tasks", len(still_running))
                for task in still_running:
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass
        
        # Wait for workers to finish
        if self._workers:
            await asyncio.gather(*self._workers, return_exceptions=True)
        
        _LOG.info("Task manager shutdown complete")
    
    async def submit(
        self,
        coro: Coroutine,
        name: str = "",
        priority: TaskPriority = TaskPriority.NORMAL,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Submit a coroutine for background execution.
        
        Args:
            coro: Coroutine to execute
            name: Human-readable task name
            priority: Task priority level
            metadata: Optional metadata dict
        
        Returns:
            task_id: UUID for tracking the task
        
        Example:
            task_id = await manager.submit(
                process_document(doc_id),
                name="process_doc",
                priority=TaskPriority.HIGH,
            )
        """
        if not self._running:
            raise RuntimeError("Task manager not started")
        
        task_id = str(uuid.uuid4())
        now = time.time()
        
        entry = TaskEntry(
            priority=-priority.value,  # Negative for max-heap behavior
            timestamp=now,
            task_id=task_id,
            coroutine=coro,
            name=name or f"task_{task_id[:8]}",
            metadata=metadata or {},
        )
        
        # Create completion event
        self._task_events[task_id] = asyncio.Event()
        
        # Track task info
        self._task_info[task_id] = TaskInfo(
            task_id=task_id,
            name=entry.name,
            status=TaskStatus.PENDING,
            priority=priority.name,
            submitted_at=now,
            metadata=entry.metadata,
        )
        
        # Add progress tracker if enabled
        if self.enable_progress:
            self._progress[task_id] = ProgressTracker(task_id)
        
        # Queue the task
        self._pending_tasks[task_id] = entry
        await self._queue.put(entry)
        
        _LOG.debug(
            "Task submitted: id=%s, name=%s, priority=%s",
            task_id[:8], entry.name, priority.name
        )
        
        return task_id
    
    async def wait(
        self, 
        task_id: str, 
        timeout: Optional[float] = None
    ) -> TaskResult:
        """
        Wait for a task to complete.
        
        Args:
            task_id: Task UUID from submit()
            timeout: Optional timeout in seconds
        
        Returns:
            TaskResult with status, result/error, and timing info
        """
        event = self._task_events.get(task_id)
        if not event:
            return TaskResult(
                task_id=task_id,
                status=TaskStatus.FAILED,
                error=f"Unknown task: {task_id}",
            )
        
        try:
            await asyncio.wait_for(event.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            return TaskResult(
                task_id=task_id,
                status=TaskStatus.RUNNING,
                error="Wait timeout",
            )
        
        return self._results.get(task_id, TaskResult(
            task_id=task_id,
            status=TaskStatus.FAILED,
            error="Result not found",
        ))
    
    def get_status(self, task_id: str) -> Optional[TaskInfo]:
        """Get current status information for a task."""
        info = self._task_info.get(task_id)
        if info and task_id in self._results:
            # Update with result info
            result = self._results[task_id]
            info.status = result.status
            info.completed_at = result.completed_at
            info.error = result.error
        return info
    
    def get_progress(self, task_id: str) -> Tuple[float, str]:
        """Get progress for a task (0.0 to 1.0, message)."""
        tracker = self._progress.get(task_id)
        if tracker:
            return tracker.progress, tracker.message
        return 0.0, ""
    
    async def cancel(self, task_id: str) -> bool:
        """
        Cancel a pending or running task.
        
        Returns True if task was found and cancelled.
        """
        # Cancel running task
        if task_id in self._running_tasks:
            task = self._running_tasks[task_id]
            task.cancel()
            _LOG.info("Cancelled running task: %s", task_id[:8])
            return True
        
        # Mark pending task as cancelled
        if task_id in self._pending_tasks:
            self._mark_cancelled(task_id, "User cancelled")
            _LOG.info("Cancelled pending task: %s", task_id[:8])
            return True
        
        return False
    
    def list_tasks(
        self,
        status: Optional[TaskStatus] = None,
        limit: int = 100,
    ) -> List[TaskInfo]:
        """List tasks with optional status filter."""
        tasks = list(self._task_info.values())
        
        if status:
            tasks = [t for t in tasks if t.status == status]
        
        # Sort by submission time (newest first)
        tasks.sort(key=lambda t: t.submitted_at, reverse=True)
        
        return tasks[:limit]
    
    def stats(self) -> Dict[str, Any]:
        """Get task manager statistics."""
        status_counts = {}
        for info in self._task_info.values():
            status_counts[info.status.value] = status_counts.get(info.status.value, 0) + 1
        
        return {
            "running": self._running,
            "uptime_seconds": time.time() - self._started_at if self._started_at else 0,
            "max_workers": self.max_workers,
            "active_workers": len(self._running_tasks),
            "queue_size": self._queue.qsize(),
            "total_tasks": len(self._task_info),
            "status_counts": status_counts,
        }
    
    # =========================================================================
    # Internal Methods
    # =========================================================================
    
    def _mark_cancelled(self, task_id: str, reason: str) -> None:
        """Mark a task as cancelled."""
        now = time.time()
        self._results[task_id] = TaskResult(
            task_id=task_id,
            status=TaskStatus.CANCELLED,
            error=reason,
            completed_at=now,
        )
        
        # Update task info
        if task_id in self._task_info:
            self._task_info[task_id].status = TaskStatus.CANCELLED
            self._task_info[task_id].completed_at = now
            self._task_info[task_id].error = reason
        
        # Signal completion
        event = self._task_events.get(task_id)
        if event:
            event.set()
        
        # Cleanup
        self._pending_tasks.pop(task_id, None)
    
    async def _worker(self, worker_id: int) -> None:
        """Worker coroutine that processes tasks from the queue."""
        _LOG.debug("Worker %d started", worker_id)
        
        while self._running or not self._queue.empty():
            try:
                # Check for shutdown
                if self._shutdown_event.is_set() and self._queue.empty():
                    break
                
                # Get next task with timeout for shutdown checks
                try:
                    entry = await asyncio.wait_for(
                        self._queue.get(),
                        timeout=1.0
                    )
                except asyncio.TimeoutError:
                    continue
                
                task_id = entry.task_id
                
                # Check if cancelled while queued
                if task_id in self._results:
                    status = self._results[task_id].status
                    if status == TaskStatus.CANCELLED:
                        _LOG.debug("Skipping cancelled task: %s", task_id[:8])
                        self._queue.task_done()
                        continue
                
                # Remove from pending
                self._pending_tasks.pop(task_id, None)
                
                # Execute with semaphore control
                await self._execute_task(entry)
                
            except asyncio.CancelledError:
                _LOG.debug("Worker %d cancelled", worker_id)
                break
            except Exception as e:
                _LOG.error("Worker %d error: %s", worker_id, e, exc_info=True)
        
        _LOG.debug("Worker %d stopped", worker_id)
    
    async def _execute_task(self, entry: TaskEntry) -> None:
        """Execute a single task with proper tracking."""
        task_id = entry.task_id
        started_at = time.time()
        
        async with self._semaphore:
            try:
                # Track as running
                current_task = asyncio.current_task()
                if current_task:
                    self._running_tasks[task_id] = current_task
                
                # Update task info
                if task_id in self._task_info:
                    self._task_info[task_id].status = TaskStatus.RUNNING
                    self._task_info[task_id].started_at = started_at
                
                _LOG.debug(
                    "Executing task: id=%s, name=%s",
                    task_id[:8], entry.name
                )
                
                # Execute the coroutine
                result = await entry.coroutine
                
                # Success
                completed_at = time.time()
                self._results[task_id] = TaskResult(
                    task_id=task_id,
                    status=TaskStatus.COMPLETED,
                    result=result,
                    started_at=started_at,
                    completed_at=completed_at,
                )
                
                _LOG.debug(
                    "Task completed: id=%s, duration=%.2fms",
                    task_id[:8], (completed_at - started_at) * 1000
                )
                
            except asyncio.CancelledError:
                self._results[task_id] = TaskResult(
                    task_id=task_id,
                    status=TaskStatus.CANCELLED,
                    error="Task cancelled",
                    started_at=started_at,
                    completed_at=time.time(),
                )
                raise
                
            except Exception as e:
                completed_at = time.time()
                error_msg = str(e)
                
                self._results[task_id] = TaskResult(
                    task_id=task_id,
                    status=TaskStatus.FAILED,
                    error=error_msg,
                    started_at=started_at,
                    completed_at=completed_at,
                )
                
                _LOG.error(
                    "Task failed: id=%s, error=%s",
                    task_id[:8], error_msg
                )
                
            finally:
                # Cleanup tracking
                self._running_tasks.pop(task_id, None)
                self._pending_tasks.pop(task_id, None)
                
                # Update task info with final status
                if task_id in self._task_info and task_id in self._results:
                    result = self._results[task_id]
                    self._task_info[task_id].status = result.status
                    self._task_info[task_id].completed_at = result.completed_at
                    self._task_info[task_id].error = result.error
                
                # Signal completion
                event = self._task_events.get(task_id)
                if event:
                    event.set()
                
                self._queue.task_done()


# =============================================================================
# Global Instance
# =============================================================================

_task_manager: Optional[AsyncTaskManager] = None


def get_task_manager() -> AsyncTaskManager:
    """Get the global task manager instance."""
    global _task_manager
    if _task_manager is None:
        _task_manager = AsyncTaskManager()
    return _task_manager


async def start_task_manager(
    max_workers: int = 10,
    shutdown_timeout: float = 30.0,
) -> None:
    """Start the global task manager."""
    global _task_manager
    if _task_manager is None:
        _task_manager = AsyncTaskManager(
            max_workers=max_workers,
            shutdown_timeout=shutdown_timeout,
        )
    await _task_manager.start()


async def shutdown_task_manager() -> None:
    """Shutdown the global task manager."""
    global _task_manager
    if _task_manager:
        await _task_manager.shutdown()
        _task_manager = None
