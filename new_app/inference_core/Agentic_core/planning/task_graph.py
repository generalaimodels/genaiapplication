"""
Task Dependency Graph.

Adheres to:
- Algorithmic Complexity: O(1) dependency checking using bitmasks (limited to 64/128 deps usually) or O(E) for general DAGs.
- Here we use adjacency sets for flexibility, optimized for fast lookups.
"""
from typing import Dict, Set, List, Optional
from dataclasses import dataclass, field
import enum

class TaskStatus(enum.Enum):
    PENDING = "PENDING"
    READY = "READY"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"

@dataclass
class TaskNode:
    task_id: str
    description: str
    dependencies: Set[str] = field(default_factory=set)
    dependents: Set[str] = field(default_factory=set)
    status: TaskStatus = TaskStatus.PENDING
    result: Optional[str] = None
    priority: int = 0

class TaskGraph:
    """
    Directed Acyclic Graph (DAG) manager for tasks.
    """
    def __init__(self):
        self.nodes: Dict[str, TaskNode] = {}
        self.ready_queue: List[str] = []

    def add_task(self, task_id: str, description: str, dependencies: List[str] = None):
        if dependencies is None:
            dependencies = []
            
        node = TaskNode(task_id=task_id, description=description)
        
        # O(N) where N is number of dependencies
        for dep_id in dependencies:
            if dep_id not in self.nodes:
                raise ValueError(f"Dependency {dep_id} does not exist")
            node.dependencies.add(dep_id)
            self.nodes[dep_id].dependents.add(task_id)
            
        self.nodes[task_id] = node
        
        if not node.dependencies:
            node.status = TaskStatus.READY
            self.ready_queue.append(task_id)

    def mark_completed(self, task_id: str, result: str) -> List[str]:
        """
        Marks task as complete and returns newly ready tasks.
        Complexity: O(D) where D is number of dependents.
        """
        node = self.nodes[task_id]
        node.status = TaskStatus.COMPLETED
        node.result = result
        
        newly_ready = []
        for dep_id in node.dependents:
            dep_node = self.nodes[dep_id]
            # Check if all dependencies are satisfied
            if all(self.nodes[d].status == TaskStatus.COMPLETED for d in dep_node.dependencies):
                dep_node.status = TaskStatus.READY
                newly_ready.append(dep_id)
                self.ready_queue.append(dep_id)
                
        return newly_ready

    def get_ready_tasks(self) -> List[TaskNode]:
        """Return tasks that can be executed immediately."""
        # Ideally this consumes from a queue
        ready = [self.nodes[tid] for tid in self.ready_queue if self.nodes[tid].status == TaskStatus.READY]
        return ready
