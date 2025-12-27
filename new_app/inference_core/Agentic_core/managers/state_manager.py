"""
State Manager with Event Sourcing and Time-Travel Debugging.

Adheres to:
- Lifecycle & Resource Determinism: Immutable state snapshots (copy-on-write).
- Failure Domain: Atomic transitions with rollback capability.
- Algorithmic Complexity: O(1) state access, O(n) for replay where n=events.
- Memory Layout: Event log (append-only) + materialized views.
"""
import asyncio
import logging
import time
import json
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from copy import deepcopy
from enum import Enum
from ..core.result import Result, Ok, Err

logger = logging.getLogger(__name__)

# ============================================================================
# EVENT SOURCING ARCHITECTURE
# ============================================================================
# Pattern: Event Sourcing with Command-Query Responsibility Segregation (CQRS)
#
# Components:
# 1. Event Log: Append-only record of all state changes
# 2. State Snapshots: Materialized views at specific points in time
# 3. Event Replay: Reconstruct state by replaying events
# 4. Time Travel: Jump to any historical state
#
# Guarantees:
# - Atomicity: All-or-nothing state transitions
# - Auditability: Complete history of changes
# - Reproducibility: Deterministic state reconstruction
#
# Complexity:
# - Append event: O(1)
# - Get current state: O(1) (materialized view)
# - Replay to timestamp: O(n) where n=events since snapshot
# ============================================================================

class EventType(Enum):
    """Types of state mutation events."""
    SET = "set"
    DELETE = "delete"
    MERGE = "merge"
    SNAPSHOT = "snapshot"


@dataclass
class StateEvent:
    """
    Single state mutation event.
    
    Field ordering (descending size):
    - payload: Dict (8 bytes pointer)
    - event_id: str (8 bytes pointer)
    - event_type: EventType (8 bytes)
    - timestamp: float (8 bytes)
    """
    event_id: str
    event_type: EventType
    timestamp: float
    payload: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StateSnapshot:
    """
    Immutable state snapshot at a point in time.
    """
    snapshot_id: str
    timestamp: float
    state: Dict[str, Any]
    event_count: int


class StateManager:
    """
    Global state coordinator with event sourcing.
    
    Performance Characteristics:
    - Read state: O(1) from materialized view
    - Write state: O(1) append to event log
    - Snapshot: O(n) where n=state size (deep copy)
    - Replay: O(e) where e=events since snapshot
    - Storage: ~1KB per event (JSON serialized)
    """
    
    def __init__(
        self,
        snapshot_interval: int = 100,
        max_events_in_memory: int = 10000
    ):
        """
        Initialize state manager.
        
        Args:
            snapshot_interval: Create snapshot every N events
            max_events_in_memory: Maximum events before archiving old ones
        """
        # Current materialized state
        self._current_state: Dict[str, Any] = {}
        
        # Event log (append-only)
        self._event_log: List[StateEvent] = []
        
        # Snapshots for fast replay
        self._snapshots: List[StateSnapshot] = []
        
        # Configuration
        self.snapshot_interval = snapshot_interval
        self.max_events_in_memory = max_events_in_memory
        
        # Concurrency control
        self._lock = asyncio.Lock()
        
        # Metrics
        self._event_counter = 0
        self._snapshot_counter = 0
        
        logger.info("State Manager initialized with event sourcing")
    
    async def set(
        self,
        key: str,
        value: Any,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Result[bool, Exception]:
        """
        Set state value (creates SET event).
        
        Complexity: O(1)
        
        Args:
            key: State key
            value: Value to store
            metadata: Optional event metadata
            
        Returns:
            Ok(True) on success
        """
        try:
            event = self._create_event(
                EventType.SET,
                {"key": key, "value": value},
                metadata
            )
            
            async with self._lock:
                # Apply to current state
                self._current_state[key] = value
                
                # Append to event log
                self._event_log.append(event)
                self._event_counter += 1
                
                # Periodic snapshot
                await self._maybe_snapshot()
            
            logger.debug(f"State set: {key} = {value}")
            return Ok(True)
            
        except Exception as e:
            return Err(e)
    
    async def get(
        self,
        key: str,
        default: Any = None
    ) -> Any:
        """
        Get current state value.
        
        Complexity: O(1)
        """
        # No lock needed for read (atomic dict access)
        return self._current_state.get(key, default)
    
    async def delete(
        self,
        key: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Result[bool, Exception]:
        """
        Delete state key (creates DELETE event).
        
        Complexity: O(1)
        """
        try:
            event = self._create_event(
                EventType.DELETE,
                {"key": key},
                metadata
            )
            
            async with self._lock:
                if key in self._current_state:
                    del self._current_state[key]
                
                self._event_log.append(event)
                self._event_counter += 1
                
                await self._maybe_snapshot()
            
            logger.debug(f"State deleted: {key}")
            return Ok(True)
            
        except Exception as e:
            return Err(e)
    
    async def merge(
        self,
        updates: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ) -> Result[bool, Exception]:
        """
        Merge multiple updates atomically (single MERGE event).
        
        Complexity: O(k) where k=len(updates)
        """
        try:
            event = self._create_event(
                EventType.MERGE,
                {"updates": updates},
                metadata
            )
            
            async with self._lock:
                # Atomic merge
                self._current_state.update(updates)
                
                self._event_log.append(event)
                self._event_counter += 1
                
                await self._maybe_snapshot()
            
            logger.debug(f"State merged: {len(updates)} keys")
            return Ok(True)
            
        except Exception as e:
            return Err(e)
    
    async def get_all(self) -> Dict[str, Any]:
        """
        Get complete current state (shallow copy).
        
        Complexity: O(n) where n=state size
        """
        async with self._lock:
            return dict(self._current_state)
    
    async def create_snapshot(self) -> Result[StateSnapshot, Exception]:
        """
        Create immutable state snapshot.
        
        Complexity: O(n) deep copy
        """
        try:
            async with self._lock:
                snapshot = StateSnapshot(
                    snapshot_id=f"snapshot_{self._snapshot_counter}",
                    timestamp=time.time(),
                    state=deepcopy(self._current_state),
                    event_count=len(self._event_log)
                )
                
                self._snapshots.append(snapshot)
                self._snapshot_counter += 1
                
                logger.info(
                    f"Created snapshot {snapshot.snapshot_id} "
                    f"({len(self._current_state)} keys)"
                )
                
                return Ok(snapshot)
                
        except Exception as e:
            return Err(e)
    
    async def _maybe_snapshot(self) -> None:
        """
        Conditionally create snapshot based on interval.
        
        INTERNAL: Called under lock.
        """
        if self._event_counter % self.snapshot_interval == 0:
            await self.create_snapshot()
    
    async def replay_to_timestamp(
        self,
        target_timestamp: float
    ) -> Result[Dict[str, Any], Exception]:
        """
        Reconstruct state at specific timestamp (time travel).
        
        Complexity: O(e) where e=events to replay
        
        Strategy:
        1. Find latest snapshot before target
        2. Replay events from snapshot to target
        3. Return reconstructed state
        
        Args:
            target_timestamp: Unix timestamp to replay to
            
        Returns:
            Ok(state_dict) at that timestamp
        """
        try:
            async with self._lock:
                # Find latest snapshot before target
                applicable_snapshots = [
                    s for s in self._snapshots 
                    if s.timestamp <= target_timestamp
                ]
                
                if applicable_snapshots:
                    base_snapshot = applicable_snapshots[-1]
                    reconstructed = deepcopy(base_snapshot.state)
                    start_index = base_snapshot.event_count
                else:
                    # No snapshot, start from empty
                    reconstructed = {}
                    start_index = 0
                
                # Replay events from snapshot to target
                for event in self._event_log[start_index:]:
                    if event.timestamp > target_timestamp:
                        break
                    
                    self._apply_event(reconstructed, event)
                
                logger.info(f"Replayed state to timestamp {target_timestamp}")
                return Ok(reconstructed)
                
        except Exception as e:
            return Err(e)
    
    def _apply_event(self, state: Dict[str, Any], event: StateEvent) -> None:
        """
        Apply single event to state dict.
        
        INTERNAL: Mutates state in-place.
        """
        if event.event_type == EventType.SET:
            key = event.payload["key"]
            value = event.payload["value"]
            state[key] = value
            
        elif event.event_type == EventType.DELETE:
            key = event.payload["key"]
            if key in state:
                del state[key]
                
        elif event.event_type == EventType.MERGE:
            updates = event.payload["updates"]
            state.update(updates)
    
    def _create_event(
        self,
        event_type: EventType,
        payload: Dict[str, Any],
        metadata: Optional[Dict[str, Any]]
    ) -> StateEvent:
        """
        Create new state event.
        
        Complexity: O(1)
        """
        import uuid
        return StateEvent(
            event_id=str(uuid.uuid4()),
            event_type=event_type,
            timestamp=time.time(),
            payload=payload,
            metadata=metadata or {}
        )
    
    async def get_event_log(
        self,
        since_timestamp: Optional[float] = None,
        limit: int = 100
    ) -> Result[List[StateEvent], Exception]:
        """
        Get event log (for auditing).
        
        Complexity: O(n) where n=filtered events
        """
        try:
            async with self._lock:
                if since_timestamp:
                    filtered = [
                        e for e in self._event_log 
                        if e.timestamp >= since_timestamp
                    ]
                else:
                    filtered = self._event_log
                
                return Ok(filtered[-limit:])
                
        except Exception as e:
            return Err(e)
    
    async def rollback_to_snapshot(
        self,
        snapshot_id: str
    ) -> Result[bool, Exception]:
        """
        Rollback state to specific snapshot.
        
        WARNING: Destructive operation, truncates events after snapshot.
        
        Complexity: O(n) deep copy
        """
        try:
            async with self._lock:
                # Find snapshot
                snapshot = None
                for s in self._snapshots:
                    if s.snapshot_id == snapshot_id:
                        snapshot = s
                        break
                
                if not snapshot:
                    return Err(ValueError(f"Snapshot {snapshot_id} not found"))
                
                # Rollback state
                self._current_state = deepcopy(snapshot.state)
                
                # Truncate event log
                self._event_log = self._event_log[:snapshot.event_count]
                
                logger.warning(
                    f"Rolled back to snapshot {snapshot_id}, "
                    f"truncated {self._event_counter - snapshot.event_count} events"
                )
                
                return Ok(True)
                
        except Exception as e:
            return Err(e)
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get state manager statistics."""
        async with self._lock:
            return {
                "current_state_keys": len(self._current_state),
                "total_events": len(self._event_log),
                "snapshots": len(self._snapshots),
                "event_counter": self._event_counter,
                "snapshot_counter": self._snapshot_counter,
                "latest_snapshot": (
                    self._snapshots[-1].snapshot_id if self._snapshots else None
                )
            }
