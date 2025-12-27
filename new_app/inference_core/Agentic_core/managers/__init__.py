"""
Agentic Framework 3.0 - Managers Module.

Exports:
- AgentManager: Multi-agent coordination
- ResourceManager: GPU/Memory/Concurrency allocation
- StateManager: Global state with event sourcing
"""
from .agent_manager import (
    AgentManager,
    AgentCapability,
    AgentStatus,
    AgentMetrics,
    RegisteredAgent
)
from .resource_manager import (
    ResourceManager,
    ResourceType,
    GPUInfo,
    MemoryQuota,
    ConcurrencyQuota
)
from .state_manager import (
    StateManager,
    StateEvent,
    StateSnapshot,
    EventType
)

__all__ = [
    # Agent Management
    "AgentManager",
    "AgentCapability",
    "AgentStatus",
    "AgentMetrics",
    "RegisteredAgent",
    
    # Resource Management
    "ResourceManager",
    "ResourceType",
    "GPUInfo",
    "MemoryQuota",
    "ConcurrencyQuota",
    
    # State Management
    "StateManager",
    "StateEvent",
    "StateSnapshot",
    "EventType",
]
