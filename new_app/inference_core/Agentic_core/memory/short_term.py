"""
Short-Term Working Memory (SOTA).

Adheres to:
- Memory Layout: Hybrid structure. Linear Chat Deque for narrative + Key-Value Map for structured artifacts.
- "Mapping Idea": Explicitly maps task outputs to accessible keys for "Immediate Focus".
"""
from collections import deque
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
import datetime
import json

@dataclass
class MemoryItem:
    role: str
    content: str
    timestamp: float = field(default_factory=lambda: datetime.datetime.utcnow().timestamp())
    type: str = "text"  # text, image, tool_result

@dataclass
class Artifact:
    """Structured data produced by tasks."""
    key: str
    value: Any
    description: str
    source_task_id: str
    timestamp: float = field(default_factory=lambda: datetime.datetime.utcnow().timestamp())

class ShortTermMemory:
    """
    Dual-Stream Memory:
    1. Stream: Linear conversation history (Narrative).
    2. Map: Structured artifact storage (Concepts/Results) for O(1) retrieval.
    """
    def __init__(self, capacity: int = 50):
        self._buffer: deque[MemoryItem] = deque(maxlen=capacity)
        # The "Mapping Idea": Key-Value store for immediate context resolving
        self._artifact_map: Dict[str, Artifact] = {}
        self._focus_keys: List[str] = [] # Keys currently in "Focus"

    def add(self, role: str, content: str, msg_type: str = "text"):
        """Add narrative entry."""
        item = MemoryItem(role=role, content=content, type=msg_type)
        self._buffer.append(item)

    def store_artifact(self, key: str, value: Any, description: str, task_id: str):
        """
        Maps a specific result (Concept) to memory.
        Example: store_artifact("weather_sf", "20C", "Current weather in SF", "task_1")
        """
        artifact = Artifact(key, value, description, task_id)
        self._artifact_map[key] = artifact
        # Auto-focus recent artifacts?
        if key not in self._focus_keys:
            self._focus_keys.append(key)

    def get_artifact(self, key: str) -> Optional[Any]:
        """O(1) Immediate retrieval."""
        art = self._artifact_map.get(key)
        return art.value if art else None

    def get_recent(self, k: int = 10) -> List[MemoryItem]:
        """Get last k narrative items."""
        if k >= len(self._buffer):
            return list(self._buffer)
        return list(self._buffer)[-k:]

    def get_working_memory_summary(self) -> str:
        """
        Returns a context string of currently mapped artifacts.
        This provides the "Focusing Context" for the LLM.
        """
        if not self._artifact_map:
            return ""
        
        summary = ["--- WORKING MEMORY MAP ---"]
        for key, art in self._artifact_map.items():
            # Truncate value if too long (sanity check)
            val_str = str(art.value)
            if len(val_str) > 200:
                val_str = val_str[:200] + "..."
            summary.append(f"[{key}] ({art.description}): {val_str}")
        summary.append("--------------------------")
        return "\n".join(summary)

    def to_prompt_format(self) -> List[Dict[str, str]]:
        """
        Injects Narrative + Working Memory Map into prompt messages.
        """
        msgs = [{"role": m.role, "content": m.content} for m in self._buffer]
        
        # Inject Memory Map as a System 'Thought' at the end of context if exists
        # or prepended to the last user message?
        # A common SOTA pattern is a System message right before generation.
        wm_context = self.get_working_memory_summary()
        if wm_context:
            msgs.append({"role": "system", "content": f"Current Knowledge Context:\n{wm_context}"})
            
        return msgs
    
    def clear(self):
        self._buffer.clear()
        self._artifact_map.clear()
        self._focus_keys.clear()
