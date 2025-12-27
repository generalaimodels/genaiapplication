"""
Episodic Memory Module.

Adheres to:
- Long-term Memory: Interface to Vector Store for retrieving past episodes.
"""
from typing import List, Dict
from ..memory.vector_store import VectorStore
from ..core.result import Result

class EpisodicMemory:
    def __init__(self):
        self.vector_store = VectorStore(collection_name="episodic_log")

    async def remember(self, episode_summary: str, metadata: Dict):
        """Save a completed episode."""
        await self.vector_store.add_texts([episode_summary], [metadata])

    async def recall_similar(self, query: str, k: int = 3) -> List[str]:
        """Retrieve relevant past experiences."""
        res = await self.vector_store.search(query, k=k)
        if res.is_ok:
            # Unpack tuples (text, score, meta)
            return [f"Past Experience: {item[0]}" for item in res.value]
        return []
