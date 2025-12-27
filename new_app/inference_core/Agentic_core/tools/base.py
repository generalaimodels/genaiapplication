"""
Base Tool Protocol.

Adheres to:
- Zero-Cost Abstraction: Protocol-based (duck typing) or Abstract Base Class.
- Standardization: Enforces schema generation compatibility.
"""
from typing import Dict, Any, Protocol, runtime_checkable
from ..core.result import Result

@runtime_checkable
class BaseTool(Protocol):
    name: str
    description: str
    
    async def execute(self, **kwargs) -> Result[str, Exception]:
        """
        Execute the tool with provided arguments.
        Must return a Result type.
        """
        ...

    def get_schema(self) -> Dict[str, Any]:
        """
        Return OpenAI-compatible JSON schema.
        """
        ...
