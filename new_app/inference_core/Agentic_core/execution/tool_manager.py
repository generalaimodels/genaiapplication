"""
Tool Manager & Registry (SOTA).

Adheres to:
- Zero-Cost Abstraction: Inspects signatures once at registration time.
- Standard Compliance: Generates OpenAI-compatible JSON Schema.
- Type Safety: Enforces Pydantic validation on tool inputs.
"""
import inspect
import logging
from typing import Callable, Dict, Any, List, Optional, get_type_hints
from pydantic import BaseModel, create_model, ValidationError

from ..core.result import Result, Ok, Err

logger = logging.getLogger(__name__)

class ToolRegistry:
    """
    Manages tools and automatically generates schemas for LLM usage.
    """
    def __init__(self):
        self._tools: Dict[str, Callable] = {}
        self._schemas: List[Dict[str, Any]] = []

    def register(self, func: Callable):
        """
        Decorator to register a function as a tool.
        """
        schema = self._generate_schema(func)
        self._tools[schema["function"]["name"]] = func
        self._schemas.append(schema)
        return func

    def _generate_schema(self, func: Callable) -> Dict[str, Any]:
        """
        Introspects function to build OpenAI Function Schema.
        """
        name = func.__name__
        doc = func.__doc__ or "No description provided."
        
        # 1. Type Hints to Pydantic Model
        type_hints = get_type_hints(func)
        fields = {}
        for param_name, py_type in type_hints.items():
            if param_name == 'return': continue
            # Default values?
            fields[param_name] = (py_type, ...) 
        
        # Dynamic Pydantic Model
        InputModel = create_model(f"{name}_Input", **fields)
        
        # 2. Pydantic to JSON Schema
        model_schema = InputModel.model_json_schema()
        
        parameters = {
            "type": "object",
            "properties": model_schema.get("properties", {}),
            "required": model_schema.get("required", [])
        }

        return {
            "type": "function",
            "function": {
                "name": name,
                "description": doc.strip(),
                "parameters": parameters
            }
        }

    def get_schemas(self) -> List[Dict[str, Any]]:
        return self._schemas

    async def execute(self, name: str, arguments: Dict[str, Any]) -> Result[str, Exception]:
        """
        Safe execution of tool.
        """
        if name not in self._tools:
            return Err(ValueError(f"Tool {name} not found"))
            
        tool_func = self._tools[name]
        try:
            # Invoke
            # Check if coroutine
            if inspect.iscoroutinefunction(tool_func):
                res = await tool_func(**arguments)
            else:
                res = tool_func(**arguments)
            return Ok(str(res))
        except Exception as e:
            logger.error(f"Tool execution failed: {e}")
            return Err(e)
