"""
File System Tool (Safe).

Adheres to:
- Security: Path traversal prevention (chroot-like jail).
- Atomic Operations: Writes are atomic where possible.
"""
import os
import logging
from typing import Dict, Any, Optional
from pathlib import Path
from .base import BaseTool
from ..core.result import Result, Ok, Err

logger = logging.getLogger(__name__)

class FileSystemTool(BaseTool):
    name = "file_system"
    description = "Read and write files safely within the workspace."

    def __init__(self, root_dir: str = "./workspace"):
        self.root_dir = Path(root_dir).resolve()
        if not self.root_dir.exists():
            self.root_dir.mkdir(parents=True, exist_ok=True)

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "operation": {
                            "type": "string", 
                            "enum": ["read", "write", "list"],
                            "description": "Operation to perform."
                        },
                        "path": {
                            "type": "string",
                            "description": "Relative path to file."
                        },
                        "content": {
                            "type": "string",
                            "description": "Content to write (for write op)."
                        }
                    },
                    "required": ["operation", "path"]
                }
            }
        }

    def _safe_path(self, path_str: str) -> Result[Path, Exception]:
        """Enforce jail."""
        try:
            # Resolve relative to root
            target = (self.root_dir / path_str).resolve()
            # Check if it starts with root_dir
            if not str(target).startswith(str(self.root_dir)):
                return Err(PermissionError(f"Access denied: {path_str} is outside workspace."))
            return Ok(target)
        except Exception as e:
            return Err(e)

    async def execute(self, operation: str, path: str, content: Optional[str] = None) -> Result[str, Exception]:
        path_res = self._safe_path(path)
        if path_res.is_err:
            return Err(path_res.error)
        
        target_path = path_res.value
        
        try:
            if operation == "read":
                if not target_path.exists():
                    return Err(FileNotFoundError(f"File not found: {path}"))
                text = target_path.read_text(encoding='utf-8')
                return Ok(text)
            
            elif operation == "write":
                if content is None:
                    return Err(ValueError("Content required for write."))
                # Atomic write pattern
                tmp_path = target_path.with_suffix(target_path.suffix + ".tmp")
                target_path.parent.mkdir(parents=True, exist_ok=True)
                tmp_path.write_text(content, encoding='utf-8')
                tmp_path.replace(target_path)
                return Ok(f"Successfully wrote to {path}")
                
            elif operation == "list":
                if not target_path.exists():
                     return Err(FileNotFoundError(f"Directory not found: {path}"))
                if not target_path.is_dir():
                    return Err(NotADirectoryError(f"Not a directory: {path}"))
                
                files = [p.name for p in target_path.iterdir()]
                return Ok(", ".join(files))
                
            return Err(ValueError(f"Unknown operation: {operation}"))

        except Exception as e:
            logger.error(f"FS operation failed: {e}")
            return Err(e)
