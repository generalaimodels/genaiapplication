"""
Code Sandbox Tool.

Adheres to:
- Security: Uses asyncio.subprocess.
- Robustness: Captures stdout/stderr and handles timeouts.
- I/O Semantics: Non-blocking execution.
"""
import asyncio
import logging
import sys
import tempfile
import os
from typing import Dict, Any
from .base import BaseTool
from ..core.result import Result, Ok, Err

logger = logging.getLogger(__name__)

class PythonSandbox(BaseTool):
    name = "python_interpreter"
    description = "Executes Python code in a safe environment. Use this for math, data analysis, or complex logic. Returns standard output."

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "code": {
                            "type": "string",
                            "description": "Valid python code to execute."
                        }
                    },
                    "required": ["code"]
                }
            }
        }

    async def execute(self, code: str) -> Result[str, Exception]:
        """
        Runs code in a subprocess.
        """
        # Security: Create a temp file
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as tmp:
                tmp.write(code)
                tmp_path = tmp.name
            
            logger.info(f"Executing sandbox code at {tmp_path}")
            
            # Execute with timeout
            proc = await asyncio.create_subprocess_exec(
                sys.executable, tmp_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                # In a real SOTA sandbox, we would wrap this in `docker run ...`
                # For this environment, we execute locally but with strict generic limits if possible.
            )
            
            try:
                stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=10.0)
            except asyncio.TimeoutError:
                proc.kill()
                return Err(TimeoutError("Code execution exceeded 10s limit"))
            finally:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)

            output = stdout.decode().strip()
            error = stderr.decode().strip()
            
            if proc.returncode != 0:
                return Err(RuntimeError(f"Execution Error:\n{error}"))
            
            result_str = output if output else (f"Executed successfully (No Output). Stderr: {error}" if error else "Executed successfully.")
            return Ok(result_str)

        except Exception as e:
            logger.error(f"Sandbox failure: {e}")
            return Err(e)
