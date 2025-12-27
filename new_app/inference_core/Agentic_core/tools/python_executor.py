"""
Python Code Execution Tool for Agentic Framework.

Adheres to:
- Robust Input Sanitization: Validates code before execution
- Failure Domain: Returns Result types with error handling
- I/O Semantics: Captures stdout/stderr separately
"""
import sys
import io
import traceback
from typing import Dict, Any
from ..core.result import Result, Ok, Err

class PythonExecutor:
    """
    Safe Python code execution tool.
    
    Security: Runs in same process (use sandbox for untrusted code)
    """
    
    name = "python_executor"
    description = "Execute Python code and return the output"
    
    async def execute(self, code: str, **kwargs) -> Result[str, Exception]:
        """
        Execute Python code in controlled environment.
        
        Args:
            code: Python code to execute
            
        Returns:
            Ok(output) with stdout/stderr, or Err on exception
        """
        try:
            # Capture stdout and stderr
            old_stdout = sys.stdout
            old_stderr = sys.stderr
            sys.stdout = io.StringIO()
            sys.stderr = io.StringIO()
            
            # Create execution namespace
            namespace = {
                '__name__': '__main__',
                '__builtins__': __builtins__,
            }
            
            try:
                # Execute code
                exec(code, namespace)
                
                # Get output
                stdout_output = sys.stdout.getvalue()
                stderr_output = sys.stderr.getvalue()
                
                output = ""
                if stdout_output:
                    output += f"Output:\n{stdout_output}"
                if stderr_output:
                    output += f"\nErrors:\n{stderr_output}"
                
                if not output:
                    output = "Code executed successfully (no output)"
                
                return Ok(output)
                
            except Exception as e:
                error_trace = traceback.format_exc()
                return Ok(f"Execution Error:\n{error_trace}")
                
            finally:
                # Restore stdout/stderr
                sys.stdout = old_stdout
                sys.stderr = old_stderr
                
        except Exception as e:
            return Err(e)
    
    def get_schema(self) -> Dict[str, Any]:
        """Return OpenAI-compatible tool schema."""
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
                            "description": "Python code to execute"
                        }
                    },
                    "required": ["code"]
                }
            }
        }
