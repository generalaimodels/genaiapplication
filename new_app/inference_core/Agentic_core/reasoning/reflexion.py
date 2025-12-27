"""
Reflexion Reasoning Module.

Adheres to:
- Reasoning: Implements "Critique -> Fix" loop.
- SOTA: Uses separate "Critic" role prompting.
"""
from typing import Optional, List, Dict
import logging
from ..core.inference_wrapper import get_inference_client
from ..core.result import Result, Ok, Err

logger = logging.getLogger(__name__)

class ReflexionEngine:
    """
    Critiques and refines agent outputs.
    """
    async def critique(self, task_description: str, result_content: str) -> Result[Optional[str], Exception]:
        """
        Analyzes the result. Returns a 'Critique' string if flaws found, or None if acceptable.
        """
        system_prompt = (
            "You are a strict Critic. precise pass/fail"
            "Review the user's task and the agent's result.\n"
            "If the result is correct and sufficient, reply with 'PASS'.\n"
            "If there are errors, missing info, or hallucinations, reply with 'FAIL: <reason>'"
        )
        
        user_prompt = f"Task: {task_description}\nResult: {result_content}"
        
        try:
            async with get_inference_client() as client:
                response = await client.chat_completion(
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.0 # Deterministic critique
                )
                
                content = response.content.strip() if hasattr(response, 'content') else ""
                
                if content.startswith("PASS"):
                    return Ok(None) # No critique needed
                else:
                    return Ok(content) # Return the critique prompt for the next try
                    
        except Exception as e:
            logger.error(f"Reflexion failed: {e}")
            return Err(e)
