"""
Web Browser Tool.

Adheres to:
- I/O Semantics: Uses httpx for non-blocking requests.
- Input Sanitization: Strips HTML to Markdown for safe LLM consumption.
"""
import httpx
import logging
from typing import Dict, Any
import re
from .base import BaseTool
from ..core.result import Result, Ok, Err

logger = logging.getLogger(__name__)

class WebBrowser(BaseTool):
    name = "web_browser"
    description = "Retrieves content from a URL and converts it to readable markdown. Use for external research."

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "url": {
                            "type": "string",
                            "description": "The URL to visit."
                        }
                    },
                    "required": ["url"]
                }
            }
        }

    async def execute(self, url: str) -> Result[str, Exception]:
        """
        Fetches URL.
        """
        try:
            async with httpx.AsyncClient(follow_redirects=True, timeout=10.0) as client:
                resp = await client.get(url)
                resp.raise_for_status()
                
                # Simple HTML to Markdown conversion (SOTA would use readability.js or similar)
                html = resp.text
                
                # 1. Strip scripts/styles
                html = re.sub(r'<(script|style)[^>]*>.*?</\1>', '', html, flags=re.DOTALL)
                
                # 2. Extract text (Approximation)
                # Removing tags
                text = re.sub(r'<[^>]+>', ' ', html)
                # Collapse whitespace
                text = re.sub(r'\s+', ' ', text).strip()
                
                # Limit length for context window safety
                if len(text) > 8000:
                    text = text[:8000] + "... [Truncated]"
                    
                return Ok(text)

        except Exception as e:
            logger.error(f"Browser failure: {e}")
            return Err(e)
