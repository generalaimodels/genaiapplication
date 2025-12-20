# =============================================================================
# RESPONSE HANDLER - Response Processing and Formatting
# =============================================================================
# Handles response processing, validation, and formatting.
# =============================================================================

from __future__ import annotations
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
import re
import logging

from clients.base_client import ChatResponse

logger = logging.getLogger(__name__)


@dataclass
class ProcessedResponse:
    """
    Processed and validated response.
    
    Attributes:
        content: Processed content
        original_content: Original content
        format: Content format (text, json, markdown)
        is_valid: Whether response passed validation
        metadata: Processing metadata
    """
    content: str
    original_content: str
    format: str = "text"
    is_valid: bool = True
    validation_errors: List[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.validation_errors is None:
            self.validation_errors = []
        if self.metadata is None:
            self.metadata = {}


class ResponseHandler:
    """
    Handles response processing and formatting.
    
    Features:
        - Response validation
        - Format detection and conversion
        - Content cleaning
        - Metadata extraction
    
    Example:
        >>> handler = ResponseHandler()
        >>> 
        >>> # Process LLM response
        >>> processed = handler.process(response)
        >>> print(processed.content)
    """
    
    def __init__(
        self,
        max_length: Optional[int] = None,
        strip_thinking: bool = True,
        validate: bool = True
    ):
        """
        Initialize response handler.
        
        Args:
            max_length: Maximum response length
            strip_thinking: Remove thinking tags
            validate: Enable validation
        """
        self.max_length = max_length
        self.strip_thinking = strip_thinking
        self.validate = validate
    
    def process(
        self,
        response: ChatResponse,
        expected_format: Optional[str] = None
    ) -> ProcessedResponse:
        """
        Process a chat response.
        
        Args:
            response: Raw ChatResponse
            expected_format: Expected content format
            
        Returns:
            ProcessedResponse with processed content
        """
        content = response.content or ""
        original = content
        
        # Clean content
        content = self._clean_content(content)
        
        # Detect format
        detected_format = self._detect_format(content)
        final_format = expected_format or detected_format
        
        # Validate
        errors = []
        is_valid = True
        if self.validate:
            is_valid, errors = self._validate_content(content, final_format)
        
        # Trim if needed
        if self.max_length and len(content) > self.max_length:
            content = content[:self.max_length] + "..."
        
        return ProcessedResponse(
            content=content,
            original_content=original,
            format=final_format,
            is_valid=is_valid,
            validation_errors=errors,
            metadata={
                "model": response.model,
                "finish_reason": response.finish_reason,
                "usage": response.usage,
            }
        )
    
    def _clean_content(self, content: str) -> str:
        """Clean response content."""
        # Strip whitespace
        content = content.strip()
        
        # Remove thinking tags if enabled
        if self.strip_thinking:
            content = re.sub(r'<thinking>.*?</thinking>', '', content, flags=re.DOTALL)
            content = re.sub(r'<thought>.*?</thought>', '', content, flags=re.DOTALL)
        
        # Normalize whitespace
        content = re.sub(r'\n{3,}', '\n\n', content)
        
        return content.strip()
    
    def _detect_format(self, content: str) -> str:
        """Detect content format."""
        content_stripped = content.strip()
        
        # Check for JSON
        if content_stripped.startswith('{') or content_stripped.startswith('['):
            return "json"
        
        # Check for Markdown indicators
        if any(marker in content for marker in ['```', '##', '**', '- ', '1. ']):
            return "markdown"
        
        return "text"
    
    def _validate_content(
        self,
        content: str,
        format_type: str
    ) -> tuple[bool, List[str]]:
        """Validate content."""
        errors = []
        
        # Check for empty
        if not content:
            errors.append("Empty response")
            return False, errors
        
        # Format-specific validation
        if format_type == "json":
            try:
                import json
                json.loads(content)
            except json.JSONDecodeError as e:
                errors.append(f"Invalid JSON: {e}")
        
        return len(errors) == 0, errors
    
    def format_for_display(
        self,
        content: str,
        format_type: str = "text"
    ) -> str:
        """Format content for display."""
        if format_type == "json":
            try:
                import json
                parsed = json.loads(content)
                return json.dumps(parsed, indent=2)
            except Exception:
                pass
        
        return content
    
    def extract_code_blocks(self, content: str) -> List[Dict[str, str]]:
        """Extract code blocks from markdown content."""
        pattern = r'```(\w*)\n(.*?)```'
        matches = re.findall(pattern, content, re.DOTALL)
        
        return [
            {"language": lang or "text", "code": code.strip()}
            for lang, code in matches
        ]
    
    def extract_json(self, content: str) -> Optional[Dict]:
        """Extract JSON from content."""
        # Try direct parse
        try:
            import json
            return json.loads(content)
        except Exception:
            pass
        
        # Try to find JSON in content
        json_pattern = r'\{[^{}]*\}'
        matches = re.findall(json_pattern, content, re.DOTALL)
        
        for match in matches:
            try:
                import json
                return json.loads(match)
            except Exception:
                continue
        
        return None
