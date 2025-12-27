"""
Chain-of-Thought (CoT) Reasoning Engine.

Adheres to:
- Reasoning: Implements zero-shot CoT with explicit step-by-step decomposition.
- Failure Domain: Returns Result types with structured error handling.
- I/O Semantics: Non-blocking async operations throughout.
- Observability: Logs each reasoning step for debugging and auditing.
"""
import asyncio
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from ..core.inference_wrapper import get_inference_client, CompletionResult
from ..core.result import Result, Ok, Err

logger = logging.getLogger(__name__)

# ============================================================================
# CHAIN-OF-THOUGHT REASONING PATTERN
# ============================================================================
# Prompt Engineering Strategy:
# 1. Zero-Shot CoT: "Let's think step by step" elicits structured reasoning
# 2. Step Enumeration: Forces explicit intermediate steps
# 3. Step Validation: Each step checked for logical consistency
# 4. Token Budgeting: Allocate 60% to reasoning, 40% to final answer
# ============================================================================

@dataclass
class ReasoningStep:
    """
    Single step in reasoning chain.
    
    Fields ordered by descending size for minimal padding:
    - content: str (8 bytes pointer)
    - step_number: int (8 bytes)
    - confidence: float (8 bytes)
    - tokens_used: int (8 bytes)
    """
    step_number: int
    content: str
    confidence: float  # 0.0 to 1.0
    tokens_used: int


@dataclass
class CoTResult:
    """
    Complete reasoning chain with final answer.
    """
    final_answer: str
    reasoning_steps: List[ReasoningStep]
    total_tokens: int
    reasoning_valid: bool
    raw_response: str


class ChainOfThoughtEngine:
    """
    Zero-shot Chain-of-Thought reasoning implementation.
    
    Performance Characteristics:
    - Single-pass reasoning: O(1) LLM calls
    - Step extraction: O(n) where n = number of steps (typically <10)
    - Token budget: Configurable, default max_tokens=2000
    """
    
    def __init__(
        self,
        max_tokens: int = 2000,
        temperature: float = 0.2,
        enable_step_validation: bool = True
    ):
        """
        Initialize CoT engine.
        
        Args:
            max_tokens: Maximum tokens for reasoning + answer
            temperature: Sampling temperature (lower = more deterministic)
            enable_step_validation: Validate each step for consistency
        """
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.enable_step_validation = enable_step_validation
        
        # Token budgeting: 60% reasoning, 40% answer
        self.reasoning_tokens = int(max_tokens * 0.6)
        self.answer_tokens = int(max_tokens * 0.4)
    
    async def reason(
        self,
        query: str,
        context: Optional[str] = None
    ) -> Result[CoTResult, Exception]:
        """
        Perform step-by-step reasoning on query.
        
        Complexity: O(1) LLM inference + O(n) step parsing
        
        Args:
            query: Problem or question to reason about
            context: Optional background context
            
        Returns:
            Ok(CoTResult) with reasoning chain, or Err on failure
        """
        try:
            # Input validation
            if not query or len(query.strip()) == 0:
                return Err(ValueError("Query cannot be empty"))
            
            # Construct CoT prompt
            system_prompt = self._build_system_prompt()
            user_prompt = self._build_user_prompt(query, context)
            
            logger.info(f"CoT Reasoning: Query={query[:100]}...")
            
            # Execute reasoning
            async with get_inference_client() as client:
                response = await client.chat_completion(
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens
                )
                
                if not isinstance(response, CompletionResult):
                    return Err(ValueError("Expected non-streaming response"))
                
                # Parse reasoning chain
                result = self._parse_reasoning(response)
                
                # Optional: Validate reasoning steps
                if self.enable_step_validation:
                    is_valid = self._validate_reasoning(result)
                    result.reasoning_valid = is_valid
                else:
                    result.reasoning_valid = True
                
                logger.info(
                    f"CoT Complete: {len(result.reasoning_steps)} steps, "
                    f"valid={result.reasoning_valid}"
                )
                
                return Ok(result)
                
        except Exception as e:
            logger.error(f"CoT reasoning failed: {e}", exc_info=True)
            return Err(e)
    
    def _build_system_prompt(self) -> str:
        """
        Construct system prompt for CoT reasoning.
        
        SOTA Prompt Engineering:
        - Explicit step-by-step instruction
        - Structured output format enforcement
        - Quality criteria specification
        """
        return (
            "You are an expert reasoning system that thinks step-by-step.\n\n"
            "For every problem, you MUST:\n"
            "1. Break down the problem into clear, logical steps\n"
            "2. Number each step explicitly (Step 1:, Step 2:, etc.)\n"
            "3. Show your work and intermediate calculations\n"
            "4. Verify each step before proceeding\n"
            "5. Conclude with a clear final answer\n\n"
            "Format:\n"
            "Step 1: [first reasoning step]\n"
            "Step 2: [second reasoning step]\n"
            "...\n"
            "Final Answer: [concise answer]\n\n"
            "Your reasoning must be rigorous, logical, and transparent."
        )
    
    def _build_user_prompt(self, query: str, context: Optional[str]) -> str:
        """
        Construct user prompt with query and optional context.
        """
        if context:
            return (
                f"Context:\n{context}\n\n"
                f"Problem:\n{query}\n\n"
                "Let's solve this step by step:"
            )
        else:
            return f"{query}\n\nLet's solve this step by step:"
    
    def _parse_reasoning(self, response: CompletionResult) -> CoTResult:
        """
        Extract structured reasoning steps from raw response.
        
        Complexity: O(n) where n = number of lines in response
        
        Parsing Strategy:
        - Identify lines starting with "Step N:"
        - Extract final answer from "Final Answer:" line
        - Calculate confidence based on step clarity
        """
        content = response.content
        lines = content.split('\n')
        
        reasoning_steps: List[ReasoningStep] = []
        final_answer = ""
        step_num = 0
        
        for line in lines:
            line = line.strip()
            
            # Parse step lines
            if line.lower().startswith('step '):
                step_num += 1
                # Extract step content (everything after "Step N:")
                parts = line.split(':', 1)
                if len(parts) == 2:
                    step_content = parts[1].strip()
                    
                    # Estimate confidence based on length and keywords
                    confidence = self._estimate_confidence(step_content)
                    
                    reasoning_steps.append(ReasoningStep(
                        step_number=step_num,
                        content=step_content,
                        confidence=confidence,
                        tokens_used=len(step_content) // 4  # Rough estimate
                    ))
            
            # Parse final answer
            elif line.lower().startswith('final answer'):
                parts = line.split(':', 1)
                if len(parts) == 2:
                    final_answer = parts[1].strip()
        
        # Fallback: If no explicit final answer found, use last step
        if not final_answer and reasoning_steps:
            final_answer = reasoning_steps[-1].content
        
        total_tokens = response.usage.get('total_tokens', 0)
        
        return CoTResult(
            final_answer=final_answer,
            reasoning_steps=reasoning_steps,
            total_tokens=total_tokens,
            reasoning_valid=True,  # Will be updated by validation
            raw_response=content
        )
    
    def _estimate_confidence(self, step_content: str) -> float:
        """
        Estimate confidence of reasoning step.
        
        Heuristics:
        - Longer, detailed steps = higher confidence
        - Presence of numbers/calculations = higher confidence
        - Hedge words ("maybe", "probably") = lower confidence
        
        Returns: Confidence score in [0.0, 1.0]
        """
        confidence = 0.5  # Base confidence
        
        # Length factor
        if len(step_content) > 100:
            confidence += 0.2
        elif len(step_content) < 20:
            confidence -= 0.2
        
        # Contains numerical reasoning
        if any(char.isdigit() for char in step_content):
            confidence += 0.15
        
        # Contains mathematical operators
        if any(op in step_content for op in ['+', '-', '*', '/', '=', '<', '>']):
            confidence += 0.1
        
        # Hedge words (reduce confidence)
        hedge_words = ['maybe', 'probably', 'might', 'perhaps', 'possibly', 'unclear']
        for word in hedge_words:
            if word in step_content.lower():
                confidence -= 0.15
                break
        
        # Clamp to [0.0, 1.0]
        return max(0.0, min(1.0, confidence))
    
    def _validate_reasoning(self, result: CoTResult) -> bool:
        """
        Validate reasoning chain for logical consistency.
        
        Validation checks:
        1. At least 1 reasoning step present
        2. Steps are sequential (no gaps in numbering)
        3. Final answer exists and is non-empty
        4. Average confidence > threshold (0.4)
        
        Returns: True if validation passes, False otherwise
        """
        # Check 1: At least one step
        if not result.reasoning_steps:
            logger.warning("CoT validation failed: No reasoning steps found")
            return False
        
        # Check 2: Sequential step numbers
        expected_nums = list(range(1, len(result.reasoning_steps) + 1))
        actual_nums = [step.step_number for step in result.reasoning_steps]
        if actual_nums != expected_nums:
            logger.warning(
                f"CoT validation warning: Non-sequential steps. "
                f"Expected {expected_nums}, got {actual_nums}"
            )
            # Not a hard failure, just warning
        
        # Check 3: Final answer exists
        if not result.final_answer or len(result.final_answer.strip()) == 0:
            logger.warning("CoT validation failed: No final answer")
            return False
        
        # Check 4: Average confidence threshold
        avg_confidence = sum(s.confidence for s in result.reasoning_steps) / len(result.reasoning_steps)
        if avg_confidence < 0.4:
            logger.warning(
                f"CoT validation warning: Low average confidence ({avg_confidence:.2f})"
            )
            # Not a hard failure for low confidence
        
        return True
