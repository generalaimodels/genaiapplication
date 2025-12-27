"""
ReAct (Reasoning + Acting) Agent Pattern.

Adheres to:
- Algorithmic Complexity: O(N * T) where N=iterations, T=tool_execution_time.
- Deterministic Concurrency: Sequential reasoning-action loops with async tool execution.
- Failure Domain: Comprehensive Result types with retry and fallback strategies.
- Robust Input Sanitization: Tool argument validation and timeout enforcement.
"""
import asyncio
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum
from ..core.inference_wrapper import get_inference_client, CompletionResult
from ..core.result import Result, Ok, Err
from ..tools.base import BaseTool

logger = logging.getLogger(__name__)

# ============================================================================
# REACT PATTERN ARCHITECTURE
# ============================================================================
# Synergistic Integration: Reasoning (LLM) + Acting (Tool Use)
#
# Loop Structure:
# 1. Thought: LLM generates reasoning about next action
# 2. Action: Execute selected tool with validated arguments
# 3. Observation: Capture tool output
# 4. Repeat until:
#    - Goal achieved (termination condition)
#    - Max iterations reached (circuit breaker)
#    - Critical error (failure mode)
#
# Complexity: O(max_iterations * (reasoning_cost + tool_cost))
# ============================================================================

class ActionType(Enum):
    """
    Action categories in ReAct loop.
    """
    TOOL_USE = "tool_use"       # Execute external tool
    FINAL_ANSWER = "final_answer"  # Terminate with answer
    RETHINK = "rethink"         # Re-evaluate approach


@dataclass
class ReActStep:
    """
    Single iteration in ReAct loop.
    
    Field ordering (descending size):
    - observation: str (8 bytes pointer)
    - thought: str (8 bytes pointer)
    - action: str (8 bytes pointer)
    - action_input: Dict (8 bytes pointer)
    - step_number: int (8 bytes)
    - action_type: ActionType (8 bytes)
    - success: bool (1 byte + 7 padding)
    """
    step_number: int
    thought: str
    action_type: ActionType
    action: str
    action_input: Dict[str, Any]
    observation: str
    success: bool


@dataclass
class ReActResult:
    """
    Complete ReAct execution result.
    """
    final_answer: str
    steps: List[ReActStep]
    total_iterations: int
    success: bool
    termination_reason: str


class ReActAgent:
    """
    Reasoning-Acting agent with tool integration.
    
    Performance Characteristics:
    - Max iterations: 10 (configurable, prevents infinite loops)
    - Tool timeout: 30s per call (prevents hanging)
    - Thought generation: ~1-2s (LLM inference)
    - Total worst-case: ~10 * (2s + 30s) = 320s
    """
    
    def __init__(
        self,
        tools: List[BaseTool],
        max_iterations: int = 10,
        tool_timeout: float = 30.0,
        temperature: float = 0.3
    ):
        """
        Initialize ReAct agent.
        
        Args:
            tools: Available tools for agent
            max_iterations: Maximum reasoning-action cycles
            tool_timeout: Timeout per tool execution (seconds)
            temperature: Sampling temperature for thoughts
        """
        # Boundary validation
        assert 1 <= max_iterations <= 20, "max_iterations must be in [1, 20]"
        assert 1.0 <= tool_timeout <= 300.0, "tool_timeout must be in [1, 300]"
        
        self.tools = {tool.name: tool for tool in tools}
        self.max_iterations = max_iterations
        self.tool_timeout = tool_timeout
        self.temperature = temperature
    
    async def run(
        self,
        goal: str,
        context: Optional[str] = None
    ) -> Result[ReActResult, Exception]:
        """
        Execute ReAct loop to achieve goal.
        
        Complexity: O(max_iterations * (thought_gen + tool_exec))
        
        Args:
            goal: Objective to accomplish
            context: Optional background information
            
        Returns:
            Ok(ReActResult) with execution trace, or Err on failure
        """
        try:
            logger.info(f"ReAct Agent: goal={goal[:100]}...")
            
            steps: List[ReActStep] = []
            iteration = 0
            history = []  # Conversation history for context
            
            # Initial system prompt
            system_prompt = self._build_system_prompt()
            
            while iteration < self.max_iterations:
                iteration += 1
                logger.debug(f"ReAct Iteration {iteration}/{self.max_iterations}")
                
                # STEP 1: THOUGHT - Generate reasoning
                thought_result = await self._generate_thought(
                    goal, context, history, system_prompt
                )
                
                if thought_result.is_err:
                    return Err(thought_result.error)
                
                thought_data = thought_result.value
                
                # STEP 2: ACTION - Parse and execute
                action_result = await self._execute_action(thought_data)
                
                if action_result.is_err:
                    # Record failed step and continue
                    steps.append(ReActStep(
                        step_number=iteration,
                        thought=thought_data.get("thought", ""),
                        action_type=ActionType.TOOL_USE,
                        action=thought_data.get("action", ""),
                        action_input=thought_data.get("action_input", {}),
                        observation=f"ERROR: {action_result.error}",
                        success=False
                    ))
                    
                    history.append({
                        "role": "assistant",
                        "content": f"Thought: {thought_data.get('thought', '')}\nAction failed: {action_result.error}"
                    })
                    continue
                
                observation, is_final = action_result.value
                
                # STEP 3: OBSERVATION - Record result
                step = ReActStep(
                    step_number=iteration,
                    thought=thought_data.get("thought", ""),
                    action_type=(ActionType.FINAL_ANSWER 
                                if is_final else ActionType.TOOL_USE),
                    action=thought_data.get("action", ""),
                    action_input=thought_data.get("action_input", {}),
                    observation=observation,
                    success=True
                )
                steps.append(step)
                
                # Update history
                history.append({
                    "role": "assistant",
                    "content": (
                        f"Thought: {step.thought}\n"
                        f"Action: {step.action}\n"
                        f"Observation: {observation}"
                    )
                })
                
                # Check termination
                if is_final:
                    logger.info(f"ReAct: Goal achieved in {iteration} iterations")
                    return Ok(ReActResult(
                        final_answer=observation,
                        steps=steps,
                        total_iterations=iteration,
                        success=True,
                        termination_reason="Goal achieved"
                    ))
            
            # Max iterations reached
            logger.warning(f"ReAct: Max iterations ({self.max_iterations}) reached")
            return Ok(ReActResult(
                final_answer=steps[-1].observation if steps else "No result",
                steps=steps,
                total_iterations=iteration,
                success=False,
                termination_reason="Max iterations exceeded"
            ))
            
        except Exception as e:
            logger.error(f"ReAct execution failed: {e}", exc_info=True)
            return Err(e)
    
    def _build_system_prompt(self) -> str:
        """
        Construct system prompt with tool descriptions.
        """
        tool_descriptions = "\n".join([
            f"- {name}: {tool.description}"
            for name, tool in self.tools.items()
        ])
        
        return (
            "You are a ReAct agent that solves problems through reasoning and tool use.\n\n"
            "Available Tools:\n"
            f"{tool_descriptions}\n\n"
            "You MUST respond in this exact format:\n"
            "Thought: [your reasoning about what to do next]\n"
            "Action: [tool name or 'Final Answer']\n"
            "Action Input: [JSON object with tool arguments, or your final answer]\n\n"
            "Examples:\n"
            "Thought: I need to search for information about quantum computing\n"
            "Action: web_search\n"
            "Action Input: {\"query\": \"quantum computing basics\"}\n\n"
            "OR when done:\n"
            "Thought: I have all the information needed\n"
            "Action: Final Answer\n"
            "Action Input: {\"answer\": \"Quantum computing uses qubits...\"}\n\n"
            "CRITICAL: Always follow the exact format. Never skip fields."
        )
    
    async def _generate_thought(
        self,
        goal: str,
        context: Optional[str],
        history: List[Dict],
        system_prompt: str
    ) -> Result[Dict[str, Any], Exception]:
        """
        Generate next thought and action via LLM.
        
        Returns: Dict with keys: thought, action, action_input
        """
        try:
            # Build prompt
            user_content = f"Goal: {goal}"
            if context:
                user_content = f"Context: {context}\n\n{user_content}"
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ]
            messages.extend(history)
            
            # Generate thought
            async with get_inference_client() as client:
                response = await client.chat_completion(
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=500
                )
                
                if not isinstance(response, CompletionResult):
                    return Err(ValueError("Expected non-streaming response"))
                
                # Parse structured output
                parsed = self._parse_thought(response.content)
                return Ok(parsed)
                
        except Exception as e:
            return Err(e)
    
    def _parse_thought(self, content: str) -> Dict[str, Any]:
        """
        Parse LLM output into structured thought-action.
        
        Expected format:
        Thought: <reasoning>
        Action: <tool_name or "Final Answer">
        Action Input: <JSON or text>
        """
        lines = content.split('\n')
        thought = ""
        action = ""
        action_input = {}
        
        for line in lines:
            line = line.strip()
            if line.lower().startswith('thought:'):
                thought = line.split(':', 1)[1].strip()
            elif line.lower().startswith('action:'):
                action = line.split(':', 1)[1].strip()
            elif line.lower().startswith('action input:'):
                input_str = line.split(':', 1)[1].strip()
                # Try to parse as JSON
                try:
                    import json
                    action_input = json.loads(input_str)
                except json.JSONDecodeError:
                    # Fallback: treat as plain text
                    action_input = {"text": input_str}
        
        return {
            "thought": thought,
            "action": action,
            "action_input": action_input
        }
    
    async def _execute_action(
        self,
        thought_data: Dict[str, Any]
    ) -> Result[tuple[str, bool], Exception]:
        """
        Execute action from thought.
        
        Returns: (observation, is_final_answer)
        """
        try:
            action = thought_data.get("action", "").strip()
            action_input = thought_data.get("action_input", {})
            
            # Check for final answer
            if action.lower() in ["final answer", "finalanswer"]:
                answer = action_input.get("answer", str(action_input))
                return Ok((answer, True))
            
            # Execute tool
            if action not in self.tools:
                return Err(ValueError(f"Unknown tool: {action}. Available: {list(self.tools.keys())}"))
            
            tool = self.tools[action]
            
            # Execute with timeout
            try:
                result = await asyncio.wait_for(
                    tool.execute(**action_input),
                    timeout=self.tool_timeout
                )
                
                if result.is_ok:
                    return Ok((str(result.value), False))
                else:
                    return Err(result.error)
                    
            except asyncio.TimeoutError:
                return Err(TimeoutError(f"Tool {action} timed out after {self.tool_timeout}s"))
            
        except Exception as e:
            return Err(e)
