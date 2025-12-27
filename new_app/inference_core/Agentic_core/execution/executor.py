"""
Agent Executor (Orchestrator) - SOTA Update.

Adheres to:
- Robustness: Fully integrated Result type handling. 
- Observability: Logs every step.
"""
import asyncio
import logging
import traceback
from typing import List, Optional, Dict

from ..core.config import get_config
from ..core.result import Result, Ok, Err
from ..core.inference_wrapper import get_inference_client
from ..planning.planner import Planner
from ..planning.task_graph import TaskStatus
from ..memory.short_term import ShortTermMemory
from ..context.context_manager import ContextManager
from ..context.context_manager import ContextManager
from ..storage.session_db import SessionDB
from .tool_manager import ToolRegistry

logger = logging.getLogger(__name__)

class AgentExecutor:
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.planner = Planner()
        self.memory = ShortTermMemory()
        self.context_mgr = ContextManager()
        self.db = SessionDB()
        self.tool_registry = ToolRegistry() # SOTA Tool Management
        
        # Advanced Reasoning
        from ..reasoning.reflexion import ReflexionEngine
        from ..memory.episodic import EpisodicMemory
        self.reflexion = ReflexionEngine()
        self.episodic = EpisodicMemory()
        self.context_mgr = ContextManager()
        self.db = SessionDB()
        self.tool_registry = ToolRegistry() # SOTA Tool Management
        
        # Register default tools (Example)
        @self.tool_registry.register
        def get_current_time(timezone: str = "UTC") -> str:
            """Returns current time in ISO format."""
            from datetime import datetime
            return datetime.utcnow().isoformat()

    async def initialize(self):
        res = await self.db.initialize()
        if res.is_err:
            raise res.error
        
        # Create session if not exists
        await self.db.create_session(self.session_id)

    async def run(self, user_objective: str) -> str:
        """
        Main Execution Loop with Real-Time Continuous Scheduling.
        Solves the "Fast vs Heavy" task problem by scheduling dependents immediately upon completion.
        """
        # 1. State Persistence & Recall
        self.memory.add("user", user_objective)
        await self.db.add_message(self.session_id, "user", user_objective)
        
        # SOTA: Episodic Recall
        past_wisdom = await self.episodic.recall_similar(user_objective)
        if past_wisdom:
            self.memory.add("system", f"Relevant Past Experiences:\n" + "\n".join(past_wisdom))
            logger.info("Injected episodic memory into context.")

        # 2. Robust Planning
        plan_result = await self.planner.create_plan(user_objective)
        if plan_result.is_err:
            return f"Planning failure: {plan_result.error}"
            
        plan = plan_result.value
        logger.info(f"Plan created with {len(plan.nodes)} tasks")
        
        final_response_buffer = []

        # 3. Continuous Execution Loop
        # We need to track running tasks mapped back to their Task IDs
        # Map[Future, str (task_id)]
        running_futures: Dict[asyncio.Future, str] = {}
        
        # Concurrency Control (SOTA: Resource management)
        # Using a semaphore to prevent exploding heavy tasks if graph is wide
        concurrency_limit = 10 
        semaphore = asyncio.Semaphore(concurrency_limit)

        while True:
            # A. Schedule New Tasks
            # Get all currently ready tasks that aren't already running
            # Note: TaskGraph must handle state carefully. 
            # In our implementation, `get_ready_tasks` returns nodes with status READY.
            # Once we pick them up, we MUST mark them RUNNING immediately to avoid re-scheduling.
            ready_nodes = plan.get_ready_tasks()
            
            for node in ready_nodes:
                # Check if we have capacity
                if semaphore.locked():
                    break # Wait for something to finish
                    
                await semaphore.acquire()
                
                # Mark as running in graph so it's not returned by get_ready_tasks again
                node.status = TaskStatus.RUNNING
                
                # Create Task
                # distinct execution wrapper to handle semaphore release
                future = asyncio.create_task(self._execute_wrapper(node, semaphore))
                running_futures[future] = node.task_id
                logger.info(f"Started task: {node.description}")

            # B. Wait for Events (The "Heartbeat")
            if not running_futures:
                # If nothing running and nothing ready, we are done (or deadlocked)
                # Check if all completed
                if all(n.status in (TaskStatus.COMPLETED, TaskStatus.FAILED) for n in plan.nodes.values()):
                    break
                
                # If nodes are pending but nothing running/ready: Deadlock
                if any(n.status == TaskStatus.PENDING for n in plan.nodes.values()):
                    # Double check if we missed something due to semaphore full
                    if ready_nodes:
                        # We have ready nodes but semaphore was full, continue waiting
                        pass 
                    else:
                        logger.error("Deadlock detected in dependency graph.")
                        break
            
            # Wait for at least one task to complete (or new tasks to be ready if we blocked)
            if running_futures:
                # Wait for FIRST_COMPLETED
                done_set, _ = await asyncio.wait(
                    running_futures.keys(), 
                    return_when=asyncio.FIRST_COMPLETED
                )
                
                # C. Process Completed Tasks
                for done_future in done_set:
                    task_id = running_futures.pop(done_future)
                    
                    try:
                        # Result is robust Result type from _execute_wrapper
                        res: Result = done_future.result()
                        
                        task_node = plan.nodes[task_id]
                        
                        if res.is_ok:
                            # IMPORTANT: This update potentially unlocks NEW ready tasks
                            plan.mark_completed(task_id, res.value)
                            
                            # SOTA: Store in Working Memory Map for Immediate Context Resolution
                            self.memory.store_artifact(
                                key=task_id, # Using Task ID as key for now, could be semantic tag
                                value=res.value,
                                description=f"Result of {task_node.description}",
                                task_id=task_id
                            )
                            
                            msg = f"✓ Task '{task_node.description}' completed."
                            final_response_buffer.append(msg)
                            self.memory.add("system", f"Task {task_id} result: {res.value}")
                        else:
                            # Handle Failure
                            task_node.status = TaskStatus.FAILED
                            logger.error(f"Task {task_id} failed: {res.error}")
                            final_response_buffer.append(f"✗ Task {task_id} failed.")
                            
                    except Exception as e:
                        logger.critical(f"System Panic in task processing: {e}")
            else:
                # If we have ready tasks but no running futures (rare edge case of start), just loop
                pass

        # 4. Finalizing
        final_output = "\n".join(final_response_buffer)
        self.memory.add("assistant", final_output)
        await self.db.add_message(self.session_id, "assistant", final_output)
        
        return final_output

    async def _execute_wrapper(self, task, semaphore) -> Result[str, Exception]:
        """
        Wraps execution to ensure semaphore release even on crash.
        """
        try:
            return await self._execute_with_retry(task)
        finally:
            semaphore.release()

    async def _execute_with_retry(self, task) -> Result[str, Exception]:
        """
        Executes task with resilience policy.
        """
        from ..utils.resilence import retry_with_backoff
        
        async def _work():
            logger.info(f"Executing task: {task.description}")
            task.status = TaskStatus.RUNNING
            
            # Prepare contextual prompt
            msgs = self.memory.to_prompt_format()
            # Use SOTA Context Composition
            msgs = self.context_mgr.compose_context(msgs)
            
            # Specialized system prompt for task execution
            msgs.append({
                "role": "system", 
                "content": f"You are a specialized worker agent. Execute this task strictly: {task.description}. Return only the result."
            })
            
            # SOTA: Include Tools
            tools = self.tool_registry.get_schemas()
            
            async with get_inference_client() as client:
                result = await client.chat_completion(
                    messages=msgs,
                    tools=tools if tools else None,
                    tool_choice="auto" if tools else None 
                )
                
                # Zero-copy check for tool calls
                raw_msg = result.raw_response["choices"][0]["message"]
                
                if raw_msg.get("tool_calls"):
                    # Handle Tool Calls (SOTA: Recursive execution)
                    tool_calls = raw_msg["tool_calls"]
                    msgs.append(raw_msg) # Add assistant's tool call msg
                    
                    for tc in tool_calls:
                        import orjson
                        fn_name = tc["function"]["name"]
                        fn_args = orjson.loads(tc["function"]["arguments"])
                        
                        # Execute Tool
                        logger.info(f"Invoking tool: {fn_name}")
                        tool_res = await self.tool_registry.execute(fn_name, fn_args)
                        
                        # Add Result
                        msgs.append({
                            "role": "tool", 
                            "tool_call_id": tc["id"],
                            "content": tool_res.unwrap_or(str(tool_res.error))
                        })
                        
                    # Recurse: Get final answer with tool outputs
                    final_res = await client.chat_completion(messages=msgs)
                    return final_res.content
                    
                if hasattr(result, 'content'):
                    return result.content
                raise ValueError("Invalid response format")

        try:
            # SOTA: Reflexion Loop (Critique -> Fix)
            # We attempt the task. If generic retry fails or result is bad, we critique.
            
            # Initial Attempt
            val = await retry_with_backoff(_work, retries=2)
            
            # Critique
            critique_res = await self.reflexion.critique(task.description, val)
            if critique_res.is_ok and critique_res.value is None:
                # PASS
                return Ok(val)
            elif critique_res.is_ok:
                # FAIL with reason
                critique = critique_res.value
                logger.warning(f"Reflexion Critique: {critique}. Retrying with feedback.")
                
                # Update context with critique for the next attempt
                self.memory.add("system", f"Previous attempt for '{task.description}' failed critique: {critique}. Fix it.")
                
                # Retry once with feedback
                # In full SOTA we would have a loop here, limiting to 1 retry for latency now.
                val_retry = await retry_with_backoff(_work, retries=1)
                return Ok(val_retry)
                
            return Ok(val)
        except Exception as e:
            return Err(e)
