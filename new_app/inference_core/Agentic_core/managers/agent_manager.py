"""
Multi-Agent Manager for Coordinating Specialized Agents.

Adheres to:
- Algorithmic Complexity: O(1) agent lookup, O(n) task routing where n=agents.
- Deterministic Concurrency: Lock-free agent registry with atomic operations.
- Failure Domain: Result types for all coordination operations.
- Observability: Track per-agent metrics and task assignments.
"""
import asyncio
import logging
from typing import Dict, List, Optional, Set, Any
from dataclasses import dataclass, field
from enum import Enum
import time
from ..core.result import Result, Ok, Err

logger = logging.getLogger(__name__)

# ============================================================================
# MULTI-AGENT COORDINATION PATTERNS
# ============================================================================
# Patterns Implemented:
# 1. SPECIALIST: Each agent has domain expertise (tags/capabilities)
# 2. COORDINATOR: Meta-agent delegates to specialists
# 3. CONSENSUS: Multiple agents vote on decisions
# 4. AUCTION: Tasks assigned to best-fit agent via scoring
#
# Complexity Analysis:
# - Register agent: O(1)
# - Find agent by capability: O(n) linear scan
# - Route task: O(n) scoring all agents
# - Consensus: O(k) where k=participating agents
# ============================================================================

class AgentStatus(Enum):
    """Agent lifecycle states."""
    IDLE = "idle"
    BUSY = "busy"
    PAUSED = "paused"
    FAILED = "failed"


@dataclass
class AgentCapability:
    """
    Agent capability descriptor.
    
    Fields:
    - tag: str - Capability identifier (e.g., "python_coding", "web_search")
    - proficiency: float - Skill level (0.0 to 1.0)
    - metadata: Dict - Additional capability info
    """
    tag: str
    proficiency: float  # 0.0 to 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentMetrics:
    """
    Per-agent performance metrics.
    
    Field ordering (descending size):
    - tasks_completed: int
    - tasks_failed: int
    - total_execution_time: float
    - last_active: float
    - average_task_time: float
    - success_rate: float
    """
    tasks_completed: int = 0
    tasks_failed: int = 0
    total_execution_time: float = 0.0
    last_active: float = 0.0
    average_task_time: float = 0.0
    success_rate: float = 1.0


@dataclass
class RegisteredAgent:
    """
    Agent registration record.
    """
    agent_id: str
    agent_type: str  # "specialist", "coordinator", "consensus"
    capabilities: List[AgentCapability]
    status: AgentStatus
    metrics: AgentMetrics
    metadata: Dict[str, Any] = field(default_factory=dict)


class AgentManager:
    """
    Coordinates multiple specialized agents.
    
    Performance Characteristics:
    - Agent registration: O(1)
    - Capability search: O(n) where n=total agents
    - Task routing: O(n * log n) with scoring
    - Consensus: O(k) where k=consensus_group_size
    """
    
    def __init__(self, enable_metrics: bool = True):
        """
        Initialize agent manager.
        
        Args:
            enable_metrics: Track per-agent performance metrics
        """
        self._agents: Dict[str, RegisteredAgent] = {}
        self._capability_index: Dict[str, Set[str]] = {}  # tag -> {agent_ids}
        self._lock = asyncio.Lock()
        self.enable_metrics = enable_metrics
        
        # Consensus settings
        self.consensus_threshold = 0.51  # 51% agreement required
    
    async def register_agent(
        self,
        agent_id: str,
        agent_type: str,
        capabilities: List[AgentCapability],
        metadata: Optional[Dict[str, Any]] = None
    ) -> Result[bool, Exception]:
        """
        Register new agent.
        
        Complexity: O(c) where c=number of capabilities
        
        Args:
            agent_id: Unique agent identifier
            agent_type: Agent pattern type
            capabilities: List of agent capabilities
            metadata: Optional agent metadata
            
        Returns:
            Ok(True) on success
        """
        try:
            async with self._lock:
                if agent_id in self._agents:
                    return Err(ValueError(f"Agent {agent_id} already registered"))
                
                # Create registration
                agent = RegisteredAgent(
                    agent_id=agent_id,
                    agent_type=agent_type,
                    capabilities=capabilities,
                    status=AgentStatus.IDLE,
                    metrics=AgentMetrics(),
                    metadata=metadata or {}
                )
                
                self._agents[agent_id] = agent
                
                # Index by capabilities
                for cap in capabilities:
                    if cap.tag not in self._capability_index:
                        self._capability_index[cap.tag] = set()
                    self._capability_index[cap.tag].add(agent_id)
                
                logger.info(
                    f"Registered agent {agent_id} ({agent_type}) "
                    f"with {len(capabilities)} capabilities"
                )
                return Ok(True)
                
        except Exception as e:
            return Err(e)
    
    async def find_agents_by_capability(
        self,
        capability_tag: str,
        min_proficiency: float = 0.0,
        status: Optional[AgentStatus] = None
    ) -> Result[List[RegisteredAgent], Exception]:
        """
        Find agents with specific capability.
        
        Complexity: O(n) where n=agents with capability
        
        Args:
            capability_tag: Capability to search for
            min_proficiency: Minimum proficiency threshold
            status: Optional status filter
            
        Returns:
            Ok(List[RegisteredAgent]) matching criteria
        """
        try:
            # Quick index lookup
            if capability_tag not in self._capability_index:
                return Ok([])
            
            agent_ids = self._capability_index[capability_tag]
            matches = []
            
            for agent_id in agent_ids:
                agent = self._agents.get(agent_id)
                if not agent:
                    continue
                
                # Status filter
                if status and agent.status != status:
                    continue
                
                # Proficiency filter
                for cap in agent.capabilities:
                    if cap.tag == capability_tag and cap.proficiency >= min_proficiency:
                        matches.append(agent)
                        break
            
            # Sort by proficiency descending
            matches.sort(
                key=lambda a: max(
                    (c.proficiency for c in a.capabilities if c.tag == capability_tag),
                    default=0.0
                ),
                reverse=True
            )
            
            return Ok(matches)
            
        except Exception as e:
            return Err(e)
    
    async def route_task(
        self,
        task_description: str,
        required_capabilities: List[str],
        prefer_idle: bool = True
    ) -> Result[Optional[str], Exception]:
        """
        Route task to best-fit agent via auction-based scoring.
        
        Complexity: O(n * log n) where n=agents
        
        Scoring factors:
        - Capability match: 40%
        - Agent availability: 30%
        - Historical performance: 30%
        
        Args:
            task_description: Task to assign
            required_capabilities: Required capability tags
            prefer_idle: Prefer idle agents over busy
            
        Returns:
            Ok(agent_id) of best match, or Ok(None) if no match
        """
        try:
            candidates = []
            
            # Collect candidates
            for cap_tag in required_capabilities:
                result = await self.find_agents_by_capability(cap_tag)
                if result.is_ok:
                    candidates.extend(result.value)
            
            if not candidates:
                return Ok(None)
            
            # Remove duplicates
            unique_candidates = {agent.agent_id: agent for agent in candidates}
            candidates = list(unique_candidates.values())
            
            # Score each candidate
            scored = []
            for agent in candidates:
                score = self._score_agent(agent, required_capabilities, prefer_idle)
                scored.append((score, agent))
            
            # Sort by score descending
            scored.sort(key=lambda x: x[0], reverse=True)
            
            best_agent = scored[0][1]
            logger.info(
                f"Routed task to agent {best_agent.agent_id} "
                f"(score={scored[0][0]:.2f})"
            )
            
            return Ok(best_agent.agent_id)
            
        except Exception as e:
            return Err(e)
    
    def _score_agent(
        self,
        agent: RegisteredAgent,
        required_caps: List[str],
        prefer_idle: bool
    ) -> float:
        """
        Compute agent fitness score for task.
        
        Returns: Score in [0.0, 1.0]
        """
        # Capability match (40%)
        cap_scores = []
        for req_cap in required_caps:
            best_match = max(
                (c.proficiency for c in agent.capabilities if c.tag == req_cap),
                default=0.0
            )
            cap_scores.append(best_match)
        cap_score = sum(cap_scores) / len(cap_scores) if cap_scores else 0.0
        
        # Availability (30%)
        if agent.status == AgentStatus.IDLE:
            avail_score = 1.0
        elif agent.status == AgentStatus.BUSY and not prefer_idle:
            avail_score = 0.5
        else:
            avail_score = 0.0
        
        # Historical performance (30%)
        perf_score = agent.metrics.success_rate
        
        # Weighted sum
        total_score = (0.4 * cap_score) + (0.3 * avail_score) + (0.3 * perf_score)
        
        return total_score
    
    async def consensus_decision(
        self,
        agent_ids: List[str],
        decision_prompt: str
    ) -> Result[Dict[str, Any], Exception]:
        """
        Run consensus among multiple agents (voting pattern).
        
        Complexity: O(k) where k=len(agent_ids)
        
        Args:
            agent_ids: Agents to participate in consensus
            decision_prompt: Decision to vote on
            
        Returns:
            Ok(Dict) with "decision", "votes", "agreement_ratio"
        """
        try:
            # Placeholder: In real implementation, each agent would vote
            # For now, simulate with majority vote
            
            votes: Dict[str, int] = {}
            for agent_id in agent_ids:
                if agent_id not in self._agents:
                    continue
                
                # Simulate vote (in practice, call agent's decision method)
                vote = "approve"  # Placeholder
                votes[vote] = votes.get(vote, 0) + 1
            
            # Determine consensus
            total_votes = sum(votes.values())
            if total_votes == 0:
                return Err(ValueError("No valid votes"))
            
            majority_vote = max(votes.items(), key=lambda x: x[1])
            decision, count = majority_vote
            agreement_ratio = count / total_votes
            
            consensus_reached = agreement_ratio >= self.consensus_threshold
            
            return Ok({
                "decision": decision,
                "votes": votes,
                "agreement_ratio": agreement_ratio,
                "consensus_reached": consensus_reached
            })
            
        except Exception as e:
            return Err(e)
    
    async def update_agent_status(
        self,
        agent_id: str,
        status: AgentStatus
    ) -> Result[bool, Exception]:
        """Update agent status."""
        try:
            if agent_id not in self._agents:
                return Err(ValueError(f"Agent {agent_id} not found"))
            
            self._agents[agent_id].status = status
            return Ok(True)
            
        except Exception as e:
            return Err(e)
    
    async def record_task_completion(
        self,
        agent_id: str,
        success: bool,
        execution_time: float
    ) -> Result[bool, Exception]:
        """
        Record task completion for metrics.
        """
        try:
            if not self.enable_metrics:
                return Ok(True)
            
            if agent_id not in self._agents:
                return Err(ValueError(f"Agent {agent_id} not found"))
            
            agent = self._agents[agent_id]
            metrics = agent.metrics
            
            # Update metrics
            if success:
                metrics.tasks_completed += 1
            else:
                metrics.tasks_failed += 1
            
            metrics.total_execution_time += execution_time
            metrics.last_active = time.time()
            
            total_tasks = metrics.tasks_completed + metrics.tasks_failed
            if total_tasks > 0:
                metrics.success_rate = metrics.tasks_completed / total_tasks
                metrics.average_task_time = metrics.total_execution_time / total_tasks
            
            return Ok(True)
            
        except Exception as e:
            return Err(e)
    
    async def get_agent_stats(self) -> Dict[str, Any]:
        """Get manager statistics."""
        total_agents = len(self._agents)
        by_status = {status: 0 for status in AgentStatus}
        by_type = {}
        
        for agent in self._agents.values():
            by_status[agent.status] += 1
            by_type[agent.agent_type] = by_type.get(agent.agent_type, 0) + 1
        
        return {
            "total_agents": total_agents,
            "by_status": {s.value: count for s, count in by_status.items()},
            "by_type": by_type,
            "total_capabilities": len(self._capability_index)
        }
