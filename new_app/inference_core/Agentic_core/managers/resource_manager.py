"""
Resource Manager for GPU/Memory/Concurrency Allocation.

Adheres to:
- Hardware Architecture: GPU affinity tracking, memory high-water marks.
- Deterministic Concurrency: Atomic resource allocation with quotas.
- Failure Domain: Result types with allocation failure handling.
- Observability: Real-time resource utilization metrics.
"""
import asyncio
import psutil
import logging
from typing import Dict, List, Optional, Set, Any
from dataclasses import dataclass
from enum import Enum
import os
from ..core.result import Result, Ok, Err
from ..core.config import get_config

logger = logging.getLogger(__name__)
CONFIG = get_config()

# ============================================================================
# RESOURCE MANAGEMENT STRATEGY
# ============================================================================
# Resources Managed:
# 1. GPU Allocation: Track which models on which GPUs
# 2. Memory Quotas: Prevent OOM via reservation system
# 3. Concurrency Limits: Dynamic scaling based on load
# 4. Compute Budget: Track inference token consumption
#
# Allocation Algorithm:
# - Best-fit: Allocate to least-loaded resource
# - Quota enforcement: Hard limits with graceful degradation
# - Auto-scaling: Adjust concurrency based on utilization
# ============================================================================

class ResourceType(Enum):
    """Types of managed resources."""
    GPU = "gpu"
    MEMORY = "memory"
    CONCURRENCY = "concurrency"
    COMPUTE = "compute"


@dataclass
class GPUInfo:
    """
    GPU resource descriptor.
    
    Fields:
    - gpu_id: int - GPU device ID
    - total_memory_mb: int - Total VRAM
    - allocated_memory_mb: int - Currently allocated
    - assigned_models: Set[str] - Models loaded on this GPU
    - utilization: float - Current utilization (0.0 to 1.0)
    """
    gpu_id: int
    total_memory_mb: int
    allocated_memory_mb: int = 0
    assigned_models: Set[str] = None
    utilization: float = 0.0
    
    def __post_init__(self):
        if self.assigned_models is None:
            self.assigned_models = set()


@dataclass
class MemoryQuota:
    """
    Memory allocation quota.
    """
    total_mb: int
    allocated_mb: int = 0
    reserved_mb: int = 0
    high_water_mark_mb: int = 0


@dataclass
class ConcurrencyQuota:
    """
    Concurrency limit management.
    """
    max_concurrent: int
    current_active: int = 0
    peak_active: int = 0
    average_utilization: float = 0.0


class ResourceManager:
    """
    Manages GPU, memory, and concurrency resources.
    
    Performance Characteristics:
    - Allocate GPU: O(g) where g=num_gpus (best-fit search)
    - Memory check: O(1) quota validation
    - Concurrency control: O(1) atomic counter
    - Metrics collection: O(1) snapshot
    """
    
    def __init__(
        self,
        enable_gpu_tracking: bool = True,
        memory_reserve_mb: int = 1024,
        auto_scale_concurrency: bool = True
    ):
        """
        Initialize resource manager.
        
        Args:
            enable_gpu_tracking: Track GPU allocation (requires CUDA)
            memory_reserve_mb: Reserve memory for system
            auto_scale_concurrency: Dynamically adjust concurrency limits
        """
        self.enable_gpu_tracking = enable_gpu_tracking
        self.memory_reserve_mb = memory_reserve_mb
        self.auto_scale_concurrency = auto_scale_concurrency
        
        # GPU tracking
        self._gpus: Dict[int, GPUInfo] = {}
        self._model_to_gpu: Dict[str, int] = {}  # model_name -> gpu_id
        
        # Memory management
        self._memory_quota = MemoryQuota(
            total_mb=int(psutil.virtual_memory().total / (1024 * 1024))
        )
        
        # Concurrency management
        self._concurrency_quota = ConcurrencyQuota(
            max_concurrent=CONFIG.max_concurrent_agents
        )
        self._semaphore = asyncio.Semaphore(CONFIG.max_concurrent_agents)
        
        # Locks
        self._gpu_lock = asyncio.Lock()
        self._memory_lock = asyncio.Lock()
        
        logger.info(
            f"Resource Manager initialized: "
            f"Memory={self._memory_quota.total_mb}MB, "
            f"Concurrency={self._concurrency_quota.max_concurrent}"
        )
    
    async def initialize_gpus(self) -> Result[bool, Exception]:
        """
        Detect and initialize GPU resources.
        
        Requires: nvidia-ml-py3 or similar GPU library
        """
        if not self.enable_gpu_tracking:
            logger.info("GPU tracking disabled")
            return Ok(True)
        
        try:
            # Try to import NVIDIA management library
            try:
                import pynvml
                pynvml.nvmlInit()
                gpu_count = pynvml.nvmlDeviceGetCount()
                
                async with self._gpu_lock:
                    for gpu_id in range(gpu_count):
                        handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
                        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                        
                        self._gpus[gpu_id] = GPUInfo(
                            gpu_id=gpu_id,
                            total_memory_mb=int(mem_info.total / (1024 * 1024))
                        )
                
                logger.info(f"Initialized {gpu_count} GPUs")
                return Ok(True)
                
            except ImportError:
                logger.warning("pynvml not installed, GPU tracking unavailable")
                return Ok(True)
                
        except Exception as e:
            logger.warning(f"GPU tracking unavailable: {e}")
            return Ok(True)  # Non-critical failure
    
    async def allocate_gpu(
        self,
        model_name: str,
        required_memory_mb: int,
        preferred_gpu: Optional[int] = None
    ) -> Result[int, Exception]:
        """
        Allocate GPU for model.
        
        Complexity: O(g) where g=num_gpus
        
        Strategy: Best-fit (least-loaded GPU with sufficient memory)
        
        Args:
            model_name: Model identifier
            required_memory_mb: Required VRAM
            preferred_gpu: Optional GPU preference
            
        Returns:
            Ok(gpu_id) on success
        """
        try:
            if not self.enable_gpu_tracking or not self._gpus:
                return Ok(0)  # Default GPU
            
            async with self._gpu_lock:
                # Check if model already allocated
                if model_name in self._model_to_gpu:
                    return Ok(self._model_to_gpu[model_name])
                
                # Preferred GPU
                if preferred_gpu is not None and preferred_gpu in self._gpus:
                    gpu = self._gpus[preferred_gpu]
                    available = gpu.total_memory_mb - gpu.allocated_memory_mb
                    if available >= required_memory_mb:
                        gpu.allocated_memory_mb += required_memory_mb
                        gpu.assigned_models.add(model_name)
                        self._model_to_gpu[model_name] = preferred_gpu
                        logger.info(f"Allocated {model_name} to GPU {preferred_gpu}")
                        return Ok(preferred_gpu)
                
                # Best-fit search
                candidates = []
                for gpu_id, gpu in self._gpus.items():
                    available = gpu.total_memory_mb - gpu.allocated_memory_mb
                    if available >= required_memory_mb:
                        candidates.append((available, gpu_id, gpu))
                
                if not candidates:
                    return Err(ValueError("No GPU with sufficient memory available"))
                
                # Sort by available memory (ascending = best fit)
                candidates.sort(key=lambda x: x[0])
                _, best_gpu_id, best_gpu = candidates[0]
                
                # Allocate
                best_gpu.allocated_memory_mb += required_memory_mb
                best_gpu.assigned_models.add(model_name)
                self._model_to_gpu[model_name] = best_gpu_id
                
                logger.info(f"Allocated {model_name} to GPU {best_gpu_id}")
                return Ok(best_gpu_id)
                
        except Exception as e:
            return Err(e)
    
    async def reserve_memory(
        self,
        amount_mb: int,
        purpose: str = "general"
    ) -> Result[bool, Exception]:
        """
        Reserve system memory.
        
        Complexity: O(1)
        
        Prevents OOM by enforcing quotas.
        """
        try:
            async with self._memory_lock:
                available = (
                    self._memory_quota.total_mb 
                    - self._memory_quota.allocated_mb 
                    - self.memory_reserve_mb
                )
                
                if amount_mb > available:
                    return Err(ValueError(
                        f"Insufficient memory: requested={amount_mb}MB, "
                        f"available={available}MB"
                    ))
                
                self._memory_quota.allocated_mb += amount_mb
                self._memory_quota.high_water_mark_mb = max(
                    self._memory_quota.high_water_mark_mb,
                    self._memory_quota.allocated_mb
                )
                
                logger.debug(f"Reserved {amount_mb}MB for {purpose}")
                return Ok(True)
                
        except Exception as e:
            return Err(e)
    
    async def release_memory(self, amount_mb: int) -> Result[bool, Exception]:
        """Release reserved memory."""
        try:
            async with self._memory_lock:
                self._memory_quota.allocated_mb -= amount_mb
                self._memory_quota.allocated_mb = max(0, self._memory_quota.allocated_mb)
                return Ok(True)
                
        except Exception as e:
            return Err(e)
    
    async def acquire_concurrency_slot(
        self,
        timeout: Optional[float] = None
    ) -> Result[bool, Exception]:
        """
        Acquire concurrency slot (semaphore-based).
        
        Complexity: O(1)
        
        Use with context manager for RAII:
        ```
        result = await resource_mgr.acquire_concurrency_slot()
        if result.is_ok:
            try:
                # Do work
            finally:
                await resource_mgr.release_concurrency_slot()
        ```
        
        Args:
            timeout: Optional timeout in seconds
            
        Returns:
            Ok(True) on success, Err on timeout
        """
        try:
            if timeout:
                acquired = await asyncio.wait_for(
                    self._semaphore.acquire(),
                    timeout=timeout
                )
            else:
                await self._semaphore.acquire()
                acquired = True
            
            if acquired:
                self._concurrency_quota.current_active += 1
                self._concurrency_quota.peak_active = max(
                    self._concurrency_quota.peak_active,
                    self._concurrency_quota.current_active
                )
                return Ok(True)
            else:
                return Err(TimeoutError("Concurrency slot acquisition timed out"))
                
        except asyncio.TimeoutError:
            return Err(TimeoutError("Concurrency slot acquisition timed out"))
        except Exception as e:
            return Err(e)
    
    async def release_concurrency_slot(self) -> None:
        """Release concurrency slot."""
        self._concurrency_quota.current_active -= 1
        self._concurrency_quota.current_active = max(0, self._concurrency_quota.current_active)
        self._semaphore.release()
    
    async def auto_scale_concurrency(self) -> None:
        """
        Automatically adjust concurrency limits based on utilization.
        
        Strategy:
        - If sustained >90% utilization, reduce limit (prevent overload)
        - If sustained <50% utilization, increase limit (better throughput)
        """
        if not self.auto_scale_concurrency:
            return
        
        utilization = self._concurrency_quota.current_active / self._concurrency_quota.max_concurrent
        
        # Calculate rolling average (simplified)
        self._concurrency_quota.average_utilization = (
            0.8 * self._concurrency_quota.average_utilization + 
            0.2 * utilization
        )
        
        avg_util = self._concurrency_quota.average_utilization
        
        # Scale down if overloaded
        if avg_util > 0.9:
            new_limit = max(4, int(self._concurrency_quota.max_concurrent * 0.8))
            if new_limit != self._concurrency_quota.max_concurrent:
                logger.warning(
                    f"Scaling down concurrency: {self._concurrency_quota.max_concurrent} -> {new_limit}"
                )
                self._concurrency_quota.max_concurrent = new_limit
                # Note: Semaphore limit cannot be changed dynamically in asyncio
                # Would need custom semaphore implementation
        
        # Scale up if underutilized
        elif avg_util < 0.5:
            new_limit = min(128, int(self._concurrency_quota.max_concurrent * 1.2))
            if new_limit != self._concurrency_quota.max_concurrent:
                logger.info(
                    f"Scaling up concurrency: {self._concurrency_quota.max_concurrent} -> {new_limit}"
                )
                self._concurrency_quota.max_concurrent = new_limit
    
    async def get_resource_stats(self) -> Dict[str, Any]:
        """Get comprehensive resource statistics."""
        # Memory stats
        current_memory = psutil.virtual_memory()
        memory_stats = {
            "total_mb": self._memory_quota.total_mb,
            "allocated_mb": self._memory_quota.allocated_mb,
            "available_mb": self._memory_quota.total_mb - self._memory_quota.allocated_mb,
            "system_used_mb": int(current_memory.used / (1024 * 1024)),
            "system_percent": current_memory.percent,
            "high_water_mark_mb": self._memory_quota.high_water_mark_mb
        }
        
        # Concurrency stats
        concurrency_stats = {
            "max_concurrent": self._concurrency_quota.max_concurrent,
            "current_active": self._concurrency_quota.current_active,
            "peak_active": self._concurrency_quota.peak_active,
            "utilization": (
                self._concurrency_quota.current_active / self._concurrency_quota.max_concurrent
                if self._concurrency_quota.max_concurrent > 0 else 0.0
            )
        }
        
        # GPU stats
        gpu_stats = {}
        if self.enable_gpu_tracking and self._gpus:
            for gpu_id, gpu in self._gpus.items():
                gpu_stats[f"gpu_{gpu_id}"] = {
                    "total_memory_mb": gpu.total_memory_mb,
                    "allocated_memory_mb": gpu.allocated_memory_mb,
                    "available_memory_mb": gpu.total_memory_mb - gpu.allocated_memory_mb,
                    "assigned_models": list(gpu.assigned_models),
                    "utilization": gpu.utilization
                }
        
        return {
            "memory": memory_stats,
            "concurrency": concurrency_stats,
            "gpus": gpu_stats
        }
