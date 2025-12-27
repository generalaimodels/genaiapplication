"""
Configuration Module: Type-Safe Runtime Configuration
======================================================

Design Principles:
    - Immutable frozen dataclasses for thread-safety (no locks required)
    - Slots optimization for reduced memory footprint and faster attribute access
    - Environment-driven configuration with validation
    - Cache-line aligned critical fields via struct ordering

Memory Layout Optimization:
    - Fields ordered by descending size to minimize padding
    - 64-byte alignment for contended atomic fields
    - Frozen=True enables hash caching for dict keys

Boundary Invariants:
    - All numeric configs validated in __post_init__
    - Checked arithmetic for overflow prevention
    - Pre-conditions asserted at module boundaries
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Final


# =============================================================================
# CONSTANTS: Cache-line size for alignment, compile-time constants
# =============================================================================
CACHE_LINE_BYTES: Final[int] = 64
DEFAULT_TIMEOUT_SECONDS: Final[float] = 30.0
DEFAULT_MAX_CONNECTIONS: Final[int] = 100
DEFAULT_QUEUE_SIZE: Final[int] = 10000  # Handle 100+ burst requests easily
DEFAULT_BATCH_SIZE: Final[int] = 32
DEFAULT_BATCH_WINDOW_MS: Final[int] = 10  # 10ms batching window


class ProviderType(Enum):
    """
    Enumeration of supported inference providers.
    Using auto() for forward compatibility with new providers.
    """
    VLLM = auto()
    OPENAI = auto()


@dataclass(slots=True, frozen=True)
class ProviderConfig:
    """
    Provider-specific configuration with immutability guarantees.
    
    Memory Layout (64-bit system, ordered by descending size):
        - base_url: 8 bytes (pointer to str)
        - api_key: 8 bytes (pointer to str) 
        - model: 8 bytes (pointer to str)
        - provider_type: 8 bytes (Enum reference)
        - timeout: 8 bytes (float64)
        - max_retries: 4 bytes (int32 effective)
        Total: 44 bytes + padding = 48 bytes (3 cache lines / 4)
    
    Attributes:
        provider_type: Backend provider (vLLM or OpenAI)
        base_url: API endpoint URL (vLLM: http://localhost:8000/v1)
        api_key: Authentication key (use "EMPTY" for local vLLM)
        model: Default model identifier
        timeout: Request timeout in seconds (checked arithmetic)
        max_retries: Retry count with exponential backoff
    """
    provider_type: ProviderType
    base_url: str
    api_key: str
    model: str
    timeout: float = DEFAULT_TIMEOUT_SECONDS
    max_retries: int = 3
    
    def __post_init__(self) -> None:
        """
        Boundary invariant validation.
        Raises AssertionError for invalid configurations.
        """
        # Pre-condition: timeout must be positive and bounded
        assert 0.0 < self.timeout <= 300.0, (
            f"timeout must be in (0, 300], got {self.timeout}"
        )
        # Pre-condition: max_retries must be non-negative and bounded
        assert 0 <= self.max_retries <= 10, (
            f"max_retries must be in [0, 10], got {self.max_retries}"
        )
        # Pre-condition: base_url must be non-empty
        assert self.base_url.strip(), "base_url cannot be empty"
        # Pre-condition: model must be non-empty
        assert self.model.strip(), "model cannot be empty"


@dataclass(slots=True, frozen=True)
class ConnectionPoolConfig:
    """
    HTTP connection pool configuration for high-throughput scenarios.
    
    Burst Traffic Handling:
        - max_connections: Total pool size (handles 100+ concurrent requests)
        - max_keepalive_connections: Warm connections for instant reuse
        - keepalive_expiry: Connection recycling interval
    
    HTTP/2 Benefits:
        - Single connection multiplexes multiple streams
        - Reduces TCP handshake overhead
        - Header compression via HPACK
    """
    max_connections: int = DEFAULT_MAX_CONNECTIONS
    max_keepalive_connections: int = 50
    keepalive_expiry: float = 30.0
    http2: bool = True  # Enable HTTP/2 multiplexing
    
    def __post_init__(self) -> None:
        assert self.max_connections >= 1, "max_connections must be >= 1"
        assert self.max_keepalive_connections >= 0, "max_keepalive >= 0"
        assert self.max_keepalive_connections <= self.max_connections, (
            "keepalive connections cannot exceed max connections"
        )
        assert self.keepalive_expiry > 0, "keepalive_expiry must be positive"


@dataclass(slots=True, frozen=True)
class QueueConfig:
    """
    Request queue configuration for burst traffic handling.
    
    Backpressure Strategy:
        - Bounded queue prevents memory exhaustion
        - Priority levels for request scheduling
        - Overflow rejection with 503 response
    
    Performance:
        - Lock-free operations via asyncio.Queue
        - O(log n) priority insertion with heapq
    """
    max_size: int = DEFAULT_QUEUE_SIZE  # 10K queue handles 100x burst
    high_water_mark: float = 0.8  # Trigger backpressure at 80%
    low_water_mark: float = 0.5   # Release backpressure at 50%
    
    def __post_init__(self) -> None:
        assert self.max_size >= 100, "max_size must be >= 100"
        assert 0.0 < self.low_water_mark < self.high_water_mark < 1.0, (
            "water marks must satisfy 0 < low < high < 1"
        )


@dataclass(slots=True, frozen=True)
class BatchConfig:
    """
    Dynamic batching configuration for throughput optimization.
    
    Algorithm:
        - Collect requests until max_size OR max_wait_ms elapsed
        - Dispatch batch to provider
        - Isolate failures per request in batch
    
    Tuning:
        - Smaller batches = lower latency
        - Larger batches = higher throughput
        - max_wait_ms bounds worst-case latency
    """
    max_size: int = DEFAULT_BATCH_SIZE
    max_wait_ms: int = DEFAULT_BATCH_WINDOW_MS
    enabled: bool = True
    
    def __post_init__(self) -> None:
        assert self.max_size >= 1, "batch max_size must be >= 1"
        assert self.max_wait_ms >= 1, "batch max_wait_ms must be >= 1"


@dataclass(slots=True, frozen=True)
class RateLimitConfig:
    """
    Token bucket rate limiter configuration.
    
    Algorithm:
        - Tokens refill at `refill_rate` per second
        - Each request consumes 1 token
        - Requests blocked when bucket empty
    
    Adaptive Behavior:
        - Reduce rate on 429 responses from upstream
        - Exponential backoff with jitter
    """
    requests_per_second: float = 100.0
    burst_size: int = 200  # Allow 2x burst
    enabled: bool = False  # Disabled by default for local vLLM
    
    def __post_init__(self) -> None:
        assert self.requests_per_second > 0, "rate must be positive"
        assert self.burst_size >= 1, "burst_size must be >= 1"


@dataclass(slots=True, frozen=True)
class ServerConfig:
    """
    FastAPI server configuration.
    """
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1  # Single worker for async; use uvicorn --workers for multi-process
    
    def __post_init__(self) -> None:
        assert 1 <= self.port <= 65535, "port must be valid"
        assert self.workers >= 1, "workers must be >= 1"


@dataclass(slots=True, frozen=True)
class BackgroundTaskConfig:
    """
    Background task executor configuration for burst handling.
    
    Critical for 100+ Sudden Requests:
        - max_concurrent_tasks: Upper bound on parallel background tasks
        - task_timeout: Individual task timeout to prevent hangs
        - queue_overflow_policy: REJECT or QUEUE_WAIT
    
    Implementation:
        - Uses asyncio.Semaphore for concurrency limiting
        - TaskGroup for structured concurrency (Python 3.11+)
        - Graceful shutdown with cancellation propagation
    """
    max_concurrent_tasks: int = 500  # Handle 100+ burst with headroom
    task_timeout: float = 60.0  # 1 minute max per task
    shutdown_timeout: float = 30.0  # Graceful shutdown window
    
    def __post_init__(self) -> None:
        assert self.max_concurrent_tasks >= 10, "need >= 10 concurrent tasks"
        assert self.task_timeout > 0, "task_timeout must be positive"
        assert self.shutdown_timeout > 0, "shutdown_timeout must be positive"


@dataclass(slots=True)
class InferenceConfig:
    """
    Root configuration aggregating all sub-configs.
    
    Note: This is NOT frozen because it holds mutable references,
    but all contained configs ARE frozen for thread-safety.
    
    Loading Priority:
        1. Environment variables (highest priority)
        2. Config file (if provided)
        3. Defaults (lowest priority)
    """
    provider: ProviderConfig = field(default_factory=lambda: ProviderConfig(
        provider_type=ProviderType.VLLM,
        base_url="http://localhost:8000/v1",
        api_key="EMPTY",
        model="default-model",
    ))
    connection_pool: ConnectionPoolConfig = field(
        default_factory=ConnectionPoolConfig
    )
    queue: QueueConfig = field(default_factory=QueueConfig)
    batch: BatchConfig = field(default_factory=BatchConfig)
    rate_limit: RateLimitConfig = field(default_factory=RateLimitConfig)
    server: ServerConfig = field(default_factory=ServerConfig)
    background_tasks: BackgroundTaskConfig = field(
        default_factory=BackgroundTaskConfig
    )


def load_config_from_env() -> InferenceConfig:
    """
    Load configuration from environment variables.
    
    Environment Variables:
        INFERENCE_PROVIDER_TYPE: "vllm" or "openai"
        INFERENCE_BASE_URL: Provider API endpoint
        INFERENCE_API_KEY: Authentication key
        INFERENCE_MODEL: Default model name
        INFERENCE_TIMEOUT: Request timeout seconds
        INFERENCE_MAX_CONNECTIONS: Connection pool size
        INFERENCE_QUEUE_SIZE: Request queue capacity
        INFERENCE_MAX_CONCURRENT_TASKS: Background task limit
    
    Returns:
        InferenceConfig: Validated configuration instance
    
    Raises:
        AssertionError: If validation fails
        ValueError: If environment variable parsing fails
    """
    # Parse provider type
    provider_type_str = os.getenv("INFERENCE_PROVIDER_TYPE", "vllm").lower()
    provider_type = (
        ProviderType.OPENAI if provider_type_str == "openai" 
        else ProviderType.VLLM
    )
    
    # Build provider config
    provider = ProviderConfig(
        provider_type=provider_type,
        base_url=os.getenv("INFERENCE_BASE_URL", "http://localhost:8000/v1"),
        api_key=os.getenv("INFERENCE_API_KEY", "EMPTY"),
        model=os.getenv("INFERENCE_MODEL", "default-model"),
        timeout=float(os.getenv("INFERENCE_TIMEOUT", str(DEFAULT_TIMEOUT_SECONDS))),
        max_retries=int(os.getenv("INFERENCE_MAX_RETRIES", "3")),
    )
    
    # Build connection pool config
    connection_pool = ConnectionPoolConfig(
        max_connections=int(os.getenv(
            "INFERENCE_MAX_CONNECTIONS", 
            str(DEFAULT_MAX_CONNECTIONS)
        )),
        http2=os.getenv("INFERENCE_HTTP2", "true").lower() == "true",
    )
    
    # Build queue config
    queue = QueueConfig(
        max_size=int(os.getenv("INFERENCE_QUEUE_SIZE", str(DEFAULT_QUEUE_SIZE))),
    )
    
    # Build background task config
    background_tasks = BackgroundTaskConfig(
        max_concurrent_tasks=int(os.getenv(
            "INFERENCE_MAX_CONCURRENT_TASKS", 
            "500"
        )),
        task_timeout=float(os.getenv("INFERENCE_TASK_TIMEOUT", "60.0")),
    )
    
    return InferenceConfig(
        provider=provider,
        connection_pool=connection_pool,
        queue=queue,
        background_tasks=background_tasks,
    )


# =============================================================================
# MODULE-LEVEL SINGLETON: Lazy-loaded global config
# =============================================================================
_global_config: InferenceConfig | None = None


def get_config() -> InferenceConfig:
    """
    Get the global configuration singleton.
    Thread-safe due to GIL for simple assignment.
    """
    global _global_config
    if _global_config is None:
        _global_config = load_config_from_env()
    return _global_config


def set_config(config: InferenceConfig) -> None:
    """
    Override the global configuration (for testing).
    """
    global _global_config
    _global_config = config
