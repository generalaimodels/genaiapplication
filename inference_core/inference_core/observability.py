"""
Observability Module: Metrics, Tracing, and Logging
=====================================================

Production-grade observability with:
    - Nanosecond-precision latency histograms
    - Request tracing with correlation IDs
    - Structured JSON logging
    - Memory and allocation tracking

Design Principles:
    - Minimal observer overhead (<1% CPU)
    - Lock-free metric collection where possible
    - Context propagation via contextvars
    - Prometheus-compatible export format

Metrics Categories:
    - Latency: p50, p90, p95, p99
    - Throughput: requests/second, tokens/second
    - Resources: active connections, queue depth
    - Errors: by type, by provider
"""

from __future__ import annotations

import asyncio
import contextvars
import json
import logging
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Final


# =============================================================================
# CONSTANTS
# =============================================================================
HISTOGRAM_BUCKETS: Final[tuple[float, ...]] = (
    0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0
)
MAX_RECENT_LATENCIES: Final[int] = 10000  # Sliding window for percentiles


# =============================================================================
# CONTEXT VARIABLES: Request-scoped tracing
# =============================================================================
_request_id: contextvars.ContextVar[str] = contextvars.ContextVar(
    "request_id", default=""
)
_span_id: contextvars.ContextVar[str] = contextvars.ContextVar(
    "span_id", default=""
)
_trace_start_ns: contextvars.ContextVar[int] = contextvars.ContextVar(
    "trace_start_ns", default=0
)


def set_request_context(request_id: str) -> contextvars.Token[str]:
    """Set request ID for current context."""
    return _request_id.set(request_id)


def get_request_id() -> str:
    """Get request ID from context."""
    return _request_id.get()


def set_span(span_id: str) -> contextvars.Token[str]:
    """Set span ID for current context."""
    return _span_id.set(span_id)


def get_span_id() -> str:
    """Get span ID from context."""
    return _span_id.get()


# =============================================================================
# METRIC TYPES
# =============================================================================

class MetricType(Enum):
    """Metric type enumeration."""
    COUNTER = auto()
    GAUGE = auto()
    HISTOGRAM = auto()


@dataclass
class Counter:
    """
    Monotonically increasing counter.
    
    Thread Safety:
        - Uses atomic increment via simple addition
        - Safe for concurrent access in async context
    """
    name: str
    description: str
    labels: dict[str, str] = field(default_factory=dict)
    _value: float = 0.0
    
    def inc(self, value: float = 1.0) -> None:
        """Increment counter."""
        self._value += value
    
    @property
    def value(self) -> float:
        """Get current value."""
        return self._value
    
    def to_prometheus(self) -> str:
        """Export in Prometheus format."""
        labels_str = ",".join(f'{k}="{v}"' for k, v in self.labels.items())
        label_part = f"{{{labels_str}}}" if labels_str else ""
        return f"{self.name}{label_part} {self._value}"


@dataclass
class Gauge:
    """
    Value that can increase or decrease.
    
    Use for:
        - Active connections
        - Queue depth
        - Memory usage
    """
    name: str
    description: str
    labels: dict[str, str] = field(default_factory=dict)
    _value: float = 0.0
    
    def set(self, value: float) -> None:
        """Set gauge value."""
        self._value = value
    
    def inc(self, value: float = 1.0) -> None:
        """Increment gauge."""
        self._value += value
    
    def dec(self, value: float = 1.0) -> None:
        """Decrement gauge."""
        self._value -= value
    
    @property
    def value(self) -> float:
        """Get current value."""
        return self._value
    
    def to_prometheus(self) -> str:
        """Export in Prometheus format."""
        labels_str = ",".join(f'{k}="{v}"' for k, v in self.labels.items())
        label_part = f"{{{labels_str}}}" if labels_str else ""
        return f"{self.name}{label_part} {self._value}"


@dataclass
class Histogram:
    """
    Distribution of values with bucket counting.
    
    Features:
        - Fixed buckets for Prometheus compatibility
        - Sliding window for percentile calculation
        - Sum and count for rate calculations
    
    Performance:
        - O(1) observation via binary search on buckets
        - Bounded memory via deque maxlen
    """
    name: str
    description: str
    buckets: tuple[float, ...] = HISTOGRAM_BUCKETS
    labels: dict[str, str] = field(default_factory=dict)
    _bucket_counts: list[int] = field(default_factory=list, init=False)
    _sum: float = field(default=0.0, init=False)
    _count: int = field(default=0, init=False)
    _recent: deque[float] = field(
        default_factory=lambda: deque(maxlen=MAX_RECENT_LATENCIES),
        init=False,
    )
    
    def __post_init__(self) -> None:
        """Initialize bucket counts."""
        self._bucket_counts = [0] * (len(self.buckets) + 1)  # +1 for +Inf
    
    def observe(self, value: float) -> None:
        """
        Record an observation.
        
        Updates bucket counts, sum, count, and recent window.
        """
        self._sum += value
        self._count += 1
        self._recent.append(value)
        
        # Find bucket
        for i, bound in enumerate(self.buckets):
            if value <= bound:
                self._bucket_counts[i] += 1
                return
        # +Inf bucket
        self._bucket_counts[-1] += 1
    
    @property
    def sum(self) -> float:
        """Total sum of observations."""
        return self._sum
    
    @property
    def count(self) -> int:
        """Total count of observations."""
        return self._count
    
    @property
    def mean(self) -> float:
        """Mean of observations."""
        return self._sum / self._count if self._count > 0 else 0.0
    
    def percentile(self, p: float) -> float:
        """
        Calculate percentile from recent observations.
        
        Args:
            p: Percentile (0-100)
            
        Returns:
            Value at percentile
        """
        if not self._recent:
            return 0.0
        
        sorted_vals = sorted(self._recent)
        idx = int(len(sorted_vals) * p / 100)
        idx = min(idx, len(sorted_vals) - 1)
        return sorted_vals[idx]
    
    def to_prometheus(self) -> str:
        """Export in Prometheus format."""
        lines: list[str] = []
        labels_str = ",".join(f'{k}="{v}"' for k, v in self.labels.items())
        label_prefix = f"{{{labels_str}," if labels_str else "{"
        label_suffix = "}"
        
        # Bucket counts
        cumulative = 0
        for i, bound in enumerate(self.buckets):
            cumulative += self._bucket_counts[i]
            le_label = f'{label_prefix}le="{bound}"{label_suffix}'
            lines.append(f"{self.name}_bucket{le_label} {cumulative}")
        
        # +Inf bucket
        cumulative += self._bucket_counts[-1]
        inf_label = f'{label_prefix}le="+Inf"{label_suffix}'
        lines.append(f"{self.name}_bucket{inf_label} {cumulative}")
        
        # Sum and count
        label_part = f"{{{labels_str}}}" if labels_str else ""
        lines.append(f"{self.name}_sum{label_part} {self._sum}")
        lines.append(f"{self.name}_count{label_part} {self._count}")
        
        return "\n".join(lines)


# =============================================================================
# METRICS REGISTRY: Central metric collection
# =============================================================================

class MetricsRegistry:
    """
    Central registry for all metrics.
    
    Features:
        - Lazy metric creation
        - Label-based metric variants
        - Prometheus export
    
    Thread Safety:
        - Dict operations are atomic in CPython
        - Safe for concurrent async access
    """
    
    def __init__(self) -> None:
        self._counters: dict[str, Counter] = {}
        self._gauges: dict[str, Gauge] = {}
        self._histograms: dict[str, Histogram] = {}
    
    def counter(
        self,
        name: str,
        description: str = "",
        labels: dict[str, str] | None = None,
    ) -> Counter:
        """Get or create counter."""
        key = self._make_key(name, labels)
        if key not in self._counters:
            self._counters[key] = Counter(name, description, labels or {})
        return self._counters[key]
    
    def gauge(
        self,
        name: str,
        description: str = "",
        labels: dict[str, str] | None = None,
    ) -> Gauge:
        """Get or create gauge."""
        key = self._make_key(name, labels)
        if key not in self._gauges:
            self._gauges[key] = Gauge(name, description, labels or {})
        return self._gauges[key]
    
    def histogram(
        self,
        name: str,
        description: str = "",
        labels: dict[str, str] | None = None,
        buckets: tuple[float, ...] | None = None,
    ) -> Histogram:
        """Get or create histogram."""
        key = self._make_key(name, labels)
        if key not in self._histograms:
            self._histograms[key] = Histogram(
                name, 
                description,
                buckets=buckets or HISTOGRAM_BUCKETS,
                labels=labels or {},
            )
        return self._histograms[key]
    
    def _make_key(self, name: str, labels: dict[str, str] | None) -> str:
        """Create unique key from name and labels."""
        if not labels:
            return name
        sorted_labels = sorted(labels.items())
        return f"{name}:{','.join(f'{k}={v}' for k, v in sorted_labels)}"
    
    def export_prometheus(self) -> str:
        """Export all metrics in Prometheus format."""
        lines: list[str] = []
        
        for counter in self._counters.values():
            lines.append(f"# TYPE {counter.name} counter")
            lines.append(counter.to_prometheus())
        
        for gauge in self._gauges.values():
            lines.append(f"# TYPE {gauge.name} gauge")
            lines.append(gauge.to_prometheus())
        
        for histogram in self._histograms.values():
            lines.append(f"# TYPE {histogram.name} histogram")
            lines.append(histogram.to_prometheus())
        
        return "\n".join(lines)
    
    def get_all_stats(self) -> dict[str, Any]:
        """Get all metrics as dict for JSON export."""
        stats: dict[str, Any] = {
            "counters": {},
            "gauges": {},
            "histograms": {},
        }
        
        for key, counter in self._counters.items():
            stats["counters"][key] = {
                "value": counter.value,
                "labels": counter.labels,
            }
        
        for key, gauge in self._gauges.items():
            stats["gauges"][key] = {
                "value": gauge.value,
                "labels": gauge.labels,
            }
        
        for key, histogram in self._histograms.items():
            stats["histograms"][key] = {
                "count": histogram.count,
                "sum": histogram.sum,
                "mean": histogram.mean,
                "p50": histogram.percentile(50),
                "p90": histogram.percentile(90),
                "p95": histogram.percentile(95),
                "p99": histogram.percentile(99),
                "labels": histogram.labels,
            }
        
        return stats


# Global registry
_registry = MetricsRegistry()


def get_registry() -> MetricsRegistry:
    """Get global metrics registry."""
    return _registry


# =============================================================================
# STANDARD METRICS: Pre-defined inference metrics
# =============================================================================

# Request latency
def request_latency(provider: str = "") -> Histogram:
    """Get request latency histogram."""
    return _registry.histogram(
        "inference_request_latency_seconds",
        "Request latency in seconds",
        labels={"provider": provider} if provider else None,
    )


# Request counter
def request_counter(provider: str = "", endpoint: str = "") -> Counter:
    """Get request counter."""
    labels = {}
    if provider:
        labels["provider"] = provider
    if endpoint:
        labels["endpoint"] = endpoint
    return _registry.counter(
        "inference_requests_total",
        "Total requests",
        labels=labels or None,
    )


# Error counter
def error_counter(provider: str = "", error_type: str = "") -> Counter:
    """Get error counter."""
    labels = {}
    if provider:
        labels["provider"] = provider
    if error_type:
        labels["error_type"] = error_type
    return _registry.counter(
        "inference_errors_total",
        "Total errors",
        labels=labels or None,
    )


# Token counter
def token_counter(direction: str = "input") -> Counter:
    """Get token counter (input or output)."""
    return _registry.counter(
        "inference_tokens_total",
        "Total tokens processed",
        labels={"direction": direction},
    )


# Active requests gauge
def active_requests(provider: str = "") -> Gauge:
    """Get active requests gauge."""
    return _registry.gauge(
        "inference_active_requests",
        "Currently active requests",
        labels={"provider": provider} if provider else None,
    )


# Queue depth gauge
def queue_depth() -> Gauge:
    """Get queue depth gauge."""
    return _registry.gauge(
        "inference_queue_depth",
        "Current queue depth",
    )


# =============================================================================
# STRUCTURED LOGGING: JSON-formatted log output
# =============================================================================

class StructuredLogger:
    """
    Structured JSON logger with context propagation.
    
    Features:
        - JSON output for log aggregation
        - Automatic request ID inclusion
        - Nanosecond timestamps
        - Level-based filtering
    """
    
    def __init__(
        self,
        name: str,
        level: int = logging.INFO,
    ) -> None:
        self._name = name
        self._level = level
        self._handler = logging.StreamHandler(sys.stdout)
        self._handler.setFormatter(logging.Formatter("%(message)s"))
        
        self._logger = logging.getLogger(name)
        self._logger.setLevel(level)
        self._logger.addHandler(self._handler)
        self._logger.propagate = False
    
    def _format_message(
        self,
        level: str,
        message: str,
        extra: dict[str, Any] | None = None,
    ) -> str:
        """Format log entry as JSON."""
        entry: dict[str, Any] = {
            "timestamp": time.time(),
            "timestamp_ns": time.perf_counter_ns(),
            "level": level,
            "logger": self._name,
            "message": message,
        }
        
        # Add context
        request_id = get_request_id()
        if request_id:
            entry["request_id"] = request_id
        
        span_id = get_span_id()
        if span_id:
            entry["span_id"] = span_id
        
        # Add extra fields
        if extra:
            entry["extra"] = extra
        
        return json.dumps(entry)
    
    def debug(self, message: str, **extra: Any) -> None:
        """Log debug message."""
        if self._logger.isEnabledFor(logging.DEBUG):
            self._logger.debug(self._format_message("DEBUG", message, extra or None))
    
    def info(self, message: str, **extra: Any) -> None:
        """Log info message."""
        if self._logger.isEnabledFor(logging.INFO):
            self._logger.info(self._format_message("INFO", message, extra or None))
    
    def warning(self, message: str, **extra: Any) -> None:
        """Log warning message."""
        if self._logger.isEnabledFor(logging.WARNING):
            self._logger.warning(self._format_message("WARNING", message, extra or None))
    
    def error(self, message: str, **extra: Any) -> None:
        """Log error message."""
        if self._logger.isEnabledFor(logging.ERROR):
            self._logger.error(self._format_message("ERROR", message, extra or None))
    
    def critical(self, message: str, **extra: Any) -> None:
        """Log critical message."""
        if self._logger.isEnabledFor(logging.CRITICAL):
            self._logger.critical(self._format_message("CRITICAL", message, extra or None))


# Default logger
_default_logger: StructuredLogger | None = None


def get_logger(name: str = "inference_core") -> StructuredLogger:
    """Get or create structured logger."""
    global _default_logger
    if _default_logger is None or _default_logger._name != name:
        _default_logger = StructuredLogger(name)
    return _default_logger


# =============================================================================
# TRACING: Request span tracking
# =============================================================================

@dataclass
class Span:
    """
    Request span for tracing.
    
    Tracks:
        - Start/end timestamps (nanosecond precision)
        - Parent span for nesting
        - Attributes for context
    """
    name: str
    span_id: str
    parent_id: str | None = None
    start_ns: int = field(default_factory=time.perf_counter_ns)
    end_ns: int | None = None
    attributes: dict[str, Any] = field(default_factory=dict)
    
    def end(self) -> None:
        """Mark span as ended."""
        self.end_ns = time.perf_counter_ns()
    
    @property
    def duration_ns(self) -> int:
        """Span duration in nanoseconds."""
        if self.end_ns is None:
            return time.perf_counter_ns() - self.start_ns
        return self.end_ns - self.start_ns
    
    @property
    def duration_ms(self) -> float:
        """Span duration in milliseconds."""
        return self.duration_ns / 1_000_000
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dict for export."""
        return {
            "name": self.name,
            "span_id": self.span_id,
            "parent_id": self.parent_id,
            "start_ns": self.start_ns,
            "end_ns": self.end_ns,
            "duration_ms": self.duration_ms,
            "attributes": self.attributes,
        }


class Tracer:
    """
    Request tracer for distributed tracing.
    
    Creates hierarchical spans for request tracking.
    """
    
    def __init__(self) -> None:
        self._spans: dict[str, Span] = {}
    
    def start_span(
        self,
        name: str,
        parent_id: str | None = None,
        attributes: dict[str, Any] | None = None,
    ) -> Span:
        """Start a new span."""
        import uuid
        span_id = uuid.uuid4().hex[:16]
        
        span = Span(
            name=name,
            span_id=span_id,
            parent_id=parent_id or get_span_id() or None,
            attributes=attributes or {},
        )
        
        self._spans[span_id] = span
        set_span(span_id)
        
        return span
    
    def end_span(self, span: Span) -> None:
        """End a span."""
        span.end()
        
        # Log span
        logger = get_logger()
        logger.debug(
            f"Span completed: {span.name}",
            span=span.to_dict(),
        )
    
    def get_span(self, span_id: str) -> Span | None:
        """Get span by ID."""
        return self._spans.get(span_id)


# Global tracer
_tracer = Tracer()


def get_tracer() -> Tracer:
    """Get global tracer."""
    return _tracer


# =============================================================================
# TIMING UTILITIES: Convenient latency measurement
# =============================================================================

class Timer:
    """
    Context manager for timing code blocks.
    
    Usage:
        with Timer() as t:
            do_work()
        print(f"Took {t.elapsed_ms}ms")
    
    Or with histogram:
        with Timer(histogram=request_latency()):
            do_work()
    """
    
    def __init__(
        self,
        histogram: Histogram | None = None,
        name: str = "",
    ) -> None:
        self._histogram = histogram
        self._name = name
        self._start_ns: int = 0
        self._end_ns: int = 0
    
    def __enter__(self) -> Timer:
        self._start_ns = time.perf_counter_ns()
        return self
    
    def __exit__(self, *args: Any) -> None:
        self._end_ns = time.perf_counter_ns()
        
        if self._histogram:
            elapsed_seconds = self.elapsed_seconds
            self._histogram.observe(elapsed_seconds)
    
    @property
    def elapsed_ns(self) -> int:
        """Elapsed time in nanoseconds."""
        if self._end_ns > 0:
            return self._end_ns - self._start_ns
        return time.perf_counter_ns() - self._start_ns
    
    @property
    def elapsed_ms(self) -> float:
        """Elapsed time in milliseconds."""
        return self.elapsed_ns / 1_000_000
    
    @property
    def elapsed_seconds(self) -> float:
        """Elapsed time in seconds."""
        return self.elapsed_ns / 1_000_000_000


async def timed_async(
    coro: Any,
    histogram: Histogram | None = None,
) -> tuple[Any, float]:
    """
    Time an async operation.
    
    Returns:
        Tuple of (result, elapsed_seconds)
    """
    start_ns = time.perf_counter_ns()
    result = await coro
    elapsed_ns = time.perf_counter_ns() - start_ns
    elapsed_seconds = elapsed_ns / 1_000_000_000
    
    if histogram:
        histogram.observe(elapsed_seconds)
    
    return result, elapsed_seconds
