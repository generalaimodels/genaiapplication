"""
inference_core: High-Performance Unified Inference Backend
===========================================================

Production-grade async inference engine combining vLLM and OpenAI capabilities
with lock-free concurrency, zero-copy data paths, and burst traffic handling.

Architecture:
    - Lock-free request queue with priority scheduling
    - Adaptive batch processing for throughput optimization  
    - Connection pooling with HTTP/2 multiplexing
    - SSE streaming with cooperative cancellation
    - Provider abstraction for vLLM/OpenAI backends

Performance Targets:
    - <10ms p50 latency overhead
    - 1000+ concurrent connections
    - 100+ burst request handling via background tasks
"""

__version__ = "0.1.0"
__all__ = ["__version__"]
