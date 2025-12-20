# -*- coding: utf-8 -*-
# =================================================================================================
# api/__init__.py
# =================================================================================================
# Package initialization for the Advanced FastAPI Backend API.
#
# Design Principles Applied:
# --------------------------
#   • Lazy imports: Heavy modules (torch, LLM clients) are NOT imported at package level to allow
#     fast startup for health checks and reduce cold-start latency in serverless deployments.
#   • Version exposure: API version is centralized here for consistent versioning across endpoints.
#   • Namespace hygiene: Only essential symbols are exported via __all__.
#
# Architecture:
# -------------
#   api/
#   ├── __init__.py         <- YOU ARE HERE
#   ├── config.py           <- Configuration management (env + YAML)
#   ├── database.py         <- Async database connection pooling
#   ├── models.py           <- ORM models for history, sessions, responses
#   ├── schemas.py          <- Pydantic request/response models
#   ├── dependencies.py     <- FastAPI dependency injection
#   ├── exceptions.py       <- Custom exception hierarchy
#   ├── middleware.py       <- Request logging, timing, rate limiting
#   ├── main.py             <- FastAPI application entrypoint
#   └── routers/
#       ├── __init__.py
#       ├── conversations.py   <- Chat history & session management
#       ├── documents.py       <- Document conversion & chunking
#       ├── search.py          <- Vector search & retrieval
#       ├── generation.py      <- LLM generation endpoints
#       └── health.py          <- Health checks & metrics
#
# =================================================================================================

__version__ = "1.0.0"
__api_version__ = "v1"
__author__ = "CCA Chatbot Team"

# -----------------------------------------------------------------------------
# Minimal exports: defer heavy imports to actual usage sites.
# -----------------------------------------------------------------------------
__all__ = [
    "__version__",
    "__api_version__",
]
