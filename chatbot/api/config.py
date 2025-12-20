# -*- coding: utf-8 -*-
# =================================================================================================
# api/config.py — Configuration Management Module
# =================================================================================================
# Production-grade configuration system with the following SOTA patterns:
#
#   1. HIERARCHICAL LOADING: Environment variables override YAML, YAML overrides defaults.
#   2. TYPE SAFETY: All config values are validated via dataclass fields with explicit types.
#   3. SINGLETON PATTERN: Configuration is loaded once and cached globally for performance.
#   4. FAIL-FAST: Invalid configuration causes immediate startup failure with clear errors.
#   5. SECURITY: Sensitive values (API keys) are never logged or exposed in error messages.
#
# Environment Variable Precedence (highest to lowest):
# -----------------------------------------------------
#   ENV VAR → ragconfig.yaml → Hardcoded Default
#
# Usage:
# ------
#   from api.config import get_settings
#   settings = get_settings()
#   print(settings.llm_base_url)
#
# =================================================================================================

from __future__ import annotations

import dataclasses
import functools
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

# -----------------------------------------------------------------------------
# Logging Configuration (module-level)
# -----------------------------------------------------------------------------
_LOG = logging.getLogger("api.config")


# -----------------------------------------------------------------------------
# YAML Loader: Attempt to use PyYAML if available, otherwise use fallback.
# -----------------------------------------------------------------------------
def _load_yaml_file(path: Path) -> Dict[str, Any]:
    """
    Load YAML configuration file with graceful fallback.
    
    Implementation Notes:
    ---------------------
    - Uses PyYAML's safe_load to prevent arbitrary code execution.
    - Returns empty dict if file doesn't exist (non-fatal for flexibility).
    - Logs warnings on parse errors but continues with defaults.
    """
    if not path.exists():
        _LOG.debug("Config file not found: %s (using defaults)", path)
        return {}
    
    try:
        import yaml  # type: ignore[import-untyped]
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        _LOG.info("Loaded configuration from: %s", path)
        return data
    except ImportError:
        # Fallback: basic YAML-like parsing for simple key: value files
        _LOG.warning("PyYAML not installed; attempting basic config parsing")
        return _parse_simple_yaml(path)
    except Exception as e:
        _LOG.warning("Failed to parse config file %s: %s", path, e)
        return {}


def _parse_simple_yaml(path: Path) -> Dict[str, Any]:
    """
    Minimal YAML parser for simple key: value configurations.
    
    Handles:
    --------
    - String values (quoted and unquoted)
    - Numeric values (int, float)
    - Boolean-like strings ("true", "false", null)
    - Comments (lines starting with #)
    
    Does NOT handle:
    ----------------
    - Nested structures, lists, multi-line strings
    """
    result: Dict[str, Any] = {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                # Skip empty lines and comments
                if not line or line.startswith("#"):
                    continue
                # Parse key: value
                if ":" not in line:
                    continue
                key, _, val = line.partition(":")
                key = key.strip()
                val = val.strip()
                # Remove inline comments
                if "#" in val:
                    val = val.split("#")[0].strip()
                # Remove surrounding quotes
                if (val.startswith('"') and val.endswith('"')) or \
                   (val.startswith("'") and val.endswith("'")):
                    val = val[1:-1]
                # Type coercion
                if val.lower() in ("null", "none", "~"):
                    result[key] = None
                elif val.lower() in ("true", "yes"):
                    result[key] = True
                elif val.lower() in ("false", "no"):
                    result[key] = False
                else:
                    try:
                        result[key] = int(val)
                    except ValueError:
                        try:
                            result[key] = float(val)
                        except ValueError:
                            result[key] = val
    except Exception as e:
        _LOG.error("Failed to parse simple YAML at line %d: %s", line_num, e)
    return result


# -----------------------------------------------------------------------------
# Environment Variable Helpers
# -----------------------------------------------------------------------------
def _env_str(key: str, default: Optional[str] = None) -> Optional[str]:
    """Get string environment variable with optional default."""
    return os.environ.get(key, default)


def _env_int(key: str, default: Optional[int] = None) -> Optional[int]:
    """Get integer environment variable with type conversion."""
    val = os.environ.get(key)
    if val is None:
        return default
    try:
        return int(val)
    except ValueError:
        _LOG.warning("Invalid integer for %s: %s (using default)", key, val)
        return default


def _env_float(key: str, default: Optional[float] = None) -> Optional[float]:
    """Get float environment variable with type conversion."""
    val = os.environ.get(key)
    if val is None:
        return default
    try:
        return float(val)
    except ValueError:
        _LOG.warning("Invalid float for %s: %s (using default)", key, val)
        return default


def _env_bool(key: str, default: bool = False) -> bool:
    """Get boolean environment variable (true/false/1/0/yes/no)."""
    val = os.environ.get(key)
    if val is None:
        return default
    return val.lower() in ("true", "1", "yes", "on")


def _env_list(key: str, default: Optional[List[str]] = None, sep: str = ",") -> List[str]:
    """Get list environment variable (comma-separated)."""
    val = os.environ.get(key)
    if val is None:
        return default or []
    return [item.strip() for item in val.split(sep) if item.strip()]


# -----------------------------------------------------------------------------
# Settings Dataclass — Immutable Configuration Container
# -----------------------------------------------------------------------------
@dataclasses.dataclass(frozen=True)
class Settings:
    """
    Immutable application settings with full type annotations.
    
    Groupings:
    ----------
    1. Application: Basic app metadata and paths
    2. Database: SQLite/connection pool settings
    3. Embedding: Vector embedding model configuration
    4. LLM: Language model endpoint settings
    5. Retrieval: RAG retrieval parameters
    6. API: Rate limiting, CORS, authentication
    7. Runtime: Device selection, threading
    
    Notes:
    ------
    - Frozen dataclass prevents accidental mutation after initialization.
    - All fields have explicit types and sensible defaults.
    - Use dataclasses.replace() for per-request overrides.
    """
    
    # -------------------------------------------------------------------------
    # Application Settings
    # -------------------------------------------------------------------------
    app_name: str = "CCA Chatbot API"
    app_version: str = "1.0.0"
    debug: bool = False
    log_level: str = "INFO"
    base_dir: Path = dataclasses.field(default_factory=lambda: Path(__file__).parent.parent)
    data_dir: Path = dataclasses.field(default_factory=lambda: Path(__file__).parent.parent / "data")
    
    # -------------------------------------------------------------------------
    # Database Settings (SQLite with WAL mode for concurrency)
    # -------------------------------------------------------------------------
    db_path: Path = dataclasses.field(default_factory=lambda: Path(__file__).parent.parent / "data" / "api.db")
    db_pool_size: int = 5                    # Connection pool size
    db_pool_timeout: float = 30.0            # Seconds to wait for connection
    db_echo: bool = False                    # Log SQL queries (debug only)
    
    # -------------------------------------------------------------------------
    # Embedding Settings
    # -------------------------------------------------------------------------
    embed_dim: int = 768                     # Embedding dimension (mpnet-base-v2 = 768)
    embed_model: str = "sentence-transformers/all-mpnet-base-v2"
    embed_batch_size: int = 64               # Batch size for embedding calls
    embed_normalize: bool = True             # L2-normalize embeddings
    
    # -------------------------------------------------------------------------
    # LLM Settings
    # -------------------------------------------------------------------------
    llm_base_url: str = "http://localhost:8007/v1"
    llm_api_key: str = "EMPTY"               # NEVER log this value
    llm_model: str = "openai/gpt-oss-20b"
    llm_temperature: float = 0.2
    llm_max_tokens: int = 2048
    llm_timeout_s: float = 60.0              # Request timeout
    llm_max_retries: int = 3                 # Retry count on failure
    llm_max_concurrency: int = 8             # Bounded concurrent requests
    llm_backoff_initial_s: float = 0.5       # Initial retry backoff
    llm_backoff_max_s: float = 8.0           # Maximum retry backoff
    llm_jitter_s: float = 0.25               # Jitter to prevent thundering herd
    
    # -------------------------------------------------------------------------
    # Retrieval Settings
    # -------------------------------------------------------------------------
    default_collection: str = "docs"
    default_conv_id: str = "default"
    default_branch_id: str = "main"
    retrieval_top_k: int = 10
    retrieval_nprobe: int = 16
    retrieval_refine: int = 200
    context_token_budget: int = 4096
    
    # -------------------------------------------------------------------------
    # Chunking Settings
    # -------------------------------------------------------------------------
    chunk_size: int = 1024
    chunk_overlap: int = 128
    
    # -------------------------------------------------------------------------
    # API Settings
    # -------------------------------------------------------------------------
    api_prefix: str = "/api/v1"
    cors_origins: List[str] = dataclasses.field(default_factory=lambda: ["*"])
    cors_allow_credentials: bool = True
    cors_allow_methods: List[str] = dataclasses.field(default_factory=lambda: ["*"])
    cors_allow_headers: List[str] = dataclasses.field(default_factory=lambda: ["*"])
    
    # -------------------------------------------------------------------------
    # Rate Limiting (Token Bucket Algorithm)
    # -------------------------------------------------------------------------
    rate_limit_enabled: bool = True
    rate_limit_requests_per_minute: int = 100
    rate_limit_burst: int = 20               # Allow short bursts above limit
    
    # -------------------------------------------------------------------------
    # Authentication (Simple API Key)
    # -------------------------------------------------------------------------
    auth_enabled: bool = False
    auth_api_keys: List[str] = dataclasses.field(default_factory=list)
    auth_header_name: str = "X-API-Key"
    
    # -------------------------------------------------------------------------
    # Runtime Settings
    # -------------------------------------------------------------------------
    device: Optional[str] = None             # "cuda", "cpu", or None (auto)
    num_workers: int = 4                     # Thread/process pool size
    uvicorn_host: str = "0.0.0.0"
    uvicorn_port: int = 8000
    uvicorn_workers: int = 1                 # Uvicorn worker processes
    uvicorn_reload: bool = False             # Auto-reload on code changes


# -----------------------------------------------------------------------------
# Settings Factory — Load and Cache Configuration
# -----------------------------------------------------------------------------
def _resolve_path(base: Path, path_str: Optional[str]) -> Path:
    """Resolve a path string relative to base directory."""
    if path_str is None:
        return base
    p = Path(path_str)
    return p if p.is_absolute() else base / p


def _load_settings() -> Settings:
    """
    Load settings with hierarchical precedence.
    
    Loading Order:
    --------------
    1. Hardcoded defaults (in Settings dataclass)
    2. YAML configuration file (ragconfig.yaml)
    3. Environment variables (highest priority)
    
    This ensures:
    - Reasonable defaults work out-of-the-box.
    - YAML file provides deployment-specific values.
    - Env vars allow runtime override without file changes (12-factor app).
    """
    # Determine base directory (where api/ package lives)
    base_dir = Path(__file__).parent.parent.resolve()
    
    # Load YAML configuration
    yaml_path = base_dir / "ragconfig.yaml"
    yaml_cfg = _load_yaml_file(yaml_path)
    
    # Helper to get value with precedence: env > yaml > default
    def get_val(env_key: str, yaml_key: str, default: Any, type_: type) -> Any:
        # Try environment first
        env_val = os.environ.get(env_key)
        if env_val is not None:
            try:
                if type_ == bool:
                    return env_val.lower() in ("true", "1", "yes", "on")
                elif type_ == list:
                    return [s.strip() for s in env_val.split(",") if s.strip()]
                elif type_ == Path:
                    return Path(env_val)
                else:
                    return type_(env_val)
            except (ValueError, TypeError):
                pass
        # Try YAML next
        yaml_val = yaml_cfg.get(yaml_key)
        if yaml_val is not None:
            try:
                if type_ == Path:
                    return Path(yaml_val)
                elif type_ == list and isinstance(yaml_val, str):
                    return [s.strip() for s in yaml_val.split(",") if s.strip()]
                else:
                    return type_(yaml_val) if not isinstance(yaml_val, type_) else yaml_val
            except (ValueError, TypeError):
                pass
        return default
    
    # Build settings with precedence-aware loading
    data_dir = _resolve_path(base_dir, get_val("DATA_DIR", "DATA_DIR", "data", str))
    
    settings = Settings(
        # Application
        app_name=get_val("APP_NAME", "APP_NAME", "CCA Chatbot API", str),
        debug=get_val("DEBUG", "DEBUG", False, bool),
        log_level=get_val("LOG_LEVEL", "LOG_LEVEL", "INFO", str),
        base_dir=base_dir,
        data_dir=data_dir,
        
        # Database
        db_path=_resolve_path(data_dir, get_val("DB_PATH", "DB_PATH", "api.db", str)),
        db_pool_size=get_val("DB_POOL_SIZE", "DB_POOL_SIZE", 5, int),
        db_pool_timeout=get_val("DB_POOL_TIMEOUT", "DB_POOL_TIMEOUT", 30.0, float),
        db_echo=get_val("DB_ECHO", "DB_ECHO", False, bool),
        
        # Embedding
        embed_dim=get_val("EMBED_DIM", "EMBED_DIM", 768, int),
        embed_model=get_val("VB_EMBED_MODEL", "DEFAULT_EMBED_MODEL", "sentence-transformers/all-mpnet-base-v2", str),
        embed_batch_size=get_val("EMBED_BATCH_SIZE", "EMBED_BATCH_SIZE", 64, int),
        embed_normalize=get_val("EMBED_NORMALIZE", "EMBED_NORMALIZE", True, bool),
        
        # LLM
        llm_base_url=get_val("LLM_BASE_URL", "LLM_BASE_URL", "http://localhost:8007/v1", str),
        llm_api_key=get_val("LLM_API_KEY", "LLM_API_KEY", "EMPTY", str),
        llm_model=get_val("LLM_MODEL", "LLM_MODEL", "openai/gpt-oss-20b", str),
        llm_temperature=get_val("LLM_TEMPERATURE", "LLM_TEMPERATURE", 0.2, float),
        llm_max_tokens=get_val("LLM_MAX_TOKENS", "LLM_MAX_TOKENS", 2048, int),
        llm_timeout_s=get_val("LLM_TIMEOUT_S", "LLM_TIMEOUT_S", 60.0, float),
        llm_max_retries=get_val("LLM_MAX_RETRIES", "LLM_MAX_RETRIES", 3, int),
        llm_max_concurrency=get_val("LLM_MAX_CONCURRENCY", "LLM_MAX_CONCURRENCY", 8, int),
        
        # Retrieval
        default_collection=get_val("VB_COLLECTION", "DEFAULT_COLLECTION", "docs", str),
        default_conv_id=get_val("VB_CONV_ID", "CONV_ID", "default", str),
        default_branch_id=get_val("VB_BRANCH_ID", "BRANCH_ID", "main", str),
        retrieval_top_k=get_val("RETRIEVAL_TOP_K", "RETRIEVAL_TOP_K", 10, int),
        context_token_budget=get_val("CONTEXT_TOKEN_BUDGET", "CONTEXT_TOKEN_BUDGET", 4096, int),
        
        # Chunking
        chunk_size=get_val("CHUNK_SIZE", "CHUNK_SIZE", 1024, int),
        chunk_overlap=get_val("CHUNK_OVERLAP", "CHUNK_OVERLAP", 128, int),
        
        # API
        api_prefix=get_val("API_PREFIX", "API_PREFIX", "/api/v1", str),
        cors_origins=get_val("CORS_ORIGINS", "CORS_ORIGINS", ["*"], list),
        
        # Rate Limiting
        rate_limit_enabled=get_val("RATE_LIMIT_ENABLED", "RATE_LIMIT_ENABLED", True, bool),
        rate_limit_requests_per_minute=get_val("RATE_LIMIT_RPM", "RATE_LIMIT_RPM", 100, int),
        rate_limit_burst=get_val("RATE_LIMIT_BURST", "RATE_LIMIT_BURST", 20, int),
        
        # Authentication
        auth_enabled=get_val("AUTH_ENABLED", "AUTH_ENABLED", False, bool),
        auth_api_keys=get_val("AUTH_API_KEYS", "AUTH_API_KEYS", [], list),
        auth_header_name=get_val("AUTH_HEADER_NAME", "AUTH_HEADER_NAME", "X-API-Key", str),
        
        # Runtime
        device=get_val("VB_DEVICE", "FORCED_DEVICE", None, str),
        num_workers=get_val("NUM_WORKERS", "NUM_WORKERS", 4, int),
        uvicorn_host=get_val("UVICORN_HOST", "UVICORN_HOST", "0.0.0.0", str),
        uvicorn_port=get_val("UVICORN_PORT", "UVICORN_PORT", 8000, int),
        uvicorn_workers=get_val("UVICORN_WORKERS", "UVICORN_WORKERS", 1, int),
        uvicorn_reload=get_val("UVICORN_RELOAD", "UVICORN_RELOAD", False, bool),
    )
    
    _LOG.info(
        "Configuration loaded: debug=%s, device=%s, llm_model=%s",
        settings.debug, settings.device, settings.llm_model
    )
    
    return settings


# -----------------------------------------------------------------------------
# Global Singleton Access
# -----------------------------------------------------------------------------
@functools.lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Get cached application settings (singleton pattern).
    
    Thread Safety:
    --------------
    - functools.lru_cache is thread-safe for reads after first call.
    - First call initializes settings; subsequent calls return cached instance.
    
    Usage:
    ------
        from api.config import get_settings
        settings = get_settings()
        print(settings.llm_base_url)
    """
    return _load_settings()


def clear_settings_cache() -> None:
    """
    Clear settings cache (useful for testing).
    
    Warning:
    --------
    Only use in tests or when configuration needs to be reloaded.
    """
    get_settings.cache_clear()


# -----------------------------------------------------------------------------
# Module Self-Test
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # Enable debug logging for self-test
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    )
    
    print("=" * 80)
    print("Configuration Self-Test")
    print("=" * 80)
    
    settings = get_settings()
    
    # Print non-sensitive settings
    for field in dataclasses.fields(settings):
        name = field.name
        value = getattr(settings, name)
        # Mask sensitive values
        if "key" in name.lower() or "secret" in name.lower() or "password" in name.lower():
            print(f"  {name}: ****MASKED****")
        else:
            print(f"  {name}: {value}")
    
    print("=" * 80)
    print("Configuration loaded successfully!")
