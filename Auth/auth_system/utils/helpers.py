# =============================================================================
# SOTA AUTHENTICATION SYSTEM - UTILITIES
# =============================================================================
# File: utils/helpers.py
# Description: Common utility functions used across the application
# =============================================================================

from typing import Optional, Any, Dict
from datetime import datetime, timezone
import json
import hashlib
import base64
import re
from urllib.parse import urlparse


def utc_now() -> datetime:
    """
    Get current UTC timestamp.
    
    Returns:
        datetime: Current UTC time with timezone info
    """
    return datetime.now(timezone.utc)


def hash_string(value: str, algorithm: str = "sha256") -> str:
    """
    Create hash of a string.
    
    Args:
        value: String to hash
        algorithm: Hash algorithm (sha256, sha512, md5)
        
    Returns:
        str: Hex-encoded hash
    """
    hasher = hashlib.new(algorithm)
    hasher.update(value.encode())
    return hasher.hexdigest()


def base64_encode(value: str) -> str:
    """
    Encode string to base64.
    
    Args:
        value: String to encode
        
    Returns:
        str: Base64 encoded string
    """
    return base64.b64encode(value.encode()).decode()


def base64_decode(value: str) -> str:
    """
    Decode base64 string.
    
    Args:
        value: Base64 encoded string
        
    Returns:
        str: Decoded string
    """
    return base64.b64decode(value.encode()).decode()


def is_valid_email(email: str) -> bool:
    """
    Validate email format.
    
    Args:
        email: Email address to validate
        
    Returns:
        bool: True if valid format
    """
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))


def is_valid_username(username: str) -> bool:
    """
    Validate username format.
    
    Rules:
        - 3-50 characters
        - Starts with letter
        - Alphanumeric and underscores only
    
    Args:
        username: Username to validate
        
    Returns:
        bool: True if valid format
    """
    if not 3 <= len(username) <= 50:
        return False
    pattern = r'^[a-zA-Z][a-zA-Z0-9_]*$'
    return bool(re.match(pattern, username))


def mask_email(email: str) -> str:
    """
    Mask email address for display/logging.
    
    Example: test@example.com -> t***@example.com
    
    Args:
        email: Email to mask
        
    Returns:
        str: Masked email
    """
    if "@" not in email:
        return email
    
    local, domain = email.split("@", 1)
    
    if len(local) <= 1:
        return f"{local}***@{domain}"
    
    return f"{local[0]}***@{domain}"


def mask_ip(ip: str) -> str:
    """
    Mask IP address for privacy.
    
    Example: 192.168.1.100 -> 192.168.x.x
    
    Args:
        ip: IP address to mask
        
    Returns:
        str: Masked IP
    """
    if ":" in ip:  # IPv6
        parts = ip.split(":")
        if len(parts) >= 2:
            return f"{parts[0]}:{parts[1]}:xxxx:xxxx"
        return ip
    
    # IPv4
    parts = ip.split(".")
    if len(parts) == 4:
        return f"{parts[0]}.{parts[1]}.x.x"
    return ip


def sanitize_url(url: str) -> str:
    """
    Sanitize URL by removing sensitive query parameters.
    
    Args:
        url: URL to sanitize
        
    Returns:
        str: Sanitized URL
    """
    sensitive_params = {"token", "key", "password", "secret", "code"}
    
    parsed = urlparse(url)
    
    if not parsed.query:
        return url
    
    # Parse and filter query params
    from urllib.parse import parse_qs, urlencode
    
    params = parse_qs(parsed.query)
    filtered = {
        k: v for k, v in params.items()
        if k.lower() not in sensitive_params
    }
    
    # Reconstruct URL
    new_query = urlencode(filtered, doseq=True)
    return f"{parsed.scheme}://{parsed.netloc}{parsed.path}?{new_query}"


def parse_user_agent(user_agent: Optional[str]) -> Dict[str, Any]:
    """
    Parse user agent string into device info.
    
    Args:
        user_agent: User agent string
        
    Returns:
        dict: Parsed device information
    """
    if not user_agent:
        return {"browser": "unknown", "os": "unknown", "device": "unknown"}
    
    info = {
        "browser": "unknown",
        "os": "unknown",
        "device": "desktop",
        "raw": user_agent[:200],  # Truncate for storage
    }
    
    ua_lower = user_agent.lower()
    
    # Detect browser
    if "firefox" in ua_lower:
        info["browser"] = "Firefox"
    elif "edg" in ua_lower:
        info["browser"] = "Edge"
    elif "chrome" in ua_lower:
        info["browser"] = "Chrome"
    elif "safari" in ua_lower:
        info["browser"] = "Safari"
    elif "opera" in ua_lower:
        info["browser"] = "Opera"
    
    # Detect OS
    if "windows" in ua_lower:
        info["os"] = "Windows"
    elif "mac" in ua_lower:
        info["os"] = "macOS"
    elif "linux" in ua_lower:
        info["os"] = "Linux"
    elif "android" in ua_lower:
        info["os"] = "Android"
        info["device"] = "mobile"
    elif "iphone" in ua_lower:
        info["os"] = "iOS"
        info["device"] = "mobile"
    elif "ipad" in ua_lower:
        info["os"] = "iOS"
        info["device"] = "tablet"
    
    return info


def truncate_string(value: str, max_length: int = 100, suffix: str = "...") -> str:
    """
    Truncate string to maximum length.
    
    Args:
        value: String to truncate
        max_length: Maximum length
        suffix: Suffix to add when truncated
        
    Returns:
        str: Truncated string
    """
    if len(value) <= max_length:
        return value
    return value[:max_length - len(suffix)] + suffix


def safe_json_loads(value: str, default: Any = None) -> Any:
    """
    Safely parse JSON string.
    
    Args:
        value: JSON string
        default: Default value if parsing fails
        
    Returns:
        Parsed value or default
    """
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return default


def safe_json_dumps(value: Any, default: str = "{}") -> str:
    """
    Safely serialize to JSON string.
    
    Args:
        value: Value to serialize
        default: Default string if serialization fails
        
    Returns:
        JSON string or default
    """
    try:
        return json.dumps(value)
    except (TypeError, ValueError):
        return default
