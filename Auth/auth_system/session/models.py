# =============================================================================
# SOTA AUTHENTICATION SYSTEM - SESSION MODELS
# =============================================================================
# File: session/models.py
# Description: Pydantic models for session data representation
# =============================================================================

from typing import Optional, List
from datetime import datetime
from pydantic import BaseModel, Field


class SessionData(BaseModel):
    """
    Session data stored in Redis.
    
    Represents the cached session information for fast access.
    """
    user_id: str = Field(..., description="User UUID")
    email: str = Field(..., description="User email")
    username: str = Field(..., description="Username")
    roles: List[str] = Field(default_factory=list, description="User roles")
    created_at: datetime = Field(..., description="Session creation time")
    last_activity: datetime = Field(..., description="Last activity time")
    device_info: Optional[dict] = Field(None, description="Device information")
    ip_address: str = Field(..., description="Client IP address")
    
    class Config:
        from_attributes = True


class SessionInfo(BaseModel):
    """Session information for API responses."""
    session_id: str = Field(..., description="Session identifier")
    user_id: str = Field(..., description="User UUID")
    created_at: datetime = Field(..., description="Session creation time")
    last_activity: datetime = Field(..., description="Last activity time")
    expires_at: datetime = Field(..., description="Session expiration time")
    device_info: Optional[dict] = Field(None, description="Device information")
    ip_address: str = Field(..., description="Client IP address")
    is_current: bool = Field(False, description="Whether this is current session")


class SessionList(BaseModel):
    """List of sessions for API response."""
    sessions: List[SessionInfo] = Field(default_factory=list)
    total: int = Field(0, description="Total number of sessions")
