# =============================================================================
# AI_CORE - Advanced Chatbot Core AI System
# =============================================================================
# Author: Advanced AI Engineering Team
# Version: 1.0.0
# Description: Production-grade chatbot core supporting 100+ LLM providers
#              via LiteLLM, with session management, document augmentation,
#              streaming, and batch processing capabilities.
# =============================================================================

from main import ChatbotCoreAI
from config.settings import Settings
from models.schemas import (
    ChatMessage,
    ChatRequest,
    ChatResponse,
    ChatChunk,
    SessionConfig,
    DocumentContext,
    FeedbackData,
    ProviderConfig
)

# =============================================================================
# VERSION INFORMATION
# =============================================================================
__version__ = "1.0.0"
__author__ = "Advanced AI Engineering Team"

# =============================================================================
# PUBLIC API EXPORTS
# =============================================================================
__all__ = [
    # Main Entry Point
    "ChatbotCoreAI",
    
    # Configuration
    "Settings",
    
    # Data Models
    "ChatMessage",
    "ChatRequest", 
    "ChatResponse",
    "ChatChunk",
    "SessionConfig",
    "DocumentContext",
    "FeedbackData",
    "ProviderConfig",
]
