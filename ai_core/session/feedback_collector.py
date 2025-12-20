# =============================================================================
# FEEDBACK COLLECTOR - User Feedback Collection System
# =============================================================================
# Collects and manages user feedback on AI responses.
# =============================================================================

from __future__ import annotations
import threading
from typing import Dict, List, Optional, Any, Literal
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class FeedbackType(str, Enum):
    """Types of feedback."""
    THUMBS_UP = "thumbs_up"
    THUMBS_DOWN = "thumbs_down"
    RATING = "rating"
    TEXT = "text"
    CORRECTION = "correction"
    REPORT = "report"


@dataclass
class Feedback:
    """
    Represents user feedback on a response.
    
    Attributes:
        feedback_id: Unique feedback identifier
        session_id: Associated session
        message_id: Associated message (if any)
        feedback_type: Type of feedback
        value: Feedback value (bool, int, or str)
        comment: Optional text comment
        categories: Feedback categories/tags
        created_at: Feedback timestamp
    """
    feedback_id: str
    session_id: str
    message_id: Optional[str] = None
    feedback_type: FeedbackType = FeedbackType.THUMBS_UP
    value: Any = None
    comment: Optional[str] = None
    categories: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def is_positive(self) -> bool:
        """Check if feedback is positive."""
        if self.feedback_type == FeedbackType.THUMBS_UP:
            return True
        if self.feedback_type == FeedbackType.THUMBS_DOWN:
            return False
        if self.feedback_type == FeedbackType.RATING and isinstance(self.value, int):
            return self.value >= 4  # Assuming 1-5 scale
        return False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "feedback_id": self.feedback_id,
            "session_id": self.session_id,
            "message_id": self.message_id,
            "feedback_type": self.feedback_type.value,
            "value": self.value,
            "comment": self.comment,
            "categories": self.categories,
            "created_at": self.created_at.isoformat(),
            "metadata": self.metadata,
        }


class FeedbackCollector:
    """
    Collects and manages user feedback.
    
    Features:
        - Multiple feedback types (thumbs, ratings, text)
        - Feedback aggregation per session
        - Analytics support
        - Thread-safe operations
    
    Example:
        >>> collector = FeedbackCollector()
        >>> 
        >>> # Submit thumbs up
        >>> collector.submit_thumbs_up("session-1", "msg-1")
        >>> 
        >>> # Submit rating
        >>> collector.submit_rating("session-1", "msg-2", rating=5)
        >>> 
        >>> # Get session feedback
        >>> feedback = collector.get_session_feedback("session-1")
    """
    
    def __init__(self, max_feedback_per_session: int = 100):
        """
        Initialize feedback collector.
        
        Args:
            max_feedback_per_session: Maximum feedback entries per session
        """
        self.max_feedback = max_feedback_per_session
        self._feedback: Dict[str, List[Feedback]] = {}
        self._lock = threading.RLock()
        self._counter = 0
    
    def _generate_id(self) -> str:
        """Generate unique feedback ID."""
        with self._lock:
            self._counter += 1
            return f"fb-{self._counter}"
    
    def submit_feedback(
        self,
        session_id: str,
        feedback_type: FeedbackType,
        value: Any = None,
        message_id: Optional[str] = None,
        comment: Optional[str] = None,
        categories: Optional[List[str]] = None,
        **metadata
    ) -> Feedback:
        """
        Submit feedback.
        
        Args:
            session_id: Session identifier
            feedback_type: Type of feedback
            value: Feedback value
            message_id: Associated message ID
            comment: Text comment
            categories: Feedback categories
            **metadata: Additional metadata
            
        Returns:
            Created Feedback object
        """
        with self._lock:
            feedback = Feedback(
                feedback_id=self._generate_id(),
                session_id=session_id,
                message_id=message_id,
                feedback_type=feedback_type,
                value=value,
                comment=comment,
                categories=categories or [],
                metadata=metadata
            )
            
            if session_id not in self._feedback:
                self._feedback[session_id] = []
            
            self._feedback[session_id].append(feedback)
            
            # Trim if exceeds max
            if len(self._feedback[session_id]) > self.max_feedback:
                self._feedback[session_id] = self._feedback[session_id][-self.max_feedback:]
            
            logger.debug(f"Feedback submitted: {feedback.feedback_id}")
            return feedback
    
    def submit_thumbs_up(
        self,
        session_id: str,
        message_id: Optional[str] = None,
        comment: Optional[str] = None
    ) -> Feedback:
        """Submit thumbs up feedback."""
        return self.submit_feedback(
            session_id=session_id,
            feedback_type=FeedbackType.THUMBS_UP,
            value=True,
            message_id=message_id,
            comment=comment
        )
    
    def submit_thumbs_down(
        self,
        session_id: str,
        message_id: Optional[str] = None,
        comment: Optional[str] = None,
        categories: Optional[List[str]] = None
    ) -> Feedback:
        """Submit thumbs down feedback."""
        return self.submit_feedback(
            session_id=session_id,
            feedback_type=FeedbackType.THUMBS_DOWN,
            value=False,
            message_id=message_id,
            comment=comment,
            categories=categories
        )
    
    def submit_rating(
        self,
        session_id: str,
        message_id: Optional[str] = None,
        rating: int = 5,
        comment: Optional[str] = None
    ) -> Feedback:
        """Submit rating feedback (1-5 scale)."""
        rating = max(1, min(5, rating))  # Clamp to 1-5
        return self.submit_feedback(
            session_id=session_id,
            feedback_type=FeedbackType.RATING,
            value=rating,
            message_id=message_id,
            comment=comment
        )
    
    def submit_text_feedback(
        self,
        session_id: str,
        text: str,
        message_id: Optional[str] = None,
        categories: Optional[List[str]] = None
    ) -> Feedback:
        """Submit text feedback."""
        return self.submit_feedback(
            session_id=session_id,
            feedback_type=FeedbackType.TEXT,
            value=text,
            message_id=message_id,
            comment=text,
            categories=categories
        )
    
    def submit_correction(
        self,
        session_id: str,
        message_id: str,
        correction: str,
        original: Optional[str] = None
    ) -> Feedback:
        """Submit correction feedback."""
        return self.submit_feedback(
            session_id=session_id,
            feedback_type=FeedbackType.CORRECTION,
            value=correction,
            message_id=message_id,
            comment=correction,
            original_content=original
        )
    
    def get_session_feedback(self, session_id: str) -> List[Feedback]:
        """Get all feedback for a session."""
        with self._lock:
            return self._feedback.get(session_id, []).copy()
    
    def get_message_feedback(
        self,
        session_id: str,
        message_id: str
    ) -> List[Feedback]:
        """Get feedback for a specific message."""
        with self._lock:
            return [
                fb for fb in self._feedback.get(session_id, [])
                if fb.message_id == message_id
            ]
    
    def get_session_stats(self, session_id: str) -> Dict[str, Any]:
        """
        Get feedback statistics for a session.
        
        Returns:
            Dictionary with positive/negative counts, average rating, etc.
        """
        with self._lock:
            feedback_list = self._feedback.get(session_id, [])
            
            if not feedback_list:
                return {"total": 0}
            
            positive = sum(1 for fb in feedback_list if fb.is_positive)
            negative = sum(1 for fb in feedback_list if not fb.is_positive)
            
            ratings = [
                fb.value for fb in feedback_list
                if fb.feedback_type == FeedbackType.RATING and isinstance(fb.value, int)
            ]
            
            return {
                "total": len(feedback_list),
                "positive": positive,
                "negative": negative,
                "positive_rate": positive / len(feedback_list) if feedback_list else 0,
                "average_rating": sum(ratings) / len(ratings) if ratings else None,
                "rating_count": len(ratings),
            }
    
    def clear_session_feedback(self, session_id: str) -> None:
        """Clear feedback for a session."""
        with self._lock:
            if session_id in self._feedback:
                del self._feedback[session_id]
    
    def export_all(self) -> Dict[str, List[Dict]]:
        """Export all feedback as dictionary."""
        with self._lock:
            return {
                sid: [fb.to_dict() for fb in fbs]
                for sid, fbs in self._feedback.items()
            }
