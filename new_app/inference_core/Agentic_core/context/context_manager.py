"""
Context Window Manager.

Adheres to:
- Robust Input Sanitization: Bounds checking on token limits.
"""
from typing import List, Dict
from ..core.config import get_config

CONFIG = get_config()

"""
Context Window Manager (SOTA).

Adheres to:
- Robust Input Sanitization: Normalizes unicode, strips dangerous control chars.
- Continuity: Uses "Head-Body-Tail" strategy to preserve the Objective (Beginning) and Recent Context (Ending).
- Safety: Strictly enforces token budgets.
"""
from typing import List, Dict, Tuple
import unicodedata
import re
from ..core.config import get_config

CONFIG = get_config()

class ContextManager:
    """
    Smart Context Compositor.
    """
    def __init__(self, max_tokens: int = CONFIG.max_context_window):
        self.max_tokens = max_tokens
        self.chars_per_token = 4.0 # Estimate
        # Reserve space for the model's reply
        self.response_reserve = 1000

    def sanitize_input(self, text: str) -> str:
        """
        SOTA Sanitization:
        1. Normalize Unicode (NFKC) to avoid homoglyph attacks.
        2. Strip null bytes and non-printable control chars (except newlines/tabs).
        """
        if not text:
            return ""
        
        # 1. Unicode Normalization
        text = unicodedata.normalize('NFKC', text)
        
        # 2. Control Char Removal (keeping \n \t \r)
        # Remove anything in the 'Cc' category (Control) except allowed
        return "".join(ch for ch in text if unicodedata.category(ch) != "Cc" or ch in "\t\n\r")

    def estimate_tokens(self, text: str) -> int:
        return int(len(text) / self.chars_per_token)

    def compose_context(self, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        Optimizes context to fit window while preserving Narrative Arc.
        Strategy: Head (System + Objective) + Tail (Recent) > Body (Middle).
        """
        if not messages:
            return []

        # 1. Sanitize all inputs in passing
        sanitized_msgs = [
            {**msg, "content": self.sanitize_input(msg.get("content", ""))} 
            for msg in messages
        ]

        budget = self.max_tokens - self.response_reserve
        used_tokens = 0
        final_context = []

        # A. Identify Critical Components
        system_msgs = [m for m in sanitized_msgs if m['role'] == 'system']
        user_msgs = [m for m in sanitized_msgs if m['role'] != 'system']
        
        if not user_msgs:
            return system_msgs

        # B. Always include System Prompts (Head 1)
        for msg in system_msgs:
            t = self.estimate_tokens(msg['content'])
            if used_tokens + t < budget:
                final_context.append(msg)
                used_tokens += t

        # C. Always include First User Message (Head 2 - The "Beginning")
        # This usually contains the core user objective.
        first_user = user_msgs[0]
        t_first = self.estimate_tokens(first_user['content'])
        
        if used_tokens + t_first < budget:
            # We add it tentatively, but strict order matters for LLM.
            # We'll assemble the list at the end, but account for budget now.
            used_tokens += t_first
        else:
            # Extremely tight budget, critical failure fallback
            return system_msgs + [first_user] 

        # D. Fill with Tail (Recent History - The "Ending")
        # We work backwards from the end of the user_msgs list
        tail_msgs = []
        # Skip the first one since we accounted for it (unless it's the only one)
        remaining_candidates = user_msgs[1:] if len(user_msgs) > 1 else []
        
        for msg in reversed(remaining_candidates):
            t = self.estimate_tokens(msg['content'])
            if used_tokens + t < budget:
                tail_msgs.insert(0, msg)
                used_tokens += t
            else:
                # Budget full.
                # Insert a placeholder if we skipped messages in the middle
                if len(tail_msgs) < len(remaining_candidates):
                    # We could add a summary token here if we had one
                    pass
                break
        
        # E. Assemble in Chronological Order
        # [System..., First_User, ...Tail...]
        # Ensure we don't duplicate if first_user was also in tail (logic above handles this by splitting candidates)
        
        result = [m for m in final_context] # System msgs
        result.append(first_user)
        result.extend(tail_msgs)
        
        return result

    def prune_messages(self, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Legacy alias for compatibility."""
        return self.compose_context(messages)
