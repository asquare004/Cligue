"""
Context and conversation memory management.
"""
from typing import List, Dict
from src.utils.config import AGENT_MEMORY_K


class MemoryManager:
    """Manages the conversation history for the agent."""

    def __init__(self, k: int = AGENT_MEMORY_K):
        self.k = k
        self.history: List[Dict[str, str]] = []

    def add_message(self, role: str, content: str):
        """Adds a message to the history."""
        self.history.append({"role": role, "content": content})

    def get_history(self) -> List[Dict[str, str]]:
        """Retrieves the last k messages from the history."""
        return self.history[-self.k * 2:]  # k pairs of user/assistant messages

    def clear(self):
        """Clears the conversation history."""
        self.history = []
