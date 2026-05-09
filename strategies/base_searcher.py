from abc import ABC, abstractmethod

from state_manager import StateManager


class BaseSearcher(ABC):
    def __init__(self, llm_agent):
        self.llm_agent = llm_agent
        self.current_strategic_experience = "None available."

    def update_strategic_experience(self, new_experience: str):
        """Updates the high-level strategy passed down by the Reflection Engine."""
        self.current_strategic_experience = new_experience

    @abstractmethod
    def run_search(self, initial_state: StateManager, session_folder: str, iteration: int) -> dict:
        """
        Explores the state space and returns the single best immediate action.
        Must return a dict: {"job": j, "op": o, "machine": m}
        """
        pass
