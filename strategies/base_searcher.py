from abc import ABC, abstractmethod
from typing import Optional

import config
from state_manager import StateManager
from utils.graph_builder import build_disjunctive_graph, summarize_graph
from models.graph_diffusion import PlaceholderGraphDiffusion, Heatmap
from utilities.logger import log_diffusion_heatmap


class BaseSearcher(ABC):
    def __init__(self, llm_agent):
        self.llm_agent = llm_agent
        self.current_strategic_experience = "None available."

        self.use_diffusion = getattr(config, "USE_DIFFUSION_HEATMAP", False)
        if self.use_diffusion:
            seed = getattr(config, "DIFFUSION_SEED", None)
            self.diffusion_model = PlaceholderGraphDiffusion(seed=seed)
        else:
            self.diffusion_model = None

    def update_strategic_experience(self, new_experience: str):
        """Updates the high-level strategy passed down by the Reflection Engine."""
        self.current_strategic_experience = new_experience

    def generate_heatmap(
        self,
        state: StateManager,
        session_folder: str,
        iteration: int,
        search_context: str = "General",
    ) -> Optional[Heatmap]:
        """Build the disjunctive graph and run the diffusion model over it.

        Returns None when the diffusion pipeline is disabled in config.
        """
        if not self.use_diffusion or self.diffusion_model is None:
            return None

        graph = build_disjunctive_graph(state)
        heatmap = self.diffusion_model.predict(graph)

        graph_stats = summarize_graph(graph)
        log_diffusion_heatmap(
            session_folder, iteration,
            state.current_time,
            graph_stats,
            heatmap.to_dict(),
            search_context=search_context,
        )

        return heatmap

    @abstractmethod
    def run_search(self, initial_state: StateManager, session_folder: str, iteration: int) -> dict:
        """
        Explores the state space and returns the single best immediate action.
        Must return a dict: {"job": j, "op": o, "machine": m}
        """
        pass
