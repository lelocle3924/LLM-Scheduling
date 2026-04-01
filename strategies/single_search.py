from strategies.base_searcher import BaseSearcher
from state_manager import StateManager

class SingleSearcher(BaseSearcher):
    def __init__(self, llm_agent):
        super().__init__(llm_agent)

    def run_search(self, initial_state: StateManager, session_folder: str, iteration: int) -> dict:
        """
        Baseline strategy: No tree search. Directly asks the LLM for 1 action.
        """
        feasible_actions = initial_state.get_feasible_actions()
        
        if not feasible_actions:
            return None

        heatmap = self.generate_heatmap(
            initial_state, session_folder, iteration,
            search_context="Single_Action",
        )
        if heatmap is not None:
            sample_weights = list(heatmap.node_weights.items())[:5]
            print(f"    [DIFFUSION] Single heatmap ({len(heatmap.node_weights)} nodes). Sample: {sample_weights}")

        print(f">>> [SingleSearch] Evaluating {len(feasible_actions)} immediate actions...")
        
        decision = self.llm_agent.get_action(
            state=initial_state,
            feasible_actions=feasible_actions,
            strategic_experience=self.current_strategic_experience,
            session_folder=session_folder,
            iteration=iteration
        )
        
        return decision