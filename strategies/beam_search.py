import copy
from strategies.base_searcher import BaseSearcher
from state_manager import StateManager
from  utilities.logger import log_beam_search_step
import config

class BeamSearcher(BaseSearcher):
    def __init__(self, llm_agent):
        super().__init__(llm_agent)
        self.beam_width = config.BEAM_WIDTH
        self.max_depth = config.SEARCH_DEPTH
        self.value_cache = {} # The Transposition Table

    def run_search(self, initial_state: StateManager, session_folder: str, iteration: int) -> dict:
        print(f">>> [BeamSearch] Starting lookahead (Width: {self.beam_width}, Depth: {self.max_depth})")
        
        cache_hits = 0
        
        initial_actions = initial_state.get_feasible_actions()
        if not initial_actions:
            return None

        # --- Auxiliary: Diffusion heatmap at root ---
        heatmap = self.generate_heatmap(
            initial_state, session_folder, iteration,
            search_context="Beam_Root",
        )
        if heatmap is not None:
            sample_weights = list(heatmap.node_weights.items())[:5]
            print(f"    [DIFFUSION] Beam root heatmap ({len(heatmap.node_weights)} nodes). Sample: {sample_weights}")

        # Level 0: Expand the root node
        current_beams = []
        for action in initial_actions:
            cloned_state = copy.deepcopy(initial_state)
            cloned_state.execute_action(action["job"], action["op"], action["machine"])
            self._fast_forward_to_next_decision(cloned_state)
            
            # Use Cache Evaluation
            val = self._evaluate_state(cloned_state, session_folder, iteration)
            current_beams.append({"state": cloned_state, "first_action": action, "value": val})

        current_beams = sorted(current_beams, key=lambda x: x["value"], reverse=True)[:self.beam_width]
        log_beam_search_step(session_folder, iteration, initial_state.current_time, 0, self.current_strategic_experience, current_beams)

        # Level 1 to max_depth: Deepen the tree
        for depth in range(1, self.max_depth):
            print(f"    - Exploring Depth {depth}/{self.max_depth}... (Cache size: {len(self.value_cache)})")
            next_beams = []
            
            for beam in current_beams:
                state = beam["state"]
                actions = state.get_feasible_actions()
                
                if not actions and all(s == 'completed' for s in state.job_status.values()):
                    next_beams.append(beam)
                    continue
                    
                for action in actions:
                    cloned_state = copy.deepcopy(state)
                    cloned_state.execute_action(action["job"], action["op"], action["machine"])
                    self._fast_forward_to_next_decision(cloned_state)
                    
                    # Use Cache Evaluation
                    val, hit = self._evaluate_state_with_hit_tracking(cloned_state, session_folder, iteration)
                    if hit: cache_hits += 1
                    
                    next_beams.append({"state": cloned_state, "first_action": beam["first_action"], "value": val})
            
            if not next_beams:
                break
            current_beams = sorted(next_beams, key=lambda x: x["value"], reverse=True)[:self.beam_width]
            log_beam_search_step(session_folder, iteration, initial_state.current_time, depth, self.current_strategic_experience, current_beams)

        best_timeline = current_beams[0]
        print(f">>> [BeamSearch] Lookahead complete. Best value: {best_timeline['value']:.2f} | Cache Hits Prevented {cache_hits} API Calls.")
        return best_timeline["first_action"]

    def _evaluate_state_with_hit_tracking(self, state: StateManager, session_folder: str, iteration: int) -> tuple[float, bool]:
        """Checks the transposition table before querying the LLM. Returns (value, is_cache_hit)."""
        state_hash = state.get_state_hash()
        
        if state_hash in self.value_cache:
            return self.value_cache[state_hash], True
            
        # Cache Miss: Query the LLM Brain
        val = self.llm_agent.get_value(state, self.current_strategic_experience, session_folder, iteration)
        self.value_cache[state_hash] = val
        return val, False

    def _evaluate_state(self, state: StateManager, session_folder: str, iteration: int) -> float:
        """Wrapper for level 0 where we just want the value without hit tracking."""
        val, _ = self._evaluate_state_with_hit_tracking(state, session_folder, iteration)
        return val

    def _fast_forward_to_next_decision(self, state: StateManager):
        while not state.get_feasible_actions() and not all(s == 'completed' for s in state.job_status.values()):
            event_type, _, _ = state.process_next_event()
            if event_type is None:
                break