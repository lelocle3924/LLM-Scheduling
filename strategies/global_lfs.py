import copy
from typing import List, Dict
from strategies.base_searcher import BaseSearcher
from state_manager import StateManager
from pragmatics.logger import log_event
import config_lfs as config

class GlobalLFSNode:
    """A state node for the Global LLM-First Search tree."""
    def __init__(self, state: StateManager, parent=None, action=None):
        self.state = state
        self.parent = parent
        self.action = action
        self.children = {}
        self.value = 0.0
        
        self.feasible_actions = self.state.get_feasible_actions()
        self.unexpanded_actions = self.feasible_actions.copy()
        
        self.is_terminal = len(self.feasible_actions) == 0 and all(s == 'completed' for s in self.state.job_status.values())
        self.lower_bound = self.state.calculate_lower_bound()
        self.state_hash = self.state.get_state_hash()

class GlobalLFSSearcher(BaseSearcher):
    """
    Offline Global LFS: Exhausts a massive search budget in one go to find a complete end-to-end schedule.
    """
    def __init__(self, llm_agent):
        super().__init__(llm_agent)
        self.num_iterations = getattr(config, 'GLOBAL_LFS_BUDGET', 1000) # Recommend starting at 1000 before 5000
        self.best_timeline_makespan = float('inf')
        self.best_terminal_node = None
        
        # Massive Caches for the 3-hour run
        self.prior_cache = {}
        self.value_cache = {}
        self.explore_cache = {}

    def run_search(self, initial_state: StateManager, session_folder: str, iteration: int = 1) -> List[Dict]:
        root = GlobalLFSNode(copy.deepcopy(initial_state))
        
        # Initialize Root Value
        root.value = self.llm_agent.get_value(root.state, self.current_strategic_experience, session_folder, iteration)
        self.value_cache[root.state_hash] = root.value
        
        frontier = [root]
        current_node = root
        completed_schedules = 0
        
        print(f"\n{'='*60}")
        print(f">>> [GLOBAL LFS] Launching Massive Search (Budget: {self.num_iterations})")
        print(f"{'='*60}\n")
        
        for lfs_iter in range(self.num_iterations):
            if lfs_iter % 50 == 0:
                print(f"--- Budget Burned: {lfs_iter}/{self.num_iterations} | Frontier Size: {len(frontier)} | Completed: {completed_schedules} ---")
            
            # 1. Evaluate Terminal State (The Vault)
            if current_node.is_terminal:
                completed_schedules += 1
                if current_node.state.current_time < self.best_timeline_makespan:
                    self.best_timeline_makespan = current_node.state.current_time
                    self.best_terminal_node = current_node
                    print(f"\n    [!] NEW GLOBAL BEST SCHEDULE FOUND! Makespan: {self.best_timeline_makespan:.2f} [!]\n")
                
                # Force backtrack from terminal state
                should_explore = True 
            else:
                # 2. Query LLM: Explore vs Continue (With Caching)
                if len(frontier) <= 1 and current_node == root:
                    should_explore = False 
                else:
                    if current_node.state_hash in self.explore_cache:
                        should_explore = self.explore_cache[current_node.state_hash]
                    else:
                        should_explore = self.llm_agent.get_explore_decision(
                            current_node.state, self.current_strategic_experience, session_folder, lfs_iter
                        )
                        self.explore_cache[current_node.state_hash] = should_explore
            
            # 3. Execute Traversal Decision
            if should_explore and frontier:
                current_node = max(frontier, key=lambda n: n.value)
                
            # 4. Handle Fully Expanded Nodes
            if not current_node.unexpanded_actions:
                if current_node in frontier:
                    frontier.remove(current_node)
                if frontier:
                    current_node = max(frontier, key=lambda n: n.value)
                continue
                
            # 5. Expand (Priors with Caching)
            if current_node.state_hash in self.prior_cache:
                priors = self.prior_cache[current_node.state_hash]
            else:
                try:
                    priors = self.llm_agent.get_priors(current_node.state, current_node.unexpanded_actions, self.current_strategic_experience, session_folder, lfs_iter)
                    self.prior_cache[current_node.state_hash] = priors
                except AttributeError:
                    priors = {str(i): 1.0 / len(current_node.unexpanded_actions) for i in range(len(current_node.unexpanded_actions))}
            
            try:
                best_action_idx = int(max(priors.items(), key=lambda x: x[1])[0])
                if best_action_idx >= len(current_node.unexpanded_actions):
                    best_action_idx = 0
            except:
                best_action_idx = 0
                
            selected_action = current_node.unexpanded_actions.pop(best_action_idx)
            action_key = f"{selected_action['job']}_{selected_action['op']}_{selected_action['machine']}"
            
            # Create Branch & Fast Forward
            new_state = copy.deepcopy(current_node.state)
            new_state.execute_action(selected_action["job"], selected_action["op"], selected_action["machine"])
            self._fast_forward_to_next_decision(new_state)
            
            child_node = GlobalLFSNode(new_state, parent=current_node, action=selected_action)
            
            # 6. Evaluate New State (Value with Caching)
            if child_node.state_hash in self.value_cache:
                child_node.value = self.value_cache[child_node.state_hash]
            else:
                child_node.value = self.llm_agent.get_value(child_node.state, self.current_strategic_experience, session_folder, lfs_iter)
                self.value_cache[child_node.state_hash] = child_node.value
            
            # OR Grounding
            if child_node.lower_bound > self.best_timeline_makespan:
                child_node.value = 0.0
                
            current_node.children[action_key] = child_node
            
            # Manage Frontier
            if not child_node.is_terminal and child_node.unexpanded_actions:
                frontier.append(child_node)
            if not current_node.unexpanded_actions and current_node in frontier:
                frontier.remove(current_node)
                
            current_node = child_node

        # 7. Post-Search Action Extraction
        print(f"\n>>> [GLOBAL LFS] Budget Exhausted. Total Valid Schedules Discovered: {completed_schedules}")
        if self.best_terminal_node:
            print(f">>> Extracting Optimal Sequence (Makespan: {self.best_timeline_makespan:.2f})...")
            return self._extract_action_sequence(self.best_terminal_node)
        else:
            print(">>> WARNING: Budget exhausted before finding a complete schedule. Returning best partial trajectory.")
            best_partial = max(frontier, key=lambda n: n.value)
            return self._extract_action_sequence(best_partial)

    def _extract_action_sequence(self, terminal_node: GlobalLFSNode) -> List[Dict]:
        """Walks backwards from a leaf node to the root to extract the chronological list of actions."""
        sequence = []
        current = terminal_node
        while current.parent is not None:
            sequence.insert(0, current.action)
            current = current.parent
        return sequence

    def _fast_forward_to_next_decision(self, state: StateManager):
        while not state.get_feasible_actions() and not all(s == 'completed' for s in state.job_status.values()):
            event_type, _, _ = state.process_next_event()
            if event_type is None: break