import copy
from strategies.base_searcher import BaseSearcher
from state_manager import StateManager
from utilities.logger import log_lfs_step, log_lfs_summary
import config

class LFSNode:
    """A state node for the LLM-First Search tree."""
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

class LFSSearcher(BaseSearcher):
    """
    LLM-First Search (LFS). 
    The LLM explicitly decides whether to delve deeper into a timeline or backtrack to the frontier.
    """
    def __init__(self, llm_agent):
        super().__init__(llm_agent)
        self.num_iterations = getattr(config, 'LFS_ITERATIONS', 15)
        self.best_timeline_makespan = float('inf')
        
    def run_search(self, initial_state: StateManager, session_folder: str, iteration: int) -> dict:
        feasible_actions = initial_state.get_feasible_actions()
        
        # --- Short-Circuit Optimization ---
        if not feasible_actions:
            return None
        if len(feasible_actions) == 1:
            print(f">>> [LFS] Short-circuit: Only 1 feasible action available. Skipping tree search.")
            return feasible_actions[0]

        # Initialize Root & Frontier
        root = LFSNode(copy.deepcopy(initial_state))
        root.value = self.llm_agent.get_value(root.state, self.current_strategic_experience, session_folder, iteration)
        
        frontier = [root]
        current_node = root
        
        print(f">>> [LFS] Starting LLM-First Search (Iterations: {self.num_iterations})")
        
        for lfs_iter in range(self.num_iterations):
            
            # 1. Evaluate Terminal State
            if current_node.is_terminal:
                if current_node.state.current_time < self.best_timeline_makespan:
                    self.best_timeline_makespan = current_node.state.current_time
                    print(f"    -> [LFS] New Upper Bound Found: {self.best_timeline_makespan:.2f}")
                should_explore = True 
                decision_str = "TERMINAL" # <--- 2. Track decision for log
                
            else:
                # 2. Query LLM: Explore (Backtrack) or Continue?
                if len(frontier) <= 1 and current_node == root:
                    should_explore = False 
                    decision_str = "ROOT_CONT" # <--- 2. Track decision for log
                else:
                    should_explore = self.llm_agent.get_explore_decision(
                        current_node.state, 
                        self.current_strategic_experience, 
                        session_folder, 
                        iteration
                    )
                    decision_str = "EXPLORE" if should_explore else "CONTINUE" # <--- 2. Track decision for log
            
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
                
            # 5. Expand (Using Priors to select the smartest unexpanded branch)
            heatmap = self.generate_heatmap(
                current_node.state, session_folder, iteration,
                search_context="LFS_Expand",
            )
            if heatmap is not None:
                sample_weights = list(heatmap.node_weights.items())[:5]
                print(f"    [DIFFUSION] LFS heatmap ({len(heatmap.node_weights)} nodes). Sample: {sample_weights}")

            priors = self.llm_agent.get_priors(current_node.state, current_node.unexpanded_actions, self.current_strategic_experience, session_folder, iteration)
            
            # <--- 3. EXTRACT PRIOR VALUE FOR THE LOG --->
            try:
                best_action_key_str = max(priors.items(), key=lambda x: x[1])[0]
                best_action_idx = int(best_action_key_str)
                if best_action_idx >= len(current_node.unexpanded_actions):
                    best_action_idx = 0
                prior_val = priors.get(best_action_key_str, 0.0)
            except:
                best_action_idx = 0
                prior_val = 0.0
                
            selected_action = current_node.unexpanded_actions.pop(best_action_idx)
            action_key = f"{selected_action['job']}_{selected_action['op']}_{selected_action['machine']}"
            
            # Create the new reality branch
            new_state = copy.deepcopy(current_node.state)
            new_state.execute_action(selected_action["job"], selected_action["op"], selected_action["machine"])
            self._fast_forward_to_next_decision(new_state)
            
            child_node = LFSNode(new_state, parent=current_node, action=selected_action)
            
            # Evaluate the new state's Value
            child_node.value = self.llm_agent.get_value(child_node.state, self.current_strategic_experience, session_folder, iteration)
            
            # OR Grounding (Mathematical Pruning)
            if child_node.lower_bound > self.best_timeline_makespan:
                child_node.value = 0.0
                
            current_node.children[action_key] = child_node
            
            # Manage Frontier
            if not child_node.is_terminal and child_node.unexpanded_actions:
                frontier.append(child_node)
                
            if not current_node.unexpanded_actions and current_node in frontier:
                frontier.remove(current_node)

            # <--- 4. TRIGGER THE STEP LOG --->
            log_lfs_step(
                session_folder=session_folder,
                global_iteration=iteration,
                lfs_iter=lfs_iter,
                decision=decision_str,
                current_node_value=current_node.value,
                frontier_size=len(frontier),
                action_str=action_key,
                prior_prob=prior_val,
                child_value=child_node.value,
                child_lb=child_node.lower_bound,
                global_ub=self.best_timeline_makespan
            )
            
            # Move traversal pointer down
            current_node = child_node
            
        # 6. Final Execution Decision
        if not root.children:
            return feasible_actions[0]
            
        best_action_key = max(root.children.keys(), key=lambda k: root.children[k].value)
        best_child = root.children[best_action_key]
        
        # <--- 5. TRIGGER THE SUMMARY LOG --->
        log_lfs_summary(session_folder, iteration, best_child.action, best_child.value)

        print(f">>> [LFS] Lookahead complete. Selected action with anticipated Value: {best_child.value:.3f}")
        return best_child.action

    def _fast_forward_to_next_decision(self, state: StateManager):
        while not state.get_feasible_actions() and not all(s == 'completed' for s in state.job_status.values()):
            event_type, _, _ = state.process_next_event()
            if event_type is None:
                break