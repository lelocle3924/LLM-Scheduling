import math
import copy
from typing import Dict, Optional
from strategies.base_searcher import BaseSearcher
from state_manager import StateManager
from models.graph_diffusion import Heatmap
from utilities.logger import log_event, log_mcts_tree
import config

class MCTSNode:
    """Represents a single state in the MCTS tree."""
    def __init__(self, state: StateManager, parent=None, action=None, prior_prob=0.0, diffusion_heatmap: Optional[Heatmap] = None):
        self.state = state
        self.parent = parent
        self.action = action
        self.children: Dict[str, 'MCTSNode'] = {}
        
        # AlphaZero / MCTS Trackers
        self.visits = 0               # N
        self.total_value = 0.0        # W
        self.q_value = 0.0            # Q (Average Value: W/N)
        self.prior_prob = prior_prob  # P (Policy prior from LLM)
        
        # Auxiliary diffusion signal (None when pipeline is disabled)
        self.diffusion_heatmap = diffusion_heatmap
        
        # Ground Truth Mathematical Bound
        self.lower_bound = self.state.calculate_lower_bound()

    def is_fully_expanded(self):
        """Checks if all feasible actions have been instantiated as child nodes."""
        feasible_actions = self.state.get_feasible_actions()
        return len(self.children) == len(feasible_actions)

    def is_terminal(self):
        """Checks if the scheduling episode is completely finished."""
        return not self.state.get_feasible_actions() and all(s == 'completed' for s in self.state.job_status.values())


class MCTSSearcher(BaseSearcher):
    """
    Monte Carlo Tree Search driven by LLM Policy (Priors) and Value (Scores).
    Enforces OR rigorousness via mathematical Lower Bound pruning.
    """
    def __init__(self, llm_agent):
        super().__init__(llm_agent)
        self.num_iterations = getattr(config, 'MCTS_ITERATIONS', 10)
        self.c_param = getattr(config, 'MCTS_C_PARAM', 1.0)
        self.use_diffusion_as_prior = getattr(config, 'USE_DIFFUSION_AS_PRIOR', False)
        
        self.best_timeline_makespan = float('inf') 
        self.prior_cache = {}

    def run_search(self, initial_state: StateManager, session_folder: str, iteration: int) -> dict:
        feasible_actions = initial_state.get_feasible_actions()
        
        if not feasible_actions:
            return None
            
        if len(feasible_actions) == 1:
            forced_action = feasible_actions[0]
            print(f">>> [MCTS] Short-circuit: Only 1 feasible action available. Skipping tree search.")
            return forced_action
        
        root = MCTSNode(state=copy.deepcopy(initial_state))
        
        print(f">>> [MCTS] Starting Lookahead (Iterations: {self.num_iterations}, C={self.c_param})")
        
        for mcts_iter in range(self.num_iterations):
            # 1. SELECTION
            node = self._select(root)
            
            # 2. EXPANSION & EVALUATION
            if not node.is_terminal():
                node = self._expand(node, session_folder, iteration)
                

                eval_state = copy.deepcopy(node.state)
                
                # Update our global best Upper Bound if we organically stumble on a finished schedule
                if not eval_state.get_feasible_actions() and all(s == 'completed' for s in eval_state.job_status.values()):
                    if eval_state.current_time < self.best_timeline_makespan:
                        self.best_timeline_makespan = eval_state.current_time
                        print(f"    -> [MCTS] New Upper Bound Found: {self.best_timeline_makespan:.2f}")

                # Query the LLM Value Network (grounded by the LB in the prompt)
                value = self.llm_agent.get_value(eval_state, self.current_strategic_experience, session_folder, iteration)
            else:
                # Terminal node evaluation
                value = 1.0 if node.state.current_time <= self.best_timeline_makespan else 0.0

            # 3. BACKPROPAGATION
            self._backpropagate(node, value)
            
        # 4. ACTION EXECUTION (Choose the most robustly explored path)
        if not root.children:
            return None
        
        log_mcts_tree(session_folder, iteration, initial_state.current_time, self.best_timeline_makespan, root)

        best_action_key = max(root.children.keys(), key=lambda k: root.children[k].visits)
        best_child = root.children[best_action_key]
        
        print(f">>> [MCTS] Lookahead complete. Selected action visits: {best_child.visits}/{self.num_iterations}")
        return best_child.action

    def _select(self, node: MCTSNode) -> MCTSNode:
        """Walks down the tree using the PUCT formula until an unexpanded node is reached."""
        current = node
        while current.is_fully_expanded() and not current.is_terminal():
            current = self._get_best_puct_child(current)
        return current
    def _expand(self, node: MCTSNode, session_folder: str, iteration: int) -> MCTSNode:
        """Generates a new child node based on feasible actions and LLM or diffusion priors."""
        feasible_actions = node.state.get_feasible_actions()
        state_hash = node.state.get_state_hash()
        heatmap = None

        if state_hash in self.prior_cache:
            priors = self.prior_cache[state_hash]
        elif self.use_diffusion_as_prior:
            heatmap = self.generate_heatmap(
                node.state, session_folder, iteration,
                search_context="MCTS_Expand",
            )
            priors = self._heatmap_to_priors(heatmap, feasible_actions)
            self.prior_cache[state_hash] = priors
            sample = list(heatmap.node_weights.items())[:5] if heatmap else []
            print(f"    [DIFFUSION->PRIOR] Heatmap-derived priors for {len(feasible_actions)} actions. Sample nodes: {sample}")
        else:
            try:
                priors = self.llm_agent.get_priors(node.state, feasible_actions, self.current_strategic_experience, session_folder, iteration)
                self.prior_cache[state_hash] = priors
            except AttributeError:
                priors = {str(i): 1.0 / len(feasible_actions) for i in range(len(feasible_actions))}

            heatmap = self.generate_heatmap(
                node.state, session_folder, iteration,
                search_context="MCTS_Expand",
            )
            if heatmap is not None:
                sample = list(heatmap.node_weights.items())[:5]
                print(f"    [DIFFUSION] Heatmap generated ({len(heatmap.node_weights)} nodes). Sample: {sample}")

        for idx, action in enumerate(feasible_actions):
            action_key = f"{action['job']}_{action['op']}_{action['machine']}"
            
            if action_key not in node.children:
                new_state = copy.deepcopy(node.state)
                new_state.execute_action(action["job"], action["op"], action["machine"])
                self._fast_forward_to_next_decision(new_state)
                
                prior_prob = float(priors.get(str(idx), priors.get(idx, 1.0 / len(feasible_actions))))
                
                child_node = MCTSNode(
                    state=new_state, parent=node, action=action,
                    prior_prob=prior_prob, diffusion_heatmap=heatmap,
                )
                node.children[action_key] = child_node
                return child_node
                
        return node 

    @staticmethod
    def _heatmap_to_priors(heatmap: Heatmap, feasible_actions: list) -> Dict[str, float]:
        """Convert heatmap node weights into a normalized prior distribution over actions.

        Each action targets operation O_{job}_{op}.  We look up its heatmap
        weight and normalise across all feasible actions so the values sum to 1.
        """
        if heatmap is None:
            uniform = 1.0 / max(len(feasible_actions), 1)
            return {str(i): uniform for i in range(len(feasible_actions))}

        raw: Dict[str, float] = {}
        for idx, action in enumerate(feasible_actions):
            node_id = f"O_{action['job']}_{action['op']}"
            weight = heatmap.get_node_weight(node_id, default=0.5)
            raw[str(idx)] = max(weight, 0.001)

        total = sum(raw.values())
        return {key: value / total for key, value in raw.items()}

    def _get_best_puct_child(self, node: MCTSNode) -> MCTSNode:
        """Calculates PUCT score with OR-driven Lower Bound Pruning."""
        best_score = -float('inf')
        best_child = None
        
        for child in node.children.values():
            eff_q = child.q_value
            
            # --- THE "LOWER BOUND" PRUNING (OR Grounding) ---
            # If the mathematical floor of this timeline is already worse than 
            # our known upper bound (best completed schedule), zero out its Q-value.
            # This causes PUCT to naturally abandon the doomed timeline without wasting API calls.
            if child.lower_bound > self.best_timeline_makespan:
                eff_q = 0.0 
                
            # AlphaZero PUCT Formula
            puct_score = eff_q + self.c_param * child.prior_prob * math.sqrt(node.visits) / (1 + child.visits)
            
            if puct_score > best_score:
                best_score = puct_score
                best_child = child
                
        return best_child

    def _backpropagate(self, node: MCTSNode, value: float):
        """Propagates the evaluated value up the tree to the root."""
        current = node
        while current is not None:
            current.visits += 1
            current.total_value += value
            current.q_value = current.total_value / current.visits  # Recompute average
            current = current.parent

    def _fast_forward_to_next_decision(self, state: StateManager):
        """Simulates time forward until a new action must be taken."""
        while not state.get_feasible_actions() and not all(s == 'completed' for s in state.job_status.values()):
            event_type, _, _ = state.process_next_event()
            if event_type is None:
                break