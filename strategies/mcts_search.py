import math
import copy
import os
import sys
from typing import Dict
from strategies.base_searcher import BaseSearcher
from state_manager import StateManager
from utilities.logger import log_event, log_mcts_tree
from utilities.stochastic_rollout import stochastic_rollout, random_rollout
from utilities.array_tree_state import ArrayTreeState
from utilities.tree_visualizer import draw_mcts_tree
import config

class MCTSNode:
    """Represents a single state in the MCTS tree."""
    def __init__(self, state: StateManager, parent=None, action=None, prior_prob=0.0):
        self.state = state
        self.parent = parent
        self.action = action
        self.children: Dict[str, 'MCTSNode'] = {}
        
        # AlphaZero / MCTS Trackers
        self.visits = 0               # N
        self.total_value = 0.0        # W
        self.q_value = 0.0            # Q (Average Utility: W/N), higher is better
        self.total_raw_value = 0.0    # Sum of un-normalized tardiness values
        self.raw_q_value = 0.0        # Average un-normalized tardiness (non-negative reference)
        self.prior_prob = prior_prob  # P (Policy prior from LLM)

        # Ground Truth Mathematical Bound
        self.lower_bound = self.state.calculate_lower_bound()

    def is_fully_expanded(self):
        """Checks if all feasible actions have been instantiated as child nodes."""
        feasible_actions = self.state.get_feasible_actions()
        return len(self.children) == len(feasible_actions)

    def is_terminal(self):
        """Checks if the scheduling episode is completely finished."""
        return not self.state.get_feasible_actions() and _state_all_jobs_completed(self.state)


def _state_all_jobs_completed(state) -> bool:
    if hasattr(state, "all_jobs_completed"):
        return state.all_jobs_completed()
    return all(status == "completed" for status in state.job_status.values())


def _clone_state(state):
    if hasattr(state, "clone"):
        return state.clone()
    return copy.deepcopy(state)


class MCTSSearcher(BaseSearcher):
    """
    Monte Carlo Tree Search driven by LLM Policy (Priors) and Value (Scores).
    Enforces OR rigorousness via mathematical Lower Bound pruning.
    """
    replay_buffer = []

    def __init__(self, llm_agent):
        super().__init__(llm_agent)
        self.replay_buffer = []
        self.max_iterations = getattr(config, "MCTS_MAX_ITERATIONS", 25)
        self.min_iterations = getattr(config, "MCTS_MIN_ITERATIONS", 5)
        self.max_c_param = getattr(config, "MCTS_C_PARAM_MAX", getattr(config, "MCTS_C_PARAM", 1.0))
        self.min_c_param = getattr(config, "MCTS_C_PARAM_MIN", getattr(config, "MCTS_C_PARAM", 1.0))
        self.prior_mode = str(getattr(config, "MCTS_PRIOR_MODE", "llm")).lower()
        self.rollouts_per_evaluation = getattr(config, "MCTS_ROLLOUTS_PER_EVAL", 5)
        self.rollout_policy = str(getattr(config, "MCTS_ROLLOUT_POLICY", "pdr")).lower()
        self.tree_visualization_enabled = bool(
            getattr(config, "MCTS_TREE_VISUALIZATION_ENABLED", False)
        )
        self.tree_state_backend = str(getattr(config, "MCTS_TREE_STATE_BACKEND", "array")).lower()
        self.save_rollouts_to_replay_buffer = bool(
            getattr(config, "MCTS_SAVE_ROLLOUTS_TO_REPLAY_BUFFER", False)
        )
        self.normalize_q_for_puct = bool(
            getattr(config, "MCTS_NORMALIZE_Q_FOR_PUCT", True)
        )
        
        self.best_timeline_tardiness = float('inf')
        self.zero_tardiness_tiebreak_epsilon = float(
            getattr(config, "MCTS_ZERO_TARDINESS_TIEBREAK_EPS", 1e-6)
        )
        self.prior_cache = {}

    def run_search(self, initial_state: StateManager, session_folder: str, iteration: int) -> dict:
        feasible_actions = initial_state.get_feasible_actions()
        
        if not feasible_actions:
            return None
            
        if len(feasible_actions) == 1:
            forced_action = feasible_actions[0]
            print(f">>> [MCTS] Short-circuit: Only 1 feasible action available. Skipping tree search.")
            return forced_action
        
        total_ops = sum(len(operations) for operations in initial_state.jobs.values())
        completed_ops = sum(initial_state.job_progress.values())
        progress = completed_ops / total_ops if total_ops > 0 else 1.0
        current_iterations = max(
            self.min_iterations,
            int(self.max_iterations - (self.max_iterations - self.min_iterations) * progress),
        )
        visualize_this_decision = self.tree_visualization_enabled and initial_state.current_time == 0.0

        root_state = self._build_root_tree_state(initial_state)
        root = MCTSNode(state=root_state)
        
        print(f">>> [MCTS] Budget: {current_iterations} iterations (Progress: {progress:.1%})")
        print(
            f">>> [MCTS] Starting Lookahead (Iterations: {current_iterations}, "
            f"C=[{self.max_c_param:.3f}->{self.min_c_param:.3f}], "
            f"PriorMode={self.prior_mode}, RolloutPolicy={self.rollout_policy})"
        )
        c_param_log_indices = {0, max(0, current_iterations // 2), max(0, current_iterations - 1)}
        
        for mcts_iter in range(current_iterations):
            current_c_param = self._compute_iteration_c_param(mcts_iter, current_iterations)
            if mcts_iter in c_param_log_indices:
                print(
                    f"    [MCTS] Iter {mcts_iter + 1}/{current_iterations} "
                    f"using C={current_c_param:.3f}"
                )
            # 1. SELECTION
            node = self._select(root, current_c_param)
            
            # 2. EXPANSION & EVALUATION
            if not node.is_terminal():
                node = self._expand(node, session_folder, iteration)
                
                eval_state = _clone_state(node.state)
                
                # Update our global best Upper Bound if we organically stumble on a finished schedule
                if not eval_state.get_feasible_actions() and _state_all_jobs_completed(eval_state):
                    terminal_tardiness = float(eval_state.calculate_actual_tardiness())
                    if terminal_tardiness < self.best_timeline_tardiness:
                        self.best_timeline_tardiness = terminal_tardiness
                        print(f"    -> [MCTS] New Upper Bound Found: {self.best_timeline_tardiness:.2f}")

                rollout_tardinesses = []
                rollout_samples = []
                for _ in range(self.rollouts_per_evaluation):
                    rollout_state = _clone_state(eval_state)
                    tardiness, trajectory, analytics = self._run_configured_rollout(rollout_state)
                    rollout_samples.append(
                        {
                            "tardiness": tardiness,
                            "makespan": rollout_state.current_time,
                            "trajectory": trajectory,
                            "analytics": analytics,
                        }
                    )
                    rollout_tardinesses.append(tardiness)

                if self.save_rollouts_to_replay_buffer:
                    # Keep replay memory compact: persist only extremal outcomes
                    # per evaluation batch (best and worst rollout).
                    if rollout_samples:
                        best_rollout = min(
                            rollout_samples,
                            key=lambda rollout_item: (
                                float(rollout_item.get("tardiness", float("inf"))),
                                float(rollout_item.get("makespan", float("inf"))),
                            ),
                        )
                        worst_rollout = max(
                            rollout_samples,
                            key=lambda rollout_item: (
                                float(rollout_item.get("tardiness", float("-inf"))),
                                float(rollout_item.get("makespan", float("-inf"))),
                            ),
                        )
                        self.replay_buffer.append(best_rollout)
                        if worst_rollout is not best_rollout:
                            self.replay_buffer.append(worst_rollout)
                    self.replay_buffer.sort(
                        key=lambda rollout_item: (
                            float(rollout_item.get("tardiness", float("inf"))),
                            float(rollout_item.get("makespan", float("inf"))),
                        )
                    )
                average_tardiness = (
                    sum(rollout_tardinesses) / len(rollout_tardinesses)
                    if rollout_tardinesses
                    else float(eval_state.calculate_actual_tardiness())
                )
                # PUCT in the paper maximizes W/N; convert minimization objective
                # (tardiness) into utility by negating the value.
                if average_tardiness <= 0.0:
                    average_tardiness += self.zero_tardiness_tiebreak_epsilon * float(eval_state.current_time)
                value = -average_tardiness
                raw_value = average_tardiness
            else:
                # Terminal node evaluation
                terminal_tardiness = float(node.state.calculate_actual_tardiness())
                if terminal_tardiness <= 0.0:
                    terminal_tardiness += self.zero_tardiness_tiebreak_epsilon * float(node.state.current_time)
                value = -terminal_tardiness
                raw_value = terminal_tardiness

            # 3. BACKPROPAGATION
            self._backpropagate(node, value, raw_value)

            # --- TREE VISUALIZATION INJECTION ---
            if visualize_this_decision:
                vis_path = os.path.join(session_folder, "tree_visualizations", f"loop_{mcts_iter + 1}")
                draw_mcts_tree(root, node, vis_path)
            
        # 4. ACTION EXECUTION (Choose the most robustly explored path)
        if not root.children:
            return None
        
        normalized_q_terms = self._compute_normalized_q_terms(root)
        log_mcts_tree(
            session_folder,
            iteration,
            initial_state.current_time,
            self.best_timeline_tardiness,
            root,
            normalized_q_terms=normalized_q_terms,
        )

        best_action_key = max(root.children.keys(), key=lambda k: root.children[k].visits)
        best_child = root.children[best_action_key]
        
        print(f">>> [MCTS] Lookahead complete. Selected action visits: {best_child.visits}/{current_iterations}")
        if visualize_this_decision:
            print(f">>> [VISUALIZER] Successfully generated {current_iterations} tree images.")
            sys.exit(0)

        return best_child.action

    def _select(self, node: MCTSNode, current_c_param: float) -> MCTSNode:
        """Walks down the tree using the PUCT formula until an unexpanded node is reached."""
        current = node
        while current.is_fully_expanded() and not current.is_terminal():
            next_child = self._get_best_puct_child(current, current_c_param)
            if next_child is None:
                # Defensive stop: no selectable child despite expansion status.
                # Returning current avoids crashing and lets expansion/evaluation continue.
                return current
            current = next_child
        return current
    def _expand(self, node: MCTSNode, session_folder: str, iteration: int) -> MCTSNode:
        """Generates a new child node based on feasible actions and priors."""
        feasible_actions = node.state.get_feasible_actions()
        state_hash = node.state.get_state_hash()
        uniform_priors = {
            str(index): 1.0 / len(feasible_actions)
            for index in range(len(feasible_actions))
        } if feasible_actions else {}

        if state_hash in self.prior_cache:
            priors = self.prior_cache[state_hash]
        else:
            if self.prior_mode == "uniform":
                priors = uniform_priors
                self.prior_cache[state_hash] = priors
            elif self.prior_mode == "llm":
                try:
                    priors = self.llm_agent.get_priors(
                        node.state,
                        feasible_actions,
                        self.current_strategic_experience,
                        session_folder,
                        iteration,
                    )
                    self.prior_cache[state_hash] = priors
                except AttributeError:
                    priors = uniform_priors
            else:
                priors = uniform_priors

        for idx, action in enumerate(feasible_actions):
            action_key = f"{action['job']}_{action['op']}_{action['machine']}"
            
            if action_key not in node.children:
                new_state = _clone_state(node.state)
                new_state.execute_action(action["job"], action["op"], action["machine"])
                self._fast_forward_to_next_decision(new_state)
                
                prior_prob = float(priors.get(str(idx), priors.get(idx, 1.0 / len(feasible_actions))))
                
                child_node = MCTSNode(
                    state=new_state, parent=node, action=action,
                    prior_prob=prior_prob,
                )
                node.children[action_key] = child_node
                return child_node
                
        return node 

    def _get_best_puct_child(self, node: MCTSNode, current_c_param: float) -> MCTSNode:
        """Calculates PUCT score with OR-driven Lower Bound Pruning."""
        best_score = -float('inf')
        best_child = None
        fallback_child = None
        normalized_q_terms = self._compute_normalized_q_terms(node)
        
        for child in node.children.values():
            if fallback_child is None:
                fallback_child = child
            else:
                is_lower_bound_better = child.lower_bound < fallback_child.lower_bound
                is_same_bound_higher_visits = (
                    child.lower_bound == fallback_child.lower_bound
                    and child.visits > fallback_child.visits
                )
                if is_lower_bound_better or is_same_bound_higher_visits:
                    fallback_child = child

            exploitation_term = (
                normalized_q_terms.get(child, 0.5)
                if self.normalize_q_for_puct
                else child.q_value
            )
            
            # --- THE "LOWER BOUND" PRUNING (OR Grounding) ---
            # If the mathematical floor of this timeline is already worse than 
            # our known upper bound (best completed schedule), zero out its Q-value.
            # This causes PUCT to naturally abandon the doomed timeline without wasting API calls.
            if child.lower_bound > self.best_timeline_tardiness:
                exploitation_term = -float("inf")
                
            # AlphaZero/Paper PUCT Formula:
            # a = argmax_a W(s,a)/N(s,a) + c * P(s,a) * sqrt(N(s)) / (1 + N(s,a))
            puct_score = (
                exploitation_term
                + current_c_param * child.prior_prob * math.sqrt(node.visits) / (1 + child.visits)
            )
            
            if puct_score > best_score:
                best_score = puct_score
                best_child = child

        # If all children were pruned to -inf (or selection tied at -inf),
        # return deterministic fallback: min lower_bound, then max visits.
        if best_child is None:
            return fallback_child

        return best_child

    def _backpropagate(self, node: MCTSNode, value: float, raw_value: float):
        """Propagates the evaluated value up the tree to the root."""
        current = node
        while current is not None:
            current.visits += 1
            current.total_value += value
            current.total_raw_value += raw_value
            current.q_value = current.total_value / current.visits  # Recompute average
            current.raw_q_value = max(0.0, current.total_raw_value / current.visits)
            current = current.parent

    def _compute_normalized_q_terms(self, node: MCTSNode) -> dict:
        """Min-max normalize child Q values to [0, 1] for stable PUCT scaling."""
        children = list(node.children.values())
        if not children:
            return {}

        q_values = [child.q_value for child in children]
        q_min = min(q_values)
        q_max = max(q_values)
        q_span = q_max - q_min
        if q_span <= 1e-9:
            return {child: 0.5 for child in children}

        return {
            child: (child.q_value - q_min) / q_span
            for child in children
        }

    def _run_configured_rollout(self, rollout_state):
        """Dispatch rollout behavior based on config-selected rollout policy."""
        if self.rollout_policy == "random":
            return random_rollout(rollout_state)
        if self.rollout_policy == "pdr":
            return stochastic_rollout(rollout_state)
        return stochastic_rollout(rollout_state)

    def _compute_iteration_c_param(self, mcts_iteration_index: int, total_iterations: int) -> float:
        """Linearly decay exploration factor from max to min within one MCTS search call."""
        if total_iterations <= 1:
            return self.max_c_param

        iteration_progress = mcts_iteration_index / (total_iterations - 1)
        decayed_c_param = self.max_c_param - (
            (self.max_c_param - self.min_c_param) * iteration_progress
        )
        return max(self.min_c_param, decayed_c_param)

    def _fast_forward_to_next_decision(self, state: StateManager):
        """Simulates time forward until a new action must be taken."""
        while not state.get_feasible_actions() and not _state_all_jobs_completed(state):
            event_type, _, _ = state.process_next_event()
            if event_type is None:
                break

    def _build_root_tree_state(self, initial_state: StateManager):
        if self.tree_state_backend != "array":
            return copy.deepcopy(initial_state)

        array_state = ArrayTreeState.from_state_manager(initial_state)
        if not array_state.supports_dynamic_features:
            print(">>> [MCTS] Array tree backend skipped (dynamic features present). Falling back to object deepcopy.")
            return copy.deepcopy(initial_state)
        return array_state