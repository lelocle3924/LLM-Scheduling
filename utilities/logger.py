import json
import os
import time

def setup_session_folder(session_name: str) -> str:
    """Creates a folder for the session if it doesn't already exist."""
    os.makedirs(session_name, exist_ok=True)
    return session_name

def log_event(session_folder: str, timestamp: float, event_type: str, details: str):
    """Logs environment events (breakdowns, completions, etc.) to a central ledger."""
    filepath = os.path.join(session_folder, f"events_log_{session_folder}.txt")
    with open(filepath, 'a', encoding='utf-8') as f:
        f.write(f"[Time: {timestamp:>6.2f}] {event_type:<25} | {details}\n")

def log_llm_call(
    session_folder: str,
    iteration: int,
    call_type: str,
    model_name: str,
    prompt_text: str,
    llm_response: str,
    latency: float,
):
    """Log the full prompt sent to the LLM and its response as a markdown file.

    Each call produces one ``.md`` file inside ``<session>/interactions/``.
    The file contains the exact prompt text that was sent (not the
    decomposed template variables) followed by the raw LLM output.
    """
    interactions_path = os.path.join(session_folder, "interactions")
    os.makedirs(interactions_path, exist_ok=True)

    timestamp_str = time.strftime("%H%M%S")
    filepath = os.path.join(
        interactions_path,
        f"iter{iteration}_{call_type}_{timestamp_str}.md",
    )

    with open(filepath, "w", encoding="utf-8") as handle:
        handle.write(f"# LLM Call: {call_type}\n\n")
        handle.write(f"| Field | Value |\n")
        handle.write(f"|-------|-------|\n")
        handle.write(f"| Iteration | {iteration} |\n")
        handle.write(f"| Model | `{model_name}` |\n")
        handle.write(f"| Latency | {latency:.2f}s |\n\n")
        handle.write(f"---\n\n")
        handle.write(f"## Prompt Sent\n\n")
        handle.write(prompt_text)
        handle.write(f"\n\n---\n\n")
        handle.write(f"## LLM Response\n\n")
        handle.write(llm_response)
        handle.write("\n")

def log_mcts_tree(session_folder: str, iteration: int, current_time: float, best_timeline_makespan: float, root_node):
    """Logs the evaluation of the MCTS tree, specifically the root's children, bounds, and AlphaZero statistics."""
    filepath = os.path.join(session_folder, f"mcts_log_{session_folder}.txt")
    with open(filepath, 'a', encoding='utf-8') as f:
        f.write(f"\n{'='*65}\n")
        f.write(f"Search Iteration: {iteration} | Factory Clock: {current_time:.2f}\n")
        f.write(f"Global Upper Bound (Best Found): {best_timeline_makespan}\n")
        f.write(f"Root Lower Bound: {root_node.lower_bound:.2f}\n")
        f.write(f"Total Root Visits: {root_node.visits}\n")
        f.write(f"{'-'*65}\n")
        f.write(f"{'Action (J,O,M)':<15} | {'Visits (N)':<10} | {'Prior (P)':<10} | {'Q-Value':<10} | {'Lower Bound'}\n")
        f.write(f"{'-'*65}\n")
        
        for action_key, child in root_node.children.items():
            f.write(f"{action_key:<15} | {child.visits:<10} | {child.prior_prob:<10.3f} | {child.q_value:<10.3f} | {child.lower_bound:.2f}\n")

def log_reflection_cycle(session_folder: str, timestamp: float, event_info: str, prompt_text: str, results: list, best: dict, worst: dict, new_exp: str, raw_llm: str):
    """Logs the full Simulate-Reflect-Refine cycle to the reflections ledger."""
    filepath = os.path.join(session_folder, f"reflections_log_{session_folder}.md")
    with open(filepath, 'a', encoding='utf-8') as f:
        f.write(f"==================================================\n")
        f.write(f"TIME: {timestamp:>6.2f} | TRIGGER: {event_info}\n")
        f.write(f"==================================================\n")
        f.write(f"[Prompt text]\n{prompt_text}\n\n")
        
        f.write(f"[ROLLOUTS EVALUATED]\n")
        for r in results:
            f.write(f"  - Action: {r['action']:<15} | Makespan: {r['makespan']:.2f}\n")
        f.write("\n")
        
        f.write(f"[BEST PATH]\n")
        f.write(f"  Action: {best['action']}\n")
        f.write(f"  Makespan: {best['makespan']:.2f}\n")
        f.write(f"  Trajectory: {' -> '.join(best['trajectory'][:15])}...\n\n")
        
        f.write(f"[WORST PATH]\n")
        f.write(f"  Action: {worst['action']}\n")
        f.write(f"  Makespan: {worst['makespan']:.2f}\n")
        f.write(f"  Trajectory: {' -> '.join(worst['trajectory'][:15])}...\n\n")
        
        f.write(f"[RAW LLM ANALYSIS]\n{raw_llm}\n\n")
        
        f.write(f"[NEW STRATEGIC EXPERIENCE]\n{new_exp}\n")
        f.write(f"==================================================\n\n")

def log_skipped_reflection(session_folder: str, timestamp: float, event_info: str, reason: str):
    """Logs instances where a reflection was triggered but aborted."""
    filepath = os.path.join(session_folder, f"reflections_log_{session_folder}.md")
    with open(filepath, 'a', encoding='utf-8') as f:
        f.write(f"==================================================\n")
        f.write(f"TIME: {timestamp:>6.2f} | TRIGGER: {event_info}\n")
        f.write(f"SKIPPED: {reason}\n")
        f.write(f"==================================================\n\n")

def log_lfs_step(session_folder: str, global_iteration: int, lfs_iter: int, 
                 decision: str, current_node_value: float, frontier_size: int, 
                 action_str: str, prior_prob: float, child_value: float, 
                 child_lb: float, global_ub: float):
    """
    Logs a single step of the LLM-First Search (LFS) process, tracking the 
    LLM's semantic navigation and mathematical bounding.
    """
    filepath = os.path.join(session_folder, f"lfs_log_{session_folder}.txt")
    
    with open(filepath, 'a', encoding='utf-8') as f:
        # If it's the first LFS iteration of this global decision step, print a header
        if lfs_iter == 0:
            f.write(f"\n{'='*110}\n")
            f.write(f"GLOBAL DECISION ITERATION: {global_iteration}\n")
            f.write(f"{'='*110}\n")
            f.write(f"{'LFS Iter':<10} | {'Decision':<10} | {'Node Val':<10} | {'Frontier':<10} | {'Action Expanded':<18} | {'Prior':<8} | {'New Val':<10} | {'New LB':<8} | {'Global UB'}\n")
            f.write(f"{'-'*110}\n")
            
        # Format Pruning tags if the LB exceeded the UB
        pruned_tag = " (PRUNED)" if child_lb > global_ub else ""
        val_str = f"{child_value:.3f}{pruned_tag}"
        action_formatted = action_str if action_str else "Terminal/None"
        prior_str = f"{prior_prob:.3f}" if prior_prob is not None else "N/A"
        
        f.write(f"{lfs_iter:<10} | {decision:<10} | {current_node_value:<10.3f} | {frontier_size:<10} | {action_formatted:<18} | {prior_str:<8} | {val_str:<10} | {child_lb:<8.2f} | {global_ub:.2f}\n")

def log_lfs_summary(session_folder: str, global_iteration: int, chosen_action: dict, anticipated_value: float):
    """Logs the final execution decision made by LFS."""
    filepath = os.path.join(session_folder, f"lfs_log_{session_folder}.txt")
    with open(filepath, 'a', encoding='utf-8') as f:
        action_str = f"{chosen_action['job']}_{chosen_action['op']}_{chosen_action['machine']}"
        f.write(f"{'-'*110}\n")
        f.write(f">>> LFS Complete. Executing: {action_str} (Anticipated Value: {anticipated_value:.3f})\n")

def log_beam_search_step(session_folder: str, iteration: int, timestamp: float, beam_depth: int, strategic_experience: str, beams: list):
    """Logs the detailed state of the Beam Search at a specific depth level."""
    filepath = os.path.join(session_folder, f"beam_search_log.txt")
    
    with open(filepath, 'a', encoding='utf-8') as f:
        f.write(f"=== Iteration {iteration} | Clock: {timestamp} | Depth: {beam_depth} ===\n")
        
        # Log Strategic Experience exactly once per step
        f.write(f"[Strategic Experience]\n{strategic_experience}\n\n")
            
        for i, beam in enumerate(beams):
            state = beam["state"]
            val = beam["value"]
            first_action = beam["first_action"]
            fa_str = f"J{first_action['job']}O{first_action['op']}->M{first_action['machine']}" if first_action else "None"
            
            f.write(f"--- Beam {i+1} | Value: {val:.2f} | First Action: {fa_str} ---\n")
            
            # --- Format Machine States ---
            # e.g., M0:J1O2, M1: free, M2: broken
            m_states = []
            for m in range(state.num_machines):
                if m in state.broken_machines:
                    m_states.append(f"M{m}: broken")
                elif state.machine_current_op[m]:
                    queue = state.machine_current_op[m]
                    ops_str = ",".join(f"J{j}O{o}" for j, o in queue)
                    m_states.append(f"M{m}:{ops_str}")
                else:
                    m_states.append(f"M{m}: free")
            f.write(f"[Machine States] {', '.join(m_states)}\n")
            
            # --- Format Ready Operations ---
            # Group feasible actions by (job, op) to get the machine list and min_pt
            actions = state.get_feasible_actions()
            op_map = {}
            for a in actions:
                key = (a["job"], a["op"])
                if key not in op_map:
                    op_map[key] = {
                        "machines": [], 
                        "pts": [], 
                        "rem_work": state._calculate_rem_work(a["job"])
                    }
                op_map[key]["machines"].append(a["machine"])
                op_map[key]["pts"].append(a["processing_time"])
            
            r_ops = []
            for (j, o), data in op_map.items():
                min_pt = min(data["pts"])
                mach_list = data["machines"]
                rem_work = data["rem_work"]
                r_ops.append(f"J{j}O{o}: min_pt={min_pt}, rem_work={rem_work:.1f}, machine={mach_list}")
            
            if r_ops:
                f.write(f"[Ready Operations] {', '.join(r_ops)}\n\n")
            else:
                f.write(f"[Ready Operations] None (Waiting or Finished)\n\n")


def log_diffusion_heatmap(
    session_folder: str,
    iteration: int,
    current_time: float,
    graph_summary: dict,
    heatmap_snapshot: dict,
    search_context: str = "General",
):
    """Log the diffusion heatmap alongside the graph summary for auditing.

    Parameters
    ----------
    session_folder : str
        Active session directory.
    iteration : int
        Global planning iteration.
    current_time : float
        Factory clock when the heatmap was generated.
    graph_summary : dict
        Lightweight stats from ``summarize_graph``.
    heatmap_snapshot : dict
        Output of ``Heatmap.to_dict()`` (may be truncated for large graphs).
    search_context : str
        Which search phase triggered this (e.g. "MCTS_Expand", "LFS_Expand").
    """
    filepath = os.path.join(session_folder, f"diffusion_heatmap_log.jsonl")

    record = {
        "iteration": iteration,
        "time": round(current_time, 2),
        "context": search_context,
        "graph": graph_summary,
        "heatmap_nodes_sample": _truncate_dict(heatmap_snapshot.get("node_weights", {}), max_keys=20),
        "heatmap_edges_sample": _truncate_dict(heatmap_snapshot.get("edge_weights", {}), max_keys=20),
    }

    with open(filepath, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def _truncate_dict(data: dict, max_keys: int = 20) -> dict:
    """Keep only the first ``max_keys`` entries to avoid gigantic log lines."""
    items = list(data.items())[:max_keys]
    truncated = dict(items)
    if len(data) > max_keys:
        truncated["__truncated__"] = f"{len(data) - max_keys} more entries"
    return truncated