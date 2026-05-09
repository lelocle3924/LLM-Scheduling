import os
import time
import json
import csv
from datetime import datetime
import config
from utilities.numeric_precision import cap_numeric_precision, dumps_capped, format_decimal


def _get_safe_session_name(session_folder: str) -> str:
    return os.path.basename(str(session_folder).rstrip("\\/"))

def setup_session_folder(session_name: str) -> str:
    """Creates a session folder inside RESULTS_FOLDER."""
    results_root = str(getattr(config, "RESULTS_FOLDER", "results") or "results")
    session_folder = os.path.join(results_root, session_name)
    os.makedirs(session_folder, exist_ok=True)
    return session_folder

def log_event(session_folder: str, timestamp: float, event_type: str, details: str):
    """Logs environment events (breakdowns, completions, etc.) to a central ledger."""
    safe_session_name = _get_safe_session_name(session_folder)
    filepath = os.path.join(session_folder, f"events_log_{safe_session_name}.txt")
    with open(filepath, 'a', encoding='utf-8') as f:
        f.write(f"[Time: {format_decimal(timestamp):>6}] {event_type:<25} | {details}\n")

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
        handle.write(f"| Latency | {format_decimal(latency)}s |\n\n")
        handle.write(f"---\n\n")
        handle.write(f"## Prompt Sent\n\n")
        handle.write(prompt_text)
        handle.write(f"\n\n---\n\n")
        handle.write(f"## LLM Response\n\n")
        handle.write(llm_response)
        handle.write("\n")

def log_mcts_tree(
    session_folder: str,
    iteration: int,
    current_time: float,
    best_timeline_tardiness: float,
    root_node,
    normalized_q_terms: dict = None,
):
    """Logs the evaluation of the MCTS tree, specifically the root's children, bounds, and AlphaZero statistics."""
    safe_session_name = _get_safe_session_name(session_folder)
    filepath = os.path.join(session_folder, f"mcts_log_{safe_session_name}.txt")
    normalized_q_terms = normalized_q_terms or {}
    with open(filepath, 'a', encoding='utf-8') as f:
        f.write(f"\n{'='*65}\n")
        f.write(f"Search Iteration: {iteration} | Factory Clock: {format_decimal(current_time)}\n")
        f.write(f"Global Upper Bound (Best Tardiness Found): {best_timeline_tardiness}\n")
        f.write(f"Root Lower Bound: {format_decimal(root_node.lower_bound)}\n")
        f.write(f"Total Root Visits: {root_node.visits}\n")
        f.write(f"{'-'*65}\n")
        f.write(
            f"{'Action (J,O,M)':<15} | {'Visits (N)':<10} | {'Prior (P)':<10} | "
            f"{'Raw-Q':<10} | {'Norm-Q':<10} | {'Lower Bound'}\n"
        )
        f.write(f"{'-'*65}\n")
        
        for action_key, child in root_node.children.items():
            normalized_q = normalized_q_terms.get(child, 0.5)
            raw_q_value = getattr(child, "raw_q_value", max(0.0, -getattr(child, "q_value", 0.0)))
            f.write(
                f"{action_key:<15} | {child.visits:<10} | {format_decimal(child.prior_prob):<10} | "
                f"{format_decimal(raw_q_value):<10} | {format_decimal(normalized_q):<10} | {format_decimal(child.lower_bound)}\n"
            )

def log_reflection_cycle(session_folder: str, timestamp: float, event_info: str, prompt_text: str, results: list, best: dict, worst: dict, new_exp: str, raw_llm: str):
    """Logs the full Simulate-Reflect-Refine cycle to the reflections ledger."""
    safe_session_name = _get_safe_session_name(session_folder)
    filepath = os.path.join(session_folder, f"reflections_log_{safe_session_name}.md")
    with open(filepath, 'a', encoding='utf-8') as f:
        f.write(f"==================================================\n")
        f.write(f"TIME: {format_decimal(timestamp):>6} | TRIGGER: {event_info}\n")
        f.write(f"==================================================\n")
        f.write(f"[Prompt text]\n{prompt_text}\n\n")
        
        f.write(f"[ROLLOUTS EVALUATED]\n")
        for r in results:
            f.write(f"  - Action: {r['action']:<15} | Tardiness: {format_decimal(r['tardiness'])}\n")
        f.write("\n")
        
        f.write(f"[BEST PATH]\n")
        f.write(f"  Action: {best['action']}\n")
        f.write(f"  Tardiness: {format_decimal(best['tardiness'])}\n")
        f.write(f"  Trajectory: {' -> '.join(best['trajectory'][:15])}...\n\n")
        
        f.write(f"[WORST PATH]\n")
        f.write(f"  Action: {worst['action']}\n")
        f.write(f"  Tardiness: {format_decimal(worst['tardiness'])}\n")
        f.write(f"  Trajectory: {' -> '.join(worst['trajectory'][:15])}...\n\n")
        
        f.write(f"[RAW LLM ANALYSIS]\n{raw_llm}\n\n")
        
        f.write(f"[NEW STRATEGIC EXPERIENCE]\n{new_exp}\n")
        f.write(f"==================================================\n\n")

def log_skipped_reflection(session_folder: str, timestamp: float, event_info: str, reason: str):
    """Logs instances where a reflection was triggered but aborted."""
    safe_session_name = _get_safe_session_name(session_folder)
    filepath = os.path.join(session_folder, f"reflections_log_{safe_session_name}.md")
    with open(filepath, 'a', encoding='utf-8') as f:
        f.write(f"==================================================\n")
        f.write(f"TIME: {format_decimal(timestamp):>6} | TRIGGER: {event_info}\n")
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
    safe_session_name = _get_safe_session_name(session_folder)
    filepath = os.path.join(session_folder, f"lfs_log_{safe_session_name}.txt")
    
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
        val_str = f"{format_decimal(child_value)}{pruned_tag}"
        action_formatted = action_str if action_str else "Terminal/None"
        prior_str = format_decimal(prior_prob) if prior_prob is not None else "N/A"
        
        f.write(
            f"{lfs_iter:<10} | {decision:<10} | {format_decimal(current_node_value):<10} | "
            f"{frontier_size:<10} | {action_formatted:<18} | {prior_str:<8} | {val_str:<10} | "
            f"{format_decimal(child_lb):<8} | {format_decimal(global_ub)}\n"
        )

def log_lfs_summary(session_folder: str, global_iteration: int, chosen_action: dict, anticipated_value: float):
    """Logs the final execution decision made by LFS."""
    safe_session_name = _get_safe_session_name(session_folder)
    filepath = os.path.join(session_folder, f"lfs_log_{safe_session_name}.txt")
    with open(filepath, 'a', encoding='utf-8') as f:
        action_str = f"{chosen_action['job']}_{chosen_action['op']}_{chosen_action['machine']}"
        f.write(f"{'-'*110}\n")
        f.write(f">>> LFS Complete. Executing: {action_str} (Anticipated Value: {format_decimal(anticipated_value)})\n")

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
            
            f.write(f"--- Beam {i+1} | Value: {format_decimal(val)} | First Action: {fa_str} ---\n")
            
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
                r_ops.append(
                    f"J{j}O{o}: min_pt={format_decimal(min_pt)}, rem_work={format_decimal(rem_work)}, machine={mach_list}"
                )
            
            if r_ops:
                f.write(f"[Ready Operations] {', '.join(r_ops)}\n\n")
            else:
                f.write(f"[Ready Operations] None (Waiting or Finished)\n\n")


def log_final_results(
    session_folder: str,
    problem_data: dict,
    final_schedule: dict,
    total_runtime_seconds: float,
    iteration_count: int,
) -> str:
    """Persist final run summary to results_<session>.json."""
    safe_session_name = os.path.basename(session_folder.rstrip("\\/"))
    results_filepath = os.path.join(session_folder, f"results_{safe_session_name}.json")

    jobs = problem_data.get("jobs", [])
    operation_count = sum(len(job_operations) for job_operations in jobs) if isinstance(jobs, list) else 0
    machine_count = int(problem_data.get("machines", 0) or 0)
    due_dates = problem_data.get("due_dates", {})
    dynamic_events = problem_data.get("dynamic_events", [])

    results_payload = {
        "session_name": safe_session_name,
        "generated_at_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "problem_summary": {
            "machine_count": machine_count,
            "job_count": len(jobs) if isinstance(jobs, list) else 0,
            "operation_count": operation_count,
            "due_date_count": len(due_dates) if isinstance(due_dates, dict) else 0,
            "dynamic_event_count": len(dynamic_events) if isinstance(dynamic_events, list) else 0,
            "metadata": problem_data.get("metadata", {}),
        },
        "run_summary": {
            "search_strategy": final_schedule.get("search_strategy"),
            "final_tardiness": cap_numeric_precision(float(final_schedule.get("final_tardiness", 0.0) or 0.0)),
            "total_tardiness": cap_numeric_precision(float(
                final_schedule.get(
                    "total_tardiness",
                    final_schedule.get("final_tardiness", 0.0),
                )
                or 0.0
            )),
            "max_tardiness": cap_numeric_precision(float(
                final_schedule.get(
                    "max_tardiness",
                    final_schedule.get("final_tardiness", 0.0),
                )
                or 0.0
            )),
            "completed_jobs": int(final_schedule.get("completed_jobs", 0) or 0),
            "total_jobs": int(final_schedule.get("total_jobs", 0) or 0),
            "completion_ratio": (
                float(final_schedule.get("completed_jobs", 0) or 0)
                / float(final_schedule.get("total_jobs", 1) or 1)
            ),
            "broken_machines": final_schedule.get("broken_machines", []),
            "replay_buffer_size": int(final_schedule.get("replay_buffer_size", 0) or 0),
            "iteration_count": int(iteration_count),
            "total_runtime_seconds": cap_numeric_precision(float(total_runtime_seconds)),
        },
        "final_schedule": cap_numeric_precision(final_schedule),
    }

    with open(results_filepath, "w", encoding="utf-8") as results_file:
        json.dump(cap_numeric_precision(results_payload), results_file, indent=2)

    return results_filepath


def save_to_csv(filepath: str, replay_buffer_list: list) -> None:
    """Save replay buffer entries to CSV with JSON-safe fields."""
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    fieldnames = ["tardiness", "makespan", "trajectory", "analytics"]

    with open(filepath, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for rollout in replay_buffer_list:
            trajectory_value = rollout.get("trajectory", [])
            analytics_value = rollout.get("analytics", {})
            writer.writerow(
                {
                    "tardiness": cap_numeric_precision(float(rollout.get("tardiness", 0.0) or 0.0)),
                    "makespan": cap_numeric_precision(float(rollout.get("makespan", 0.0) or 0.0)),
                    "trajectory": dumps_capped(trajectory_value),
                    "analytics": dumps_capped(analytics_value),
                }
            )


def load_from_csv(filepath: str) -> list:
    """Load replay buffer entries from CSV and deserialize JSON fields."""
    if not os.path.exists(filepath):
        return []

    loaded_rollouts = []
    with open(filepath, "r", newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            try:
                trajectory = json.loads(row.get("trajectory", "[]") or "[]")
                analytics = json.loads(row.get("analytics", "{}") or "{}")
            except json.JSONDecodeError:
                trajectory = []
                analytics = {}

            loaded_rollouts.append(
                {
                    "tardiness": float(row.get("tardiness", 0.0) or 0.0),
                    "makespan": float(row.get("makespan", 0.0) or 0.0),
                    "trajectory": trajectory if isinstance(trajectory, list) else [],
                    "analytics": analytics if isinstance(analytics, dict) else {},
                }
            )

    return loaded_rollouts

