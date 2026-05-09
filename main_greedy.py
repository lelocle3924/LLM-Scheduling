import argparse
import json
import os
from datetime import datetime

from state_manager import StateManager
import config
from simple_gantt import generate_gantt_from_events_log
from utilities.logger import setup_session_folder, log_event, log_final_results


def main():
    total_start_time = datetime.now()
    print(">>> 1. Loading problem data...")
    with open(config.PROBLEM_FILE, "r") as f:
        problem_data = json.load(f)
    
    print(">>> 2. Initializing State Manager and Logger...")
    sm = StateManager(problem_data)
    
    # We append "_greedy" to the session name so it doesn't overwrite your LLM logs
    session_folder = setup_session_folder(config.SESSION_NAME + "_greedy")
    print(f">>> Starting greedy simulation. Logging to folder: '{session_folder}'")

    iteration = 1
    
    # --- CONTINUOUS SCHEDULING LOOP ---
    while not all(status == 'completed' for status in sm.job_status.values()):
        actions = sm.get_feasible_actions()
        
        # If no actions, fast-forward time to the next event in the queue
        if not actions:
            event_type, timestamp, data = sm.process_next_event()
            if event_type is None:
                print("\nWARNING: Deadlock detected or no more events in queue.")
                break
            
            # Log the dynamic event to the events_log
            log_event(session_folder, timestamp, event_type, str(data))
            
            if event_type in ["Machine_Breakdown", "Job_Emergency", "Job_Arrival"]:
                print(f"\n>>> [DYNAMIC EVENT] {event_type} at time {timestamp}.")
            continue
            
        print(f"\n--- Iteration {iteration} | Clock: {sm.current_time} ---")
        
        # --- GREEDY DECISION LOGIC ---
        # Use minimum slack time with shortest processing-time tie-break.
        def get_slack(action):
            due_date = action.get("due_date")
            if due_date is None:
                return float("inf")
            rem_work = sm._calculate_rem_work(action["job"])
            return due_date - sm.current_time - rem_work

        decision = min(
            actions,
            key=lambda action: (
                get_slack(action),
                action["processing_time"],
            ),
        )
        
        # Execute the chosen action
        sm.execute_action(decision["job"], decision["op"], decision["machine"])
        log_event(session_folder, sm.current_time, "Action_Executed", f"Job {decision['job']}, Op {decision['op']} -> Mach {decision['machine']}")
        
        print(f">>> GREEDY DECISION: Job {decision['job']}, Op {decision['op']} assigned to Mach {decision['machine']} (Processing Time: {decision['processing_time']}).")
        
        iteration += 1

    total_runtime_seconds = (datetime.now() - total_start_time).total_seconds()
    tardiness_metrics = sm.calculate_tardiness_metrics()
    total_tardiness = float(tardiness_metrics["total_tardiness"])
    max_tardiness = float(tardiness_metrics["max_tardiness"])
    final_tardiness = float(sm.calculate_actual_tardiness())
    print(f"\n>>> Simulation Complete! Final Tardiness: {final_tardiness:.2f}")
    print(f"Total simulation time: {datetime.now() - total_start_time}")
    
    # Log the final completion time
    log_event(session_folder, sm.current_time, "Simulation_Complete", f"Total time: {datetime.now() - total_start_time}")

    machine_available_times = {
        str(machine_id): float(available_time)
        for machine_id, available_time in sm.machine_avail.items()
    }
    completed_jobs = sum(1 for status in sm.job_status.values() if status == "completed")
    total_jobs = len(sm.job_status)
    final_schedule = {
        "final_tardiness": final_tardiness,
        "total_tardiness": total_tardiness,
        "max_tardiness": max_tardiness,
        "completed_jobs": completed_jobs,
        "total_jobs": total_jobs,
        "machine_available_times": machine_available_times,
        "broken_machines": sorted(list(sm.broken_machines)),
        "search_strategy": "GreedyMSTEDD",
        "replay_buffer_size": 0,
    }
    results_path = log_final_results(
        session_folder=session_folder,
        problem_data=problem_data,
        final_schedule=final_schedule,
        total_runtime_seconds=total_runtime_seconds,
        iteration_count=max(iteration - 1, 0),
    )
    print(f">>> Saved results file: {results_path}")

    if config.AUTO_GENERATE_GANTT:
        safe_session_name = os.path.basename(session_folder.rstrip("\\/"))
        events_log_filename = f"events_log_{safe_session_name}.txt"
        events_log_path = os.path.join(session_folder, events_log_filename)
        output_gantt_path = os.path.join(session_folder, f"final_gantt_{safe_session_name}.png")

        print(f">>> 3. Auto-generating Gantt chart from: {events_log_path}")
        try:
            generate_gantt_from_events_log(events_log_path, output_gantt_path)
        except (OSError, ValueError) as error:
            print(f"WARNING: Failed to generate Gantt chart. Reason: {error}")


def run_batch_folder(batch_folder_path: str) -> None:
    """Run greedy solver for every JSON instance in a folder."""
    normalized_batch_folder = os.path.normpath(batch_folder_path)
    if not os.path.isdir(normalized_batch_folder):
        raise ValueError(f"--batch-folder is not a valid directory: {normalized_batch_folder}")

    json_file_paths = sorted(
        os.path.join(normalized_batch_folder, file_name)
        for file_name in os.listdir(normalized_batch_folder)
        if file_name.lower().endswith(".json")
    )
    if not json_file_paths:
        raise ValueError(f"No JSON files found in batch folder: {normalized_batch_folder}")

    base_session_name = str(getattr(config, "SESSION_NAME", "session")).strip() or "session"
    print(f">>> Batch mode: solving {len(json_file_paths)} instance(s) from {normalized_batch_folder}")
    for file_index, instance_path in enumerate(json_file_paths, start=1):
        instance_name = os.path.splitext(os.path.basename(instance_path))[0]
        config.PROBLEM_FILE = instance_path
        config.SESSION_NAME = f"{base_session_name}_{instance_name}"
        print(f"\n=== [{file_index}/{len(json_file_paths)}] {instance_path} ===")
        print(f">>> Session: {config.SESSION_NAME}")
        main()


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run greedy DFJSP solver on one instance or a folder of instances.",
    )
    parser.add_argument(
        "--batch-folder",
        default="",
        help="Folder containing JSON instances to solve in batch.",
    )
    return parser


if __name__ == "__main__":
    arguments = _build_argument_parser().parse_args()
    if arguments.batch_folder:
        run_batch_folder(arguments.batch_folder)
    else:
        main()