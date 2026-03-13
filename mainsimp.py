import json
import os
from datetime import datetime

from state_manager import StateManager
import config
from logger import setup_session_folder, log_event

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
        # Selects the action with the shortest processing time (SPT heuristic)
        decision = min(actions, key=lambda x: x["processing_time"])
        
        # Execute the chosen action
        sm.execute_action(decision["job"], decision["op"], decision["machine"])
        log_event(session_folder, sm.current_time, "Action_Executed", f"Job {decision['job']}, Op {decision['op']} -> Mach {decision['machine']}")
        
        print(f">>> GREEDY DECISION: Job {decision['job']}, Op {decision['op']} assigned to Mach {decision['machine']} (Processing Time: {decision['processing_time']}).")
        
        iteration += 1

    print(f"\n>>> Simulation Complete! Final Makespan: {sm.current_time}")
    print(f"Total simulation time: {datetime.now() - total_start_time}")
    
    # Log the final completion time
    log_event(session_folder, sm.current_time, "Simulation_Complete", f"Total time: {datetime.now() - total_start_time}")

if __name__ == "__main__":
    main()