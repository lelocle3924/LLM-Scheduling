import json
import os
import shutil
import re
from datetime import datetime

from state_manager import StateManager
import config
from logger import setup_session_folder, log_file, log_event
from llm_inference import Sched, Reflec

def load_text_file(filepath: str) -> str:
    """Helper to load prompt templates from text files."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.read()

def parse_action_from_file(filepath: str) -> dict:
    """Helper to extract the JSON decision from a historical log file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    response_section = content.split("=== FULL LLM RESPONSE ===")[-1] if "=== FULL LLM RESPONSE ===" in content else content
    match = re.search(r'\{.*?\}', response_section, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError: pass
    return None

def main():
    total_start_time = datetime.now()
    print(">>> 1. Loading problem data and prompt templates...")
    with open(config.PROBLEM_FILE, "r") as f:
        problem_data = json.load(f)
    
    # Initialize the inference modules
    sched = Sched(load_text_file("decision_prompt.txt"))
    reflec = Reflec(load_text_file("reflection_prompt.txt")) 
    
    print(">>> 2. Initializing State Manager and Logger...")
    sm = StateManager(problem_data)
    session_folder = setup_session_folder(config.SESSION_NAME)
    
    iteration = 1
    current_strategic_experience = "Strategic Experience: Maintain flow. No dynamic events have occurred yet."
    
    # --- CHECKPOINT RESUMPTION LOGIC ---
    checkpoint_path = getattr(config, "CHECKPOINT_PATH", "")
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f">>> 3. Resuming from checkpoint: {checkpoint_path}")
        src_folder = os.path.dirname(checkpoint_path)
        checkpoint_iter = int(os.path.splitext(os.path.basename(checkpoint_path))[0])
        
        print(f">>> Fast-forwarding state to iteration {checkpoint_iter}...")
        for i in range(1, checkpoint_iter + 1):
            src_file = os.path.join(src_folder, f"{i}.txt")
            dest_file = os.path.join(session_folder, f"{i}.txt")
            if os.path.exists(src_file):
                shutil.copy2(src_file, dest_file)
                action = parse_action_from_file(src_file)
                if action:
                    while True:
                        actions = sm.get_feasible_actions()
                        if not actions:
                            # Safely unpack the tuple from the new state manager logic
                            event_type, timestamp, data = sm.process_next_event()
                            if event_type is None: 
                                break
                            # Keep the events ledger accurate during fast-forwarding
                            log_event(session_folder, timestamp, event_type, str(data))
                            continue
                        
                        is_valid = any(a["job"] == action["job"] and a["op"] == action["op"] and a["machine"] == action["machine"] for a in actions)
                        job_id, op_id, machine_id = (action["job"], action["op"], action["machine"]) if is_valid else (actions[0]["job"], actions[0]["op"], actions[0]["machine"])
                        break
                    
                    sm.execute_action(job_id, op_id, machine_id)
                    # Log the historical action executions
                    # log_event(session_folder, sm.current_time, "Action_Executed", f"Job {job_id}, Op {op_id} -> Mach {machine_id} (Fast-forwarded)")
        
        iteration = checkpoint_iter + 1
        print(f">>> State restored successfully. Clock is now at {sm.current_time}.")
    else:
        print(f">>> Starting fresh simulation. Logging to folder: '{session_folder}'")

    # --- CONTINUOUS SCHEDULING LOOP ---
    while not all(status == 'completed' for status in sm.job_status.values()):
        actions = sm.get_feasible_actions()
        
        # If no actions, fast-forward time to the next event
        if not actions:
            event_type, timestamp, data = sm.process_next_event()
            if event_type is None:
                print("\nWARNING: Deadlock detected or no more events in queue.")
                break
            
            # Log the event
            log_event(session_folder, timestamp, event_type, str(data))
            
            # TRIGGER REFLECTION ON DYNAMIC EVENTS
            if event_type in ["Machine_Breakdown", "Job_Emergency", "Job_Arrival"]:
                print(f"\n>>> [DYNAMIC EVENT] {event_type} at time {timestamp}. Triggering Reflection...")
                # Note: Currently passing dummy rollout data to the placeholder Reflec module
                new_experience = reflec.generate_reflection(rollout_data={"event": event_type, "data": data})
                if new_experience:
                    current_strategic_experience = new_experience
                    print(">>> Strategic Experience Updated.")
            continue
            
        print(f"\n--- Iteration {iteration} | Clock: {sm.current_time} ---")
        iteration_start = datetime.now()
        
        prompt_inputs = sm.compile_prompt_elements()
        prompt_inputs['strategic_experience'] = current_strategic_experience
        
        decision = sched.make_decision(prompt_inputs, actions, session_folder, iteration)
        
        if decision:
            sm.execute_action(decision["job"], decision["op"], decision["machine"])
            # log_event(session_folder, sm.current_time, "Action_Executed", f"Job {decision['job']}, Op {decision['op']} -> Mach {decision['machine']}")
            print(f">>> SUCCESS: Job {decision['job']}, Op {decision['op']} assigned to Machine {decision['machine']}.")
        else:
            fallback = actions[0]
            sm.execute_action(fallback["job"], fallback["op"], fallback["machine"])
            log_event(session_folder, sm.current_time, "Fallback_Executed", f"Job {fallback['job']}, Op {fallback['op']} -> Mach {fallback['machine']}")
        
        log_file(session_folder, iteration, f"Iteration Time: {datetime.now() - iteration_start}")
        iteration += 1

    print(f"\n>>> Simulation Complete! Final Makespan: {sm.current_time}")
    print(f"Total simulation time: {datetime.now() - total_start_time}")
    log_event(session_folder, sm.current_time, "Simulation_Complete", f"Total time: {datetime.now() - total_start_time}")


if __name__ == "__main__":
    main()