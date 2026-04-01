import os
import json
from datetime import datetime

from state_manager import StateManager
from llm_agent import LLMAgent
from strategies.global_lfs import GlobalLFSSearcher
from pragmatics.logger import setup_session_folder, log_event
import config_lfs as config

def load_text_file(filepath: str) -> str:
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.read()

def main():
    total_start_time = datetime.now()
    
    print(">>> 1. Loading Problem Data...")
    with open(config.PROBLEM_FILE, "r") as f:
        problem_data = json.load(f)
    
    # Initialize Reality (The Ground Truth State)
    sm = StateManager(problem_data)
    session_folder = setup_session_folder(f"{config.SESSION_NAME}_GLOBAL_LFS")
    
    print(">>> 2. Initializing LLM Brains...")
    action_template = load_text_file(config.ACTION_PROMPT_FILE)
    value_template = load_text_file(config.VALUE_PROMPT_FILE)
    prior_template = load_text_file(getattr(config, 'PRIOR_PROMPT_FILE', 'prompts/prior_prompt.md'))
    explore_template = load_text_file(getattr(config, 'EXPLORE_PROMPT_FILE', 'prompts/explore_prompt.md'))
    
    agent = LLMAgent(
        action_prompt_template=action_template, 
        value_prompt_template=value_template,
        prior_prompt_template=prior_template,
        explore_prompt_template=explore_template
    )
    
    # Initialize the Global Searcher
    print(">>> 3. Handing control to Global LFS...")
    searcher = GlobalLFSSearcher(llm_agent=agent)
    
    # ---> THE GIGANTIC ITERATION <---
    # The searcher will run for 'GLOBAL_LFS_BUDGET' iterations and return the complete optimal path
    optimal_action_sequence = searcher.run_search(initial_state=sm, session_folder=session_folder)
    
    print("\n>>> 4. Search Complete! Replaying Optimal Sequence in Ground Truth Reality...")
    
    # Execute the extracted path sequentially
    for step_num, action in enumerate(optimal_action_sequence, 1):
        
        # Advance clock to the moment this action was actually supposed to occur
        while True:
            feasible = sm.get_feasible_actions()
            # If the specific action the LLM chose is available right now, execute it
            if any(a["job"] == action["job"] and a["op"] == action["op"] and a["machine"] == action["machine"] for a in feasible):
                break
                
            # If not, time must move forward
            event_type, timestamp, data = sm.process_next_event()
            if event_type is None:
                break
        
        sm.execute_action(action["job"], action["op"], action["machine"])
        print(f"   [Step {step_num:02d}] Executed: Job {action['job']}, Op {action['op']} -> Mach {action['machine']}")
        log_event(session_folder, sm.current_time, "Action_Executed", f"Job {action['job']} -> Mach {action['machine']}")

    # Fast forward any remaining simulation time until all jobs are technically 'completed'
    while not all(status == 'completed' for status in sm.job_status.values()):
        sm.process_next_event()

    print(f"\n{'='*40}")
    print(f">>> FINAL VALIDATED MAKESPAN: {sm.current_time:.2f}")
    print(f">>> TOTAL WALL-CLOCK TIME: {datetime.now() - total_start_time}")
    print(f"{'='*40}")

if __name__ == "__main__":
    # Ensure you add GLOBAL_LFS_BUDGET = 5000 to config.py before running!
    main()