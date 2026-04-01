import os
import json
from datetime import datetime

from llm_agent import LLMAgent
from reflection_engine import Reflec
from state_manager import StateManager
from strategies.lfs_search import LFSSearcher
from strategies.beam_search import BeamSearcher
from strategies.mcts_search import MCTSSearcher
from strategies.single_search import SingleSearcher
from utilities.logger import setup_session_folder, log_event

import config

def load_text_file(filepath: str) -> str:
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.read()

def main():
    total_start_time = datetime.now()
    print(">>> 1. Loading problem data...")
    with open(config.PROBLEM_FILE, "r") as f:
        problem_data = json.load(f)
    
    # 1. Initialize Reality
    sm = StateManager(problem_data)
    session_folder = setup_session_folder(config.SESSION_NAME)
    
    # 1.5 Save a copy of the config file
    config_copy_path = os.path.join(session_folder, "config_copy.json")
    config_dict = {k: v for k, v in vars(config).items() if k.isupper() and not k.startswith('_')}
    with open(config_copy_path, "w") as f:
        json.dump(config_dict, f, indent=4)
    
    # 2. Initialize the AI Brain
    action_template = load_text_file(config.ACTION_PROMPT_FILE)
    value_template = load_text_file(config.VALUE_PROMPT_FILE)
    prior_template = load_text_file(config.PRIOR_PROMPT_FILE)
    reflect_template = load_text_file(config.REFLECT_PROMPT_FILE)
    explore_template = load_text_file(config.EXPLORE_PROMPT_FILE)

    agent = LLMAgent(
        action_prompt_template=action_template, 
        value_prompt_template=value_template,
        prior_prompt_template=prior_template,
        explore_prompt_template=explore_template
    )

    reflector = Reflec(prompt_template=reflect_template)
    
    # 3. Instantiate the Chosen Search Strategy
    print(f">>> 2. Initializing Search Framework: {config.SEARCH_STRATEGY}")
    if config.SEARCH_STRATEGY == "SingleSearch":
        searcher = SingleSearcher(llm_agent=agent)
    elif config.SEARCH_STRATEGY == "BeamSearch":
        searcher = BeamSearcher(llm_agent=agent)
    elif config.SEARCH_STRATEGY == "MCTSSearch":
        searcher = MCTSSearcher(llm_agent=agent)
    elif config.SEARCH_STRATEGY == "LFSSearch":
        searcher = LFSSearcher(llm_agent=agent)
    else:
        raise ValueError(f"Unknown SEARCH_STRATEGY in config: {config.SEARCH_STRATEGY}")

    iteration = 1
    
    # 4. The Online Planning Loop
    while not all(status == 'completed' for status in sm.job_status.values()):
        actions = sm.get_feasible_actions()
        
        # Handle Time Advance & Dynamic Events
        if not actions:
            event_type, timestamp, data = sm.process_next_event()
            if event_type is None:
                break
            log_event(session_folder, timestamp, event_type, str(data))
            
            if event_type not in ["Operation_Completion"]:
                print(f"\n>>> [DISRUPTION] {event_type} detected at T={timestamp}. Triggering Reflection Engine...")
                current_exp = searcher.current_strategic_experience
                event_info = f"{event_type} at T={timestamp}"
                
                new_exp = reflector.generate_reflection(sm, current_exp, event_info, session_folder)
                searcher.update_strategic_experience(new_exp) # Pass the new rule down to the searcher
            
            continue
            
        print(f"\n--- Iteration {iteration} | Clock: {sm.current_time} ---")
        
        # 5. Execute the Search Framework
        decision = searcher.run_search(initial_state=sm, session_folder=session_folder, iteration=iteration)
        
        # 6. Execute Action in Ground Truth Reality
        if decision:
            sm.execute_action(decision["job"], decision["op"], decision["machine"])
            log_event(session_folder, sm.current_time, "Action_Executed", f"Job {decision['job']} -> Mach {decision['machine']}")
            print(f">>> SUCCESS: Job {decision['job']}, Op {decision['op']} -> Mach {decision['machine']}.")
        else:
            # Fallback to SPT if search fails completely (API crash, etc.)
            fallback = min(actions, key=lambda x: x["processing_time"])
            sm.execute_action(fallback["job"], fallback["op"], fallback["machine"])
            print(">>> FALLBACK: Executed SPT due to search failure.")
            
        iteration += 1

    print(f"\n>>> Simulation Complete! Final Makespan: {sm.current_time}")
    print(f"Total processing time: {datetime.now() - total_start_time}")

if __name__ == "__main__":
    main()