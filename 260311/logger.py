import os

def setup_session_folder(session_name: str) -> str:
    """Creates a folder for the session if it doesn't already exist."""
    # Ensure the directory exists; ignore if it already does
    os.makedirs(session_name, exist_ok=True)
    return session_name

def log_interaction(problem_file:str, model_name:str, session_folder: str, iteration: int, prompt_inputs: dict, llm_response: str):
    """Logs the state and the LLM's full response to a numbered text file."""
    filepath = os.path.join(session_folder, f"{iteration}.txt")
    
    with open(filepath, 'a', encoding='utf-8') as f:
        f.write(f"--- PROBLEM FILE: {problem_file} ---\n")
        f.write(f"--- MODEL NAME: {model_name} ---\n")

        f.write(f"--- ITERATION {iteration} ---\n")
        f.write(f"Timestamp: {prompt_inputs.get('timestamp', 'N/A')}\n\n")
        
        f.write("=== MACHINES STATES ===\n")
        f.write(prompt_inputs.get('machines_states', '') + "\n\n")
        
        f.write("=== EMERGENCY JOBS ===\n")
        f.write(prompt_inputs.get('emergency_jobs', '') + "\n\n")
        
        f.write("=== STRATEGIC EXPERIENCE ===\n")
        f.write(prompt_inputs.get('strategic_experience', 'None available.') + "\n\n")
        
        f.write("=== READY OPERATIONS ===\n")
        f.write(prompt_inputs.get('ready_operations', '') + "\n\n")
        
        f.write("=== CANDIDATE ACTIONS (JSON) ===\n")
        f.write(prompt_inputs.get('actions_json', '') + "\n\n")
        
        f.write("=== FULL LLM RESPONSE ===\n")
        f.write(llm_response + "\n")

def log_file(session_folder: str, iteration: int, text:str):
    """Additional logging"""
    filepath = os.path.join(session_folder, f"{iteration}.txt")
    with open(filepath, 'a', encoding='utf-8') as f:
        f.write(text+"\n")