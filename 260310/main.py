# with retry logic for hallucinations
import json
import os
import re
import requests
from state_manager import StateManager
from dotenv import load_dotenv
import config
from logger import setup_session_folder, log_interaction

# Configuration
load_dotenv()
API_KEY = os.getenv("OPENROUTER_API_KEY")

def load_text_file(filepath: str) -> str:
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.read()

def main():
    print(">>> 1. Loading problem data and prompt template...")
    with open(config.PROBLEM_FILE, "r") as f:
        problem_data = json.load(f)
    
    prompt_template = load_text_file("decision_prompt.txt")
    
    print(">>> 2. Initializing State Manager and Logger...")
    sm = StateManager(problem_data)
    session_folder = setup_session_folder(config.SESSION_NAME)
    
    iteration = 1
    print(f">>> Starting continuous simulation. Logging to folder: '{session_folder}'")
    
    # 3. Continuous Scheduling Loop
    while not all(status == 'completed' for status in sm.job_status.values()):
        
        actions = sm.get_feasible_actions()
        
        # If no actions are feasible, advance the simulation clock
        if not actions:
            has_more_events = sm.process_next_event()
            if not has_more_events:
                print("\nWARNING: No more events in the queue, but jobs are not finished. Deadlock detected.")
                break
            continue
            
        print(f"\n--- Iteration {iteration} | Clock: {sm.current_time} ---")
        
        # Extract the current dynamic state into text elements
        prompt_inputs = sm.compile_prompt_elements()
        prompt_inputs['strategic_experience'] = "Strategic Experience: None available for this step."
        
        # Inject state data into the base prompt
        base_prompt_text = prompt_template.replace("{snapshot['timestamp']}", str(prompt_inputs['timestamp']))
        base_prompt_text = base_prompt_text.replace("{snapshot['timestamp']}", str(prompt_inputs['timestamp'])) 
        base_prompt_text = base_prompt_text.replace("{Machines States}", prompt_inputs['machines_states'])
        base_prompt_text = base_prompt_text.replace("{Emergency Jobs}", prompt_inputs['emergency_jobs'])
        base_prompt_text = base_prompt_text.replace("{Strategic Experience}", prompt_inputs['strategic_experience'])
        base_prompt_text = base_prompt_text.replace("{Ready Operations}", prompt_inputs['ready_operations'])
        base_prompt_text = base_prompt_text.replace("{actions_json}", prompt_inputs['actions_json'])

        MAX_RETRIES = 3
        action_executed = False
        current_prompt_text = base_prompt_text

        # --- RETRY LOOP ---
        for attempt in range(1, MAX_RETRIES + 1):
            if attempt > 1:
                print(f">>> Retrying LLM call (Attempt {attempt}/{MAX_RETRIES})...")
            else:
                print(">>> Calling LLM via OpenRouter API...")

            headers = {
                "Authorization": f"Bearer {API_KEY}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": config.MODEL_NAME,
                "messages": [
                    {"role": "user", "content": current_prompt_text}
                ],
                "temperature": config.TEMPERATURE,
                "max_tokens": config.MAX_TOKENS
            }

            try:
                response = requests.post(
                    url="https://openrouter.ai/api/v1/chat/completions",
                    headers=headers,
                    data=json.dumps(payload)
                )
                response.raise_for_status()
                result = response.json()
                
                llm_output = result['choices'][0]['message']['content']
                
                # Log the full interaction (will overwrite with the latest attempt for this iteration)
                log_interaction(session_folder, iteration, prompt_inputs, llm_output)
                if attempt == 1:
                    print(f">>> Logged interaction to {session_folder}/{iteration}.txt")
                
                # Parse LLM output using Regex
                match = re.search(r'\{.*?\}', llm_output, re.DOTALL)
                
                if match:
                    clean_output = match.group(0)
                    decision = json.loads(clean_output)
                    
                    job_id = decision.get("job")
                    op_id = decision.get("op")
                    machine_id = decision.get("machine")
                    
                    if job_id is not None and op_id is not None and machine_id is not None:
                        # Validate the LLM's choice against actual feasible actions
                        is_valid = any(
                            a["job"] == job_id and a["op"] == op_id and a["machine"] == machine_id 
                            for a in actions
                        )
                        
                        if is_valid:
                            sm.execute_action(job_id, op_id, machine_id)
                            print(f">>> SUCCESS: Executed decision -> Job {job_id}, Op {op_id} assigned to Machine {machine_id}.")
                            action_executed = True
                            break # Break the retry loop on success
                        else:
                            print(f">>> WARNING: LLM hallucinated invalid action (Job {job_id}, Op {op_id}, Mach {machine_id}).")
                            # Append error feedback so the LLM knows what to fix on the next attempt
                            current_prompt_text = base_prompt_text + f"\n\n[SYSTEM WARNING]: Your previous choice ({{\"job\": {job_id}, \"op\": {op_id}, \"machine\": {machine_id}}}) was INVALID. You MUST strictly choose an action that exists in the Candidate Actions list."
                    else:
                        print(">>> ERROR: LLM returned incomplete JSON keys.")
                        current_prompt_text = base_prompt_text + "\n\n[SYSTEM WARNING]: Your previous response was missing required JSON keys. Please provide exactly {\"job\": int, \"op\": int, \"machine\": int}."
                else:
                    print(f">>> ERROR: Could not find JSON object in LLM response.")
                    current_prompt_text = base_prompt_text + "\n\n[SYSTEM WARNING]: Could not parse JSON from your response. Ensure you output a valid JSON dictionary."

            except requests.exceptions.RequestException as e:
                print(f"API Error: {e}")
                if 'response' in locals() and response is not None:
                    print(response.text)
                break # Hard network fail, break retry loop
            except json.JSONDecodeError:
                print(">>> ERROR: Could not parse the LLM output as JSON.")
                current_prompt_text = base_prompt_text + "\n\n[SYSTEM WARNING]: Your previous JSON was improperly formatted."
                
        # --- FALLBACK LOGIC ---
        if not action_executed:
            print(f">>> FALLBACK: LLM failed {MAX_RETRIES} times. Automatically selecting the first valid action.")
            fallback = actions[0]
            sm.execute_action(fallback["job"], fallback["op"], fallback["machine"])
            print(f">>> FALLBACK SUCCESS: Executed -> Job {fallback['job']}, Op {fallback['op']} assigned to Machine {fallback['machine']}.")
            
        iteration += 1

    print(f"\n>>> Simulation Complete! Final Makespan: {sm.current_time}")

if __name__ == "__main__":
    main()