import os
import json
import re
import requests
from datetime import datetime
from dotenv import load_dotenv

import config
from logger import log_interaction, log_file

load_dotenv()
API_KEY = os.getenv("OPENROUTER_API_KEY")

class Sched:
    """The Scheduling Decision Maker."""
    def __init__(self, prompt_template: str):
        self.prompt_template = prompt_template

    def make_decision(self, prompt_inputs: dict, feasible_actions: list, session_folder: str, iteration: int) -> dict:
        """
        Takes environment state, queries the LLM, handles errors, and returns a validated decision.
        Returns None if the LLM fails after MAX_RETRIES.
        """
        # Inject state data into the base prompt
        base_prompt_text = self.prompt_template.replace("{snapshot['timestamp']}", str(prompt_inputs['timestamp']))
        base_prompt_text = base_prompt_text.replace("{Machines States}", prompt_inputs['machines_states'])
        base_prompt_text = base_prompt_text.replace("{Emergency Jobs}", prompt_inputs['emergency_jobs'])
        base_prompt_text = base_prompt_text.replace("{Strategic Experience}", prompt_inputs.get('strategic_experience', 'None available.'))
        base_prompt_text = base_prompt_text.replace("{Ready Operations}", prompt_inputs['ready_operations'])
        base_prompt_text = base_prompt_text.replace("{actions_json}", prompt_inputs['actions_json'])

        current_prompt_text = base_prompt_text
        
        for attempt in range(1, config.MAX_RETRIES + 1):
            if attempt > 1:
                print(f">>> Retrying LLM call (Attempt {attempt}/{config.MAX_RETRIES})...")
            else:
                print(">>> Calling LLM via OpenRouter API...")

            headers = {
                "Authorization": f"Bearer {API_KEY}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": config.MODEL_NAME,
                "messages": [{"role": "user", "content": current_prompt_text}],
                "temperature": config.TEMPERATURE,
                "max_tokens": config.MAX_TOKENS
            }

            try:
                response_start = datetime.now()
                response = requests.post(
                    url="https://openrouter.ai/api/v1/chat/completions",
                    headers=headers,
                    data=json.dumps(payload)
                )
                response.raise_for_status()
                result = response.json()
                
                llm_output = result['choices'][0]['message'].get('content') or ""
                
                if not llm_output.strip():
                    print(">>> ERROR: LLM returned a completely blank response.")
                    log_interaction(config.PROBLEM_FILE, config.MODEL_NAME, session_folder, iteration, prompt_inputs, "[BLANK RESPONSE]")
                    current_prompt_text = base_prompt_text + "\n\n[SYSTEM WARNING]: Your previous response was completely blank. You MUST provide a JSON dictionary with your decision."
                    continue

                # Log the successful interaction
                log_interaction(config.PROBLEM_FILE, config.MODEL_NAME, session_folder, iteration, prompt_inputs, llm_output)
                log_file(session_folder, iteration, f"Response Time: {datetime.now() - response_start}")

                if attempt == 1:
                    print(f">>> Logged interaction to {session_folder}/{iteration}.txt")
                
                # Parse LLM output using Regex
                match = re.search(r'\{.*?\}', llm_output, re.DOTALL)
                
                if match:
                    decision = json.loads(match.group(0))
                    job_id = decision.get("job")
                    op_id = decision.get("op")
                    machine_id = decision.get("machine")
                    
                    if job_id is not None and op_id is not None and machine_id is not None:
                        # Validate the LLM's choice against actual feasible actions
                        is_valid = any(
                            a["job"] == job_id and a["op"] == op_id and a["machine"] == machine_id 
                            for a in feasible_actions
                        )
                        
                        if is_valid:
                            return decision # SUCCESS
                        else:
                            print(f">>> WARNING: LLM hallucinated invalid action (Job {job_id}, Op {op_id}, Mach {machine_id}).")
                            current_prompt_text = base_prompt_text + f"\n\n[SYSTEM WARNING]: Your previous choice ({{\"job\": {job_id}, \"op\": {op_id}, \"machine\": {machine_id}}}) was INVALID. You MUST strictly choose an action that exists in the Candidate Actions list."
                    else:
                        print(">>> ERROR: LLM returned incomplete JSON keys.")
                        current_prompt_text = base_prompt_text + "\n\n[SYSTEM WARNING]: Your previous response was missing required JSON keys. Please provide exactly {\"job\": int, \"op\": int, \"machine\": int}."
                else:
                    print(">>> ERROR: Could not find JSON object in LLM response.")
                    current_prompt_text = base_prompt_text + "\n\n[SYSTEM WARNING]: Could not parse JSON from your response. Ensure you output a valid JSON dictionary."

            except requests.exceptions.RequestException as e:
                print(f"API Error: {e}")
                if 'response' in locals() and response is not None:
                    print(response.text)
                break 
            except json.JSONDecodeError:
                print(">>> ERROR: Could not parse the LLM output as JSON.")
                current_prompt_text = base_prompt_text + "\n\n[SYSTEM WARNING]: Your previous JSON was improperly formatted."

        return None # Failed after MAX_RETRIES

class Reflec:
    """The Strategic Reflection Maker."""
    def __init__(self, prompt_template: str):
        self.prompt_template = prompt_template

    def generate_reflection(self, rollout_data: dict) -> str:
        """
        Placeholder for the semantic reasoning module.
        Will eventually compare rollouts and return a strategic rule.
        """
        # TODO: Implement Reflec logic using self.prompt_template
        pass