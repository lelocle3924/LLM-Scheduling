import json
import re
import os
import time
import requests
from dotenv import load_dotenv
from utilities.logger import log_llm_call
import config

load_dotenv()
API_KEY = os.getenv("OPENROUTER_API_KEY")

class LLMAgent:
    """The Unified LLM Brain for Policy and Value estimations."""
    
    def __init__(self, action_prompt_template: str, value_prompt_template: str, prior_prompt_template: str, explore_prompt_template: str):
        self.action_prompt = action_prompt_template
        self.value_prompt = value_prompt_template
        self.prior_prompt = prior_prompt_template
        self.explore_prompt = explore_prompt_template

    def _call_api(self, prompt_text: str, session_folder: str = None, iteration: int = 0, call_type: str = "General") -> str:
        """Centralized API calling with full-prompt logging."""
        for attempt in range(config.MAX_RETRIES):
            try:
                start_time = time.time()
                response = requests.post(
                    url="https://openrouter.ai/api/v1/chat/completions",
                    headers={"Authorization": f"Bearer {API_KEY}"},
                    json={"model": config.MODEL_NAME, "messages": [{"role": "user", "content": prompt_text}], "temperature": config.TEMPERATURE}
                )
                response.raise_for_status()
                latency = time.time() - start_time
                
                llm_output = response.json()['choices'][0]['message']['content']
                
                if session_folder:
                    log_llm_call(
                        session_folder=session_folder,
                        iteration=iteration,
                        call_type=call_type,
                        model_name=config.MODEL_NAME,
                        prompt_text=prompt_text,
                        llm_response=llm_output,
                        latency=latency,
                    )
                    
                return llm_output
            except Exception as e:
                print(f"API attempt {attempt+1} failed for {call_type}: {e}")
        return ""

    def get_action(self, state, feasible_actions: list, strategic_experience: str, session_folder: str, iteration: int) -> dict:
        """Acts as the Policy Network. Returns a single action."""
        prompt_inputs = state.compile_prompt_elements()
        prompt_inputs['strategic_experience'] = strategic_experience
        
        prompt_text = self.action_prompt.replace("{snapshot['timestamp']}", str(prompt_inputs['timestamp']))
        prompt_text = prompt_text.replace("{Machines States}", prompt_inputs['machines_states'])
        prompt_text = prompt_text.replace("{Emergency Jobs}", prompt_inputs['emergency_jobs'])
        prompt_text = prompt_text.replace("{Strategic Experience}", prompt_inputs['strategic_experience'])
        prompt_text = prompt_text.replace("{Ready Operations}", prompt_inputs['ready_operations'])
        prompt_text = prompt_text.replace("{Full State Information}", prompt_inputs['full_state'])
        prompt_text = prompt_text.replace("{actions_json}", prompt_inputs['actions_json'])

        llm_output = self._call_api(prompt_text, session_folder, iteration, call_type="Action_Policy")
        
        match = re.search(r'\{.*?\}', llm_output, re.DOTALL)
        if match:
            try:
                decision = json.loads(match.group(0))
                # Validate against feasible actions
                for a in feasible_actions:
                    if a["job"] == decision.get("job") and a["op"] == decision.get("op") and a["machine"] == decision.get("machine"):
                        return decision
            except json.JSONDecodeError:
                pass
                
        return None 

    def get_value(self, state, strategic_experience: str, session_folder: str, iteration: int) -> float:
        """Acts as the Value Network. Scores the current state from 0.0 to 1.0."""
        prompt_inputs = state.compile_prompt_elements()
        prompt_inputs['strategic_experience'] = strategic_experience
        
        prompt_text = self.value_prompt.replace("{snapshot['timestamp']}", str(prompt_inputs['timestamp']))
        prompt_text = prompt_text.replace("{Machines States}", prompt_inputs['machines_states'])
        prompt_text = prompt_text.replace("{Ready Operations}", prompt_inputs['ready_operations'])
        prompt_text = prompt_text.replace("{Full State Information}", prompt_inputs['full_state'])
        prompt_text = prompt_text.replace("{Strategic Experience}", prompt_inputs['strategic_experience'])
        
        lb = prompt_inputs.get('lower_bound', str(prompt_inputs['timestamp'])) 
        prompt_text = prompt_text.replace("{snapshot['lower_bound']}", lb)

        llm_output = self._call_api(prompt_text, session_folder, iteration, call_type="Value_Estimation")
        
        match = re.search(r'<score>\s*([\d\.]+)\s*</score>', llm_output, re.IGNORECASE)
        if match:
            return max(0.0, min(1.0, float(match.group(1))))
                
        return 0.5

    def get_priors(self, state, feasible_actions: list, strategic_experience: str, session_folder: str, iteration: int) -> dict:
        """Acts as the Probabilistic Policy Network for MCTS Expansion."""
        if not feasible_actions:
            return {}

        prompt_inputs = state.compile_prompt_elements()
        
        # Inject explicit string indices into the actions
        indexed_actions = [{"index": str(i), **a} for i, a in enumerate(feasible_actions)]
        prompt_inputs['actions_json'] = json.dumps(indexed_actions, indent=2)
        prompt_inputs['strategic_experience'] = strategic_experience
        
        prompt_text = self.prior_prompt.replace("{snapshot['timestamp']}", str(prompt_inputs['timestamp']))
        prompt_text = prompt_text.replace("{Machines States}", prompt_inputs['machines_states'])
        prompt_text = prompt_text.replace("{Emergency Jobs}", prompt_inputs['emergency_jobs'])
        prompt_text = prompt_text.replace("{Ready Operations}", prompt_inputs['ready_operations'])
        prompt_text = prompt_text.replace("{Full State Information}", prompt_inputs['full_state'])
        prompt_text = prompt_text.replace("{Strategic Experience}", prompt_inputs['strategic_experience'])
        prompt_text = prompt_text.replace("{actions_json}", prompt_inputs['actions_json'])

        llm_output = self._call_api(prompt_text, session_folder, iteration, call_type="Prior_Probabilities")
        
        match = re.search(r'\{.*\}', llm_output, re.DOTALL)
        priors = {}
        if match:
            try:
                data = json.loads(match.group(0))
                if "operation_scores" in data:
                    priors = {str(k): float(v) for k, v in data["operation_scores"].items()}
            except (json.JSONDecodeError, ValueError):
                pass

        # Fallback & Normalization Safety Net
        final_priors = {}
        total_prob = 0.0
        
        for i in range(len(feasible_actions)):
            idx_str = str(i)
            prob = priors.get(idx_str, 1.0 / len(feasible_actions))
            prob = max(0.001, prob) 
            final_priors[idx_str] = prob
            total_prob += prob
            
        if total_prob > 0:
            final_priors = {k: v / total_prob for k, v in final_priors.items()}
            
        return final_priors

    def get_explore_decision(self, state, strategic_experience: str, session_folder: str, iteration: int) -> bool:
        """LFS specific: Asks the LLM if it should backtrack (EXPLORE) or CONTINUE."""
        if not self.explore_prompt:
            return False 
            
        prompt_inputs = state.compile_prompt_elements()
        prompt_inputs['strategic_experience'] = strategic_experience
        
        prompt_text = self.explore_prompt.replace("{snapshot['timestamp']}", str(prompt_inputs['timestamp']))
        prompt_text = prompt_text.replace("{Machines States}", prompt_inputs['machines_states'])
        prompt_text = prompt_text.replace("{Ready Operations}", prompt_inputs['ready_operations'])
        prompt_text = prompt_text.replace("{Full State Information}", prompt_inputs['full_state'])
        prompt_text = prompt_text.replace("{Strategic Experience}", prompt_inputs['strategic_experience'])
        lb = prompt_inputs.get('lower_bound', str(prompt_inputs['timestamp'])) 
        prompt_text = prompt_text.replace("{snapshot['lower_bound']}", lb)

        llm_output = self._call_api(prompt_text, session_folder, iteration, call_type="Explore_Decision")
        
        match = re.search(r'<decision>(EXPLORE|CONTINUE)</decision>', llm_output, re.IGNORECASE)
        if match:
            decision = match.group(1).upper()
            return decision == "EXPLORE"
            
        return False