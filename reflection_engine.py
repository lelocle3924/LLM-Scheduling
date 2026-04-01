import os
import re
import json
import copy
import random
import requests
from dotenv import load_dotenv

from utilities.logger import log_reflection_cycle, log_skipped_reflection
import config

load_dotenv()
API_KEY = os.getenv("OPENROUTER_API_KEY")

class Reflec:
    """The Strategic Reflection Maker ($L=0$ MVP)."""
    def __init__(self, prompt_template: str):
        self.prompt_template = prompt_template

    def _stochastic_rollout(self, sm_clone, initial_action=None) -> tuple:
        """Runs a fast-forward stochastic simulation, tracking time and bottlenecks."""
        trajectory = []
        machine_busy = {m: 0 for m in range(sm_clone.num_machines)}
        
        if initial_action:
            machine_busy[initial_action['machine']] += initial_action['processing_time']

        def spt(acts, sm): return min(acts, key=lambda x: x["processing_time"]) 
        def lpt(acts, sm): return max(acts, key=lambda x: x["processing_time"]) 
        def mwkr(acts, sm): return max(acts, key=lambda x: sm._calculate_rem_work(x["job"])) 
        def lwkr(acts, sm): return min(acts, key=lambda x: sm._calculate_rem_work(x["job"])) 
        def est(acts, sm): return min(acts, key=lambda x: max(sm.current_time, sm.machine_avail[x["machine"]]))

        pdr_pool = [spt, lpt, mwkr, lwkr, est]
        weights = [0.2, 0.2, 0.2, 0.2, 0.2] 

        while not all(status == 'completed' for status in sm_clone.job_status.values()):
            actions = sm_clone.get_feasible_actions()
            
            if not actions:
                event_type, timestamp, data = sm_clone.process_next_event()
                if event_type is None:
                    break
                continue
            
            selected_pdr = random.choices(pdr_pool, weights=weights, k=1)[0]
            decision = selected_pdr(actions, sm_clone)
            
            m = decision["machine"]
            pt = decision["processing_time"]
            
            valid_avail = sm_clone.machine_avail[m] if m not in sm_clone.broken_machines else sm_clone.current_time
            est_start = max(sm_clone.current_time, valid_avail)
            
            sm_clone.execute_action(decision["job"], decision["op"], decision["machine"])
            
            machine_busy[m] += pt
            trajectory.append(f"[T:{est_start:.1f}-{est_start+pt:.1f}] J{decision['job']}O{decision['op']}@M{m} (pt:{pt})")
            
        makespan = sm_clone.current_time
        bottleneck_machine = max(sm_clone.machine_avail, key=sm_clone.machine_avail.get)
        
        analytics = {
            "bottleneck": bottleneck_machine,
            "busy_times": machine_busy
        }
            
        return makespan, trajectory, analytics

    def generate_reflection(self, current_sm, current_experience: str, event_info: str, session_folder: str) -> str:
        if not getattr(config, "USE_REFLECTION", True):
            return current_experience
            
        L = getattr(config, "REFLECTION_LEVELS", 2)
        R = getattr(config, "ROLLOUTS_PER_LEVEL", 12)
        current_exp = current_experience
        
        print(f"\n>>> [REFLEC] Initiating Hierarchical Reflection (Levels: {L}, Rollouts/Level: {R})...")
        
        for l in range(L, -1, -1):
            print(f"\n>>> [REFLEC] --- Running Level {l} Reflection ---")
            results = []
            base_clone = copy.deepcopy(current_sm)
            
            if l > 0:
                for _ in range(R):
                    rollout_clone = copy.deepcopy(base_clone)
                    makespan, traj, analytics = self._stochastic_rollout(rollout_clone)
                    if not traj: continue
                    results.append({"action": traj[0], "makespan": makespan, "trajectory": traj, "analytics": analytics})
            else:
                while True:
                    actions = base_clone.get_feasible_actions()
                    if actions: break
                    ev_type, ts, data = base_clone.process_next_event()
                    if ev_type is None: break
                        
                if not actions or len(actions) < 2:
                    print(f">>> [REFLEC] Level 0: Not enough branching options to compare. Skipping.")
                    continue
                    
                for action in actions:
                    branch_clone = copy.deepcopy(base_clone)
                    branch_clone.execute_action(action["job"], action["op"], action["machine"])
                    trajectory = [f"J{action['job']}O{action['op']}@M{action['machine']}"]
                    
                    makespan, remaining_traj, analytics = self._stochastic_rollout(branch_clone, initial_action=action)
                    trajectory.extend(remaining_traj)
                    results.append({"action": f"J{action['job']}O{action['op']}@M{action['machine']}", "makespan": makespan, "trajectory": trajectory, "analytics": analytics})
            
            if not results: continue
                
            best_result = min(results, key=lambda x: x["makespan"])
            worst_result = max(results, key=lambda x: x["makespan"])
            
            if best_result["makespan"] == worst_result["makespan"]:
                print(f">>> [REFLEC] Level {l}: Identical makespans ({best_result['makespan']}). Skipping LLM synthesis.")
                continue
                
            print(f">>> [REFLEC] Level {l}: Contrastive signals found! Best: {best_result['makespan']}, Worst: {worst_result['makespan']}")
            
            best_path_str = " -> ".join(best_result["trajectory"][:])
            worst_path_str = " -> ".join(worst_result["trajectory"][:])

            sim_outcomes = (
                f"**Best Path** (Makespan: {best_result['makespan']}):\n"
                f"Bottleneck Machine: M{best_result['analytics']['bottleneck']}\n"
                f"Initial Decision: {best_result['action']}\n"
                f"Decision Path: {best_path_str}\n\n"
                f"**Worst Path** (Makespan: {worst_result['makespan']}):\n"
                f"Bottleneck Machine: M{worst_result['analytics']['bottleneck']}\n"
                f"Initial Decision: {worst_result['action']}\n"
                f"Decision Path: {worst_path_str}"
            )

            origin_state = current_sm.compile_prompt_elements()
            state_desc = (
                f"Timestamp: {origin_state['timestamp']}\n"
                f"Machine States:\n{origin_state['machines_states']}\n"
                f"Emergency Jobs: {origin_state['emergency_jobs']}\n"
                f"Event Triggered: {event_info}\n"
                f"Planning Level: L={l}"
            )

            prompt_text = self.prompt_template.replace("{The Existing Strategic Principle}", current_exp)
            prompt_text = prompt_text.replace("{The Originating Decision-Point State}", state_desc)
            prompt_text = prompt_text.replace("{Summarized Simulation Outcomes}", sim_outcomes)

            print(f">>> [REFLEC] Level {l}: Querying LLM for strategic synthesis...")
            try:
                response = requests.post(
                    url="https://openrouter.ai/api/v1/chat/completions",
                    headers={"Authorization": f"Bearer {API_KEY}"},
                    json={"model": config.MODEL_NAME, "messages": [{"role": "user", "content": prompt_text}], "temperature": 0.3}
                )
                response.raise_for_status()
                llm_output = response.json()['choices'][0]['message']['content']
                
                match = re.search(r'<key_insights>(.*?)</key_insights>', llm_output, re.DOTALL | re.IGNORECASE)
                if match:
                    new_experience = match.group(1).strip()
                    print(f">>> [REFLEC] Level {l} Successfully synthesized refined experience:\n{new_experience}")
                    log_reflection_cycle(session_folder, current_sm.current_time, f"{event_info} (Level {l})", prompt_text, results, best_result, worst_result, new_experience, llm_output)
                    current_exp = new_experience
                else:
                    print(f">>> [REFLEC] Level {l}: Failed to parse <key_insights> tag.")
                    log_reflection_cycle(session_folder, current_sm.current_time, f"{event_info} (Level {l})", prompt_text, results, best_result, worst_result, "[FAILED TO PARSE]", llm_output)

            except Exception as e:
                print(f">>> [REFLEC] Level {l} API Error: {e}")
                log_skipped_reflection(session_folder, current_sm.current_time, f"{event_info} (Level {l})", f"API Error: {str(e)}")

        return current_exp