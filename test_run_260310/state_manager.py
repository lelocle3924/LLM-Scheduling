import json
import heapq
from typing import List, Dict, Set, Any

class StateManager:
    def __init__(self, problem_data: Dict[str, Any], start_time: float = 0.0):
        # --- 1. Initialization ---
        self.num_machines = problem_data["machines"]
        self.jobs = problem_data["jobs"]
        self.current_time = start_time
        
        # Event queue (min-heap based on timestamp)
        # Events are tuples: (timestamp, event_id, event_type, data_dict)
        self.event_queue = []
        self._event_counter = 0 # To prevent ties in heapq
        
        # --- 2. State Variables ---
        # machine_avail: When each machine will be free (rk(t))
        self.machine_avail = {m: self.current_time for m in range(self.num_machines)}
        
        # job_progress: The index of the next operation to be scheduled for each job
        self.job_progress = {j: 0 for j in range(len(self.jobs))}
        
        # job_status: 'idle' (ready to schedule), 'running', or 'completed'
        self.job_status = {j: 'idle' for j in range(len(self.jobs))}
        
        self.broken_machines: Set[int] = set()
        self.emergency_jobs: Set[int] = set()
        self.interrupted_ops: Set[tuple] = set() # (job_id, op_id)

    def add_event(self, timestamp: float, event_type: str, data: dict):
        """Helper to push events to the priority queue."""
        heapq.heappush(self.event_queue, (timestamp, self._event_counter, event_type, data))
        self._event_counter += 1

    # --- 3. Event Handling ---
    def process_next_event(self):
        """Pops and processes the next event from the queue."""
        if not self.event_queue:
            return False
            
        timestamp, _, event_type, data = heapq.heappop(self.event_queue)
        self.current_time = timestamp
        
        if event_type == "Operation_Completion":
            job_id = data["job_id"]
            # Mark the job as ready for its next operation
            self.job_status[job_id] = 'idle'
            self.job_progress[job_id] += 1
            if self.job_progress[job_id] >= len(self.jobs[job_id]):
                self.job_status[job_id] = 'completed'
                
        elif event_type == "Machine_Breakdown":
            machine_id = data["machine_id"]
            self.broken_machines.add(machine_id)
            # Logic to interrupt current operation would go here...
            
        elif event_type == "Job_Emergency":
            job_id = data["job_id"]
            self.emergency_jobs.add(job_id)
            
        return True

    # --- 4. Feasible Action Generation ---
    def get_feasible_actions(self) -> List[dict]:
        """Identifies all operations that are ready and their available candidate machines."""
        feasible_actions = []
        for job_id, status in self.job_status.items():
            if status == 'idle':
                op_id = self.job_progress[job_id]
                candidates = self.jobs[job_id][op_id]
                
                for cand in candidates:
                    machine_id = cand["machine"]
                    # Check if machine is available (not broken and not busy right now)
                    if machine_id not in self.broken_machines and self.machine_avail[machine_id] <= self.current_time:
                        feasible_actions.append({
                            "job": job_id,
                            "op": op_id,
                            "machine": machine_id,
                            "processing_time": cand["processing"]
                        })
        return feasible_actions

    # --- 5. Action Execution ---
    def execute_action(self, job_id: int, op_id: int, machine_id: int):
        """Executes LLM decision, updates state, and schedules completion event."""
        # Find processing time
        processing_time = next(
            cand["processing"] for cand in self.jobs[job_id][op_id] if cand["machine"] == machine_id
        )
        
        # Update Machine and Job State
        start_time = max(self.current_time, self.machine_avail[machine_id])
        end_time = start_time + processing_time
        
        self.machine_avail[machine_id] = end_time
        self.job_status[job_id] = 'running'
        
        # Queue the completion event
        self.add_event(end_time, "Operation_Completion", {"job_id": job_id, "op_id": op_id})

    # --- Utilities for LLM Prompt Compilation ---
    def _calculate_rem_work(self, job_id: int) -> int:
        """Calculates remaining work by summing the minimum processing time of remaining ops[cite: 6]."""
        rem_work = 0
        for op_idx in range(self.job_progress[job_id], len(self.jobs[job_id])):
            min_time = min(cand["processing"] for cand in self.jobs[job_id][op_idx])
            rem_work += min_time
        return rem_work

    def _calculate_machine_contention(self) -> dict:
        """Counts how many future operations can potentially use each machine[cite: 4]."""
        contention = {m: 0 for m in range(self.num_machines)}
        for job_id, status in self.job_status.items():
            if status != 'completed':
                # Start from next operation (or current if idle)
                start_idx = self.job_progress[job_id] if status == 'idle' else self.job_progress[job_id] + 1
                for op_idx in range(start_idx, len(self.jobs[job_id])):
                    for cand in self.jobs[job_id][op_idx]:
                        contention[cand["machine"]] += 1
        return contention

    def compile_prompt_elements(self) -> dict:
        """Compiles the dynamic strings required for the LLM prompt template."""
        actions = self.get_feasible_actions()
        contention = self._calculate_machine_contention()
        
        # 1. {Machines States} [cite: 3, 4]
        machine_states_str = ""
        for m in range(self.num_machines):
            status = "Broken" if m in self.broken_machines else ("Busy" if self.machine_avail[m] > self.current_time else "Available")
            machine_states_str += f"- Machine {m}: status={status}, contention={contention[m]}\n"
            
        # 2. {Emergency Jobs} [cite: 8]
        if self.emergency_jobs:
            emergencies_str = "- " + ", ".join([f"Job {j}" for j in self.emergency_jobs])
        else:
            emergencies_str = "- None"
            
        # 3. {Ready Operations} [cite: 5, 6, 7]
        ready_ops_str = ""
        ready_jobs = set(a["job"] for a in actions)
        for job_id in ready_jobs:
            op_id = self.job_progress[job_id]
            candidates = self.jobs[job_id][op_id]
            flexibility = len(candidates)
            rem_work = self._calculate_rem_work(job_id)
            is_emerg = job_id in self.emergency_jobs
            
            ready_ops_str += f"- Job {job_id}, Op {op_id}: rem_work={rem_work}, flexibility={flexibility}, [EMERGENCY]={is_emerg}\n"
            
        # 4. {actions_json} [cite: 9]
        clean_actions = [{"job": a["job"], "op": a["op"], "machine": a["machine"]} for a in actions]
        
        return {
            "timestamp": self.current_time,
            "machines_states": machine_states_str.strip(),
            "emergency_jobs": emergencies_str,
            "ready_operations": ready_ops_str.strip(),
            "actions_json": json.dumps(clean_actions, indent=2)
        }

# ==========================================
# Example Usage with mk01.json
# ==========================================
if __name__ == "__main__":
    # Load your JSON data
    with open("mk01.json", "r") as f:
        problem_data = json.load(f)
        
    # Initialize State Manager
    state_manager = StateManager(problem_data)
    
    # Optional: Simulate a dynamic event at time 0
    state_manager.add_event(0.0, "Job_Emergency", {"job_id": 2})
    state_manager.process_next_event()
    
    # Compile prompt elements for the LLM
    prompt_inputs = state_manager.compile_prompt_elements()
    
    print("--- GENERATED PROMPT ELEMENTS ---")
    print(f"Timestamp: {prompt_inputs['timestamp']}\n")
    print("Machines States:\n" + prompt_inputs['machines_states'] + "\n")
    print("Emergency Jobs:\n" + prompt_inputs['emergency_jobs'] + "\n")
    print("Ready Operations:\n" + prompt_inputs['ready_operations'] + "\n")
    print("Actions JSON:\n" + prompt_inputs['actions_json'])