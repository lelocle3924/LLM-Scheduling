import json
import heapq
import random
import os
from typing import List, Dict, Set, Any
import config

class StateManager:
    def __init__(self, problem_data: Dict[str, Any], start_time: float = 0.0):
        self.num_machines = problem_data["machines"]
        self.jobs = problem_data["jobs"]
        self.current_time = start_time
        
        self.event_queue = []
        self._event_counter = 0 
        
        self.machine_avail = {m: self.current_time for m in range(self.num_machines)}
        self.machine_current_op = {m: None for m in range(self.num_machines)}
        
        # Track start times to calculate remaining work if interrupted
        self.op_start_times = {} 
        # ADDED: Track expected completion times to filter out stale 'ghost' events
        self.op_expected_end_time = {}
        
        self.job_progress = {j: 0 for j in range(len(self.jobs))}
        self.job_status = {j: 'idle' for j in range(len(self.jobs))}
        
        self.broken_machines: Set[int] = set()
        self.emergency_jobs: Set[int] = set()
        
        # Interrupted ops: mapping (job_id, op_id) -> {"machine": m, "work_done": float, "total_p": float}
        self.interrupted_ops: Dict[tuple, dict] = {} 

        self._initialize_dynamic_events()

    def _initialize_dynamic_events(self):
        """Initializes events based on Mode 1 (Random) or Mode 2 (JSON file)."""
        if config.DYNAMIC_EVENTS_FILE and os.path.exists(config.DYNAMIC_EVENTS_FILE):
            print(f">>> Mode 2: Loading user-specified events from {config.DYNAMIC_EVENTS_FILE}")
            with open(config.DYNAMIC_EVENTS_FILE, 'r') as f:
                events = json.load(f)
                for ev in events:
                    self.add_event(ev["timestamp"], ev["event_type"], ev["data"])
        else:
            print(">>> Mode 1: Generating random dynamic events.")
            if config.RANDOM_SEED is not None:
                random.seed(config.RANDOM_SEED)
                
            # Random Breakdowns & Repairs
            for _ in range(config.NUM_RANDOM_BREAKDOWNS):
                m_id = random.randint(0, self.num_machines - 1)
                t_break = random.uniform(10.0, 50.0) 
                t_repair = t_break + random.uniform(5.0, 15.0)
                
                self.add_event(t_break, "Machine_Breakdown", {"machine_id": m_id, "repair_time": t_repair})
                self.add_event(t_repair, "Machine_Repair", {"machine_id": m_id})
                
            # Random Emergencies
            for _ in range(config.NUM_RANDOM_EMERGENCIES):
                j_id = random.randint(0, len(self.jobs) - 1)
                t_emerg = random.uniform(5.0, 40.0)
                self.add_event(t_emerg, "Job_Emergency", {"job_id": j_id})

    def add_event(self, timestamp: float, event_type: str, data: dict):
        heapq.heappush(self.event_queue, (timestamp, self._event_counter, event_type, data))
        self._event_counter += 1

    def process_next_event(self) -> tuple:
        """Pops and processes the next event. Returns (event_type, timestamp, data) for logging."""
        while True:
            if not self.event_queue:
                return None, None, None
                
            timestamp, _, event_type, data = heapq.heappop(self.event_queue)
            self.current_time = timestamp
            
            if event_type == "Operation_Completion":
                job_id, op_id, machine_id = data["job_id"], data["op_id"], data.get("machine_id")
                
                # FIX: Check if this completion event matches the currently expected end time
                expected_time = self.op_expected_end_time.get((job_id, op_id))
                if expected_time is None or abs(expected_time - timestamp) > 1e-5:
                    continue # Skip this stale event silently and pull the next one

                if machine_id is not None and self.machine_current_op[machine_id] == (job_id, op_id):
                    self.machine_current_op[machine_id] = None

                self.job_status[job_id] = 'idle'
                self.job_progress[job_id] += 1
                if self.job_progress[job_id] >= len(self.jobs[job_id]):
                    self.job_status[job_id] = 'completed'
                    
            elif event_type == "Machine_Breakdown":
                machine_id = data["machine_id"]
                repair_time = data["repair_time"]
                self.broken_machines.add(machine_id)
                self.machine_avail[machine_id] = repair_time # Postpone availability
                
                # Preempt-Resume Logic: Interrupt current operation
                current_op = self.machine_current_op[machine_id]
                if current_op:
                    job_id, op_id = current_op
                    start_t = self.op_start_times[(job_id, op_id)]
                    work_done = self.current_time - start_t
                    total_p = next(cand["processing"] for cand in self.jobs[job_id][op_id] if cand["machine"] == machine_id)
                    
                    # Move to interrupted set
                    self.interrupted_ops[(job_id, op_id)] = {"machine": machine_id, "work_done": work_done, "total_p": total_p}
                    self.machine_current_op[machine_id] = None # Clear machine
                    self.op_expected_end_time[(job_id, op_id)] = None 
                    
                    # ENHANCEMENT: Explicitly log the interrupted operation
                    data["interrupted_job"] = job_id
                    data["interrupted_op"] = op_id
                
            elif event_type == "Machine_Repair":
                machine_id = data["machine_id"]
                if machine_id in self.broken_machines:
                    self.broken_machines.remove(machine_id)
                    
                # Resume any interrupted operations
                for (job_id, op_id), info in list(self.interrupted_ops.items()):
                    if info["machine"] == machine_id:
                        rem_work = info["total_p"] - info["work_done"]
                        new_completion_time = self.current_time + rem_work # Calculate new completion
                        
                        self.machine_avail[machine_id] = new_completion_time
                        self.machine_current_op[machine_id] = (job_id, op_id)
                        self.op_start_times[(job_id, op_id)] = self.current_time # Reset start time for remaining work
                        self.op_expected_end_time[(job_id, op_id)] = new_completion_time 
                        
                        # Re-queue completion
                        self.add_event(new_completion_time, "Operation_Completion", {"job_id": job_id, "op_id": op_id, "machine_id": machine_id})
                        del self.interrupted_ops[(job_id, op_id)] # Remove from interrupted set
                        
                        # ENHANCEMENT: Explicitly log the resumed operation
                        data["resumed_job"] = job_id
                        data["resumed_op"] = op_id
                        break 
                
            elif event_type == "Job_Emergency":
                job_id = data["job_id"]
                self.emergency_jobs.add(job_id) # Handled by prioritization in the prompt
                
            return event_type, timestamp, data

    def get_feasible_actions(self) -> List[dict]:
        feasible_actions = []
        for job_id, status in self.job_status.items():
            if status == 'idle' and (job_id, self.job_progress[job_id]) not in self.interrupted_ops:
                op_id = self.job_progress[job_id]
                candidates = self.jobs[job_id][op_id]
                
                for cand in candidates:
                    machine_id = cand["machine"]
                    if machine_id not in self.broken_machines and self.machine_avail[machine_id] <= self.current_time:
                        feasible_actions.append({
                            "job": job_id, "op": op_id, "machine": machine_id, "processing_time": cand["processing"]
                        })
        return feasible_actions

    def execute_action(self, job_id: int, op_id: int, machine_id: int):
        processing_time = next(cand["processing"] for cand in self.jobs[job_id][op_id] if cand["machine"] == machine_id)
        start_time = max(self.current_time, self.machine_avail[machine_id])
        end_time = start_time + processing_time
        
        self.machine_avail[machine_id] = end_time
        self.job_status[job_id] = 'running'
        self.machine_current_op[machine_id] = (job_id, op_id)
        
        # Record start time for potential interruption calculations
        self.op_start_times[(job_id, op_id)] = start_time 
        self.op_expected_end_time[(job_id, op_id)] = end_time
        
        self.add_event(end_time, "Operation_Completion", {"job_id": job_id, "op_id": op_id, "machine_id": machine_id})

    # --- Utilities for LLM Prompt Compilation ---
    def _calculate_rem_work(self, job_id: int) -> int:
        """Calculates remaining work by summing the minimum processing time of remaining ops."""
        rem_work = 0
        for op_idx in range(self.job_progress[job_id], len(self.jobs[job_id])):
            min_time = min(cand["processing"] for cand in self.jobs[job_id][op_idx])
            rem_work += min_time
        return rem_work

    def _calculate_machine_contention(self) -> dict:
        """Counts how many future operations can potentially use each machine."""
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
        
        # 1. {Machines States}
        machine_states_str = ""
        for m in range(self.num_machines):
            occupying_info = ""
            buffer = ""
            free_at = self.machine_avail[m]
            
            # wait_time is 0 if the machine is already free, otherwise the difference
            wait_time = max(0.0, free_at - self.current_time)
            
            if m in self.broken_machines:
                status = "Broken"
                buffer = "   "
            elif free_at > self.current_time:
                status = "Busy"
                buffer = "     "
                occupant = self.machine_current_op.get(m)
                if occupant:
                    occupying_info = f", occupying_job={occupant[0]}, occupying_op={occupant[1]}"
            else:
                status = "Available"
                
            machine_states_str += f"- Machine {m}: status={status}{buffer}, free_at={free_at}, wait_time={wait_time}, contention={contention[m]}{occupying_info}\n"
            
        # 2. {Emergency Jobs}
        if self.emergency_jobs:
            emergencies_str = "- " + ", ".join([f"Job {j}" for j in self.emergency_jobs])
        else:
            emergencies_str = "- None"
            
        # 3. {Ready Operations}
        ready_ops_str = ""
        ready_jobs = set(a["job"] for a in actions)
        for job_id in ready_jobs:
            op_id = self.job_progress[job_id]
            candidates = self.jobs[job_id][op_id]
            flexibility = len(candidates)
            rem_work = self._calculate_rem_work(job_id)
            is_emerg = job_id in self.emergency_jobs
            
            # Minimum possible processing time across all machine candidates for this operation
            min_pt = min(cand["processing"] for cand in candidates)
            
            # Earliest Start Time (est): The minimum 'free_at' time among eligible machines.
            # We use max() to ensure it can never start in the past.
            valid_machines_avail = [self.machine_avail[cand["machine"]] for cand in candidates if cand["machine"] not in self.broken_machines]
            est = max(self.current_time, min(valid_machines_avail)) if valid_machines_avail else self.current_time
            
            ready_ops_str += f"- Job {job_id}, Op {op_id}: est={est}, min_pt={min_pt}, rem_work={rem_work}, flexibility={flexibility}, [EMERGENCY]={is_emerg}\n"
            
        # 4. {actions_json}
        clean_actions = [{"job": a["job"], "op": a["op"], "machine": a["machine"], "processing_time": a["processing_time"]} for a in actions]
        
        return {
            "timestamp": self.current_time,
            "machines_states": machine_states_str.strip(),
            "emergency_jobs": emergencies_str,
            "ready_operations": ready_ops_str.strip(),
            "actions_json": json.dumps(clean_actions, indent=2)
        }