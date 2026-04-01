import json
import heapq
import random
import os
import copy
from typing import List, Dict, Set, Any
import config

class StateManager:
    def __init__(self, problem_data: Dict[str, Any], start_time: float = 0.0):
        self.problem_data = problem_data
        self.num_machines = problem_data["machines"]
        self.jobs = {i: job for i, job in enumerate(problem_data["jobs"])}
        self.current_time = start_time
        
        self.event_queue = []
        self._event_counter = 0 
        
        self.machine_avail = {m: self.current_time for m in range(self.num_machines)}
        
        # CHANGED: Upgraded to a list to support queueing multiple operations
        self.machine_current_op = {m: [] for m in range(self.num_machines)}
        
        self.op_start_times = {} 
        self.op_expected_end_time = {}
        
        self.job_progress = {j: 0 for j in self.jobs.keys()}
        self.job_status = {j: 'idle' for j in self.jobs.keys()}
        
        self.broken_machines: Set[int] = set()
        self.emergency_jobs: Set[int] = set()
        
        self.interrupted_ops = {}
        self.completed_jobs = 0

        self._load_dynamic_events()

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result

        result.problem_data = self.problem_data
        result.jobs = self.jobs
        result.num_machines = self.num_machines

        result.current_time = self.current_time
        result._event_counter = self._event_counter
        result.completed_jobs = self.completed_jobs
        
        result.machine_avail = copy.deepcopy(self.machine_avail, memo)
        result.job_status = copy.deepcopy(self.job_status, memo)
        result.job_progress = copy.deepcopy(self.job_progress, memo)
        
        # This will flawlessly deepcopy the new Queue lists
        result.machine_current_op = copy.deepcopy(self.machine_current_op, memo)
        
        result.op_expected_end_time = copy.deepcopy(self.op_expected_end_time, memo)
        result.op_start_times = copy.deepcopy(self.op_start_times, memo)
        
        result.event_queue = copy.deepcopy(self.event_queue, memo)
        
        if hasattr(self, 'broken_machines'):
            result.broken_machines = copy.deepcopy(self.broken_machines, memo)
        if hasattr(self, 'interrupted_ops'):
            result.interrupted_ops = copy.deepcopy(self.interrupted_ops, memo)
        if hasattr(self, 'emergency_jobs'):
            result.emergency_jobs = copy.deepcopy(self.emergency_jobs, memo)

        return result

    def get_state_hash(self) -> str:
        status_tup = tuple(sorted(self.job_status.items()))
        progress_tup = tuple(sorted(self.job_progress.items()))
        avail_tup = tuple(sorted((m, round(t, 2)) for m, t in self.machine_avail.items()))
        return hash((status_tup, progress_tup, avail_tup))

    def get_feasible_actions(self) -> List[dict]:
        feasible_actions = []
        for job_id, status in self.job_status.items():
            if status == 'idle' and (job_id, self.job_progress[job_id]) not in self.interrupted_ops:
                op_id = self.job_progress[job_id]
                candidates = self.jobs[job_id][op_id]
                
                for cand in candidates:
                    machine_id = cand["machine"]
                    if machine_id not in self.broken_machines: 
                        actual_start = max(self.current_time, self.machine_avail[machine_id])
                        
                        feasible_actions.append({
                            "job": job_id, 
                            "op": op_id, 
                            "machine": machine_id, 
                            "processing_time": cand["processing"],
                            "start_time": actual_start,
                            "wait_time": actual_start - self.current_time 
                        })
        return feasible_actions

    def execute_action(self, job_id: int, op_id: int, machine_id: int):
        candidates = self.jobs[job_id][op_id]
        processing_time = next(cand["processing"] for cand in candidates if cand["machine"] == machine_id)
        
        start_time = max(self.current_time, self.machine_avail[machine_id])
        completion_time = start_time + processing_time
        
        self.machine_avail[machine_id] = completion_time
        
        self.job_status[job_id] = 'in_progress'
        
        # CHANGED: Append to queue instead of overwriting
        self.machine_current_op[machine_id].append((job_id, op_id))
        
        self.op_start_times[(job_id, op_id)] = start_time
        self.op_expected_end_time[(job_id, op_id)] = completion_time
        
        heapq.heappush(self.event_queue, (completion_time, self._event_counter, 'Operation_Completion', {'job_id': job_id, 'op_id': op_id, 'machine_id': machine_id}))
        self._event_counter += 1

    def process_next_event(self) -> tuple:
        if not self.event_queue:
            return None, None, None
            
        event_time, _, event_type, event_data = heapq.heappop(self.event_queue)
        self.current_time = max(self.current_time, event_time)
        
        if event_type == 'Operation_Completion':
            job_id = event_data['job_id']
            op_id = event_data['op_id']
            machine_id = event_data['machine_id']
            
            expected_end = self.op_expected_end_time.get((job_id, op_id))
            if expected_end is None or expected_end > self.current_time:
                return self.process_next_event()
            
            # CHANGED: Safely remove the completed op from the machine's queue list
            if (job_id, op_id) in self.machine_current_op[machine_id]:
                self.machine_current_op[machine_id].remove((job_id, op_id))
                
            self.job_status[job_id] = 'idle'
            self.job_progress[job_id] += 1
            
            if self.job_progress[job_id] >= len(self.jobs[job_id]):
                self.job_status[job_id] = 'completed'
                self.completed_jobs += 1
                
        elif event_type == 'Machine_Breakdown':
            self.break_machine(event_data['machine_id'])
        elif event_type == 'Machine_Repair':
            self.repair_machine(event_data['machine_id'])
        elif event_type == 'Job_Arrival':
            self._handle_job_arrival(event_data)
        elif event_type == 'Emergency_Job_Arrival':
            self.add_emergency_job(event_data['job_id'])
            
        return event_type, event_time, event_data

    def calculate_lower_bound(self) -> float:
        max_bound = self.current_time
        for job_id, status in self.job_status.items():
            if status != 'completed':
                rem_work = self._calculate_rem_work(job_id)
                job_bound = self.current_time + rem_work
                if job_bound > max_bound:
                    max_bound = job_bound
        return max_bound

    def _calculate_rem_work(self, job_id: int) -> float:
        if self.job_status[job_id] == 'completed':
            return 0.0
        
        rem_work = 0.0
        start_idx = self.job_progress[job_id]
        
        if self.job_status[job_id] == 'in_progress':
            op_id = start_idx
            if (job_id, op_id) in self.op_expected_end_time:
                expected_end = self.op_expected_end_time[(job_id, op_id)]
                rem_work += max(0.0, expected_end - self.current_time)
            start_idx += 1
            
        for op_id in range(start_idx, len(self.jobs[job_id])):
            candidates = self.jobs[job_id][op_id]
            rem_work += min(cand["processing"] for cand in candidates)
            
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

    def break_machine(self, machine_id: int):
        if machine_id in self.broken_machines:
            return
        self.broken_machines.add(machine_id)
        
        # CHANGED: Interrupt the currently active job (the first one in the queue)
        if self.machine_current_op[machine_id]:
            current_op = self.machine_current_op[machine_id][0]
            job_id, op_id = current_op
            
            rem_work = self.op_expected_end_time[(job_id, op_id)] - self.current_time
            if rem_work > 0:
                self.interrupted_ops[(job_id, op_id)] = rem_work
            
            self.job_status[job_id] = 'idle'
            self.machine_current_op[machine_id].remove(current_op)

    def repair_machine(self, machine_id: int):
        if machine_id in self.broken_machines:
            self.broken_machines.remove(machine_id)
            self.machine_avail[machine_id] = self.current_time

    def add_emergency_job(self, job_id: int):
        self.emergency_jobs.add(job_id)

    def _load_dynamic_events(self):
        events_file = getattr(config, 'DYNAMIC_EVENTS_FILE', '')
        if not events_file:
            return

        if not os.path.exists(events_file):
            print(f"Warning: Dynamic events file '{events_file}' not found. Skipping.")
            return

        with open(events_file, 'r', encoding='utf-8') as f:
            events = json.load(f)

        for event in events:
            timestamp = event['timestamp']
            event_type = event['event_type']
            data = event['data']
            heapq.heappush(
                self.event_queue,
                (timestamp, self._event_counter, event_type, data)
            )
            self._event_counter += 1

    def _handle_job_arrival(self, event_data: dict):
        job_id = event_data['job_id']
        operations = event_data['operations']

        self.jobs[job_id] = operations
        self.job_progress[job_id] = 0
        self.job_status[job_id] = 'idle'

        if event_data.get('is_emergency', False):
            self.emergency_jobs.add(job_id)

    def compile_prompt_elements(self) -> Dict[str, str]:
        machine_states_str = "Machine States:\n"
        contention = self._calculate_machine_contention()
        for m in range(self.num_machines):
            if m in self.broken_machines:
                status = "BROKEN"
            else:
                queue = self.machine_current_op.get(m, [])
                if queue:
                    active_jobs = []
                    queued_jobs = []
                    
                    # CHANGED: Separate the active job from the waiting queue
                    for (j_id, o_id) in queue:
                        start_t = self.op_start_times.get((j_id, o_id), 0)
                        if start_t <= self.current_time:
                            active_jobs.append(f"Job {j_id} (Op {o_id})")
                        else:
                            queued_jobs.append(f"Job {j_id} (Op {o_id})")
                    
                    parts = []
                    if active_jobs:
                        parts.append(f"Processing {', '.join(active_jobs)}")
                    else:
                        parts.append("Available")
                        
                    if queued_jobs:
                        parts.append(f"Queue: {', '.join(queued_jobs)}")
                        
                    status = " | ".join(parts)
                else:
                    status = "Available"
            
            machine_states_str += f"- Machine {m}: {status}, Available from T={max(self.current_time, self.machine_avail[m]):.1f}, Contention: {contention[m]}\n"

        ready_ops_str = "Ready Operations:\n"
        actions = self.get_feasible_actions()
        
        rem_works = [self._calculate_rem_work(a["job"]) for a in actions]
        max_rem_work = max(rem_works) if rem_works else 0
        
        seen_ops = set()
        for a in actions:
            job_id, op_id = a["job"], a["op"]
            if (job_id, op_id) in seen_ops:
                continue
            seen_ops.add((job_id, op_id))
            
            candidates = self.jobs[job_id][op_id]
            flexibility = len(candidates)
            rem_work = self._calculate_rem_work(job_id)
            is_emerg = job_id in self.emergency_jobs
            
            is_critical = (rem_work == max_rem_work) and (rem_work > 0)
            
            min_pt = min(cand["processing"] for cand in candidates)
            
            valid_machines_avail = [self.machine_avail[cand["machine"]] for cand in candidates if cand["machine"] not in self.broken_machines]
            est = max(self.current_time, min(valid_machines_avail)) if valid_machines_avail else self.current_time
            
            ready_ops_str += f"- Job {job_id}, Op {op_id}: est={est:.1f}, min_pt={min_pt}, rem_work={rem_work}, flexibility={flexibility}, is_critical={is_critical}, [EMERGENCY]={is_emerg}\n"
            
        clean_actions = [{"job": a["job"], "op": a["op"], "machine": a["machine"], "processing_time": a["processing_time"], "wait_time": a["wait_time"], "is_critical": (self._calculate_rem_work(a["job"]) == max_rem_work)} for a in actions]
            
        state_lower_bound = self.calculate_lower_bound()

        result = {
            "timestamp": self.current_time,
            "machines_states": machine_states_str.strip(),
            "emergency_jobs": str(list(self.emergency_jobs)),
            "ready_operations": ready_ops_str.strip(),
            "actions_json": json.dumps(clean_actions, indent=2),
            "lower_bound": f"{state_lower_bound:.2f}",
            "full_state": "",
        }

        if getattr(config, "INCLUDE_FULL_STATE_IN_PROMPT", False):
            result["full_state"] = self._compile_full_state_table()

        return result

    def _compile_full_state_table(self) -> str:
        """Build a complete processing-time table for every job and operation.

        Format per operation:
            Job 0, Op 1 [PENDING]: M0=3, M2=5, M4=2
            Job 0, Op 0 [COMPLETED]: M0=3, M2=5, M4=2

        This gives the LLM full visibility into machine-specific costs and
        remaining routing flexibility for all jobs, not just ready operations.
        """
        lines: List[str] = ["Complete Processing Time Table:"]

        for job_id in sorted(self.jobs.keys()):
            operations = self.jobs[job_id]
            job_status = self.job_status[job_id]
            progress = self.job_progress[job_id]

            for op_id, candidates in enumerate(operations):
                if op_id < progress:
                    status_tag = "DONE"
                elif op_id == progress and job_status == "in_progress":
                    status_tag = "IN_PROGRESS"
                elif op_id == progress and job_status == "idle":
                    status_tag = "READY"
                else:
                    status_tag = "WAITING"

                machine_costs = ", ".join(
                    f"M{c['machine']}={c['processing']}"
                    for c in candidates
                )
                lines.append(
                    f"- Job {job_id}, Op {op_id} [{status_tag}]: {machine_costs}"
                )

        return "\n".join(lines)