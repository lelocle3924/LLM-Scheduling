import heapq
import json
from typing import Any, Dict, List, Set

import numpy as np

import config
from utilities.numeric_precision import cap_numeric_precision, dumps_capped, format_decimal


class ArrayTreeState:
    """Array-backed simulation state for fast MCTS tree cloning and transitions."""

    STATUS_IDLE = 0
    STATUS_IN_PROGRESS = 1
    STATUS_COMPLETED = 2

    EVENT_OPERATION_COMPLETION = 0
    EVENT_MACHINE_BREAKDOWN = 1
    EVENT_MACHINE_REPAIR = 2
    EVENT_JOB_ARRIVAL = 3
    EVENT_EMERGENCY_JOB_ARRIVAL = 4
    EVENT_JOB_CANCELLATION = 5

    def __init__(
        self,
        *,
        problem_data: Dict[str, Any],
        num_machines: int,
        job_ids: List[int],
        due_dates: np.ndarray,
        job_completion_times: np.ndarray,
        operation_count_per_job: np.ndarray,
        candidate_machine_ids: np.ndarray,
        candidate_processing_times: np.ndarray,
        candidate_count_per_op: np.ndarray,
        current_time: float,
        machine_avail: np.ndarray,
        job_progress: np.ndarray,
        job_status_codes: np.ndarray,
        event_queue: List[tuple],
        event_counter: int,
        broken_machine_mask: np.ndarray,
        emergency_jobs: Set[int],
        supports_dynamic_features: bool,
    ):
        self.problem_data = problem_data
        self.num_machines = num_machines
        self.job_ids = np.asarray(job_ids, dtype=np.int32)
        self.job_ids_list = [int(job_id) for job_id in self.job_ids.tolist()]
        self.job_id_to_index = {job_id: index for index, job_id in enumerate(self.job_ids_list)}
        self.due_dates = due_dates
        self.job_completion_times = job_completion_times
        self.operation_count_per_job = operation_count_per_job
        self.candidate_machine_ids = candidate_machine_ids
        self.candidate_processing_times = candidate_processing_times
        self.candidate_count_per_op = candidate_count_per_op

        self.current_time = float(current_time)
        self.machine_avail = machine_avail
        self.job_progress = job_progress
        self.job_status_codes = job_status_codes
        self.event_queue = event_queue
        self._event_counter = int(event_counter)
        self.broken_machine_mask = broken_machine_mask
        self.emergency_jobs = set(emergency_jobs)
        self.supports_dynamic_features = supports_dynamic_features
        self.completed_jobs = int(np.count_nonzero(self.job_status_codes == self.STATUS_COMPLETED))

    @classmethod
    def from_state_manager(cls, state_manager) -> "ArrayTreeState":
        job_ids = sorted(state_manager.jobs.keys())
        job_count = len(job_ids)
        job_id_to_index = {job_id: index for index, job_id in enumerate(job_ids)}
        due_dates_raw = getattr(state_manager, "problem_data", {}).get("due_dates", {})
        due_dates = np.full(job_count, np.inf, dtype=np.float64)
        job_completion_times = np.full(job_count, -1.0, dtype=np.float64)
        for job_id in job_ids:
            job_index = job_id_to_index[job_id]
            due_date_value = None
            if isinstance(due_dates_raw, dict):
                if job_id in due_dates_raw:
                    due_date_value = due_dates_raw[job_id]
                elif str(job_id) in due_dates_raw:
                    due_date_value = due_dates_raw[str(job_id)]
            if due_date_value is not None:
                due_dates[job_index] = float(due_date_value)
            if state_manager.job_status.get(job_id) == "completed":
                job_completion_times[job_index] = float(state_manager.current_time)

        operation_count_per_job = np.array(
            [len(state_manager.jobs[job_id]) for job_id in job_ids],
            dtype=np.int32,
        )
        max_operations = int(operation_count_per_job.max()) if job_count > 0 else 0
        max_candidates = 1
        for job_id in job_ids:
            for operation_candidates in state_manager.jobs[job_id]:
                max_candidates = max(max_candidates, len(operation_candidates))

        candidate_machine_ids = np.full((job_count, max_operations, max_candidates), -1, dtype=np.int32)
        candidate_processing_times = np.zeros((job_count, max_operations, max_candidates), dtype=np.float64)
        candidate_count_per_op = np.zeros((job_count, max_operations), dtype=np.int32)

        for job_id in job_ids:
            job_index = job_id_to_index[job_id]
            for operation_index, operation_candidates in enumerate(state_manager.jobs[job_id]):
                candidate_count_per_op[job_index, operation_index] = len(operation_candidates)
                for candidate_index, candidate in enumerate(operation_candidates):
                    candidate_machine_ids[job_index, operation_index, candidate_index] = int(candidate["machine"])
                    candidate_processing_times[job_index, operation_index, candidate_index] = float(candidate["processing"])

        machine_avail = np.array(
            [float(state_manager.machine_avail[machine_index]) for machine_index in range(state_manager.num_machines)],
            dtype=np.float64,
        )
        job_progress = np.array(
            [int(state_manager.job_progress[job_id]) for job_id in job_ids],
            dtype=np.int32,
        )
        job_status_codes = np.zeros(job_count, dtype=np.int8)
        for job_id in job_ids:
            status_text = state_manager.job_status[job_id]
            job_index = job_id_to_index[job_id]
            if status_text == "idle":
                job_status_codes[job_index] = cls.STATUS_IDLE
            elif status_text == "in_progress":
                job_status_codes[job_index] = cls.STATUS_IN_PROGRESS
            else:
                job_status_codes[job_index] = cls.STATUS_COMPLETED

        broken_machine_mask = np.zeros(state_manager.num_machines, dtype=np.bool_)
        for machine_index in state_manager.broken_machines:
            if 0 <= machine_index < state_manager.num_machines:
                broken_machine_mask[machine_index] = True

        supports_dynamic_features = True
        encoded_event_queue = []
        for event_time, event_counter, event_type, event_data in state_manager.event_queue:
            # Prevent Data Leakage: Hide future stochastic events from the tree/rollouts
            if event_type not in {"Operation_Completion", "Machine_Repair"} and event_time > state_manager.current_time:
                continue
            event_type_code = cls._map_event_type_to_code(event_type)
            if event_type_code is None:
                supports_dynamic_features = False
                continue
            encoded_event = cls._encode_event_payload(
                event_time=event_time,
                event_counter=event_counter,
                event_type_code=event_type_code,
                event_data=event_data,
                job_id_to_index=job_id_to_index,
            )
            if encoded_event is None:
                supports_dynamic_features = False
                continue
            encoded_event_queue.append(encoded_event)
        heapq.heapify(encoded_event_queue)

        return cls(
            problem_data=state_manager.problem_data,
            num_machines=state_manager.num_machines,
            job_ids=job_ids,
            due_dates=due_dates,
            job_completion_times=job_completion_times,
            operation_count_per_job=operation_count_per_job,
            candidate_machine_ids=candidate_machine_ids,
            candidate_processing_times=candidate_processing_times,
            candidate_count_per_op=candidate_count_per_op,
            current_time=float(state_manager.current_time),
            machine_avail=machine_avail,
            job_progress=job_progress,
            job_status_codes=job_status_codes,
            event_queue=encoded_event_queue,
            event_counter=int(state_manager._event_counter),
            broken_machine_mask=broken_machine_mask,
            emergency_jobs=set(getattr(state_manager, "emergency_jobs", set())),
            supports_dynamic_features=supports_dynamic_features,
        )

    @classmethod
    def _map_event_type_to_code(cls, event_type: str):
        event_mapping = {
            "Operation_Completion": cls.EVENT_OPERATION_COMPLETION,
            "Machine_Breakdown": cls.EVENT_MACHINE_BREAKDOWN,
            "Machine_Repair": cls.EVENT_MACHINE_REPAIR,
            "Job_Arrival": cls.EVENT_JOB_ARRIVAL,
            "Emergency_Job_Arrival": cls.EVENT_EMERGENCY_JOB_ARRIVAL,
            "Job_Cancellation": cls.EVENT_JOB_CANCELLATION,
        }
        return event_mapping.get(str(event_type))

    @classmethod
    def _encode_event_payload(
        cls,
        *,
        event_time: float,
        event_counter: int,
        event_type_code: int,
        event_data: Dict[str, Any],
        job_id_to_index: Dict[int, int],
    ):
        payload = event_data or {}
        if event_type_code == cls.EVENT_OPERATION_COMPLETION:
            job_id = int(payload.get("job_id"))
            if job_id not in job_id_to_index:
                return None
            return (
                float(event_time),
                int(event_counter),
                event_type_code,
                {
                    "job_index": int(job_id_to_index[job_id]),
                    "job_id": job_id,
                    "op_id": int(payload.get("op_id")),
                    "machine_id": int(payload.get("machine_id")),
                },
            )
        return (
            float(event_time),
            int(event_counter),
            event_type_code,
            dict(payload),
        )

    def clone(self) -> "ArrayTreeState":
        cloned_state = self.__class__.__new__(self.__class__)
        cloned_state.problem_data = self.problem_data
        cloned_state.num_machines = self.num_machines
        cloned_state.job_ids = self.job_ids.copy()
        cloned_state.job_ids_list = list(self.job_ids_list)
        cloned_state.job_id_to_index = dict(self.job_id_to_index)
        cloned_state.due_dates = self.due_dates.copy()
        cloned_state.job_completion_times = self.job_completion_times.copy()
        cloned_state.operation_count_per_job = self.operation_count_per_job.copy()
        cloned_state.candidate_machine_ids = self.candidate_machine_ids.copy()
        cloned_state.candidate_processing_times = self.candidate_processing_times.copy()
        cloned_state.candidate_count_per_op = self.candidate_count_per_op.copy()

        cloned_state.current_time = self.current_time
        cloned_state.machine_avail = self.machine_avail.copy()
        cloned_state.job_progress = self.job_progress.copy()
        cloned_state.job_status_codes = self.job_status_codes.copy()
        cloned_state.event_queue = list(self.event_queue)
        cloned_state._event_counter = self._event_counter
        cloned_state.broken_machine_mask = self.broken_machine_mask.copy()
        cloned_state.emergency_jobs = set(self.emergency_jobs)
        cloned_state.supports_dynamic_features = self.supports_dynamic_features
        cloned_state.completed_jobs = self.completed_jobs
        return cloned_state

    def all_jobs_completed(self) -> bool:
        return self.completed_jobs >= self.job_ids.shape[0]

    def _calculate_rem_work(self, job_id: int) -> float:
        job_index = self.job_id_to_index[job_id]
        if self.job_status_codes[job_index] == self.STATUS_COMPLETED:
            return 0.0

        remaining_work = 0.0
        start_operation_index = int(self.job_progress[job_index])
        if self.job_status_codes[job_index] == self.STATUS_IN_PROGRESS:
            start_operation_index += 1

        for operation_index in range(start_operation_index, int(self.operation_count_per_job[job_index])):
            candidate_count = int(self.candidate_count_per_op[job_index, operation_index])
            minimum_processing_time = float("inf")
            for candidate_index in range(candidate_count):
                processing_time = float(self.candidate_processing_times[job_index, operation_index, candidate_index])
                if processing_time < minimum_processing_time:
                    minimum_processing_time = processing_time
            if minimum_processing_time != float("inf"):
                remaining_work += minimum_processing_time
        return remaining_work

    def calculate_lower_bound(self) -> float:
        tardiness_objective = str(getattr(config, "TARDINESS_OBJECTIVE", "total")).lower()
        total_tardiness_lower_bound = 0.0
        maximum_tardiness_lower_bound = 0.0
        for job_index, job_id in enumerate(self.job_ids_list):
            if self.job_status_codes[job_index] == self.STATUS_COMPLETED:
                continue
            remaining_work = self._calculate_rem_work(job_id)
            minimum_completion_time = self.current_time + remaining_work
            due_date = float(self.due_dates[job_index])
            if np.isfinite(due_date):
                minimum_tardiness = max(0.0, minimum_completion_time - due_date)
                total_tardiness_lower_bound += minimum_tardiness
                if minimum_tardiness > maximum_tardiness_lower_bound:
                    maximum_tardiness_lower_bound = minimum_tardiness
        if tardiness_objective == "max":
            return maximum_tardiness_lower_bound
        return total_tardiness_lower_bound

    def calculate_actual_tardiness(self) -> float:
        tardiness_objective = str(getattr(config, "TARDINESS_OBJECTIVE", "total")).lower()
        total_tardiness = 0.0
        maximum_tardiness = 0.0
        for job_index in range(self.job_ids.shape[0]):
            completion_time = float(self.job_completion_times[job_index])
            if completion_time < 0.0:
                completion_time = self.current_time
            due_date = float(self.due_dates[job_index])
            if not np.isfinite(due_date):
                continue
            tardiness = max(0.0, completion_time - due_date)
            total_tardiness += tardiness
            if tardiness > maximum_tardiness:
                maximum_tardiness = tardiness
        if tardiness_objective == "max":
            return maximum_tardiness
        return total_tardiness

    def get_state_hash(self) -> str:
        machine_avail_rounded = tuple(np.round(self.machine_avail, 2).tolist())
        return hash(
            (
                tuple(self.job_progress.tolist()),
                tuple(self.job_status_codes.tolist()),
                machine_avail_rounded,
            )
        )

    def _calculate_machine_contention(self) -> dict:
        contention = {machine_index: 0 for machine_index in range(self.num_machines)}
        for job_index, job_id in enumerate(self.job_ids_list):
            if self.job_status_codes[job_index] == self.STATUS_COMPLETED:
                continue
            start_operation_index = int(self.job_progress[job_index])
            if self.job_status_codes[job_index] == self.STATUS_IN_PROGRESS:
                start_operation_index += 1
            operation_count = int(self.operation_count_per_job[job_index])
            for operation_index in range(start_operation_index, operation_count):
                candidate_count = int(self.candidate_count_per_op[job_index, operation_index])
                for candidate_index in range(candidate_count):
                    machine_index = int(self.candidate_machine_ids[job_index, operation_index, candidate_index])
                    if 0 <= machine_index < self.num_machines:
                        contention[machine_index] += 1
        return contention

    def _build_machine_processing_view(self) -> dict:
        machine_processing = {machine_index: [] for machine_index in range(self.num_machines)}
        for event in self.event_queue:
            if not isinstance(event, tuple) or len(event) != 4:
                continue
            _, _, event_type, event_data = event
            if int(event_type) != self.EVENT_OPERATION_COMPLETION:
                continue
            payload = event_data or {}
            job_index = int(payload.get("job_index", -1))
            machine_index = int(payload.get("machine_id", -1))
            op_id = int(payload.get("op_id", -1))
            if (
                job_index < 0
                or job_index >= self.job_status_codes.shape[0]
                or machine_index < 0
                or machine_index >= self.num_machines
            ):
                continue
            if self.job_status_codes[job_index] != self.STATUS_IN_PROGRESS:
                continue
            job_id = int(self.job_ids[job_index])
            machine_processing[machine_index].append((job_id, op_id))
        return machine_processing

    def get_feasible_actions(self) -> List[dict]:
        feasible_actions: List[dict] = []
        for job_index, job_id in enumerate(self.job_ids_list):
            if self.job_status_codes[job_index] != self.STATUS_IDLE:
                continue

            operation_index = int(self.job_progress[job_index])
            if operation_index >= int(self.operation_count_per_job[job_index]):
                continue

            candidate_count = int(self.candidate_count_per_op[job_index, operation_index])
            for candidate_index in range(candidate_count):
                machine_index = int(self.candidate_machine_ids[job_index, operation_index, candidate_index])
                if machine_index < 0 or self.broken_machine_mask[machine_index]:
                    continue
                processing_time = float(self.candidate_processing_times[job_index, operation_index, candidate_index])
                start_time = max(self.current_time, float(self.machine_avail[machine_index]))
                feasible_actions.append(
                    {
                        "job": int(job_id),
                        "op": int(operation_index),
                        "machine": int(machine_index),
                        "processing_time": processing_time,
                        "start_time": start_time,
                        "wait_time": start_time - self.current_time,
                        "due_date": None if not np.isfinite(self.due_dates[job_index]) else float(self.due_dates[job_index]),
                    }
                )
        return feasible_actions

    def execute_action(self, job_id: int, op_id: int, machine_id: int):
        job_index = self.job_id_to_index[job_id]
        candidate_count = int(self.candidate_count_per_op[job_index, op_id])

        processing_time = None
        for candidate_index in range(candidate_count):
            candidate_machine = int(self.candidate_machine_ids[job_index, op_id, candidate_index])
            if candidate_machine == machine_id:
                processing_time = float(self.candidate_processing_times[job_index, op_id, candidate_index])
                break

        if processing_time is None:
            raise ValueError(f"Machine {machine_id} is not valid for job={job_id}, op={op_id}.")

        start_time = max(self.current_time, float(self.machine_avail[machine_id]))
        completion_time = start_time + processing_time
        self.machine_avail[machine_id] = completion_time
        self.job_status_codes[job_index] = self.STATUS_IN_PROGRESS
        heapq.heappush(
            self.event_queue,
            (
                completion_time,
                self._event_counter,
                self.EVENT_OPERATION_COMPLETION,
                {
                    "job_index": int(job_index),
                    "job_id": int(job_id),
                    "op_id": int(op_id),
                    "machine_id": int(machine_id),
                },
            ),
        )
        self._event_counter += 1

    def process_next_event(self):
        if not self.event_queue:
            return None, None, None

        event_time, _, event_type, event_data = heapq.heappop(self.event_queue)
        self.current_time = max(self.current_time, float(event_time))
        if event_type == self.EVENT_OPERATION_COMPLETION:
            job_index = int(event_data["job_index"])
            op_id = int(event_data["op_id"])
            machine_id = int(event_data["machine_id"])
            if self.job_status_codes[job_index] != self.STATUS_IN_PROGRESS:
                return self.process_next_event()
            self.job_progress[job_index] += 1
            if self.job_progress[job_index] >= self.operation_count_per_job[job_index]:
                self.job_status_codes[job_index] = self.STATUS_COMPLETED
                self.job_completion_times[job_index] = self.current_time
                self.completed_jobs += 1
            else:
                self.job_status_codes[job_index] = self.STATUS_IDLE
            job_id = int(self.job_ids[job_index])
            return "Operation_Completion", event_time, {"job_id": job_id, "op_id": op_id, "machine_id": machine_id}

        if event_type == self.EVENT_MACHINE_BREAKDOWN:
            machine_id = int((event_data or {}).get("machine_id", -1))
            if 0 <= machine_id < self.num_machines:
                self.broken_machine_mask[machine_id] = True
            return "Machine_Breakdown", event_time, {"machine_id": machine_id}

        if event_type == self.EVENT_MACHINE_REPAIR:
            machine_id = int((event_data or {}).get("machine_id", -1))
            if 0 <= machine_id < self.num_machines:
                self.broken_machine_mask[machine_id] = False
                self.machine_avail[machine_id] = self.current_time
            return "Machine_Repair", event_time, {"machine_id": machine_id}

        if event_type == self.EVENT_JOB_ARRIVAL:
            self._handle_job_arrival(event_data=event_data, is_emergency=False)
            return "Job_Arrival", event_time, dict(event_data or {})

        if event_type == self.EVENT_EMERGENCY_JOB_ARRIVAL:
            self._handle_job_arrival(event_data=event_data, is_emergency=True)
            return "Emergency_Job_Arrival", event_time, dict(event_data or {})

        if event_type == self.EVENT_JOB_CANCELLATION:
            return "Job_Cancellation", event_time, dict(event_data or {})

        return None, None, None

    def _handle_job_arrival(self, event_data: Dict[str, Any], is_emergency: bool) -> None:
        payload = dict(event_data or {})
        if "job_id" not in payload or "operations" not in payload:
            return

        job_id = int(payload["job_id"])
        operations = payload.get("operations", [])
        if not isinstance(operations, list) or not operations:
            return

        sanitized_operations = []
        for operation_candidates in operations:
            if not isinstance(operation_candidates, list):
                continue
            valid_candidates = []
            for candidate in operation_candidates:
                machine_id = int(candidate.get("machine", -1))
                if 0 <= machine_id < self.num_machines:
                    valid_candidates.append(
                        {
                            "machine": machine_id,
                            "processing": float(candidate.get("processing", 0.0)),
                        }
                    )
            if valid_candidates:
                sanitized_operations.append(valid_candidates)
        if not sanitized_operations:
            return

        if job_id in self.job_id_to_index:
            return

        self._append_dynamic_job(job_id, sanitized_operations, payload.get("due_date"))
        if is_emergency or bool(payload.get("is_emergency", False)):
            self.emergency_jobs.add(job_id)

    def _append_dynamic_job(self, job_id: int, operations: List[List[Dict[str, float]]], due_date_value) -> None:
        new_job_count = self.job_ids.shape[0] + 1
        required_operations = len(operations)
        required_candidates = max(len(operation_candidates) for operation_candidates in operations)
        self._ensure_candidate_capacity(
            target_job_count=new_job_count,
            target_operation_count=required_operations,
            target_candidate_count=required_candidates,
        )

        job_index = self.job_ids.shape[0]
        self.job_ids = np.concatenate([self.job_ids, np.array([job_id], dtype=np.int32)])
        self.job_ids_list.append(job_id)
        self.job_id_to_index[job_id] = job_index
        self.due_dates = np.concatenate(
            [self.due_dates, np.array([np.inf if due_date_value is None else float(due_date_value)], dtype=np.float64)]
        )
        self.job_completion_times = np.concatenate([self.job_completion_times, np.array([-1.0], dtype=np.float64)])
        self.operation_count_per_job = np.concatenate(
            [self.operation_count_per_job, np.array([required_operations], dtype=np.int32)]
        )
        self.job_progress = np.concatenate([self.job_progress, np.array([0], dtype=np.int32)])
        self.job_status_codes = np.concatenate([self.job_status_codes, np.array([self.STATUS_IDLE], dtype=np.int8)])

        for operation_index, operation_candidates in enumerate(operations):
            self.candidate_count_per_op[job_index, operation_index] = len(operation_candidates)
            for candidate_index, candidate in enumerate(operation_candidates):
                self.candidate_machine_ids[job_index, operation_index, candidate_index] = int(candidate["machine"])
                self.candidate_processing_times[job_index, operation_index, candidate_index] = float(candidate["processing"])

    def _ensure_candidate_capacity(
        self,
        *,
        target_job_count: int,
        target_operation_count: int,
        target_candidate_count: int,
    ) -> None:
        current_jobs, current_operations, current_candidates = self.candidate_machine_ids.shape
        required_jobs = max(current_jobs, target_job_count)
        required_operations = max(current_operations, target_operation_count)
        required_candidates = max(current_candidates, target_candidate_count)

        if (
            required_jobs == current_jobs
            and required_operations == current_operations
            and required_candidates == current_candidates
        ):
            return

        expanded_machine_ids = np.full(
            (required_jobs, required_operations, required_candidates),
            -1,
            dtype=np.int32,
        )
        expanded_processing_times = np.zeros(
            (required_jobs, required_operations, required_candidates),
            dtype=np.float64,
        )
        expanded_candidate_counts = np.zeros(
            (required_jobs, required_operations),
            dtype=np.int32,
        )

        expanded_machine_ids[:current_jobs, :current_operations, :current_candidates] = self.candidate_machine_ids
        expanded_processing_times[:current_jobs, :current_operations, :current_candidates] = self.candidate_processing_times
        expanded_candidate_counts[:current_jobs, :current_operations] = self.candidate_count_per_op

        self.candidate_machine_ids = expanded_machine_ids
        self.candidate_processing_times = expanded_processing_times
        self.candidate_count_per_op = expanded_candidate_counts

    def compile_prompt_elements(self) -> Dict[str, str]:
        machine_states = []
        slack_guidance = (
            "Slack Guidance:\n"
            "- slack = due_date - current_time - rem_work\n"
            "- Smaller slack means higher urgency\n"
            "- Negative slack means the job is mathematically guaranteed to be tardy"
        )
        contention = self._calculate_machine_contention()
        processing_view = self._build_machine_processing_view()
        for machine_index in range(self.num_machines):
            if self.broken_machine_mask[machine_index]:
                machine_status = "BROKEN"
            else:
                active_operations = processing_view.get(machine_index, [])
                if active_operations:
                    machine_status = "Processing " + ", ".join(
                        f"Job {job_id} (Op {op_id})" for job_id, op_id in active_operations
                    )
                else:
                    machine_status = "Available"
            available_from = max(self.current_time, float(self.machine_avail[machine_index]))
            machine_states.append(
                f"- Machine {machine_index}: {machine_status}, Available from T={available_from:.1f}, Contention: {contention[machine_index]}"
            )

        actions = self.get_feasible_actions()
        rem_works = [self._calculate_rem_work(action["job"]) for action in actions]
        max_rem_work = max(rem_works) if rem_works else 0.0
        ready_operations = []
        seen_operations = set()
        for action in actions:
            operation_key = (action["job"], action["op"])
            if operation_key in seen_operations:
                continue
            seen_operations.add(operation_key)

            job_id = int(action["job"])
            op_id = int(action["op"])
            job_index = self.job_id_to_index[job_id]
            rem_work = self._calculate_rem_work(job_id)
            candidate_count = int(self.candidate_count_per_op[job_index, op_id])
            min_processing_time = float("inf")
            valid_machine_availability = []
            for candidate_index in range(candidate_count):
                candidate_machine = int(self.candidate_machine_ids[job_index, op_id, candidate_index])
                candidate_processing = float(self.candidate_processing_times[job_index, op_id, candidate_index])
                if candidate_processing < min_processing_time:
                    min_processing_time = candidate_processing
                if (
                    0 <= candidate_machine < self.num_machines
                    and not self.broken_machine_mask[candidate_machine]
                ):
                    valid_machine_availability.append(float(self.machine_avail[candidate_machine]))
            est = (
                max(self.current_time, min(valid_machine_availability))
                if valid_machine_availability
                else self.current_time
            )
            due_date = action.get("due_date")
            slack = None if due_date is None else float(due_date) - self.current_time - rem_work
            due_date_text = "None" if due_date is None else format_decimal(float(due_date))
            slack_text = "None" if slack is None else format_decimal(float(slack))
            ready_operations.append(
                f"- Job {job_id}, Op {op_id}: est={format_decimal(est)}, min_pt={format_decimal(min_processing_time)}, "
                f"rem_work={format_decimal(rem_work)}, due_date={due_date_text}, slack={slack_text}, "
                f"flexibility={candidate_count}, is_critical={rem_work == max_rem_work and rem_work > 0}, "
                f"[EMERGENCY]={job_id in self.emergency_jobs}"
            )

        clean_actions = []
        for action in actions:
            job_id = int(action["job"])
            action_rem_work = self._calculate_rem_work(job_id)
            action_due_date = action.get("due_date")
            action_slack = (
                None
                if action_due_date is None
                else float(action_due_date) - self.current_time - action_rem_work
            )
            clean_actions.append(
                {
                    "job": job_id,
                    "op": int(action["op"]),
                    "machine": int(action["machine"]),
                    "processing_time": float(action["processing_time"]),
                    "wait_time": float(action["wait_time"]),
                    "due_date": action_due_date,
                    "slack": action_slack,
                    "is_critical": (action_rem_work == max_rem_work),
                }
            )

        actions_json = dumps_capped(clean_actions, indent=2)
        full_state = ""
        if getattr(config, "INCLUDE_FULL_STATE_IN_PROMPT", False):
            lines = ["Complete Processing Time Table:"]
            for job_index, job_id in enumerate(self.job_ids_list):
                for operation_index in range(int(self.operation_count_per_job[job_index])):
                    candidate_count = int(self.candidate_count_per_op[job_index, operation_index])
                    candidate_text = []
                    for candidate_index in range(candidate_count):
                        machine_index = int(self.candidate_machine_ids[job_index, operation_index, candidate_index])
                        processing_time = float(self.candidate_processing_times[job_index, operation_index, candidate_index])
                        candidate_text.append(f"M{machine_index}={format_decimal(processing_time)}")
                    if operation_index < int(self.job_progress[job_index]):
                        status_tag = "DONE"
                    elif (
                        operation_index == int(self.job_progress[job_index])
                        and self.job_status_codes[job_index] == self.STATUS_IN_PROGRESS
                    ):
                        status_tag = "IN_PROGRESS"
                    elif (
                        operation_index == int(self.job_progress[job_index])
                        and self.job_status_codes[job_index] == self.STATUS_IDLE
                    ):
                        status_tag = "READY"
                    else:
                        status_tag = "WAITING"
                    lines.append(
                        f"- Job {job_id}, Op {operation_index} [{status_tag}]: {', '.join(candidate_text)}"
                    )
            full_state = "\n".join(lines)

        return {
            "timestamp": cap_numeric_precision(self.current_time),
            "machines_states": "Machine States:\n" + "\n".join(machine_states),
            "emergency_jobs": str(sorted(list(self.emergency_jobs))),
            "ready_operations": "Ready Operations:\n" + ("\n".join(ready_operations) if ready_operations else "None"),
            "actions_json": actions_json,
            "lower_bound": format_decimal(self.calculate_lower_bound()),
            "slack_guidance": slack_guidance,
            "full_state": full_state,
        }
