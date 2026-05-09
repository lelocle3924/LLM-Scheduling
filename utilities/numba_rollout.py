from typing import Any, Dict, Optional, Tuple

import numpy as np
import config

try:
    from numba import njit

    NUMBA_AVAILABLE = True
except Exception:
    NUMBA_AVAILABLE = False

    def njit(*args, **kwargs):
        def wrapper(func):
            return func

        return wrapper


_PROBLEM_ARRAY_CACHE: Dict[Tuple[Any, ...], Dict[str, Any]] = {}


@njit(cache=True)
def _compute_remaining_work_for_idle_job(
    job_index,
    job_progress,
    operation_count_per_job,
    candidate_count_per_op,
    candidate_processing_times,
):
    remaining_work = 0.0
    start_operation_index = job_progress[job_index]
    for operation_index in range(start_operation_index, operation_count_per_job[job_index]):
        candidate_count = candidate_count_per_op[job_index, operation_index]
        minimum_processing_time = 1e30
        for candidate_index in range(candidate_count):
            processing_time = candidate_processing_times[job_index, operation_index, candidate_index]
            if processing_time < minimum_processing_time:
                minimum_processing_time = processing_time
        if minimum_processing_time < 1e29:
            remaining_work += minimum_processing_time
    return remaining_work


@njit(cache=True)
def _select_action_index(
    policy_mode,
    feasible_count,
    feasible_job,
    feasible_start_time,
    feasible_processing,
    due_dates,
    current_time,
    job_progress,
    operation_count_per_job,
    candidate_count_per_op,
    candidate_processing_times,
    g_parameter=0.5,
):
    # Mode 0: random rollout.
    if policy_mode == 0:
        return np.random.randint(feasible_count)

    # Mode 1: PDR-mixture rollout with equal rule weights.
    # 0=SPT, 1=ATC, 2=MDD, 3=LWR, 4=EST
    selected_rule_index = np.random.randint(5)
    best_index = 0

    if selected_rule_index == 0:
        best_value = feasible_processing[0]
        for action_index in range(1, feasible_count):
            candidate_value = feasible_processing[action_index]
            if candidate_value < best_value:
                best_value = candidate_value
                best_index = action_index
        return best_index

    if selected_rule_index == 1:
        average_processing_time = 0.0
        for action_index in range(feasible_count):
            average_processing_time += feasible_processing[action_index]
        average_processing_time /= feasible_count
        if average_processing_time <= 1e-12:
            average_processing_time = 1e-12

        first_processing = feasible_processing[0]
        if first_processing <= 1e-12:
            first_processing = 1e-12
        first_due_date = due_dates[feasible_job[0]]
        first_slack = first_due_date - current_time - first_processing
        if first_slack < 0.0:
            first_slack = 0.0
        denominator = g_parameter * average_processing_time
        if denominator <= 1e-12:
            denominator = 1e-12
        best_value = (1.0 / first_processing) * np.exp(-first_slack / denominator)

        for action_index in range(1, feasible_count):
            processing_time = feasible_processing[action_index]
            if processing_time <= 1e-12:
                processing_time = 1e-12
            due_date = due_dates[feasible_job[action_index]]
            slack = due_date - current_time - processing_time
            if slack < 0.0:
                slack = 0.0
            candidate_value = (1.0 / processing_time) * np.exp(-slack / denominator)
            if candidate_value > best_value:
                best_value = candidate_value
                best_index = action_index
        return best_index

    if selected_rule_index == 2:
        first_due_date = due_dates[feasible_job[0]]
        first_completion_time = current_time + feasible_processing[0]
        best_value = first_due_date if first_due_date > first_completion_time else first_completion_time
        for action_index in range(1, feasible_count):
            due_date = due_dates[feasible_job[action_index]]
            completion_time = current_time + feasible_processing[action_index]
            candidate_value = due_date if due_date > completion_time else completion_time
            if candidate_value < best_value:
                best_value = candidate_value
                best_index = action_index
        return best_index

    if selected_rule_index == 4:
        best_value = feasible_start_time[0]
        for action_index in range(1, feasible_count):
            candidate_value = feasible_start_time[action_index]
            if candidate_value < best_value:
                best_value = candidate_value
                best_index = action_index
        return best_index

    # Remaining work based rule: LWR.
    best_job_index = feasible_job[0]
    best_value = _compute_remaining_work_for_idle_job(
        best_job_index,
        job_progress,
        operation_count_per_job,
        candidate_count_per_op,
        candidate_processing_times,
    )
    for action_index in range(1, feasible_count):
        candidate_job_index = feasible_job[action_index]
        candidate_value = _compute_remaining_work_for_idle_job(
            candidate_job_index,
            job_progress,
            operation_count_per_job,
            candidate_count_per_op,
            candidate_processing_times,
        )
        if candidate_value < best_value:
            best_value = candidate_value
            best_index = action_index
    return best_index


@njit(cache=True)
def _simulate_rollout_array(
    candidate_machine_ids,
    candidate_processing_times,
    candidate_count_per_op,
    operation_count_per_job,
    due_dates,
    job_progress,
    job_status_codes,
    machine_available_time,
    initial_event_time,
    initial_event_job,
    initial_event_count,
    current_time,
    max_actions,
    policy_mode,
    tardiness_objective_mode,
):
    machine_count = machine_available_time.shape[0]
    job_count = job_progress.shape[0]
    total_completed_jobs = 0
    completion_times = np.full(job_count, -1.0, dtype=np.float64)
    for job_index in range(job_count):
        if job_status_codes[job_index] == 2:
            total_completed_jobs += 1
            completion_times[job_index] = current_time

    total_event_capacity = max_actions + initial_event_count + 1
    event_time = np.empty(total_event_capacity, dtype=np.float64)
    event_job = np.empty(total_event_capacity, dtype=np.int32)
    event_valid = np.zeros(total_event_capacity, dtype=np.uint8)
    event_count = initial_event_count
    for event_index in range(initial_event_count):
        event_time[event_index] = initial_event_time[event_index]
        event_job[event_index] = initial_event_job[event_index]
        event_valid[event_index] = 1

    scheduled_job = np.empty(max_actions, dtype=np.int32)
    scheduled_op = np.empty(max_actions, dtype=np.int32)
    scheduled_machine = np.empty(max_actions, dtype=np.int32)
    scheduled_start = np.empty(max_actions, dtype=np.float64)
    scheduled_processing = np.empty(max_actions, dtype=np.float64)
    scheduled_count = 0

    feasible_capacity = job_count * candidate_machine_ids.shape[2] + 1
    feasible_job = np.empty(feasible_capacity, dtype=np.int32)
    feasible_op = np.empty(feasible_capacity, dtype=np.int32)
    feasible_machine = np.empty(feasible_capacity, dtype=np.int32)
    feasible_start_time = np.empty(feasible_capacity, dtype=np.float64)
    feasible_processing = np.empty(feasible_capacity, dtype=np.float64)

    while total_completed_jobs < job_count:
        feasible_count = 0
        for job_index in range(job_count):
            if job_status_codes[job_index] != 0:
                continue
            operation_index = job_progress[job_index]
            if operation_index >= operation_count_per_job[job_index]:
                continue

            candidate_count = candidate_count_per_op[job_index, operation_index]
            for candidate_index in range(candidate_count):
                machine_index = candidate_machine_ids[job_index, operation_index, candidate_index]
                processing_time = candidate_processing_times[job_index, operation_index, candidate_index]
                feasible_job[feasible_count] = job_index
                feasible_op[feasible_count] = operation_index
                feasible_machine[feasible_count] = machine_index
                start_time = current_time
                if machine_available_time[machine_index] > start_time:
                    start_time = machine_available_time[machine_index]
                feasible_start_time[feasible_count] = start_time
                feasible_processing[feasible_count] = processing_time
                feasible_count += 1

        if feasible_count == 0:
            earliest_index = -1
            earliest_time = 1e30
            for event_index in range(event_count):
                if event_valid[event_index] == 0:
                    continue
                if event_time[event_index] < earliest_time:
                    earliest_time = event_time[event_index]
                    earliest_index = event_index

            if earliest_index < 0:
                break

            event_valid[earliest_index] = 0
            current_time = max(current_time, event_time[earliest_index])
            completed_job_index = event_job[earliest_index]
            job_progress[completed_job_index] += 1
            if job_progress[completed_job_index] >= operation_count_per_job[completed_job_index]:
                if job_status_codes[completed_job_index] != 2:
                    total_completed_jobs += 1
                job_status_codes[completed_job_index] = 2
                completion_times[completed_job_index] = current_time
            else:
                job_status_codes[completed_job_index] = 0
            continue

        selected_index = _select_action_index(
            policy_mode,
            feasible_count,
            feasible_job,
            feasible_start_time,
            feasible_processing,
            due_dates,
            current_time,
            job_progress,
            operation_count_per_job,
            candidate_count_per_op,
            candidate_processing_times,
        )
        selected_job = feasible_job[selected_index]
        selected_op = feasible_op[selected_index]
        selected_machine = feasible_machine[selected_index]
        selected_processing = feasible_processing[selected_index]
        start_time = feasible_start_time[selected_index]
        completion_time = start_time + selected_processing
        machine_available_time[selected_machine] = completion_time
        job_status_codes[selected_job] = 1

        event_time[event_count] = completion_time
        event_job[event_count] = selected_job
        event_valid[event_count] = 1
        event_count += 1

        scheduled_job[scheduled_count] = selected_job
        scheduled_op[scheduled_count] = selected_op
        scheduled_machine[scheduled_count] = selected_machine
        scheduled_start[scheduled_count] = start_time
        scheduled_processing[scheduled_count] = selected_processing
        scheduled_count += 1

    total_tardiness = 0.0
    maximum_tardiness = 0.0
    for job_index in range(job_count):
        completion_time = completion_times[job_index]
        if completion_time < 0.0:
            completion_time = current_time
        tardiness = completion_time - due_dates[job_index]
        if tardiness > 0.0:
            total_tardiness += tardiness
            if tardiness > maximum_tardiness:
                maximum_tardiness = tardiness
    objective_value = total_tardiness
    if tardiness_objective_mode == 1:
        objective_value = maximum_tardiness
    return (
        objective_value,
        machine_available_time,
        scheduled_job,
        scheduled_op,
        scheduled_machine,
        scheduled_start,
        scheduled_processing,
        scheduled_count,
    )


def _encode_job_status(job_status_map: Dict[int, str], ordered_job_ids: Tuple[int, ...]):
    encoded = np.zeros(len(ordered_job_ids), dtype=np.int8)
    for job_index, job_id in enumerate(ordered_job_ids):
        status_text = job_status_map[job_id]
        if status_text == "idle":
            encoded[job_index] = 0
        elif status_text == "in_progress":
            encoded[job_index] = 1
        else:
            encoded[job_index] = 2
    return encoded


def _extract_due_dates_array(state_manager_clone, ordered_job_ids: Optional[Tuple[int, ...]] = None) -> np.ndarray:
    due_dates_raw = getattr(state_manager_clone, "problem_data", {}).get("due_dates", {})
    if ordered_job_ids is not None:
        job_ids = list(ordered_job_ids)
    elif hasattr(state_manager_clone, "job_ids"):
        job_ids = [int(job_id) for job_id in state_manager_clone.job_ids.tolist()]
    else:
        job_ids = sorted(int(job_id) for job_id in state_manager_clone.jobs.keys())

    due_dates = np.full(len(job_ids), 1e12, dtype=np.float64)
    if not isinstance(due_dates_raw, dict):
        return due_dates

    for job_index, job_id in enumerate(job_ids):
        if job_id in due_dates_raw:
            due_dates[job_index] = float(due_dates_raw[job_id])
        elif str(job_id) in due_dates_raw:
            due_dates[job_index] = float(due_dates_raw[str(job_id)])
    return due_dates


def _build_problem_arrays(state_manager_clone) -> Dict[str, Any]:
    jobs = state_manager_clone.jobs
    ordered_job_ids = tuple(sorted(int(job_id) for job_id in jobs.keys()))
    jobs_signature = tuple(
        (job_id, len(jobs[job_id]), tuple(len(operation) for operation in jobs[job_id]))
        for job_id in ordered_job_ids
    )
    cache_key = (id(state_manager_clone.problem_data), ordered_job_ids, jobs_signature)
    if cache_key in _PROBLEM_ARRAY_CACHE:
        return _PROBLEM_ARRAY_CACHE[cache_key]

    job_count = len(ordered_job_ids)
    operation_count_per_job = np.array([len(jobs[job_id]) for job_id in ordered_job_ids], dtype=np.int32)
    max_operations = int(operation_count_per_job.max()) if job_count > 0 else 0
    max_candidates = 1
    for job_id in ordered_job_ids:
        for operation in jobs[job_id]:
            if len(operation) > max_candidates:
                max_candidates = len(operation)

    candidate_machine_ids = np.full((job_count, max_operations, max_candidates), -1, dtype=np.int32)
    candidate_processing_times = np.zeros((job_count, max_operations, max_candidates), dtype=np.float64)
    candidate_count_per_op = np.zeros((job_count, max_operations), dtype=np.int32)

    for job_index, job_id in enumerate(ordered_job_ids):
        for operation_index, operation_candidates in enumerate(jobs[job_id]):
            candidate_count_per_op[job_index, operation_index] = len(operation_candidates)
            for candidate_index, candidate in enumerate(operation_candidates):
                candidate_machine_ids[job_index, operation_index, candidate_index] = int(candidate["machine"])
                candidate_processing_times[job_index, operation_index, candidate_index] = float(candidate["processing"])

    arrays = {
        "job_ids": ordered_job_ids,
        "operation_count_per_job": operation_count_per_job,
        "candidate_machine_ids": candidate_machine_ids,
        "candidate_processing_times": candidate_processing_times,
        "candidate_count_per_op": candidate_count_per_op,
        "due_dates": _extract_due_dates_array(state_manager_clone, ordered_job_ids=ordered_job_ids),
    }
    _PROBLEM_ARRAY_CACHE[cache_key] = arrays
    return arrays


def can_use_numba_rollout(state_manager_clone) -> bool:
    if not NUMBA_AVAILABLE:
        return False
    if hasattr(state_manager_clone, "candidate_machine_ids"):
        # ArrayTreeState dynamic event payloads use a 4-field tuple
        # (time, counter, type_code, payload_dict), while numba rollout
        # currently expects the legacy 6-field operation-completion format.
        # Skip numba when dynamic-compatible event encoding is present.
        for event in state_manager_clone.event_queue:
            if not isinstance(event, tuple):
                return False
            if len(event) != 6:
                return False
            if int(event[2]) != 0:
                return False
        return True
    if not hasattr(state_manager_clone, "interrupted_ops"):
        return False
    if state_manager_clone.interrupted_ops:
        return False
    if state_manager_clone.broken_machines:
        return False

    for _, _, event_type, event_data in state_manager_clone.event_queue:
        if event_type != "Operation_Completion":
            return False
        if not isinstance(event_data.get("job_id"), int):
            return False
    return True


def _run_numba_rollout(state_manager_clone, policy_mode: int) -> Optional[Tuple[float, list, dict]]:
    if not can_use_numba_rollout(state_manager_clone):
        return None

    is_array_tree_state = hasattr(state_manager_clone, "candidate_machine_ids")

    if is_array_tree_state:
        problem_arrays = {
            "operation_count_per_job": state_manager_clone.operation_count_per_job,
            "candidate_machine_ids": state_manager_clone.candidate_machine_ids,
            "candidate_processing_times": state_manager_clone.candidate_processing_times,
            "candidate_count_per_op": state_manager_clone.candidate_count_per_op,
        }
        rollout_due_dates = state_manager_clone.due_dates
        job_count = int(state_manager_clone.job_ids.shape[0])
        state_job_progress = state_manager_clone.job_progress
    else:
        problem_arrays = _build_problem_arrays(state_manager_clone)
        rollout_due_dates = problem_arrays["due_dates"]
        ordered_job_ids = problem_arrays["job_ids"]
        job_count = len(ordered_job_ids)
        state_job_progress = np.array(
            [state_manager_clone.job_progress[job_id] for job_id in ordered_job_ids],
            dtype=np.int32,
        )
    machine_count = state_manager_clone.num_machines
    max_actions = sum(
        max(0, problem_arrays["operation_count_per_job"][job_index] - state_job_progress[job_index])
        for job_index in range(job_count)
    )
    max_actions = max(max_actions, 1)

    if is_array_tree_state:
        job_progress = state_manager_clone.job_progress.astype(np.int32, copy=True)
        job_status_codes = state_manager_clone.job_status_codes.astype(np.int8, copy=True)
        machine_available_time = state_manager_clone.machine_avail.astype(np.float64, copy=True)
        initial_event_time = np.empty(len(state_manager_clone.event_queue), dtype=np.float64)
        initial_event_job = np.empty(len(state_manager_clone.event_queue), dtype=np.int32)
        initial_event_count = 0
        for event in state_manager_clone.event_queue:
            if not isinstance(event, tuple) or len(event) != 6:
                continue
            event_time, _, event_type, job_index, _, _ = event
            if int(event_type) != 0:
                continue
            initial_event_time[initial_event_count] = float(event_time)
            initial_event_job[initial_event_count] = int(job_index)
            initial_event_count += 1
    else:
        job_id_to_index = {job_id: index for index, job_id in enumerate(ordered_job_ids)}
        job_progress = state_job_progress.copy()
        job_status_codes = _encode_job_status(state_manager_clone.job_status, ordered_job_ids)
        machine_available_time = np.array(
            [state_manager_clone.machine_avail[machine_index] for machine_index in range(machine_count)],
            dtype=np.float64,
        )
        initial_event_time = np.empty(len(state_manager_clone.event_queue), dtype=np.float64)
        initial_event_job = np.empty(len(state_manager_clone.event_queue), dtype=np.int32)
        initial_event_count = 0
        for event_time, _, event_type, event_data in state_manager_clone.event_queue:
            if event_type != "Operation_Completion":
                continue
            job_id = int(event_data["job_id"])
            if job_id not in job_id_to_index:
                continue
            initial_event_time[initial_event_count] = float(event_time)
            initial_event_job[initial_event_count] = int(job_id_to_index[job_id])
            initial_event_count += 1

    tardiness_objective = str(getattr(config, "TARDINESS_OBJECTIVE", "total")).lower()
    tardiness_objective_mode = 1 if tardiness_objective == "max" else 0

    (
        total_tardiness,
        machine_available_time_after,
        scheduled_job,
        scheduled_op,
        scheduled_machine,
        scheduled_start,
        scheduled_processing,
        scheduled_count,
    ) = _simulate_rollout_array(
        problem_arrays["candidate_machine_ids"],
        problem_arrays["candidate_processing_times"],
        problem_arrays["candidate_count_per_op"],
        problem_arrays["operation_count_per_job"],
        rollout_due_dates,
        job_progress,
        job_status_codes,
        machine_available_time,
        initial_event_time,
        initial_event_job,
        int(initial_event_count),
        float(state_manager_clone.current_time),
        int(max_actions),
        int(policy_mode),
        int(tardiness_objective_mode),
    )

    trajectory = []
    machine_busy_time = {machine_index: 0.0 for machine_index in range(machine_count)}
    for index in range(scheduled_count):
        machine_index = int(scheduled_machine[index])
        processing_time = float(scheduled_processing[index])
        start_time = float(scheduled_start[index])
        job_index = int(scheduled_job[index])
        operation_index = int(scheduled_op[index])
        machine_busy_time[machine_index] += processing_time
        trajectory.append(
            f"[T:{start_time:.1f}-{start_time + processing_time:.1f}] "
            f"J{job_index}O{operation_index}@M{machine_index} (pt:{processing_time:.1f})"
        )

    bottleneck_machine = int(np.argmax(machine_available_time_after)) if machine_count > 0 else 0
    analytics = {
        "bottleneck": bottleneck_machine,
        "busy_times": machine_busy_time,
    }
    return float(total_tardiness), trajectory, analytics


def run_numba_random_rollout(state_manager_clone) -> Optional[Tuple[float, list, dict]]:
    return _run_numba_rollout(state_manager_clone, policy_mode=0)


def run_numba_pdr_rollout(state_manager_clone) -> Optional[Tuple[float, list, dict]]:
    return _run_numba_rollout(state_manager_clone, policy_mode=1)
