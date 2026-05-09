import random
import math

import config
from utilities.numba_rollout import run_numba_pdr_rollout, run_numba_random_rollout


def _all_jobs_completed(state) -> bool:
    if hasattr(state, "all_jobs_completed"):
        return state.all_jobs_completed()
    return all(status == "completed" for status in state.job_status.values())


def _calculate_remaining_work(state, job_id: int) -> float:
    if hasattr(state, "_calculate_rem_work"):
        return state._calculate_rem_work(job_id)
    return 0.0


def _build_rollout_result(state_manager_clone, trajectory, machine_busy_time):
    tardiness = float(state_manager_clone.calculate_actual_tardiness())
    if hasattr(state_manager_clone.machine_avail, "get"):
        bottleneck_machine = max(state_manager_clone.machine_avail, key=state_manager_clone.machine_avail.get)
    else:
        bottleneck_machine = int(max(range(state_manager_clone.num_machines), key=lambda machine_id: state_manager_clone.machine_avail[machine_id]))
    analytics = {
        "bottleneck": bottleneck_machine,
        "busy_times": machine_busy_time,
    }
    return tardiness, trajectory, analytics


def stochastic_rollout(state_manager_clone, initial_action=None):
    """Run one PDR-mixture stochastic rollout and return tardiness, trajectory, and analytics."""
    if (
        initial_action is None
        and getattr(config, "MCTS_NUMBA_ROLLOUT_ENABLED", False)
    ):
        numba_result = run_numba_pdr_rollout(state_manager_clone)
        if numba_result is not None:
            return numba_result

    trajectory = []
    machine_busy_time = {machine_index: 0 for machine_index in range(state_manager_clone.num_machines)}

    if initial_action:
        machine_busy_time[initial_action["machine"]] += initial_action["processing_time"]

    def shortest_processing_time(actions, state_manager):
        return min(actions, key=lambda action: action["processing_time"])

    def apparent_tardiness_cost(actions, state_manager, g_parameter=0.5):
        average_processing_time = sum(action["processing_time"] for action in actions) / len(actions)
        denominator = max(g_parameter * average_processing_time, 1e-12)

        def atc_priority(action):
            processing_time = max(action["processing_time"], 1e-12)
            due_date = float("inf") if action.get("due_date") is None else float(action["due_date"])
            slack = max(0.0, due_date - state_manager.current_time - processing_time)
            return (1.0 / processing_time) * math.exp(-slack / denominator)

        return max(actions, key=atc_priority)

    def modified_due_date(actions, state_manager):
        return min(
            actions,
            key=lambda action: max(
                float("inf") if action.get("due_date") is None else float(action["due_date"]),
                state_manager.current_time + action["processing_time"],
            ),
        )

    def least_work_remaining(actions, state_manager):
        return min(actions, key=lambda action: _calculate_remaining_work(state_manager, action["job"]))

    def earliest_start_time(actions, state_manager):
        return min(
            actions,
            key=lambda action: max(state_manager.current_time, state_manager.machine_avail[action["machine"]]),
        )

    dispatching_rules = [
        ("SPT", shortest_processing_time),
        ("ATC", apparent_tardiness_cost),
        ("MDD", modified_due_date),
        ("LWR", least_work_remaining),
        ("EST", earliest_start_time),
    ]
    dispatching_rule_weights = [0.2, 0.2, 0.2, 0.2, 0.2]

    while not _all_jobs_completed(state_manager_clone):
        feasible_actions = state_manager_clone.get_feasible_actions()

        if not feasible_actions:
            event_type, _, _ = state_manager_clone.process_next_event()
            if event_type is None:
                break
            continue

        selected_rule_name, selected_dispatching_rule = random.choices(
            dispatching_rules,
            weights=dispatching_rule_weights,
            k=1,
        )[0]
        selected_action = selected_dispatching_rule(feasible_actions, state_manager_clone)

        machine_index = selected_action["machine"]
        processing_time = selected_action["processing_time"]

        machine_available_time = state_manager_clone.machine_avail[machine_index]
        estimated_start_time = max(state_manager_clone.current_time, machine_available_time)

        state_manager_clone.execute_action(
            selected_action["job"],
            selected_action["op"],
            selected_action["machine"],
        )

        machine_busy_time[machine_index] += processing_time
        trajectory.append(
            f"[R:{selected_rule_name}] [T:{estimated_start_time:.1f}-{estimated_start_time + processing_time:.1f}] "
            f"J{selected_action['job']}O{selected_action['op']}@M{machine_index} (pt:{processing_time})"
        )

    return _build_rollout_result(state_manager_clone, trajectory, machine_busy_time)


def random_rollout(state_manager_clone, initial_action=None):
    """Run one truly-random rollout and return tardiness, trajectory, and analytics."""
    if (
        initial_action is None
        and getattr(config, "MCTS_NUMBA_ROLLOUT_ENABLED", False)
    ):
        numba_result = run_numba_random_rollout(state_manager_clone)
        if numba_result is not None:
            return numba_result

    trajectory = []
    machine_busy_time = {machine_index: 0 for machine_index in range(state_manager_clone.num_machines)}

    if initial_action:
        machine_busy_time[initial_action["machine"]] += initial_action["processing_time"]

    while not _all_jobs_completed(state_manager_clone):
        feasible_actions = state_manager_clone.get_feasible_actions()

        if not feasible_actions:
            event_type, _, _ = state_manager_clone.process_next_event()
            if event_type is None:
                break
            continue

        selected_action = random.choice(feasible_actions)
        machine_index = selected_action["machine"]
        processing_time = selected_action["processing_time"]
        estimated_start_time = max(state_manager_clone.current_time, state_manager_clone.machine_avail[machine_index])

        state_manager_clone.execute_action(
            selected_action["job"],
            selected_action["op"],
            selected_action["machine"],
        )

        machine_busy_time[machine_index] += processing_time
        trajectory.append(
            f"[T:{estimated_start_time:.1f}-{estimated_start_time + processing_time:.1f}] "
            f"J{selected_action['job']}O{selected_action['op']}@M{machine_index} (pt:{processing_time})"
        )

    return _build_rollout_result(state_manager_clone, trajectory, machine_busy_time)
