import json
import os
import argparse
from datetime import datetime
from ortools.sat.python import cp_model
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import config
from utilities.numeric_precision import cap_numeric_precision

TIME_SCALE = 1000
RESULTS_EXACT_DIRNAME = "results_exact"


def _to_scaled_int(value: float) -> int:
    return int(round(float(value) * TIME_SCALE))


def _strip_job_cancellation_events(problem_data: dict) -> int:
    """Remove Job_Cancellation events from instance dynamic event list.

    Returns the number of removed cancellation events.
    """
    dynamic_events = problem_data.get("dynamic_events", [])
    if not isinstance(dynamic_events, list):
        return 0

    filtered_events = []
    removed_count = 0
    for event in dynamic_events:
        event_type = str((event or {}).get("event_type", ""))
        if event_type == "Job_Cancellation":
            removed_count += 1
            continue
        filtered_events.append(event)

    problem_data["dynamic_events"] = filtered_events
    return removed_count


def _build_output_gantt_path(json_file_path: str) -> str:
    base_name = os.path.splitext(os.path.basename(json_file_path))[0]
    return os.path.join(os.path.dirname(json_file_path), f"exact_gantt_{base_name}.png")


def _build_exact_results_folder_path(json_file_path: str) -> str:
    instance_name = os.path.splitext(os.path.basename(json_file_path))[0]
    date_prefix = datetime.now().strftime("%y%m%d")
    folder_name = f"{date_prefix}_{instance_name}_exact"
    workspace_root = os.path.dirname(os.path.abspath(__file__))
    results_root = os.path.join(workspace_root, RESULTS_EXACT_DIRNAME)
    os.makedirs(results_root, exist_ok=True)
    return os.path.join(results_root, folder_name)


def _build_exact_results_json_path(results_folder_path: str) -> str:
    folder_name = os.path.basename(results_folder_path.rstrip("\\/"))
    return os.path.join(results_folder_path, f"results_{folder_name}.json")


def _build_exact_folder_gantt_path(results_folder_path: str) -> str:
    folder_name = os.path.basename(results_folder_path.rstrip("\\/"))
    return os.path.join(results_folder_path, f"final_gantt_{folder_name}.png")


def _compute_exact_tardiness_metrics(
    solver: cp_model.CpSolver,
    completion_end_by_job: dict,
    due_dates: dict,
) -> tuple[float, float]:
    total_tardiness = 0.0
    max_tardiness = 0.0
    for job_id, completion_end in completion_end_by_job.items():
        due_date = due_dates.get(job_id)
        if due_date is None:
            continue
        completion_time = float(solver.Value(completion_end)) / TIME_SCALE
        tardiness_value = max(0.0, completion_time - float(due_date))
        total_tardiness += tardiness_value
        if tardiness_value > max_tardiness:
            max_tardiness = tardiness_value
    return float(total_tardiness), float(max_tardiness)


def _render_exact_gantt(schedule_blocks, output_path: str, objective_label: str, objective_value: float) -> None:
    if not schedule_blocks:
        return

    machine_ids = sorted({int(block["machine"]) for block in schedule_blocks})
    y_position_by_machine = {machine_id: idx for idx, machine_id in enumerate(machine_ids)}
    # Keep color style consistent with simple_gantt.py:
    # stable mapping by job id modulo tab20 palette size.
    color_map = plt.get_cmap("tab20")

    plt.figure(figsize=(14, max(4, len(machine_ids) * 0.55 + 2)))
    for block in schedule_blocks:
        machine_id = int(block["machine"])
        y_position = y_position_by_machine[machine_id]
        start_time = float(block["start"])
        duration = float(block["end"] - block["start"])
        job_id = int(block["job"])
        operation_id = int(block["op"])

        plt.barh(
            y=y_position,
            width=duration,
            left=start_time,
            height=0.7,
            color=color_map(job_id % 20),
            edgecolor="black",
            linewidth=0.5,
        )
        plt.text(
            start_time + duration / 2.0,
            y_position,
            f"J{job_id}-O{operation_id}",
            ha="center",
            va="center",
            fontsize=7,
            color="black",
        )

    plt.yticks(
        ticks=list(y_position_by_machine.values()),
        labels=[f"M{machine_id}" for machine_id in machine_ids],
    )
    plt.xlabel("Time")
    plt.ylabel("Machine")
    plt.title(f"Exact CP-SAT Schedule Gantt ({objective_label}: {objective_value:.2f})")
    plt.grid(axis="x", linestyle="--", alpha=0.4)

    unique_jobs = sorted({int(block["job"]) for block in schedule_blocks})
    legend_patches = [
        mpatches.Patch(color=color_map(job_id % 20), label=f"Job {job_id}")
        for job_id in unique_jobs
    ]
    if legend_patches:
        plt.legend(
            handles=legend_patches,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.08),
            ncol=min(6, len(legend_patches)),
            fontsize=8,
            frameon=False,
        )

    max_end_time = max(float(block["end"]) for block in schedule_blocks)
    plt.xlim(left=0, right=max(max_end_time * 1.02, 1))
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def _build_staticized_jobs(problem_data: dict):
    """Create a static job list from base jobs + supported dynamic arrivals.

    Supported dynamic events:
    - Job_Arrival
    - Emergency_Job_Arrival

    Supported dynamic events (approximated):
    - Machine_Breakdown / Machine_Repair are converted into machine
      downtime windows (stand-in approximation in static CP-SAT).

    Job_Cancellation events are pre-stripped before this function.
    """
    jobs = [job for job in problem_data.get("jobs", [])]
    release_times_raw = problem_data.get("release_times", {})
    due_dates_raw = problem_data.get("due_dates", {})
    dynamic_events = problem_data.get("dynamic_events", [])

    release_times = {}
    due_dates = {}
    machine_downtimes = {}
    for job_id in range(len(jobs)):
        if isinstance(release_times_raw, dict):
            release_times[job_id] = float(release_times_raw.get(job_id, release_times_raw.get(str(job_id), 0.0)))
        else:
            release_times[job_id] = 0.0
        if isinstance(due_dates_raw, dict):
            due_date_value = due_dates_raw.get(job_id, due_dates_raw.get(str(job_id), None))
            if due_date_value is not None:
                due_dates[job_id] = float(due_date_value)

    if not isinstance(dynamic_events, list):
        dynamic_events = []

    sorted_events = sorted(dynamic_events, key=lambda event_item: float(event_item.get("timestamp", 0.0)))
    active_breakdowns = {}
    for event in sorted_events:
        event_type = str(event.get("event_type", ""))
        event_data = event.get("data", {}) or {}
        event_timestamp = float(event.get("timestamp", 0.0))

        if event_type == "Machine_Breakdown":
            machine_id = int(event_data.get("machine_id", -1))
            if machine_id >= 0 and machine_id not in active_breakdowns:
                active_breakdowns[machine_id] = event_timestamp
            continue
        if event_type == "Machine_Repair":
            machine_id = int(event_data.get("machine_id", -1))
            if machine_id < 0:
                continue
            if machine_id in active_breakdowns:
                downtime_start = active_breakdowns.pop(machine_id)
                downtime_end = max(downtime_start, event_timestamp)
                if downtime_end > downtime_start:
                    machine_downtimes.setdefault(machine_id, []).append((downtime_start, downtime_end))
            continue
        if event_type not in {"Job_Arrival", "Emergency_Job_Arrival"}:
            continue

        if "job_id" not in event_data or "operations" not in event_data:
            continue
        job_id = int(event_data["job_id"])
        if job_id < 0:
            continue
        while len(jobs) <= job_id:
            jobs.append([])
        jobs[job_id] = event_data["operations"]
        release_times[job_id] = float(event_data.get("release_time", event.get("timestamp", 0.0)))
        if event_data.get("due_date") is not None:
            due_dates[job_id] = float(event_data["due_date"])

    fallback_horizon = float(problem_data.get("simulation_horizon", 0.0) or 0.0)
    if fallback_horizon <= 0.0:
        fallback_horizon = max(
            [0.0]
            + [release_times.get(job_id, 0.0) for job_id in release_times]
            + [sum(max(float(alt.get("processing", 0.0)) for alt in op) for op in job) for job in jobs if job]
        )
    for machine_id, downtime_start in active_breakdowns.items():
        if fallback_horizon > downtime_start:
            machine_downtimes.setdefault(machine_id, []).append((downtime_start, fallback_horizon))

    return jobs, release_times, due_dates, machine_downtimes

def solve_fjsp_exact(json_file_path, time_limit_seconds=300):
    # 1. Load Data
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    removed_cancellations = _strip_job_cancellation_events(data)
    if removed_cancellations > 0:
        print(
            f">>> Removed {removed_cancellations} Job_Cancellation event(s) "
            "from dynamic events for exact-model compatibility."
        )

    jobs, release_times, due_dates, machine_downtimes = _build_staticized_jobs(data)
    if not jobs:
        raise ValueError("No jobs found in input data.")

    # Sometimes machine IDs are 1-indexed, so we'll collect unique machines directly
    all_machines = set()
    horizon = 0
    
    for job in jobs:
        for operation in job:
            max_duration = 0
            for alt in operation:
                all_machines.add(alt['machine'])
                max_duration = max(max_duration, alt['processing'])
            horizon += max_duration
    for machine_id, downtime_windows in machine_downtimes.items():
        all_machines.add(machine_id)
        for downtime_start, downtime_end in downtime_windows:
            horizon = max(horizon, downtime_end)
    scaled_horizon = max(1, _to_scaled_int(horizon))

    # 2. Initialize Model
    model = cp_model.CpModel()
    
    # Store machine intervals for NoOverlap constraint
    machine_to_intervals = {m: [] for m in all_machines}
    selected_machine_for_operation = {}
    operation_starts = {}
    operation_ends = {}
    
    job_ends = []
    completion_end_by_job = {}

    # 3. Create Variables and Constraints
    for job_id, job in enumerate(jobs):
        if not job:
            continue
        prev_end_var = None
        release_time_scaled = _to_scaled_int(release_times.get(job_id, 0.0))
        
        for op_id, operation in enumerate(job):
            # Main variables for the operation
            op_start = model.NewIntVar(0, scaled_horizon, f'start_j{job_id}_o{op_id}')
            op_end = model.NewIntVar(0, scaled_horizon, f'end_j{job_id}_o{op_id}')
            op_duration = model.NewIntVar(0, scaled_horizon, f'duration_j{job_id}_o{op_id}')
            operation_starts[(job_id, op_id)] = op_start
            operation_ends[(job_id, op_id)] = op_end
            
            # Precedence Constraint (Operation must start after previous operation ends)
            if prev_end_var is not None:
                model.Add(op_start >= prev_end_var)
            else:
                model.Add(op_start >= release_time_scaled)
            prev_end_var = op_end

            # Track end of the final operation in the job
            if op_id == len(job) - 1:
                job_ends.append(op_end)
                completion_end_by_job[job_id] = op_end

            # Variables for alternative machines
            alt_presences = []
            
            for alt_id, alt in enumerate(operation):
                m_id = alt['machine']
                duration = _to_scaled_int(alt['processing'])
                
                # Boolean variable: True if this machine is selected for this operation
                presence = model.NewBoolVar(f'presence_j{job_id}_o{op_id}_a{alt_id}')
                alt_start = model.NewIntVar(0, scaled_horizon, f'alt_start_j{job_id}_o{op_id}_a{alt_id}')
                alt_end = model.NewIntVar(0, scaled_horizon, f'alt_end_j{job_id}_o{op_id}_a{alt_id}')
                
                # Optional Interval Variable (Only active if presence == True)
                alt_interval = model.NewOptionalIntervalVar(
                    alt_start, duration, alt_end, presence, 
                    f'interval_j{job_id}_o{op_id}_a{alt_id}'
                )
                
                alt_presences.append(presence)
                machine_to_intervals[m_id].append(alt_interval)
                selected_machine_for_operation[(job_id, op_id, alt_id)] = (presence, m_id)
                
                # Link alternative variables to main operation variables if selected
                model.Add(op_start == alt_start).OnlyEnforceIf(presence)
                model.Add(op_duration == duration).OnlyEnforceIf(presence)
                model.Add(op_end == alt_end).OnlyEnforceIf(presence)
                
            # Constraint: Exactly ONE alternative must be selected
            model.AddExactlyOne(alt_presences)

    # 4. Machine Constraints (No overlaps on the same machine)
    # Add machine downtime as fixed, non-optional intervals.
    for machine_id, downtime_windows in machine_downtimes.items():
        for downtime_index, (downtime_start, downtime_end) in enumerate(downtime_windows):
            start_scaled = _to_scaled_int(downtime_start)
            end_scaled = _to_scaled_int(downtime_end)
            duration_scaled = max(0, end_scaled - start_scaled)
            if duration_scaled <= 0:
                continue
            downtime_interval = model.NewIntervalVar(
                start_scaled,
                duration_scaled,
                end_scaled,
                f"downtime_m{machine_id}_{downtime_index}",
            )
            machine_to_intervals.setdefault(machine_id, []).append(downtime_interval)
    for m_id, intervals in machine_to_intervals.items():
        if intervals:
            model.AddNoOverlap(intervals)

    # 5. Objective: Minimize Tardiness
    tardiness_objective = str(getattr(config, "TARDINESS_OBJECTIVE", "total")).lower()
    tardiness_vars = []
    for job_id, completion_end in completion_end_by_job.items():
        due_date_scaled = _to_scaled_int(due_dates.get(job_id, horizon))
        tardiness_var = model.NewIntVar(0, scaled_horizon * 2, f"tardiness_j{job_id}")
        model.Add(tardiness_var >= completion_end - due_date_scaled)
        model.Add(tardiness_var >= 0)
        tardiness_vars.append(tardiness_var)

    if not tardiness_vars:
        raise ValueError("No tardiness variables were created; check input jobs/due dates.")

    if tardiness_objective == "max":
        max_tardiness = model.NewIntVar(0, scaled_horizon * 2, "max_tardiness")
        model.AddMaxEquality(max_tardiness, tardiness_vars)
        model.Minimize(max_tardiness)
        objective_label = "Maximum Tardiness"
    else:
        model.Minimize(sum(tardiness_vars))
        objective_label = "Total Tardiness"

    # 6. Solve
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit_seconds
    solver.parameters.log_search_progress = True # Shows bounds and gap in real-time
    
    # Optional: Use multiple workers for speed
    solver.parameters.num_search_workers = 8 

    status = solver.Solve(model)

    # 7. Output Results
    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        objective_value_scaled = solver.ObjectiveValue()
        objective_value = float(objective_value_scaled) / TIME_SCALE
        print(f"\nStatus: {solver.StatusName(status)}")
        print(f"Optimal {objective_label}: {objective_value:.3f}")
        print(f"Time Taken: {solver.WallTime():.2f} seconds")
        print(f"Branches: {solver.NumBranches()}")

        schedule_blocks = []
        for job_id, job in enumerate(jobs):
            for op_id, operation in enumerate(job):
                assigned_machine = None
                for alt_id, _ in enumerate(operation):
                    presence_var, machine_id = selected_machine_for_operation[(job_id, op_id, alt_id)]
                    if solver.Value(presence_var) == 1:
                        assigned_machine = machine_id
                        break
                if assigned_machine is None:
                    continue
                start_value = float(solver.Value(operation_starts[(job_id, op_id)])) / TIME_SCALE
                end_value = float(solver.Value(operation_ends[(job_id, op_id)])) / TIME_SCALE
                schedule_blocks.append(
                    {
                        "job": job_id,
                        "op": op_id,
                        "machine": assigned_machine,
                        "start": start_value,
                        "end": end_value,
                    }
                )

        output_gantt_path = _build_output_gantt_path(json_file_path)
        exact_results_folder_path = _build_exact_results_folder_path(json_file_path)
        os.makedirs(exact_results_folder_path, exist_ok=True)
        exact_folder_gantt_path = _build_exact_folder_gantt_path(exact_results_folder_path)
        _render_exact_gantt(
            schedule_blocks=schedule_blocks,
            output_path=output_gantt_path,
            objective_label=objective_label,
            objective_value=objective_value,
        )
        _render_exact_gantt(
            schedule_blocks=schedule_blocks,
            output_path=exact_folder_gantt_path,
            objective_label=objective_label,
            objective_value=objective_value,
        )
        total_tardiness, max_tardiness = _compute_exact_tardiness_metrics(
            solver=solver,
            completion_end_by_job=completion_end_by_job,
            due_dates=due_dates,
        )

        exact_results_payload = {
            "session_name": os.path.basename(exact_results_folder_path.rstrip("\\/")),
            "generated_at_utc": datetime.now().isoformat(timespec="seconds") + "Z",
            "instance_file": json_file_path,
            "objective": {
                "type": str(getattr(config, "TARDINESS_OBJECTIVE", "total")).lower(),
                "label": objective_label,
                "value": objective_value,
                "total_tardiness": total_tardiness,
                "max_tardiness": max_tardiness,
            },
            "solver_stats": {
                "status": solver.StatusName(status),
                "time_taken_seconds": float(solver.WallTime()),
                "num_branches": int(solver.NumBranches()),
                "time_limit_seconds": float(time_limit_seconds),
            },
            "problem_summary": {
                "job_count": len(jobs),
                "machine_count": len(all_machines),
                "operation_count": sum(len(job) for job in jobs),
                "dynamic_event_count": len(data.get("dynamic_events", []))
                if isinstance(data.get("dynamic_events", []), list)
                else 0,
                "removed_job_cancellations": int(removed_cancellations),
            },
            "schedule_summary": {
                "scheduled_operation_count": len(schedule_blocks),
                "machine_downtime_window_count": sum(len(windows) for windows in machine_downtimes.values()),
                "total_tardiness": total_tardiness,
                "max_tardiness": max_tardiness,
            },
        }
        exact_results_json_path = _build_exact_results_json_path(exact_results_folder_path)
        with open(exact_results_json_path, "w", encoding="utf-8") as results_file:
            json.dump(cap_numeric_precision(exact_results_payload), results_file, indent=2)

        print(f"Gantt chart saved (problem_data folder): {output_gantt_path}")
        print(f"Gantt chart saved (exact results folder): {exact_folder_gantt_path}")
        print(f"Exact results saved: {exact_results_json_path}")
    else:
        print("\nNo solution found within the time limit.")


def solve_fjsp_exact_batch(batch_folder_path: str, time_limit_seconds: int = 600) -> None:
    """Run exact solver for all JSON files in a folder."""
    normalized_batch_folder = os.path.normpath(batch_folder_path)
    if not os.path.isdir(normalized_batch_folder):
        raise ValueError(f"--batch-folder is not a valid directory: {normalized_batch_folder}")

    json_file_paths = sorted(
        os.path.join(normalized_batch_folder, file_name)
        for file_name in os.listdir(normalized_batch_folder)
        if file_name.lower().endswith(".json")
    )
    if not json_file_paths:
        raise ValueError(f"No JSON files found in batch folder: {normalized_batch_folder}")

    print(f">>> Batch mode: solving {len(json_file_paths)} instance(s) from {normalized_batch_folder}")
    for file_index, json_file_path in enumerate(json_file_paths, start=1):
        print(f"\n=== [{file_index}/{len(json_file_paths)}] {json_file_path} ===")
        solve_fjsp_exact(json_file_path=json_file_path, time_limit_seconds=time_limit_seconds)


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Solve tardiness-focused DFJSP exactly with CP-SAT.",
    )
    parser.add_argument(
        "instance_file",
        nargs="?",
        default="problem_data/large_scale_JSSP/mt6.json",
        help="Path to one JSON instance file (ignored when --batch-folder is set).",
    )
    parser.add_argument(
        "--batch-folder",
        default="",
        help="Folder containing JSON instances to solve in batch.",
    )
    parser.add_argument(
        "--time-limit-seconds",
        type=int,
        default=600,
        help="CP-SAT time limit per instance.",
    )
    return parser


if __name__ == '__main__':
    arguments = _build_argument_parser().parse_args()
    if arguments.batch_folder:
        solve_fjsp_exact_batch(
            batch_folder_path=arguments.batch_folder,
            time_limit_seconds=arguments.time_limit_seconds,
        )
    else:
        solve_fjsp_exact(
            json_file_path=arguments.instance_file,
            time_limit_seconds=arguments.time_limit_seconds,
        )