import argparse
import json
import math
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Any
from numeric_precision import cap_numeric_precision

INCLUDE_CANCELLATION = False
def format_operation_entries_in_json(json_text: str) -> str:
    """Compact operation dict formatting to match expected instance style."""
    return re.sub(
        r'\{\s*"machine":\s*(\d+),\s*"processing":\s*([0-9]+(?:\.[0-9]+)?)\s*\}',
        r'{"machine": \1,"processing": \2}',
        json_text,
    )




@dataclass(frozen=True)
class ScaleParameters:
    initial_jobs_range: Tuple[int, int]
    operations_per_job_range: Tuple[int, int]
    machines_range: Tuple[int, int]
    candidate_machines_per_operation_range: Tuple[int, int]
    processing_time_range: Tuple[float, float]
    due_date_tightness_factors: Tuple[float, ...]
    emergency_jobs_per_instance: int
    machine_failure_probability: float
    job_cancellation_probability: float




def sample_integer_inclusive(random_number_generator: random.Random, bounds: Tuple[int, int]) -> int:
    lower_bound, upper_bound = bounds
    return random_number_generator.randint(lower_bound, upper_bound)


def sample_float(random_number_generator: random.Random, bounds: Tuple[float, float]) -> float:
    lower_bound, upper_bound = bounds
    return random_number_generator.uniform(lower_bound, upper_bound)


def compute_operation_expected_processing_time(operation_candidates: List[Dict[str, float]]) -> float:
    processing_values = [candidate["processing"] for candidate in operation_candidates]
    return sum(processing_values) / len(processing_values)


def compute_job_expected_work_content(job_operations: List[List[Dict[str, float]]]) -> float:
    return sum(
        compute_operation_expected_processing_time(operation_candidates)
        for operation_candidates in job_operations
    )


def compute_due_date(
    release_time: float,
    job_operations: List[List[Dict[str, float]]],
    tightness_factor: float,
) -> int:
    expected_work_content = compute_job_expected_work_content(job_operations)
    return math.floor(release_time + tightness_factor * expected_work_content)


def generate_job_operations(
    random_number_generator: random.Random,
    number_of_machines: int,
    operations_per_job_range: Tuple[int, int],
    candidate_machines_per_operation_range: Tuple[int, int],
    processing_time_range: Tuple[float, float],
) -> List[List[Dict[str, float]]]:
    operation_count = sample_integer_inclusive(random_number_generator, operations_per_job_range)
    generated_operations: List[List[Dict[str, float]]] = []

    for _ in range(operation_count):
        candidate_machine_count = sample_integer_inclusive(
            random_number_generator,
            (
                candidate_machines_per_operation_range[0],
                min(candidate_machines_per_operation_range[1], number_of_machines),
            ),
        )
        sampled_machine_ids = random_number_generator.sample(range(number_of_machines), candidate_machine_count)

        base_processing_time = sample_float(random_number_generator, processing_time_range)
        operation_candidates: List[Dict[str, float]] = []
        for machine_identifier in sampled_machine_ids:
            perturbation_factor = random_number_generator.uniform(0.8, 1.2)
            perturbed_processing_time = round(base_processing_time * perturbation_factor, 3)
            operation_candidates.append(
                {
                    "machine": machine_identifier,
                    "processing": max(0.1, perturbed_processing_time),
                }
            )

        generated_operations.append(operation_candidates)

    return generated_operations


def compute_adaptive_horizon(
    initial_jobs: List[List[List[Dict[str, float]]]],
    machine_count: int,
    initial_due_dates: Dict[int, int],
) -> float:
    total_expected_work = sum(compute_job_expected_work_content(job_operations) for job_operations in initial_jobs)
    makespan_lower_bound = total_expected_work / max(machine_count, 1)
    lower_bound_component = 1.2 * makespan_lower_bound
    max_due_date_component = 1.1 * max(initial_due_dates.values()) if initial_due_dates else 0.0
    horizon = max(lower_bound_component, max_due_date_component)
    return round(max(horizon, 1.0), 3)


def generate_tardiness_instance(
    scale: str,
    tightness_factor: float,
    seed: int,
    dynamic_arrival_fraction: float = 0.5,
    include_cancellation: bool = INCLUDE_CANCELLATION,
) -> Dict[str, Any]:
    if scale not in SCALE_CONFIGURATION:
        raise ValueError(f"Unsupported scale '{scale}'. Supported values: {sorted(SCALE_CONFIGURATION.keys())}")

    if dynamic_arrival_fraction < 0.0 or dynamic_arrival_fraction > 1.0:
        raise ValueError("dynamic_arrival_fraction must be in [0.0, 1.0].")

    scale_parameters = SCALE_CONFIGURATION[scale]
    if tightness_factor not in scale_parameters.due_date_tightness_factors:
        raise ValueError(
            f"Unsupported tightness factor '{tightness_factor}' for scale '{scale}'. "
            f"Allowed values: {scale_parameters.due_date_tightness_factors}"
        )

    random_number_generator = random.Random(seed)
    machine_count = sample_integer_inclusive(random_number_generator, scale_parameters.machines_range)
    initial_job_count = sample_integer_inclusive(random_number_generator, scale_parameters.initial_jobs_range)

    initial_jobs: List[List[List[Dict[str, float]]]] = []
    initial_due_dates: Dict[int, int] = {}
    release_times: Dict[int, float] = {}

    for job_id in range(initial_job_count):
        job_operations = generate_job_operations(
            random_number_generator=random_number_generator,
            number_of_machines=machine_count,
            operations_per_job_range=scale_parameters.operations_per_job_range,
            candidate_machines_per_operation_range=scale_parameters.candidate_machines_per_operation_range,
            processing_time_range=scale_parameters.processing_time_range,
        )
        initial_jobs.append(job_operations)
        release_times[job_id] = 0.0
        initial_due_dates[job_id] = compute_due_date(
            release_time=0.0,
            job_operations=job_operations,
            tightness_factor=tightness_factor,
        )

    simulation_horizon = compute_adaptive_horizon(
        initial_jobs=initial_jobs,
        machine_count=machine_count,
        initial_due_dates=initial_due_dates,
    )

    dynamic_events: List[Dict[str, Any]] = []
    due_dates: Dict[int, int] = dict(initial_due_dates)
    all_jobs: List[List[List[Dict[str, float]]]] = list(initial_jobs)

    dynamic_job_count = max(1, int(round(dynamic_arrival_fraction * initial_job_count)))
    for _ in range(dynamic_job_count):
        arriving_job_id = len(all_jobs)
        arriving_job_operations = generate_job_operations(
            random_number_generator=random_number_generator,
            number_of_machines=machine_count,
            operations_per_job_range=scale_parameters.operations_per_job_range,
            candidate_machines_per_operation_range=scale_parameters.candidate_machines_per_operation_range,
            processing_time_range=scale_parameters.processing_time_range,
        )
        arrival_time = round(random_number_generator.uniform(0.0, 0.5 * simulation_horizon), 3)
        release_times[arriving_job_id] = arrival_time
        due_dates[arriving_job_id] = compute_due_date(
            release_time=arrival_time,
            job_operations=arriving_job_operations,
            tightness_factor=tightness_factor,
        )

        all_jobs.append(arriving_job_operations)
        dynamic_events.append(
            {
                "timestamp": arrival_time,
                "event_type": "Job_Arrival",
                "data": {
                    "job_id": arriving_job_id,
                    "operations": arriving_job_operations,
                    "release_time": arrival_time,
                    "due_date": due_dates[arriving_job_id],
                    "is_emergency": False,
                },
            }
        )

    for _ in range(scale_parameters.emergency_jobs_per_instance):
        emergency_job_id = len(all_jobs)
        emergency_job_operations = generate_job_operations(
            random_number_generator=random_number_generator,
            number_of_machines=machine_count,
            operations_per_job_range=scale_parameters.operations_per_job_range,
            candidate_machines_per_operation_range=scale_parameters.candidate_machines_per_operation_range,
            processing_time_range=scale_parameters.processing_time_range,
        )
        emergency_arrival_time = round(random_number_generator.uniform(0.0, 0.5 * simulation_horizon), 3)
        emergency_tightness_factor = random_number_generator.uniform(1.0, 1.1)

        release_times[emergency_job_id] = emergency_arrival_time
        due_dates[emergency_job_id] = compute_due_date(
            release_time=emergency_arrival_time,
            job_operations=emergency_job_operations,
            tightness_factor=emergency_tightness_factor,
        )

        all_jobs.append(emergency_job_operations)
        dynamic_events.append(
            {
                "timestamp": emergency_arrival_time,
                "event_type": "Emergency_Job_Arrival",
                "data": {
                    "job_id": emergency_job_id,
                    "operations": emergency_job_operations,
                    "release_time": emergency_arrival_time,
                    "due_date": due_dates[emergency_job_id],
                    "is_emergency": True,
                    "tightness_factor": round(emergency_tightness_factor, 3),
                },
            }
        )

    for machine_id in range(machine_count):
        has_breakdown = random_number_generator.random() < scale_parameters.machine_failure_probability
        if not has_breakdown:
            continue

        breakdown_time = round(random_number_generator.uniform(0.05 * simulation_horizon, 0.95 * simulation_horizon), 3)
        repair_duration = round(random_number_generator.uniform(1.0, 4.0), 3)
        repair_time = round(min(simulation_horizon, breakdown_time + repair_duration), 3)

        dynamic_events.append(
            {
                "timestamp": breakdown_time,
                "event_type": "Machine_Breakdown",
                "data": {"machine_id": machine_id},
            }
        )
        dynamic_events.append(
            {
                "timestamp": repair_time,
                "event_type": "Machine_Repair",
                "data": {"machine_id": machine_id},
            }
        )

    if include_cancellation:
        for job_id in range(len(all_jobs)):
            is_cancelled = random_number_generator.random() < scale_parameters.job_cancellation_probability
            if not is_cancelled:
                continue
            cancellation_time = round(random_number_generator.uniform(0.05 * simulation_horizon, simulation_horizon), 3)
            dynamic_events.append(
                {
                    "timestamp": cancellation_time,
                    "event_type": "Job_Cancellation",
                    "data": {"job_id": job_id},
                }
            )

    dynamic_events.sort(key=lambda event: (event["timestamp"], event["event_type"]))

    return {
        "machines": machine_count,
        "jobs": initial_jobs,
        "release_times": release_times,
        "due_dates": due_dates,
        "tightness_factor": tightness_factor,
        "scale": scale,
        "simulation_horizon": simulation_horizon,
        "dynamic_events": dynamic_events,
        "metadata": {
            "seed": seed,
            "dynamic_job_count": dynamic_job_count,
            "emergency_jobs_per_instance": scale_parameters.emergency_jobs_per_instance,
            "machine_failure_probability": scale_parameters.machine_failure_probability,
            "job_cancellation_probability": (
                scale_parameters.job_cancellation_probability if include_cancellation else 0.0
            ),
            "include_cancellation": bool(include_cancellation),
            "notes": "Due dates follow TWK. Horizon follows max(1.2*makespan_lb, 1.1*max_initial_due_date).",
        },
    }


def generate_instance_pool(
    output_directory: Path,
    scale: str,
    instances_per_tightness: int,
    base_seed: int,
    include_cancellation: bool = INCLUDE_CANCELLATION,
) -> None:
    output_directory.mkdir(parents=True, exist_ok=True)
    scale_parameters = SCALE_CONFIGURATION[scale]

    file_index = 0
    for tightness_factor in scale_parameters.due_date_tightness_factors:
        for instance_counter in range(instances_per_tightness):
            seed = base_seed + file_index
            generated_instance = generate_tardiness_instance(
                scale=scale,
                tightness_factor=tightness_factor,
                seed=seed,
                include_cancellation=include_cancellation,
            )

            generated_filename = f"{scale}_k{tightness_factor}_instance_{instance_counter:03d}.json"
            output_path = output_directory / generated_filename
            formatted_json = json.dumps(cap_numeric_precision(generated_instance), indent=2)
            formatted_json = format_operation_entries_in_json(formatted_json)
            with output_path.open("w", encoding="utf-8") as output_file:
                output_file.write(formatted_json + "\n")
            file_index += 1


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate dynamic tardiness-focused scheduling instances.",
    )
    parser.add_argument("--scale", choices=sorted(SCALE_CONFIGURATION.keys()), required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where generated instance files will be written.",
    )
    parser.add_argument(
        "--instances-per-tightness",
        type=int,
        default=1,
        help="Number of instances generated for each tightness factor k.",
    )
    parser.add_argument(
        "--base-seed",
        type=int,
        default=260427,
        help="Base seed used to create deterministic seeds per instance.",
    )
    parser.add_argument(
        "--include-cancellation",
        action="store_true",
        help="If set, include Job_Cancellation dynamic events in generated instances.",
    )
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    generate_instance_pool(
        output_directory=arguments.output_dir,
        scale=arguments.scale,
        instances_per_tightness=arguments.instances_per_tightness,
        base_seed=arguments.base_seed,
        include_cancellation=arguments.include_cancellation,
    )


SCALE_CONFIGURATION: Dict[str, ScaleParameters] = {
    "normal": ScaleParameters(
        initial_jobs_range=(15, 20),
        operations_per_job_range=(2, 4),
        machines_range=(3, 5),
        candidate_machines_per_operation_range=(1, 3),
        processing_time_range=(1.0, 5.0),
        due_date_tightness_factors=(1.5, 2.0),
        emergency_jobs_per_instance=1,
        machine_failure_probability=0.5,
        job_cancellation_probability=0.0,
    ),
    "small": ScaleParameters(
        initial_jobs_range=(3, 5),
        operations_per_job_range=(2, 4),
        machines_range=(3, 5),
        candidate_machines_per_operation_range=(1, 3),
        processing_time_range=(1.0, 3.0),
        due_date_tightness_factors=(1.5, 2.0),
        emergency_jobs_per_instance=1,
        machine_failure_probability=0.3,
        job_cancellation_probability=0.0,
    ),
    "large": ScaleParameters(
        initial_jobs_range=(25, 45),
        operations_per_job_range=(2, 4),
        machines_range=(6, 7),
        candidate_machines_per_operation_range=(2, 3),
        processing_time_range=(5.0, 10.0),
        due_date_tightness_factors=(3.0,),
        emergency_jobs_per_instance=3,
        machine_failure_probability=0.3,
        job_cancellation_probability=0.0,
    ),
}

if __name__ == "__main__":
    main()
