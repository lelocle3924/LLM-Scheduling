import argparse
import json
import os
from datetime import datetime

import config
from llm_agent import LLMAgent
from reflection_engine import Reflec
from state_manager import StateManager
from strategies.mcts_search import MCTSSearcher
from strategies.single_search import SingleSearcher
from utilities.logger import log_event, log_final_results, setup_session_folder
from utilities.numeric_precision import cap_numeric_precision, format_decimal


def load_text_file(filepath: str) -> str:
    with open(filepath, "r", encoding="utf-8") as file_handle:
        return file_handle.read()


def _save_json(filepath: str, payload) -> None:
    with open(filepath, "w", encoding="utf-8") as file_handle:
        json.dump(cap_numeric_precision(payload), file_handle, indent=2)


def _generate_final_gantt_if_enabled(session_folder: str) -> None:
    if not getattr(config, "AUTO_GENERATE_GANTT", True):
        return

    try:
        from simple_gantt import build_schedule_data, parse_events_log, render_gantt

        safe_session_name = os.path.basename(session_folder.rstrip("\\/"))
        events_log_path = os.path.join(session_folder, f"events_log_{safe_session_name}.txt")
        output_gantt_path = os.path.join(session_folder, f"final_gantt_{safe_session_name}.png")

        events = parse_events_log(events_log_path)
        if not events:
            print(">>> Gantt generation skipped: events log is empty or missing.")
            return

        schedule_blocks, breakdowns, emergency_jobs, max_machine, max_time = build_schedule_data(events)
        render_gantt(
            blocks=schedule_blocks,
            breakdowns=breakdowns,
            emergency_jobs=emergency_jobs,
            num_machines=max_machine,
            max_time=max_time,
            output_path=output_gantt_path,
        )
    except Exception as gantt_error:
        print(f">>> Gantt generation failed: {gantt_error}")


def _build_searcher(agent: LLMAgent):
    print(f">>> Initializing Search Framework: {config.SEARCH_STRATEGY}")
    if config.SEARCH_STRATEGY == "SingleSearch":
        return SingleSearcher(llm_agent=agent)
    if config.SEARCH_STRATEGY == "MCTSSearch":
        return MCTSSearcher(llm_agent=agent)
    raise ValueError(f"Unknown SEARCH_STRATEGY in config: {config.SEARCH_STRATEGY}")


def _write_lessons_file(session_folder: str, lessons_text: str) -> None:
    lessons_path = os.path.join(session_folder, "lessons.md")
    with open(lessons_path, "w", encoding="utf-8") as lessons_file:
        lessons_file.write(str(lessons_text or "").strip())


def main():
    total_start_time = datetime.now()
    print(">>> Loading problem data...")
    with open(config.PROBLEM_FILE, "r", encoding="utf-8") as problem_file:
        problem_data = json.load(problem_file)

    session_folder = setup_session_folder(config.SESSION_NAME)
    config_copy_path = os.path.join(session_folder, "config_copy.json")
    config_dict = {key: value for key, value in vars(config).items() if key.isupper() and not key.startswith("_")}
    _save_json(config_copy_path, config_dict)

    action_template = load_text_file(config.ACTION_PROMPT_FILE)
    prior_template = load_text_file(config.PRIOR_PROMPT_FILE)
    reflect_template = load_text_file(config.REFLECT_PROMPT_FILE)

    agent = LLMAgent(action_prompt_template=action_template, prior_prompt_template=prior_template)
    reflector = Reflec(prompt_template=reflect_template)
    searcher = _build_searcher(agent)
    sm = StateManager(problem_data)

    safe_session_name = os.path.basename(session_folder.rstrip("\\/"))
    events_log_path = os.path.join(session_folder, f"events_log_{safe_session_name}.txt")
    with open(events_log_path, "w", encoding="utf-8"):
        pass

    iteration = 1

    while not all(status == "completed" for status in sm.job_status.values()):
        feasible_actions = sm.get_feasible_actions()

        if (
            getattr(config, "USE_REFLECTION", True)
            and feasible_actions
            and getattr(sm, "requires_reflection", False)
        ):
            print(
                f"\n>>> [EVENT TRIGGER] {sm.last_dynamic_event} at T={format_decimal(sm.current_time)}. "
                "Executing Hierarchical Reflection..."
            )
            lessons = reflector.execute_hierarchical_reflection(
                sm, sm.last_dynamic_event, session_folder, iteration
            )
            searcher.update_strategic_experience(lessons)
            _write_lessons_file(session_folder, lessons)
            sm.requires_reflection = False

        if not feasible_actions:
            event_type, timestamp, event_data = sm.process_next_event()
            if event_type is None:
                break
            log_event(session_folder, timestamp, event_type, str(event_data))
            continue

        print(f"\n--- Iteration {iteration} | Clock: {sm.current_time} ---")
        decision = searcher.run_search(
            initial_state=sm,
            session_folder=session_folder,
            iteration=iteration,
        )

        if decision:
            sm.execute_action(decision["job"], decision["op"], decision["machine"])
            log_event(
                session_folder,
                sm.current_time,
                "Action_Executed",
                f"Job {decision['job']}, Op {decision['op']} -> Mach {decision['machine']}",
            )
            print(f">>> SUCCESS: Job {decision['job']}, Op {decision['op']} -> Mach {decision['machine']}.")
        else:
            fallback_action = min(feasible_actions, key=lambda action: action["processing_time"])
            sm.execute_action(
                fallback_action["job"],
                fallback_action["op"],
                fallback_action["machine"],
            )
            log_event(
                session_folder,
                sm.current_time,
                "Fallback_Executed",
                f"Job {fallback_action['job']}, Op {fallback_action['op']} -> Mach {fallback_action['machine']}",
            )
            print(">>> FALLBACK: Executed SPT due to search failure.")

        iteration += 1

    tardiness_metrics = sm.calculate_tardiness_metrics()
    total_tardiness = float(tardiness_metrics["total_tardiness"])
    max_tardiness = float(tardiness_metrics["max_tardiness"])
    final_tardiness = float(sm.calculate_actual_tardiness())
    final_schedule = {
        "tardiness": final_tardiness,
        "final_tardiness": final_tardiness,
        "total_tardiness": total_tardiness,
        "max_tardiness": max_tardiness,
        "makespan": float(sm.current_time),
        "completed_jobs": sm.completed_jobs,
        "total_jobs": len(sm.jobs),
        "machine_available_times": sm.machine_avail,
        "broken_machines": sorted(sm.broken_machines),
        "search_strategy": config.SEARCH_STRATEGY,
    }

    runtime_seconds = (datetime.now() - total_start_time).total_seconds()
    log_final_results(
        session_folder=session_folder,
        problem_data=problem_data,
        final_schedule=final_schedule,
        total_runtime_seconds=runtime_seconds,
        iteration_count=max(0, iteration - 1),
    )
    _generate_final_gantt_if_enabled(session_folder)

    print("\n>>> Event-driven run complete.")
    print(f">>> Session folder: {session_folder}")
    print(f"Total processing time: {datetime.now() - total_start_time}")


def run_batch_folder(batch_folder_path: str) -> None:
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

    base_session_name = str(getattr(config, "SESSION_NAME", "session")).strip() or "session"
    print(f">>> Batch mode: solving {len(json_file_paths)} instance(s) from {normalized_batch_folder}")
    for file_index, instance_path in enumerate(json_file_paths, start=1):
        instance_name = os.path.splitext(os.path.basename(instance_path))[0]
        config.PROBLEM_FILE = instance_path
        config.SESSION_NAME = f"{base_session_name}_{instance_name}"
        print(f"\n=== [{file_index}/{len(json_file_paths)}] {instance_path} ===")
        print(f">>> Session: {config.SESSION_NAME}")
        main()


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run DFJSP solver on one instance or a folder of instances.",
    )
    parser.add_argument(
        "--batch-folder",
        default="",
        help="Folder containing JSON instances to solve in batch.",
    )
    return parser


if __name__ == "__main__":
    arguments = _build_argument_parser().parse_args()
    if arguments.batch_folder:
        run_batch_folder(arguments.batch_folder)
    else:
        main()
