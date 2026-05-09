import os
import re
import ast
import argparse
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.cm as cm
import numpy as np

import config

def build_io_paths(target_folder_path, use_greedy_suffix=False):
    """Build input/output file paths from a folder path."""
    normalized_folder_path = os.path.normpath(target_folder_path)
    folder_name = os.path.basename(normalized_folder_path)

    if not folder_name:
        raise ValueError("Target folder path is invalid.")

    if use_greedy_suffix:
        log_filename = f"events_log_{folder_name}_greedy.txt"
        output_filename = f"final_gantt_{folder_name}_greedy.png"
    else:
        log_filename = f"events_log_{folder_name}.txt"
        output_filename = f"final_gantt_{folder_name}.png"

    events_log_path = os.path.join(normalized_folder_path, log_filename)
    output_path = os.path.join(normalized_folder_path, output_filename)
    return events_log_path, output_path

def parse_events_log(filepath):
    """Parses the events log and extracts structured data."""
    events = []
    if not os.path.exists(filepath):
        print(f"Error: File '{filepath}' not found.")
        return events

    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            # Match format: [Time:   0.00] Event_Type | Details
            match = re.search(r'\[Time:\s*([\d\.]+)\]\s*([A-Za-z_]+)\s*\|\s*(.*)', line)
            if match:
                timestamp = float(match.group(1))
                event_type = match.group(2)
                details_str = match.group(3)

                try:
                    if details_str.startswith("{"):
                        details = ast.literal_eval(details_str)
                    else:
                        details = details_str
                except (ValueError, SyntaxError):
                    details = details_str

                events.append({
                    "time": timestamp,
                    "type": event_type,
                    "details": details
                })
    return events

def build_schedule_data(events):
    """Processes chronological events to build operation blocks and breakdown intervals.

    Supports machine queueing: multiple jobs can be assigned to a machine before
    the current occupant completes. Jobs are processed in FIFO order.
    """
    machine_queues = {}
    machine_proc_start = {}
    job_next_op = {}

    completed_blocks = []
    breakdowns = []
    emergency_jobs = set()

    max_time = 0.0
    max_machine = 0

    for ev in events:
        t = ev['time']
        e_type = ev['type']
        details = ev['details']

        max_time = max(max_time, t)

        if e_type == "Job_Emergency":
            if isinstance(details, dict) and 'job_id' in details:
                emergency_jobs.add(details['job_id'])

        elif e_type in ("Action_Executed", "Fallback_Executed"):
            match = re.search(r'Job (\d+)(?:, Op (\d+))? -> Mach (\d+)', str(details))
            if match:
                j = int(match.group(1))
                if match.group(2):
                    o = int(match.group(2))
                    job_next_op[j] = max(job_next_op.get(j, 0), o)
                else:
                    o = job_next_op.get(j, 0)
                m = int(match.group(3))
                max_machine = max(max_machine, m)

                queue = machine_queues.setdefault(m, [])
                queue.append({'job': j, 'op': o})

                if len(queue) == 1:
                    machine_proc_start[m] = t

        elif e_type == "Machine_Breakdown":
            m = details.get('machine_id') if isinstance(details, dict) else None
            if m is None:
                continue
            breakdowns.append({'machine': m, 'start': t, 'end': None})

            queue = machine_queues.get(m, [])
            if queue:
                front = queue.pop(0)
                start = machine_proc_start.get(m, t)
                completed_blocks.append({
                    'machine': m, 'job': front['job'], 'op': front['op'],
                    'start': start, 'end': t,
                    'interrupted': True
                })

        elif e_type == "Machine_Repair":
            m = details.get('machine_id') if isinstance(details, dict) else None
            if m is None:
                continue

            for b in breakdowns:
                if b['machine'] == m and b['end'] is None:
                    b['end'] = t
                    break

            resumed_job = details.get('resumed_job') if isinstance(details, dict) else None
            resumed_op = details.get('resumed_op') if isinstance(details, dict) else None

            if resumed_job is not None and resumed_op is not None:
                queue = machine_queues.setdefault(m, [])
                queue.insert(0, {'job': resumed_job, 'op': resumed_op})

            queue = machine_queues.get(m, [])
            if queue:
                machine_proc_start[m] = t

        elif e_type == "Operation_Completion":
            if isinstance(details, dict):
                m = details.get('machine_id')
                j = details.get('job_id')
                o = details.get('op_id')

                queue = machine_queues.get(m, [])
                if not queue:
                    continue

                matched_index = None
                for idx, queued_op in enumerate(queue):
                    if queued_op['job'] == j and queued_op['op'] == o:
                        matched_index = idx
                        break

                if matched_index is None:
                    continue

                # Normal case: completion matches current machine front.
                # Fallback case: parser recovered a non-front completion from noisy/mixed logs.
                if matched_index == 0:
                    start = machine_proc_start.get(m, 0)
                else:
                    start = t

                completed_blocks.append({
                    'machine': m, 'job': j, 'op': o,
                    'start': start, 'end': t,
                    'interrupted': False
                })
                del queue[matched_index]
                job_next_op[j] = max(job_next_op.get(j, 0), o + 1)

                if matched_index == 0 and queue:
                    machine_proc_start[m] = t

    for b in breakdowns:
        if b['end'] is None:
            b['end'] = max_time

    for m, queue in machine_queues.items():
        if queue:
            front = queue[0]
            start = machine_proc_start.get(m, max_time)
            completed_blocks.append({
                'machine': m, 'job': front['job'], 'op': front['op'],
                'start': start, 'end': max_time,
                'interrupted': False
            })

    return completed_blocks, breakdowns, emergency_jobs, max_machine, max_time

def render_gantt(blocks, breakdowns, emergency_jobs, num_machines, max_time, output_path):
    """Draws the Gantt chart using Matplotlib."""
    fig, ax = plt.subplots(figsize=(14, 7))
    cmap = plt.get_cmap('tab20')
    
    # Setup Axes
    yticks = np.arange(num_machines + 1)
    ax.set_yticks(yticks)
    ax.set_yticklabels([f"Machine {m}" for m in yticks])
    ax.set_ylim(-0.5, num_machines + 0.5)
    ax.set_xlim(0, max_time + 2)
    ax.set_xlabel("Time")
    ax.set_title("DFJSP Schedule with Dynamic Events")
    ax.grid(True, axis='x', linestyle='--', alpha=0.6)

    # 1. Plot Breakdowns (Red Backgrounds & X marks)
    for b in breakdowns:
        m, start, end = b['machine'], b['start'], b['end']
        ax.axvspan(start, end, ymin=(m)/(num_machines+1), ymax=(m+1)/(num_machines+1), 
                   facecolor='red', alpha=0.25)
        ax.text(start, m, 'X', color='red', fontweight='bold', ha='center', va='center', fontsize=12, zorder=5)
        ax.text(end, m, 'X', color='green', fontweight='bold', ha='center', va='center', fontsize=12, zorder=5)

    # 2. Plot Operations
    for blk in blocks:
        m, j, o = blk['machine'], blk['job'], blk['op']
        start, end = blk['start'], blk['end']
        duration = end - start
        
        color = cmap(j % 20)
        
        rect = mpatches.Rectangle((start, m - 0.35), duration, 0.7, 
                                  facecolor=color, edgecolor='black', linewidth=1, zorder=3)
        ax.add_patch(rect)
        
        prefix = "!" if j in emergency_jobs else ""
        label = f"{prefix}J{j}O{o}"
        
        x_center = start + (duration / 2)
        ax.text(x_center, m, label, ha='center', va='center', 
                color='black', fontweight='normal', fontsize=8, zorder=4)

    plt.tight_layout()
    plt.savefig(output_path, dpi=400)
    print(f">>> Gantt chart successfully saved to: {output_path}")
    plt.close()


def generate_gantt_from_events_log(events_log_path, output_path):
    """Generate a Gantt chart image from an events log file."""
    events = parse_events_log(events_log_path)
    if not events:
        raise ValueError(f"Could not load any events from: {events_log_path}")

    blocks, breakdowns, emergencies, max_machine_id, max_time = build_schedule_data(events)
    render_gantt(
        blocks=blocks,
        breakdowns=breakdowns,
        emergency_jobs=emergencies,
        num_machines=max_machine_id,
        max_time=max_time,
        output_path=output_path,
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate a Gantt chart from an events log inside a session folder."
    )
    parser.add_argument(
        "session_folder",
        nargs="?",
        default=config.SESSION_NAME,
        help="Path to the session folder containing the events log file.",
    )
    parser.add_argument(
        "--greedy",
        action="store_true",
        help="Read/write *_greedy files.",
    )
    arguments = parser.parse_args()

    events_log_file_path, output_path = build_io_paths(
        target_folder_path=arguments.session_folder,
        use_greedy_suffix=arguments.greedy,
    )

    events = parse_events_log(events_log_file_path)
    if not events:
        print("Failed to load events. Exiting.")
        exit(1)
        
    blocks, breakdowns, emergencies, max_m, max_t = build_schedule_data(events)
    render_gantt(blocks, breakdowns, emergencies, max_m, max_t, output_path)