import os
import re
import ast
import argparse
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.cm as cm
import numpy as np
import config

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
    """Processes chronological events to build operation blocks and breakdown intervals."""
    active_ops = {}       
    pending_resumes = {}  # FALLBACK: Tracks operations waiting for a repair
    
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

        elif e_type == "Action_Executed":
            match = re.search(r'Job (\d+), Op (\d+) -> Mach (\d+)', str(details))
            if match:
                j, o, m = int(match.group(1)), int(match.group(2)), int(match.group(3))
                active_ops[m] = {'job': j, 'op': o, 'start': t}
                max_machine = max(max_machine, m)

        elif e_type == "Machine_Breakdown":
            m = details.get('machine_id') if isinstance(details, dict) else None
            if m is None: continue
            breakdowns.append({'machine': m, 'start': t, 'end': None})
            
            # Close the first half of the operation block
            if m in active_ops:
                op_data = active_ops.pop(m)
                completed_blocks.append({
                    'machine': m, 'job': op_data['job'], 'op': op_data['op'],
                    'start': op_data['start'], 'end': t,
                    'interrupted': True
                })
                # ALWAYS save to pending_resumes as a bulletproof fallback
                pending_resumes[m] = {'job': op_data['job'], 'op': op_data['op']}

        elif e_type == "Machine_Repair":
            m = details.get('machine_id') if isinstance(details, dict) else None
            if m is None: continue
            
            # Close the breakdown interval
            for b in breakdowns:
                if b['machine'] == m and b['end'] is None:
                    b['end'] = t
                    break
            
            # Try to grab explicit log data first
            resumed_job = details.get('resumed_job') if isinstance(details, dict) else None
            resumed_op = details.get('resumed_op') if isinstance(details, dict) else None
            
            if resumed_job is not None and resumed_op is not None:
                # Explicit resume (New state_manager logic)
                active_ops[m] = {'job': resumed_job, 'op': resumed_op, 'start': t}
                if m in pending_resumes:
                    del pending_resumes[m] # Clean up fallback
            elif m in pending_resumes:
                # Implicit resume (Fallback for older logs)
                resumed_op_data = pending_resumes.pop(m)
                active_ops[m] = {'job': resumed_op_data['job'], 'op': resumed_op_data['op'], 'start': t}

        elif e_type == "Operation_Completion":
            if isinstance(details, dict):
                m = details.get('machine_id')
                j = details.get('job_id')
                o = details.get('op_id')
                
                # Check if this completion matches an active operation
                if m in active_ops and active_ops[m]['job'] == j and active_ops[m]['op'] == o:
                    # Pop from active and finally append to completed_blocks!
                    op_data = active_ops.pop(m)
                    completed_blocks.append({
                        'machine': m, 'job': op_data['job'], 'op': op_data['op'],
                        'start': op_data['start'], 'end': t,
                        'interrupted': False
                    })
                
    # Cleanup any unfinished breakdowns or active ops at the end of the log
    for b in breakdowns:
        if b['end'] is None:
            b['end'] = max_time
            
    for m, op_data in active_ops.items():
        completed_blocks.append({
            'machine': m, 'job': op_data['job'], 'op': op_data['op'],
            'start': op_data['start'], 'end': max_time,
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
                color='black', fontweight='bold', fontsize=5, zorder=4)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f">>> Gantt chart successfully saved to: {output_path}")
    plt.close()

if __name__ == "__main__":
    session_folder = config.SESSION_NAME + "_greedy"
    log_file = session_folder + "/events_log_"+config.SESSION_NAME+"_greedy.txt"
    output_path = session_folder + "/final_gantt_"+config.SESSION_NAME+"_greedy.png"
    events = parse_events_log(log_file)
    if not events:
        print("Failed to load events. Exiting.")
        exit(1)
        
    blocks, breakdowns, emergencies, max_m, max_t = build_schedule_data(events)
    render_gantt(blocks, breakdowns, emergencies, max_m, max_t, output_path)