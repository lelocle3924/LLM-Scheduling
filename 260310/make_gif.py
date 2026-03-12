import os
import json
import re
import glob
from PIL import Image
from state_manager import StateManager
import config
import gantt

def parse_action_from_file(filepath):
    """Extracts the JSON decision from a logged interaction text file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Isolate the FULL LLM RESPONSE section to avoid matching random JSON in the prompt
    if "=== FULL LLM RESPONSE ===" in content:
        response_section = content.split("=== FULL LLM RESPONSE ===")[-1]
    else:
        response_section = content
        
    # Extract the dictionary
    match = re.search(r'\{.*?\}', response_section, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
    return None

def main():
    session_folder = config.SESSION_NAME
    problem_file = config.PROBLEM_FILE
    
    print(f"Reading problem data from {problem_file}...")
    with open(problem_file, 'r') as f:
        problem_data = json.load(f)
        
    machines_count = problem_data['machines']
    
    print("Initializing State Manager for replay...")
    sm = StateManager(problem_data)
    
    # Get all .txt interaction files and sort them numerically (1.txt, 2.txt, ... 55.txt)
    txt_files = glob.glob(os.path.join(session_folder, "*.txt"))
    txt_files.sort(key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
    
    history_packages = []
    image_files = []
    
    # Folder to store the individual PNG snapshots
    frames_dir = f"gantt_frames_{config.SESSION_NAME}"
    os.makedirs(frames_dir, exist_ok=True)
    
    print(f"Processing {len(txt_files)} recorded interactions...")
    
    for idx, txt_file in enumerate(txt_files):
        action = parse_action_from_file(txt_file)
        if not action:
            print(f"Skipping {txt_file} - No valid JSON action found.")
            continue
            
        job_id = action.get('job')
        op_id = action.get('op')
        machine_id = action.get('machine')
        
        if job_id is None or op_id is None or machine_id is None:
            continue

        # Replay Logic: Sync the StateManager clock to make the action feasible
        while True:
            actions = sm.get_feasible_actions()
            if not actions:
                # If no actions are ready, advance the simulation clock
                has_more = sm.process_next_event()
                if not has_more:
                    break
                continue
                
            # Validate if the LLM's action from the log is actually feasible
            is_valid = any(a["job"] == job_id and a["op"] == op_id and a["machine"] == machine_id for a in actions)
            
            if is_valid:
                break
            else:
                # In your main3.py, if the LLM hallucinated heavily, it used a fallback.
                # To perfectly mimic the log's reality, we must use the exact same fallback.
                fallback = actions[0]
                job_id, op_id, machine_id = fallback["job"], fallback["op"], fallback["machine"]
                break

        # Calculate exact start and end times
        processing_time = next(cand["processing"] for cand in sm.jobs[job_id][op_id] if cand["machine"] == machine_id)
        start_time = max(sm.current_time, sm.machine_avail[machine_id])
        end_time = start_time + processing_time
        
        # Execute the action to update internal tracking
        sm.execute_action(job_id, op_id, machine_id)
        
        # Record the block for the Gantt chart
        history_packages.append({
            'start': start_time,
            'end': end_time,
            'machine': machine_id,
            'job': job_id,
            'op': op_id
        })
        
        # Render and save the frame
        frame_path = os.path.join(frames_dir, f"frame_{idx+1:03d}.png")
        gantt.save_gantt(history_packages, machines_count, frame_path)
        image_files.append(frame_path)
        print(f"Rendered frame {idx+1}/{len(txt_files)}")
        
    # Compile the PNGs into an animated GIF
    print("Compiling GIF animation...")
    if image_files:
        frames = [Image.open(img) for img in image_files]
        # Duration is the time per frame in milliseconds (e.g., 400ms = 2.5 fps)
        frames[0].save(
            f'scheduling_progression_{config.SESSION_NAME}.gif', 
            format='GIF', 
            append_images=frames[1:], 
            save_all=True, 
            duration=400, 
            loop=0
        )
        print(f">>> SUCCESS: Created 'scheduling_progression_{config.SESSION_NAME}.gif'!")
    else:
        print(">>> ERROR: No frames were generated.")

if __name__ == '__main__':
    main()