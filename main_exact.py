import json
from ortools.sat.python import cp_model

def solve_fjsp_exact(json_file_path, time_limit_seconds=300):
    # 1. Load Data
    with open(json_file_path, 'r') as f:
        data = json.load(f)
        
    jobs = data['jobs']
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

    # 2. Initialize Model
    model = cp_model.CpModel()
    
    # Store machine intervals for NoOverlap constraint
    machine_to_intervals = {m: [] for m in all_machines}
    
    job_ends = []

    # 3. Create Variables and Constraints
    for job_id, job in enumerate(jobs):
        prev_end_var = None
        
        for op_id, operation in enumerate(job):
            # Main variables for the operation
            op_start = model.NewIntVar(0, horizon, f'start_j{job_id}_o{op_id}')
            op_end = model.NewIntVar(0, horizon, f'end_j{job_id}_o{op_id}')
            op_duration = model.NewIntVar(0, horizon, f'duration_j{job_id}_o{op_id}')
            
            # Precedence Constraint (Operation must start after previous operation ends)
            if prev_end_var is not None:
                model.Add(op_start >= prev_end_var)
            prev_end_var = op_end

            # Track end of the final operation in the job for Makespan calculation
            if op_id == len(job) - 1:
                job_ends.append(op_end)

            # Variables for alternative machines
            alt_presences = []
            
            for alt_id, alt in enumerate(operation):
                m_id = alt['machine']
                duration = alt['processing']
                
                # Boolean variable: True if this machine is selected for this operation
                presence = model.NewBoolVar(f'presence_j{job_id}_o{op_id}_a{alt_id}')
                alt_start = model.NewIntVar(0, horizon, f'alt_start_j{job_id}_o{op_id}_a{alt_id}')
                alt_end = model.NewIntVar(0, horizon, f'alt_end_j{job_id}_o{op_id}_a{alt_id}')
                
                # Optional Interval Variable (Only active if presence == True)
                alt_interval = model.NewOptionalIntervalVar(
                    alt_start, duration, alt_end, presence, 
                    f'interval_j{job_id}_o{op_id}_a{alt_id}'
                )
                
                alt_presences.append(presence)
                machine_to_intervals[m_id].append(alt_interval)
                
                # Link alternative variables to main operation variables if selected
                model.Add(op_start == alt_start).OnlyEnforceIf(presence)
                model.Add(op_duration == duration).OnlyEnforceIf(presence)
                model.Add(op_end == alt_end).OnlyEnforceIf(presence)
                
            # Constraint: Exactly ONE alternative must be selected
            model.AddExactlyOne(alt_presences)

    # 4. Machine Constraints (No overlaps on the same machine)
    for m_id, intervals in machine_to_intervals.items():
        if intervals:
            model.AddNoOverlap(intervals)

    # 5. Objective: Minimize Makespan
    makespan = model.NewIntVar(0, horizon, 'makespan')
    model.AddMaxEquality(makespan, job_ends)
    model.Minimize(makespan)

    # 6. Solve
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit_seconds
    solver.parameters.log_search_progress = True # Shows bounds and gap in real-time
    
    # Optional: Use multiple workers for speed
    solver.parameters.num_search_workers = 8 

    status = solver.Solve(model)

    # 7. Output Results
    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        print(f"\nStatus: {solver.StatusName(status)}")
        print(f"Optimal Makespan: {solver.ObjectiveValue()}")
        print(f"Time Taken: {solver.WallTime():.2f} seconds")
        print(f"Branches: {solver.NumBranches()}")
    else:
        print("\nNo solution found within the time limit.")

if __name__ == '__main__':
    import sys
    # Try it on one of your provided datasets
    if len(sys.argv) > 1:
        solve_fjsp_exact(sys.argv[1], time_limit_seconds=600)
    else:
        solve_fjsp_exact('problem_data/large_scale_JSSP/mt6.json', time_limit_seconds=600)