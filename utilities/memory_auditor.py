import copy
import tracemalloc
import time
import json
import os
import sys

# Adjust path if running directly from the pragmatics folder
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from state_manager import StateManager
import config

def audit_deepcopy_cost(problem_filepath: str, iterations: int = 100):
    """
    Profiles the exact memory footprint and execution time of deepcopying the StateManager.
    Runs multiple iterations to get a stable average.
    """
    print(f"=== Auditing Deepcopy Cost: {os.path.basename(problem_filepath)} ===")
    
    # 1. Initialize the Ground Truth State
    with open(problem_filepath, "r") as f:
        problem_data = json.load(f)
    sm = StateManager(problem_data)
    
    # Fast-forward slightly to populate some dynamic variables (like ready operations)
    if sm.get_feasible_actions():
        action = sm.get_feasible_actions()[0]
        sm.execute_action(action["job"], action["op"], action["machine"])
    
    # 2. Warm up the JIT compiler/memory allocator
    _ = copy.deepcopy(sm)

    # 3. Track a Single Copy for Exact Memory Metrics
    tracemalloc.start()
    start_time = time.perf_counter()
    
    sm_clone = copy.deepcopy(sm)
    
    end_time = time.perf_counter()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    single_time_ms = (end_time - start_time) * 1000
    kept_kb = current / 1024
    peak_kb = peak / 1024
    
    # 4. Track Bulk Copies for Tree Search Extrapolations
    start_time_bulk = time.perf_counter()
    bulk_clones = []
    for _ in range(iterations):
        bulk_clones.append(copy.deepcopy(sm))
    end_time_bulk = time.perf_counter()
    
    avg_time_ms = ((end_time_bulk - start_time_bulk) / iterations) * 1000

    print(f"\n--- Single Clone Metrics ---")
    print(f"Memory Kept:     {kept_kb:.2f} KB  (Actual size of the cloned state)")
    print(f"Memory Spiked:   {peak_kb:.2f} KB  (Peak RAM used during the copy process)")
    print(f"Time Taken:      {single_time_ms:.2f} ms")
    
    print(f"\n--- Tree Search Projections ({iterations} clones) ---")
    print(f"Average Time:    {avg_time_ms:.2f} ms per clone")
    print(f"Est. RAM Burden: {(kept_kb * iterations) / 1024:.2f} MB held in memory")
    
    # 5. Provide Architectural Recommendations
    print("\n--- Diagnostic ---")
    if avg_time_ms > 5.0:
        print("[WARNING] Deepcopy is taking >5ms. In a lookahead search with hundreds of nodes, this will bottleneck the CPU.")
    if kept_kb > 50:
        print("[WARNING] StateManager footprint is >50KB. Consider implementing __deepcopy__ to skip caching static problem data (like the raw job dictionary).")

def track_function_memory(func):
    """A decorator you can add to any function to log its memory usage."""
    def wrapper(*args, **kwargs):
        tracemalloc.start()
        start_t = time.perf_counter()
        
        result = func(*args, **kwargs)
        
        end_t = time.perf_counter()
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        print(f"[AUDIT] {func.__name__} | Time: {(end_t - start_t)*1000:.2f}ms | Added RAM: {current/1024:.2f}KB | Peak RAM: {peak/1024:.2f}KB")
        return result
    return wrapper

if __name__ == "__main__":
    # Test on the current problem file from config
    audit_deepcopy_cost(config.PROBLEM_FILE, iterations=100)