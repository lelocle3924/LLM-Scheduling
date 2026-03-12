import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

def save_gantt(packages, num_machines, filepath):
    """
    Renders the current state of the schedule and saves it as an image.
    :param packages: List of dictionaries containing {'start', 'end', 'machine', 'job'}
    :param num_machines: Total number of machines
    :param filepath: Path to save the PNG image
    """
    # Clear any previous figures to avoid overlapping plots and memory leaks
    plt.clf() 
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Define a colormap to keep job colors consistent across frames
    cmap = plt.get_cmap('tab20')
    
    ypos = np.arange(num_machines, 0, -1)
    labels = [f"Machine {i}" for i in range(num_machines)]
    
    max_end = 0
    for pkg in packages:
        start = pkg['start']
        end = pkg['end']
        machine = pkg['machine']
        job = pkg['job']
        
        max_end = max(max_end, end)
        
        # Assign consistent color based on job ID
        color = cmap(job % 20)
        
        # Draw the rectangle for the operation
        rect = mpatches.Rectangle((start, num_machines - machine - 0.25),
                                  end - start, 0.5, facecolor=color, edgecolor='black', linewidth=1)
        ax.add_patch(rect)
        
        # Add the job number as text in the center of the block
        x_center = start + (end - start) / 2
        y_center = num_machines - machine
        ax.text(x_center, y_center, str(job), ha='center', va='center', 
                color='white', fontweight='bold', fontsize=9)

    ax.set_yticks(ypos)
    ax.set_yticklabels(labels)
    # Dynamically expand the X-axis as time progresses
    ax.set_xlim(0, max_end + 5 if max_end > 0 else 10)
    ax.set_ylim(0.5, num_machines + 0.5)
    
    ax.set_xlabel("Time")
    ax.set_title("DFJSP Scheduling Progression")
    ax.grid(axis='x', linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(filepath, dpi=150)
    plt.close(fig)