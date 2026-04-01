#=============================================================
# PATHS & PROMPTS
#=============================================================
PROBLEM_FILE = r"problem_data/brandimarte/mk01.json"
SESSION_NAME = "Global_LFS_5000_Run"

ACTION_PROMPT_FILE = "prompts/decision_prompt.md"
VALUE_PROMPT_FILE = "prompts/value_prompt.md"
PRIOR_PROMPT_FILE = "prompts/prior_prompt.md"
EXPLORE_PROMPT_FILE = "prompts/explore_prompt.md"
REFLECT_PROMPT_FILE = "prompts/reflection_prompt.md"

#=============================================================
# LLM PARAMETERS
#=============================================================
MODEL_NAME = "google/gemini-2.0-flash-lite-001"
TEMPERATURE = 0.3
MAX_TOKENS = 5000
MAX_RETRIES = 3 

#=============================================================
# STRATEGIC REFLECTION PARAMETERS
#=============================================================
# Note: Reflection is typically turned off during a massive offline search 
# to save compute, but the parameters remain if you build an offline wrapper.
USE_REFLECTION = False
REFLECTION_LEVELS = 2       
ROLLOUTS_PER_LEVEL = 12     

#=============================================================
# SEARCH FRAMEWORK PARAMETERS
#=============================================================
# Options: "SingleSearch", "BeamSearch", "MCTSSearch", "LFSSearch", "GlobalLFS"
SEARCH_STRATEGY = "GlobalLFS" 

# --- Global LFS Specific ---
GLOBAL_LFS_BUDGET = 5000    # The massive offline exploration budget

# --- Legacy Search Parameters (Kept for quick toggling) ---
LFS_ITERATIONS = 15         # For standard online LFS
MCTS_ITERATIONS = 50        # For standard online MCTS
MCTS_C_PARAM = 0.5          # MCTS exploration constant
BEAM_WIDTH = 5              # Beam search parallel timelines
SEARCH_DEPTH = 3            # Beam search depth