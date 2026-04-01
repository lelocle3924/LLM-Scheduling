#=============================================================
# PATHS & PROMPTS
#=============================================================
PROBLEM_FILE = r"problem_data/brandimarte/mk01.json"
#SESSION_NAME = "260401_0020_mk01_MCTS_C0.5_Iter21"
SESSION_NAME = "260401_1130_mk01_SingleSearch"
#SESSION_NAME = "260401_0008_mk01_BeamSearch_k5_d3"

ACTION_PROMPT_FILE = "prompts/decision_prompt.md"
VALUE_PROMPT_FILE = "prompts/value_prompt.md"
PRIOR_PROMPT_FILE = "prompts/prior_prompt.md"
EXPLORE_PROMPT_FILE = "prompts/explore_prompt.md"
REFLECT_PROMPT_FILE = "prompts/reflection_prompt.md"

#=============================================================
# STATE VARIABLES
#=============================================================
ALLOW_WAIT = True # manage state to allow jobs assignment before machine available

# Mode 2: User-specified events (Leave as "" to use Mode 1)
DYNAMIC_EVENTS_FILE = "problem_data/events_mk01_few.json"

# Mode 1: Random events generator
RANDOM_SEED = 0 
NUM_RANDOM_BREAKDOWNS = 0
NUM_RANDOM_EMERGENCIES = 0

#=============================================================
# LLM PARAMETERS
#=============================================================
MODEL_NAME = "google/gemini-3.1-flash-lite-preview"
TEMPERATURE = 0.3
MAX_TOKENS = 5000
MAX_RETRIES = 3 

#=============================================================
# STRATEGIC REFLECTION PARAMETERS
#=============================================================
USE_REFLECTION = True
REFLECTION_LEVELS = 2       
ROLLOUTS_PER_LEVEL = 12     

#=============================================================
# SEARCH FRAMEWORK PARAMETERS
#=============================================================
# Options: "SingleSearch", "BeamSearch", "MCTSSearch", "LFSSearch"
SEARCH_STRATEGY = "SingleSearch" 

# Beam Search specific
BEAM_WIDTH = 3          # k: How many parallel timelines to keep at each depth
SEARCH_DEPTH = 3        # d: How many decisions to look ahead before acting

# MCTS specific 
MCTS_ITERATIONS = 21
MCTS_C_PARAM = 0.5

# LFS specific 
LFS_ITERATIONS = 100

#=============================================================
# PROMPT DETAIL LEVEL
#=============================================================
INCLUDE_FULL_STATE_IN_PROMPT = True  # Inject complete job-operation-machine processing time table (token-heavy)

#=============================================================
# GRAPH DIFFUSION MODEL PARAMETERS
#=============================================================
USE_DIFFUSION_HEATMAP = False    # Toggle the auxiliary diffusion pipeline
DIFFUSION_SEED = 42             # RNG seed for reproducible placeholder weights
USE_DIFFUSION_AS_PRIOR = False  # When True, heatmap weights replace LLM get_priors in MCTS