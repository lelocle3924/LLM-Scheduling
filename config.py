#=============================================================
# PATHS & PROMPTS
#=============================================================
PROBLEM_FILE = ""
SESSION_NAME = ""
RESULTS_FOLDER = ""

ACTION_PROMPT_FILE = "prompts/decision_prompt.md"
PRIOR_PROMPT_FILE = "prompts/prior_prompt.md"
REFLECT_PROMPT_FILE = "prompts/reflection_prompt.md"

#=============================================================
# STATE VARIABLES
#=============================================================
ALLOW_WAIT = True # manage state to allow jobs assignment before machine available
TARDINESS_OBJECTIVE = "total"  # Options: "total" or "max"

AUTO_GENERATE_GANTT = True  # Generate gantt chart automatically when an instance finishes

# Mode 1: Random events generator
RANDOM_SEED = 0 
NUM_RANDOM_BREAKDOWNS = 0
NUM_RANDOM_EMERGENCIES = 0

# Mode 2: User-specified events file
DYNAMIC_EVENTS_FILE = ""

#=============================================================
# LLM PARAMETERS
#=============================================================
# Global compressed-output persona for all API calls.
CAVEMAN_SYSTEM_PROMPT = (
    "Terse like caveman. Technical substance exact. No fluff. "
    "Drop articles, filler words, pleasantries, and hedging. "
    "Use short sentence fragments. Prefer short synonyms and abbreviations when clear. "
    "Pattern: [thing] [action] [reason]. [next step]. "
    "Keep technical accuracy perfect. Never change requested code semantics. "
    "Never wrap requested JSON outputs in markdown or conversational text. Output requested formats perfectly."
)

LLM_THINKING_EXCLUDE = True  # If True, hide reasoning tokens from API response text field
MAX_TOKENS = 5000
MAX_RETRIES = 4 

PRIOR_MODEL_NAME = "openai/gpt-oss-120b"   #model names can be found on openrouter website      
PRIOR_LLM_TEMPERATURE = 1
PRIOR_THINKING_ENABLED = True
PRIOR_THINKING_EFFORT = "medium"
PRIOR_THINKING_MAX_TOKENS = 500

# REFLECTION LLM PARAMS
REFLECT_MODEL_NAME = "google/gemini-3-flash-preview"
REFLECT_LLM_TEMPERATURE = 1
REFLECT_THINKING_ENABLED = True
REFLECT_THINKING_EFFORT = "high"
REFLECT_THINKING_MAX_TOKENS = 0

#=============================================================
# SEARCH FRAMEWORK PARAMETERS
#=============================================================

# Options: "SingleSearch", "MCTSSearch"
SEARCH_STRATEGY = "MCTSSearch" 

MCTS_MAX_ITERATIONS = 30
MCTS_MIN_ITERATIONS = 10
MCTS_C_PARAM_MAX = 2
MCTS_C_PARAM_MIN = 2
MCTS_PRIOR_MODE = "llm"  # Options: "llm" or "uniform"
MCTS_TREE_VISUALIZATION_ENABLED = False  # If True: draw tree and force exactly 1 scheduling iteration
MCTS_ROLLOUT_POLICY = "random"  # Options: "pdr" or "random"
MCTS_ROLLOUTS_PER_EVAL = 100

MCTS_NORMALIZE_Q_FOR_PUCT = True  # Min-max normalize Q term per parent before PUCT selection
MCTS_NUMBA_ROLLOUT_ENABLED = True  # Uses array-state + JIT path for compatible random rollouts
MCTS_TREE_STATE_BACKEND = "array"  # Options: "array" or "object"
#=============================================================
# STRATEGIC REFLECTION PARAMETERS
#=============================================================
USE_REFLECTION = True
REFLECTION_LEVELS = 1
REFLECTION_MACRO_ROLLOUTS = 12
REFLECTION_MICRO_ROLLOUTS_PER_ACTION = 5

#=============================================================
# PROMPT DETAIL LEVEL
#=============================================================
INCLUDE_FULL_STATE_IN_PROMPT = False  # Inject complete job-operation-machine processing time table (token-heavy)
