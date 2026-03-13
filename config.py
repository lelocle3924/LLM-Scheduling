#=============================================================
# PATHS
#=============================================================
PROBLEM_FILE = r"brandimarte/mk04.json"

#=============================================================
# DYNAMIC EVENTS
#=============================================================
# Mode 2: User-specified events (Leave as "" to use Mode 1)
DYNAMIC_EVENTS_FILE = "events_mk01_many.json" 

# Mode 1: Random events generator (Used if DYNAMIC_EVENTS_FILE is empty or not found)
RANDOM_SEED = 42 # Set to None for true randomness
NUM_RANDOM_BREAKDOWNS = 0
NUM_RANDOM_EMERGENCIES = 0

#=============================================================
# LLM PARAMETERS
#=============================================================
#MODEL_NAME = "nvidia/nemotron-3-nano-30b-a3b" # SUPER CHEAP
#MODEL_NAME = "liquid/lfm2-8b-a1b" # 0.01, 0.02
#MODEL_NAME = "openrouter/hunter-alpha"
MODEL_NAME = "google/gemini-2.0-flash-lite-001" # 0.075, 0.3 ~5s per request
TEMPERATURE = 0.0
MAX_TOKENS = 5000
MAX_RETRIES = 3 # max number of invalid responses from LLM

#=============================================================
# SESSION PARAMETERS
#=============================================================
#SESSION_NAME = "ABCXYZ_YYMMDD_HHmm_mkXX"
SESSION_NAME = "test_run_260313_1615_mk04"

#CHECKPOINT_PATH = r"test_run_260312/test_run_260313_1120_mk01/20.txt" # Leave as "" to run a fresh simulation

