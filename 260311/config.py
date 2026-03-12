#=============================================================
# PATHS
#=============================================================
PROBLEM_FILE = r"brandimarte/mk01.json"


#=============================================================
# LLM PARAMETERS
#=============================================================
#MODEL_NAME = "arcee-ai/trinity-large-preview:free"
#MODEL_NAME = "stepfun/step-3.5-flash:free"
#MODEL_NAME = "qwen/qwen3-next-80b-a3b-instruct:free"
#MODEL_NAME = "openai/gpt-oss-120b:free"
#MODEL_NAME = "openai/gpt-oss-20b:free"

#MODEL_NAME = "nvidia/nemotron-3-nano-30b-a3b" # SUPER CHEAP
MODEL_NAME = "qwen/qwen3-vl-30b-a3b-thinking" # kinda slow ~100 tps, completely free
TEMPERATURE = 0.0
MAX_TOKENS = 10000
MAX_RETRIES = 3 # max number of invalid responses from LLM

#=============================================================
# SESSION PARAMETERS
#=============================================================
#SESSION_NAME = "ABCXYZ_YYMMDD_HHmm_mkXX"
SESSION_NAME = "test_run_260311_1530_mk01"

#CHECKPOINT_PATH = r"test_run_260310_2300\116.txt" # Leave as "" to run a fresh simulation

