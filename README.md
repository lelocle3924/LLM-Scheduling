# INTEGRATING LARGE LANGUAGE MODELS IN A MONTE CARLO TREE SEARCH FRAMEWORK FOR OPTIMIZATION
This repository holds the code and data for my thesis project at International University - VNUHCMC
Developed from the key reference: Cao, S., & Yuan, Y. (2025). ReflecSched: Solving Dynamic Flexible Job-Shop Scheduling via LLM-Powered Hierarchical Reflection. arXiv preprint arXiv:2508.01724.

### Input

### To run the code
```
pip install -r requirement.txt
```
- Register an OpenRouter account for an API key. Then create a file named ".env", and write a variable in it as follows: OPENROUTER_API_KEY=sk_your_api_key
- setup the parameter settings in config.py
- to run one problem file, specified in config.py
```
python main.py
```
- to run many problem files at once inside a parent folder
```
python main.py --batch-folder path_to_parent_folder
```
- The same is applied to main_greedy and main_exact