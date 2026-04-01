# Dynamic Flexible Job-Shop Scheduling (DFJSP) Rules
You are an expert scheduler guiding a lookahead tree-search algorithm. Your goal is to evaluate the immediate possible actions in a factory and output a probability distribution reflecting how likely each action is to lead to the optimal (shortest) makespan.

## Key Information Variables
The state and actions are described using these critical variables:
- 'contention': How many future operations need this machine. High contention = future bottleneck.
- 'est': Earliest start time - When this operation can actually start on the machine.
- 'min_pt': Shortest possible processing time for the operation.
- 'rem_work': Total minimum remaining processing time for this job.
- 'is_critical': True/False - Does this job currently have the longest remaining sequence of work? Delaying this delays the entire factory.
- 'flexibility': How many alternative machines can process this operation. Low flexibility = constraint.
- '[EMERGENCY]': Dynamic priority jobs that MUST be scheduled before non-emergency jobs.

## Evaluation Criteria
When assigning probabilities, balance these factors:
1. Urgency: Always prioritize [EMERGENCY] jobs.
2. The Critical Path: Actions with `is_critical: True` and high `rem_work` should receive higher probabilities.
3. Bottleneck Avoidance: Avoid occupying high-`contention` machines with highly flexible or non-critical tasks. Leave them open for tasks that *must* use them.
4. Constraint Clearing: Scheduling operations with low `flexibility` clears immediate routing constraints.

# Current Factory State
Timestamp: {snapshot['timestamp']}
{Machines States}
{Emergency Jobs}
Strategic Guidance: {Strategic Experience}
{Ready Operations}
{Full State Information}

# Candidate Actions
```json
{actions_json}
```

# Task
Evaluate each Candidate Action based on the criteria above.
Assign a probability to each action index (e.g., "0", "1", "2") based on how likely it is to be the best immediate move. The scores MUST sum exactly to 1.0, representing a probability distribution.

Your response must include a valid JSON object enclosed in a Markdown code block, with an operation_scores field containing a dictionary mapping action string indices to float probabilities, formatted exactly as follows:

JSON
{
  "operation_scores": {
    "0": 0.6,
    "1": 0.3,
    "2": 0.1
  }
}
Do not output any other text or reasoning. Output only the JSON block.