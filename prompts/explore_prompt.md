# Factory State Exploration Decision
You are an expert scheduler guiding a lookahead tree-search algorithm. 
You must evaluate the current factory state and decide whether to CONTINUE scheduling down this specific timeline, or EXPLORE a different timeline (backtrack).

## Decision Logic
- **CONTINUE**: Choose this if the flow looks healthy, high-contention machines are well utilized, and the Lower Bound is stable.
- **EXPLORE**: Choose this if the state looks deadlocked, bad jobs were assigned to critical machines, or the Mathematical Lower Bound is unexpectedly high.

# Current Factory State
Timestamp: {snapshot['timestamp']}
{Machines States}
{Ready Operations}
{Full State Information}

Mathematical Constraint (Lower Bound): 
The absolute minimum theoretical makespan from this state is {snapshot['lower_bound']}.

# Strategic Guidance
{Strategic Experience}

# Task
Evaluate the state. Do not output lengthy reasoning. 
You MUST output your decision as either EXPLORE or CONTINUE inside XML tags exactly like this:
<decision>CONTINUE</decision>