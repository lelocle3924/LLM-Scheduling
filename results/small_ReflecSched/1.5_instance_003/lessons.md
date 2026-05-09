<comparison_summary>
Micro analysis confirms Level 1. J4O0 to M4 reduces tardiness 5x-7x vs M3. M3 saturated by J1, J3, and pending J0O2. J4 delay for M4 availability (T=4.2) prevents M3/M0 queue collapse. J0O1 must seize M2 immediately to maintain flow.
</comparison_summary>

<key_insights>
**Banned Behaviors:**
- DO NOT route J4O0 to M3; forces J0O2/J5O1 into catastrophic delays.
- DO NOT use M0 for J5O2; keep clear for J5O1 (T=5.3).
- DO NOT delay J0O1 start on M2; immediate start critical for J1O3/J3O1 sequence.

**Bottleneck Focus:**
- Machine 3: Extreme saturation. Reserve for J1, J3, and late J0/J5 steps.
- Machine 2: Sequence critical for J0, J1, J3 throughput.

**Current Routing Priorities:**
- J4O0 -> M4 (Queue behind J5O0).
- J0O1 -> M2 (Start immediate).
- Reserve M1 for J4O1 and J5O2 cross-load.
</key_insights>