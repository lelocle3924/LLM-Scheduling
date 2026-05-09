<comparison_summary>
J4O0 to M4 superior. J4O0 to M3 causes 8x tardiness increase. M3 congestion from J4O0 delays J0O2 and J5O1 sequence. Level 0 validates Level 1: M4 queueing J4 is optimal despite J5 currently processing there.
</comparison_summary>

<key_insights>
**Banned Behaviors:**
- DO NOT route J4O0 to M3; triggers catastrophic ripple delays on J0 and J5.
- DO NOT delay J0O1 start on M2; M2 must clear for J1 and J3 immediately.
- DO NOT route J0O2 to M3 if J5O1 already assigned there.

**Bottleneck Focus:**
- M3: Strategic preserve. Avoid stacking J4 and J0/J5 here.
- M4: Primary sink for J4O0 and J5O2.
- M2: Short-term throughput critical.

**Current Routing Priorities:**
- J4O0 -> M4. Wait for J5O0 completion (T=3.7).
- J0O1 -> M2. Execution at T=2.8 mandatory.
- J5O1 -> M3. Only after J0O1 clears M2 to balance downstream load.
- J1O3 -> M2. Second in M2 priority after J0O1.
</key_insights>