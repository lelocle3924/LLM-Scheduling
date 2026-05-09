<comparison_summary>
All Level 0 rollouts confirm J21O1@M3 as the optimal immediate move. Divergence occurs in post-repair routing of J18O2 and J19O2. Worst outcomes (346-348 tardiness) result from routing both J18 and J19 to M4, overloading the M4/M1 sequence. Best Level 0 outcomes (340 tardiness) utilize M0, but remain inferior to Level 1 Macro insights (334 tardiness) which favored routing J18O2 to M1 and J19O2 to M2.
</comparison_summary>

<key_insights>
**Banned Behaviors:**
- DO NOT route J19O2 or J18O2 to M4; causes cascading delay on M1/M4 (Worst Outcomes 1 & 2).
- DO NOT route J8O2 to M3; M3 must remain clear for J16O2 and J1O2 sequences.
- DO NOT allow J24O1 to occupy M3; keep on M4 or M0.

**Bottleneck Focus:**
- Machine 1: Critical for late-stage completions (J22, J20). Must avoid mid-process congestion.
- Machine 4: High contention (8); avoid long operations like J19O2.

**Current Routing Priorities:**
- Start J21O1 on M3 immediately (PT: 2.196).
- Assign J17O2 to M4 or M1 (Short PT) to clear M2 queue for J19O2.
- Execute Level 1 Macro strategy: Route J18O2 to M1 (T=30.2) and J19O2 to M2 (T=31.2).
</key_insights>