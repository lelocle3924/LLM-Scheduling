<comparison_summary>
Micro simulations confirm M0 as the fatal bottleneck. Divergence is extreme: J3O0 to M1 (Best) vs. J3O0 to M0 (Worst). Routing J3O0 to M1 outperforms the Macro-level suggestion of M2 by securing earlier processing without blocking M2 for J4’s requirements.
</comparison_summary>

<key_insights>
**Banned Behaviors:** 
- DO NOT route J3O0 to M0; it causes a 150% increase in total tardiness by blocking late-stage operations.
- DO NOT assign J5O1 to M0 if J1O2 is pending; prioritize M0 for J1/J0 throughput.

**Bottleneck Focus:** 
- M0: Critical for J1 and J0 completion. Must remain available by T=4.9.
- M1: Secondary bottleneck; must absorb J3O0 to protect M0.

**Current Routing Priorities:** 
- J3O0 -> M1: Execute immediately to balance load away from M0 and M2.
- J5O1 -> M1 or M0: Sequence after J3O0 or J1O2 respectively.
- J4O2 -> M2: Utilize M2 availability at T=5.5 for J4 progression.
</key_insights>