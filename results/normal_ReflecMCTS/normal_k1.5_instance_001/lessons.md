<comparison_summary>
Best outcomes utilize M0 as relief valve for M1. J21O1->M0 immediate start is optimal. Divergence caused by M1 overloading; worst outcomes (tardiness 788+) result from routing J20, J10, and J24 all to M1 while M0 sits idle (Busy 7.7 vs 23.5). Shift J20 and J10 to M0 to maintain M1 flow for J18, J13.
</comparison_summary>

<key_insights>
**Banned Behaviors:**
- DO NOT route J10O3 to M1; it is too long (pt: 3.9-5.0) and will block J17/J27.
- DO NOT route J20O1 to M2; waiting until T=58.9 kills makespan.
- DO NOT let M0 idle after J26O3; must take J20O1 or J10O3.

**Bottleneck Focus:**
- M1 is flow bottleneck; strictly limit it to medium-duration tasks (J18, J13, J17).
- M0 is capacity relief; use to absorb long tasks (J9, J10, J20).

**Current Routing Priorities:**
- Execute J21O1@M0 immediately.
- Next: J26O3@M0 -> J20O1@M0.
- Parallel: J5O3@M1 (current) -> J25O1@M1 -> J18O2@M1.
- M2: Idle until T=58.9, then take J22O2 and J13O3 (if M1 backed up).
</key_insights>