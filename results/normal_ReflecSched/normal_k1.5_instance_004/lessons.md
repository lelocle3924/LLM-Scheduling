<comparison_summary>
Best outcomes leverage Machine 1's immediate availability to offload Machine 0. Divergence driven by late-stage routing: Best rolls utilize M1/M2 for J23O2, J18O2, and J13O3. Worst rolls overload M0 post-T=61.9, increasing makespan by ~12% and tardiness by ~30. Best scenario uses M1 as primary sink for next 15 time units.
</comparison_summary>

<key_insights>
**Banned Behaviors:**
- DO NOT route J23O2 or J18O2 to Machine 0; M0 is already over-committed (Contention 8).
- DO NOT leave Machine 1 idle post-repair; immediate utilization is required to stabilize makespan.
- DO NOT assign J13O3 to M0 if M1 is available; M1 processing is significantly faster for the current flow.

**Bottleneck Focus:**
- Machine 0 is the critical bottleneck (Busy until 61.9, 8 jobs queued). All routing must aim to minimize further M0 arrivals.

**Current Routing Priorities:**
- J21O2 -> M1 immediately (T=45.8).
- Target M2 for J23O2 and J18O2 once M2 becomes available (T=53.3).
- Reserve M1 for J13O3 and J16O3 to utilize high-speed processing on just-repaired capacity.
</key_insights>