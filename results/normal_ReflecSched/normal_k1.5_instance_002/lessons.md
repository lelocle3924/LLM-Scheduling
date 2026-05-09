<comparison_summary>
Best outcomes route J18O1 to M1 or M0. Worst outcomes route J18O1 to M4 or result in poor sequencing on M1. M4 is primary global bottleneck (Contention 12). Loading M4 further spikes tardiness (+43.8s) and makespan. M1 loading (Best 1) yields lowest tardiness despite M1 contention (10), provided M1 sequence stays lean.
</comparison_summary>

<key_insights>
**Banned Behaviors:**
- DO NOT route J18 to M4. M4 contention is critical; routing there delays downstream Ops for J14, J17, J13.
- DO NOT sequence J21O1 to M1 or M4 if M2 is available.

**Bottleneck Focus:**
- Machine 4: Highest contention (12). Process J0O3 immediately upon M4 availability to clear downstream flow.
- Machine 1: High contention (10). Priority to J18O1 and J6O2.

**Current Routing Priorities:**
- Route J18O1 to M1. This minimizes tardiness by utilizing M1 capacity before J6O2 arrival.
- Assign J15O2 to M0 immediately upon M0 availability to offload M1/M2.
- Priority sequence on M2: J21O1 -> J3O2 -> J7O3.
- Priority sequence on M0: J15O2 -> J22O2 -> J1O3.
</key_insights>