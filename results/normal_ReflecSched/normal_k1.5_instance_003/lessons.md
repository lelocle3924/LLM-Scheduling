<comparison_summary>
Best vs Worst divergence driven by machine selection for J17O2, J18O2, and J11O3. Best rollout utilizes M4 availability at T=26.3 for J18 and J17. Worst rollouts offload these to M1 (available T=40.8), causing cascading delays. Efficiency depends on bypassing overloaded M1.
</comparison_summary>

<key_insights>
**Banned Behaviors:**
- DO NOT route J17, J18, or J21 to M1; M1 backlog until T=40.8 is prohibitive.
- DO NOT assign J11O2 to M1 if M2 (T=35.3) or M4 (T=26.3) available first.
- DO NOT leave M3 idle; it must process J16O2 immediately following repair.

**Bottleneck Focus:**
- M3: Highest contention (10), must prioritize J16 -> J24 sequence.
- M4: Critical for offloading M1/M2 tasks.

**Current Routing Priorities:**
- Assign J16O2 to M3 now.
- Priority for M4 (at T=26.3): J18O2 > J17O2.
- Priority for M3 (after J16O2): J24O1 > J22O1 > J23O1.
- Shift J11O2 and J6O2 to M2 upon T=35.3.
</key_insights>