<comparison_summary>
J4O1 to M0 outperforms J4O1 to M2 by 3.3s tardiness reduction. M2 routing creates critical path congestion for J2 and J6 successors. Sim results validate L1 ban on M2 for J4, but reveal failure to utilize M1 (repaired) in tested trajectories. M0 queueing is acceptable; M2 queueing is terminal.
</comparison_summary>

<key_insights>
**Banned Behaviors:** DO NOT route J4O1 to M2. DO NOT allow M1 to remain idle at T=7.4 while M0 and M2 have pending queues. DO NOT assign J2O2 to M0 if M2 is free at T=13.9.

**Bottleneck Focus:** Machine 2 delay (T=13.9) dictates schedule tail. Machine 0 requires strict sequence [J4, J5, J3] to minimize successor interference.

**Current Routing Priorities:** Dispatch J4O1 to M1 immediately (earliest start). Assign J1O2 and J6O1 to M1 or M0 if pt < wait time for M2. Use M1 to offload M0 volume. Ensure J5O1 starts on M0 at T=11.2 to trigger M0 availability for remaining tasks.
</key_insights>