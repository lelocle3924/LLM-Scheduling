<comparison_summary>
Best outcomes balance M0 and M2 loads. Best 1 (27.362 tardy) uses M2 for J2/J1 and M0 for J4/J5. Worst 1 (29.258 tardy) overloads M2 with four consecutive operations (J2, J4, J1, J6), creating a secondary bottleneck. Macro insight to avoid M0 refined: load balancing across M0 and M2 is superior to strictly avoiding M0.
</comparison_summary>

<key_insights>
**Banned Behaviors:** 
- DO NOT stack J2O2 and J4O1 sequentially on M2; this forces J1/J6 delays.
- DO NOT route J6O3 to M1; high processing time vs M0/M2 options.
- DO NOT assign J2O2 to M0 if J4O1 and J5O1 are queued for M0 (Worst 2).

**Bottleneck Focus:** 
- M2: Becomes primary bottleneck if J2 and J4 are both routed there.
- M0: Critical for J4/J5/J6 sequence; must remain clear of J2.

**Current Routing Priorities:** 
- Act: Assign J1O1 to M1 immediately (T=7.0).
- Decision: Pair J2O2 with M2 and J4O1 with M0 to parallelize processing.
- Sequence: Prioritize J4O1 on M0 after J2O1 completes (T=11.5).
</key_insights>