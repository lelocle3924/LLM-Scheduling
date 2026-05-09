<comparison_summary>
Micro-simulations confirm Macro prediction: J3O0 to M0 is catastrophic, increasing tardiness by 193%+ compared to M1. Routing J3O0 to M1 is 55% more effective than M2 (5.4 vs 8.4 tardiness), as it preserves M2 capacity for J4O2 and balances the M1 queue.
</comparison_summary>

<key_insights>
**Banned Behaviors:** DO NOT route J3O0 to M0; it forces a massive queue backlog for J0 and J1. DO NOT route J3O0 to M2; this creates a secondary bottleneck that delays J4O2 by 3.3 units. 
**Bottleneck Focus:** Machine 0 (Critical). Machine 1 (Secondary). Careful staggering of J3 and J5 operations is required to prevent concurrent peaks.
**Current Routing Priorities:** J3O0 -> M1 (Immediate). Preserve M0 for early completion of J1O2 (T=4.9) and J0O3. J5O1 -> M1 if M1 becomes free before M0, otherwise M0 at T=7.0.
</key_insights>