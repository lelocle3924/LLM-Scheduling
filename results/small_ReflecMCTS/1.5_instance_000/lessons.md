<comparison_summary>
Micro simulations confirm Level 1 macro strategy: routing J4O0 to M1 at T=4.0 is optimal. Diversion to M2 increases tardiness by 150% to 360% due to stacking behind J0 and J5. Worst outcomes show M2 load exceeding 7.0 units, while best outcomes redistribute work to M0 and M1, keeping M2 load under 1.5 units.
</comparison_summary>

<key_insights>
**Banned Behaviors:** 
- DO NOT route J4O0 to M2; this is the primary cause of extreme tardiness.
- DO NOT delay J4O0 start past T=4.0; M1 is idle and ready.
- DO NOT assign J5O2 to M2; move to M0 to relieve main bottleneck.

**Bottleneck Focus:** 
- Machine 2 (Primary): Currently over-subscribed. Use M1 and M0 for all flexible operations.
- Machine 0 (Secondary): Critical for J5O2 and J3O1 downstream.

**Current Routing Priorities:** 
- Priority 1: Dispatch J4O0 to M1 immediately (T=4.0).
- Priority 2: Prepare M1 for J2O2 at T=5.4 (sequential flow).
- Priority 3: Route J5O2 to M0 at T=6.5 to bypass M2 congestion.
</key_insights>