<comparison_summary>
Micro simulations contradict Macro insight: J4O0 -> M1 outperforms J4O0 -> M2 despite later start (T=5.1 vs T=4.4). Offloading J4O0 to M1 mitigates severe M2 contention (5) and enables earlier J3O1 start on M2. Worst outcomes confirm J3O1 -> M0 remains a catastrophic choice, delaying J5 sequence.
</comparison_summary>

<key_insights>
**Banned Behaviors:**
- DO NOT route J4O0 to M2; though M2 opens sooner, its high contention makes it less efficient than M1.
- DO NOT route J3O1 to M0; consumes capacity required for J5O2.
- DO NOT route J4O1 to M2 if M3 is available; underutilizing M3 drives tardiness.

**Bottleneck Focus:**
- M2: Extreme contention (5). Use strictly for J3O1 and J5O2 following J0 completion.
- M1: Secondary bottleneck; must absorb J4O0 to relieve M2.

**Current Routing Priorities:**
- Route J4O0 to M1 (available T=5.1).
- Route J3O1 to M2 immediately after J0 finish (T=4.4, but J3 ready T=5.8).
- Route J4O1 to M3 (available T=5.8).
- Reserve M0 for J5O1/J5O2.
</key_insights>