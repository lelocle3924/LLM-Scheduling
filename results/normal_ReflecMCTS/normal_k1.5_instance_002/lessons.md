<comparison_summary>
Immediate action J0O3->M4 is fixed. Divergence occurs in secondary assignments for J13O3 and J16O1. Best performance (T=335) requires J13O3->M0 immediately at T=24.0, which offloads M1 for J14 and J18. Worst outcomes route J13O3->M1 or J16O1->M3, causing late-stage M3 saturation (busy time >19) and shifting bottleneck pressure to M1 or M4 improperly. Best results keep M3 busy time below 14.
</comparison_summary>

<key_insights>
**Banned Behaviors:**
- DO NOT route J13O3 to M1; it must go to M0 to preserve M1 capacity for J14/J18.
- DO NOT assign J16O1 to M3; this triggers a bottleneck cascade on the most constrained machine.
- DO NOT allow M0 to remain idle past T=24.0; it must start J13O3 or J16O1 (J13O3 preferred).

**Bottleneck Focus:**
- Machine 3 remains the primary global bottleneck; M1 is the secondary local bottleneck for this window.
- Machine 0 is the primary relief valve.

**Current Routing Priorities:**
- J13O3 -> M0 (Critical: must start at T=24.0).
- J16O1 -> M1 (If M0 is busy with J13O3) or M0.
- J14O3 -> M4 (To prevent M1 overload).
- J23O1 -> M1 (Earliest possible window to free J23 for subsequent stages).
</key_insights>