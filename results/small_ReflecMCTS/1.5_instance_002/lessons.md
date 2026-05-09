<comparison_summary>
Micro rollouts validate Level 1 insights. J4O3->M1 reduces makespan by 0.226 vs J4O3->M0. Tardiness parity (3.963) confirms delay is job-intrinsic, not routing-dependent. M1 repair restoration is critical path for throughput.
</comparison_summary>

<key_insights>
**Banned Behaviors:** DO NOT assign J4O3 to M0. DO NOT hold J4O3 for M4.
**Bottleneck Focus:** M1. Proper utilization critical for makespan minimization.
**Current Routing Priorities:** Immediately dispatch J4O3 to M1. Prioritize M1 for J4 sequence. Reserve M0/M3 for overflow or distinct job streams.
</key_insights>