# Rollout policy comparison (RPD): random vs PDR


Averaging RPD across instances reduces skew from raw tardiness magnitude (e.g. normal vs small scale). **Lower** average RPD is better for that policy.

---

## Normal instances

### Per-instance (raw tardiness and RPD)

| Instance | \(T_{\mathrm{random}}\) | \(T_{\mathrm{pdr}}\) | Winner | RPD random (%) | RPD PDR (%) |
| --- | ---: | ---: | --- | ---: | ---: |
| `normal_k1.5_instance_000` | 384.357 | 392.527 | Random | 0 | 2.126 |
| `normal_k1.5_instance_001` | 676.365 | 685.298 | Random | 0 | 1.321 |
| `normal_k1.5_instance_002` | 276.199 | 308.803 | Random | 0 | 11.8 |
| `normal_k1.5_instance_003` | 323.449 | 287.962 | PDR | 12.32 | 0 |
| `normal_k1.5_instance_004` | 703.491 | 708.742 | Random | 0 | 0.7464 |
| `normal_k2.0_instance_000` | 1017.16 | 973.312 | PDR | 4.505 | 0 |
| `normal_k2.0_instance_001` | 440.754 | 469.679 | Random | 0 | 6.563 |
| `normal_k2.0_instance_002` | 179.884 | 197.69 | Random | 0 | 9.899 |
| `normal_k2.0_instance_003` | 376.559 | 356.94 | PDR | 5.496 | 0 |
| `normal_k2.0_instance_004` | 353.128 | 385.803 | Random | 0 | 9.253 |

---

## Small instances

### Per-instance (raw tardiness and RPD)

| Instance | \(T_{\mathrm{random}}\) | \(T_{\mathrm{pdr}}\) | Winner | RPD random (%) | RPD PDR (%) |
| --- | ---: | ---: | --- | ---: | ---: |
| `small_k1.5_instance_000` | 3.733 | 1.294 | PDR | 188.5 | 0 |
| `small_k1.5_instance_001` | 24.442 | 31.462 | Random | 0 | 28.72 |
| `small_k1.5_instance_002` | 0.514 | 1.959 | Random | 0 | 281.1 |
| `small_k1.5_instance_003` | 0.333 | 0.333 | Tie | 0 | 0 |
| `small_k1.5_instance_004` | 11.751 | 9.696 | PDR | 21.19 | 0 |
| `small_k2.0_instance_000` | 4.196 | 3.773 | PDR | 11.21 | 0 |
| `small_k2.0_instance_001` | 3.252 | 2.533 | PDR | 28.39 | 0 |
| `small_k2.0_instance_002` | 0 | 0 | Tie | 0 | 0 |
| `small_k2.0_instance_003` | 4.14 | 4.14 | Tie | 0 | 0 |
| `small_k2.0_instance_004` | 2.328 | 1.045 | PDR | 122.8 | 0 |

