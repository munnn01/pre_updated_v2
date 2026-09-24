# Paper-validation reanalysis: H264

Paired source videos: **1000**. This TEST set was previously
inspected; these added comparisons are **exploratory**, not an
independent confirmation. All fixed candidates and frozen A/B/C
arms are reported, without choosing a winner on TEST.

| Arm | Analyzer | BD-rate Top-1 (%) | 95% clip-bootstrap CI | BD-accuracy (pp) | Minimum same-QP Top-1 gap (pp) |
|---|---|---:|---:|---:|---:|
| `area112` | `r2plus1d_18` | -3.38 | [-6.10, -0.63] | +1.37 | -8.00 |
| `area112` | `r3d_18` | -0.65 | [-3.79, +2.33] | +0.24 | -8.90 |
| `area96` | `r2plus1d_18` | +3.67 | [+0.46, +6.82] | -1.64 | -19.90 |
| `area96` | `r3d_18` | +12.07 | [+8.13, +16.18] | -4.91 | -20.50 |
| `area112_up128` | `r2plus1d_18` | +8.45 | [+5.43, +11.43] | -3.54 | -12.20 |
| `area112_up128` | `r3d_18` | +16.12 | [+12.21, +19.89] | -5.98 | -14.90 |
| `blur020_128` | `r2plus1d_18` | +1.06 | [-1.53, +3.53] | -0.30 | -4.70 |
| `blur020_128` | `r3d_18` | +7.35 | [+3.97, +10.42] | -2.75 | -7.20 |
| `blur040_128` | `r2plus1d_18` | +7.92 | [+4.96, +10.80] | -3.15 | -11.70 |
| `blur040_128` | `r3d_18` | +16.80 | [+12.92, +20.64] | -6.09 | -13.80 |
| `v1` | `r2plus1d_18` | -24.88 | [-26.84, -23.00] | +12.17 | +0.70 |
| `v1` | `r3d_18` | -1.03 | [-3.46, +1.55] | +0.47 | -6.00 |
| `A` | `r2plus1d_18` | -12.88 | [-14.29, -11.51] | +5.56 | +0.50 |
| `A` | `r3d_18` | -9.34 | [-10.59, -8.08] | +3.56 | +0.30 |
| `B` | `r2plus1d_18` | -16.33 | [-17.91, -14.74] | +7.35 | +0.50 |
| `B` | `r3d_18` | -11.64 | [-12.98, -10.12] | +4.65 | +0.60 |
| `C` | `r2plus1d_18` | -22.01 | [-23.85, -20.24] | +10.40 | +0.80 |
| `C` | `r3d_18` | -14.47 | [-15.69, -13.21] | +6.09 | +0.70 |

## Direct frozen C versus frozen B

This is a paired C-vs-B rate--accuracy comparison, not the
difference between their BD-rates against identity.

| Analyzer | C vs B BD-rate Top-1 (%) | 95% clip-bootstrap CI |
|---|---:|---:|
| `r2plus1d_18` | -6.61 | [-7.91, -5.34] |
| `r3d_18` | -3.24 | [-4.23, -2.36] |

A negative BD-rate favors the arm named first in the
header only for C vs B; in the main table it favors the
listed arm over the identity128 anchor. The full JSON contains
every QP's curve, choices, provenance and bootstrap counts.
