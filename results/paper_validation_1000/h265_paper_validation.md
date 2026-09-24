# Paper-validation reanalysis: H265

Paired source videos: **1000**. This TEST set was previously
inspected; these added comparisons are **exploratory**, not an
independent confirmation. All fixed candidates and frozen A/B/C
arms are reported, without choosing a winner on TEST.

| Arm | Analyzer | BD-rate Top-1 (%) | 95% clip-bootstrap CI | BD-accuracy (pp) | Minimum same-QP Top-1 gap (pp) |
|---|---|---:|---:|---:|---:|
| `area112` | `r2plus1d_18` | -0.34 | [-2.23, +1.59] | +0.22 | -9.90 |
| `area112` | `r3d_18` | +0.40 | [-1.64, +2.42] | -0.22 | -8.80 |
| `area96` | `r2plus1d_18` | +5.94 | [+3.80, +7.95] | -4.11 | -19.50 |
| `area96` | `r3d_18` | +8.26 | [+6.01, +10.59] | -6.18 | -19.70 |
| `area112_up128` | `r2plus1d_18` | +7.47 | [+5.40, +9.54] | -4.76 | -12.60 |
| `area112_up128` | `r3d_18` | +10.44 | [+8.23, +12.88] | -6.26 | -15.10 |
| `blur020_128` | `r2plus1d_18` | +3.04 | [+1.11, +4.87] | -1.81 | -6.20 |
| `blur020_128` | `r3d_18` | +3.75 | [+1.78, +5.72] | -2.10 | -6.60 |
| `blur040_128` | `r2plus1d_18` | +7.20 | [+5.26, +9.19] | -4.36 | -13.80 |
| `blur040_128` | `r3d_18` | +11.24 | [+9.09, +13.54] | -6.78 | -14.60 |
| `v1` | `r2plus1d_18` | -16.09 | [-17.61, -14.63] | +11.51 | +2.50 |
| `v1` | `r3d_18` | -0.99 | [-2.41, +0.45] | +0.67 | -5.50 |
| `A` | `r2plus1d_18` | -8.86 | [-9.90, -7.83] | +5.77 | +0.80 |
| `A` | `r3d_18` | -6.14 | [-6.96, -5.30] | +3.69 | +0.50 |
| `B` | `r2plus1d_18` | -11.02 | [-12.20, -9.84] | +7.50 | +1.20 |
| `B` | `r3d_18` | -7.77 | [-8.80, -6.77] | +4.79 | +0.70 |
| `C` | `r2plus1d_18` | -14.03 | [-15.39, -12.77] | +9.92 | +2.90 |
| `C` | `r3d_18` | -8.72 | [-9.62, -7.82] | +5.53 | +0.70 |

## Direct frozen C versus frozen B

This is a paired C-vs-B rate--accuracy comparison, not the
difference between their BD-rates against identity.

| Analyzer | C vs B BD-rate Top-1 (%) | 95% clip-bootstrap CI |
|---|---:|---:|
| `r2plus1d_18` | -3.36 | [-4.16, -2.56] |
| `r3d_18` | -1.10 | [-1.66, -0.50] |

A negative BD-rate favors the arm named first in the
header only for C vs B; in the main table it favors the
listed arm over the identity128 anchor. The full JSON contains
every QP's curve, choices, provenance and bootstrap counts.
