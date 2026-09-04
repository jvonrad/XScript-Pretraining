## Per-fact question alignment (d') vs answer consistency, PolyFact, 30B finals

| partner | cond | peak layer | mean d' @peak | mean d' @L12 | AUC(d'→consistent) all | AUC on wrong-in-both | P(consistent) | P(consistent \| wrong-in-both) |
|---|---|---|---|---|---|---|---|---|
| de | fair | L1 | 9.035 | 3.866 | 0.495 | 0.462 | 0.762 | 0.817 |
| de | starved | L1 | 8.780 | 3.993 | 0.484 | 0.483 | 0.729 | 0.791 |
| fr | fair | L1 | 7.981 | 3.812 | 0.528 | 0.486 | 0.766 | 0.814 |
| fr | starved | L1 | 9.290 | 3.701 | 0.499 | 0.491 | 0.726 | 0.809 |
| ar | fair | L8 | 3.318 | 3.171 | 0.526 | 0.513 | 0.662 | 0.770 |
| ar | starved | L16 | 3.113 | 2.792 | 0.499 | 0.462 | 0.618 | 0.725 |
| zh | fair | L4 | 3.756 | 3.108 | 0.512 | 0.555 | 0.661 | 0.734 |
| zh | starved | L16 | 3.739 | 3.346 | 0.575 | 0.588 | 0.642 | 0.752 |

## fair − starved: alignment (paired over facts) and mediation (logistic: consistent ~ fair [+ d'])

| partner | Δ mean d' @peak [95% CI] | Δ d' @L12 | β_fair (no d') | β_fair (with d') | shrink | β_d' | same on wrong-in-both: β_fair no/with d' |
|---|---|---|---|---|---|---|---|
| de | **+0.255** [+0.131, +0.372]* | -0.127 | +0.173 | +0.176 | -1% | -0.034 | +0.169 / +0.173 (β_d' -0.099) |
| fr | **-1.309** [-1.414, -1.201]* | +0.111 | +0.210 | +0.223 | -6% | +0.034 | +0.034 / +0.020 (β_d' -0.042) |
| ar | **+0.205** [+0.170, +0.239]* | +0.379 | +0.187 | +0.184 | +2% | +0.020 | +0.235 / +0.250 (β_d' -0.091) |
| zh | **+0.017** [-0.062, +0.100] | -0.237 | +0.082 | +0.083 | -1% | +0.133 | -0.092 / -0.115 (β_d' +0.252) |
