# Bilingual alignment trajectory: layer x budget x pooling

`mutual_nn`, `centered` variant, each bilingual on its OWN trained pair, FLORES+ dev+devtest (n=2009). 48/48 checkpoints.

**Budgets are TOTAL tokens**; the bilinguals mix 50/50, so a 23B run has seen ~11.4B of each language. Every checkpoint at 23B and below is mid-stable at peak LR 3.0e-3 (decay starts at 24B), so those columns are LR-matched by construction; the **30B column is cooled** and is NOT on the same curve -- see CLAUDE.md section 6.

`SAT` = both arms above 0.95: the peak gap there is a ceiling effect, not a measurement. Do not read a decay-with-budget trend across a row that changes from `no` to `SAT`.

## Cross-pooling summary

`mid` = mean fair-starved gap over L5-8; `peak` = gap between each arm's own best layer (⚠ = ceilinged).

### en-de

| pooling | 2B mid / peak | 5B mid / peak | 10B mid / peak | 15B mid / peak | 23B mid / peak | 30B mid / peak |
|---|---|---|---|---|---|---|
| mean | +0.305 / +0.005 ⚠ | +0.210 / +0.000 ⚠ | +0.208 / -0.001 ⚠ | +0.178 / +0.001 ⚠ | -0.023 / -0.001 ⚠ | +0.300 / +0.001 ⚠ |
| mean_nobos | +0.303 / +0.005 ⚠ | +0.212 / +0.001 ⚠ | +0.162 / +0.000 ⚠ | +0.217 / +0.001 ⚠ | -0.016 / +0.000 ⚠ | +0.293 / +0.002 ⚠ |
| weighted | +0.321 / +0.009 ⚠ | +0.239 / +0.001 ⚠ | +0.180 / +0.000 ⚠ | +0.266 / +0.004 ⚠ | -0.035 / +0.001 ⚠ | +0.338 / +0.001 ⚠ |
| last | +0.287 / +0.046 | +0.317 / +0.008 | +0.293 / +0.024 | +0.477 / +0.280 | -0.083 / +0.103 | +0.017 / +0.027 |

### en-fr

| pooling | 2B mid / peak | 5B mid / peak | 10B mid / peak | 15B mid / peak | 23B mid / peak | 30B mid / peak |
|---|---|---|---|---|---|---|
| mean | +0.116 / +0.003 ⚠ | +0.087 / +0.000 ⚠ | +0.103 / +0.000 ⚠ | +0.325 / +0.002 ⚠ | +0.146 / +0.001 ⚠ | +0.145 / -0.000 ⚠ |
| mean_nobos | +0.209 / +0.003 ⚠ | +0.100 / +0.000 ⚠ | +0.072 / -0.000 ⚠ | +0.342 / +0.001 ⚠ | +0.134 / +0.003 ⚠ | +0.095 / +0.000 ⚠ |
| weighted | +0.218 / +0.002 ⚠ | +0.127 / +0.000 ⚠ | +0.113 / +0.000 ⚠ | +0.476 / +0.001 ⚠ | +0.218 / +0.002 ⚠ | +0.229 / +0.000 ⚠ |
| last | +0.206 / +0.011 ⚠ | +0.263 / +0.006 ⚠ | +0.236 / +0.001 ⚠ | +0.700 / +0.598 | +0.046 / +0.513 | +0.025 / +0.377 |

### en-ar

| pooling | 2B mid / peak | 5B mid / peak | 10B mid / peak | 15B mid / peak | 23B mid / peak | 30B mid / peak |
|---|---|---|---|---|---|---|
| mean | +0.197 / +0.130 | +0.605 / +0.027 ⚠ | +0.642 / +0.015 ⚠ | +0.418 / +0.013 ⚠ | +0.384 / +0.003 ⚠ | +0.456 / +0.001 ⚠ |
| mean_nobos | +0.214 / +0.129 | +0.605 / +0.030 ⚠ | +0.652 / +0.007 ⚠ | +0.599 / +0.006 ⚠ | +0.460 / +0.006 ⚠ | +0.493 / -0.004 ⚠ |
| weighted | +0.234 / +0.144 | +0.606 / +0.024 ⚠ | +0.621 / +0.004 ⚠ | +0.550 / +0.005 ⚠ | +0.365 / -0.015 ⚠ | +0.429 / -0.011 ⚠ |
| last | +0.084 / +0.104 | +0.295 / +0.069 | +0.387 / +0.058 | +0.416 / +0.059 | +0.071 / -0.369 | +0.130 / -0.005 |

### en-zh

| pooling | 2B mid / peak | 5B mid / peak | 10B mid / peak | 15B mid / peak | 23B mid / peak | 30B mid / peak |
|---|---|---|---|---|---|---|
| mean | +0.254 / +0.129 | +0.468 / -0.002 | +0.487 / -0.009 ⚠ | +0.703 / +0.029 ⚠ | +0.412 / +0.002 ⚠ | +0.362 / +0.006 ⚠ |
| mean_nobos | +0.203 / +0.177 | +0.388 / +0.022 ⚠ | +0.189 / +0.004 ⚠ | +0.700 / +0.023 ⚠ | +0.634 / +0.016 ⚠ | +0.612 / +0.027 ⚠ |
| weighted | +0.193 / +0.148 | +0.391 / +0.007 ⚠ | +0.221 / +0.003 ⚠ | +0.772 / +0.036 ⚠ | +0.668 / +0.020 ⚠ | +0.580 / +0.025 ⚠ |
| last | +0.065 / +0.103 | +0.236 / +0.055 | +0.243 / +0.035 | +0.426 / +0.276 | +0.414 / +0.213 | +0.496 / +0.210 |

