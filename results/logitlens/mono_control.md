# Monolingual control for the logit-lens depth effects (CLAUDE.md 6l)

All numbers are fair - starved, paired over identical items, 95% bootstrap CI.

## 1. Commitment layer, English monolinguals repeating ENGLISH words

`en-fair` and `en-starved` are trained on English only. Nothing here is cross-lingual.

| word list | n | d commit layer [95% CI] |
|---|---|---|
| en-de word set | 110 | -0.63 [-0.77, -0.49]* |
| en-fr word set | 99 | -0.55 [-0.68, -0.41]* |
| en-ar word set | 136 | -0.54 [-0.64, -0.43]* |
| en-zh word set | 139 | -0.65 [-0.77, -0.54]* |

## 2. Commitment layer, partner-language monolinguals on their OWN language

| lang | pair | n | d commit layer (rep X→X) | d commit layer (rep en→en) | note |
|---|---|---|---|---|---|
| fr | fr-fair vs fr-starved | 99 | -0.55 [-0.68, -0.42]* | -0.09 [-0.19, +0.01] | both 30B cooled |
| ar | ar-fair vs ar-starved | 134 | -0.75 [-0.87, -0.64]* | -0.23 [-0.32, -0.15]* | both 30B cooled |
| zh | zh-fair-15b vs zh-starved-15b | 136 | -2.04 [-2.11, -1.99]* | +0.10 [-0.01, +0.20] | both 15B mid-stable |
| de | de-fair vs de-starved-16b | 104 | -1.19 [-1.31, -1.07]* | +0.29 [+0.19, +0.40]* | NOT LR-matched: 30B cooled vs 16B mid-stable |

## 3. The bilinguals, same word lists, for reference

| pair | n | d commit layer (rep X→X) | d commit layer (rep en→en) |
|---|---|---|---|
| en-de | 107 | -1.25 [-1.39, -1.10]* | -1.16 [-1.29, -1.04]* |
| en-fr | 99 | -0.98 [-1.16, -0.80]* | -1.17 [-1.30, -1.03]* |
| en-ar | 136 | -0.80 [-0.93, -0.68]* | -0.86 [-0.97, -0.76]* |
| en-zh | 139 | -1.18 [-1.28, -1.08]* | -1.45 [-1.55, -1.33]* |

## 4. Interior factual accuracy (PolyFact, 800 facts -- the set the monolinguals were scored on)

| kind | lang | pair | d acc @L14 [95% CI] | d acc @output [95% CI] | d settle layer [CI] (n) |
|---|---|---|---|---|---|
| mono | fr | fr-fair vs fr-starved | +0.031 [+0.003, +0.061]* | +0.011 [-0.016, +0.039] | -0.67 [-1.02, -0.32]* (111) |
| mono | ar | ar-fair vs ar-starved | +0.072 [+0.037, +0.109]* | +0.028 [+0.003, +0.051]* | -2.42 [-3.02, -1.77]* (52) |
| mono | zh | zh-fair-15b vs zh-starved-15b | +0.065 [+0.037, +0.091]* | +0.022 [-0.005, +0.049] | -1.51 [-1.96, -1.09]* (47) |
| mono | de | de-fair vs de-starved-16b | +0.084 [+0.050, +0.116]* | +0.077 [+0.046, +0.109]* | -1.24 [-1.77, -0.74]* (78) |
| mono | en | en-fair vs en-starved | +0.062 [+0.030, +0.094]* | +0.050 [+0.022, +0.077]* | -0.64 [-0.98, -0.29]* (95) |
| bi | de | en-de-fair vs en-de-starved | +0.076 [+0.045, +0.106]* | +0.005 [-0.021, +0.033] | -0.56 [-0.90, -0.21]* (87) |
| bi | fr | en-fr-fair vs en-fr-starved | +0.064 [+0.036, +0.092]* | +0.006 [-0.021, +0.035] | -0.71 [-1.12, -0.31]* (89) |
| bi | ar | en-ar-fair vs en-ar-starved | +0.081 [+0.044, +0.116]* | +0.010 [-0.016, +0.035] | -1.26 [-2.05, -0.38]* (42) |
| bi | zh | en-zh-fair vs en-zh-starved | +0.075 [+0.049, +0.101]* | +0.009 [-0.014, +0.031] | -0.51 [-0.99, -0.01]* (71) |

Mean d acc @L14: monolingual **+0.058** (LR-matched pairs plus English) vs bilingual **+0.074**.

**Reading.** The effect is present, and of comparable size, in models that have no second language at all. It is therefore a property of what a better tokenizer does to a language model's depth profile, NOT evidence about cross-lingual representation alignment. Note also that a monolingual shows the effect only in the language it was trained on (rep en→en is ~0 for the fr/zh monolinguals), so it requires competence in the language, not merely a shared vocabulary.
