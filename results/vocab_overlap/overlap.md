# Vocabulary overlap between languages, per tokenizer

FLORES+ dev+devtest, 2009 parallel sentences per language (identical content in all five). Specials excluded.


## fair (`unigram_destarved`)

| lang | types used | content types | tokens |
|---|---|---|---|
| en | 8119 | 7740 | 58809 |
| de | 9282 | 8902 | 66451 |
| fr | 8685 | 8371 | 72512 |
| ar | 8943 | 8575 | 60793 |
| zh | 7311 | 6948 | 59519 |

Pieces used by >=1 language: 34111 of 65536; by >=2: 4444; by all 5: 336.


### all pieces

| pair | group | shared types | Jaccard | overlap coef | cov A->B | cov B->A |
|---|---|---|---|---|---|---|
| en-de | same-script | 2844 | 0.195 | 0.350 | 0.553 | 0.481 |
| en-fr | same-script | 3412 | 0.255 | 0.420 | 0.577 | 0.617 |
| en-ar | cross-script | 479 | 0.029 | 0.059 | 0.217 | 0.084 |
| en-zh | cross-script | 1123 | 0.078 | 0.154 | 0.379 | 0.153 |
| de-fr | same-script | 2487 | 0.161 | 0.286 | 0.445 | 0.536 |
| de-ar | cross-script | 453 | 0.025 | 0.051 | 0.173 | 0.083 |
| de-zh | cross-script | 1096 | 0.071 | 0.150 | 0.267 | 0.152 |
| fr-ar | cross-script | 422 | 0.025 | 0.049 | 0.193 | 0.083 |
| fr-zh | cross-script | 1028 | 0.069 | 0.141 | 0.343 | 0.146 |
| ar-zh | cross-script | 399 | 0.025 | 0.055 | 0.080 | 0.126 |

### content pieces only (letter-bearing)

| pair | group | shared types | Jaccard | overlap coef | cov A->B | cov B->A |
|---|---|---|---|---|---|---|
| en-de | same-script | 2533 | 0.180 | 0.327 | 0.496 | 0.413 |
| en-fr | same-script | 3138 | 0.242 | 0.405 | 0.525 | 0.564 |
| en-ar | cross-script | 177 | 0.011 | 0.023 | 0.117 | 0.004 |
| en-zh | cross-script | 832 | 0.060 | 0.120 | 0.301 | 0.034 |
| de-fr | same-script | 2213 | 0.147 | 0.264 | 0.375 | 0.469 |
| de-ar | cross-script | 174 | 0.010 | 0.020 | 0.066 | 0.004 |
| de-zh | cross-script | 825 | 0.055 | 0.119 | 0.173 | 0.034 |
| fr-ar | cross-script | 163 | 0.010 | 0.019 | 0.083 | 0.004 |
| fr-zh | cross-script | 786 | 0.054 | 0.113 | 0.248 | 0.033 |
| ar-zh | cross-script | 128 | 0.008 | 0.018 | 0.003 | 0.010 |

### matched-size overlap (top-K types per language, K equal in both langs and both tokenizers)

| pair | group | K=500 shared | K=1000 shared | K=2000 shared | K=2000 Jaccard | K=2000 mass A | K=2000 mass B |
|---|---|---|---|---|---|---|---|
| en-de | same-script | 87 | 181 | 398 | 0.110 | 0.355 | 0.282 |
| en-fr | same-script | 99 | 213 | 492 | 0.140 | 0.362 | 0.428 |
| en-ar | cross-script | 13 | 26 | 49 | 0.012 | 0.102 | 0.073 |
| en-zh | cross-script | 13 | 36 | 79 | 0.020 | 0.153 | 0.116 |
| de-fr | same-script | 69 | 140 | 341 | 0.093 | 0.292 | 0.354 |
| de-ar | cross-script | 11 | 19 | 46 | 0.012 | 0.109 | 0.072 |
| de-zh | cross-script | 13 | 30 | 83 | 0.021 | 0.184 | 0.116 |
| fr-ar | cross-script | 8 | 20 | 41 | 0.010 | 0.087 | 0.070 |
| fr-zh | cross-script | 11 | 31 | 72 | 0.018 | 0.143 | 0.111 |
| ar-zh | cross-script | 6 | 16 | 40 | 0.010 | 0.065 | 0.105 |

### genuine subwords only (letter-bearing, >=3 chars)

| pair | group | A types | B types | shared | Jaccard | overlap coef | top-500 shared | top-1000 shared |
|---|---|---|---|---|---|---|---|---|
| en-de | same-script | 7281 | 8377 | 2125 | 0.157 | 0.292 | 30 | 84 |
| en-fr | same-script | 7281 | 7883 | 2744 | 0.221 | 0.377 | 50 | 144 |
| en-ar | cross-script | 7281 | 7903 | 87 | 0.006 | 0.012 | 0 | 0 |
| en-zh | cross-script | 7281 | 1193 | 539 | 0.068 | 0.452 | 17 | 73 |
| de-fr | same-script | 8377 | 7883 | 1826 | 0.127 | 0.232 | 18 | 62 |
| de-ar | cross-script | 8377 | 7903 | 84 | 0.005 | 0.011 | 0 | 0 |
| de-zh | cross-script | 8377 | 1193 | 524 | 0.058 | 0.439 | 14 | 75 |
| fr-ar | cross-script | 7883 | 7903 | 79 | 0.005 | 0.010 | 0 | 0 |
| fr-zh | cross-script | 7883 | 1193 | 502 | 0.059 | 0.421 | 15 | 52 |
| ar-zh | cross-script | 7903 | 1193 | 46 | 0.005 | 0.039 | 0 | 0 |

Script of each language's own >=3-char pieces (a language whose long pieces are mostly Latin can only 'share vocabulary' with English trivially):

| lang | >=3-char types | script mix |
|---|---|---|
| en | 7281 | Latin 7281 |
| de | 8377 | Latin 8377 |
| fr | 7883 | Latin 7883 |
| ar | 7903 | Arabic 7814, Latin 89 |
| zh | 1193 | Latin 613, Han 580 |

Mean emitted piece length (chars): en 4.41, de 4.55, fr 4.25, ar 3.77, zh 1.47.


### what the shared pieces ARE

| pair | group | shared types | multi-char % | top script buckets |
|---|---|---|---|---|
| en-de | same-script | 2844 | 96.7 | Latin 2533, sym_num_space 311 |
| en-fr | same-script | 3412 | 97.5 | Latin 3138, sym_num_space 274 |
| en-ar | cross-script | 479 | 89.6 | sym_num_space 302, Latin 177 |
| en-zh | cross-script | 1123 | 91.9 | Latin 832, sym_num_space 291 |
| de-fr | same-script | 2487 | 96.6 | Latin 2213, sym_num_space 274 |
| de-ar | cross-script | 453 | 89.0 | sym_num_space 279, Latin 174 |
| de-zh | cross-script | 1096 | 91.7 | Latin 825, sym_num_space 271 |
| fr-ar | cross-script | 422 | 88.9 | sym_num_space 259, Latin 163 |
| fr-zh | cross-script | 1028 | 92.0 | Latin 786, sym_num_space 242 |
| ar-zh | cross-script | 399 | 87.7 | sym_num_space 271, Latin 128 |

## starved (`unigram_starved`)

| lang | types used | content types | tokens |
|---|---|---|---|
| en | 5767 | 5426 | 70581 |
| de | 5274 | 4928 | 91130 |
| fr | 5285 | 4997 | 94374 |
| ar | 2431 | 2101 | 89725 |
| zh | 3912 | 3562 | 77637 |

Pieces used by >=1 language: 13980 of 65536; by >=2: 4654; by all 5: 352.


### all pieces

| pair | group | shared types | Jaccard | overlap coef | cov A->B | cov B->A |
|---|---|---|---|---|---|---|
| en-de | same-script | 3072 | 0.385 | 0.582 | 0.665 | 0.633 |
| en-fr | same-script | 3572 | 0.478 | 0.676 | 0.713 | 0.746 |
| en-ar | cross-script | 476 | 0.062 | 0.196 | 0.219 | 0.070 |
| en-zh | cross-script | 1136 | 0.133 | 0.290 | 0.403 | 0.125 |
| de-fr | same-script | 2790 | 0.359 | 0.529 | 0.628 | 0.647 |
| de-ar | cross-script | 458 | 0.063 | 0.188 | 0.178 | 0.068 |
| de-zh | cross-script | 1137 | 0.141 | 0.291 | 0.337 | 0.125 |
| fr-ar | cross-script | 428 | 0.059 | 0.176 | 0.184 | 0.068 |
| fr-zh | cross-script | 1048 | 0.129 | 0.268 | 0.345 | 0.118 |
| ar-zh | cross-script | 399 | 0.067 | 0.164 | 0.067 | 0.103 |

### content pieces only (letter-bearing)

| pair | group | shared types | Jaccard | overlap coef | cov A->B | cov B->A |
|---|---|---|---|---|---|---|
| en-de | same-script | 2775 | 0.366 | 0.563 | 0.630 | 0.601 |
| en-fr | same-script | 3315 | 0.466 | 0.663 | 0.685 | 0.722 |
| en-ar | cross-script | 193 | 0.026 | 0.092 | 0.137 | 0.003 |
| en-zh | cross-script | 860 | 0.106 | 0.241 | 0.342 | 0.025 |
| de-fr | same-script | 2528 | 0.342 | 0.513 | 0.597 | 0.611 |
| de-ar | cross-script | 190 | 0.028 | 0.090 | 0.103 | 0.003 |
| de-zh | cross-script | 877 | 0.115 | 0.246 | 0.277 | 0.026 |
| fr-ar | cross-script | 180 | 0.026 | 0.086 | 0.102 | 0.003 |
| fr-zh | cross-script | 813 | 0.105 | 0.228 | 0.275 | 0.024 |
| ar-zh | cross-script | 147 | 0.027 | 0.070 | 0.003 | 0.008 |

### matched-size overlap (top-K types per language, K equal in both langs and both tokenizers)

| pair | group | K=500 shared | K=1000 shared | K=2000 shared | K=2000 Jaccard | K=2000 mass A | K=2000 mass B |
|---|---|---|---|---|---|---|---|
| en-de | same-script | 117 | 250 | 643 | 0.192 | 0.493 | 0.430 |
| en-fr | same-script | 151 | 329 | 836 | 0.264 | 0.486 | 0.557 |
| en-ar | cross-script | 9 | 13 | 79 | 0.020 | 0.135 | 0.063 |
| en-zh | cross-script | 7 | 22 | 169 | 0.044 | 0.234 | 0.100 |
| de-fr | same-script | 103 | 222 | 586 | 0.172 | 0.425 | 0.460 |
| de-ar | cross-script | 8 | 11 | 71 | 0.018 | 0.121 | 0.062 |
| de-zh | cross-script | 6 | 17 | 168 | 0.044 | 0.202 | 0.099 |
| fr-ar | cross-script | 5 | 8 | 65 | 0.017 | 0.119 | 0.060 |
| fr-zh | cross-script | 4 | 16 | 151 | 0.039 | 0.210 | 0.094 |
| ar-zh | cross-script | 4 | 6 | 111 | 0.029 | 0.060 | 0.092 |

### genuine subwords only (letter-bearing, >=3 chars)

| pair | group | A types | B types | shared | Jaccard | overlap coef | top-500 shared | top-1000 shared |
|---|---|---|---|---|---|---|---|---|
| en-de | same-script | 4945 | 4344 | 2338 | 0.336 | 0.538 | 68 | 177 |
| en-fr | same-script | 4945 | 4431 | 2893 | 0.446 | 0.653 | 109 | 291 |
| en-ar | cross-script | 4945 | 1468 | 80 | 0.013 | 0.054 | 0 | 0 |
| en-zh | cross-script | 4945 | 625 | 555 | 0.111 | 0.888 | 57 | 105 |
| de-fr | same-script | 4344 | 4431 | 2107 | 0.316 | 0.485 | 63 | 170 |
| de-ar | cross-script | 4344 | 1468 | 78 | 0.014 | 0.053 | 0 | 0 |
| de-zh | cross-script | 4344 | 625 | 553 | 0.125 | 0.885 | 79 | 144 |
| fr-ar | cross-script | 4431 | 1468 | 74 | 0.013 | 0.050 | 0 | 0 |
| fr-zh | cross-script | 4431 | 625 | 526 | 0.116 | 0.842 | 54 | 93 |
| ar-zh | cross-script | 1468 | 625 | 42 | 0.020 | 0.067 | 0 | 0 |

Script of each language's own >=3-char pieces (a language whose long pieces are mostly Latin can only 'share vocabulary' with English trivially):

| lang | >=3-char types | script mix |
|---|---|---|
| en | 4945 | Latin 4945 |
| de | 4344 | Latin 4344 |
| fr | 4431 | Latin 4431 |
| ar | 1468 | Arabic 1386, Latin 82 |
| zh | 625 | Latin 621, Han 4 |

Mean emitted piece length (chars): en 3.67, de 3.32, fr 3.27, ar 2.55, zh 1.13.


### what the shared pieces ARE

| pair | group | shared types | multi-char % | top script buckets |
|---|---|---|---|---|
| en-de | same-script | 3072 | 97.2 | Latin 2775, sym_num_space 297 |
| en-fr | same-script | 3572 | 97.7 | Latin 3315, sym_num_space 257 |
| en-ar | cross-script | 476 | 88.0 | sym_num_space 283, Latin 193 |
| en-zh | cross-script | 1136 | 92.7 | Latin 860, sym_num_space 276 |
| de-fr | same-script | 2790 | 97.1 | Latin 2528, sym_num_space 262 |
| de-ar | cross-script | 458 | 87.8 | sym_num_space 268, Latin 190 |
| de-zh | cross-script | 1137 | 92.9 | Latin 877, sym_num_space 260 |
| fr-ar | cross-script | 428 | 86.7 | sym_num_space 248, Latin 180 |
| fr-zh | cross-script | 1048 | 92.6 | Latin 813, sym_num_space 235 |
| ar-zh | cross-script | 399 | 86.0 | sym_num_space 252, Latin 147 |

## fair - starved (all pieces / content only)

| pair | group | dJaccard | dJaccard content | dOverlapCoef | dOverlapCoef content |
|---|---|---|---|---|---|
| en-de | same-script | -0.190 | -0.187 | -0.232 | -0.236 |
| en-fr | same-script | -0.223 | -0.224 | -0.256 | -0.258 |
| en-ar | cross-script | -0.033 | -0.015 | -0.137 | -0.069 |
| en-zh | cross-script | -0.054 | -0.046 | -0.137 | -0.122 |
| de-fr | same-script | -0.198 | -0.195 | -0.243 | -0.249 |
| de-ar | cross-script | -0.038 | -0.018 | -0.138 | -0.070 |
| de-zh | cross-script | -0.071 | -0.060 | -0.141 | -0.127 |
| fr-ar | cross-script | -0.034 | -0.016 | -0.127 | -0.066 |
| fr-zh | cross-script | -0.060 | -0.051 | -0.127 | -0.115 |
| ar-zh | cross-script | -0.042 | -0.018 | -0.110 | -0.052 |

## fair - starved, SIZE-MATCHED (top-2000 types per language)

| pair | group | fair shared | starved shared | delta | fair mass A | starved mass A | delta |
|---|---|---|---|---|---|---|---|
| en-de | same-script | 398 | 643 | -245 | 0.355 | 0.493 | -0.138 |
| en-fr | same-script | 492 | 836 | -344 | 0.362 | 0.486 | -0.124 |
| en-ar | cross-script | 49 | 79 | -30 | 0.102 | 0.135 | -0.033 |
| en-zh | cross-script | 79 | 169 | -90 | 0.153 | 0.234 | -0.080 |
| de-fr | same-script | 341 | 586 | -245 | 0.292 | 0.425 | -0.133 |
| de-ar | cross-script | 46 | 71 | -25 | 0.109 | 0.121 | -0.012 |
| de-zh | cross-script | 83 | 168 | -85 | 0.184 | 0.202 | -0.018 |
| fr-ar | cross-script | 41 | 65 | -24 | 0.087 | 0.119 | -0.031 |
| fr-zh | cross-script | 72 | 151 | -79 | 0.143 | 0.210 | -0.067 |
| ar-zh | cross-script | 40 | 111 | -71 | 0.065 | 0.060 | +0.006 |
