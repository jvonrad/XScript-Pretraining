# Alignment: fr-fair-15b (FLORES+ both, n=2009)

Languages embedded: en, de, fr, ar, zh; trained on: fr.

### raw, ref layer

`lex` is the model-free TF-IDF token-overlap floor; `vs lex` below 0 means the model adds nothing on that pair.

| pair | script | trained | layer | top1 A->B | top1 B->A | mutual-NN | cos margin | CKA | lex | vs lex |
|---|---|---|---|---|---|---|---|---|---|---|
| en-de | same | no | 12 | 0.490 | 0.551 | 0.398 | 0.065 | 0.707 | 0.434 | +0.086 |
| en-fr | same | no | 12 | 0.972 | 0.980 | 0.962 | 0.116 | 0.823 | 0.580 | +0.396 |
| en-ar | cross | no | 12 | 0.004 | 0.006 | 0.000 | 0.016 | 0.497 | 0.134 | -0.128 |
| en-zh | cross | no | 12 | 0.342 | 0.390 | 0.208 | 0.054 | 0.623 | 0.198 | +0.168 |
| de-fr | same | no | 12 | 0.200 | 0.368 | 0.159 | 0.056 | 0.656 | 0.342 | -0.059 |
| de-ar | cross | no | 12 | 0.010 | 0.010 | 0.002 | 0.017 | 0.513 | 0.114 | -0.104 |
| de-zh | cross | no | 12 | 0.156 | 0.108 | 0.052 | 0.038 | 0.550 | 0.181 | -0.049 |
| fr-ar | cross | no | 12 | 0.004 | 0.003 | 0.000 | 0.014 | 0.446 | 0.111 | -0.107 |
| fr-zh | cross | no | 12 | 0.290 | 0.120 | 0.061 | 0.049 | 0.572 | 0.173 | +0.032 |
| ar-zh | cross | no | 12 | 0.010 | 0.012 | 0.002 | 0.018 | 0.525 | 0.102 | -0.091 |

### raw, best layer

`lex` is the model-free TF-IDF token-overlap floor; `vs lex` below 0 means the model adds nothing on that pair.

| pair | script | trained | layer | top1 A->B | top1 B->A | mutual-NN | cos margin | CKA | lex | vs lex |
|---|---|---|---|---|---|---|---|---|---|---|
| en-de | same | no | 12 | 0.490 | 0.551 | 0.398 | 0.065 | 0.707 | 0.434 | +0.086 |
| en-fr | same | no | 13 | 0.988 | 0.992 | 0.982 | 0.139 | 0.840 | 0.580 | +0.410 |
| en-ar | cross | no | 0 | 0.035 | 0.029 | 0.022 | 0.042 | 0.164 | 0.134 | -0.102 |
| en-zh | cross | no | 12 | 0.342 | 0.390 | 0.208 | 0.054 | 0.623 | 0.198 | +0.168 |
| de-fr | same | no | 13 | 0.278 | 0.323 | 0.172 | 0.064 | 0.619 | 0.342 | -0.042 |
| de-ar | cross | no | 0 | 0.038 | 0.032 | 0.021 | 0.041 | 0.139 | 0.114 | -0.080 |
| de-zh | cross | no | 12 | 0.156 | 0.108 | 0.052 | 0.038 | 0.550 | 0.181 | -0.049 |
| fr-ar | cross | no | 0 | 0.033 | 0.023 | 0.019 | 0.031 | 0.123 | 0.111 | -0.083 |
| fr-zh | cross | no | 14 | 0.189 | 0.232 | 0.070 | 0.060 | 0.563 | 0.173 | +0.038 |
| ar-zh | cross | no | 0 | 0.023 | 0.015 | 0.009 | 0.040 | 0.153 | 0.102 | -0.083 |

### centered, ref layer

`lex` is the model-free TF-IDF token-overlap floor; `vs lex` below 0 means the model adds nothing on that pair.

| pair | script | trained | layer | top1 A->B | top1 B->A | mutual-NN | cos margin | CKA | lex | vs lex |
|---|---|---|---|---|---|---|---|---|---|---|
| en-de | same | no | 12 | 0.907 | 0.884 | 0.837 | 0.464 | 0.693 | 0.434 | +0.461 |
| en-fr | same | no | 12 | 0.996 | 0.999 | 0.995 | 0.690 | 0.820 | 0.580 | +0.417 |
| en-ar | cross | no | 12 | 0.086 | 0.047 | 0.028 | 0.156 | 0.476 | 0.134 | -0.068 |
| en-zh | cross | no | 12 | 0.867 | 0.771 | 0.711 | 0.433 | 0.625 | 0.198 | +0.620 |
| de-fr | same | no | 12 | 0.793 | 0.826 | 0.716 | 0.391 | 0.644 | 0.342 | +0.467 |
| de-ar | cross | no | 12 | 0.065 | 0.039 | 0.023 | 0.187 | 0.483 | 0.114 | -0.062 |
| de-zh | cross | no | 12 | 0.480 | 0.418 | 0.298 | 0.349 | 0.547 | 0.181 | +0.268 |
| fr-ar | cross | no | 12 | 0.070 | 0.032 | 0.018 | 0.132 | 0.429 | 0.111 | -0.060 |
| fr-zh | cross | no | 12 | 0.841 | 0.714 | 0.655 | 0.383 | 0.580 | 0.173 | +0.605 |
| ar-zh | cross | no | 12 | 0.038 | 0.048 | 0.020 | 0.211 | 0.489 | 0.102 | -0.059 |

### centered, best layer

`lex` is the model-free TF-IDF token-overlap floor; `vs lex` below 0 means the model adds nothing on that pair.

| pair | script | trained | layer | top1 A->B | top1 B->A | mutual-NN | cos margin | CKA | lex | vs lex |
|---|---|---|---|---|---|---|---|---|---|---|
| en-de | same | no | 13 | 0.908 | 0.907 | 0.858 | 0.474 | 0.665 | 0.434 | +0.473 |
| en-fr | same | no | 13 | 0.998 | 0.999 | 0.997 | 0.716 | 0.838 | 0.580 | +0.418 |
| en-ar | cross | no | 0 | 0.074 | 0.067 | 0.054 | 0.072 | 0.163 | 0.134 | -0.063 |
| en-zh | cross | no | 14 | 0.873 | 0.839 | 0.784 | 0.463 | 0.627 | 0.198 | +0.658 |
| de-fr | same | no | 13 | 0.833 | 0.829 | 0.750 | 0.400 | 0.612 | 0.342 | +0.489 |
| de-ar | cross | no | 0 | 0.049 | 0.048 | 0.036 | 0.071 | 0.136 | 0.114 | -0.066 |
| de-zh | cross | no | 14 | 0.558 | 0.514 | 0.416 | 0.370 | 0.539 | 0.181 | +0.355 |
| fr-ar | cross | no | 0 | 0.052 | 0.052 | 0.040 | 0.048 | 0.124 | 0.111 | -0.059 |
| fr-zh | cross | no | 14 | 0.804 | 0.805 | 0.716 | 0.407 | 0.575 | 0.173 | +0.632 |
| ar-zh | cross | no | 0 | 0.049 | 0.045 | 0.030 | 0.075 | 0.152 | 0.102 | -0.055 |

