# Alignment: en-starved (FLORES+ both, n=2009)

Languages embedded: en, de, fr, ar, zh; trained on: en.

### raw, ref layer

`lex` is the model-free TF-IDF token-overlap floor; `vs lex` below 0 means the model adds nothing on that pair.

| pair | script | trained | layer | top1 A->B | top1 B->A | mutual-NN | cos margin | CKA | lex | vs lex |
|---|---|---|---|---|---|---|---|---|---|---|
| en-de | same | no | 12 | 0.044 | 0.109 | 0.036 | 0.041 | 0.629 | 0.457 | -0.381 |
| en-fr | same | no | 12 | 0.061 | 0.369 | 0.056 | 0.057 | 0.696 | 0.645 | -0.430 |
| en-ar | cross | no | 12 | 0.001 | 0.003 | 0.000 | 0.015 | 0.542 | 0.133 | -0.131 |
| en-zh | cross | no | 12 | 0.046 | 0.451 | 0.026 | 0.055 | 0.677 | 0.196 | +0.052 |
| de-fr | same | no | 12 | 0.187 | 0.211 | 0.104 | 0.045 | 0.734 | 0.365 | -0.166 |
| de-ar | cross | no | 12 | 0.003 | 0.002 | 0.000 | 0.020 | 0.624 | 0.113 | -0.110 |
| de-zh | cross | no | 12 | 0.037 | 0.068 | 0.010 | 0.034 | 0.593 | 0.179 | -0.126 |
| fr-ar | cross | no | 12 | 0.002 | 0.009 | 0.000 | 0.020 | 0.650 | 0.113 | -0.107 |
| fr-zh | cross | no | 12 | 0.125 | 0.087 | 0.015 | 0.042 | 0.629 | 0.177 | -0.071 |
| ar-zh | cross | no | 12 | 0.009 | 0.004 | 0.001 | 0.019 | 0.616 | 0.106 | -0.099 |

### raw, best layer

`lex` is the model-free TF-IDF token-overlap floor; `vs lex` below 0 means the model adds nothing on that pair.

| pair | script | trained | layer | top1 A->B | top1 B->A | mutual-NN | cos margin | CKA | lex | vs lex |
|---|---|---|---|---|---|---|---|---|---|---|
| en-de | same | no | 0 | 0.225 | 0.154 | 0.139 | 0.110 | 0.202 | 0.457 | -0.267 |
| en-fr | same | no | 0 | 0.336 | 0.208 | 0.187 | 0.140 | 0.235 | 0.645 | -0.373 |
| en-ar | cross | no | 0 | 0.029 | 0.023 | 0.015 | 0.026 | 0.122 | 0.133 | -0.107 |
| en-zh | cross | no | 0 | 0.091 | 0.053 | 0.039 | 0.070 | 0.176 | 0.196 | -0.124 |
| de-fr | same | no | 13 | 0.204 | 0.279 | 0.125 | 0.051 | 0.720 | 0.365 | -0.123 |
| de-ar | cross | no | 0 | 0.018 | 0.017 | 0.011 | 0.022 | 0.098 | 0.113 | -0.096 |
| de-zh | cross | no | 0 | 0.037 | 0.036 | 0.024 | 0.045 | 0.172 | 0.179 | -0.142 |
| fr-ar | cross | no | 0 | 0.017 | 0.019 | 0.012 | 0.019 | 0.099 | 0.113 | -0.095 |
| fr-zh | cross | no | 0 | 0.029 | 0.026 | 0.018 | 0.042 | 0.148 | 0.177 | -0.150 |
| ar-zh | cross | no | 0 | 0.021 | 0.009 | 0.005 | 0.025 | 0.120 | 0.106 | -0.091 |

### centered, ref layer

`lex` is the model-free TF-IDF token-overlap floor; `vs lex` below 0 means the model adds nothing on that pair.

| pair | script | trained | layer | top1 A->B | top1 B->A | mutual-NN | cos margin | CKA | lex | vs lex |
|---|---|---|---|---|---|---|---|---|---|---|
| en-de | same | no | 12 | 0.727 | 0.602 | 0.528 | 0.347 | 0.633 | 0.457 | +0.208 |
| en-fr | same | no | 12 | 0.903 | 0.864 | 0.815 | 0.443 | 0.697 | 0.645 | +0.238 |
| en-ar | cross | no | 12 | 0.081 | 0.027 | 0.015 | 0.154 | 0.536 | 0.133 | -0.079 |
| en-zh | cross | no | 12 | 0.948 | 0.867 | 0.839 | 0.462 | 0.682 | 0.196 | +0.711 |
| de-fr | same | no | 12 | 0.557 | 0.627 | 0.453 | 0.435 | 0.706 | 0.365 | +0.227 |
| de-ar | cross | no | 12 | 0.054 | 0.032 | 0.012 | 0.256 | 0.593 | 0.113 | -0.070 |
| de-zh | cross | no | 12 | 0.383 | 0.404 | 0.232 | 0.349 | 0.588 | 0.179 | +0.215 |
| fr-ar | cross | no | 12 | 0.075 | 0.032 | 0.014 | 0.243 | 0.619 | 0.113 | -0.059 |
| fr-zh | cross | no | 12 | 0.651 | 0.600 | 0.449 | 0.400 | 0.624 | 0.177 | +0.448 |
| ar-zh | cross | no | 12 | 0.026 | 0.060 | 0.010 | 0.243 | 0.577 | 0.106 | -0.063 |

### centered, best layer

`lex` is the model-free TF-IDF token-overlap floor; `vs lex` below 0 means the model adds nothing on that pair.

| pair | script | trained | layer | top1 A->B | top1 B->A | mutual-NN | cos margin | CKA | lex | vs lex |
|---|---|---|---|---|---|---|---|---|---|---|
| en-de | same | no | 13 | 0.725 | 0.643 | 0.566 | 0.362 | 0.610 | 0.457 | +0.227 |
| en-fr | same | no | 13 | 0.901 | 0.888 | 0.838 | 0.451 | 0.677 | 0.645 | +0.249 |
| en-ar | cross | no | 0 | 0.065 | 0.058 | 0.047 | 0.045 | 0.125 | 0.133 | -0.071 |
| en-zh | cross | no | 13 | 0.934 | 0.887 | 0.854 | 0.471 | 0.675 | 0.196 | +0.714 |
| de-fr | same | no | 14 | 0.602 | 0.651 | 0.514 | 0.444 | 0.673 | 0.365 | +0.262 |
| de-ar | cross | no | 0 | 0.039 | 0.042 | 0.027 | 0.045 | 0.097 | 0.113 | -0.073 |
| de-zh | cross | no | 14 | 0.437 | 0.444 | 0.305 | 0.362 | 0.589 | 0.179 | +0.262 |
| fr-ar | cross | no | 0 | 0.038 | 0.034 | 0.027 | 0.038 | 0.098 | 0.113 | -0.077 |
| fr-zh | cross | no | 14 | 0.692 | 0.631 | 0.536 | 0.409 | 0.617 | 0.177 | +0.484 |
| ar-zh | cross | no | 14 | 0.038 | 0.060 | 0.017 | 0.220 | 0.502 | 0.106 | -0.056 |

