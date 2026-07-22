# Alignment: ar-fair (FLORES+ both, n=2009)

Languages embedded: en, de, fr, ar, zh; trained on: ar.

### raw, ref layer

`lex` is the model-free TF-IDF token-overlap floor; `vs lex` below 0 means the model adds nothing on that pair.

| pair | script | trained | layer | top1 A->B | top1 B->A | mutual-NN | cos margin | CKA | lex | vs lex |
|---|---|---|---|---|---|---|---|---|---|---|
| en-de | same | no | 12 | 0.044 | 0.062 | 0.029 | 0.056 | 0.754 | 0.434 | -0.381 |
| en-fr | same | no | 12 | 0.388 | 0.737 | 0.371 | 0.093 | 0.800 | 0.580 | -0.018 |
| en-ar | cross | no | 12 | 0.937 | 0.938 | 0.887 | 0.127 | 0.776 | 0.134 | +0.803 |
| en-zh | cross | no | 12 | 0.015 | 0.048 | 0.007 | 0.049 | 0.675 | 0.198 | -0.167 |
| de-fr | same | no | 12 | 0.050 | 0.065 | 0.028 | 0.053 | 0.760 | 0.342 | -0.285 |
| de-ar | cross | no | 12 | 0.062 | 0.023 | 0.014 | 0.049 | 0.667 | 0.114 | -0.072 |
| de-zh | cross | no | 12 | 0.013 | 0.018 | 0.006 | 0.041 | 0.618 | 0.181 | -0.166 |
| fr-ar | cross | no | 12 | 0.542 | 0.168 | 0.129 | 0.086 | 0.730 | 0.111 | +0.244 |
| fr-zh | cross | no | 12 | 0.015 | 0.018 | 0.003 | 0.045 | 0.628 | 0.173 | -0.156 |
| ar-zh | cross | no | 12 | 0.011 | 0.028 | 0.005 | 0.046 | 0.627 | 0.102 | -0.082 |

### raw, best layer

`lex` is the model-free TF-IDF token-overlap floor; `vs lex` below 0 means the model adds nothing on that pair.

| pair | script | trained | layer | top1 A->B | top1 B->A | mutual-NN | cos margin | CKA | lex | vs lex |
|---|---|---|---|---|---|---|---|---|---|---|
| en-de | same | no | 0 | 0.193 | 0.183 | 0.140 | 0.138 | 0.224 | 0.434 | -0.246 |
| en-fr | same | no | 11 | 0.459 | 0.810 | 0.443 | 0.087 | 0.808 | 0.580 | +0.055 |
| en-ar | cross | no | 13 | 0.957 | 0.967 | 0.933 | 0.155 | 0.768 | 0.134 | +0.828 |
| en-zh | cross | no | 0 | 0.059 | 0.039 | 0.027 | 0.069 | 0.141 | 0.198 | -0.150 |
| de-fr | same | no | 0 | 0.136 | 0.115 | 0.087 | 0.096 | 0.156 | 0.342 | -0.217 |
| de-ar | cross | no | 0 | 0.044 | 0.059 | 0.029 | 0.045 | 0.130 | 0.114 | -0.063 |
| de-zh | cross | no | 0 | 0.064 | 0.040 | 0.027 | 0.060 | 0.137 | 0.181 | -0.129 |
| fr-ar | cross | no | 12 | 0.542 | 0.168 | 0.129 | 0.086 | 0.730 | 0.111 | +0.244 |
| fr-zh | cross | no | 0 | 0.037 | 0.030 | 0.015 | 0.051 | 0.120 | 0.173 | -0.139 |
| ar-zh | cross | no | 0 | 0.040 | 0.012 | 0.008 | 0.033 | 0.126 | 0.102 | -0.076 |

### centered, ref layer

`lex` is the model-free TF-IDF token-overlap floor; `vs lex` below 0 means the model adds nothing on that pair.

| pair | script | trained | layer | top1 A->B | top1 B->A | mutual-NN | cos margin | CKA | lex | vs lex |
|---|---|---|---|---|---|---|---|---|---|---|
| en-de | same | no | 12 | 0.500 | 0.340 | 0.262 | 0.406 | 0.728 | 0.434 | -0.014 |
| en-fr | same | no | 12 | 0.918 | 0.900 | 0.843 | 0.599 | 0.783 | 0.580 | +0.329 |
| en-ar | cross | no | 12 | 0.958 | 0.968 | 0.931 | 0.655 | 0.773 | 0.134 | +0.829 |
| en-zh | cross | no | 12 | 0.236 | 0.107 | 0.056 | 0.331 | 0.642 | 0.198 | -0.027 |
| de-fr | same | no | 12 | 0.297 | 0.402 | 0.216 | 0.424 | 0.719 | 0.342 | +0.007 |
| de-ar | cross | no | 12 | 0.191 | 0.332 | 0.122 | 0.315 | 0.649 | 0.114 | +0.147 |
| de-zh | cross | no | 12 | 0.094 | 0.065 | 0.024 | 0.354 | 0.595 | 0.181 | -0.102 |
| fr-ar | cross | no | 12 | 0.777 | 0.827 | 0.671 | 0.491 | 0.723 | 0.111 | +0.691 |
| fr-zh | cross | no | 12 | 0.172 | 0.086 | 0.035 | 0.339 | 0.600 | 0.173 | -0.044 |
| ar-zh | cross | no | 12 | 0.187 | 0.077 | 0.034 | 0.278 | 0.598 | 0.102 | +0.030 |

### centered, best layer

`lex` is the model-free TF-IDF token-overlap floor; `vs lex` below 0 means the model adds nothing on that pair.

| pair | script | trained | layer | top1 A->B | top1 B->A | mutual-NN | cos margin | CKA | lex | vs lex |
|---|---|---|---|---|---|---|---|---|---|---|
| en-de | same | no | 4 | 0.447 | 0.455 | 0.392 | 0.263 | 0.435 | 0.434 | +0.017 |
| en-fr | same | no | 13 | 0.930 | 0.929 | 0.881 | 0.595 | 0.759 | 0.580 | +0.349 |
| en-ar | cross | no | 14 | 0.979 | 0.972 | 0.958 | 0.673 | 0.775 | 0.134 | +0.842 |
| en-zh | cross | no | 14 | 0.233 | 0.148 | 0.093 | 0.286 | 0.544 | 0.198 | -0.008 |
| de-fr | same | no | 14 | 0.341 | 0.416 | 0.273 | 0.385 | 0.613 | 0.342 | +0.036 |
| de-ar | cross | no | 14 | 0.268 | 0.322 | 0.185 | 0.280 | 0.531 | 0.114 | +0.180 |
| de-zh | cross | no | 0 | 0.086 | 0.073 | 0.052 | 0.107 | 0.129 | 0.181 | -0.102 |
| fr-ar | cross | no | 14 | 0.855 | 0.817 | 0.756 | 0.474 | 0.667 | 0.111 | +0.725 |
| fr-zh | cross | no | 14 | 0.181 | 0.110 | 0.064 | 0.296 | 0.525 | 0.173 | -0.027 |
| ar-zh | cross | no | 14 | 0.190 | 0.112 | 0.068 | 0.233 | 0.495 | 0.102 | +0.049 |

