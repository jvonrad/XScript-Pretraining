# Alignment: ar-fair-15b (FLORES+ both, n=2009)

Languages embedded: en, de, fr, ar, zh; trained on: ar.

### raw, ref layer

`lex` is the model-free TF-IDF token-overlap floor; `vs lex` below 0 means the model adds nothing on that pair.

| pair | script | trained | layer | top1 A->B | top1 B->A | mutual-NN | cos margin | CKA | lex | vs lex |
|---|---|---|---|---|---|---|---|---|---|---|
| en-de | same | no | 12 | 0.056 | 0.044 | 0.023 | 0.036 | 0.665 | 0.434 | -0.385 |
| en-fr | same | no | 12 | 0.306 | 0.399 | 0.222 | 0.063 | 0.738 | 0.580 | -0.228 |
| en-ar | cross | no | 12 | 0.952 | 0.930 | 0.895 | 0.101 | 0.766 | 0.134 | +0.807 |
| en-zh | cross | no | 12 | 0.014 | 0.012 | 0.006 | 0.026 | 0.593 | 0.198 | -0.185 |
| de-fr | same | no | 12 | 0.042 | 0.072 | 0.023 | 0.032 | 0.682 | 0.342 | -0.285 |
| de-ar | cross | no | 12 | 0.034 | 0.022 | 0.012 | 0.029 | 0.586 | 0.114 | -0.086 |
| de-zh | cross | no | 12 | 0.012 | 0.017 | 0.008 | 0.022 | 0.545 | 0.181 | -0.166 |
| fr-ar | cross | no | 12 | 0.251 | 0.088 | 0.053 | 0.053 | 0.672 | 0.111 | +0.058 |
| fr-zh | cross | no | 12 | 0.016 | 0.013 | 0.004 | 0.023 | 0.557 | 0.173 | -0.158 |
| ar-zh | cross | no | 12 | 0.009 | 0.017 | 0.003 | 0.023 | 0.569 | 0.102 | -0.089 |

### raw, best layer

`lex` is the model-free TF-IDF token-overlap floor; `vs lex` below 0 means the model adds nothing on that pair.

| pair | script | trained | layer | top1 A->B | top1 B->A | mutual-NN | cos margin | CKA | lex | vs lex |
|---|---|---|---|---|---|---|---|---|---|---|
| en-de | same | no | 1 | 0.200 | 0.143 | 0.129 | 0.088 | 0.278 | 0.434 | -0.263 |
| en-fr | same | no | 11 | 0.449 | 0.541 | 0.366 | 0.063 | 0.760 | 0.580 | -0.085 |
| en-ar | cross | no | 13 | 0.958 | 0.928 | 0.899 | 0.121 | 0.753 | 0.134 | +0.809 |
| en-zh | cross | no | 0 | 0.059 | 0.037 | 0.025 | 0.065 | 0.143 | 0.198 | -0.150 |
| de-fr | same | no | 10 | 0.120 | 0.211 | 0.094 | 0.036 | 0.731 | 0.342 | -0.177 |
| de-ar | cross | no | 0 | 0.037 | 0.046 | 0.024 | 0.043 | 0.120 | 0.114 | -0.073 |
| de-zh | cross | no | 0 | 0.049 | 0.035 | 0.020 | 0.057 | 0.130 | 0.181 | -0.139 |
| fr-ar | cross | no | 11 | 0.344 | 0.104 | 0.068 | 0.052 | 0.690 | 0.111 | +0.113 |
| fr-zh | cross | no | 0 | 0.033 | 0.027 | 0.016 | 0.049 | 0.118 | 0.173 | -0.143 |
| ar-zh | cross | no | 1 | 0.010 | 0.015 | 0.007 | 0.018 | 0.176 | 0.102 | -0.090 |

### centered, ref layer

`lex` is the model-free TF-IDF token-overlap floor; `vs lex` below 0 means the model adds nothing on that pair.

| pair | script | trained | layer | top1 A->B | top1 B->A | mutual-NN | cos margin | CKA | lex | vs lex |
|---|---|---|---|---|---|---|---|---|---|---|
| en-de | same | no | 12 | 0.420 | 0.295 | 0.245 | 0.323 | 0.641 | 0.434 | -0.077 |
| en-fr | same | no | 12 | 0.883 | 0.853 | 0.792 | 0.499 | 0.721 | 0.580 | +0.288 |
| en-ar | cross | no | 12 | 0.975 | 0.976 | 0.956 | 0.597 | 0.765 | 0.134 | +0.842 |
| en-zh | cross | no | 12 | 0.111 | 0.052 | 0.034 | 0.224 | 0.565 | 0.198 | -0.117 |
| de-fr | same | no | 12 | 0.232 | 0.301 | 0.180 | 0.341 | 0.640 | 0.342 | -0.076 |
| de-ar | cross | no | 12 | 0.160 | 0.256 | 0.111 | 0.232 | 0.571 | 0.114 | +0.094 |
| de-zh | cross | no | 12 | 0.061 | 0.041 | 0.021 | 0.268 | 0.529 | 0.181 | -0.130 |
| fr-ar | cross | no | 12 | 0.726 | 0.747 | 0.595 | 0.385 | 0.664 | 0.111 | +0.625 |
| fr-zh | cross | no | 12 | 0.084 | 0.041 | 0.024 | 0.242 | 0.524 | 0.173 | -0.110 |
| ar-zh | cross | no | 12 | 0.075 | 0.027 | 0.013 | 0.182 | 0.543 | 0.102 | -0.051 |

### centered, best layer

`lex` is the model-free TF-IDF token-overlap floor; `vs lex` below 0 means the model adds nothing on that pair.

| pair | script | trained | layer | top1 A->B | top1 B->A | mutual-NN | cos margin | CKA | lex | vs lex |
|---|---|---|---|---|---|---|---|---|---|---|
| en-de | same | no | 7 | 0.462 | 0.444 | 0.381 | 0.305 | 0.575 | 0.434 | +0.019 |
| en-fr | same | no | 13 | 0.873 | 0.870 | 0.812 | 0.490 | 0.674 | 0.580 | +0.291 |
| en-ar | cross | no | 13 | 0.984 | 0.978 | 0.967 | 0.612 | 0.756 | 0.134 | +0.847 |
| en-zh | cross | no | 0 | 0.103 | 0.086 | 0.062 | 0.104 | 0.141 | 0.198 | -0.104 |
| de-fr | same | no | 7 | 0.279 | 0.300 | 0.227 | 0.267 | 0.488 | 0.342 | -0.053 |
| de-ar | cross | no | 14 | 0.202 | 0.239 | 0.145 | 0.204 | 0.417 | 0.114 | +0.106 |
| de-zh | cross | no | 0 | 0.083 | 0.068 | 0.046 | 0.101 | 0.121 | 0.181 | -0.106 |
| fr-ar | cross | no | 13 | 0.758 | 0.755 | 0.648 | 0.386 | 0.618 | 0.111 | +0.645 |
| fr-zh | cross | no | 0 | 0.073 | 0.063 | 0.046 | 0.090 | 0.113 | 0.173 | -0.105 |
| ar-zh | cross | no | 1 | 0.033 | 0.034 | 0.026 | 0.042 | 0.177 | 0.102 | -0.069 |

