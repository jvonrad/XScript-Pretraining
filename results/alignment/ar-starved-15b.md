# Alignment: ar-starved-15b (FLORES+ both, n=2009)

Languages embedded: en, de, fr, ar, zh; trained on: ar.

### raw, ref layer

`lex` is the model-free TF-IDF token-overlap floor; `vs lex` below 0 means the model adds nothing on that pair.

| pair | script | trained | layer | top1 A->B | top1 B->A | mutual-NN | cos margin | CKA | lex | vs lex |
|---|---|---|---|---|---|---|---|---|---|---|
| en-de | same | no | 12 | 0.029 | 0.043 | 0.021 | 0.027 | 0.645 | 0.457 | -0.421 |
| en-fr | same | no | 12 | 0.073 | 0.197 | 0.064 | 0.039 | 0.691 | 0.645 | -0.510 |
| en-ar | cross | no | 12 | 0.460 | 0.577 | 0.328 | 0.054 | 0.745 | 0.133 | +0.386 |
| en-zh | cross | no | 12 | 0.006 | 0.008 | 0.003 | 0.018 | 0.546 | 0.196 | -0.189 |
| de-fr | same | no | 12 | 0.039 | 0.057 | 0.025 | 0.026 | 0.685 | 0.365 | -0.317 |
| de-ar | cross | no | 12 | 0.010 | 0.005 | 0.002 | 0.018 | 0.554 | 0.113 | -0.106 |
| de-zh | cross | no | 12 | 0.009 | 0.015 | 0.004 | 0.018 | 0.569 | 0.179 | -0.166 |
| fr-ar | cross | no | 12 | 0.027 | 0.011 | 0.003 | 0.028 | 0.619 | 0.113 | -0.094 |
| fr-zh | cross | no | 12 | 0.010 | 0.015 | 0.004 | 0.019 | 0.550 | 0.177 | -0.165 |
| ar-zh | cross | no | 12 | 0.002 | 0.005 | 0.000 | 0.013 | 0.488 | 0.106 | -0.102 |

### raw, best layer

`lex` is the model-free TF-IDF token-overlap floor; `vs lex` below 0 means the model adds nothing on that pair.

| pair | script | trained | layer | top1 A->B | top1 B->A | mutual-NN | cos margin | CKA | lex | vs lex |
|---|---|---|---|---|---|---|---|---|---|---|
| en-de | same | no | 0 | 0.101 | 0.096 | 0.061 | 0.091 | 0.188 | 0.457 | -0.358 |
| en-fr | same | no | 1 | 0.267 | 0.115 | 0.101 | 0.052 | 0.290 | 0.645 | -0.454 |
| en-ar | cross | no | 13 | 0.580 | 0.608 | 0.419 | 0.067 | 0.731 | 0.133 | +0.461 |
| en-zh | cross | no | 0 | 0.027 | 0.021 | 0.014 | 0.042 | 0.173 | 0.196 | -0.172 |
| de-fr | same | no | 11 | 0.057 | 0.075 | 0.033 | 0.025 | 0.725 | 0.365 | -0.299 |
| de-ar | cross | no | 0 | 0.016 | 0.019 | 0.010 | 0.021 | 0.111 | 0.113 | -0.095 |
| de-zh | cross | no | 0 | 0.015 | 0.018 | 0.009 | 0.031 | 0.149 | 0.179 | -0.162 |
| fr-ar | cross | no | 0 | 0.007 | 0.019 | 0.005 | 0.021 | 0.118 | 0.113 | -0.100 |
| fr-zh | cross | no | 0 | 0.014 | 0.014 | 0.008 | 0.030 | 0.138 | 0.177 | -0.163 |
| ar-zh | cross | no | 0 | 0.010 | 0.005 | 0.003 | 0.015 | 0.119 | 0.106 | -0.098 |

### centered, ref layer

`lex` is the model-free TF-IDF token-overlap floor; `vs lex` below 0 means the model adds nothing on that pair.

| pair | script | trained | layer | top1 A->B | top1 B->A | mutual-NN | cos margin | CKA | lex | vs lex |
|---|---|---|---|---|---|---|---|---|---|---|
| en-de | same | no | 12 | 0.349 | 0.287 | 0.223 | 0.282 | 0.625 | 0.457 | -0.139 |
| en-fr | same | no | 12 | 0.674 | 0.630 | 0.548 | 0.399 | 0.672 | 0.645 | +0.007 |
| en-ar | cross | no | 12 | 0.942 | 0.952 | 0.909 | 0.478 | 0.737 | 0.133 | +0.814 |
| en-zh | cross | no | 12 | 0.091 | 0.047 | 0.031 | 0.179 | 0.532 | 0.196 | -0.127 |
| de-fr | same | no | 12 | 0.198 | 0.219 | 0.146 | 0.289 | 0.644 | 0.365 | -0.157 |
| de-ar | cross | no | 12 | 0.129 | 0.160 | 0.076 | 0.170 | 0.538 | 0.113 | +0.031 |
| de-zh | cross | no | 12 | 0.036 | 0.034 | 0.014 | 0.203 | 0.530 | 0.179 | -0.144 |
| fr-ar | cross | no | 12 | 0.359 | 0.437 | 0.256 | 0.262 | 0.606 | 0.113 | +0.285 |
| fr-zh | cross | no | 12 | 0.054 | 0.034 | 0.020 | 0.201 | 0.517 | 0.177 | -0.133 |
| ar-zh | cross | no | 12 | 0.049 | 0.027 | 0.013 | 0.122 | 0.479 | 0.106 | -0.067 |

### centered, best layer

`lex` is the model-free TF-IDF token-overlap floor; `vs lex` below 0 means the model adds nothing on that pair.

| pair | script | trained | layer | top1 A->B | top1 B->A | mutual-NN | cos margin | CKA | lex | vs lex |
|---|---|---|---|---|---|---|---|---|---|---|
| en-de | same | no | 8 | 0.347 | 0.342 | 0.291 | 0.223 | 0.374 | 0.457 | -0.112 |
| en-fr | same | no | 13 | 0.663 | 0.649 | 0.554 | 0.380 | 0.631 | 0.645 | +0.011 |
| en-ar | cross | no | 13 | 0.943 | 0.948 | 0.909 | 0.499 | 0.726 | 0.133 | +0.812 |
| en-zh | cross | no | 0 | 0.069 | 0.056 | 0.040 | 0.092 | 0.174 | 0.196 | -0.134 |
| de-fr | same | no | 8 | 0.197 | 0.187 | 0.147 | 0.185 | 0.310 | 0.365 | -0.173 |
| de-ar | cross | no | 14 | 0.119 | 0.138 | 0.078 | 0.135 | 0.345 | 0.113 | +0.015 |
| de-zh | cross | no | 0 | 0.043 | 0.029 | 0.020 | 0.072 | 0.140 | 0.179 | -0.143 |
| fr-ar | cross | no | 14 | 0.383 | 0.399 | 0.283 | 0.239 | 0.489 | 0.113 | +0.278 |
| fr-zh | cross | no | 13 | 0.060 | 0.037 | 0.024 | 0.193 | 0.478 | 0.177 | -0.129 |
| ar-zh | cross | no | 1 | 0.028 | 0.023 | 0.018 | 0.031 | 0.144 | 0.106 | -0.080 |

