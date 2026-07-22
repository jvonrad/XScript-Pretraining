# Alignment: zh-fair-12b (FLORES+ both, n=2009)

Languages embedded: en, de, fr, ar, zh; trained on: zh.

### raw, ref layer

`lex` is the model-free TF-IDF token-overlap floor; `vs lex` below 0 means the model adds nothing on that pair.

| pair | script | trained | layer | top1 A->B | top1 B->A | mutual-NN | cos margin | CKA | lex | vs lex |
|---|---|---|---|---|---|---|---|---|---|---|
| en-de | same | no | 12 | 0.086 | 0.057 | 0.038 | 0.038 | 0.574 | 0.434 | -0.363 |
| en-fr | same | no | 12 | 0.092 | 0.069 | 0.046 | 0.040 | 0.588 | 0.580 | -0.500 |
| en-ar | cross | no | 12 | 0.004 | 0.003 | 0.000 | 0.015 | 0.233 | 0.134 | -0.130 |
| en-zh | cross | no | 12 | 0.717 | 0.817 | 0.634 | 0.118 | 0.710 | 0.198 | +0.569 |
| de-fr | same | no | 12 | 0.085 | 0.076 | 0.046 | 0.029 | 0.592 | 0.342 | -0.262 |
| de-ar | cross | no | 12 | 0.003 | 0.001 | 0.000 | 0.016 | 0.244 | 0.114 | -0.112 |
| de-zh | cross | no | 12 | 0.009 | 0.034 | 0.006 | 0.031 | 0.470 | 0.181 | -0.159 |
| fr-ar | cross | no | 12 | 0.005 | 0.002 | 0.000 | 0.014 | 0.260 | 0.111 | -0.108 |
| fr-zh | cross | no | 12 | 0.008 | 0.039 | 0.005 | 0.032 | 0.478 | 0.173 | -0.149 |
| ar-zh | cross | no | 12 | 0.002 | 0.002 | 0.000 | 0.014 | 0.209 | 0.102 | -0.100 |

### raw, best layer

`lex` is the model-free TF-IDF token-overlap floor; `vs lex` below 0 means the model adds nothing on that pair.

| pair | script | trained | layer | top1 A->B | top1 B->A | mutual-NN | cos margin | CKA | lex | vs lex |
|---|---|---|---|---|---|---|---|---|---|---|
| en-de | same | no | 0 | 0.181 | 0.189 | 0.139 | 0.129 | 0.241 | 0.434 | -0.249 |
| en-fr | same | no | 0 | 0.199 | 0.150 | 0.115 | 0.127 | 0.250 | 0.580 | -0.405 |
| en-ar | cross | no | 0 | 0.052 | 0.036 | 0.029 | 0.047 | 0.124 | 0.134 | -0.090 |
| en-zh | cross | no | 11 | 0.713 | 0.825 | 0.642 | 0.109 | 0.704 | 0.198 | +0.571 |
| de-fr | same | no | 0 | 0.137 | 0.108 | 0.087 | 0.091 | 0.145 | 0.342 | -0.220 |
| de-ar | cross | no | 0 | 0.046 | 0.030 | 0.022 | 0.044 | 0.106 | 0.114 | -0.076 |
| de-zh | cross | no | 1 | 0.018 | 0.041 | 0.017 | 0.037 | 0.203 | 0.181 | -0.151 |
| fr-ar | cross | no | 0 | 0.038 | 0.024 | 0.018 | 0.036 | 0.092 | 0.111 | -0.080 |
| fr-zh | cross | no | 0 | 0.027 | 0.036 | 0.016 | 0.045 | 0.116 | 0.173 | -0.141 |
| ar-zh | cross | no | 0 | 0.026 | 0.017 | 0.009 | 0.021 | 0.115 | 0.102 | -0.081 |

### centered, ref layer

`lex` is the model-free TF-IDF token-overlap floor; `vs lex` below 0 means the model adds nothing on that pair.

| pair | script | trained | layer | top1 A->B | top1 B->A | mutual-NN | cos margin | CKA | lex | vs lex |
|---|---|---|---|---|---|---|---|---|---|---|
| en-de | same | no | 12 | 0.370 | 0.305 | 0.248 | 0.285 | 0.555 | 0.434 | -0.097 |
| en-fr | same | no | 12 | 0.504 | 0.451 | 0.375 | 0.317 | 0.557 | 0.580 | -0.103 |
| en-ar | cross | no | 12 | 0.021 | 0.009 | 0.003 | 0.098 | 0.222 | 0.134 | -0.118 |
| en-zh | cross | no | 12 | 0.972 | 0.959 | 0.937 | 0.557 | 0.711 | 0.198 | +0.767 |
| de-fr | same | no | 12 | 0.202 | 0.218 | 0.155 | 0.309 | 0.550 | 0.342 | -0.132 |
| de-ar | cross | no | 12 | 0.017 | 0.009 | 0.004 | 0.140 | 0.225 | 0.114 | -0.101 |
| de-zh | cross | no | 12 | 0.173 | 0.208 | 0.109 | 0.200 | 0.468 | 0.181 | +0.009 |
| fr-ar | cross | no | 12 | 0.015 | 0.007 | 0.003 | 0.135 | 0.241 | 0.111 | -0.100 |
| fr-zh | cross | no | 12 | 0.224 | 0.270 | 0.154 | 0.216 | 0.468 | 0.173 | +0.074 |
| ar-zh | cross | no | 12 | 0.004 | 0.016 | 0.001 | 0.080 | 0.201 | 0.102 | -0.092 |

### centered, best layer

`lex` is the model-free TF-IDF token-overlap floor; `vs lex` below 0 means the model adds nothing on that pair.

| pair | script | trained | layer | top1 A->B | top1 B->A | mutual-NN | cos margin | CKA | lex | vs lex |
|---|---|---|---|---|---|---|---|---|---|---|
| en-de | same | no | 5 | 0.396 | 0.395 | 0.334 | 0.251 | 0.434 | 0.434 | -0.039 |
| en-fr | same | no | 5 | 0.508 | 0.489 | 0.439 | 0.278 | 0.412 | 0.580 | -0.081 |
| en-ar | cross | no | 0 | 0.092 | 0.080 | 0.063 | 0.091 | 0.125 | 0.134 | -0.048 |
| en-zh | cross | no | 12 | 0.972 | 0.959 | 0.937 | 0.557 | 0.711 | 0.198 | +0.767 |
| de-fr | same | no | 5 | 0.217 | 0.222 | 0.178 | 0.203 | 0.311 | 0.342 | -0.123 |
| de-ar | cross | no | 0 | 0.065 | 0.059 | 0.046 | 0.086 | 0.102 | 0.114 | -0.052 |
| de-zh | cross | no | 2 | 0.203 | 0.213 | 0.162 | 0.115 | 0.272 | 0.181 | +0.027 |
| fr-ar | cross | no | 0 | 0.065 | 0.058 | 0.047 | 0.076 | 0.091 | 0.111 | -0.050 |
| fr-zh | cross | no | 5 | 0.259 | 0.249 | 0.178 | 0.153 | 0.336 | 0.173 | +0.081 |
| ar-zh | cross | no | 1 | 0.030 | 0.032 | 0.024 | 0.023 | 0.116 | 0.102 | -0.071 |

