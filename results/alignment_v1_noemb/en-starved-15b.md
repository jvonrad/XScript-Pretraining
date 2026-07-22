# Alignment: en-starved-15b (FLORES+ both, n=2009)

Languages embedded: en, de, fr, ar, zh; trained on: en.

### raw, ref layer

`lex` is the model-free TF-IDF token-overlap floor; `vs lex` below 0 means the model adds nothing on that pair.

| pair | script | trained | layer | top1 A->B | top1 B->A | mutual-NN | cos margin | CKA | lex | vs lex |
|---|---|---|---|---|---|---|---|---|---|---|
| en-de | same | no | 12 | 0.027 | 0.062 | 0.022 | 0.029 | 0.485 | 0.457 | -0.413 |
| en-fr | same | no | 12 | 0.047 | 0.106 | 0.038 | 0.039 | 0.552 | 0.645 | -0.569 |
| en-ar | cross | no | 12 | 0.001 | 0.001 | 0.000 | 0.008 | 0.352 | 0.133 | -0.132 |
| en-zh | cross | no | 12 | 0.014 | 0.121 | 0.009 | 0.036 | 0.554 | 0.196 | -0.129 |
| de-fr | same | no | 12 | 0.131 | 0.134 | 0.077 | 0.029 | 0.643 | 0.365 | -0.232 |
| de-ar | cross | no | 12 | 0.003 | 0.003 | 0.000 | 0.012 | 0.504 | 0.113 | -0.110 |
| de-zh | cross | no | 12 | 0.028 | 0.044 | 0.011 | 0.021 | 0.513 | 0.179 | -0.142 |
| fr-ar | cross | no | 12 | 0.002 | 0.005 | 0.000 | 0.011 | 0.523 | 0.113 | -0.110 |
| fr-zh | cross | no | 12 | 0.035 | 0.063 | 0.012 | 0.024 | 0.537 | 0.177 | -0.128 |
| ar-zh | cross | no | 12 | 0.003 | 0.006 | 0.001 | 0.012 | 0.527 | 0.106 | -0.101 |

### raw, best layer

`lex` is the model-free TF-IDF token-overlap floor; `vs lex` below 0 means the model adds nothing on that pair.

| pair | script | trained | layer | top1 A->B | top1 B->A | mutual-NN | cos margin | CKA | lex | vs lex |
|---|---|---|---|---|---|---|---|---|---|---|
| en-de | same | no | 0 | 0.186 | 0.137 | 0.114 | 0.107 | 0.183 | 0.457 | -0.296 |
| en-fr | same | no | 0 | 0.294 | 0.184 | 0.162 | 0.139 | 0.224 | 0.645 | -0.406 |
| en-ar | cross | no | 0 | 0.024 | 0.019 | 0.013 | 0.024 | 0.111 | 0.133 | -0.111 |
| en-zh | cross | no | 0 | 0.071 | 0.043 | 0.030 | 0.062 | 0.151 | 0.196 | -0.139 |
| de-fr | same | no | 11 | 0.180 | 0.182 | 0.111 | 0.027 | 0.658 | 0.365 | -0.184 |
| de-ar | cross | no | 0 | 0.015 | 0.009 | 0.007 | 0.020 | 0.095 | 0.113 | -0.101 |
| de-zh | cross | no | 0 | 0.033 | 0.028 | 0.019 | 0.041 | 0.167 | 0.179 | -0.148 |
| fr-ar | cross | no | 0 | 0.017 | 0.013 | 0.007 | 0.017 | 0.099 | 0.113 | -0.098 |
| fr-zh | cross | no | 0 | 0.029 | 0.024 | 0.016 | 0.039 | 0.149 | 0.177 | -0.151 |
| ar-zh | cross | no | 0 | 0.016 | 0.010 | 0.004 | 0.020 | 0.115 | 0.106 | -0.093 |

### centered, ref layer

`lex` is the model-free TF-IDF token-overlap floor; `vs lex` below 0 means the model adds nothing on that pair.

| pair | script | trained | layer | top1 A->B | top1 B->A | mutual-NN | cos margin | CKA | lex | vs lex |
|---|---|---|---|---|---|---|---|---|---|---|
| en-de | same | no | 12 | 0.597 | 0.544 | 0.456 | 0.258 | 0.496 | 0.457 | +0.114 |
| en-fr | same | no | 12 | 0.792 | 0.785 | 0.707 | 0.333 | 0.558 | 0.645 | +0.143 |
| en-ar | cross | no | 12 | 0.048 | 0.023 | 0.012 | 0.088 | 0.362 | 0.133 | -0.098 |
| en-zh | cross | no | 12 | 0.815 | 0.735 | 0.671 | 0.326 | 0.568 | 0.196 | +0.579 |
| de-fr | same | no | 12 | 0.414 | 0.448 | 0.326 | 0.338 | 0.613 | 0.365 | +0.066 |
| de-ar | cross | no | 12 | 0.034 | 0.026 | 0.011 | 0.173 | 0.479 | 0.113 | -0.083 |
| de-zh | cross | no | 12 | 0.220 | 0.222 | 0.121 | 0.259 | 0.503 | 0.179 | +0.043 |
| fr-ar | cross | no | 12 | 0.036 | 0.025 | 0.009 | 0.160 | 0.495 | 0.113 | -0.083 |
| fr-zh | cross | no | 12 | 0.334 | 0.306 | 0.190 | 0.284 | 0.525 | 0.177 | +0.143 |
| ar-zh | cross | no | 12 | 0.023 | 0.032 | 0.008 | 0.176 | 0.483 | 0.106 | -0.078 |

### centered, best layer

`lex` is the model-free TF-IDF token-overlap floor; `vs lex` below 0 means the model adds nothing on that pair.

| pair | script | trained | layer | top1 A->B | top1 B->A | mutual-NN | cos margin | CKA | lex | vs lex |
|---|---|---|---|---|---|---|---|---|---|---|
| en-de | same | no | 7 | 0.574 | 0.566 | 0.501 | 0.231 | 0.457 | 0.457 | +0.113 |
| en-fr | same | no | 12 | 0.792 | 0.785 | 0.707 | 0.333 | 0.558 | 0.645 | +0.143 |
| en-ar | cross | no | 0 | 0.051 | 0.045 | 0.034 | 0.041 | 0.113 | 0.133 | -0.085 |
| en-zh | cross | no | 12 | 0.815 | 0.735 | 0.671 | 0.326 | 0.568 | 0.196 | +0.579 |
| de-fr | same | no | 14 | 0.429 | 0.455 | 0.348 | 0.352 | 0.570 | 0.365 | +0.077 |
| de-ar | cross | no | 0 | 0.030 | 0.030 | 0.020 | 0.041 | 0.094 | 0.113 | -0.083 |
| de-zh | cross | no | 14 | 0.265 | 0.248 | 0.158 | 0.268 | 0.474 | 0.179 | +0.078 |
| fr-ar | cross | no | 0 | 0.031 | 0.027 | 0.021 | 0.035 | 0.098 | 0.113 | -0.084 |
| fr-zh | cross | no | 14 | 0.373 | 0.372 | 0.252 | 0.291 | 0.495 | 0.177 | +0.195 |
| ar-zh | cross | no | 0 | 0.021 | 0.019 | 0.011 | 0.047 | 0.113 | 0.106 | -0.086 |

