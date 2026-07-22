# Alignment: zh-starved-12b (FLORES+ both, n=2009)

Languages embedded: en, de, fr, ar, zh; trained on: zh.

### raw, ref layer

`lex` is the model-free TF-IDF token-overlap floor; `vs lex` below 0 means the model adds nothing on that pair.

| pair | script | trained | layer | top1 A->B | top1 B->A | mutual-NN | cos margin | CKA | lex | vs lex |
|---|---|---|---|---|---|---|---|---|---|---|
| en-de | same | no | 12 | 0.028 | 0.031 | 0.015 | 0.019 | 0.413 | 0.457 | -0.427 |
| en-fr | same | no | 12 | 0.024 | 0.030 | 0.013 | 0.022 | 0.449 | 0.645 | -0.618 |
| en-ar | cross | no | 12 | 0.001 | 0.000 | 0.000 | 0.006 | 0.295 | 0.133 | -0.132 |
| en-zh | cross | no | 12 | 0.595 | 0.232 | 0.191 | 0.052 | 0.635 | 0.196 | +0.218 |
| de-fr | same | no | 12 | 0.047 | 0.041 | 0.021 | 0.016 | 0.479 | 0.365 | -0.321 |
| de-ar | cross | no | 12 | 0.002 | 0.003 | 0.001 | 0.007 | 0.365 | 0.113 | -0.111 |
| de-zh | cross | no | 12 | 0.004 | 0.004 | 0.002 | 0.011 | 0.268 | 0.179 | -0.174 |
| fr-ar | cross | no | 12 | 0.002 | 0.002 | 0.000 | 0.007 | 0.381 | 0.113 | -0.111 |
| fr-zh | cross | no | 12 | 0.004 | 0.003 | 0.002 | 0.013 | 0.299 | 0.177 | -0.174 |
| ar-zh | cross | no | 12 | 0.001 | 0.000 | 0.000 | 0.004 | 0.181 | 0.106 | -0.105 |

### raw, best layer

`lex` is the model-free TF-IDF token-overlap floor; `vs lex` below 0 means the model adds nothing on that pair.

| pair | script | trained | layer | top1 A->B | top1 B->A | mutual-NN | cos margin | CKA | lex | vs lex |
|---|---|---|---|---|---|---|---|---|---|---|
| en-de | same | no | 0 | 0.120 | 0.146 | 0.089 | 0.093 | 0.202 | 0.457 | -0.324 |
| en-fr | same | no | 0 | 0.178 | 0.133 | 0.096 | 0.110 | 0.243 | 0.645 | -0.490 |
| en-ar | cross | no | 0 | 0.016 | 0.023 | 0.012 | 0.030 | 0.108 | 0.133 | -0.114 |
| en-zh | cross | no | 13 | 0.620 | 0.341 | 0.278 | 0.068 | 0.655 | 0.196 | +0.285 |
| de-fr | same | no | 8 | 0.087 | 0.088 | 0.053 | 0.011 | 0.621 | 0.365 | -0.277 |
| de-ar | cross | no | 0 | 0.018 | 0.016 | 0.011 | 0.026 | 0.102 | 0.113 | -0.096 |
| de-zh | cross | no | 0 | 0.035 | 0.019 | 0.011 | 0.026 | 0.151 | 0.179 | -0.152 |
| fr-ar | cross | no | 0 | 0.013 | 0.017 | 0.008 | 0.022 | 0.088 | 0.113 | -0.098 |
| fr-zh | cross | no | 0 | 0.021 | 0.015 | 0.008 | 0.026 | 0.133 | 0.177 | -0.160 |
| ar-zh | cross | no | 0 | 0.011 | 0.006 | 0.004 | 0.012 | 0.103 | 0.106 | -0.097 |

### centered, ref layer

`lex` is the model-free TF-IDF token-overlap floor; `vs lex` below 0 means the model adds nothing on that pair.

| pair | script | trained | layer | top1 A->B | top1 B->A | mutual-NN | cos margin | CKA | lex | vs lex |
|---|---|---|---|---|---|---|---|---|---|---|
| en-de | same | no | 12 | 0.379 | 0.326 | 0.271 | 0.220 | 0.416 | 0.457 | -0.105 |
| en-fr | same | no | 12 | 0.538 | 0.495 | 0.420 | 0.273 | 0.448 | 0.645 | -0.129 |
| en-ar | cross | no | 12 | 0.028 | 0.015 | 0.010 | 0.074 | 0.291 | 0.133 | -0.111 |
| en-zh | cross | no | 12 | 0.969 | 0.937 | 0.921 | 0.437 | 0.635 | 0.196 | +0.757 |
| de-fr | same | no | 12 | 0.188 | 0.190 | 0.136 | 0.237 | 0.438 | 0.365 | -0.176 |
| de-ar | cross | no | 12 | 0.012 | 0.011 | 0.003 | 0.112 | 0.336 | 0.113 | -0.101 |
| de-zh | cross | no | 12 | 0.144 | 0.175 | 0.097 | 0.119 | 0.288 | 0.179 | -0.019 |
| fr-ar | cross | no | 12 | 0.020 | 0.013 | 0.006 | 0.116 | 0.351 | 0.113 | -0.097 |
| fr-zh | cross | no | 12 | 0.205 | 0.252 | 0.142 | 0.145 | 0.313 | 0.177 | +0.051 |
| ar-zh | cross | no | 12 | 0.009 | 0.017 | 0.003 | 0.044 | 0.196 | 0.106 | -0.093 |

### centered, best layer

`lex` is the model-free TF-IDF token-overlap floor; `vs lex` below 0 means the model adds nothing on that pair.

| pair | script | trained | layer | top1 A->B | top1 B->A | mutual-NN | cos margin | CKA | lex | vs lex |
|---|---|---|---|---|---|---|---|---|---|---|
| en-de | same | no | 10 | 0.386 | 0.339 | 0.280 | 0.257 | 0.548 | 0.457 | -0.095 |
| en-fr | same | no | 13 | 0.523 | 0.508 | 0.429 | 0.275 | 0.421 | 0.645 | -0.130 |
| en-ar | cross | no | 0 | 0.064 | 0.055 | 0.045 | 0.069 | 0.108 | 0.133 | -0.073 |
| en-zh | cross | no | 12 | 0.969 | 0.937 | 0.921 | 0.437 | 0.635 | 0.196 | +0.757 |
| de-fr | same | no | 13 | 0.197 | 0.206 | 0.152 | 0.242 | 0.433 | 0.365 | -0.163 |
| de-ar | cross | no | 0 | 0.044 | 0.043 | 0.031 | 0.062 | 0.098 | 0.113 | -0.070 |
| de-zh | cross | no | 1 | 0.144 | 0.160 | 0.121 | 0.073 | 0.219 | 0.179 | -0.027 |
| fr-ar | cross | no | 0 | 0.043 | 0.037 | 0.027 | 0.058 | 0.087 | 0.113 | -0.073 |
| fr-zh | cross | no | 13 | 0.216 | 0.241 | 0.148 | 0.155 | 0.298 | 0.177 | +0.051 |
| ar-zh | cross | no | 0 | 0.028 | 0.029 | 0.013 | 0.028 | 0.102 | 0.106 | -0.077 |

