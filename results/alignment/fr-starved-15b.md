# Alignment: fr-starved-15b (FLORES+ both, n=2009)

Languages embedded: en, de, fr, ar, zh; trained on: fr.

### raw, ref layer

`lex` is the model-free TF-IDF token-overlap floor; `vs lex` below 0 means the model adds nothing on that pair.

| pair | script | trained | layer | top1 A->B | top1 B->A | mutual-NN | cos margin | CKA | lex | vs lex |
|---|---|---|---|---|---|---|---|---|---|---|
| en-de | same | no | 12 | 0.095 | 0.202 | 0.086 | 0.033 | 0.514 | 0.457 | -0.309 |
| en-fr | same | no | 12 | 0.699 | 0.947 | 0.684 | 0.078 | 0.323 | 0.645 | +0.178 |
| en-ar | cross | no | 12 | 0.001 | 0.006 | 0.001 | 0.010 | 0.390 | 0.133 | -0.129 |
| en-zh | cross | no | 12 | 0.019 | 0.193 | 0.015 | 0.030 | 0.577 | 0.196 | -0.090 |
| de-fr | same | no | 12 | 0.056 | 0.066 | 0.031 | 0.029 | 0.192 | 0.365 | -0.304 |
| de-ar | cross | no | 12 | 0.002 | 0.005 | 0.000 | 0.010 | 0.372 | 0.113 | -0.110 |
| de-zh | cross | no | 12 | 0.022 | 0.038 | 0.008 | 0.018 | 0.450 | 0.179 | -0.149 |
| fr-ar | cross | no | 12 | 0.001 | 0.002 | 0.000 | 0.008 | 0.149 | 0.113 | -0.111 |
| fr-zh | cross | no | 12 | 0.011 | 0.091 | 0.006 | 0.029 | 0.226 | 0.177 | -0.126 |
| ar-zh | cross | no | 12 | 0.009 | 0.007 | 0.002 | 0.011 | 0.431 | 0.106 | -0.097 |

### raw, best layer

`lex` is the model-free TF-IDF token-overlap floor; `vs lex` below 0 means the model adds nothing on that pair.

| pair | script | trained | layer | top1 A->B | top1 B->A | mutual-NN | cos margin | CKA | lex | vs lex |
|---|---|---|---|---|---|---|---|---|---|---|
| en-de | same | no | 0 | 0.187 | 0.181 | 0.138 | 0.113 | 0.198 | 0.457 | -0.273 |
| en-fr | same | no | 14 | 0.877 | 0.984 | 0.867 | 0.116 | 0.693 | 0.645 | +0.285 |
| en-ar | cross | no | 0 | 0.025 | 0.020 | 0.015 | 0.028 | 0.127 | 0.133 | -0.111 |
| en-zh | cross | no | 0 | 0.074 | 0.055 | 0.034 | 0.075 | 0.174 | 0.196 | -0.132 |
| de-fr | same | no | 0 | 0.104 | 0.129 | 0.085 | 0.078 | 0.149 | 0.365 | -0.249 |
| de-ar | cross | no | 0 | 0.019 | 0.014 | 0.009 | 0.022 | 0.110 | 0.113 | -0.097 |
| de-zh | cross | no | 0 | 0.043 | 0.033 | 0.021 | 0.050 | 0.173 | 0.179 | -0.140 |
| fr-ar | cross | no | 0 | 0.026 | 0.018 | 0.013 | 0.019 | 0.104 | 0.113 | -0.091 |
| fr-zh | cross | no | 0 | 0.068 | 0.038 | 0.028 | 0.052 | 0.135 | 0.177 | -0.124 |
| ar-zh | cross | no | 0 | 0.016 | 0.013 | 0.006 | 0.026 | 0.132 | 0.106 | -0.091 |

### centered, ref layer

`lex` is the model-free TF-IDF token-overlap floor; `vs lex` below 0 means the model adds nothing on that pair.

| pair | script | trained | layer | top1 A->B | top1 B->A | mutual-NN | cos margin | CKA | lex | vs lex |
|---|---|---|---|---|---|---|---|---|---|---|
| en-de | same | no | 12 | 0.713 | 0.694 | 0.623 | 0.316 | 0.513 | 0.457 | +0.247 |
| en-fr | same | no | 12 | 0.989 | 0.949 | 0.945 | 0.545 | 0.383 | 0.645 | +0.324 |
| en-ar | cross | no | 12 | 0.084 | 0.065 | 0.039 | 0.115 | 0.387 | 0.133 | -0.058 |
| en-zh | cross | no | 12 | 0.788 | 0.721 | 0.662 | 0.330 | 0.579 | 0.196 | +0.559 |
| de-fr | same | no | 12 | 0.562 | 0.471 | 0.413 | 0.228 | 0.229 | 0.365 | +0.152 |
| de-ar | cross | no | 12 | 0.044 | 0.039 | 0.022 | 0.127 | 0.357 | 0.113 | -0.072 |
| de-zh | cross | no | 12 | 0.256 | 0.214 | 0.151 | 0.223 | 0.443 | 0.179 | +0.056 |
| fr-ar | cross | no | 12 | 0.053 | 0.037 | 0.022 | 0.082 | 0.169 | 0.113 | -0.068 |
| fr-zh | cross | no | 12 | 0.694 | 0.685 | 0.577 | 0.268 | 0.269 | 0.177 | +0.512 |
| ar-zh | cross | no | 12 | 0.035 | 0.037 | 0.014 | 0.153 | 0.397 | 0.106 | -0.069 |

### centered, best layer

`lex` is the model-free TF-IDF token-overlap floor; `vs lex` below 0 means the model adds nothing on that pair.

| pair | script | trained | layer | top1 A->B | top1 B->A | mutual-NN | cos margin | CKA | lex | vs lex |
|---|---|---|---|---|---|---|---|---|---|---|
| en-de | same | no | 13 | 0.731 | 0.754 | 0.662 | 0.338 | 0.514 | 0.457 | +0.286 |
| en-fr | same | no | 14 | 0.996 | 0.992 | 0.989 | 0.640 | 0.715 | 0.645 | +0.349 |
| en-ar | cross | no | 13 | 0.089 | 0.067 | 0.040 | 0.115 | 0.333 | 0.133 | -0.055 |
| en-zh | cross | no | 14 | 0.829 | 0.820 | 0.753 | 0.382 | 0.599 | 0.196 | +0.628 |
| de-fr | same | no | 14 | 0.629 | 0.558 | 0.497 | 0.252 | 0.394 | 0.365 | +0.229 |
| de-ar | cross | no | 14 | 0.052 | 0.053 | 0.031 | 0.135 | 0.327 | 0.113 | -0.061 |
| de-zh | cross | no | 14 | 0.337 | 0.301 | 0.215 | 0.258 | 0.474 | 0.179 | +0.140 |
| fr-ar | cross | no | 13 | 0.061 | 0.045 | 0.027 | 0.083 | 0.209 | 0.113 | -0.060 |
| fr-zh | cross | no | 13 | 0.770 | 0.777 | 0.691 | 0.308 | 0.404 | 0.177 | +0.596 |
| ar-zh | cross | no | 13 | 0.048 | 0.049 | 0.025 | 0.153 | 0.361 | 0.106 | -0.057 |

