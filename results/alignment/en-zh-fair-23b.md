# Alignment: en-zh-fair-23b (FLORES+ both, n=2009)

Languages embedded: en, de, fr, ar, zh; trained on: en, zh.

### raw, ref layer

`lex` is the model-free TF-IDF token-overlap floor; `vs lex` below 0 means the model adds nothing on that pair.

| pair | script | trained | layer | top1 A->B | top1 B->A | mutual-NN | cos margin | CKA | lex | vs lex |
|---|---|---|---|---|---|---|---|---|---|---|
| en-de | same | no | 12 | 0.080 | 0.133 | 0.056 | 0.066 | 0.667 | 0.434 | -0.328 |
| en-fr | same | no | 12 | 0.101 | 0.208 | 0.075 | 0.073 | 0.692 | 0.580 | -0.426 |
| en-ar | cross | no | 12 | 0.004 | 0.004 | 0.000 | 0.023 | 0.392 | 0.134 | -0.130 |
| en-zh | cross | yes | 12 | 0.968 | 0.976 | 0.953 | 0.161 | 0.742 | 0.198 | +0.774 |
| de-fr | same | no | 12 | 0.253 | 0.257 | 0.157 | 0.060 | 0.725 | 0.342 | -0.088 |
| de-ar | cross | no | 12 | 0.010 | 0.007 | 0.001 | 0.029 | 0.435 | 0.114 | -0.106 |
| de-zh | cross | no | 12 | 0.078 | 0.056 | 0.027 | 0.059 | 0.589 | 0.181 | -0.114 |
| fr-ar | cross | no | 12 | 0.011 | 0.009 | 0.002 | 0.028 | 0.452 | 0.111 | -0.101 |
| fr-zh | cross | no | 12 | 0.127 | 0.067 | 0.033 | 0.063 | 0.609 | 0.173 | -0.075 |
| ar-zh | cross | no | 12 | 0.002 | 0.004 | 0.000 | 0.025 | 0.406 | 0.102 | -0.099 |

### raw, best layer

`lex` is the model-free TF-IDF token-overlap floor; `vs lex` below 0 means the model adds nothing on that pair.

| pair | script | trained | layer | top1 A->B | top1 B->A | mutual-NN | cos margin | CKA | lex | vs lex |
|---|---|---|---|---|---|---|---|---|---|---|
| en-de | same | no | 0 | 0.213 | 0.219 | 0.164 | 0.141 | 0.225 | 0.434 | -0.219 |
| en-fr | same | no | 1 | 0.219 | 0.271 | 0.182 | 0.082 | 0.304 | 0.580 | -0.335 |
| en-ar | cross | no | 0 | 0.024 | 0.035 | 0.019 | 0.038 | 0.146 | 0.134 | -0.104 |
| en-zh | cross | yes | 13 | 0.971 | 0.972 | 0.954 | 0.172 | 0.762 | 0.198 | +0.773 |
| de-fr | same | no | 10 | 0.259 | 0.290 | 0.177 | 0.059 | 0.754 | 0.342 | -0.068 |
| de-ar | cross | no | 0 | 0.035 | 0.031 | 0.019 | 0.039 | 0.117 | 0.114 | -0.081 |
| de-zh | cross | no | 0 | 0.072 | 0.043 | 0.029 | 0.067 | 0.186 | 0.181 | -0.123 |
| fr-ar | cross | no | 0 | 0.036 | 0.032 | 0.021 | 0.031 | 0.112 | 0.111 | -0.077 |
| fr-zh | cross | no | 12 | 0.127 | 0.067 | 0.033 | 0.063 | 0.609 | 0.173 | -0.075 |
| ar-zh | cross | no | 0 | 0.018 | 0.007 | 0.005 | 0.026 | 0.136 | 0.102 | -0.090 |

### centered, ref layer

`lex` is the model-free TF-IDF token-overlap floor; `vs lex` below 0 means the model adds nothing on that pair.

| pair | script | trained | layer | top1 A->B | top1 B->A | mutual-NN | cos margin | CKA | lex | vs lex |
|---|---|---|---|---|---|---|---|---|---|---|
| en-de | same | no | 12 | 0.649 | 0.540 | 0.446 | 0.368 | 0.647 | 0.434 | +0.160 |
| en-fr | same | no | 12 | 0.777 | 0.696 | 0.617 | 0.413 | 0.676 | 0.580 | +0.156 |
| en-ar | cross | no | 12 | 0.034 | 0.010 | 0.005 | 0.139 | 0.392 | 0.134 | -0.111 |
| en-zh | cross | yes | 12 | 0.986 | 0.980 | 0.968 | 0.676 | 0.747 | 0.198 | +0.784 |
| de-fr | same | no | 12 | 0.454 | 0.486 | 0.346 | 0.454 | 0.690 | 0.342 | +0.128 |
| de-ar | cross | no | 12 | 0.031 | 0.016 | 0.007 | 0.235 | 0.428 | 0.114 | -0.091 |
| de-zh | cross | no | 12 | 0.400 | 0.464 | 0.279 | 0.328 | 0.588 | 0.181 | +0.251 |
| fr-ar | cross | no | 12 | 0.032 | 0.017 | 0.008 | 0.224 | 0.442 | 0.111 | -0.086 |
| fr-zh | cross | no | 12 | 0.531 | 0.567 | 0.388 | 0.358 | 0.612 | 0.173 | +0.376 |
| ar-zh | cross | no | 12 | 0.012 | 0.026 | 0.007 | 0.151 | 0.407 | 0.102 | -0.083 |

### centered, best layer

`lex` is the model-free TF-IDF token-overlap floor; `vs lex` below 0 means the model adds nothing on that pair.

| pair | script | trained | layer | top1 A->B | top1 B->A | mutual-NN | cos margin | CKA | lex | vs lex |
|---|---|---|---|---|---|---|---|---|---|---|
| en-de | same | no | 4 | 0.580 | 0.601 | 0.509 | 0.273 | 0.441 | 0.434 | +0.156 |
| en-fr | same | no | 12 | 0.777 | 0.696 | 0.617 | 0.413 | 0.676 | 0.580 | +0.156 |
| en-ar | cross | no | 0 | 0.078 | 0.071 | 0.052 | 0.063 | 0.146 | 0.134 | -0.059 |
| en-zh | cross | yes | 13 | 0.986 | 0.982 | 0.972 | 0.676 | 0.767 | 0.198 | +0.785 |
| de-fr | same | no | 15 | 0.443 | 0.481 | 0.358 | 0.376 | 0.517 | 0.342 | +0.119 |
| de-ar | cross | no | 0 | 0.055 | 0.053 | 0.037 | 0.070 | 0.113 | 0.114 | -0.060 |
| de-zh | cross | no | 15 | 0.444 | 0.406 | 0.311 | 0.252 | 0.436 | 0.181 | +0.244 |
| fr-ar | cross | no | 0 | 0.066 | 0.061 | 0.050 | 0.061 | 0.110 | 0.111 | -0.048 |
| fr-zh | cross | no | 15 | 0.583 | 0.483 | 0.400 | 0.275 | 0.480 | 0.173 | +0.361 |
| ar-zh | cross | no | 1 | 0.031 | 0.032 | 0.024 | 0.033 | 0.150 | 0.102 | -0.071 |

