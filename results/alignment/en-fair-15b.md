# Alignment: en-fair-15b (FLORES+ both, n=2009)

Languages embedded: en, de, fr, ar, zh; trained on: en.

### raw, ref layer

`lex` is the model-free TF-IDF token-overlap floor; `vs lex` below 0 means the model adds nothing on that pair.

| pair | script | trained | layer | top1 A->B | top1 B->A | mutual-NN | cos margin | CKA | lex | vs lex |
|---|---|---|---|---|---|---|---|---|---|---|
| en-de | same | no | 12 | 0.168 | 0.206 | 0.119 | 0.073 | 0.664 | 0.434 | -0.247 |
| en-fr | same | no | 12 | 0.237 | 0.417 | 0.192 | 0.083 | 0.697 | 0.580 | -0.253 |
| en-ar | cross | no | 12 | 0.004 | 0.005 | 0.000 | 0.022 | 0.369 | 0.134 | -0.129 |
| en-zh | cross | no | 12 | 0.128 | 0.215 | 0.066 | 0.076 | 0.619 | 0.198 | -0.027 |
| de-fr | same | no | 12 | 0.322 | 0.379 | 0.237 | 0.062 | 0.701 | 0.342 | +0.008 |
| de-ar | cross | no | 12 | 0.012 | 0.006 | 0.003 | 0.024 | 0.399 | 0.114 | -0.105 |
| de-zh | cross | no | 12 | 0.092 | 0.120 | 0.036 | 0.050 | 0.597 | 0.181 | -0.075 |
| fr-ar | cross | no | 12 | 0.015 | 0.007 | 0.004 | 0.024 | 0.397 | 0.111 | -0.100 |
| fr-zh | cross | no | 12 | 0.121 | 0.113 | 0.034 | 0.052 | 0.594 | 0.173 | -0.056 |
| ar-zh | cross | no | 12 | 0.003 | 0.019 | 0.001 | 0.025 | 0.402 | 0.102 | -0.091 |

### raw, best layer

`lex` is the model-free TF-IDF token-overlap floor; `vs lex` below 0 means the model adds nothing on that pair.

| pair | script | trained | layer | top1 A->B | top1 B->A | mutual-NN | cos margin | CKA | lex | vs lex |
|---|---|---|---|---|---|---|---|---|---|---|
| en-de | same | no | 0 | 0.279 | 0.241 | 0.206 | 0.156 | 0.207 | 0.434 | -0.174 |
| en-fr | same | no | 11 | 0.287 | 0.452 | 0.235 | 0.079 | 0.724 | 0.580 | -0.210 |
| en-ar | cross | no | 0 | 0.043 | 0.038 | 0.025 | 0.041 | 0.123 | 0.134 | -0.093 |
| en-zh | cross | no | 10 | 0.198 | 0.272 | 0.099 | 0.065 | 0.636 | 0.198 | +0.037 |
| de-fr | same | no | 10 | 0.336 | 0.359 | 0.251 | 0.058 | 0.727 | 0.342 | +0.005 |
| de-ar | cross | no | 0 | 0.035 | 0.030 | 0.018 | 0.040 | 0.107 | 0.114 | -0.082 |
| de-zh | cross | no | 13 | 0.106 | 0.113 | 0.041 | 0.050 | 0.574 | 0.181 | -0.072 |
| fr-ar | cross | no | 0 | 0.037 | 0.022 | 0.017 | 0.032 | 0.100 | 0.111 | -0.081 |
| fr-zh | cross | no | 11 | 0.152 | 0.129 | 0.048 | 0.051 | 0.591 | 0.173 | -0.032 |
| ar-zh | cross | no | 0 | 0.025 | 0.012 | 0.008 | 0.037 | 0.130 | 0.102 | -0.083 |

### centered, ref layer

`lex` is the model-free TF-IDF token-overlap floor; `vs lex` below 0 means the model adds nothing on that pair.

| pair | script | trained | layer | top1 A->B | top1 B->A | mutual-NN | cos margin | CKA | lex | vs lex |
|---|---|---|---|---|---|---|---|---|---|---|
| en-de | same | no | 12 | 0.759 | 0.679 | 0.597 | 0.382 | 0.648 | 0.434 | +0.285 |
| en-fr | same | no | 12 | 0.861 | 0.831 | 0.765 | 0.431 | 0.683 | 0.580 | +0.266 |
| en-ar | cross | no | 12 | 0.044 | 0.016 | 0.009 | 0.133 | 0.378 | 0.134 | -0.104 |
| en-zh | cross | no | 12 | 0.833 | 0.737 | 0.666 | 0.415 | 0.620 | 0.198 | +0.587 |
| de-fr | same | no | 12 | 0.580 | 0.606 | 0.475 | 0.437 | 0.671 | 0.342 | +0.251 |
| de-ar | cross | no | 12 | 0.042 | 0.024 | 0.010 | 0.203 | 0.400 | 0.114 | -0.081 |
| de-zh | cross | no | 12 | 0.406 | 0.385 | 0.254 | 0.366 | 0.586 | 0.181 | +0.215 |
| fr-ar | cross | no | 12 | 0.050 | 0.026 | 0.012 | 0.193 | 0.393 | 0.111 | -0.073 |
| fr-zh | cross | no | 12 | 0.506 | 0.452 | 0.333 | 0.377 | 0.581 | 0.173 | +0.307 |
| ar-zh | cross | no | 12 | 0.021 | 0.044 | 0.012 | 0.212 | 0.397 | 0.102 | -0.070 |

### centered, best layer

`lex` is the model-free TF-IDF token-overlap floor; `vs lex` below 0 means the model adds nothing on that pair.

| pair | script | trained | layer | top1 A->B | top1 B->A | mutual-NN | cos margin | CKA | lex | vs lex |
|---|---|---|---|---|---|---|---|---|---|---|
| en-de | same | no | 13 | 0.710 | 0.713 | 0.609 | 0.379 | 0.576 | 0.434 | +0.277 |
| en-fr | same | no | 13 | 0.839 | 0.859 | 0.774 | 0.433 | 0.634 | 0.580 | +0.269 |
| en-ar | cross | no | 0 | 0.076 | 0.066 | 0.055 | 0.059 | 0.125 | 0.134 | -0.063 |
| en-zh | cross | no | 13 | 0.825 | 0.794 | 0.722 | 0.430 | 0.598 | 0.198 | +0.611 |
| de-fr | same | no | 13 | 0.603 | 0.632 | 0.511 | 0.441 | 0.627 | 0.342 | +0.275 |
| de-ar | cross | no | 0 | 0.057 | 0.052 | 0.039 | 0.067 | 0.105 | 0.114 | -0.060 |
| de-zh | cross | no | 13 | 0.465 | 0.446 | 0.329 | 0.372 | 0.562 | 0.181 | +0.275 |
| fr-ar | cross | no | 0 | 0.059 | 0.052 | 0.040 | 0.056 | 0.098 | 0.111 | -0.056 |
| fr-zh | cross | no | 13 | 0.548 | 0.511 | 0.401 | 0.389 | 0.566 | 0.173 | +0.356 |
| ar-zh | cross | no | 0 | 0.043 | 0.041 | 0.026 | 0.070 | 0.128 | 0.102 | -0.060 |

