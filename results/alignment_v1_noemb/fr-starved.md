# Alignment: fr-starved (FLORES+ both, n=2009)

Languages embedded: en, de, fr, ar, zh; trained on: fr.

### raw, ref layer

`lex` is the model-free TF-IDF token-overlap floor; `vs lex` below 0 means the model adds nothing on that pair.

| pair | script | trained | layer | top1 A->B | top1 B->A | mutual-NN | cos margin | CKA | lex | vs lex |
|---|---|---|---|---|---|---|---|---|---|---|
| en-de | same | no | 12 | 0.133 | 0.372 | 0.126 | 0.049 | 0.646 | 0.457 | -0.204 |
| en-fr | same | no | 12 | 0.461 | 0.720 | 0.459 | 0.105 | 0.201 | 0.645 | -0.055 |
| en-ar | cross | no | 12 | 0.003 | 0.012 | 0.000 | 0.016 | 0.548 | 0.133 | -0.126 |
| en-zh | cross | no | 12 | 0.077 | 0.626 | 0.064 | 0.049 | 0.693 | 0.196 | +0.155 |
| de-fr | same | no | 12 | 0.096 | 0.094 | 0.046 | 0.045 | 0.146 | 0.365 | -0.270 |
| de-ar | cross | no | 12 | 0.007 | 0.009 | 0.000 | 0.016 | 0.527 | 0.113 | -0.105 |
| de-zh | cross | no | 12 | 0.056 | 0.081 | 0.018 | 0.030 | 0.562 | 0.179 | -0.110 |
| fr-ar | cross | no | 12 | 0.003 | 0.008 | 0.001 | 0.015 | 0.119 | 0.113 | -0.108 |
| fr-zh | cross | no | 12 | 0.066 | 0.210 | 0.028 | 0.048 | 0.158 | 0.177 | -0.039 |
| ar-zh | cross | no | 12 | 0.013 | 0.011 | 0.002 | 0.016 | 0.552 | 0.106 | -0.093 |

### raw, best layer

`lex` is the model-free TF-IDF token-overlap floor; `vs lex` below 0 means the model adds nothing on that pair.

| pair | script | trained | layer | top1 A->B | top1 B->A | mutual-NN | cos margin | CKA | lex | vs lex |
|---|---|---|---|---|---|---|---|---|---|---|
| en-de | same | no | 13 | 0.204 | 0.419 | 0.176 | 0.061 | 0.632 | 0.457 | -0.146 |
| en-fr | same | no | 14 | 0.789 | 0.976 | 0.783 | 0.141 | 0.508 | 0.645 | +0.237 |
| en-ar | cross | no | 0 | 0.038 | 0.027 | 0.020 | 0.032 | 0.130 | 0.133 | -0.100 |
| en-zh | cross | no | 13 | 0.249 | 0.637 | 0.187 | 0.062 | 0.704 | 0.196 | +0.247 |
| de-fr | same | no | 0 | 0.098 | 0.143 | 0.085 | 0.077 | 0.165 | 0.365 | -0.245 |
| de-ar | cross | no | 0 | 0.022 | 0.014 | 0.010 | 0.025 | 0.109 | 0.113 | -0.095 |
| de-zh | cross | no | 13 | 0.112 | 0.103 | 0.029 | 0.036 | 0.573 | 0.179 | -0.071 |
| fr-ar | cross | no | 0 | 0.028 | 0.022 | 0.015 | 0.020 | 0.113 | 0.113 | -0.088 |
| fr-zh | cross | no | 13 | 0.178 | 0.182 | 0.050 | 0.061 | 0.259 | 0.177 | +0.003 |
| ar-zh | cross | no | 0 | 0.019 | 0.016 | 0.005 | 0.032 | 0.132 | 0.106 | -0.088 |

### centered, ref layer

`lex` is the model-free TF-IDF token-overlap floor; `vs lex` below 0 means the model adds nothing on that pair.

| pair | script | trained | layer | top1 A->B | top1 B->A | mutual-NN | cos margin | CKA | lex | vs lex |
|---|---|---|---|---|---|---|---|---|---|---|
| en-de | same | no | 12 | 0.833 | 0.802 | 0.739 | 0.393 | 0.641 | 0.457 | +0.360 |
| en-fr | same | no | 12 | 0.974 | 0.859 | 0.852 | 0.570 | 0.226 | 0.645 | +0.271 |
| en-ar | cross | no | 12 | 0.128 | 0.096 | 0.059 | 0.175 | 0.540 | 0.133 | -0.021 |
| en-zh | cross | no | 12 | 0.938 | 0.893 | 0.855 | 0.447 | 0.695 | 0.196 | +0.720 |
| de-fr | same | no | 12 | 0.594 | 0.467 | 0.387 | 0.277 | 0.164 | 0.365 | +0.166 |
| de-ar | cross | no | 12 | 0.086 | 0.067 | 0.035 | 0.198 | 0.507 | 0.113 | -0.037 |
| de-zh | cross | no | 12 | 0.453 | 0.452 | 0.309 | 0.310 | 0.555 | 0.179 | +0.274 |
| fr-ar | cross | no | 12 | 0.094 | 0.064 | 0.034 | 0.123 | 0.130 | 0.113 | -0.034 |
| fr-zh | cross | no | 12 | 0.690 | 0.787 | 0.607 | 0.345 | 0.180 | 0.177 | +0.561 |
| ar-zh | cross | no | 12 | 0.072 | 0.077 | 0.029 | 0.217 | 0.517 | 0.106 | -0.032 |

### centered, best layer

`lex` is the model-free TF-IDF token-overlap floor; `vs lex` below 0 means the model adds nothing on that pair.

| pair | script | trained | layer | top1 A->B | top1 B->A | mutual-NN | cos margin | CKA | lex | vs lex |
|---|---|---|---|---|---|---|---|---|---|---|
| en-de | same | no | 13 | 0.862 | 0.861 | 0.799 | 0.421 | 0.632 | 0.457 | +0.405 |
| en-fr | same | no | 15 | 0.996 | 0.988 | 0.986 | 0.682 | 0.730 | 0.645 | +0.347 |
| en-ar | cross | no | 13 | 0.156 | 0.118 | 0.078 | 0.180 | 0.465 | 0.133 | +0.004 |
| en-zh | cross | no | 13 | 0.956 | 0.937 | 0.910 | 0.478 | 0.708 | 0.196 | +0.750 |
| de-fr | same | no | 14 | 0.738 | 0.645 | 0.592 | 0.324 | 0.359 | 0.365 | +0.327 |
| de-ar | cross | no | 15 | 0.110 | 0.102 | 0.053 | 0.185 | 0.379 | 0.113 | -0.007 |
| de-zh | cross | no | 14 | 0.581 | 0.556 | 0.449 | 0.350 | 0.577 | 0.179 | +0.390 |
| fr-ar | cross | no | 14 | 0.123 | 0.086 | 0.051 | 0.129 | 0.228 | 0.113 | -0.008 |
| fr-zh | cross | no | 14 | 0.879 | 0.891 | 0.830 | 0.406 | 0.417 | 0.177 | +0.708 |
| ar-zh | cross | no | 14 | 0.103 | 0.111 | 0.056 | 0.221 | 0.450 | 0.106 | +0.001 |

