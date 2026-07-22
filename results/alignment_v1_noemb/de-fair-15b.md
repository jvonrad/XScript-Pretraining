# Alignment: de-fair-15b (FLORES+ both, n=2009)

Languages embedded: en, de, fr, ar, zh; trained on: de.

### raw, ref layer

`lex` is the model-free TF-IDF token-overlap floor; `vs lex` below 0 means the model adds nothing on that pair.

| pair | script | trained | layer | top1 A->B | top1 B->A | mutual-NN | cos margin | CKA | lex | vs lex |
|---|---|---|---|---|---|---|---|---|---|---|
| en-de | same | no | 12 | 0.974 | 0.991 | 0.968 | 0.152 | 0.828 | 0.434 | +0.548 |
| en-fr | same | no | 12 | 0.781 | 0.882 | 0.747 | 0.106 | 0.769 | 0.580 | +0.251 |
| en-ar | cross | no | 12 | 0.003 | 0.002 | 0.000 | 0.022 | 0.466 | 0.134 | -0.131 |
| en-zh | cross | no | 12 | 0.039 | 0.079 | 0.018 | 0.054 | 0.595 | 0.198 | -0.139 |
| de-fr | same | no | 12 | 0.582 | 0.570 | 0.438 | 0.096 | 0.733 | 0.342 | +0.234 |
| de-ar | cross | no | 12 | 0.001 | 0.001 | 0.000 | 0.019 | 0.419 | 0.114 | -0.113 |
| de-zh | cross | no | 12 | 0.032 | 0.019 | 0.007 | 0.049 | 0.552 | 0.181 | -0.156 |
| fr-ar | cross | no | 12 | 0.009 | 0.004 | 0.000 | 0.023 | 0.479 | 0.111 | -0.104 |
| fr-zh | cross | no | 12 | 0.085 | 0.080 | 0.026 | 0.049 | 0.565 | 0.173 | -0.090 |
| ar-zh | cross | no | 12 | 0.004 | 0.016 | 0.002 | 0.027 | 0.489 | 0.102 | -0.092 |

### raw, best layer

`lex` is the model-free TF-IDF token-overlap floor; `vs lex` below 0 means the model adds nothing on that pair.

| pair | script | trained | layer | top1 A->B | top1 B->A | mutual-NN | cos margin | CKA | lex | vs lex |
|---|---|---|---|---|---|---|---|---|---|---|
| en-de | same | no | 12 | 0.974 | 0.991 | 0.968 | 0.152 | 0.828 | 0.434 | +0.548 |
| en-fr | same | no | 12 | 0.781 | 0.882 | 0.747 | 0.106 | 0.769 | 0.580 | +0.251 |
| en-ar | cross | no | 0 | 0.053 | 0.045 | 0.034 | 0.047 | 0.143 | 0.134 | -0.085 |
| en-zh | cross | no | 0 | 0.095 | 0.062 | 0.047 | 0.077 | 0.164 | 0.198 | -0.120 |
| de-fr | same | no | 12 | 0.582 | 0.570 | 0.438 | 0.096 | 0.733 | 0.342 | +0.234 |
| de-ar | cross | no | 0 | 0.039 | 0.034 | 0.024 | 0.037 | 0.099 | 0.114 | -0.078 |
| de-zh | cross | no | 0 | 0.080 | 0.051 | 0.034 | 0.065 | 0.133 | 0.181 | -0.115 |
| fr-ar | cross | no | 0 | 0.051 | 0.027 | 0.020 | 0.036 | 0.107 | 0.111 | -0.072 |
| fr-zh | cross | no | 12 | 0.085 | 0.080 | 0.026 | 0.049 | 0.565 | 0.173 | -0.090 |
| ar-zh | cross | no | 0 | 0.028 | 0.018 | 0.010 | 0.036 | 0.123 | 0.102 | -0.079 |

### centered, ref layer

`lex` is the model-free TF-IDF token-overlap floor; `vs lex` below 0 means the model adds nothing on that pair.

| pair | script | trained | layer | top1 A->B | top1 B->A | mutual-NN | cos margin | CKA | lex | vs lex |
|---|---|---|---|---|---|---|---|---|---|---|
| en-de | same | no | 12 | 0.999 | 0.997 | 0.995 | 0.701 | 0.825 | 0.434 | +0.563 |
| en-fr | same | no | 12 | 0.970 | 0.961 | 0.943 | 0.578 | 0.760 | 0.580 | +0.385 |
| en-ar | cross | no | 12 | 0.041 | 0.016 | 0.010 | 0.158 | 0.469 | 0.134 | -0.105 |
| en-zh | cross | no | 12 | 0.557 | 0.378 | 0.300 | 0.342 | 0.582 | 0.198 | +0.269 |
| de-fr | same | no | 12 | 0.938 | 0.940 | 0.899 | 0.498 | 0.727 | 0.342 | +0.597 |
| de-ar | cross | no | 12 | 0.032 | 0.012 | 0.006 | 0.133 | 0.422 | 0.114 | -0.093 |
| de-zh | cross | no | 12 | 0.489 | 0.337 | 0.247 | 0.292 | 0.546 | 0.181 | +0.232 |
| fr-ar | cross | no | 12 | 0.045 | 0.016 | 0.009 | 0.185 | 0.471 | 0.111 | -0.080 |
| fr-zh | cross | no | 12 | 0.407 | 0.271 | 0.201 | 0.346 | 0.547 | 0.173 | +0.166 |
| ar-zh | cross | no | 12 | 0.021 | 0.028 | 0.008 | 0.248 | 0.463 | 0.102 | -0.078 |

### centered, best layer

`lex` is the model-free TF-IDF token-overlap floor; `vs lex` below 0 means the model adds nothing on that pair.

| pair | script | trained | layer | top1 A->B | top1 B->A | mutual-NN | cos margin | CKA | lex | vs lex |
|---|---|---|---|---|---|---|---|---|---|---|
| en-de | same | no | 12 | 0.999 | 0.997 | 0.995 | 0.701 | 0.825 | 0.434 | +0.563 |
| en-fr | same | no | 12 | 0.970 | 0.961 | 0.943 | 0.578 | 0.760 | 0.580 | +0.385 |
| en-ar | cross | no | 0 | 0.075 | 0.068 | 0.049 | 0.070 | 0.142 | 0.134 | -0.063 |
| en-zh | cross | no | 14 | 0.512 | 0.417 | 0.324 | 0.337 | 0.541 | 0.198 | +0.266 |
| de-fr | same | no | 12 | 0.938 | 0.940 | 0.899 | 0.498 | 0.727 | 0.342 | +0.597 |
| de-ar | cross | no | 0 | 0.058 | 0.052 | 0.037 | 0.050 | 0.099 | 0.114 | -0.059 |
| de-zh | cross | no | 14 | 0.453 | 0.381 | 0.272 | 0.290 | 0.497 | 0.181 | +0.236 |
| fr-ar | cross | no | 0 | 0.065 | 0.056 | 0.042 | 0.060 | 0.104 | 0.111 | -0.051 |
| fr-zh | cross | no | 14 | 0.417 | 0.326 | 0.244 | 0.338 | 0.516 | 0.173 | +0.199 |
| ar-zh | cross | no | 0 | 0.042 | 0.032 | 0.022 | 0.062 | 0.122 | 0.102 | -0.065 |

