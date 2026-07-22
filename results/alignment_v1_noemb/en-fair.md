# Alignment: en-fair (FLORES+ both, n=2009)

Languages embedded: en, de, fr, ar, zh; trained on: en.

### raw, ref layer

`lex` is the model-free TF-IDF token-overlap floor; `vs lex` below 0 means the model adds nothing on that pair.

| pair | script | trained | layer | top1 A->B | top1 B->A | mutual-NN | cos margin | CKA | lex | vs lex |
|---|---|---|---|---|---|---|---|---|---|---|
| en-de | same | no | 12 | 0.187 | 0.373 | 0.150 | 0.092 | 0.709 | 0.434 | -0.155 |
| en-fr | same | no | 12 | 0.177 | 0.760 | 0.167 | 0.107 | 0.743 | 0.580 | -0.111 |
| en-ar | cross | no | 12 | 0.002 | 0.005 | 0.000 | 0.027 | 0.431 | 0.134 | -0.130 |
| en-zh | cross | no | 12 | 0.139 | 0.530 | 0.085 | 0.095 | 0.653 | 0.198 | +0.136 |
| de-fr | same | no | 12 | 0.557 | 0.703 | 0.479 | 0.087 | 0.743 | 0.342 | +0.288 |
| de-ar | cross | no | 12 | 0.011 | 0.010 | 0.001 | 0.031 | 0.469 | 0.114 | -0.104 |
| de-zh | cross | no | 12 | 0.149 | 0.267 | 0.069 | 0.068 | 0.635 | 0.181 | +0.027 |
| fr-ar | cross | no | 12 | 0.012 | 0.013 | 0.002 | 0.031 | 0.461 | 0.111 | -0.099 |
| fr-zh | cross | no | 12 | 0.232 | 0.218 | 0.070 | 0.073 | 0.642 | 0.173 | +0.052 |
| ar-zh | cross | no | 12 | 0.011 | 0.015 | 0.001 | 0.030 | 0.476 | 0.102 | -0.089 |

### raw, best layer

`lex` is the model-free TF-IDF token-overlap floor; `vs lex` below 0 means the model adds nothing on that pair.

| pair | script | trained | layer | top1 A->B | top1 B->A | mutual-NN | cos margin | CKA | lex | vs lex |
|---|---|---|---|---|---|---|---|---|---|---|
| en-de | same | no | 0 | 0.301 | 0.245 | 0.213 | 0.156 | 0.219 | 0.434 | -0.161 |
| en-fr | same | no | 11 | 0.324 | 0.754 | 0.291 | 0.097 | 0.764 | 0.580 | -0.041 |
| en-ar | cross | no | 0 | 0.051 | 0.050 | 0.035 | 0.043 | 0.142 | 0.134 | -0.083 |
| en-zh | cross | no | 10 | 0.376 | 0.591 | 0.225 | 0.076 | 0.648 | 0.198 | +0.285 |
| de-fr | same | no | 12 | 0.557 | 0.703 | 0.479 | 0.087 | 0.743 | 0.342 | +0.288 |
| de-ar | cross | no | 0 | 0.044 | 0.040 | 0.027 | 0.041 | 0.110 | 0.114 | -0.073 |
| de-zh | cross | no | 10 | 0.334 | 0.254 | 0.108 | 0.058 | 0.610 | 0.181 | +0.113 |
| fr-ar | cross | no | 0 | 0.040 | 0.025 | 0.019 | 0.035 | 0.104 | 0.111 | -0.078 |
| fr-zh | cross | no | 10 | 0.521 | 0.261 | 0.135 | 0.062 | 0.619 | 0.173 | +0.218 |
| ar-zh | cross | no | 0 | 0.035 | 0.018 | 0.010 | 0.043 | 0.135 | 0.102 | -0.075 |

### centered, ref layer

`lex` is the model-free TF-IDF token-overlap floor; `vs lex` below 0 means the model adds nothing on that pair.

| pair | script | trained | layer | top1 A->B | top1 B->A | mutual-NN | cos margin | CKA | lex | vs lex |
|---|---|---|---|---|---|---|---|---|---|---|
| en-de | same | no | 12 | 0.891 | 0.864 | 0.800 | 0.466 | 0.699 | 0.434 | +0.443 |
| en-fr | same | no | 12 | 0.944 | 0.940 | 0.905 | 0.518 | 0.737 | 0.580 | +0.362 |
| en-ar | cross | no | 12 | 0.072 | 0.030 | 0.017 | 0.168 | 0.433 | 0.134 | -0.083 |
| en-zh | cross | no | 12 | 0.929 | 0.882 | 0.842 | 0.510 | 0.663 | 0.198 | +0.707 |
| de-fr | same | no | 12 | 0.814 | 0.842 | 0.743 | 0.522 | 0.723 | 0.342 | +0.486 |
| de-ar | cross | no | 12 | 0.072 | 0.045 | 0.020 | 0.240 | 0.460 | 0.114 | -0.056 |
| de-zh | cross | no | 12 | 0.702 | 0.661 | 0.544 | 0.449 | 0.632 | 0.181 | +0.500 |
| fr-ar | cross | no | 12 | 0.095 | 0.051 | 0.027 | 0.231 | 0.452 | 0.111 | -0.038 |
| fr-zh | cross | no | 12 | 0.795 | 0.732 | 0.644 | 0.468 | 0.639 | 0.173 | +0.591 |
| ar-zh | cross | no | 12 | 0.040 | 0.058 | 0.018 | 0.244 | 0.455 | 0.102 | -0.053 |

### centered, best layer

`lex` is the model-free TF-IDF token-overlap floor; `vs lex` below 0 means the model adds nothing on that pair.

| pair | script | trained | layer | top1 A->B | top1 B->A | mutual-NN | cos margin | CKA | lex | vs lex |
|---|---|---|---|---|---|---|---|---|---|---|
| en-de | same | no | 12 | 0.891 | 0.864 | 0.800 | 0.466 | 0.699 | 0.434 | +0.443 |
| en-fr | same | no | 12 | 0.944 | 0.940 | 0.905 | 0.518 | 0.737 | 0.580 | +0.362 |
| en-ar | cross | no | 0 | 0.097 | 0.078 | 0.065 | 0.062 | 0.143 | 0.134 | -0.047 |
| en-zh | cross | no | 13 | 0.913 | 0.900 | 0.856 | 0.523 | 0.657 | 0.198 | +0.708 |
| de-fr | same | no | 13 | 0.821 | 0.844 | 0.760 | 0.529 | 0.692 | 0.342 | +0.490 |
| de-ar | cross | no | 0 | 0.065 | 0.066 | 0.046 | 0.069 | 0.108 | 0.114 | -0.049 |
| de-zh | cross | no | 13 | 0.709 | 0.688 | 0.570 | 0.460 | 0.616 | 0.181 | +0.517 |
| fr-ar | cross | no | 0 | 0.064 | 0.064 | 0.048 | 0.061 | 0.102 | 0.111 | -0.047 |
| fr-zh | cross | no | 13 | 0.808 | 0.747 | 0.679 | 0.480 | 0.630 | 0.173 | +0.605 |
| ar-zh | cross | no | 0 | 0.052 | 0.044 | 0.029 | 0.080 | 0.134 | 0.102 | -0.054 |

