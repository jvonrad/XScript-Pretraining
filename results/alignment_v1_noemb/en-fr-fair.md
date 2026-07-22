# Alignment: en-fr-fair (FLORES+ both, n=2009)

Languages embedded: en, de, fr, ar, zh; trained on: en, fr.

### raw, ref layer

`lex` is the model-free TF-IDF token-overlap floor; `vs lex` below 0 means the model adds nothing on that pair.

| pair | script | trained | layer | top1 A->B | top1 B->A | mutual-NN | cos margin | CKA | lex | vs lex |
|---|---|---|---|---|---|---|---|---|---|---|
| en-de | same | no | 12 | 0.432 | 0.565 | 0.305 | 0.091 | 0.694 | 0.434 | +0.064 |
| en-fr | same | yes | 12 | 0.970 | 0.993 | 0.964 | 0.163 | 0.804 | 0.580 | +0.401 |
| en-ar | cross | no | 12 | 0.004 | 0.013 | 0.000 | 0.035 | 0.575 | 0.134 | -0.125 |
| en-zh | cross | no | 12 | 0.061 | 0.634 | 0.055 | 0.093 | 0.580 | 0.198 | +0.149 |
| de-fr | same | no | 12 | 0.312 | 0.500 | 0.212 | 0.090 | 0.685 | 0.342 | +0.063 |
| de-ar | cross | no | 12 | 0.013 | 0.020 | 0.002 | 0.038 | 0.569 | 0.114 | -0.098 |
| de-zh | cross | no | 12 | 0.053 | 0.264 | 0.046 | 0.072 | 0.535 | 0.181 | -0.023 |
| fr-ar | cross | no | 12 | 0.007 | 0.006 | 0.000 | 0.035 | 0.546 | 0.111 | -0.105 |
| fr-zh | cross | no | 12 | 0.086 | 0.722 | 0.068 | 0.092 | 0.577 | 0.173 | +0.231 |
| ar-zh | cross | no | 12 | 0.005 | 0.017 | 0.001 | 0.034 | 0.405 | 0.102 | -0.091 |

### raw, best layer

`lex` is the model-free TF-IDF token-overlap floor; `vs lex` below 0 means the model adds nothing on that pair.

| pair | script | trained | layer | top1 A->B | top1 B->A | mutual-NN | cos margin | CKA | lex | vs lex |
|---|---|---|---|---|---|---|---|---|---|---|
| en-de | same | no | 12 | 0.432 | 0.565 | 0.305 | 0.091 | 0.694 | 0.434 | +0.064 |
| en-fr | same | yes | 14 | 0.992 | 0.999 | 0.992 | 0.231 | 0.857 | 0.580 | +0.415 |
| en-ar | cross | no | 0 | 0.044 | 0.037 | 0.026 | 0.046 | 0.153 | 0.134 | -0.093 |
| en-zh | cross | no | 0 | 0.202 | 0.122 | 0.092 | 0.119 | 0.196 | 0.198 | -0.036 |
| de-fr | same | no | 12 | 0.312 | 0.500 | 0.212 | 0.090 | 0.685 | 0.342 | +0.063 |
| de-ar | cross | no | 0 | 0.041 | 0.032 | 0.023 | 0.043 | 0.137 | 0.114 | -0.078 |
| de-zh | cross | no | 12 | 0.053 | 0.264 | 0.046 | 0.072 | 0.535 | 0.181 | -0.023 |
| fr-ar | cross | no | 0 | 0.034 | 0.025 | 0.018 | 0.036 | 0.123 | 0.111 | -0.082 |
| fr-zh | cross | no | 12 | 0.086 | 0.722 | 0.068 | 0.092 | 0.577 | 0.173 | +0.231 |
| ar-zh | cross | no | 0 | 0.040 | 0.019 | 0.010 | 0.045 | 0.151 | 0.102 | -0.073 |

### centered, ref layer

`lex` is the model-free TF-IDF token-overlap floor; `vs lex` below 0 means the model adds nothing on that pair.

| pair | script | trained | layer | top1 A->B | top1 B->A | mutual-NN | cos margin | CKA | lex | vs lex |
|---|---|---|---|---|---|---|---|---|---|---|
| en-de | same | no | 12 | 0.914 | 0.894 | 0.846 | 0.537 | 0.711 | 0.434 | +0.470 |
| en-fr | same | yes | 12 | 0.995 | 0.996 | 0.992 | 0.803 | 0.819 | 0.580 | +0.415 |
| en-ar | cross | no | 12 | 0.150 | 0.053 | 0.029 | 0.236 | 0.559 | 0.134 | -0.032 |
| en-zh | cross | no | 12 | 0.896 | 0.926 | 0.862 | 0.525 | 0.619 | 0.198 | +0.713 |
| de-fr | same | no | 12 | 0.857 | 0.897 | 0.799 | 0.513 | 0.700 | 0.342 | +0.535 |
| de-ar | cross | no | 12 | 0.130 | 0.061 | 0.029 | 0.299 | 0.561 | 0.114 | -0.019 |
| de-zh | cross | no | 12 | 0.701 | 0.775 | 0.614 | 0.469 | 0.572 | 0.181 | +0.557 |
| fr-ar | cross | no | 12 | 0.135 | 0.049 | 0.025 | 0.229 | 0.532 | 0.111 | -0.019 |
| fr-zh | cross | no | 12 | 0.881 | 0.900 | 0.829 | 0.499 | 0.609 | 0.173 | +0.718 |
| ar-zh | cross | no | 12 | 0.064 | 0.111 | 0.029 | 0.264 | 0.408 | 0.102 | -0.015 |

### centered, best layer

`lex` is the model-free TF-IDF token-overlap floor; `vs lex` below 0 means the model adds nothing on that pair.

| pair | script | trained | layer | top1 A->B | top1 B->A | mutual-NN | cos margin | CKA | lex | vs lex |
|---|---|---|---|---|---|---|---|---|---|---|
| en-de | same | no | 12 | 0.914 | 0.894 | 0.846 | 0.537 | 0.711 | 0.434 | +0.470 |
| en-fr | same | yes | 15 | 0.997 | 1.000 | 0.997 | 0.760 | 0.865 | 0.580 | +0.418 |
| en-ar | cross | no | 0 | 0.084 | 0.082 | 0.060 | 0.074 | 0.153 | 0.134 | -0.051 |
| en-zh | cross | no | 13 | 0.895 | 0.934 | 0.866 | 0.525 | 0.616 | 0.198 | +0.716 |
| de-fr | same | no | 12 | 0.857 | 0.897 | 0.799 | 0.513 | 0.700 | 0.342 | +0.535 |
| de-ar | cross | no | 0 | 0.066 | 0.062 | 0.041 | 0.078 | 0.133 | 0.114 | -0.051 |
| de-zh | cross | no | 12 | 0.701 | 0.775 | 0.614 | 0.469 | 0.572 | 0.181 | +0.557 |
| fr-ar | cross | no | 0 | 0.069 | 0.065 | 0.050 | 0.058 | 0.123 | 0.111 | -0.044 |
| fr-zh | cross | no | 13 | 0.878 | 0.913 | 0.839 | 0.500 | 0.615 | 0.173 | +0.722 |
| ar-zh | cross | no | 15 | 0.077 | 0.091 | 0.045 | 0.176 | 0.268 | 0.102 | -0.018 |

