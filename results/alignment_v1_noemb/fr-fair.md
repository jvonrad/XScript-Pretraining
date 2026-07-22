# Alignment: fr-fair (FLORES+ both, n=2009)

Languages embedded: en, de, fr, ar, zh; trained on: fr.

### raw, ref layer

`lex` is the model-free TF-IDF token-overlap floor; `vs lex` below 0 means the model adds nothing on that pair.

| pair | script | trained | layer | top1 A->B | top1 B->A | mutual-NN | cos margin | CKA | lex | vs lex |
|---|---|---|---|---|---|---|---|---|---|---|
| en-de | same | no | 12 | 0.499 | 0.733 | 0.447 | 0.085 | 0.772 | 0.434 | +0.181 |
| en-fr | same | no | 12 | 0.934 | 0.991 | 0.932 | 0.134 | 0.828 | 0.580 | +0.382 |
| en-ar | cross | no | 12 | 0.003 | 0.013 | 0.000 | 0.032 | 0.638 | 0.134 | -0.126 |
| en-zh | cross | no | 12 | 0.365 | 0.646 | 0.275 | 0.072 | 0.676 | 0.198 | +0.307 |
| de-fr | same | no | 12 | 0.359 | 0.439 | 0.234 | 0.079 | 0.730 | 0.342 | +0.056 |
| de-ar | cross | no | 12 | 0.008 | 0.021 | 0.002 | 0.034 | 0.650 | 0.114 | -0.100 |
| de-zh | cross | no | 12 | 0.316 | 0.282 | 0.109 | 0.058 | 0.625 | 0.181 | +0.118 |
| fr-ar | cross | no | 12 | 0.003 | 0.009 | 0.000 | 0.031 | 0.602 | 0.111 | -0.105 |
| fr-zh | cross | no | 12 | 0.395 | 0.232 | 0.094 | 0.070 | 0.625 | 0.173 | +0.141 |
| ar-zh | cross | no | 12 | 0.017 | 0.013 | 0.002 | 0.035 | 0.632 | 0.102 | -0.087 |

### raw, best layer

`lex` is the model-free TF-IDF token-overlap floor; `vs lex` below 0 means the model adds nothing on that pair.

| pair | script | trained | layer | top1 A->B | top1 B->A | mutual-NN | cos margin | CKA | lex | vs lex |
|---|---|---|---|---|---|---|---|---|---|---|
| en-de | same | no | 12 | 0.499 | 0.733 | 0.447 | 0.085 | 0.772 | 0.434 | +0.181 |
| en-fr | same | no | 14 | 0.980 | 0.985 | 0.972 | 0.158 | 0.833 | 0.580 | +0.402 |
| en-ar | cross | no | 0 | 0.040 | 0.045 | 0.028 | 0.048 | 0.165 | 0.134 | -0.091 |
| en-zh | cross | no | 14 | 0.419 | 0.615 | 0.296 | 0.087 | 0.676 | 0.198 | +0.318 |
| de-fr | same | no | 13 | 0.448 | 0.403 | 0.256 | 0.090 | 0.708 | 0.342 | +0.083 |
| de-ar | cross | no | 0 | 0.046 | 0.035 | 0.027 | 0.047 | 0.139 | 0.114 | -0.074 |
| de-zh | cross | no | 12 | 0.316 | 0.282 | 0.109 | 0.058 | 0.625 | 0.181 | +0.118 |
| fr-ar | cross | no | 0 | 0.036 | 0.028 | 0.023 | 0.039 | 0.125 | 0.111 | -0.079 |
| fr-zh | cross | no | 14 | 0.281 | 0.412 | 0.126 | 0.081 | 0.619 | 0.173 | +0.174 |
| ar-zh | cross | no | 0 | 0.035 | 0.018 | 0.010 | 0.048 | 0.153 | 0.102 | -0.075 |

### centered, ref layer

`lex` is the model-free TF-IDF token-overlap floor; `vs lex` below 0 means the model adds nothing on that pair.

| pair | script | trained | layer | top1 A->B | top1 B->A | mutual-NN | cos margin | CKA | lex | vs lex |
|---|---|---|---|---|---|---|---|---|---|---|
| en-de | same | no | 12 | 0.954 | 0.926 | 0.896 | 0.543 | 0.759 | 0.434 | +0.506 |
| en-fr | same | no | 12 | 0.995 | 0.999 | 0.995 | 0.724 | 0.823 | 0.580 | +0.417 |
| en-ar | cross | no | 12 | 0.193 | 0.085 | 0.054 | 0.245 | 0.620 | 0.134 | +0.005 |
| en-zh | cross | no | 12 | 0.923 | 0.790 | 0.744 | 0.513 | 0.674 | 0.198 | +0.658 |
| de-fr | same | no | 12 | 0.861 | 0.908 | 0.803 | 0.470 | 0.719 | 0.342 | +0.542 |
| de-ar | cross | no | 12 | 0.141 | 0.081 | 0.044 | 0.287 | 0.621 | 0.114 | -0.003 |
| de-zh | cross | no | 12 | 0.690 | 0.592 | 0.460 | 0.446 | 0.622 | 0.181 | +0.460 |
| fr-ar | cross | no | 12 | 0.172 | 0.066 | 0.040 | 0.218 | 0.584 | 0.111 | +0.008 |
| fr-zh | cross | no | 12 | 0.899 | 0.749 | 0.702 | 0.460 | 0.631 | 0.173 | +0.651 |
| ar-zh | cross | no | 12 | 0.077 | 0.092 | 0.029 | 0.318 | 0.600 | 0.102 | -0.018 |

### centered, best layer

`lex` is the model-free TF-IDF token-overlap floor; `vs lex` below 0 means the model adds nothing on that pair.

| pair | script | trained | layer | top1 A->B | top1 B->A | mutual-NN | cos margin | CKA | lex | vs lex |
|---|---|---|---|---|---|---|---|---|---|---|
| en-de | same | no | 13 | 0.957 | 0.952 | 0.923 | 0.560 | 0.745 | 0.434 | +0.520 |
| en-fr | same | no | 13 | 0.997 | 0.999 | 0.996 | 0.746 | 0.840 | 0.580 | +0.418 |
| en-ar | cross | no | 13 | 0.192 | 0.102 | 0.066 | 0.239 | 0.559 | 0.134 | +0.013 |
| en-zh | cross | no | 14 | 0.934 | 0.891 | 0.857 | 0.544 | 0.685 | 0.198 | +0.714 |
| de-fr | same | no | 13 | 0.899 | 0.917 | 0.852 | 0.488 | 0.703 | 0.342 | +0.566 |
| de-ar | cross | no | 13 | 0.162 | 0.093 | 0.055 | 0.283 | 0.581 | 0.114 | +0.013 |
| de-zh | cross | no | 14 | 0.759 | 0.711 | 0.620 | 0.470 | 0.623 | 0.181 | +0.554 |
| fr-ar | cross | no | 0 | 0.071 | 0.059 | 0.049 | 0.059 | 0.126 | 0.111 | -0.046 |
| fr-zh | cross | no | 14 | 0.903 | 0.857 | 0.809 | 0.486 | 0.639 | 0.173 | +0.707 |
| ar-zh | cross | no | 14 | 0.103 | 0.116 | 0.051 | 0.287 | 0.537 | 0.102 | +0.007 |

