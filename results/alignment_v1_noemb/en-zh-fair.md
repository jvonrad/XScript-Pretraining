# Alignment: en-zh-fair (FLORES+ both, n=2009)

Languages embedded: en, de, fr, ar, zh; trained on: en, zh.

### raw, ref layer

`lex` is the model-free TF-IDF token-overlap floor; `vs lex` below 0 means the model adds nothing on that pair.

| pair | script | trained | layer | top1 A->B | top1 B->A | mutual-NN | cos margin | CKA | lex | vs lex |
|---|---|---|---|---|---|---|---|---|---|---|
| en-de | same | no | 12 | 0.087 | 0.130 | 0.055 | 0.065 | 0.722 | 0.434 | -0.326 |
| en-fr | same | no | 12 | 0.084 | 0.258 | 0.065 | 0.074 | 0.739 | 0.580 | -0.409 |
| en-ar | cross | no | 12 | 0.002 | 0.003 | 0.000 | 0.022 | 0.451 | 0.134 | -0.131 |
| en-zh | cross | yes | 12 | 0.939 | 0.954 | 0.899 | 0.135 | 0.726 | 0.198 | +0.748 |
| de-fr | same | no | 12 | 0.251 | 0.427 | 0.181 | 0.063 | 0.762 | 0.342 | -0.003 |
| de-ar | cross | no | 12 | 0.008 | 0.004 | 0.001 | 0.029 | 0.492 | 0.114 | -0.108 |
| de-zh | cross | no | 12 | 0.060 | 0.056 | 0.016 | 0.058 | 0.627 | 0.181 | -0.123 |
| fr-ar | cross | no | 12 | 0.009 | 0.007 | 0.001 | 0.029 | 0.504 | 0.111 | -0.103 |
| fr-zh | cross | no | 12 | 0.126 | 0.053 | 0.020 | 0.064 | 0.638 | 0.173 | -0.083 |
| ar-zh | cross | no | 12 | 0.002 | 0.003 | 0.000 | 0.024 | 0.453 | 0.102 | -0.100 |

### raw, best layer

`lex` is the model-free TF-IDF token-overlap floor; `vs lex` below 0 means the model adds nothing on that pair.

| pair | script | trained | layer | top1 A->B | top1 B->A | mutual-NN | cos margin | CKA | lex | vs lex |
|---|---|---|---|---|---|---|---|---|---|---|
| en-de | same | no | 0 | 0.205 | 0.212 | 0.159 | 0.144 | 0.226 | 0.434 | -0.226 |
| en-fr | same | no | 1 | 0.168 | 0.268 | 0.148 | 0.074 | 0.318 | 0.580 | -0.362 |
| en-ar | cross | no | 0 | 0.024 | 0.036 | 0.018 | 0.038 | 0.144 | 0.134 | -0.104 |
| en-zh | cross | yes | 13 | 0.947 | 0.961 | 0.918 | 0.162 | 0.748 | 0.198 | +0.755 |
| de-fr | same | no | 9 | 0.313 | 0.390 | 0.187 | 0.037 | 0.780 | 0.342 | +0.009 |
| de-ar | cross | no | 0 | 0.038 | 0.032 | 0.020 | 0.040 | 0.122 | 0.114 | -0.080 |
| de-zh | cross | no | 0 | 0.075 | 0.046 | 0.029 | 0.069 | 0.195 | 0.181 | -0.120 |
| fr-ar | cross | no | 0 | 0.038 | 0.029 | 0.020 | 0.031 | 0.113 | 0.111 | -0.078 |
| fr-zh | cross | no | 10 | 0.203 | 0.072 | 0.027 | 0.043 | 0.615 | 0.173 | -0.035 |
| ar-zh | cross | no | 0 | 0.017 | 0.010 | 0.005 | 0.028 | 0.141 | 0.102 | -0.089 |

### centered, ref layer

`lex` is the model-free TF-IDF token-overlap floor; `vs lex` below 0 means the model adds nothing on that pair.

| pair | script | trained | layer | top1 A->B | top1 B->A | mutual-NN | cos margin | CKA | lex | vs lex |
|---|---|---|---|---|---|---|---|---|---|---|
| en-de | same | no | 12 | 0.761 | 0.653 | 0.571 | 0.422 | 0.701 | 0.434 | +0.272 |
| en-fr | same | no | 12 | 0.878 | 0.820 | 0.753 | 0.474 | 0.723 | 0.580 | +0.269 |
| en-ar | cross | no | 12 | 0.042 | 0.012 | 0.007 | 0.159 | 0.447 | 0.134 | -0.107 |
| en-zh | cross | yes | 12 | 0.978 | 0.971 | 0.952 | 0.694 | 0.728 | 0.198 | +0.776 |
| de-fr | same | no | 12 | 0.585 | 0.646 | 0.472 | 0.511 | 0.729 | 0.342 | +0.273 |
| de-ar | cross | no | 12 | 0.039 | 0.016 | 0.007 | 0.264 | 0.472 | 0.114 | -0.087 |
| de-zh | cross | no | 12 | 0.500 | 0.575 | 0.369 | 0.376 | 0.625 | 0.181 | +0.357 |
| fr-ar | cross | no | 12 | 0.046 | 0.017 | 0.008 | 0.252 | 0.483 | 0.111 | -0.080 |
| fr-zh | cross | no | 12 | 0.668 | 0.689 | 0.519 | 0.409 | 0.641 | 0.173 | +0.506 |
| ar-zh | cross | no | 12 | 0.011 | 0.037 | 0.005 | 0.169 | 0.446 | 0.102 | -0.078 |

### centered, best layer

`lex` is the model-free TF-IDF token-overlap floor; `vs lex` below 0 means the model adds nothing on that pair.

| pair | script | trained | layer | top1 A->B | top1 B->A | mutual-NN | cos margin | CKA | lex | vs lex |
|---|---|---|---|---|---|---|---|---|---|---|
| en-de | same | no | 11 | 0.778 | 0.665 | 0.582 | 0.421 | 0.715 | 0.434 | +0.287 |
| en-fr | same | no | 11 | 0.895 | 0.830 | 0.767 | 0.475 | 0.743 | 0.580 | +0.282 |
| en-ar | cross | no | 0 | 0.082 | 0.072 | 0.052 | 0.065 | 0.144 | 0.134 | -0.057 |
| en-zh | cross | yes | 15 | 0.985 | 0.984 | 0.973 | 0.668 | 0.817 | 0.198 | +0.786 |
| de-fr | same | no | 12 | 0.585 | 0.646 | 0.472 | 0.511 | 0.729 | 0.342 | +0.273 |
| de-ar | cross | no | 0 | 0.060 | 0.055 | 0.039 | 0.074 | 0.118 | 0.114 | -0.057 |
| de-zh | cross | no | 11 | 0.514 | 0.576 | 0.373 | 0.376 | 0.619 | 0.181 | +0.364 |
| fr-ar | cross | no | 0 | 0.062 | 0.061 | 0.047 | 0.062 | 0.112 | 0.111 | -0.050 |
| fr-zh | cross | no | 12 | 0.668 | 0.689 | 0.519 | 0.409 | 0.641 | 0.173 | +0.506 |
| ar-zh | cross | no | 0 | 0.039 | 0.038 | 0.022 | 0.048 | 0.140 | 0.102 | -0.063 |

