# Alignment: en-fr-starved (FLORES+ both, n=2009)

Languages embedded: en, de, fr, ar, zh; trained on: en, fr.

### raw, ref layer

`lex` is the model-free TF-IDF token-overlap floor; `vs lex` below 0 means the model adds nothing on that pair.

| pair | script | trained | layer | top1 A->B | top1 B->A | mutual-NN | cos margin | CKA | lex | vs lex |
|---|---|---|---|---|---|---|---|---|---|---|
| en-de | same | no | 12 | 0.107 | 0.165 | 0.072 | 0.046 | 0.665 | 0.457 | -0.321 |
| en-fr | same | yes | 12 | 0.984 | 0.993 | 0.978 | 0.111 | 0.768 | 0.645 | +0.343 |
| en-ar | cross | no | 12 | 0.002 | 0.009 | 0.000 | 0.020 | 0.605 | 0.133 | -0.127 |
| en-zh | cross | no | 12 | 0.011 | 0.412 | 0.009 | 0.059 | 0.615 | 0.196 | +0.015 |
| de-fr | same | no | 12 | 0.136 | 0.131 | 0.066 | 0.045 | 0.638 | 0.365 | -0.231 |
| de-ar | cross | no | 12 | 0.011 | 0.013 | 0.001 | 0.023 | 0.589 | 0.113 | -0.101 |
| de-zh | cross | no | 12 | 0.017 | 0.142 | 0.009 | 0.036 | 0.523 | 0.179 | -0.099 |
| fr-ar | cross | no | 12 | 0.003 | 0.006 | 0.000 | 0.019 | 0.544 | 0.113 | -0.108 |
| fr-zh | cross | no | 12 | 0.013 | 0.531 | 0.007 | 0.059 | 0.624 | 0.177 | +0.095 |
| ar-zh | cross | no | 12 | 0.003 | 0.033 | 0.001 | 0.019 | 0.442 | 0.106 | -0.088 |

### raw, best layer

`lex` is the model-free TF-IDF token-overlap floor; `vs lex` below 0 means the model adds nothing on that pair.

| pair | script | trained | layer | top1 A->B | top1 B->A | mutual-NN | cos margin | CKA | lex | vs lex |
|---|---|---|---|---|---|---|---|---|---|---|
| en-de | same | no | 0 | 0.240 | 0.186 | 0.169 | 0.117 | 0.228 | 0.457 | -0.244 |
| en-fr | same | yes | 14 | 0.995 | 0.999 | 0.993 | 0.151 | 0.851 | 0.645 | +0.351 |
| en-ar | cross | no | 0 | 0.044 | 0.024 | 0.018 | 0.028 | 0.138 | 0.133 | -0.099 |
| en-zh | cross | no | 0 | 0.117 | 0.043 | 0.032 | 0.070 | 0.191 | 0.196 | -0.116 |
| de-fr | same | no | 0 | 0.125 | 0.161 | 0.107 | 0.080 | 0.181 | 0.365 | -0.222 |
| de-ar | cross | no | 0 | 0.030 | 0.018 | 0.014 | 0.023 | 0.112 | 0.113 | -0.089 |
| de-zh | cross | no | 0 | 0.061 | 0.037 | 0.026 | 0.047 | 0.175 | 0.179 | -0.130 |
| fr-ar | cross | no | 0 | 0.034 | 0.022 | 0.017 | 0.021 | 0.116 | 0.113 | -0.085 |
| fr-zh | cross | no | 0 | 0.074 | 0.037 | 0.028 | 0.051 | 0.165 | 0.177 | -0.122 |
| ar-zh | cross | no | 0 | 0.021 | 0.012 | 0.005 | 0.027 | 0.134 | 0.106 | -0.089 |

### centered, ref layer

`lex` is the model-free TF-IDF token-overlap floor; `vs lex` below 0 means the model adds nothing on that pair.

| pair | script | trained | layer | top1 A->B | top1 B->A | mutual-NN | cos margin | CKA | lex | vs lex |
|---|---|---|---|---|---|---|---|---|---|---|
| en-de | same | no | 12 | 0.802 | 0.667 | 0.595 | 0.402 | 0.670 | 0.457 | +0.278 |
| en-fr | same | yes | 12 | 0.996 | 0.994 | 0.990 | 0.776 | 0.793 | 0.645 | +0.350 |
| en-ar | cross | no | 12 | 0.131 | 0.045 | 0.026 | 0.203 | 0.595 | 0.133 | -0.045 |
| en-zh | cross | no | 12 | 0.920 | 0.957 | 0.896 | 0.479 | 0.642 | 0.196 | +0.742 |
| de-fr | same | no | 12 | 0.609 | 0.748 | 0.521 | 0.380 | 0.641 | 0.365 | +0.314 |
| de-ar | cross | no | 12 | 0.085 | 0.054 | 0.020 | 0.277 | 0.581 | 0.113 | -0.044 |
| de-zh | cross | no | 12 | 0.437 | 0.579 | 0.350 | 0.348 | 0.534 | 0.179 | +0.329 |
| fr-ar | cross | no | 12 | 0.125 | 0.044 | 0.024 | 0.193 | 0.546 | 0.113 | -0.029 |
| fr-zh | cross | no | 12 | 0.899 | 0.943 | 0.868 | 0.458 | 0.635 | 0.177 | +0.744 |
| ar-zh | cross | no | 12 | 0.050 | 0.112 | 0.029 | 0.222 | 0.433 | 0.106 | -0.025 |

### centered, best layer

`lex` is the model-free TF-IDF token-overlap floor; `vs lex` below 0 means the model adds nothing on that pair.

| pair | script | trained | layer | top1 A->B | top1 B->A | mutual-NN | cos margin | CKA | lex | vs lex |
|---|---|---|---|---|---|---|---|---|---|---|
| en-de | same | no | 13 | 0.811 | 0.722 | 0.652 | 0.414 | 0.664 | 0.457 | +0.310 |
| en-fr | same | yes | 14 | 0.999 | 0.999 | 0.997 | 0.782 | 0.845 | 0.645 | +0.353 |
| en-ar | cross | no | 0 | 0.069 | 0.054 | 0.043 | 0.044 | 0.140 | 0.133 | -0.072 |
| en-zh | cross | no | 13 | 0.929 | 0.962 | 0.907 | 0.493 | 0.658 | 0.196 | +0.749 |
| de-fr | same | no | 13 | 0.662 | 0.759 | 0.579 | 0.390 | 0.629 | 0.365 | +0.345 |
| de-ar | cross | no | 0 | 0.045 | 0.040 | 0.030 | 0.042 | 0.110 | 0.113 | -0.071 |
| de-zh | cross | no | 13 | 0.500 | 0.595 | 0.400 | 0.366 | 0.529 | 0.179 | +0.369 |
| fr-ar | cross | no | 0 | 0.052 | 0.044 | 0.033 | 0.034 | 0.117 | 0.113 | -0.065 |
| fr-zh | cross | no | 13 | 0.909 | 0.942 | 0.879 | 0.468 | 0.654 | 0.177 | +0.748 |
| ar-zh | cross | no | 13 | 0.063 | 0.120 | 0.037 | 0.218 | 0.382 | 0.106 | -0.014 |

