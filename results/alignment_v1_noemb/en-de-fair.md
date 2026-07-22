# Alignment: en-de-fair (FLORES+ both, n=2009)

Languages embedded: en, de, fr, ar, zh; trained on: en, de.

### raw, ref layer

`lex` is the model-free TF-IDF token-overlap floor; `vs lex` below 0 means the model adds nothing on that pair.

| pair | script | trained | layer | top1 A->B | top1 B->A | mutual-NN | cos margin | CKA | lex | vs lex |
|---|---|---|---|---|---|---|---|---|---|---|
| en-de | same | yes | 12 | 0.987 | 0.985 | 0.973 | 0.154 | 0.772 | 0.434 | +0.552 |
| en-fr | same | no | 12 | 0.237 | 0.840 | 0.215 | 0.101 | 0.736 | 0.580 | -0.041 |
| en-ar | cross | no | 12 | 0.004 | 0.005 | 0.000 | 0.030 | 0.580 | 0.134 | -0.129 |
| en-zh | cross | no | 12 | 0.028 | 0.433 | 0.024 | 0.082 | 0.568 | 0.198 | +0.032 |
| de-fr | same | no | 12 | 0.212 | 0.712 | 0.172 | 0.098 | 0.712 | 0.342 | +0.119 |
| de-ar | cross | no | 12 | 0.005 | 0.004 | 0.000 | 0.029 | 0.533 | 0.114 | -0.110 |
| de-zh | cross | no | 12 | 0.015 | 0.421 | 0.012 | 0.082 | 0.575 | 0.181 | +0.037 |
| fr-ar | cross | no | 12 | 0.009 | 0.008 | 0.001 | 0.033 | 0.600 | 0.111 | -0.102 |
| fr-zh | cross | no | 12 | 0.069 | 0.233 | 0.046 | 0.070 | 0.552 | 0.173 | -0.021 |
| ar-zh | cross | no | 12 | 0.002 | 0.008 | 0.001 | 0.027 | 0.422 | 0.102 | -0.097 |

### raw, best layer

`lex` is the model-free TF-IDF token-overlap floor; `vs lex` below 0 means the model adds nothing on that pair.

| pair | script | trained | layer | top1 A->B | top1 B->A | mutual-NN | cos margin | CKA | lex | vs lex |
|---|---|---|---|---|---|---|---|---|---|---|
| en-de | same | yes | 14 | 0.995 | 0.997 | 0.993 | 0.205 | 0.849 | 0.434 | +0.561 |
| en-fr | same | no | 11 | 0.403 | 0.884 | 0.368 | 0.090 | 0.739 | 0.580 | +0.063 |
| en-ar | cross | no | 0 | 0.054 | 0.036 | 0.028 | 0.044 | 0.153 | 0.134 | -0.089 |
| en-zh | cross | no | 0 | 0.192 | 0.050 | 0.041 | 0.091 | 0.159 | 0.198 | -0.078 |
| de-fr | same | no | 11 | 0.381 | 0.788 | 0.320 | 0.088 | 0.709 | 0.342 | +0.242 |
| de-ar | cross | no | 0 | 0.041 | 0.031 | 0.022 | 0.036 | 0.117 | 0.114 | -0.078 |
| de-zh | cross | no | 0 | 0.149 | 0.043 | 0.032 | 0.077 | 0.155 | 0.181 | -0.085 |
| fr-ar | cross | no | 0 | 0.047 | 0.025 | 0.020 | 0.035 | 0.112 | 0.111 | -0.075 |
| fr-zh | cross | no | 12 | 0.069 | 0.233 | 0.046 | 0.070 | 0.552 | 0.173 | -0.021 |
| ar-zh | cross | no | 0 | 0.030 | 0.010 | 0.004 | 0.038 | 0.135 | 0.102 | -0.082 |

### centered, ref layer

`lex` is the model-free TF-IDF token-overlap floor; `vs lex` below 0 means the model adds nothing on that pair.

| pair | script | trained | layer | top1 A->B | top1 B->A | mutual-NN | cos margin | CKA | lex | vs lex |
|---|---|---|---|---|---|---|---|---|---|---|
| en-de | same | yes | 12 | 0.991 | 0.991 | 0.984 | 0.773 | 0.791 | 0.434 | +0.556 |
| en-fr | same | no | 12 | 0.958 | 0.954 | 0.926 | 0.582 | 0.739 | 0.580 | +0.376 |
| en-ar | cross | no | 12 | 0.078 | 0.026 | 0.015 | 0.207 | 0.547 | 0.134 | -0.082 |
| en-zh | cross | no | 12 | 0.878 | 0.904 | 0.831 | 0.495 | 0.600 | 0.198 | +0.693 |
| de-fr | same | no | 12 | 0.944 | 0.927 | 0.891 | 0.549 | 0.723 | 0.342 | +0.593 |
| de-ar | cross | no | 12 | 0.070 | 0.022 | 0.010 | 0.197 | 0.515 | 0.114 | -0.068 |
| de-zh | cross | no | 12 | 0.862 | 0.887 | 0.806 | 0.478 | 0.605 | 0.181 | +0.693 |
| fr-ar | cross | no | 12 | 0.086 | 0.032 | 0.017 | 0.261 | 0.570 | 0.111 | -0.052 |
| fr-zh | cross | no | 12 | 0.741 | 0.794 | 0.659 | 0.472 | 0.577 | 0.173 | +0.595 |
| ar-zh | cross | no | 12 | 0.027 | 0.064 | 0.017 | 0.231 | 0.411 | 0.102 | -0.056 |

### centered, best layer

`lex` is the model-free TF-IDF token-overlap floor; `vs lex` below 0 means the model adds nothing on that pair.

| pair | script | trained | layer | top1 A->B | top1 B->A | mutual-NN | cos margin | CKA | lex | vs lex |
|---|---|---|---|---|---|---|---|---|---|---|
| en-de | same | yes | 14 | 0.997 | 0.998 | 0.996 | 0.780 | 0.848 | 0.434 | +0.563 |
| en-fr | same | no | 12 | 0.958 | 0.954 | 0.926 | 0.582 | 0.739 | 0.580 | +0.376 |
| en-ar | cross | no | 0 | 0.085 | 0.076 | 0.055 | 0.064 | 0.154 | 0.134 | -0.054 |
| en-zh | cross | no | 13 | 0.884 | 0.908 | 0.846 | 0.495 | 0.611 | 0.198 | +0.697 |
| de-fr | same | no | 13 | 0.939 | 0.938 | 0.896 | 0.538 | 0.707 | 0.342 | +0.596 |
| de-ar | cross | no | 0 | 0.064 | 0.061 | 0.041 | 0.053 | 0.117 | 0.114 | -0.052 |
| de-zh | cross | no | 13 | 0.872 | 0.898 | 0.832 | 0.475 | 0.617 | 0.181 | +0.704 |
| fr-ar | cross | no | 0 | 0.069 | 0.063 | 0.047 | 0.061 | 0.110 | 0.111 | -0.045 |
| fr-zh | cross | no | 13 | 0.768 | 0.802 | 0.693 | 0.467 | 0.585 | 0.173 | +0.612 |
| ar-zh | cross | no | 0 | 0.051 | 0.044 | 0.025 | 0.073 | 0.133 | 0.102 | -0.055 |

