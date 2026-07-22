# Alignment: de-fair (FLORES+ both, n=2009)

Languages embedded: en, de, fr, ar, zh; trained on: de.

### raw, ref layer

`lex` is the model-free TF-IDF token-overlap floor; `vs lex` below 0 means the model adds nothing on that pair.

| pair | script | trained | layer | top1 A->B | top1 B->A | mutual-NN | cos margin | CKA | lex | vs lex |
|---|---|---|---|---|---|---|---|---|---|---|
| en-de | same | no | 12 | 0.958 | 0.979 | 0.940 | 0.130 | 0.760 | 0.434 | +0.534 |
| en-fr | same | no | 12 | 0.806 | 0.953 | 0.787 | 0.101 | 0.759 | 0.580 | +0.299 |
| en-ar | cross | no | 12 | 0.001 | 0.004 | 0.000 | 0.028 | 0.607 | 0.134 | -0.131 |
| en-zh | cross | no | 12 | 0.026 | 0.125 | 0.021 | 0.068 | 0.562 | 0.198 | -0.123 |
| de-fr | same | no | 12 | 0.670 | 0.801 | 0.577 | 0.097 | 0.725 | 0.342 | +0.393 |
| de-ar | cross | no | 12 | 0.001 | 0.003 | 0.000 | 0.026 | 0.534 | 0.114 | -0.112 |
| de-zh | cross | no | 12 | 0.016 | 0.143 | 0.008 | 0.067 | 0.560 | 0.181 | -0.102 |
| fr-ar | cross | no | 12 | 0.005 | 0.005 | 0.001 | 0.029 | 0.598 | 0.111 | -0.106 |
| fr-zh | cross | no | 12 | 0.043 | 0.151 | 0.031 | 0.061 | 0.532 | 0.173 | -0.076 |
| ar-zh | cross | no | 12 | 0.003 | 0.007 | 0.001 | 0.024 | 0.411 | 0.102 | -0.097 |

### raw, best layer

`lex` is the model-free TF-IDF token-overlap floor; `vs lex` below 0 means the model adds nothing on that pair.

| pair | script | trained | layer | top1 A->B | top1 B->A | mutual-NN | cos margin | CKA | lex | vs lex |
|---|---|---|---|---|---|---|---|---|---|---|
| en-de | same | no | 12 | 0.958 | 0.979 | 0.940 | 0.130 | 0.760 | 0.434 | +0.534 |
| en-fr | same | no | 12 | 0.806 | 0.953 | 0.787 | 0.101 | 0.759 | 0.580 | +0.299 |
| en-ar | cross | no | 0 | 0.061 | 0.047 | 0.037 | 0.046 | 0.128 | 0.134 | -0.080 |
| en-zh | cross | no | 0 | 0.128 | 0.069 | 0.054 | 0.088 | 0.163 | 0.198 | -0.100 |
| de-fr | same | no | 12 | 0.670 | 0.801 | 0.577 | 0.097 | 0.725 | 0.342 | +0.393 |
| de-ar | cross | no | 0 | 0.046 | 0.040 | 0.029 | 0.035 | 0.101 | 0.114 | -0.071 |
| de-zh | cross | no | 0 | 0.114 | 0.057 | 0.042 | 0.076 | 0.146 | 0.181 | -0.095 |
| fr-ar | cross | no | 0 | 0.052 | 0.031 | 0.024 | 0.033 | 0.092 | 0.111 | -0.070 |
| fr-zh | cross | no | 0 | 0.061 | 0.050 | 0.034 | 0.061 | 0.126 | 0.173 | -0.117 |
| ar-zh | cross | no | 0 | 0.032 | 0.015 | 0.010 | 0.036 | 0.118 | 0.102 | -0.079 |

### centered, ref layer

`lex` is the model-free TF-IDF token-overlap floor; `vs lex` below 0 means the model adds nothing on that pair.

| pair | script | trained | layer | top1 A->B | top1 B->A | mutual-NN | cos margin | CKA | lex | vs lex |
|---|---|---|---|---|---|---|---|---|---|---|
| en-de | same | no | 12 | 0.987 | 0.989 | 0.980 | 0.740 | 0.786 | 0.434 | +0.553 |
| en-fr | same | no | 12 | 0.969 | 0.966 | 0.948 | 0.653 | 0.771 | 0.580 | +0.387 |
| en-ar | cross | no | 12 | 0.059 | 0.018 | 0.009 | 0.217 | 0.576 | 0.134 | -0.095 |
| en-zh | cross | no | 12 | 0.811 | 0.818 | 0.723 | 0.482 | 0.599 | 0.198 | +0.616 |
| de-fr | same | no | 12 | 0.947 | 0.933 | 0.898 | 0.581 | 0.741 | 0.342 | +0.598 |
| de-ar | cross | no | 12 | 0.044 | 0.014 | 0.007 | 0.187 | 0.516 | 0.114 | -0.085 |
| de-zh | cross | no | 12 | 0.791 | 0.785 | 0.683 | 0.442 | 0.591 | 0.181 | +0.607 |
| fr-ar | cross | no | 12 | 0.058 | 0.015 | 0.011 | 0.242 | 0.564 | 0.111 | -0.074 |
| fr-zh | cross | no | 12 | 0.695 | 0.719 | 0.578 | 0.458 | 0.562 | 0.173 | +0.534 |
| ar-zh | cross | no | 12 | 0.017 | 0.047 | 0.010 | 0.228 | 0.398 | 0.102 | -0.070 |

### centered, best layer

`lex` is the model-free TF-IDF token-overlap floor; `vs lex` below 0 means the model adds nothing on that pair.

| pair | script | trained | layer | top1 A->B | top1 B->A | mutual-NN | cos margin | CKA | lex | vs lex |
|---|---|---|---|---|---|---|---|---|---|---|
| en-de | same | no | 15 | 0.996 | 0.992 | 0.989 | 0.702 | 0.821 | 0.434 | +0.559 |
| en-fr | same | no | 13 | 0.970 | 0.970 | 0.953 | 0.651 | 0.766 | 0.580 | +0.390 |
| en-ar | cross | no | 0 | 0.081 | 0.074 | 0.056 | 0.068 | 0.129 | 0.134 | -0.056 |
| en-zh | cross | no | 12 | 0.811 | 0.818 | 0.723 | 0.482 | 0.599 | 0.198 | +0.616 |
| de-fr | same | no | 13 | 0.946 | 0.940 | 0.904 | 0.576 | 0.734 | 0.342 | +0.601 |
| de-ar | cross | no | 0 | 0.062 | 0.054 | 0.041 | 0.048 | 0.102 | 0.114 | -0.056 |
| de-zh | cross | no | 13 | 0.775 | 0.790 | 0.685 | 0.435 | 0.593 | 0.181 | +0.602 |
| fr-ar | cross | no | 0 | 0.060 | 0.053 | 0.039 | 0.054 | 0.091 | 0.111 | -0.054 |
| fr-zh | cross | no | 14 | 0.691 | 0.706 | 0.587 | 0.457 | 0.572 | 0.173 | +0.526 |
| ar-zh | cross | no | 0 | 0.042 | 0.035 | 0.023 | 0.061 | 0.117 | 0.102 | -0.064 |

