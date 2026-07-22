# Alignment: ar-starved (FLORES+ both, n=2009)

Languages embedded: en, de, fr, ar, zh; trained on: ar.

### raw, ref layer

`lex` is the model-free TF-IDF token-overlap floor; `vs lex` below 0 means the model adds nothing on that pair.

| pair | script | trained | layer | top1 A->B | top1 B->A | mutual-NN | cos margin | CKA | lex | vs lex |
|---|---|---|---|---|---|---|---|---|---|---|
| en-de | same | no | 12 | 0.031 | 0.068 | 0.023 | 0.028 | 0.725 | 0.457 | -0.407 |
| en-fr | same | no | 12 | 0.077 | 0.374 | 0.073 | 0.042 | 0.764 | 0.645 | -0.420 |
| en-ar | cross | no | 12 | 0.450 | 0.851 | 0.404 | 0.056 | 0.785 | 0.133 | +0.517 |
| en-zh | cross | no | 12 | 0.002 | 0.011 | 0.001 | 0.022 | 0.685 | 0.196 | -0.189 |
| de-fr | same | no | 12 | 0.054 | 0.085 | 0.030 | 0.031 | 0.767 | 0.365 | -0.295 |
| de-ar | cross | no | 12 | 0.016 | 0.009 | 0.003 | 0.023 | 0.665 | 0.113 | -0.100 |
| de-zh | cross | no | 12 | 0.005 | 0.019 | 0.003 | 0.023 | 0.652 | 0.179 | -0.167 |
| fr-ar | cross | no | 12 | 0.070 | 0.034 | 0.008 | 0.036 | 0.721 | 0.113 | -0.061 |
| fr-zh | cross | no | 12 | 0.007 | 0.017 | 0.002 | 0.025 | 0.659 | 0.177 | -0.166 |
| ar-zh | cross | no | 12 | 0.001 | 0.009 | 0.000 | 0.020 | 0.634 | 0.106 | -0.100 |

### raw, best layer

`lex` is the model-free TF-IDF token-overlap floor; `vs lex` below 0 means the model adds nothing on that pair.

| pair | script | trained | layer | top1 A->B | top1 B->A | mutual-NN | cos margin | CKA | lex | vs lex |
|---|---|---|---|---|---|---|---|---|---|---|
| en-de | same | no | 0 | 0.112 | 0.110 | 0.071 | 0.093 | 0.182 | 0.457 | -0.346 |
| en-fr | same | no | 1 | 0.333 | 0.139 | 0.131 | 0.062 | 0.325 | 0.645 | -0.409 |
| en-ar | cross | no | 14 | 0.734 | 0.843 | 0.653 | 0.092 | 0.763 | 0.133 | +0.656 |
| en-zh | cross | no | 0 | 0.033 | 0.030 | 0.016 | 0.048 | 0.165 | 0.196 | -0.165 |
| de-fr | same | no | 11 | 0.075 | 0.114 | 0.048 | 0.029 | 0.787 | 0.365 | -0.270 |
| de-ar | cross | no | 0 | 0.015 | 0.035 | 0.011 | 0.024 | 0.117 | 0.113 | -0.088 |
| de-zh | cross | no | 0 | 0.018 | 0.024 | 0.011 | 0.035 | 0.147 | 0.179 | -0.158 |
| fr-ar | cross | no | 14 | 0.043 | 0.023 | 0.009 | 0.043 | 0.641 | 0.113 | -0.080 |
| fr-zh | cross | no | 0 | 0.016 | 0.022 | 0.010 | 0.034 | 0.131 | 0.177 | -0.158 |
| ar-zh | cross | no | 0 | 0.015 | 0.005 | 0.004 | 0.018 | 0.124 | 0.106 | -0.096 |

### centered, ref layer

`lex` is the model-free TF-IDF token-overlap floor; `vs lex` below 0 means the model adds nothing on that pair.

| pair | script | trained | layer | top1 A->B | top1 B->A | mutual-NN | cos margin | CKA | lex | vs lex |
|---|---|---|---|---|---|---|---|---|---|---|
| en-de | same | no | 12 | 0.414 | 0.292 | 0.223 | 0.340 | 0.704 | 0.457 | -0.104 |
| en-fr | same | no | 12 | 0.805 | 0.730 | 0.650 | 0.481 | 0.748 | 0.645 | +0.123 |
| en-ar | cross | no | 12 | 0.951 | 0.965 | 0.924 | 0.563 | 0.776 | 0.133 | +0.825 |
| en-zh | cross | no | 12 | 0.159 | 0.074 | 0.044 | 0.262 | 0.659 | 0.196 | -0.080 |
| de-fr | same | no | 12 | 0.224 | 0.279 | 0.149 | 0.366 | 0.729 | 0.365 | -0.113 |
| de-ar | cross | no | 12 | 0.148 | 0.220 | 0.089 | 0.236 | 0.641 | 0.113 | +0.071 |
| de-zh | cross | no | 12 | 0.062 | 0.041 | 0.017 | 0.291 | 0.615 | 0.179 | -0.127 |
| fr-ar | cross | no | 12 | 0.537 | 0.621 | 0.405 | 0.357 | 0.707 | 0.113 | +0.466 |
| fr-zh | cross | no | 12 | 0.091 | 0.052 | 0.024 | 0.289 | 0.625 | 0.177 | -0.106 |
| ar-zh | cross | no | 12 | 0.109 | 0.046 | 0.023 | 0.205 | 0.608 | 0.106 | -0.028 |

### centered, best layer

`lex` is the model-free TF-IDF token-overlap floor; `vs lex` below 0 means the model adds nothing on that pair.

| pair | script | trained | layer | top1 A->B | top1 B->A | mutual-NN | cos margin | CKA | lex | vs lex |
|---|---|---|---|---|---|---|---|---|---|---|
| en-de | same | no | 8 | 0.410 | 0.409 | 0.350 | 0.241 | 0.446 | 0.457 | -0.048 |
| en-fr | same | no | 14 | 0.772 | 0.763 | 0.672 | 0.458 | 0.712 | 0.645 | +0.122 |
| en-ar | cross | no | 14 | 0.963 | 0.963 | 0.937 | 0.612 | 0.764 | 0.133 | +0.830 |
| en-zh | cross | no | 8 | 0.108 | 0.090 | 0.059 | 0.128 | 0.407 | 0.196 | -0.098 |
| de-fr | same | no | 8 | 0.245 | 0.242 | 0.192 | 0.198 | 0.378 | 0.365 | -0.121 |
| de-ar | cross | no | 14 | 0.161 | 0.222 | 0.114 | 0.207 | 0.517 | 0.113 | +0.078 |
| de-zh | cross | no | 14 | 0.074 | 0.046 | 0.026 | 0.271 | 0.562 | 0.179 | -0.119 |
| fr-ar | cross | no | 14 | 0.590 | 0.613 | 0.482 | 0.344 | 0.644 | 0.113 | +0.488 |
| fr-zh | cross | no | 0 | 0.050 | 0.045 | 0.032 | 0.080 | 0.129 | 0.177 | -0.130 |
| ar-zh | cross | no | 14 | 0.105 | 0.046 | 0.027 | 0.175 | 0.514 | 0.106 | -0.030 |

