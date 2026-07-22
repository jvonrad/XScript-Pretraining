# Alignment: en-zh-starved-23b (FLORES+ both, n=2009)

Languages embedded: en, de, fr, ar, zh; trained on: en, zh.

### raw, ref layer

`lex` is the model-free TF-IDF token-overlap floor; `vs lex` below 0 means the model adds nothing on that pair.

| pair | script | trained | layer | top1 A->B | top1 B->A | mutual-NN | cos margin | CKA | lex | vs lex |
|---|---|---|---|---|---|---|---|---|---|---|
| en-de | same | no | 12 | 0.044 | 0.053 | 0.024 | 0.044 | 0.656 | 0.457 | -0.408 |
| en-fr | same | no | 12 | 0.048 | 0.103 | 0.037 | 0.054 | 0.690 | 0.645 | -0.570 |
| en-ar | cross | no | 12 | 0.002 | 0.002 | 0.000 | 0.024 | 0.580 | 0.133 | -0.131 |
| en-zh | cross | yes | 12 | 0.925 | 0.971 | 0.903 | 0.123 | 0.730 | 0.196 | +0.752 |
| de-fr | same | no | 12 | 0.080 | 0.126 | 0.043 | 0.044 | 0.726 | 0.365 | -0.262 |
| de-ar | cross | no | 12 | 0.002 | 0.005 | 0.000 | 0.027 | 0.605 | 0.113 | -0.110 |
| de-zh | cross | no | 12 | 0.021 | 0.043 | 0.011 | 0.041 | 0.602 | 0.179 | -0.147 |
| fr-ar | cross | no | 12 | 0.005 | 0.004 | 0.000 | 0.028 | 0.631 | 0.113 | -0.109 |
| fr-zh | cross | no | 12 | 0.042 | 0.046 | 0.012 | 0.048 | 0.616 | 0.177 | -0.133 |
| ar-zh | cross | no | 12 | 0.002 | 0.003 | 0.000 | 0.024 | 0.537 | 0.106 | -0.103 |

### raw, best layer

`lex` is the model-free TF-IDF token-overlap floor; `vs lex` below 0 means the model adds nothing on that pair.

| pair | script | trained | layer | top1 A->B | top1 B->A | mutual-NN | cos margin | CKA | lex | vs lex |
|---|---|---|---|---|---|---|---|---|---|---|
| en-de | same | no | 1 | 0.162 | 0.181 | 0.138 | 0.083 | 0.244 | 0.457 | -0.285 |
| en-fr | same | no | 1 | 0.310 | 0.276 | 0.218 | 0.113 | 0.243 | 0.645 | -0.352 |
| en-ar | cross | no | 0 | 0.034 | 0.017 | 0.013 | 0.029 | 0.113 | 0.133 | -0.108 |
| en-zh | cross | yes | 14 | 0.952 | 0.966 | 0.930 | 0.147 | 0.769 | 0.196 | +0.763 |
| de-fr | same | no | 0 | 0.122 | 0.117 | 0.083 | 0.076 | 0.156 | 0.365 | -0.245 |
| de-ar | cross | no | 0 | 0.026 | 0.012 | 0.009 | 0.024 | 0.101 | 0.113 | -0.094 |
| de-zh | cross | no | 1 | 0.048 | 0.028 | 0.021 | 0.028 | 0.210 | 0.179 | -0.141 |
| fr-ar | cross | no | 0 | 0.023 | 0.014 | 0.012 | 0.019 | 0.096 | 0.113 | -0.094 |
| fr-zh | cross | no | 1 | 0.050 | 0.033 | 0.020 | 0.033 | 0.197 | 0.177 | -0.136 |
| ar-zh | cross | no | 0 | 0.015 | 0.006 | 0.003 | 0.013 | 0.106 | 0.106 | -0.095 |

### centered, ref layer

`lex` is the model-free TF-IDF token-overlap floor; `vs lex` below 0 means the model adds nothing on that pair.

| pair | script | trained | layer | top1 A->B | top1 B->A | mutual-NN | cos margin | CKA | lex | vs lex |
|---|---|---|---|---|---|---|---|---|---|---|
| en-de | same | no | 12 | 0.451 | 0.307 | 0.231 | 0.328 | 0.648 | 0.457 | -0.078 |
| en-fr | same | no | 12 | 0.676 | 0.521 | 0.432 | 0.394 | 0.684 | 0.645 | -0.046 |
| en-ar | cross | no | 12 | 0.034 | 0.010 | 0.001 | 0.182 | 0.551 | 0.133 | -0.111 |
| en-zh | cross | yes | 12 | 0.969 | 0.962 | 0.937 | 0.664 | 0.737 | 0.196 | +0.770 |
| de-fr | same | no | 12 | 0.293 | 0.331 | 0.191 | 0.420 | 0.692 | 0.365 | -0.053 |
| de-ar | cross | no | 12 | 0.033 | 0.012 | 0.004 | 0.284 | 0.577 | 0.113 | -0.090 |
| de-zh | cross | no | 12 | 0.206 | 0.298 | 0.129 | 0.291 | 0.594 | 0.179 | +0.073 |
| fr-ar | cross | no | 12 | 0.030 | 0.013 | 0.004 | 0.280 | 0.604 | 0.113 | -0.091 |
| fr-zh | cross | no | 12 | 0.343 | 0.441 | 0.215 | 0.337 | 0.615 | 0.177 | +0.215 |
| ar-zh | cross | no | 12 | 0.009 | 0.022 | 0.003 | 0.182 | 0.510 | 0.106 | -0.090 |

### centered, best layer

`lex` is the model-free TF-IDF token-overlap floor; `vs lex` below 0 means the model adds nothing on that pair.

| pair | script | trained | layer | top1 A->B | top1 B->A | mutual-NN | cos margin | CKA | lex | vs lex |
|---|---|---|---|---|---|---|---|---|---|---|
| en-de | same | no | 4 | 0.486 | 0.472 | 0.420 | 0.205 | 0.358 | 0.457 | +0.022 |
| en-fr | same | no | 4 | 0.628 | 0.607 | 0.549 | 0.260 | 0.353 | 0.645 | -0.028 |
| en-ar | cross | no | 0 | 0.063 | 0.051 | 0.039 | 0.048 | 0.114 | 0.133 | -0.076 |
| en-zh | cross | yes | 16 | 0.983 | 0.983 | 0.970 | 0.579 | 0.796 | 0.196 | +0.787 |
| de-fr | same | no | 16 | 0.322 | 0.336 | 0.251 | 0.257 | 0.402 | 0.365 | -0.036 |
| de-ar | cross | no | 0 | 0.041 | 0.037 | 0.028 | 0.045 | 0.099 | 0.113 | -0.074 |
| de-zh | cross | no | 16 | 0.335 | 0.270 | 0.213 | 0.159 | 0.318 | 0.179 | +0.124 |
| fr-ar | cross | no | 0 | 0.039 | 0.032 | 0.029 | 0.038 | 0.094 | 0.113 | -0.078 |
| fr-zh | cross | no | 16 | 0.484 | 0.379 | 0.317 | 0.197 | 0.401 | 0.177 | +0.254 |
| ar-zh | cross | no | 3 | 0.031 | 0.034 | 0.023 | 0.029 | 0.173 | 0.106 | -0.073 |

