# Alignment: en-zh-starved (FLORES+ both, n=2009)

Languages embedded: en, de, fr, ar, zh; trained on: en, zh.

### raw, ref layer

`lex` is the model-free TF-IDF token-overlap floor; `vs lex` below 0 means the model adds nothing on that pair.

| pair | script | trained | layer | top1 A->B | top1 B->A | mutual-NN | cos margin | CKA | lex | vs lex |
|---|---|---|---|---|---|---|---|---|---|---|
| en-de | same | no | 12 | 0.038 | 0.080 | 0.027 | 0.039 | 0.642 | 0.457 | -0.398 |
| en-fr | same | no | 12 | 0.042 | 0.213 | 0.035 | 0.049 | 0.681 | 0.645 | -0.518 |
| en-ar | cross | no | 12 | 0.001 | 0.002 | 0.000 | 0.020 | 0.590 | 0.133 | -0.132 |
| en-zh | cross | yes | 12 | 0.905 | 0.960 | 0.873 | 0.098 | 0.718 | 0.196 | +0.737 |
| de-fr | same | no | 12 | 0.076 | 0.182 | 0.045 | 0.042 | 0.723 | 0.365 | -0.236 |
| de-ar | cross | no | 12 | 0.006 | 0.006 | 0.001 | 0.025 | 0.610 | 0.113 | -0.107 |
| de-zh | cross | no | 12 | 0.028 | 0.038 | 0.007 | 0.037 | 0.600 | 0.179 | -0.146 |
| fr-ar | cross | no | 12 | 0.005 | 0.005 | 0.000 | 0.026 | 0.639 | 0.113 | -0.108 |
| fr-zh | cross | no | 12 | 0.078 | 0.042 | 0.011 | 0.045 | 0.606 | 0.177 | -0.118 |
| ar-zh | cross | no | 12 | 0.001 | 0.003 | 0.000 | 0.021 | 0.546 | 0.106 | -0.104 |

### raw, best layer

`lex` is the model-free TF-IDF token-overlap floor; `vs lex` below 0 means the model adds nothing on that pair.

| pair | script | trained | layer | top1 A->B | top1 B->A | mutual-NN | cos margin | CKA | lex | vs lex |
|---|---|---|---|---|---|---|---|---|---|---|
| en-de | same | no | 0 | 0.204 | 0.171 | 0.142 | 0.111 | 0.200 | 0.457 | -0.270 |
| en-fr | same | no | 1 | 0.272 | 0.289 | 0.208 | 0.103 | 0.262 | 0.645 | -0.365 |
| en-ar | cross | no | 0 | 0.031 | 0.019 | 0.013 | 0.027 | 0.122 | 0.133 | -0.108 |
| en-zh | cross | yes | 14 | 0.942 | 0.953 | 0.909 | 0.143 | 0.755 | 0.196 | +0.751 |
| de-fr | same | no | 0 | 0.126 | 0.124 | 0.091 | 0.073 | 0.159 | 0.365 | -0.239 |
| de-ar | cross | no | 0 | 0.021 | 0.015 | 0.010 | 0.022 | 0.105 | 0.113 | -0.095 |
| de-zh | cross | no | 0 | 0.039 | 0.030 | 0.019 | 0.032 | 0.163 | 0.179 | -0.144 |
| fr-ar | cross | no | 0 | 0.024 | 0.018 | 0.012 | 0.019 | 0.099 | 0.113 | -0.092 |
| fr-zh | cross | no | 0 | 0.038 | 0.031 | 0.020 | 0.033 | 0.139 | 0.177 | -0.143 |
| ar-zh | cross | no | 0 | 0.011 | 0.007 | 0.003 | 0.013 | 0.112 | 0.106 | -0.097 |

### centered, ref layer

`lex` is the model-free TF-IDF token-overlap floor; `vs lex` below 0 means the model adds nothing on that pair.

| pair | script | trained | layer | top1 A->B | top1 B->A | mutual-NN | cos margin | CKA | lex | vs lex |
|---|---|---|---|---|---|---|---|---|---|---|
| en-de | same | no | 12 | 0.559 | 0.389 | 0.308 | 0.354 | 0.656 | 0.457 | +0.017 |
| en-fr | same | no | 12 | 0.783 | 0.651 | 0.571 | 0.427 | 0.695 | 0.645 | +0.072 |
| en-ar | cross | no | 12 | 0.045 | 0.015 | 0.005 | 0.187 | 0.572 | 0.133 | -0.103 |
| en-zh | cross | yes | 12 | 0.969 | 0.958 | 0.936 | 0.681 | 0.733 | 0.196 | +0.767 |
| de-fr | same | no | 12 | 0.384 | 0.448 | 0.273 | 0.452 | 0.706 | 0.365 | +0.051 |
| de-ar | cross | no | 12 | 0.038 | 0.020 | 0.006 | 0.300 | 0.587 | 0.113 | -0.084 |
| de-zh | cross | no | 12 | 0.277 | 0.383 | 0.180 | 0.318 | 0.606 | 0.179 | +0.151 |
| fr-ar | cross | no | 12 | 0.051 | 0.016 | 0.005 | 0.291 | 0.616 | 0.113 | -0.080 |
| fr-zh | cross | no | 12 | 0.486 | 0.570 | 0.339 | 0.369 | 0.621 | 0.177 | +0.351 |
| ar-zh | cross | no | 12 | 0.011 | 0.033 | 0.004 | 0.188 | 0.529 | 0.106 | -0.083 |

### centered, best layer

`lex` is the model-free TF-IDF token-overlap floor; `vs lex` below 0 means the model adds nothing on that pair.

| pair | script | trained | layer | top1 A->B | top1 B->A | mutual-NN | cos margin | CKA | lex | vs lex |
|---|---|---|---|---|---|---|---|---|---|---|
| en-de | same | no | 3 | 0.484 | 0.466 | 0.416 | 0.196 | 0.325 | 0.457 | +0.018 |
| en-fr | same | no | 13 | 0.771 | 0.646 | 0.572 | 0.425 | 0.684 | 0.645 | +0.063 |
| en-ar | cross | no | 0 | 0.065 | 0.063 | 0.045 | 0.048 | 0.124 | 0.133 | -0.069 |
| en-zh | cross | yes | 16 | 0.977 | 0.984 | 0.966 | 0.579 | 0.783 | 0.196 | +0.784 |
| de-fr | same | no | 14 | 0.398 | 0.463 | 0.305 | 0.437 | 0.696 | 0.365 | +0.065 |
| de-ar | cross | no | 0 | 0.054 | 0.045 | 0.033 | 0.046 | 0.103 | 0.113 | -0.063 |
| de-zh | cross | no | 16 | 0.313 | 0.280 | 0.209 | 0.167 | 0.310 | 0.179 | +0.118 |
| fr-ar | cross | no | 0 | 0.047 | 0.036 | 0.029 | 0.040 | 0.097 | 0.113 | -0.072 |
| fr-zh | cross | no | 12 | 0.486 | 0.570 | 0.339 | 0.369 | 0.621 | 0.177 | +0.351 |
| ar-zh | cross | no | 3 | 0.031 | 0.030 | 0.019 | 0.036 | 0.207 | 0.106 | -0.075 |

