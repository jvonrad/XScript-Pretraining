# Alignment: en-ar-starved (FLORES+ both, n=2009)

Languages embedded: en, de, fr, ar, zh; trained on: en, ar.

### raw, ref layer

`lex` is the model-free TF-IDF token-overlap floor; `vs lex` below 0 means the model adds nothing on that pair.

| pair | script | trained | layer | top1 A->B | top1 B->A | mutual-NN | cos margin | CKA | lex | vs lex |
|---|---|---|---|---|---|---|---|---|---|---|
| en-de | same | no | 12 | 0.026 | 0.069 | 0.020 | 0.036 | 0.663 | 0.457 | -0.409 |
| en-fr | same | no | 12 | 0.064 | 0.358 | 0.056 | 0.053 | 0.727 | 0.645 | -0.434 |
| en-ar | cross | yes | 12 | 0.942 | 0.990 | 0.936 | 0.095 | 0.809 | 0.133 | +0.833 |
| en-zh | cross | no | 12 | 0.037 | 0.339 | 0.023 | 0.048 | 0.687 | 0.196 | -0.008 |
| de-fr | same | no | 12 | 0.139 | 0.137 | 0.072 | 0.039 | 0.742 | 0.365 | -0.227 |
| de-ar | cross | no | 12 | 0.055 | 0.019 | 0.008 | 0.032 | 0.630 | 0.113 | -0.076 |
| de-zh | cross | no | 12 | 0.033 | 0.029 | 0.005 | 0.029 | 0.614 | 0.179 | -0.148 |
| fr-ar | cross | no | 12 | 0.281 | 0.071 | 0.041 | 0.048 | 0.707 | 0.113 | +0.063 |
| fr-zh | cross | no | 12 | 0.093 | 0.068 | 0.012 | 0.037 | 0.647 | 0.177 | -0.097 |
| ar-zh | cross | no | 12 | 0.059 | 0.261 | 0.016 | 0.046 | 0.676 | 0.106 | +0.054 |

### raw, best layer

`lex` is the model-free TF-IDF token-overlap floor; `vs lex` below 0 means the model adds nothing on that pair.

| pair | script | trained | layer | top1 A->B | top1 B->A | mutual-NN | cos margin | CKA | lex | vs lex |
|---|---|---|---|---|---|---|---|---|---|---|
| en-de | same | no | 0 | 0.206 | 0.158 | 0.131 | 0.117 | 0.192 | 0.457 | -0.275 |
| en-fr | same | no | 1 | 0.309 | 0.191 | 0.174 | 0.087 | 0.279 | 0.645 | -0.395 |
| en-ar | cross | yes | 14 | 0.975 | 0.984 | 0.962 | 0.125 | 0.841 | 0.133 | +0.846 |
| en-zh | cross | no | 0 | 0.107 | 0.064 | 0.041 | 0.078 | 0.167 | 0.196 | -0.111 |
| de-fr | same | no | 11 | 0.223 | 0.245 | 0.128 | 0.039 | 0.761 | 0.365 | -0.131 |
| de-ar | cross | no | 0 | 0.022 | 0.022 | 0.014 | 0.022 | 0.124 | 0.113 | -0.091 |
| de-zh | cross | no | 0 | 0.043 | 0.037 | 0.023 | 0.047 | 0.172 | 0.179 | -0.139 |
| fr-ar | cross | no | 12 | 0.281 | 0.071 | 0.041 | 0.048 | 0.707 | 0.113 | +0.063 |
| fr-zh | cross | no | 0 | 0.043 | 0.031 | 0.022 | 0.049 | 0.147 | 0.177 | -0.140 |
| ar-zh | cross | no | 12 | 0.059 | 0.261 | 0.016 | 0.046 | 0.676 | 0.106 | +0.054 |

### centered, ref layer

`lex` is the model-free TF-IDF token-overlap floor; `vs lex` below 0 means the model adds nothing on that pair.

| pair | script | trained | layer | top1 A->B | top1 B->A | mutual-NN | cos margin | CKA | lex | vs lex |
|---|---|---|---|---|---|---|---|---|---|---|
| en-de | same | no | 12 | 0.605 | 0.482 | 0.399 | 0.328 | 0.654 | 0.457 | +0.087 |
| en-fr | same | no | 12 | 0.866 | 0.809 | 0.745 | 0.437 | 0.718 | 0.645 | +0.192 |
| en-ar | cross | yes | 12 | 0.988 | 0.990 | 0.978 | 0.657 | 0.805 | 0.133 | +0.856 |
| en-zh | cross | no | 12 | 0.892 | 0.754 | 0.703 | 0.438 | 0.688 | 0.196 | +0.627 |
| de-fr | same | no | 12 | 0.405 | 0.469 | 0.306 | 0.406 | 0.708 | 0.365 | +0.072 |
| de-ar | cross | no | 12 | 0.326 | 0.414 | 0.230 | 0.276 | 0.617 | 0.113 | +0.257 |
| de-zh | cross | no | 12 | 0.260 | 0.251 | 0.134 | 0.326 | 0.601 | 0.179 | +0.077 |
| fr-ar | cross | no | 12 | 0.700 | 0.752 | 0.581 | 0.380 | 0.697 | 0.113 | +0.613 |
| fr-zh | cross | no | 12 | 0.512 | 0.441 | 0.294 | 0.380 | 0.636 | 0.177 | +0.299 |
| ar-zh | cross | no | 12 | 0.810 | 0.665 | 0.579 | 0.401 | 0.670 | 0.106 | +0.632 |

### centered, best layer

`lex` is the model-free TF-IDF token-overlap floor; `vs lex` below 0 means the model adds nothing on that pair.

| pair | script | trained | layer | top1 A->B | top1 B->A | mutual-NN | cos margin | CKA | lex | vs lex |
|---|---|---|---|---|---|---|---|---|---|---|
| en-de | same | no | 7 | 0.528 | 0.521 | 0.452 | 0.225 | 0.436 | 0.457 | +0.068 |
| en-fr | same | no | 13 | 0.860 | 0.853 | 0.788 | 0.430 | 0.674 | 0.645 | +0.211 |
| en-ar | cross | yes | 13 | 0.991 | 0.992 | 0.983 | 0.656 | 0.822 | 0.133 | +0.858 |
| en-zh | cross | no | 13 | 0.897 | 0.797 | 0.752 | 0.436 | 0.666 | 0.196 | +0.651 |
| de-fr | same | no | 14 | 0.458 | 0.496 | 0.374 | 0.390 | 0.644 | 0.365 | +0.112 |
| de-ar | cross | no | 14 | 0.372 | 0.412 | 0.274 | 0.254 | 0.495 | 0.113 | +0.278 |
| de-zh | cross | no | 15 | 0.294 | 0.290 | 0.195 | 0.281 | 0.507 | 0.179 | +0.113 |
| fr-ar | cross | no | 13 | 0.762 | 0.768 | 0.660 | 0.373 | 0.652 | 0.113 | +0.652 |
| fr-zh | cross | no | 14 | 0.541 | 0.494 | 0.378 | 0.367 | 0.599 | 0.177 | +0.340 |
| ar-zh | cross | no | 13 | 0.815 | 0.726 | 0.641 | 0.393 | 0.653 | 0.106 | +0.665 |

