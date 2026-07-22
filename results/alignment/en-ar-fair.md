# Alignment: en-ar-fair (FLORES+ both, n=2009)

Languages embedded: en, de, fr, ar, zh; trained on: en, ar.

### raw, ref layer

`lex` is the model-free TF-IDF token-overlap floor; `vs lex` below 0 means the model adds nothing on that pair.

| pair | script | trained | layer | top1 A->B | top1 B->A | mutual-NN | cos margin | CKA | lex | vs lex |
|---|---|---|---|---|---|---|---|---|---|---|
| en-de | same | no | 12 | 0.069 | 0.197 | 0.048 | 0.039 | 0.648 | 0.434 | -0.301 |
| en-fr | same | no | 12 | 0.148 | 0.690 | 0.121 | 0.053 | 0.714 | 0.580 | -0.161 |
| en-ar | cross | yes | 12 | 0.975 | 0.965 | 0.941 | 0.077 | 0.792 | 0.134 | +0.836 |
| en-zh | cross | no | 12 | 0.008 | 0.454 | 0.006 | 0.047 | 0.660 | 0.198 | +0.033 |
| de-fr | same | no | 12 | 0.226 | 0.311 | 0.122 | 0.041 | 0.691 | 0.342 | -0.074 |
| de-ar | cross | no | 12 | 0.123 | 0.030 | 0.011 | 0.033 | 0.601 | 0.114 | -0.038 |
| de-zh | cross | no | 12 | 0.014 | 0.125 | 0.005 | 0.035 | 0.549 | 0.181 | -0.111 |
| fr-ar | cross | no | 12 | 0.517 | 0.069 | 0.046 | 0.047 | 0.682 | 0.111 | +0.182 |
| fr-zh | cross | no | 12 | 0.035 | 0.254 | 0.013 | 0.042 | 0.599 | 0.173 | -0.028 |
| ar-zh | cross | no | 12 | 0.001 | 0.404 | 0.001 | 0.044 | 0.666 | 0.102 | +0.100 |

### raw, best layer

`lex` is the model-free TF-IDF token-overlap floor; `vs lex` below 0 means the model adds nothing on that pair.

| pair | script | trained | layer | top1 A->B | top1 B->A | mutual-NN | cos margin | CKA | lex | vs lex |
|---|---|---|---|---|---|---|---|---|---|---|
| en-de | same | no | 0 | 0.276 | 0.254 | 0.207 | 0.168 | 0.248 | 0.434 | -0.169 |
| en-fr | same | no | 0 | 0.337 | 0.191 | 0.173 | 0.169 | 0.257 | 0.580 | -0.316 |
| en-ar | cross | yes | 11 | 0.975 | 0.975 | 0.951 | 0.065 | 0.780 | 0.134 | +0.841 |
| en-zh | cross | no | 0 | 0.190 | 0.081 | 0.052 | 0.120 | 0.218 | 0.198 | -0.063 |
| de-fr | same | no | 10 | 0.322 | 0.394 | 0.185 | 0.032 | 0.681 | 0.342 | +0.015 |
| de-ar | cross | no | 0 | 0.055 | 0.068 | 0.035 | 0.072 | 0.162 | 0.114 | -0.053 |
| de-zh | cross | no | 0 | 0.090 | 0.046 | 0.033 | 0.077 | 0.188 | 0.181 | -0.113 |
| fr-ar | cross | no | 10 | 0.621 | 0.095 | 0.056 | 0.032 | 0.679 | 0.111 | +0.247 |
| fr-zh | cross | no | 0 | 0.051 | 0.042 | 0.022 | 0.069 | 0.155 | 0.173 | -0.126 |
| ar-zh | cross | no | 0 | 0.095 | 0.028 | 0.017 | 0.082 | 0.204 | 0.102 | -0.041 |

### centered, ref layer

`lex` is the model-free TF-IDF token-overlap floor; `vs lex` below 0 means the model adds nothing on that pair.

| pair | script | trained | layer | top1 A->B | top1 B->A | mutual-NN | cos margin | CKA | lex | vs lex |
|---|---|---|---|---|---|---|---|---|---|---|
| en-de | same | no | 12 | 0.813 | 0.719 | 0.656 | 0.424 | 0.656 | 0.434 | +0.332 |
| en-fr | same | no | 12 | 0.946 | 0.939 | 0.906 | 0.528 | 0.721 | 0.580 | +0.362 |
| en-ar | cross | yes | 12 | 0.991 | 0.989 | 0.982 | 0.700 | 0.791 | 0.134 | +0.856 |
| en-zh | cross | no | 12 | 0.900 | 0.880 | 0.830 | 0.467 | 0.667 | 0.198 | +0.691 |
| de-fr | same | no | 12 | 0.669 | 0.746 | 0.585 | 0.490 | 0.688 | 0.342 | +0.365 |
| de-ar | cross | no | 12 | 0.625 | 0.710 | 0.526 | 0.362 | 0.617 | 0.114 | +0.553 |
| de-zh | cross | no | 12 | 0.493 | 0.546 | 0.376 | 0.407 | 0.566 | 0.181 | +0.339 |
| fr-ar | cross | no | 12 | 0.909 | 0.904 | 0.848 | 0.471 | 0.690 | 0.111 | +0.795 |
| fr-zh | cross | no | 12 | 0.709 | 0.717 | 0.596 | 0.449 | 0.609 | 0.173 | +0.540 |
| ar-zh | cross | no | 12 | 0.874 | 0.858 | 0.792 | 0.430 | 0.668 | 0.102 | +0.764 |

### centered, best layer

`lex` is the model-free TF-IDF token-overlap floor; `vs lex` below 0 means the model adds nothing on that pair.

| pair | script | trained | layer | top1 A->B | top1 B->A | mutual-NN | cos margin | CKA | lex | vs lex |
|---|---|---|---|---|---|---|---|---|---|---|
| en-de | same | no | 11 | 0.831 | 0.728 | 0.668 | 0.424 | 0.672 | 0.434 | +0.345 |
| en-fr | same | no | 11 | 0.951 | 0.941 | 0.909 | 0.527 | 0.728 | 0.580 | +0.366 |
| en-ar | cross | yes | 13 | 0.991 | 0.993 | 0.985 | 0.694 | 0.811 | 0.134 | +0.858 |
| en-zh | cross | no | 12 | 0.900 | 0.880 | 0.830 | 0.467 | 0.667 | 0.198 | +0.691 |
| de-fr | same | no | 12 | 0.669 | 0.746 | 0.585 | 0.490 | 0.688 | 0.342 | +0.365 |
| de-ar | cross | no | 11 | 0.632 | 0.732 | 0.540 | 0.367 | 0.627 | 0.114 | +0.567 |
| de-zh | cross | no | 12 | 0.493 | 0.546 | 0.376 | 0.407 | 0.566 | 0.181 | +0.339 |
| fr-ar | cross | no | 11 | 0.905 | 0.917 | 0.852 | 0.475 | 0.691 | 0.111 | +0.800 |
| fr-zh | cross | no | 13 | 0.704 | 0.715 | 0.605 | 0.455 | 0.595 | 0.173 | +0.537 |
| ar-zh | cross | no | 12 | 0.874 | 0.858 | 0.792 | 0.430 | 0.668 | 0.102 | +0.764 |

