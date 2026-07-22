# Alignment: en-de-starved (FLORES+ both, n=2009)

Languages embedded: en, de, fr, ar, zh; trained on: en, de.

### raw, ref layer

`lex` is the model-free TF-IDF token-overlap floor; `vs lex` below 0 means the model adds nothing on that pair.

| pair | script | trained | layer | top1 A->B | top1 B->A | mutual-NN | cos margin | CKA | lex | vs lex |
|---|---|---|---|---|---|---|---|---|---|---|
| en-de | same | yes | 12 | 0.960 | 0.988 | 0.951 | 0.101 | 0.769 | 0.457 | +0.517 |
| en-fr | same | no | 12 | 0.082 | 0.628 | 0.072 | 0.058 | 0.727 | 0.645 | -0.290 |
| en-ar | cross | no | 12 | 0.000 | 0.003 | 0.000 | 0.021 | 0.646 | 0.133 | -0.131 |
| en-zh | cross | no | 12 | 0.008 | 0.126 | 0.006 | 0.048 | 0.591 | 0.196 | -0.129 |
| de-fr | same | no | 12 | 0.121 | 0.442 | 0.094 | 0.060 | 0.722 | 0.365 | -0.083 |
| de-ar | cross | no | 12 | 0.001 | 0.006 | 0.000 | 0.022 | 0.598 | 0.113 | -0.110 |
| de-zh | cross | no | 12 | 0.013 | 0.192 | 0.006 | 0.051 | 0.588 | 0.179 | -0.076 |
| fr-ar | cross | no | 12 | 0.001 | 0.009 | 0.000 | 0.024 | 0.663 | 0.113 | -0.108 |
| fr-zh | cross | no | 12 | 0.017 | 0.056 | 0.004 | 0.040 | 0.556 | 0.177 | -0.141 |
| ar-zh | cross | no | 12 | 0.000 | 0.007 | 0.000 | 0.020 | 0.505 | 0.106 | -0.102 |

### raw, best layer

`lex` is the model-free TF-IDF token-overlap floor; `vs lex` below 0 means the model adds nothing on that pair.

| pair | script | trained | layer | top1 A->B | top1 B->A | mutual-NN | cos margin | CKA | lex | vs lex |
|---|---|---|---|---|---|---|---|---|---|---|
| en-de | same | yes | 14 | 0.990 | 0.997 | 0.988 | 0.155 | 0.837 | 0.457 | +0.537 |
| en-fr | same | no | 0 | 0.318 | 0.220 | 0.188 | 0.150 | 0.234 | 0.645 | -0.377 |
| en-ar | cross | no | 0 | 0.037 | 0.029 | 0.019 | 0.029 | 0.118 | 0.133 | -0.100 |
| en-zh | cross | no | 0 | 0.109 | 0.042 | 0.033 | 0.072 | 0.167 | 0.196 | -0.121 |
| de-fr | same | no | 13 | 0.160 | 0.533 | 0.125 | 0.073 | 0.729 | 0.365 | -0.019 |
| de-ar | cross | no | 0 | 0.030 | 0.018 | 0.010 | 0.023 | 0.097 | 0.113 | -0.089 |
| de-zh | cross | no | 0 | 0.063 | 0.038 | 0.025 | 0.047 | 0.158 | 0.179 | -0.128 |
| fr-ar | cross | no | 0 | 0.030 | 0.015 | 0.012 | 0.020 | 0.096 | 0.113 | -0.091 |
| fr-zh | cross | no | 0 | 0.045 | 0.032 | 0.022 | 0.045 | 0.146 | 0.177 | -0.139 |
| ar-zh | cross | no | 0 | 0.017 | 0.010 | 0.003 | 0.024 | 0.128 | 0.106 | -0.092 |

### centered, ref layer

`lex` is the model-free TF-IDF token-overlap floor; `vs lex` below 0 means the model adds nothing on that pair.

| pair | script | trained | layer | top1 A->B | top1 B->A | mutual-NN | cos margin | CKA | lex | vs lex |
|---|---|---|---|---|---|---|---|---|---|---|
| en-de | same | yes | 12 | 0.989 | 0.990 | 0.983 | 0.737 | 0.796 | 0.457 | +0.532 |
| en-fr | same | no | 12 | 0.918 | 0.872 | 0.821 | 0.498 | 0.742 | 0.645 | +0.250 |
| en-ar | cross | no | 12 | 0.066 | 0.022 | 0.011 | 0.208 | 0.627 | 0.133 | -0.089 |
| en-zh | cross | no | 12 | 0.841 | 0.851 | 0.752 | 0.440 | 0.635 | 0.196 | +0.650 |
| de-fr | same | no | 12 | 0.889 | 0.835 | 0.770 | 0.471 | 0.734 | 0.365 | +0.497 |
| de-ar | cross | no | 12 | 0.060 | 0.016 | 0.007 | 0.201 | 0.586 | 0.113 | -0.075 |
| de-zh | cross | no | 12 | 0.814 | 0.824 | 0.705 | 0.424 | 0.623 | 0.179 | +0.640 |
| fr-ar | cross | no | 12 | 0.061 | 0.024 | 0.012 | 0.270 | 0.646 | 0.113 | -0.070 |
| fr-zh | cross | no | 12 | 0.556 | 0.628 | 0.417 | 0.389 | 0.580 | 0.177 | +0.415 |
| ar-zh | cross | no | 12 | 0.019 | 0.051 | 0.005 | 0.233 | 0.482 | 0.106 | -0.071 |

### centered, best layer

`lex` is the model-free TF-IDF token-overlap floor; `vs lex` below 0 means the model adds nothing on that pair.

| pair | script | trained | layer | top1 A->B | top1 B->A | mutual-NN | cos margin | CKA | lex | vs lex |
|---|---|---|---|---|---|---|---|---|---|---|
| en-de | same | yes | 16 | 0.996 | 0.998 | 0.995 | 0.584 | 0.855 | 0.457 | +0.540 |
| en-fr | same | no | 13 | 0.937 | 0.912 | 0.876 | 0.517 | 0.747 | 0.645 | +0.279 |
| en-ar | cross | no | 0 | 0.058 | 0.057 | 0.042 | 0.048 | 0.120 | 0.133 | -0.075 |
| en-zh | cross | no | 13 | 0.867 | 0.891 | 0.810 | 0.459 | 0.645 | 0.196 | +0.683 |
| de-fr | same | no | 13 | 0.912 | 0.874 | 0.827 | 0.489 | 0.732 | 0.365 | +0.528 |
| de-ar | cross | no | 0 | 0.049 | 0.039 | 0.029 | 0.039 | 0.098 | 0.113 | -0.069 |
| de-zh | cross | no | 13 | 0.862 | 0.882 | 0.794 | 0.440 | 0.636 | 0.179 | +0.693 |
| fr-ar | cross | no | 0 | 0.040 | 0.036 | 0.027 | 0.039 | 0.095 | 0.113 | -0.075 |
| fr-zh | cross | no | 14 | 0.669 | 0.708 | 0.561 | 0.400 | 0.583 | 0.177 | +0.511 |
| ar-zh | cross | no | 15 | 0.030 | 0.061 | 0.018 | 0.166 | 0.306 | 0.106 | -0.060 |

