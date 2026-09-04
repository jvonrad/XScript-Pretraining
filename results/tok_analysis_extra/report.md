# Tokenizer fertility on FLORES+ (dev+devtest)

## unigram_starved

| lang | bytes_per_token | tokens_per_char | tokens_per_word | tokens_per_sentence | parity_vs_en | pct_byte_tokens | pct_single_char_tokens | unique_tokens_used |
|---|---|---|---|---|---|---|---|---|
| en | 3.647 | 0.274 | 1.647 | 35.132 | 1.000 | 0.000 | 17.914 | 5767 |
| de | 3.350 | 0.303 | 2.136 | 45.361 | 1.291 | 0.007 | 16.445 | 5274 |
| fr | 3.375 | 0.308 | 1.896 | 46.976 | 1.337 | 0.000 | 22.366 | 5285 |
| ar | 4.590 | 0.395 | 2.337 | 44.662 | 1.271 | 0.000 | 25.380 | 2431 |
| zh | 3.066 | 0.909 | 18.890 | 38.645 | 1.100 | 0.062 | 87.226 | 3912 |

## unigram_destarved

| lang | bytes_per_token | tokens_per_char | tokens_per_word | tokens_per_sentence | parity_vs_en | pct_byte_tokens | pct_single_char_tokens | unique_tokens_used |
|---|---|---|---|---|---|---|---|---|
| en | 4.377 | 0.229 | 1.372 | 29.273 | 1.000 | 0.000 | 17.800 | 8119 |
| de | 4.594 | 0.221 | 1.558 | 33.077 | 1.130 | 0.009 | 16.797 | 9282 |
| fr | 4.392 | 0.237 | 1.457 | 36.094 | 1.233 | 0.000 | 23.877 | 8685 |
| ar | 6.774 | 0.268 | 1.583 | 30.260 | 1.034 | 0.000 | 17.407 | 8943 |
| zh | 4.000 | 0.697 | 14.482 | 29.626 | 1.012 | 0.000 | 58.640 | 7311 |

## unigram_10lang

| lang | bytes_per_token | tokens_per_char | tokens_per_word | tokens_per_sentence | parity_vs_en | pct_byte_tokens | pct_single_char_tokens | unique_tokens_used |
|---|---|---|---|---|---|---|---|---|
| en | 3.933 | 0.255 | 1.527 | 32.578 | 1.000 | 0.000 | 19.273 | 6464 |
| de | 4.224 | 0.241 | 1.694 | 35.975 | 1.104 | 0.008 | 17.775 | 7381 |
| fr | 4.083 | 0.255 | 1.568 | 38.830 | 1.192 | 0.000 | 24.392 | 7109 |
| ar | 5.755 | 0.315 | 1.864 | 35.619 | 1.093 | 0.000 | 19.961 | 5269 |
| zh | 3.768 | 0.739 | 15.370 | 31.444 | 0.965 | 0.024 | 64.510 | 6171 |

## unigram_20lang

| lang | bytes_per_token | tokens_per_char | tokens_per_word | tokens_per_sentence | parity_vs_en | pct_byte_tokens | pct_single_char_tokens | unique_tokens_used |
|---|---|---|---|---|---|---|---|---|
| en | 3.962 | 0.253 | 1.516 | 32.338 | 1.000 | 0.000 | 19.057 | 6588 |
| de | 3.946 | 0.258 | 1.813 | 38.507 | 1.191 | 0.008 | 17.842 | 6395 |
| fr | 3.853 | 0.270 | 1.661 | 41.144 | 1.272 | 0.000 | 24.332 | 6258 |
| ar | 5.304 | 0.342 | 2.022 | 38.649 | 1.195 | 0.000 | 21.020 | 3815 |
| zh | 3.524 | 0.791 | 16.437 | 33.626 | 1.040 | 0.044 | 71.697 | 5132 |

## unigram_50lang

| lang | bytes_per_token | tokens_per_char | tokens_per_word | tokens_per_sentence | parity_vs_en | pct_byte_tokens | pct_single_char_tokens | unique_tokens_used |
|---|---|---|---|---|---|---|---|---|
| en | 3.920 | 0.255 | 1.532 | 32.686 | 1.000 | 0.000 | 18.693 | 6314 |
| de | 3.545 | 0.287 | 2.019 | 42.866 | 1.311 | 0.000 | 18.446 | 5331 |
| fr | 3.529 | 0.295 | 1.814 | 44.922 | 1.374 | 0.000 | 24.133 | 5320 |
| ar | 4.661 | 0.389 | 2.301 | 43.985 | 1.346 | 0.000 | 25.132 | 2374 |
| zh | 3.221 | 0.865 | 17.984 | 36.791 | 1.126 | 0.085 | 81.404 | 4225 |

## unigram_bi_de

| lang | bytes_per_token | tokens_per_char | tokens_per_word | tokens_per_sentence | parity_vs_en | pct_byte_tokens | pct_single_char_tokens | unique_tokens_used |
|---|---|---|---|---|---|---|---|---|
| en | 4.781 | 0.209 | 1.256 | 26.799 | 1.000 | 0.000 | 16.248 | 9510 |
| de | 5.234 | 0.194 | 1.367 | 29.032 | 1.083 | 0.010 | 15.506 | 11933 |
| fr | 2.917 | 0.356 | 2.194 | 54.341 | 2.028 | 0.000 | 30.482 | 5437 |
| ar | 2.041 | 0.889 | 5.254 | 100.416 | 3.747 | 0.047 | 81.730 | 592 |
| zh | 2.368 | 1.177 | 24.459 | 50.038 | 1.867 | 27.363 | 67.783 | 2353 |

## unigram_bi_fr

| lang | bytes_per_token | tokens_per_char | tokens_per_word | tokens_per_sentence | parity_vs_en | pct_byte_tokens | pct_single_char_tokens | unique_tokens_used |
|---|---|---|---|---|---|---|---|---|
| en | 4.825 | 0.207 | 1.245 | 26.552 | 1.000 | 0.000 | 16.051 | 9648 |
| de | 2.742 | 0.371 | 2.610 | 55.423 | 2.087 | 0.005 | 27.543 | 5453 |
| fr | 4.871 | 0.213 | 1.314 | 32.546 | 1.226 | 0.000 | 22.481 | 10773 |
| ar | 2.403 | 0.755 | 4.464 | 85.318 | 3.213 | 0.006 | 77.167 | 620 |
| zh | 2.546 | 1.094 | 22.745 | 46.533 | 1.753 | 18.135 | 76.646 | 2599 |

## unigram_bi_ar

| lang | bytes_per_token | tokens_per_char | tokens_per_word | tokens_per_sentence | parity_vs_en | pct_byte_tokens | pct_single_char_tokens | unique_tokens_used |
|---|---|---|---|---|---|---|---|---|
| en | 4.712 | 0.212 | 1.275 | 27.190 | 1.000 | 0.000 | 16.269 | 9272 |
| de | 2.596 | 0.392 | 2.757 | 58.541 | 2.153 | 0.005 | 28.442 | 4458 |
| fr | 2.840 | 0.366 | 2.254 | 55.828 | 2.053 | 0.000 | 31.129 | 4783 |
| ar | 7.800 | 0.233 | 1.375 | 26.281 | 0.967 | 0.000 | 15.036 | 12453 |
| zh | 2.418 | 1.152 | 23.949 | 48.996 | 1.802 | 24.523 | 70.447 | 2401 |

## unigram_bi_zh

| lang | bytes_per_token | tokens_per_char | tokens_per_word | tokens_per_sentence | parity_vs_en | pct_byte_tokens | pct_single_char_tokens | unique_tokens_used |
|---|---|---|---|---|---|---|---|---|
| en | 4.658 | 0.215 | 1.289 | 27.505 | 1.000 | 0.000 | 16.624 | 9042 |
| de | 2.501 | 0.406 | 2.861 | 60.759 | 2.209 | 0.005 | 27.705 | 4244 |
| fr | 2.629 | 0.395 | 2.435 | 60.305 | 2.193 | 0.000 | 31.648 | 4543 |
| ar | 1.917 | 0.947 | 5.596 | 106.958 | 3.889 | 0.042 | 76.746 | 574 |
| zh | 4.490 | 0.621 | 12.899 | 26.389 | 0.959 | 0.000 | 47.566 | 9797 |

# Gate: starved/destarved token-count ratio (per flavor; `<tok>/destarved` rows = corroboration tokenizers vs destarved)

- **unigram**: en=1.200, de=1.371, fr=1.301, ar=1.476, zh=1.304
- **unigram_10lang/destarved**: en=1.113, de=1.088, fr=1.076, ar=1.177, zh=1.061
- **unigram_20lang/destarved**: en=1.105, de=1.164, fr=1.140, ar=1.277, zh=1.135
- **unigram_50lang/destarved**: en=1.117, de=1.296, fr=1.245, ar=1.454, zh=1.242
- **unigram_bi_de/destarved**: en=0.915, de=0.878, fr=1.506, ar=3.318, zh=1.689
- **unigram_bi_fr/destarved**: en=0.907, de=1.676, fr=0.902, ar=2.819, zh=1.571
- **unigram_bi_ar/destarved**: en=0.929, de=1.770, fr=1.547, ar=0.869, zh=1.654
- **unigram_bi_zh/destarved**: en=0.940, de=1.837, fr=1.671, ar=3.535, zh=0.891

# Vocab allocation (64k pieces by script)

| tokenizer | Arabic | Armenian | Cyrillic | Devanagari | Ethiopic | Georgian | Greek | Han | Hangul | Hebrew | Kana | Latin | OtherIndic | OtherSEA | OtherScript | Thai | byte_atom | mixed | special | sym_num_space |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| unigram_starved | 3718 | 459 | 6544 | 1717 | 569 | 375 | 621 | 9093 | 1571 | 746 | 265 | 29038 | 2648 | 1247 | 2640 | 254 | 256 | 823 | 4 | 2948 |
| unigram_destarved | 13132 | 29 | 125 | 40 | 1 | 22 | 73 | 17625 | 368 | 29 | 220 | 30561 | 104 | 12 | 344 | 41 | 256 | 8 | 4 | 2542 |
| unigram_10lang | 5740 | 37 | 5204 | 40 | 22 | 17 | 68 | 15350 | 417 | 28 | 2466 | 31436 | 63 | 18 | 342 | 42 | 256 | 1272 | 4 | 2714 |
| unigram_20lang | 5283 | 41 | 5042 | 35 | 65 | 20 | 70 | 11064 | 4412 | 27 | 1475 | 34584 | 48 | 25 | 221 | 43 | 256 | 517 | 4 | 2304 |
| unigram_50lang | 3431 | 39 | 5660 | 2453 | 10 | 929 | 1313 | 8280 | 2712 | 1476 | 751 | 31127 | 3023 | 40 | 263 | 1004 | 256 | 650 | 4 | 2115 |
| unigram_bi_de | 60 | 22 | 144 | 45 | 8 | 20 | 61 | 1173 | 335 | 27 | 136 | 59492 | 109 | 0 | 126 | 47 | 256 | 0 | 4 | 3471 |
| unigram_bi_fr | 86 | 20 | 171 | 44 | 8 | 17 | 67 | 1538 | 367 | 27 | 145 | 59106 | 99 | 0 | 163 | 46 | 256 | 0 | 4 | 3372 |
| unigram_bi_ar | 35184 | 24 | 127 | 44 | 9 | 12 | 61 | 1239 | 378 | 27 | 131 | 24758 | 135 | 0 | 161 | 47 | 256 | 0 | 4 | 2939 |
| unigram_bi_zh | 70 | 39 | 141 | 47 | 19 | 19 | 72 | 35783 | 699 | 27 | 233 | 23033 | 214 | 25 | 898 | 48 | 256 | 8 | 4 | 3901 |
