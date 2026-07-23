# Tokenizer fertility on FLORES+ (dev+devtest)

## unigram_starved

| lang | bytes_per_token | tokens_per_char | tokens_per_word | tokens_per_sentence | parity_vs_en | pct_byte_tokens | pct_single_char_tokens | unique_tokens_used |
|---|---|---|---|---|---|---|---|---|
| en | 3.647 | 0.274 | 1.647 | 35.132 | 1.000 | 0.000 | 17.914 | 5767 |
| de | 3.350 | 0.303 | 2.136 | 45.361 | 1.291 | 0.007 | 16.445 | 5274 |
| fr | 3.374 | 0.308 | 1.897 | 47.001 | 1.338 | 0.000 | 22.354 | 5285 |
| ar | 4.589 | 0.396 | 2.338 | 44.677 | 1.272 | 0.000 | 25.371 | 2431 |
| zh | 3.066 | 0.909 | 18.892 | 38.649 | 1.100 | 0.062 | 87.217 | 3912 |

## unigram_destarved

| lang | bytes_per_token | tokens_per_char | tokens_per_word | tokens_per_sentence | parity_vs_en | pct_byte_tokens | pct_single_char_tokens | unique_tokens_used |
|---|---|---|---|---|---|---|---|---|
| en | 4.377 | 0.229 | 1.372 | 29.273 | 1.000 | 0.000 | 17.800 | 8119 |
| de | 4.594 | 0.221 | 1.558 | 33.077 | 1.130 | 0.009 | 16.797 | 9282 |
| fr | 4.391 | 0.237 | 1.458 | 36.119 | 1.234 | 0.000 | 23.860 | 8685 |
| ar | 6.771 | 0.268 | 1.584 | 30.276 | 1.034 | 0.000 | 17.397 | 8943 |
| zh | 3.999 | 0.697 | 14.483 | 29.630 | 1.012 | 0.000 | 58.632 | 7311 |

# Gate: starved/destarved token-count ratio (per flavor)

- **unigram**: en=1.200, de=1.371, fr=1.301, ar=1.476, zh=1.304

# Vocab allocation (64k pieces by script)

| tokenizer | Arabic | Armenian | Cyrillic | Devanagari | Ethiopic | Georgian | Greek | Han | Hangul | Hebrew | Kana | Latin | OtherIndic | OtherSEA | OtherScript | Thai | byte_atom | mixed | special | sym_num_space |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| unigram_starved | 3718 | 459 | 6544 | 1717 | 569 | 375 | 621 | 9093 | 1571 | 746 | 265 | 29038 | 2648 | 1247 | 2640 | 254 | 256 | 823 | 4 | 2948 |
| unigram_destarved | 13132 | 29 | 125 | 40 | 1 | 22 | 73 | 17625 | 368 | 29 | 220 | 30561 | 104 | 12 | 344 | 41 | 256 | 8 | 4 | 2542 |
