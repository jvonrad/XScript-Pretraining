# BTS (unigram, eval on holdout)

## starved

| partner | script | BPB mono | BPB bi | BTS (total) | BTS (lang) |
|---|---|---|---|---|---|
| de | same | - | 0.8794 | - | - |
| fr | same | 0.7588 | 0.7692 | -0.0138 | 0.0809 |
| ar | cross | 0.6751 | 0.6959 | -0.0308 | 0.0657 |
| zh | cross | 1.1666 | 1.0848 | 0.0701 | 0.0701 |

## destarved

| partner | script | BPB mono | BPB bi | BTS (total) | BTS (lang) |
|---|---|---|---|---|---|
| de | same | 0.8170 | 0.8354 | -0.0225 | 0.0661 |
| fr | same | 0.7360 | 0.7458 | -0.0133 | 0.0758 |
| ar | cross | 0.6426 | 0.6620 | -0.0303 | 0.0647 |
| zh | cross | 1.1335 | 1.0610 | 0.0639 | 0.0639 |

## Interaction (same-script penalty - cross-script penalty)

- **bts_matched_total**: penalty(starved)=-0.0334, penalty(destarved)=-0.0347, **interaction=0.0013**
- **bts_matched_lang**: penalty(starved)=0.0130, penalty(destarved)=0.0066, **interaction=0.0064**

> interaction >> 0  =>  cross-script penalty is a tokenizer-starvation artifact.
> interaction ~ 0  =>  penalty persists under a fair tokenizer (genuine script effect).
