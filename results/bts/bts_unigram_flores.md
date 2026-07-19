# BTS (unigram, eval on flores)

## starved

| partner | script | BPB mono | BPB bi | BTS (total) | BTS (lang) |
|---|---|---|---|---|---|
| de | same | - | 0.9864 | - | - |
| fr | same | 0.8869 | 0.9088 | -0.0247 | 0.0464 |
| ar | cross | 0.8150 | 0.8350 | -0.0246 | 0.0260 |
| zh | cross | 1.3491 | 1.2884 | 0.0450 | 0.0450 |

## destarved

| partner | script | BPB mono | BPB bi | BTS (total) | BTS (lang) |
|---|---|---|---|---|---|
| de | same | 0.9467 | 0.9611 | -0.0152 | 0.0298 |
| fr | same | 0.8814 | 0.8827 | -0.0014 | 0.0430 |
| ar | cross | 0.7979 | 0.8063 | -0.0105 | 0.0324 |
| zh | cross | 1.3179 | 1.2664 | 0.0391 | 0.0391 |

## Interaction (same-script penalty - cross-script penalty)

- **bts_matched_total**: penalty(starved)=-0.0350, penalty(destarved)=-0.0226, **interaction=-0.0123**
- **bts_matched_lang**: penalty(starved)=0.0109, penalty(destarved)=0.0007, **interaction=0.0102**

> interaction >> 0  =>  cross-script penalty is a tokenizer-starvation artifact.
> interaction ~ 0  =>  penalty persists under a fair tokenizer (genuine script effect).
