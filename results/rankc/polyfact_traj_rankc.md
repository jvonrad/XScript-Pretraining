
## en-de

| checkpoint | B tok | LR state | bpb en | bpb X | acc en | acc X | RankC [95% CI] |
|---|---|---|---|---|---|---|---|
| en-de-fair-2b | 2 | stable | 1.1955 | 1.0720 | 0.422 | 0.399 | **0.801** [0.788, 0.813] |
| en-de-fair-5b | 5 | stable | 1.1280 | 0.9993 | 0.486 | 0.431 | **0.755** [0.741, 0.767] |
| en-de-fair-10b | 10 | stable | 1.1047 | 0.9738 | 0.502 | 0.465 | **0.767** [0.753, 0.780] |
| en-de-fair-15b | 15 | stable | 1.0939 | 0.9615 | 0.501 | 0.477 | **0.792** [0.780, 0.804] |
| en-de-fair-23b | 23 | stable | 1.0940 | 0.9691 | 0.479 | 0.463 | **0.787** [0.774, 0.800] |
| en-de-fair | 30 | cooled | 1.0353 | 0.9126 | 0.538 | 0.498 | **0.801** [0.789, 0.813] |
| en-de-starved-2b | 2 | stable | 1.2617 | 1.1423 | 0.372 | 0.337 | **0.755** [0.741, 0.769] |
| en-de-starved-5b | 5 | stable | 1.1681 | 1.0428 | 0.455 | 0.420 | **0.738** [0.723, 0.752] |
| en-de-starved-10b | 10 | stable | 1.1483 | 1.0155 | 0.418 | 0.435 | **0.739** [0.725, 0.753] |
| en-de-starved-15b | 15 | stable | 1.1359 | 1.0049 | 0.478 | 0.448 | **0.760** [0.746, 0.773] |
| en-de-starved-23b | 23 | stable | 1.1278 | 0.9980 | 0.480 | 0.464 | **0.789** [0.776, 0.802] |
| en-de-starved | 30 | cooled | 1.0608 | 0.9419 | 0.528 | 0.488 | **0.779** [0.766, 0.792] |

### matched partner-BPB pairs (en-de); Δ = fair − starved, paired over facts

| fair ckpt | starved ckpt | bpb X fair | bpb X starved | Δ bpb | Δ RankC [95% CI] |
|---|---|---|---|---|---|
| en-de-fair-2b | en-de-starved-5b | 1.0720 | 1.0428 | +0.0291 | **+0.0628** [+0.0469, +0.0778]* |
| en-de-fair-2b | en-de-starved-2b | 1.0720 | 1.1423 | -0.0703 | **+0.0462** [+0.0310, +0.0616]* |
| en-de-fair-5b | en-de-starved-23b | 0.9993 | 0.9980 | +0.0012 | **-0.0348** [-0.0497, -0.0189]* |
| en-de-fair-5b | en-de-starved-10b | 0.9993 | 1.0155 | -0.0162 | **+0.0153** [+0.0001, +0.0300]* |
| en-de-fair-5b | en-de-starved-15b | 0.9993 | 1.0049 | -0.0056 | **-0.0053** [-0.0199, +0.0100] |
| en-de-fair-10b | en-de-starved-23b | 0.9738 | 0.9980 | -0.0243 | **-0.0227** [-0.0386, -0.0075]* |
| en-de-fair-15b | en-de-starved-23b | 0.9615 | 0.9980 | -0.0365 | **+0.0029** [-0.0107, +0.0179] |
| en-de-fair-23b | en-de-starved-23b | 0.9691 | 0.9980 | -0.0289 | **-0.0024** [-0.0175, +0.0126] |
| en-de-fair | en-de-starved | 0.9126 | 0.9419 | -0.0293 | **+0.0213** [+0.0065, +0.0360]* |

**Tight matched-loss pairs (mid-stable, |Δbpb_X| ≤ 0.01), n=2: mean Δ RankC (fair − starved) = -0.0200, range -0.0348..-0.0053; pairs: 5b/15b -0.005, 5b/23b -0.035**

Linear fit RankC~bpb_X (mid-stable only): fair slope +0.146/bpb, starved slope -0.093/bpb; **fair − starved at equal BPB over the overlap [0.998, 1.072]: +0.0293** (range +0.0205..+0.0381)

## en-fr

| checkpoint | B tok | LR state | bpb en | bpb X | acc en | acc X | RankC [95% CI] |
|---|---|---|---|---|---|---|---|
| en-fr-fair-2b | 2 | stable | 1.2080 | 0.9889 | 0.405 | 0.399 | **0.801** [0.789, 0.813] |
| en-fr-fair-5b | 5 | stable | 1.1431 | 0.9130 | 0.471 | 0.435 | **0.790** [0.777, 0.803] |
| en-fr-fair-10b | 10 | stable | 1.1041 | 0.8928 | 0.467 | 0.433 | **0.775** [0.762, 0.788] |
| en-fr-fair-15b | 15 | stable | 1.0914 | 0.8830 | 0.477 | 0.470 | **0.784** [0.771, 0.797] |
| en-fr-fair-23b | 23 | stable | 1.0933 | 0.8906 | 0.477 | 0.457 | **0.791** [0.777, 0.803] |
| en-fr-fair | 30 | cooled | 1.0360 | 0.8361 | 0.505 | 0.476 | **0.806** [0.794, 0.819] |
| en-fr-starved-2b | 2 | stable | 1.2517 | 1.0057 | 0.382 | 0.359 | **0.797** [0.784, 0.810] |
| en-fr-starved-5b | 5 | stable | 1.1568 | 0.9402 | 0.435 | 0.381 | **0.754** [0.741, 0.768] |
| en-fr-starved-10b | 10 | stable | 1.1315 | 0.9188 | 0.478 | 0.424 | **0.755** [0.742, 0.770] |
| en-fr-starved-15b | 15 | stable | 1.1250 | 0.9053 | 0.462 | 0.406 | **0.760** [0.747, 0.774] |
| en-fr-starved-23b | 23 | stable | 1.1165 | 0.9006 | 0.457 | 0.410 | **0.745** [0.731, 0.759] |
| en-fr-starved | 30 | cooled | 1.0497 | 0.8526 | 0.521 | 0.462 | **0.776** [0.762, 0.789] |

### matched partner-BPB pairs (en-fr); Δ = fair − starved, paired over facts

| fair ckpt | starved ckpt | bpb X fair | bpb X starved | Δ bpb | Δ RankC [95% CI] |
|---|---|---|---|---|---|
| en-fr-fair-2b | en-fr-starved-2b | 0.9889 | 1.0057 | -0.0168 | **+0.0045** [-0.0103, +0.0197] |
| en-fr-fair-5b | en-fr-starved-10b | 0.9130 | 0.9188 | -0.0057 | **+0.0350** [+0.0194, +0.0509]* |
| en-fr-fair-5b | en-fr-starved-15b | 0.9130 | 0.9053 | +0.0077 | **+0.0297** [+0.0137, +0.0452]* |
| en-fr-fair-5b | en-fr-starved-5b | 0.9130 | 0.9402 | -0.0272 | **+0.0357** [+0.0208, +0.0513]* |
| en-fr-fair-10b | en-fr-starved-23b | 0.8928 | 0.9006 | -0.0078 | **+0.0302** [+0.0152, +0.0457]* |
| en-fr-fair-15b | en-fr-starved-23b | 0.8830 | 0.9006 | -0.0176 | **+0.0388** [+0.0238, +0.0547]* |
| en-fr-fair-23b | en-fr-starved-23b | 0.8906 | 0.9006 | -0.0100 | **+0.0455** [+0.0301, +0.0616]* |
| en-fr-fair | en-fr-starved | 0.8361 | 0.8526 | -0.0165 | **+0.0298** [+0.0150, +0.0455]* |

**Tight matched-loss pairs (mid-stable, |Δbpb_X| ≤ 0.01), n=3: mean Δ RankC (fair − starved) = +0.0316, range +0.0297..+0.0350; pairs: 10b/23b +0.030, 5b/10b +0.035, 5b/15b +0.030**

Linear fit RankC~bpb_X (mid-stable only): fair slope +0.173/bpb, starved slope +0.428/bpb; **fair − starved at equal BPB over the overlap [0.901, 0.989]: +0.0267** (range +0.0154..+0.0380)

## en-ar

| checkpoint | B tok | LR state | bpb en | bpb X | acc en | acc X | RankC [95% CI] |
|---|---|---|---|---|---|---|---|
| en-ar-fair-2b | 2 | stable | 1.2496 | 0.8938 | 0.388 | 0.320 | **0.680** [0.666, 0.696] |
| en-ar-fair-5b | 5 | stable | 1.1631 | 0.8380 | 0.448 | 0.365 | **0.682** [0.668, 0.697] |
| en-ar-fair-10b | 10 | stable | 1.1217 | 0.8280 | 0.465 | 0.363 | **0.681** [0.666, 0.695] |
| en-ar-fair-15b | 15 | stable | 1.1109 | 0.8127 | 0.454 | 0.364 | **0.705** [0.691, 0.720] |
| en-ar-fair-23b | 23 | stable | 1.1005 | 0.8128 | 0.466 | 0.387 | **0.703** [0.688, 0.717] |
| en-ar-fair | 30 | cooled | 1.0421 | 0.7727 | 0.495 | 0.413 | **0.719** [0.705, 0.733] |
| en-ar-starved-2b | 2 | stable | 1.2613 | 0.9228 | 0.363 | 0.307 | **0.665** [0.650, 0.680] |
| en-ar-starved-5b | 5 | stable | 1.1718 | 0.8738 | 0.418 | 0.317 | **0.668** [0.653, 0.683] |
| en-ar-starved-10b | 10 | stable | 1.1565 | 0.8626 | 0.436 | 0.322 | **0.674** [0.660, 0.689] |
| en-ar-starved-15b | 15 | stable | 1.1349 | 0.8438 | 0.462 | 0.349 | **0.692** [0.678, 0.706] |
| en-ar-starved-23b | 23 | stable | 1.1366 | 0.8408 | 0.452 | 0.355 | **0.699** [0.685, 0.714] |
| en-ar-starved | 30 | cooled | 1.0646 | 0.7982 | 0.515 | 0.386 | **0.687** [0.672, 0.702] |

### matched partner-BPB pairs (en-ar); Δ = fair − starved, paired over facts

| fair ckpt | starved ckpt | bpb X fair | bpb X starved | Δ bpb | Δ RankC [95% CI] |
|---|---|---|---|---|---|
| en-ar-fair-2b | en-ar-starved-5b | 0.8938 | 0.8738 | +0.0200 | **+0.0123** [-0.0043, +0.0279] |
| en-ar-fair-2b | en-ar-starved-2b | 0.8938 | 0.9228 | -0.0290 | **+0.0152** [-0.0004, +0.0307] |
| en-ar-fair-5b | en-ar-starved-23b | 0.8380 | 0.8408 | -0.0029 | **-0.0166** [-0.0337, +0.0004] |
| en-ar-fair-5b | en-ar-starved-10b | 0.8380 | 0.8626 | -0.0247 | **+0.0080** [-0.0082, +0.0249] |
| en-ar-fair-5b | en-ar-starved-15b | 0.8380 | 0.8438 | -0.0058 | **-0.0096** [-0.0253, +0.0073] |
| en-ar-fair-10b | en-ar-starved-23b | 0.8280 | 0.8408 | -0.0128 | **-0.0180** [-0.0336, -0.0015]* |
| en-ar-fair-15b | en-ar-starved-23b | 0.8127 | 0.8408 | -0.0281 | **+0.0064** [-0.0098, +0.0211] |
| en-ar-fair-23b | en-ar-starved-23b | 0.8128 | 0.8408 | -0.0281 | **+0.0036** [-0.0118, +0.0194] |
| en-ar-fair | en-ar-starved | 0.7727 | 0.7982 | -0.0255 | **+0.0320** [+0.0166, +0.0480]* |

**Tight matched-loss pairs (mid-stable, |Δbpb_X| ≤ 0.01), n=2: mean Δ RankC (fair − starved) = -0.0131, range -0.0166..-0.0096; pairs: 5b/15b -0.010, 5b/23b -0.017**

Linear fit RankC~bpb_X (mid-stable only): fair slope -0.259/bpb, starved slope -0.379/bpb; **fair − starved at equal BPB over the overlap [0.841, 0.894]: +0.0023** (range -0.0009..+0.0054)

## en-zh

| checkpoint | B tok | LR state | bpb en | bpb X | acc en | acc X | RankC [95% CI] |
|---|---|---|---|---|---|---|---|
| en-zh-fair-2b | 2 | stable | 1.2296 | 1.4140 | 0.383 | 0.305 | **0.638** [0.622, 0.653] |
| en-zh-fair-5b | 5 | stable | 1.1546 | 1.3226 | 0.421 | 0.346 | **0.669** [0.655, 0.683] |
| en-zh-fair-10b | 10 | stable | 1.1139 | 1.2927 | 0.452 | 0.388 | **0.682** [0.667, 0.697] |
| en-zh-fair-15b | 15 | stable | 1.1065 | 1.2846 | 0.452 | 0.363 | **0.713** [0.699, 0.728] |
| en-zh-fair-23b | 23 | stable | 1.0998 | 1.2688 | 0.453 | 0.391 | **0.700** [0.685, 0.715] |
| en-zh-fair | 30 | cooled | 1.0390 | 1.2019 | 0.499 | 0.422 | **0.717** [0.703, 0.731] |
| en-zh-starved-2b | 2 | stable | 1.2790 | 1.4337 | 0.363 | 0.305 | **0.678** [0.663, 0.693] |
| en-zh-starved-5b | 5 | stable | 1.1696 | 1.3547 | 0.422 | 0.346 | **0.655** [0.640, 0.670] |
| en-zh-starved-10b | 10 | stable | 1.1428 | 1.3187 | 0.423 | 0.348 | **0.663** [0.649, 0.678] |
| en-zh-starved-15b | 15 | stable | 1.1366 | 1.3064 | 0.456 | 0.391 | **0.675** [0.660, 0.689] |
| en-zh-starved-23b | 23 | stable | 1.1123 | 1.2948 | 0.452 | 0.371 | **0.680** [0.664, 0.694] |
| en-zh-starved | 30 | cooled | 1.0505 | 1.2229 | 0.497 | 0.401 | **0.705** [0.692, 0.720] |

### matched partner-BPB pairs (en-zh); Δ = fair − starved, paired over facts

| fair ckpt | starved ckpt | bpb X fair | bpb X starved | Δ bpb | Δ RankC [95% CI] |
|---|---|---|---|---|---|
| en-zh-fair-2b | en-zh-starved-2b | 1.4140 | 1.4337 | -0.0198 | **-0.0401** [-0.0555, -0.0228]* |
| en-zh-fair-5b | en-zh-starved-10b | 1.3226 | 1.3187 | +0.0039 | **+0.0058** [-0.0096, +0.0221] |
| en-zh-fair-5b | en-zh-starved-5b | 1.3226 | 1.3547 | -0.0320 | **+0.0139** [-0.0019, +0.0291] |
| en-zh-fair-10b | en-zh-starved-23b | 1.2927 | 1.2948 | -0.0021 | **+0.0025** [-0.0122, +0.0178] |
| en-zh-fair-10b | en-zh-starved-15b | 1.2927 | 1.3064 | -0.0137 | **+0.0071** [-0.0078, +0.0222] |
| en-zh-fair-15b | en-zh-starved-23b | 1.2846 | 1.2948 | -0.0101 | **+0.0337** [+0.0178, +0.0489]* |
| en-zh-fair-23b | en-zh-starved-23b | 1.2688 | 1.2948 | -0.0260 | **+0.0200** [+0.0047, +0.0359]* |
| en-zh-fair | en-zh-starved | 1.2019 | 1.2229 | -0.0210 | **+0.0119** [-0.0028, +0.0266] |

**Tight matched-loss pairs (mid-stable, |Δbpb_X| ≤ 0.01), n=2: mean Δ RankC (fair − starved) = +0.0041, range +0.0025..+0.0058; pairs: 10b/23b +0.002, 5b/10b +0.006**

Linear fit RankC~bpb_X (mid-stable only): fair slope -0.464/bpb, starved slope +0.006/bpb; **fair − starved at equal BPB over the overlap [1.295, 1.414]: -0.0074** (range -0.0354..+0.0206)
