## MuBench-BMLAMA RankC at matched validation loss (fair − starved, paired over items)

| partner | fair ckpt | starved ckpt | Δbpb partner | Δbpb en | variant | RankC fair | RankC starved | Δ RankC [95% CI] |
|---|---|---|---|---|---|---|---|---|
| de | en-de-fair-5b | en-de-starved-23b | +0.0012 | +0.0001 | meanCE | 0.598 | 0.558 | **+0.0402** [+0.0284, +0.0514]* |
| de | en-de-fair-5b | en-de-starved-23b | +0.0012 | +0.0001 | sumWhole | 0.712 | 0.746 | **-0.0342** [-0.0435, -0.0253]* |
| de | en-de-fair-5b | en-de-starved-23b | +0.0012 | +0.0001 | cand | 0.704 | 0.706 | **-0.0017** [-0.0114, +0.0075] |
| de | en-de-fair-5b | en-de-starved-15b | -0.0056 | -0.0079 | meanCE | 0.598 | 0.539 | **+0.0590** [+0.0471, +0.0712]* |
| de | en-de-fair-5b | en-de-starved-15b | -0.0056 | -0.0079 | sumWhole | 0.712 | 0.734 | **-0.0222** [-0.0315, -0.0130]* |
| de | en-de-fair-5b | en-de-starved-15b | -0.0056 | -0.0079 | cand | 0.704 | 0.691 | **+0.0128** [+0.0033, +0.0225]* |
| fr | en-fr-fair-10b | en-fr-starved-23b | -0.0078 | -0.0124 | meanCE | 0.586 | 0.529 | **+0.0565** [+0.0456, +0.0682]* |
| fr | en-fr-fair-10b | en-fr-starved-23b | -0.0078 | -0.0124 | sumWhole | 0.692 | 0.679 | **+0.0129** [+0.0040, +0.0219]* |
| fr | en-fr-fair-10b | en-fr-starved-23b | -0.0078 | -0.0124 | cand | 0.716 | 0.705 | **+0.0112** [+0.0018, +0.0205]* |
| fr | en-fr-fair-5b | en-fr-starved-10b | -0.0057 | +0.0115 | meanCE | 0.586 | 0.524 | **+0.0615** [+0.0500, +0.0729]* |
| fr | en-fr-fair-5b | en-fr-starved-10b | -0.0057 | +0.0115 | sumWhole | 0.677 | 0.665 | **+0.0119** [+0.0028, +0.0211]* |
| fr | en-fr-fair-5b | en-fr-starved-10b | -0.0057 | +0.0115 | cand | 0.711 | 0.667 | **+0.0437** [+0.0343, +0.0538]* |
| ar | en-ar-fair-5b | en-ar-starved-23b | -0.0029 | +0.0264 | meanCE | 0.350 | 0.363 | **-0.0131** [-0.0242, -0.0022]* |
| ar | en-ar-fair-5b | en-ar-starved-23b | -0.0029 | +0.0264 | sumWhole | 0.586 | 0.592 | **-0.0063** [-0.0155, +0.0037] |
| ar | en-ar-fair-5b | en-ar-starved-23b | -0.0029 | +0.0264 | cand | 0.548 | 0.595 | **-0.0461** [-0.0559, -0.0364]* |
| ar | en-ar-fair-10b | en-ar-starved-23b | -0.0128 | -0.0150 | meanCE | 0.358 | 0.363 | **-0.0046** [-0.0157, +0.0059] |
| ar | en-ar-fair-10b | en-ar-starved-23b | -0.0128 | -0.0150 | sumWhole | 0.617 | 0.592 | **+0.0249** [+0.0157, +0.0347]* |
| ar | en-ar-fair-10b | en-ar-starved-23b | -0.0128 | -0.0150 | cand | 0.614 | 0.595 | **+0.0192** [+0.0097, +0.0293]* |
| zh | en-zh-fair-10b | en-zh-starved-23b | -0.0021 | +0.0016 | meanCE | 0.382 | 0.359 | **+0.0226** [+0.0128, +0.0334]* |
| zh | en-zh-fair-10b | en-zh-starved-23b | -0.0021 | +0.0016 | sumWhole | 0.607 | 0.630 | **-0.0230** [-0.0324, -0.0130]* |
| zh | en-zh-fair-10b | en-zh-starved-23b | -0.0021 | +0.0016 | cand | 0.635 | 0.659 | **-0.0240** [-0.0341, -0.0145]* |
| zh | en-zh-fair-5b | en-zh-starved-10b | +0.0039 | +0.0118 | meanCE | 0.353 | 0.341 | **+0.0120** [+0.0015, +0.0229]* |
| zh | en-zh-fair-5b | en-zh-starved-10b | +0.0039 | +0.0118 | sumWhole | 0.576 | 0.586 | **-0.0101** [-0.0196, -0.0010]* |
| zh | en-zh-fair-5b | en-zh-starved-10b | +0.0039 | +0.0118 | cand | 0.584 | 0.557 | **+0.0277** [+0.0177, +0.0375]* |
