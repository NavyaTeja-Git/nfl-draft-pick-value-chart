# Step 4: Evaluating CUMYr5 imputation

Question: for draft classes that have not finished five seasons (2022-2025), how
well can CUMYr5 be predicted from the seasons observed so far?

## Setup

- Data: complete classes 2012-2021, n = 2,549 players.
- CUMYr5: median 8, mean 12.8, sd 14.1, max 85; 17.8% are exactly 0.
- For k = 1..4, predict CUMYr5 using only information through season k.
  k = 4 is the 2022 class's situation, k = 3 is 2023, k = 2 is 2024, k = 1 is 2025.
- Leave-one-draft-class-out cross-validation: fit on nine classes, predict the
  tenth, repeat for all ten. Smoothing is chosen inside each training set.
- Predictions are floored at CUMYrk (AV already earned is kept).
- Code: `src/step4_imputation_eval.py`. Tables: `results/tables/`. Figures 3-5.

| Model | Definition |
|---|---|
| pick_only | GAM: CUMYr5 ~ s(pick) (baseline that ignores early seasons) |
| **gam_schuckers** | **GAM: CUMYr5 ~ s(pick) + s(CUMYrk)** (the planned model) |
| gam_plus | GAM: + s(AVk) + s(CUMGk), i.e. latest season's AV and games so far |

## Results (out of sample)

Mean absolute error (AV points):

| Model | k=1 (2025) | k=2 (2024) | k=3 (2023) | k=4 (2022) |
|---|---|---|---|---|
| pick_only | 8.34 | 8.09 | 7.35 | 5.96 |
| **gam_schuckers** | **6.33** | **4.32** | **2.86** | **1.49** |
| gam_plus | 6.27 | 4.20 | 2.68 | 1.35 |

Root mean squared error:

| Model | k=1 | k=2 | k=3 | k=4 |
|---|---|---|---|---|
| pick_only | 11.11 | 10.61 | 9.37 | 7.64 |
| **gam_schuckers** | **8.81** | **6.13** | **4.21** | **2.34** |
| gam_plus | 8.64 | 6.01 | 4.05 | 2.24 |

Both GAMs are unbiased (mean error within +/-0.01); pick_only overpredicts by up
to 2.4 AV at k = 4 because of the floor.

Examples (gam_plus, each player predicted by a model that never saw his class):

| Player | Actual CUMYr5 | After 1 season | 2 | 3 | 4 |
|---|---|---|---|---|---|
| Patrick Mahomes (2017) | 75 | 16 | 60 | 65 | 70 |
| Josh Allen (2018) | 74 | 33 | 43 | 59 | 66 |
| Justin Herbert (2020) | 70 | 54 | 69 | 68 | 66 |
| Joe Burrow (2020) | 63 | 38 | 54 | 63 | 56 |
| Zach Wilson (2021) | 14 | 31 | 24 | 24 | 16 |
| Trey Lance (2021) | 6 | 26 | 13 | 7 | 7 |

## Findings

1. **Early seasons matter far more than draft position.** Once one season is
   observed, both GAMs beat pick_only; by k = 4 their error is about a quarter
   of pick_only's.
2. **Adding the latest season and games played helps, consistently.**
   gam_plus beats gam_schuckers in 7/10 held-out classes at k = 1 and 10/10 at
   k = 2, 3, 4, with 1%, 3%, 6%, 9% lower MAE. It is used in step 5.
3. **Uncertainty is not constant.** Errors spread out as the prediction grows
   (k = 1: sd 4.4 for predictions 0-3, 13.6 for predictions 25+; figure 5), so
   one fixed error size cannot describe all players. Step 5 accounts for this.
4. **The 2025 class (k = 1) is the hardest** (MAE about 6.3 vs 1.4 for 2022).
