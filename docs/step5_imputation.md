# Step 5: Multiple imputation of CUMYr5 for 2022-2025

The 2022-2025 classes have played 4, 3, 2 and 1 seasons, so their CUMYr5 is
not yet known. This step gives every one of those players **100 plausible
values** of CUMYr5 (multiple imputation). Step 6 fits the value curve to each
of the 100 completed datasets and pools the results, so the final bounds
include the uncertainty about these unfinished careers.

Code: `src/step5_impute.py`. Figures: `results/figures/fig6_imputation_coverage.png`,
`results/figures/fig7_imputed_classes_by_round.png`.

## Model

The extended GAM chosen in step 4:

    CUMYr5 ~ s(pick) + s(CUMYrk) + s(AVk) + s(CUMGk)

fit on all 2012-2021 players (n = 2,549), once per k:

| Class | Seasons observed (k) | Players |
|---|---|---|
| 2022 | 4 | 262 |
| 2023 | 3 | 259 |
| 2024 | 2 | 257 |
| 2025 | 1 | 257 |

## How each imputed value is drawn

imputed value = GAM prediction + an error borrowed from a past player

1. **Errors grow with the prediction.** Step 4 showed that errors for players
   predicted to be stars are about three times those for fringe players. So
   the error is taken from one of the 25 past players (same k) whose
   out-of-sample prediction in step 4 was closest. This also keeps the real
   shape of the errors (skewed, with many players who add no more AV).
2. **Whole classes can miss together.** For each of the 100 imputations, all
   errors come from one past draft class chosen at random, so a class that
   ends up better or worse than predicted as a whole is carried along.
3. Values are rounded to whole AV and never fall below the AV already earned.

Random seed 20260924, so the results are reproducible.

## Check on 2012-2021, where the truth is known

Each 2012-2021 class was imputed using only the other nine classes' errors
(200 draws per player), and the intervals were compared with the actual
CUMYr5.

Share of players inside the interval, by predicted CUMYr5:

| k | Interval | 0-3 | 3-8 | 8-15 | 15-25 | 25+ | all |
|---|---|---|---|---|---|---|---|
| 1 | 80% | 0.91 | 0.86 | 0.84 | 0.80 | 0.79 | 0.84 |
| 1 | 95% | 0.97 | 0.97 | 0.97 | 0.95 | 0.91 | 0.96 |
| 2 | 80% | 0.91 | 0.88 | 0.83 | 0.79 | 0.80 | 0.85 |
| 2 | 95% | 0.98 | 0.98 | 0.97 | 0.95 | 0.93 | 0.96 |
| 3 | 80% | 0.93 | 0.88 | 0.84 | 0.83 | 0.78 | 0.86 |
| 3 | 95% | 0.98 | 0.97 | 0.96 | 0.95 | 0.95 | 0.96 |
| 4 | 80% | 0.96 | 0.90 | 0.86 | 0.83 | 0.81 | 0.88 |
| 4 | 95% | 0.98 | 0.98 | 0.97 | 0.96 | 0.94 | 0.97 |

- For good players (predicted 15+), coverage is at or near the target.
- For fringe players (predicted 0-3), intervals cover more than nominal:
  AV is a whole number and many players add nothing more, so values pile up
  at the interval ends. The intervals are conservative there.
- 95% coverage for predicted stars (25+) is 0.91-0.95, slightly below target.

Average 80% interval width: 19.0, 12.4, 7.8 and 3.8 AV for k = 1, 2, 3, 4.

Round averages (what the value curve is built from), share of the 70
class-round averages per k inside the interval: 80% intervals 0.77-0.89,
95% intervals 0.90-0.99.

Caveat: the past errors used in this check come from step 4 models that were
trained with the held-out class included among their nine training classes,
so the check is slightly optimistic.

## Results for 2022-2025

Mean imputed CUMYr5 by round (averaged over the 100 imputations), next to the
2012-2021 actual means:

| Class | R1 | R2 | R3 | R4 | R5 | R6 | R7 |
|---|---|---|---|---|---|---|---|
| 2012-2021 actual | 30.0 | 20.8 | 15.2 | 10.4 | 8.6 | 5.3 | 3.6 |
| 2022 | 29.5 | 22.0 | 13.9 | 12.1 | 11.1 | 5.2 | 6.6 |
| 2023 | 33.3 | 21.0 | 16.1 | 9.1 | 8.1 | 6.8 | 3.5 |
| 2024 | 32.4 | 20.7 | 13.4 | 11.7 | 7.3 | 5.2 | 4.0 |
| 2025 | 31.1 | 21.4 | 13.6 | 13.6 | 8.8 | 5.2 | 4.8 |

The imputed classes follow the historical pattern. 2022 round 7 is high
because of Brock Purdy (pick 262; 46 AV in four seasons), Rasheed Walker and
Isiah Pacheco. Interval widths grow from 2022 to 2025 (figure 7).

## Output files

- `data/processed/cumyr5_imputed_2012_2025.csv`: all 3,584 players with
  `CUMYr5_imputed` (True for 2022-2025), `CUMYr5_observed`, and
  `imp_001` ... `imp_100`. For 2012-2021 every imputation column equals the
  observed CUMYr5, so step 6 treats all 14 classes the same way.
- `data/processed/cumyr5_imputed_summary.csv`: 2022-2025 players with
  prediction, imputation mean, and 10th/50th/90th percentiles.
- `results/tables/step5_calibration_player.csv`,
  `results/tables/step5_calibration_round_means.csv`.
