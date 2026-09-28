# Step 6: The bounded draft pick value chart

The paper's contribution: a draft pick value chart where (1) value never goes
up as the pick number goes up, and (2) every pick's value has credible-interval
bounds.

Code: `src/bnmr.py` (the model), `src/step6a_simulation_check.py` (check on
simulated data), `src/step6_value_chart.py` (the chart).
Tables: `results/tables/step6_*.csv`. Figures 8-10.

## Model: Bayesian monotone regression with Bernstein polynomials

Following Wilson, Tryner, L'Orange and Volckens (2020), "Bayesian nonparametric
monotone regression", *Environmetrics* 31(8), e2642 (arXiv:2006.00326).

- CUMYr5 = f(pick) + error, with f written as a weighted sum of M = 50
  Bernstein polynomials (smooth by construction).
- The weights are rewritten as a starting value plus 50 increments. If every
  increment is >= 0, f is monotone. The paper builds increasing curves; here
  x = 1 - (pick - 1) / 261, so increasing in x means **decreasing in pick**:
  pick 1 is worth at least as much as pick 2, and so on.
- Prior on each increment: exactly 0 with some probability (lets the curve
  flatten), otherwise drawn from a Dirichlet process with a truncated-normal
  base TN[0, inf)(mu, phi^2), which lets increments share values. CUMYr5 is
  standardized before fitting.
- Fit by Gibbs sampling as in the paper (section 2.2): each increment's
  cluster is updated by a Polya urn step, then the intercept and all cluster
  values together as one truncated multivariate normal block, then the error
  variance and the Dirichlet process concentration.
- Implemented in Python (`src/bnmr.py`; the authors' own code is an R
  package). Because f depends on the data only through the pick, the sampler
  works with per-pick totals (count, sum, sum of squares), which is exact and
  fast.

## Choosing the prior: simulation check

The paper's default prior (mu = 0.5, phi = 0.25) expects a few large steps. A
draft value curve falls about 2.5 standard deviations over many small steps,
so several settings were checked on simulated data before fitting the real
chart. Each setting was fit to 20 simulated datasets built on the real picks
(3,584 players), for two known true curves: one shaped like the real data
with noise resampled from real residuals (skewed, like AV), and a smooth
exponential curve with normal noise.

Share of picks where the 95% band contains the true value (all picks):

| Prior | Real-shaped curve | Exponential curve | Error of the mean (AV) |
|---|---|---|---|
| Paper default: M=50, mu=0.5, phi=0.25 | 0.90 | 0.94 | 0.58 / 0.56 |
| **M=50, mu=0.1, phi=0.1 (chosen)** | **0.94** | **0.95** | **0.54 / 0.54** |
| M=50, mu=0.05, phi=0.05 | 0.92 | 0.92 | 0.58 / 0.59 |
| M=20, mu=0.5, phi=0.25 | 0.92 | 0.89 | 0.50 / 0.55 |

The chosen setting has 95% bands that contain the true curve 94-95% of the
time and the smallest error overall. Coverage is a little lower for the first
round (0.91) and highest in the middle rounds. Full table:
`results/tables/step6a_simulation_check.csv`.

An earlier version of the sampler updated the intercept and cluster values one
at a time. They are strongly correlated, so that version mixed slowly and gave
bands that were too narrow. Updating them as one block, as in the paper,
fixed it.

## The chart

Data: all 14 classes, 2012-2025 (3,584 players). For 2022-2025, CUMYr5 is one
of the 100 imputations from step 5. The model is fit to each of the 100
imputed datasets (200 posterior draws each, 400 burn-in sweeps, every 5th
sweep kept, 20 chains in parallel) and the 20,000 draws are pooled. The bounds
therefore include both the uncertainty in the curve and the uncertainty in
the unfinished careers.

| Pick | Value (CUMYr5, AV) | 95% bounds | Relative (pick 1 = 100) | 95% bounds |
|---|---|---|---|---|
| 1 | 39.7 | 36.7 - 42.5 | 100 | |
| 10 | 33.7 | 32.2 - 35.2 | 85 | 80 - 91 |
| 32 | 23.9 | 22.7 - 25.0 | 60 | 55 - 66 |
| 64 | 18.0 | 16.9 - 19.0 | 45 | 41 - 50 |
| 100 | 12.7 | 11.7 - 13.6 | 32 | 29 - 36 |
| 160 | 8.3 | 7.4 - 9.2 | 21 | 18 - 24 |
| 256 | 3.6 | 1.8 - 4.9 | 9 | 5 - 12 |

Full chart for picks 1-262 (mean, median, 80% and 95% bounds, in AV and
relative to pick 1): `results/tables/step6_value_chart.csv`.

- Every one of the 20,000 posterior curves is monotone decreasing.
- The last pick of the first round (32) is worth about 60% of the first pick
  (95% bounds 55-66%). The traditional Jimmy Johnson chart puts it near 20%
  (590 vs 3,000 points), consistent with Schuckers (2011): that chart
  greatly overvalues the top picks.
- Bounds are widest at the very top (few players, large spread among stars)
  and at picks beyond 253, which exist only in some years.

Checks:
- **Convergence:** split R-hat across 20 independent chains is at most 1.008
  for picks 1-224 and 1.011-1.016 for picks 256-262
  (`results/tables/step6_convergence_rhat.csv`).
- **Effect of the imputed classes:** fitting 2012-2021 only gives almost the
  same curve (pick 1: 38.8, pick 32: 23.7, pick 64: 18.1, pick 100: 12.6);
  adding 2022-2025 narrows the average 95% band from 2.40 to 2.13 AV because
  there are more players (figure 10).

## Notes on what the bounds mean

The bounds describe uncertainty about the **average** value of a pick, which
is what a trade chart needs. Individual players at the same pick vary far
more (a pick-40 player can end with 0 or 70 AV; figure 1).
