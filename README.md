# Bounded NFL Draft Pick Value Charts

UNC Charlotte 

(mentor: Dr. Michael Schuckers) 
"Bounded Draft Pick Value Charts"

**Full write-up: [REPORT.md](REPORT.md)**

## Goal

A draft pick value chart that is monotone non-increasing in pick and gives
each pick a credible interval, using Bayesian monotone regression with
Bernstein polynomials. Outcome: CUMYr5, cumulative Approximate Value (AV)
over a player's first five seasons, for players drafted 2012-2025.

## Steps

| Step | What | Status |
|---|---|---|
| 1 | Draft list 2012-2025 (3,584 players) | done |
| 2 | Per-season games and AV for every player | done |
| 3 | Yearly and cumulative AV and games (CUMYr1-5, CUMG1-5) | done |
| 4 | Evaluate how well CUMYr5 can be predicted from early seasons | done |
| 5 | Impute CUMYr5 for 2022-2025 with uncertainty (100 imputations) | done |
| 6 | Bounded value chart: Bayesian monotone curve with credible bounds | done |
| 7 | Comparison with isotonic, monotone XGBoost, splines and the Jimmy Johnson chart | done |

Walk-through of the whole project: [docs/OVERVIEW.md](docs/OVERVIEW.md).
Every step, rule and decision in detail: [docs/METHODS.md](docs/METHODS.md).

## Main files

- **Value chart (picks 1-262, with 80% and 95% bounds):** [results/tables/step6_value_chart.csv](results/tables/step6_value_chart.csv)
- **Step 4 report:** [docs/step4_imputation_eval.md](docs/step4_imputation_eval.md)
- **Step 5 report:** [docs/step5_imputation.md](docs/step5_imputation.md)
- **Step 6 report:** [docs/step6_value_chart.md](docs/step6_value_chart.md)
- **Model comparison and Jimmy Johnson chart:** [results/tables/step7_model_curves.csv](results/tables/step7_model_curves.csv)
- **Hand check against PFR:** [docs/spot_check_results.md](docs/spot_check_results.md)

## Results so far

The bounded value chart: every pick's expected five-year AV, never increasing,
with 80% and 95% credible bounds:

![Value chart](results/figures/fig8_value_chart.png)

Relative to the first pick, the last pick of round 1 is worth about 60
(95% bounds 55-66), far more than the traditional chart's ~20:

![Relative value chart](results/figures/fig9_relative_value_chart.png)

![Versus Jimmy Johnson](results/figures/fig12_vs_jimmy_johnson.png)

Other methods predict players about as well, but only this chart is smooth,
never increasing, and has bounds:

![Model comparison](results/figures/fig11_model_comparison.png)

How the chart was built, step by step:

Five-year AV falls steeply through the first round, then flattens:

![CUMYr5 by pick](results/figures/fig1_cumyr5_vs_pick.png)

Cumulative AV rises every season and the rounds stay in draft order:

![Cumulative AV by round](results/figures/fig2_cumulative_av_by_round.png)

Predicting CUMYr5 from the first k seasons (leave-one-draft-class-out):

![Imputation error by k](results/figures/fig3_cv_error_by_k.png)

![Predicted vs actual](results/figures/fig4_predicted_vs_actual.png)

The prediction error grows with the prediction, which step 5 must account for:

![Error spread](results/figures/fig5_error_spread_by_prediction.png)

Step 5 imputes CUMYr5 for 2022-2025 with errors that grow with the prediction.
Checked on 2012-2021, the intervals cover close to their target:

![Imputation coverage](results/figures/fig6_imputation_coverage.png)

Imputed classes follow the historical pattern, with more uncertainty the fewer
seasons have been played:

![Imputed classes](results/figures/fig7_imputed_classes_by_round.png)

The chart barely changes when the imputed 2022-2025 classes are added:

![Chart comparison](results/figures/fig10_value_chart_comparison.png)

## Repository layout

```
README.md               this overview
REPORT.md               full write-up (paper style)
run_all.py              rebuilds everything, in order
requirements.txt        Python packages
docs/
  OVERVIEW.md           plain-language walk-through
  METHODS.md            every step and decision
  step4_imputation_eval.md
  step5_imputation.md
  step6_value_chart.md
  spot_check_results.md, spot_check_players.csv
src/
  step1_draft.py        combine draft tables -> draft list
  step2_stathead.py     move and check Stathead exports (manual step)
  step3_dataset.py      build the dataset
  step4_imputation_eval.py
  step5_impute.py       100 imputations of CUMYr5 for 2022-2025
  bnmr.py               Bayesian monotone regression (Wilson et al. 2020)
  step6a_simulation_check.py   checks the model on simulated data
  step6_value_chart.py  the bounded value chart
  step7_compare_models.py      other methods and the Jimmy Johnson chart
  make_figures.py
  fetch.py              PFR downloader (not used: PFR blocks scripts)
data/                   how to obtain the data (the data files are not included)
results/
  tables/               step 4-7 tables
  figures/              fig1-fig12
```

## Reproduce

The data come from Pro-Football-Reference.com and Stathead (Sports Reference
LLC) and are **not included** in this repository, because Sports Reference's
terms do not allow redistributing them. To rebuild everything:

1. Save the 2012-2025 draft tables from Pro-Football-Reference to
   `data/raw/draft/<YEAR>.csv` (columns: `data/README_draft_tables.md`).
2. Export the Stathead Player Season Finder queries listed in
   `data/STATHEAD_CHECKLIST.md` (a Stathead subscription is needed) and move
   them in with `py src/step2_stathead.py <YEAR>`.
3. Save the Jimmy Johnson chart to `data/raw/reference/jimmy_johnson_chart.csv`
   (`data/README_jimmy_johnson.md`).
4. Run:

```
py -m pip install -r requirements.txt
py run_all.py
```

`run_all.py` rebuilds steps 1, 3, 4, 5, 6a, 6 and 7 and the figures (about 10
minutes on a 20-core machine). The result tables in `results/tables/` and the
figures are included, so the results can be read without rebuilding.

## Data source and use

Pro-Football-Reference.com and Stathead (Sports Reference LLC), used for
non-commercial academic research.
