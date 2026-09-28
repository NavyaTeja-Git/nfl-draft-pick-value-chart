# Methods

This document records every step of the project in order: where the data came
from, every rule and decision applied to it, how it was checked, and how each
analysis was run. Scripts are in `src/`, numbered by step.

## Overview

| Step | What | Script | Output |
|---|---|---|---|
| 1 | Draft list, 2012-2025 | `src/step1_draft.py` | `data/processed/draft_picks_2012_2025.csv` |
| 2 | Per-season games and AV (Stathead exports) | `src/step2_stathead.py` (manual) | `data/raw/stathead/*.xls` |
| 3 | Yearly and cumulative AV and games | `src/step3_dataset.py` | `data/processed/nfl_draft_cumulative_2012_2025.csv` (+ .xlsx, `by_year/`) |
| 4 | Evaluate CUMYr5 imputation | `src/step4_imputation_eval.py` | `results/tables/`, `docs/step4_imputation_eval.md` |
| 5 | Impute CUMYr5 for 2022-2025 (100 imputations) | `src/step5_impute.py` | `data/processed/cumyr5_imputed_2012_2025.csv`, `docs/step5_imputation.md` |
| 6a | Check the Bayesian model on simulated data; choose the prior | `src/step6a_simulation_check.py` | `results/tables/step6a_simulation_check.csv` |
| 6 | Bounded value chart (Bayesian monotone regression) | `src/step6_value_chart.py` (model: `src/bnmr.py`) | `results/tables/step6_value_chart.csv`, `docs/step6_value_chart.md` |
| 7 | Comparison with other methods and the Jimmy Johnson chart | `src/step7_compare_models.py` | `results/tables/step7_*.csv` |
| - | Figures | `src/make_figures.py` | `results/figures/` |

The full write-up is `REPORT.md`.

`py run_all.py` rebuilds steps 1, 3, 4, 5, 6a, 6, 7 and the figures from `data/raw/`.

## Definitions

- **AV**: Pro-Football-Reference's Approximate Value for one regular season.
- **Season k** (k = 1..5): the season whose year minus the draft year is k - 1.
  For a 2017 draftee, season 1 is 2017 and season 5 is 2021.
- **AVk, Gk**: AV and regular-season games in season k.
- **CUMYrk** = AV1 + ... + AVk. **CUMGk** = G1 + ... + Gk.
- **CUMYr5** is the outcome for the value chart.
- **seasons_observed** = min(5, 2025 - draft year + 1). The last completed
  season is 2025.

## Step 1: Draft list

Source: the `drafts` table on
`https://www.pro-football-reference.com/years/<YEAR>/draft.htm`, 2012-2025
(main draft only; supplemental-draft picks are not in this table).

Collection: Pro-Football-Reference (PFR) now answers automated requests with a
Cloudflare JavaScript challenge (HTTP 403), so the pages cannot be downloaded
by a script (`src/fetch.py` was written for that and is not used). The 14
draft pages were opened in a normal browser, one about every 8 seconds, and
the table rows were saved to `data/raw/draft/<YEAR>.csv` together with each
player's profile URL. Column definitions: `data/raw/draft/README.md`.

`src/step1_draft.py` combines the 14 files and stops with an error unless:
- each year's picks run 1..max with no gaps or duplicates,
- rounds are in pick order,
- each year has 253-262 picks,
- no player URL appears twice.

Result: 3,584 players, 253-262 per year, every player with a profile link.

## Step 2: Per-season games and AV

A player's per-season AV is on his PFR player page, about 3,600 pages in all,
which cannot be scraped (see step 1). The same data is available in
**Stathead Football** (Sports Reference's query tool) through the Player
Season Finder, with an "Excel workbook" export.

Query: single regular seasons; players drafted in year Y; seasons Y to
min(Y + 4, 2025); sorted by AV. Exact links: `data/raw/stathead/CHECKLIST.md`.

- 2012-2017 were exported as 200-row result pages.
- A page break can fall inside a group of rows tied on AV, and tied rows can
  change order between page requests. That makes one row appear on two pages
  and another row appear on neither. This happened once in 2018 (James Daniels
  2021 appeared twice and his 2019 season was missing). Because the pages are
  separate requests, the number of repeated rows equals the number of missing
  rows; 2012-2017 had no repeats and are therefore complete.
- From 2018 on, each draft round was exported as its own query. One round's
  first five seasons fit on a single page (at most 181 rows), so nothing can be
  skipped.

Each export includes the PFR player ID for every row. `src/step2_stathead.py`
moves a class's downloads into `data/raw/stathead/`, names them
`<YEAR>_p<N>.xls` (pages) or `<YEAR>_r<ROUND>.xls` (rounds), and checks that
every row belongs to that class and its five-season window, with no repeated
player-season.

Three rows (Cobi Hamilton 2014, Shelton Gibson 2019, Phil Haynes 2019) are
roster seasons with 0 games and a blank AV; they are set to AV 0.

## Step 3: Dataset

`src/step3_dataset.py` joins the Stathead rows to the draft list by PFR player
ID and builds one row per drafted player.

Rules:
1. **Played season with no row = 0.** If season k has been played
   (year <= 2025) and the player has no row for it, AVk = 0 and Gk = 0.
2. **Unplayed season = blank.** If season k is after 2025, AVk, Gk, CUMYrk and
   CUMGk are blank. A 2025 draftee has only season 1.
3. **Traded players use the season total.** Stathead returns one row per
   player-season, with the team shown as, e.g., "CHI,BAL"; its AV and G equal
   PFR's "2TM" total row. Nothing is averaged. (An earlier version of this
   data averaged team rows; e.g. Gareon Conley 2019 was 2.67 instead of 4.)
4. **No -1 codes.** Players who never played have AV 0 and G 0; games played
   separate "played with 0 AV" from "never played".
5. **Negative AV is kept.** PFR gives a few very poor seasons negative AV
   (5 in this data, e.g. Jared Goff 2016 = -2, Ryan Lindley 2012 = -5). These
   are kept as reported, so CUMYr can dip in those seasons.
6. **Supplemental-draft picks are excluded.** Stathead counts them as drafted
   (Josh Gordon 2012, Adonis Alexander and Sam Beal 2018, Jalen Thompson
   2019), but they are not part of the main draft list.
7. **Regular season only.** Games and AV are regular-season values.

Automated checks (the script reports any failure):
- picks 1..max in every year;
- CUMYr1..CUMYr(seasons_observed) filled and later CUMYr blank, for every year;
- CUMYr never decreases except in a negative-AV season;
- no season has more than 17 games;
- Stathead's draft year and pick agree with the draft list for every row;
- for every player whose career ended inside his five-season window, career
  games on the draft page equal our cumulative games. Two players differ
  (Kerwynn Williams 40 vs 39, Thakarius Keyes 18 vs 13); in both, the
  player's PFR regular-season table sums to our number, and the career
  summary counts something the season table does not (likely playoff games).

Twelve players have games on the draft page but no season in their window:
eleven debuted in 2026 (the season in progress when the data was collected)
and Bennett Jackson (2014) first played in 2019, after his fifth season.

Hand check: 11 players were compared season by season with their PFR pages,
chosen to cover traded seasons, players who never played, missed seasons and
a 2025 rookie. All 11 match. Details: `docs/spot_check_results.md`.

Outputs:
- `data/processed/nfl_draft_cumulative_2012_2025.csv`, columns
  `name, url, year_drafted, round, pick, team, position, college, AV1..AV5,
  G1..G5, CUMYr1..CUMYr5, CUMG1..CUMG5, seasons_observed`
- the same as an Excel workbook (sheet "All" and one sheet per year)
- `data/processed/by_year/<YEAR>_NFL_Draft_Cumulative_AV.csv`

Summary (classes 2012-2021): mean CUMYr5 by round is 30.0, 20.8, 15.2, 10.4,
8.6, 5.3, 3.6 (rounds 1-7). Figures 1 and 2 in `results/figures/`.

## Step 4: Evaluating CUMYr5 imputation

Classes 2022-2025 have not played five seasons, so their CUMYr5 must be
predicted from the seasons observed so far (k = 4, 3, 2, 1 seasons). This step
measures how accurate that prediction is, using the complete classes
2012-2021 (2,549 players), before it is used.

- For k = 1..4, predict CUMYr5 from information through season k.
- **Leave-one-draft-class-out cross-validation:** fit on nine classes, predict
  the tenth, repeat for all ten. Smoothing parameters are chosen inside each
  training set.
- Models (GAMs fit with `pygam`): pick only `CUMYr5 ~ s(pick)` as a baseline;
  the planned GAM `CUMYr5 ~ s(pick) + s(CUMYrk)`; and an extended GAM that
  adds the latest season's AV and games played so far,
  `+ s(AVk) + s(CUMGk)`.
- Predictions are floored at CUMYrk.
- Measures: MAE, RMSE, bias; MAE by held-out class.

Main results (MAE, AV points; k = 1, 2, 3, 4):
- pick only: 8.34, 8.09, 7.35, 5.96
- GAM s(pick) + s(CUMYrk): 6.33, 4.32, 2.86, 1.49
- GAM + latest AV + games: 6.27, 4.20, 2.68, 1.35 (better than the planned GAM
  in 7 of 10 held-out classes for k = 1 and 10 of 10 for k = 2, 3, 4)

The error spread grows with the prediction (k = 1: sd 4.4 for predictions
0-3, 13.6 for predictions 25+), so one fixed error size cannot describe all
players. Step 5 models the spread as a function of the prediction.

Full report: `docs/step4_imputation_eval.md`. Figures 3-5.

Decision (2026-09-24): step 5 uses the extended GAM
(`s(pick) + s(CUMYrk) + s(AVk) + s(CUMGk)`), because it was more accurate than
the planned GAM in every held-out class for k >= 2.

## Step 5: Multiple imputation of CUMYr5 for 2022-2025

Each 2022-2025 player gets 100 imputed values of CUMYr5 so that the
uncertainty about unfinished careers carries into the final bounds.

- Model: the extended GAM, fit on all 2012-2021 players, once per k
  (2022: k = 4, 2023: k = 3, 2024: k = 2, 2025: k = 1).
- Imputed value = prediction + an error borrowed from one of the 25 past
  players (same k) with the closest out-of-sample prediction in step 4. This
  makes the spread grow with the prediction and keeps the real, skewed shape
  of the errors.
- For each imputation, all errors come from one randomly chosen past class,
  so a class-wide miss is carried along rather than averaged away.
- Values are rounded to whole AV and floored at AV already earned.
  Seed 20260924.

Check on 2012-2021 (each class imputed from the other nine): 80% intervals
cover 78-83% for players predicted 15+ and 84-88% overall (conservative for
fringe players because AV is whole-numbered and piles up at 0 added AV); 95%
intervals cover 96-97% overall. For round averages, 95% intervals cover
90-99% of class-round averages.

Imputed round averages for 2022-2025 follow the 2012-2021 pattern (round 1
about 30-33 vs 30.0). Full report: `docs/step5_imputation.md`. Figures 6-7.

Output: `data/processed/cumyr5_imputed_2012_2025.csv` holds all 3,584 players
with `imp_001`...`imp_100` (2012-2021 rows repeat the observed value), ready
for step 6.

## Step 6: Bounded value chart

Model: Bayesian nonparametric monotone regression with Bernstein polynomials
(Wilson, Tryner, L'Orange and Volckens 2020, *Environmetrics* 31(8), e2642),
implemented in Python in `src/bnmr.py`:
- CUMYr5 = f(pick) + normal error; f is a weighted sum of M = 50 Bernstein
  polynomials, written as an intercept plus 50 increments. Increments >= 0
  make f monotone; using x = 1 - (pick - 1) / 261 makes it **decreasing in
  pick**.
- Each increment is 0 with some probability, otherwise drawn from a Dirichlet
  process with base TN[0, inf)(mu, phi^2); CUMYr5 is standardized.
- Gibbs sampler as in the paper: Polya urn update of each increment's
  cluster, then intercept and cluster values as one truncated multivariate
  normal block, then error variance and concentration parameter. The data
  enter only through per-pick totals.

Prior choice (step 6a, `src/step6a_simulation_check.py`): four settings were
fit to 20 simulated datasets each, for two known true curves (one shaped like
the real data with real, skewed residuals; one exponential with normal
noise). The paper's default (mu = 0.5, phi = 0.25) gave 95% bands covering
90% / 94% of the true curve; **mu = 0.1, phi = 0.1 with M = 50** gave 94% /
95% and the smallest error, and is used.

Fit (`src/step6_value_chart.py`): the model is fit to each of the 100 imputed
datasets from step 5 (400 burn-in sweeps, 200 draws kept from 1,000 sweeps,
20 parallel chains, warm starts) and the 20,000 posterior draws are pooled,
so the bounds include imputation uncertainty. A comparison fit uses the
complete classes 2012-2021 only.

Results: pick 1 = 39.7 AV (95% bounds 36.7-42.5), pick 32 = 23.9
(22.7-25.0), pick 64 = 18.0 (16.9-19.0), pick 100 = 12.7 (11.7-13.6),
pick 256 = 3.6 (1.8-4.9). Relative to pick 1, pick 32 is worth 60 (55-66),
far above the Jimmy Johnson chart's ~20. All 20,000 posterior curves are
monotone. Split R-hat across chains is at most 1.008 for picks 1-224 (1.016
at pick 262). The 2012-2021-only fit gives almost the same curve.

Full report: `docs/step6_value_chart.md`. Figures 8-10.

## Step 7: Comparison with other methods

`src/step7_compare_models.py` fits, to the same data, isotonic regression
(non-increasing), XGBoost with a decreasing monotone constraint, a penalized
spline with a decreasing constraint (`pygam`), a cubic regression spline with a
knot every 32 picks (unconstrained), and the plain average at each pick. Each is
scored, like the Bayesian curve, by leave-one-draft-class-out prediction of
individual CUMYr5 on 2012-2021; the final curves use all 14 classes with the
2022-2025 imputation means.

Out-of-sample MAE: Bayesian 8.35, isotonic 8.30, XGBoost 8.27, monotone spline
8.36, 32-pick spline 8.35, per-pick average 8.54. The smoothed and monotone
methods are practically tied; the 32-pick spline rises at 5 picks and turns up
sharply at the last picks; only the Bayesian curve is smooth, monotone and has
bounds.

Jimmy Johnson chart (`data/raw/reference/`, from DraftTek, picks 1-224): on a
pick 1 = 100 scale it gives pick 32 about 20, pick 64 about 9 and pick 100
about 3; the Bayesian chart gives 60 (55-66), 45 (41-50) and 32 (29-36).

The first overall pick averaged 54.1 AV in 2012-2021 (n = 10, standard error
2.8) against 35.4 for picks 2-10. The smooth chart gives 39.7 (M = 200 gives
the same), while isotonic regression and XGBoost, which can jump at one pick,
give 51-54. The report treats pick 1 as a special case.

## Software

Python 3.13 with numpy, pandas, scipy, pygam, scikit-learn, xgboost and matplotlib (plus lxml and
openpyxl for reading and writing files; `requirements.txt`).

## Data use

Data are from Pro-Football-Reference.com and Stathead (Sports Reference LLC)
and are used for non-commercial academic research. The data files are not
included in the public repository; `README.md` explains how to obtain them.
