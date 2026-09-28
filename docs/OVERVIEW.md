# Project overview

## The question

NFL teams trade draft picks, so they need to know what each pick is worth.
The usual "value charts" give one number per pick, with no sense of how sure
that number is. This project builds a chart where:

1. a pick is never worth more than an earlier pick, and
2. every pick's value comes with a range (credible bounds), which no published
   chart has had.

Value is measured by **CUMYr5**: a player's total Approximate Value (AV, a
single-number rating from Pro-Football-Reference) over his first five seasons.

## Step 1: Who was drafted

We listed every player drafted from 2012 to 2025: 3,584 players, 253-262 per
year, with round, pick, team, position and college. The draft pages were saved
from Pro-Football-Reference in a browser, because the site now blocks
automated downloads. A script checks that no pick is missing or repeated.

## Step 2: How each player did, season by season

For every player we needed AV and games played in each of his first five
seasons. That came from Stathead, Sports Reference's own query tool, exported
class by class. One problem came up: when results span several pages, tied
rows can shift between pages, so one row shows twice and another disappears.
We caught this with a check, and from 2018 on exported one draft round at a
time so every export fits on one page.

## Step 3: The dataset

We joined the two, one row per player, with AV and games for years 1-5 and
the running totals CUMYr1-CUMYr5. Rules: a season a player missed counts as 0;
a season that hasn't happened yet (for recent draftees) is left blank; a
traded player gets his full-season total, not an average of his teams. We
compared 11 players by hand against their Pro-Football-Reference pages, and
all 11 matched.

**Result:** on average a first-round pick earns 30 AV in five years, a
seventh-rounder 3.6 (figures 1-2).

## Step 4: Can we predict CUMYr5 for players who haven't finished five years?

Players drafted 2022-2025 have only played 1-4 seasons. Before predicting
their five-year totals, we tested how well that works on 2012-2021 players,
whose totals we know: hide the total, predict it from the first few seasons,
compare. Each draft class was predicted by a model that never saw that class.

We compared three models: draft pick alone, the planned model (pick + AV so
far), and an extended model that also uses the latest season's AV and games
played. **The extended model was best**, and predictions improve fast: the
average miss is 6.3 AV after one season and 1.4 after four (figures 3-4).
We also found that misses are bigger for good players than for fringe players
(figure 5), which matters for the next step.

## Step 5: Filling in the unfinished careers, with uncertainty

Instead of one guess per 2022-2025 player, we made **100 plausible values**
each. Each value is the extended model's prediction plus a real error taken
from a past player with a similar prediction, so stars get star-sized
uncertainty and fringe players small uncertainty. We checked this on
2012-2021: the ranges contain the true value about as often as they should
(figure 6). The imputed classes look like past classes, and uncertainty is
largest for 2025, which has played only one season (figure 7).

## Step 6: The bounded value chart

We fit a smooth curve of value against pick that is **forced to go down**, using
the Bayesian method of Wilson et al. (2020), which builds the curve from
Bernstein polynomials. "Bayesian" means the result is not one curve but
thousands of plausible curves; the spread between them gives the bounds. We
fit it on all 100 versions of the data from step 5 and combined them, so the
bounds also include the uncertainty about unfinished careers.

Before trusting the bounds, we tested the method on made-up data where the
true curve is known. The method's default settings gave bands that were too
narrow, so we picked settings whose 95% bands contained the true curve about
95% of the time.

**Result (figures 8-9):** pick 1 is worth 39.7 AV (range 36.7-42.5); pick 32 is
worth 23.9, which is **60% of pick 1** (range 55-66%). The traditional Jimmy
Johnson chart says about 20%, so it greatly overvalues the top picks. Leaving
out the imputed 2022-2025 classes gives almost the same chart (figure 10).

## Step 7: How does it compare?

We fit the same data with other methods from the working paper: isotonic
regression, XGBoost forced to decrease, and splines. For predicting individual
players they are all about equally good, but they produce jagged steps or, for
the unconstrained spline, go up in places, and none of them come with bounds
(figure 11). Against the traditional Jimmy Johnson chart, our chart values
later picks far more: pick 32 is 60% of pick 1, not 20% (figure 12).

One more finding: **the first overall pick is special.** The last ten #1 picks
averaged 54 AV, while picks 2-10 averaged 35. A smooth chart spreads that jump
over nearby picks and shows pick 1 at about 40, so for the first pick its own
history is the better guide.

The full write-up is [REPORT.md](../REPORT.md).

## Questions you may ask

- **Why CUMYr5?** Five years covers a rookie contract, the years a team
  controls the player it drafted.
- **Why not just average each pick?** Averages at single picks jump around
  (only 14 players per pick); the curve borrows strength from nearby picks and
  enforces "earlier is at least as good".
- **Why impute instead of dropping 2022-2025?** Dropping them wastes four
  classes; imputing with 100 values keeps them while being honest about what
  we don't know yet. The chart is almost the same either way.
- **Why not the method's default settings?** We tested them on made-up data
  and their 95% bands only contained the truth about 90% of the time on
  draft-shaped data.
- **Is the Bayesian chart more accurate than simpler methods?** Not for
  predicting individual players; they are about tied. Its advantages are that
  it is smooth, never increases, and tells you how sure it is.
- **What do the bounds mean?** They are about the *average* value of a pick,
  which is what a trade chart needs. Individual players vary much more.
