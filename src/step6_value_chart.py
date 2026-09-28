"""Step 6: the bounded draft pick value chart.

Fits the Bayesian monotone (decreasing) Bernstein-polynomial regression of
src/bnmr.py (Wilson et al. 2020) of CUMYr5 on pick, and reports the posterior
mean value of every pick 1-262 with 80% and 95% credible bounds.

Main fit: all 14 classes (2012-2025). The 2022-2025 CUMYr5 values are the 100
imputations from step 5. The model is fit to each imputed dataset and the
posterior draws are pooled, so the bounds include both the uncertainty in the
curve and the uncertainty in the imputed careers.

Comparison fit: the complete classes 2012-2021 only (no imputed values).

Prior: chosen with the simulation check in step6a_simulation_check.py.

Outputs:
  results/tables/step6_value_chart.csv             main chart (all classes, pooled over imputations)
  results/tables/step6_value_chart_2012_2021.csv   comparison chart (complete classes only)
  results/tables/step6_convergence_rhat.csv        split R-hat of the comparison fit's 20 chains
Value columns are in AV (CUMYr5); rel_* columns scale each posterior draw so
that pick 1 = 100.
"""

from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

from bnmr import MonotoneBP, Prior

ROOT = Path(__file__).resolve().parent.parent
IMPUTED = ROOT / "data" / "processed" / "cumyr5_imputed_2012_2025.csv"
TABLES = ROOT / "results" / "tables"

PRIOR = Prior(M=50, mu=0.1, phi=0.1)        # see docs/step6_value_chart.md, simulation check
PICK_MAX = 262
GRID = np.arange(1, PICK_MAX + 1)
X_GRID = 1 - (GRID - 1) / (PICK_MAX - 1)     # pick 1 -> x = 1, so "increasing in x" = decreasing in pick
FIRST_BURN, BURN, KEEP, THIN = 2000, 400, 200, 5   # sweeps per imputed dataset: 400 burn-in, 200 draws kept from 1,000
WORKERS = 20
SEED = 20260924


def grouped(picks, y):
    g = pd.DataFrame({"pick": picks, "y": y}).groupby("pick")["y"]
    st = pd.DataFrame({"n": g.size(), "sy": g.sum(), "sy2": g.apply(lambda v: float((v ** 2).sum()))})
    return 1 - (st.index.to_numpy() - 1) / (PICK_MAX - 1), st["n"].to_numpy(), st["sy"].to_numpy(), st["sy2"].to_numpy()


def fit_block(args):
    """Fit a block of imputed datasets in sequence, warm-starting each chain from the last."""
    worker, picks, ys = args
    rng = np.random.default_rng(SEED + worker)
    state, draws = None, []
    for j, y in enumerate(ys):
        model = MonotoneBP(*grouped(picks, y), PRIOR)
        thetas, _, state = model.run(FIRST_BURN if j == 0 else BURN, KEEP, rng, state=state, thin=THIN)
        draws.append(model.curve(thetas, X_GRID))
    return np.vstack(draws)


def split_rhat(chains):
    """Split R-hat (Gelman et al.) for a list of 1-d chains of equal length."""
    half = min(len(c) for c in chains) // 2
    parts = np.array([c[i * half:(i + 1) * half] for c in chains for i in (0, 1)])
    n = parts.shape[1]
    w = parts.var(axis=1, ddof=1).mean()
    b = n * parts.mean(axis=1).var(ddof=1)
    return float(np.sqrt(((n - 1) / n * w + b / n) / w))


def summarize(F):
    rel = 100 * F / F[:, [0]]
    out = pd.DataFrame({"pick": GRID, "value_mean": F.mean(0), "value_median": np.median(F, 0)})
    for w in (80, 95):
        out[f"value_lo{w}"], out[f"value_hi{w}"] = np.percentile(F, [50 - w / 2, 50 + w / 2], axis=0)
    out["rel_mean"] = rel.mean(0)
    for w in (80, 95):
        out[f"rel_lo{w}"], out[f"rel_hi{w}"] = np.percentile(rel, [50 - w / 2, 50 + w / 2], axis=0)
    out["round_approx"] = np.minimum((GRID - 1) // 32 + 1, 7)
    return out.round(3)


def main():
    d = pd.read_csv(IMPUTED)
    picks = d["pick"].to_numpy()
    imp_cols = [c for c in d.columns if c.startswith("imp_")]

    # Main fit: every imputed dataset, spread over workers.
    blocks = np.array_split(np.arange(len(imp_cols)), WORKERS)
    jobs = [(w, picks, [d[imp_cols[i]].to_numpy(float) for i in idx]) for w, idx in enumerate(blocks)]
    # Comparison fit: complete classes only, one long chain per worker (same total draws).
    obs = d[~d["CUMYr5_imputed"]]
    obs_jobs = [(100 + w, obs["pick"].to_numpy(), [obs["CUMYr5_observed"].to_numpy(float)] * len(idx))
                for w, idx in enumerate(blocks)]
    with Pool(WORKERS) as pool:
        F_all = np.vstack(pool.map(fit_block, jobs))
        F_obs = np.vstack(pool.map(fit_block, obs_jobs))

    TABLES.mkdir(parents=True, exist_ok=True)
    main_tab, obs_tab = summarize(F_all), summarize(F_obs)
    means = d.groupby("pick")["CUMYr5_observed"].agg(["size"]).rename(columns={"size": "n_players"})
    main_tab = main_tab.merge(means, left_on="pick", right_index=True, how="left")
    main_tab.to_csv(TABLES / "step6_value_chart.csv", index=False)
    obs_tab.to_csv(TABLES / "step6_value_chart_2012_2021.csv", index=False)

    print(f"posterior draws: all classes {len(F_all)}, complete classes {len(F_obs)}")
    print("monotone decreasing in every draw:", bool(np.all(np.diff(F_all, axis=1) <= 1e-9)))
    show = [1, 2, 5, 10, 16, 32, 33, 50, 64, 100, 128, 160, 192, 224, 256, 262]
    cols = ["pick", "value_mean", "value_lo95", "value_hi95", "rel_mean", "rel_lo95", "rel_hi95"]
    print("\nValue chart, all classes 2012-2025 (AV = CUMYr5; rel: pick 1 = 100)")
    print(main_tab.set_index("pick").loc[show, cols[1:]].to_string())
    print("\nComparison, complete classes 2012-2021 only")
    print(obs_tab.set_index("pick").loc[show, cols[1:4]].to_string())
    # Convergence: the comparison fit ran as WORKERS independent chains on the same
    # data, so the split R-hat of f(pick) across them should be close to 1.
    chains = np.array_split(F_obs, WORKERS)
    rhat = pd.DataFrame({"pick": show, "rhat": [split_rhat([c[:, p - 1] for c in chains]) for p in show]})
    rhat.to_csv(TABLES / "step6_convergence_rhat.csv", index=False)
    print(f"\nConvergence (split R-hat over {WORKERS} chains, target < 1.01): "
          f"max {rhat['rhat'].max():.3f} across picks {show[0]}-{show[-1]}")
    w_all = (main_tab["value_hi95"] - main_tab["value_lo95"]).mean()
    w_obs = (obs_tab["value_hi95"] - obs_tab["value_lo95"]).mean()
    print(f"\nMean 95% band width: all classes {w_all:.2f} AV, complete classes only {w_obs:.2f} AV")


if __name__ == "__main__":
    main()
