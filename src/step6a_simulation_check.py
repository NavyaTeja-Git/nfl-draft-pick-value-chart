"""Step 6a: check the Bayesian monotone regression (src/bnmr.py) on simulated data.

Before fitting the real value chart, simulate data where the true curve is
known and check (1) the posterior mean recovers it and (2) the credible bands
contain it as often as they claim. Several prior settings are compared,
because the paper's default (mu = 0.5, phi = 0.25 on standardized y) was built
for curves with a few large steps, while a draft value curve rises about 2.5
standard deviations over 50 small increments.

Design: the real draft picks (3,584 players, picks 1-262).
True curves:
  real_shape   a monotone GAM fit to 2012-2021 CUMYr5 by pick; noise resampled
               from that fit's residuals (skewed, like real AV)
  exponential  3 + 35 exp(-(pick - 1) / 45); normal noise, sd 12
20 simulated datasets per curve and setting; 1,000 burn-in + 1,000 kept sweeps.

Output: results/tables/step6a_simulation_check.csv
"""

import zlib
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from pygam import LinearGAM, s

from bnmr import MonotoneBP, Prior

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed" / "nfl_draft_cumulative_2012_2025.csv"
OUT = ROOT / "results" / "tables" / "step6a_simulation_check.csv"

SETTINGS = {
    "paper (M=50, mu=0.5, phi=0.25)": Prior(M=50, mu=0.5, phi=0.25),
    "M=50, mu=0.1, phi=0.1": Prior(M=50, mu=0.1, phi=0.1),
    "M=50, mu=0.05, phi=0.05": Prior(M=50, mu=0.05, phi=0.05),
    "M=20, mu=0.5, phi=0.25": Prior(M=20, mu=0.5, phi=0.25),
}
REPS, BURN, KEEP = 20, 1000, 1000
REGIONS = {"picks 1-32": (1, 32), "picks 33-100": (33, 100), "picks 101-262": (101, 262)}


def true_curves(d):
    c = d[d["year_drafted"] <= 2021]
    gam = LinearGAM(s(0, constraints="monotonic_dec")).gridsearch(
        c[["pick"]].to_numpy(), c["CUMYr5"].to_numpy(), progress=False)
    grid = np.arange(1, d["pick"].max() + 1)
    resid = (c["CUMYr5"] - gam.predict(c[["pick"]].to_numpy())).to_numpy()
    return {
        "real_shape": (gam.predict(grid[:, None]), resid - resid.mean()),
        "exponential": (3 + 35 * np.exp(-(grid - 1) / 45), None),
    }


def one_run(args):
    truth_name, truth, resid, setting, prior, rep, picks, burn, keep = args
    rng = np.random.default_rng([rep, zlib.crc32((truth_name + setting).encode())])
    noise = rng.choice(resid, len(picks)) if resid is not None else rng.normal(0, 12, len(picks))
    y = truth[picks - 1] + noise
    g = pd.DataFrame({"pick": picks, "y": y}).groupby("pick")["y"]
    stats = pd.DataFrame({"n": g.size(), "sy": g.sum(), "sy2": g.apply(lambda v: (v ** 2).sum())})
    dmax = len(truth)
    model = MonotoneBP(1 - (stats.index.to_numpy() - 1) / (dmax - 1), stats["n"], stats["sy"], stats["sy2"], prior)
    thetas, _, _ = model.run(burn, keep, rng)
    grid = np.arange(1, dmax + 1)
    F = model.curve(thetas, 1 - (grid - 1) / (dmax - 1))
    mean = F.mean(axis=0)
    rows = []
    for region, (a, b) in {"all picks": (1, dmax), **REGIONS}.items():
        sl = slice(a - 1, b)
        row = {"truth": truth_name, "setting": setting, "rep": rep, "region": region,
               "rmse": np.sqrt(np.mean((mean[sl] - truth[sl]) ** 2))}
        for w in (80, 95):
            lo, hi = np.percentile(F[:, sl], [50 - w / 2, 50 + w / 2], axis=0)
            row[f"cover{w}"] = np.mean((truth[sl] >= lo) & (truth[sl] <= hi))
            row[f"width{w}"] = np.mean(hi - lo)
        rows.append(row)
    return rows


def main():
    d = pd.read_csv(DATA)
    picks = d["pick"].to_numpy()
    jobs = [(tn, tr, rs, sn, pr, rep, picks, BURN, KEEP)
            for tn, (tr, rs) in true_curves(d).items()
            for sn, pr in SETTINGS.items() for rep in range(REPS)]
    with Pool(min(20, len(jobs))) as pool:
        rows = [r for res in pool.map(one_run, jobs) for r in res]
    res = pd.DataFrame(rows)
    summary = res.groupby(["truth", "setting", "region"], sort=False)[
        ["cover80", "cover95", "width95", "rmse"]].mean().round(3)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUT)
    pd.set_option("display.width", 160)
    print(summary.to_string())


if __name__ == "__main__":
    main()
