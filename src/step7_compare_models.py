"""Step 7: compare the Bayesian value chart with other ways to draw a value curve.

Curves of CUMYr5 against pick:
  bayes_monotone   the Bayesian monotone Bernstein-polynomial model (step 6)
  isotonic         isotonic regression, forced non-increasing (a step function)
  xgboost_mono     XGBoost with a decreasing monotone constraint
  spline_mono      penalized spline (GAM) with a decreasing constraint
  spline_32        cubic regression spline with a knot every 32 picks (not monotone)
  pick_average     plain average CUMYr5 at each pick (no smoothing)

1. Accuracy: leave-one-draft-class-out on the complete classes 2012-2021. Each
   curve is fit on nine classes and scored on the held-out class's players.
2. Shape: every curve is fit on all 14 classes (2022-2025 CUMYr5 = the mean of
   the 100 step 5 imputations) and checked for places where value goes UP.
3. The Jimmy Johnson chart (data/raw/reference) is compared on a relative
   scale (pick 1 = 100), since its points are not in AV.

Outputs (results/tables/):
  step7_cv_accuracy.csv       out-of-sample error of each curve
  step7_model_curves.csv      each curve for picks 1-262, plus Jimmy Johnson
"""

from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from pygam import LinearGAM, s
from scipy.interpolate import make_lsq_spline
from sklearn.isotonic import IsotonicRegression
from xgboost import XGBRegressor

from bnmr import MonotoneBP
from step6_value_chart import GRID, PRIOR, X_GRID, grouped

ROOT = Path(__file__).resolve().parent.parent
IMPUTED = ROOT / "data" / "processed" / "cumyr5_imputed_2012_2025.csv"
JJ = ROOT / "data" / "raw" / "reference" / "jimmy_johnson_chart.csv"
TABLES = ROOT / "results" / "tables"
CLASSES = range(2012, 2022)
SEED = 20260924
MODELS = ["bayes_monotone", "isotonic", "xgboost_mono", "spline_mono", "spline_32", "pick_average"]


def fit_curve(model, picks, y, seed=SEED):
    """Fit one curve to (pick, CUMYr5) data; return its value at picks 1-262."""
    picks, y = np.asarray(picks), np.asarray(y, float)
    if model == "bayes_monotone":
        rng = np.random.default_rng(seed)
        m = MonotoneBP(*grouped(picks, y), PRIOR)
        thetas, _, _ = m.run(1500, 300, rng, thin=5)
        return m.curve(thetas, X_GRID).mean(axis=0)
    if model == "isotonic":
        iso = IsotonicRegression(increasing=False, out_of_bounds="clip").fit(picks, y)
        return iso.predict(GRID)
    if model == "xgboost_mono":
        xgb = XGBRegressor(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8,
                           monotone_constraints="(-1)", random_state=seed)
        xgb.fit(picks.reshape(-1, 1), y)
        return xgb.predict(GRID.reshape(-1, 1))
    if model == "spline_mono":
        gam = LinearGAM(s(0, constraints="monotonic_dec")).gridsearch(
            picks.reshape(-1, 1), y, progress=False)
        return gam.predict(GRID.reshape(-1, 1))
    if model == "spline_32":
        g = pd.DataFrame({"p": picks, "y": y}).groupby("p")["y"].agg(["mean", "size"])
        x = g.index.to_numpy(float)
        inner = np.arange(32.5, x.max() - 1, 32)          # knots between rounds: 32.5, 64.5, ...
        t = np.r_[[x.min()] * 4, inner, [x.max()] * 4]
        spl = make_lsq_spline(x, g["mean"].to_numpy(), t, k=3, w=np.sqrt(g["size"].to_numpy()))
        return spl(np.clip(GRID, x.min(), x.max()))
    if model == "pick_average":
        avg = pd.Series(y).groupby(picks).mean()
        return avg.reindex(GRID).interpolate(limit_direction="both").to_numpy()
    raise ValueError(model)


def cv_one(args):
    model, held, picks_tr, y_tr = args
    return model, held, fit_curve(model, picks_tr, y_tr, seed=SEED + held)


def main():
    d = pd.read_csv(IMPUTED)
    imp_cols = [c for c in d.columns if c.startswith("imp_")]
    obs = d[~d["CUMYr5_imputed"]]

    # 1. Leave-one-class-out accuracy on 2012-2021.
    jobs = []
    for held in CLASSES:
        tr = obs[obs["year_drafted"] != held]
        for model in MODELS:
            jobs.append((model, held, tr["pick"].to_numpy(), tr["CUMYr5_observed"].to_numpy()))
    with Pool(20) as pool:
        fits = pool.map(cv_one, jobs)
    rows = []
    for model, held, curve in fits:
        te = obs[obs["year_drafted"] == held]
        pred = curve[te["pick"].to_numpy() - 1]
        err = pred - te["CUMYr5_observed"].to_numpy()
        rows.append({"model": model, "held_out": held, "sse": float((err ** 2).sum()),
                     "sae": float(np.abs(err).sum()), "n": len(te)})
    r = pd.DataFrame(rows).groupby("model")[["sse", "sae", "n"]].sum()
    acc = pd.DataFrame({"RMSE": np.sqrt(r["sse"] / r["n"]), "MAE": r["sae"] / r["n"]}).loc[MODELS].round(3)
    TABLES.mkdir(parents=True, exist_ok=True)
    acc.to_csv(TABLES / "step7_cv_accuracy.csv")

    # 2. Every curve on all 14 classes (imputed classes at their imputation mean).
    y_all = d[imp_cols].mean(axis=1).to_numpy()
    with Pool(len(MODELS)) as pool:
        curves = pool.starmap(fit_curve, [(m, d["pick"].to_numpy(), y_all) for m in MODELS])
    out = pd.DataFrame({"pick": GRID, **dict(zip(MODELS, curves))})
    chart = pd.read_csv(TABLES / "step6_value_chart.csv")
    out["bayes_monotone"] = chart["value_mean"].to_numpy()       # use the full step 6 posterior mean
    out["bayes_lo95"], out["bayes_hi95"] = chart["value_lo95"].to_numpy(), chart["value_hi95"].to_numpy()
    jj = pd.read_csv(JJ)
    out = out.merge(jj, on="pick", how="left")
    for m in MODELS:
        out[f"{m}_rel"] = 100 * out[m] / out[m].iloc[0]
    out["jj_rel"] = 100 * out["jj_points"] / out["jj_points"].iloc[0]
    out["bayes_rel_lo95"], out["bayes_rel_hi95"] = chart["rel_lo95"].to_numpy(), chart["rel_hi95"].to_numpy()
    out.round(3).to_csv(TABLES / "step7_model_curves.csv", index=False)

    ups = {m: int((np.diff(out[m].to_numpy()) > 1e-9).sum()) for m in MODELS}
    pd.set_option("display.width", 160)
    print("Out-of-sample accuracy, leave-one-class-out 2012-2021 (lower is better):")
    print(acc.to_string())
    print("\nPicks where the curve goes UP (fit on all classes):", ups)
    show = [1, 10, 32, 64, 100, 160, 224]
    print("\nValue (AV) at selected picks:")
    print(out.set_index("pick").loc[show, MODELS].round(1).to_string())
    print("\nRelative to pick 1 = 100 (Bayesian with 95% bounds vs Jimmy Johnson):")
    print(out.set_index("pick").loc[show, ["bayes_monotone_rel", "bayes_rel_lo95", "bayes_rel_hi95", "jj_rel"]]
          .round(1).to_string())


if __name__ == "__main__":
    main()
