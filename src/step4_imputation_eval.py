"""Step 4: how well can CUMYr5 be predicted from the first k seasons (k = 1..4)?

Uses the complete draft classes 2012-2021, where CUMYr5 is known. Each model is
scored with leave-one-draft-class-out cross-validation: fit on nine classes,
predict the held-out class, repeat for all ten. This mirrors how the model is
used on 2022-2025, classes it has never seen.

Models (for each k):
  pick_only       GAM  CUMYr5 ~ s(pick)                          (baseline: ignores early seasons)
  gam_schuckers   GAM  CUMYr5 ~ s(pick) + s(CUMYrk)              (the planned model)
  gam_plus        GAM  CUMYr5 ~ s(pick) + s(CUMYrk) + s(AVk) + s(CUMGk)
                  (adds the latest season's AV and games played so far)

Predictions are floored at CUMYrk, since AV already earned is kept.

Outputs (results/tables/):
  impute_cv_predictions.csv   every out-of-sample prediction (used by step 5 and the figures)
  impute_cv_metrics.csv       MAE, RMSE and bias per model and k
  impute_cv_by_class.csv      MAE per held-out class
"""

from pathlib import Path

import numpy as np
import pandas as pd
from pygam import LinearGAM, s

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed" / "nfl_draft_cumulative_2012_2025.csv"
OUT = ROOT / "results" / "tables"
CLASSES = range(2012, 2022)
LAMS = np.logspace(-3, 3, 7)          # smoothing values tried by the GAM grid search
MODELS = ["pick_only", "gam_schuckers", "gam_plus"]


def features(model, k):
    return {"pick_only": ["pick"],
            "gam_schuckers": ["pick", f"CUMYr{k}"],
            "gam_plus": ["pick", f"CUMYr{k}", f"AV{k}", f"CUMG{k}"]}[model]


def predict(train, test, model, k):
    cols = features(model, k)
    terms = s(0)
    for j in range(1, len(cols)):
        terms += s(j)
    gam = LinearGAM(terms).gridsearch(train[cols].to_numpy(), train["CUMYr5"].to_numpy(),
                                      lam=LAMS, progress=False)
    return np.maximum(gam.predict(test[cols].to_numpy()), test[f"CUMYr{k}"].to_numpy())


def main() -> None:
    d = pd.read_csv(DATA)
    d = d[d["year_drafted"].isin(CLASSES)].reset_index(drop=True)
    rows = []
    for k in range(1, 5):
        for held in CLASSES:
            train, test = d[d["year_drafted"] != held], d[d["year_drafted"] == held]
            for model in MODELS:
                rows.append(pd.DataFrame({
                    "k": k, "held_out": held, "model": model, "name": test["name"].to_numpy(),
                    "pick": test["pick"].to_numpy(), "CUMYrk": test[f"CUMYr{k}"].to_numpy(),
                    "actual": test["CUMYr5"].to_numpy(), "pred": predict(train, test, model, k)}))
        print(f"k={k} done")

    p = pd.concat(rows, ignore_index=True)
    p["err"] = p["pred"] - p["actual"]
    OUT.mkdir(parents=True, exist_ok=True)
    p.to_csv(OUT / "impute_cv_predictions.csv", index=False)

    g = p.assign(abs_err=p["err"].abs(), sq_err=p["err"] ** 2).groupby(["k", "model"])
    metrics = pd.DataFrame({"MAE": g["abs_err"].mean(), "RMSE": np.sqrt(g["sq_err"].mean()),
                            "bias": g["err"].mean()}).round(2).reset_index()
    metrics.to_csv(OUT / "impute_cv_metrics.csv", index=False)
    by_class = p.assign(abs_err=p["err"].abs()).groupby(["k", "model", "held_out"])["abs_err"].mean().round(2)
    by_class.unstack("held_out").to_csv(OUT / "impute_cv_by_class.csv")

    for col in ["MAE", "RMSE", "bias"]:
        t = metrics.pivot(index="model", columns="k", values=col).loc[MODELS]
        print(f"\n{col} (leave-one-class-out, 2012-2021, n={len(d)})\n{t.to_string()}")
    wins = (by_class.unstack("model")["gam_plus"] < by_class.unstack("model")["gam_schuckers"]).groupby("k").sum()
    print("\nHeld-out classes (of 10) where gam_plus beats gam_schuckers:", wins.to_dict())


if __name__ == "__main__":
    main()
