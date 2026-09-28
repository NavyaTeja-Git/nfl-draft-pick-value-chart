"""Step 5: multiple imputation of CUMYr5 for the 2022-2025 draft classes.

Model: the extended GAM from step 4,
    CUMYr5 ~ s(pick) + s(CUMYrk) + s(AVk) + s(CUMGk),
fit on all complete classes (2012-2021), with k = seasons observed:
    2022 -> k = 4, 2023 -> k = 3, 2024 -> k = 2, 2025 -> k = 1.

Uncertainty: each imputed value is prediction + an error drawn from the
out-of-sample errors of step 4 (leave-one-class-out, same model, same k):
  - Neighbour errors: the error comes from one of the K past players whose
    prediction was closest, so the spread grows with the prediction and keeps
    the skew of real errors (step 4 showed a constant spread miscovers).
  - Class-level draws: for each imputation, all errors come from one past
    class chosen at random, so a class-wide over/under-prediction is carried
    along instead of averaging out.
  - Values are rounded to whole AV and never fall below AV already earned.

Validation: the same procedure is run on each 2012-2021 class using only the
other nine classes' errors, and interval coverage is compared with the truth,
overall, by prediction size, and for round averages.

Outputs:
  data/processed/cumyr5_imputed_2012_2025.csv   every player; imp_001..imp_100
                                                 (2012-2021 rows repeat the actual CUMYr5)
  data/processed/cumyr5_imputed_summary.csv     2022-2025: prediction, mean, 10/50/90%
  results/tables/step5_calibration_player.csv
  results/tables/step5_calibration_round_means.csv
"""

from pathlib import Path

import numpy as np
import pandas as pd
from pygam import LinearGAM, s

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed" / "nfl_draft_cumulative_2012_2025.csv"
OOF = ROOT / "results" / "tables" / "impute_cv_predictions.csv"   # from step 4
PROCESSED = ROOT / "data" / "processed"
TABLES = ROOT / "results" / "tables"

TRAIN_CLASSES = range(2012, 2022)
TARGET_K = {2022: 4, 2023: 3, 2024: 2, 2025: 1}
M = 100                  # imputations
M_CHECK = 200            # draws per player in the calibration check
K_NEIGHBOURS = 25        # past players whose errors are borrowed
LAMS = np.logspace(-3, 3, 7)
SEED = 20260924
BUCKETS = ([-1, 3, 8, 15, 25, 200], ["0-3", "3-8", "8-15", "15-25", "25+"])


def features(k):
    return ["pick", f"CUMYr{k}", f"AV{k}", f"CUMG{k}"]


def fit_extended_gam(train, k):
    X, y = train[features(k)].to_numpy(), train["CUMYr5"].to_numpy()
    return LinearGAM(s(0) + s(1) + s(2) + s(3)).gridsearch(X, y, lam=LAMS, progress=False)


def error_pools(oof_k):
    """Per past class: predictions and out-of-sample errors (actual - prediction)."""
    return {c: (g["pred"].to_numpy(), (g["actual"] - g["pred"]).to_numpy())
            for c, g in oof_k.groupby("held_out")}


def draw(pred, floor, pools, n_draws, rng):
    """n_draws x len(pred) imputed values.

    For each draw, pick one past class, then give each player the error of one
    of that class's K players with the closest prediction.
    """
    classes = list(pools)
    nearest = {}
    for c in classes:
        cp, _ = pools[c]
        dist = np.abs(pred[:, None] - cp[None, :])
        nearest[c] = np.argsort(dist, axis=1)[:, :K_NEIGHBOURS]
    out = np.empty((n_draws, len(pred)))
    rows = np.arange(len(pred))
    for m in range(n_draws):
        c = classes[rng.integers(len(classes))]
        j = nearest[c][rows, rng.integers(0, K_NEIGHBOURS, size=len(pred))]
        out[m] = pred + pools[c][1][j]
    return np.maximum(np.rint(out), floor)


def calibrate(d, oof, rng):
    """Run the procedure on 2012-2021, one held-out class at a time, and score coverage."""
    player_rows, round_rows = [], []
    for k in range(1, 5):
        ok = oof[(oof["model"] == "gam_plus") & (oof["k"] == k)]
        pools = error_pools(ok)
        for held, g in ok.groupby("held_out"):
            others = {c: v for c, v in pools.items() if c != held}
            g = g.merge(d[["name", "year_drafted", "pick", "round"]],
                        left_on=["name", "held_out", "pick"], right_on=["name", "year_drafted", "pick"])
            pred, floor, y = g["pred"].to_numpy(), g["CUMYrk"].to_numpy(), g["actual"].to_numpy()
            sims = draw(pred, floor, others, M_CHECK, rng)
            bucket = pd.cut(pred, bins=BUCKETS[0], labels=BUCKETS[1]).astype(str)
            for w in (80, 95):
                lo, hi = np.percentile(sims, [50 - w / 2, 50 + w / 2], axis=0)
                player_rows.append(pd.DataFrame({
                    "k": k, "held_out": held, "width": w, "bucket": bucket,
                    "covered": (y >= lo) & (y <= hi), "int_width": hi - lo}))
                for r in range(1, 8):
                    sel = g["round"].to_numpy() == r
                    lo_r, hi_r = np.percentile(sims[:, sel].mean(axis=1), [50 - w / 2, 50 + w / 2])
                    round_rows.append({"k": k, "held_out": held, "round": r, "width": w,
                                       "covered": lo_r <= y[sel].mean() <= hi_r})
        print(f"  calibration k={k} done")
    pc = pd.concat(player_rows, ignore_index=True)
    rc = pd.DataFrame(round_rows)
    by_bucket = pc.groupby(["k", "width", "bucket"])["covered"].mean().unstack("bucket")[BUCKETS[1]]
    overall = pc.groupby(["k", "width"])["covered"].mean().rename("all")
    player_tab = by_bucket.join(overall).round(3)
    width_tab = pc.groupby(["k", "width"])["int_width"].mean().round(1)
    round_tab = rc.groupby(["k", "width"])["covered"].mean().unstack("width").round(3)
    return player_tab, width_tab, round_tab


def impute(d, oof, rng):
    """Fit on 2012-2021 and draw M imputations for each 2022-2025 class."""
    train = d[d["year_drafted"].isin(TRAIN_CLASSES)]
    imputed, summary = [], []
    for year, k in TARGET_K.items():
        target = d[d["year_drafted"] == year]
        model = fit_extended_gam(train, k)
        pred = np.maximum(model.predict(target[features(k)].to_numpy()), target[f"CUMYr{k}"].to_numpy())
        pools = error_pools(oof[(oof["model"] == "gam_plus") & (oof["k"] == k)])
        sims = draw(pred, target[f"CUMYr{k}"].to_numpy(), pools, M, rng)
        imputed.append(pd.DataFrame(sims.T, index=target.index,
                                    columns=[f"imp_{i:03d}" for i in range(1, M + 1)]))
        summary.append(pd.DataFrame({
            "name": target["name"], "year_drafted": year, "round": target["round"], "pick": target["pick"],
            "k_seasons_observed": k, f"CUMYr_observed": target[f"CUMYr{k}"], "prediction": pred.round(2),
            "imp_mean": sims.mean(axis=0).round(2), "imp_p10": np.percentile(sims, 10, axis=0),
            "imp_p50": np.percentile(sims, 50, axis=0), "imp_p90": np.percentile(sims, 90, axis=0)}))
        print(f"  {year}: k={k}, {len(target)} players, mean prediction {pred.mean():.1f}")

    # One table for every player: observed classes repeat their actual CUMYr5 in
    # every imputation column, so later steps can treat all 14 classes the same way.
    out = d[["name", "url", "year_drafted", "round", "pick", "position"]].copy()
    out["CUMYr5_imputed"] = out["year_drafted"].isin(TARGET_K)
    out["CUMYr5_observed"] = d["CUMYr5"]
    imp_cols = [f"imp_{i:03d}" for i in range(1, M + 1)]
    observed = out.loc[~out["CUMYr5_imputed"], "CUMYr5_observed"].to_numpy()
    imp = pd.concat(imputed + [pd.DataFrame(np.repeat(observed[:, None], M, axis=1),
                                            index=out.index[~out["CUMYr5_imputed"]], columns=imp_cols)])
    return pd.concat([out, imp.loc[out.index].astype(int)], axis=1), pd.concat(summary)


def main():
    rng = np.random.default_rng(SEED)
    d = pd.read_csv(DATA)
    oof = pd.read_csv(OOF)

    print("Calibration check on 2012-2021 (each class imputed from the other nine):")
    player_tab, width_tab, round_tab = calibrate(d, oof, rng)
    TABLES.mkdir(parents=True, exist_ok=True)
    player_tab.to_csv(TABLES / "step5_calibration_player.csv")
    round_tab.to_csv(TABLES / "step5_calibration_round_means.csv")
    print("\nPlayer-level coverage by predicted CUMYr5 (target = width/100):\n", player_tab.to_string())
    print("\nMean interval width (AV):\n", width_tab.unstack("width").to_string())
    print("\nRound-average coverage (target = width/100):\n", round_tab.to_string())

    print("\nImputing 2022-2025:")
    full, summary = impute(d, oof, rng)
    full.to_csv(PROCESSED / "cumyr5_imputed_2012_2025.csv", index=False)
    summary.to_csv(PROCESSED / "cumyr5_imputed_summary.csv", index=False)
    print("\nImputed CUMYr5 by class and round (mean over imputations):")
    t = summary.groupby(["year_drafted", "round"])["imp_mean"].mean().unstack("round").round(1)
    print(t.to_string())
    ref = d[d["year_drafted"] <= 2021].groupby("round")["CUMYr5"].mean().round(1)
    print("\nFor comparison, actual 2012-2021 mean CUMYr5 by round:\n", ref.to_frame().T.to_string())


if __name__ == "__main__":
    main()
