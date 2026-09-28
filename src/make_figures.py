"""Figures for the dataset (steps 1-3), imputation (steps 4-5) and the bounded value chart (step 6).

Run after the step scripts (see run_all.py):
    py src/make_figures.py
Writes PNGs to results/figures/.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed" / "nfl_draft_cumulative_2012_2025.csv"
CV = ROOT / "results" / "tables"
FIG = ROOT / "results" / "figures"

# Reference palette (dataviz skill): categorical slots in fixed order, plus chart ink.
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "axes.titlecolor": INK,
    "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.labelsize": 10, "xtick.color": MUTED, "ytick.color": MUTED,
    "xtick.labelsize": 9, "ytick.labelsize": 9, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "lines.linewidth": 2, "font.family": "sans-serif", "figure.dpi": 150,
})


def save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / name, bbox_inches="tight")
    plt.close(fig)
    print("wrote", FIG / name)


def fig_value_by_pick(d):
    """CUMYr5 against pick: the relationship the value chart is built on."""
    c = d[d["year_drafted"] <= 2021]
    mean_by_pick = c.groupby("pick")["CUMYr5"].mean()
    smooth = mean_by_pick.rolling(15, center=True, min_periods=5).mean()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.scatter(c["pick"], c["CUMYr5"], s=6, color=MUTED, alpha=0.35, linewidths=0, label="player")
    ax.plot(smooth.index, smooth.values, color=BLUE, label="average (15-pick window)")
    ax.set_title("Five-year AV falls steeply over the first round, then flattens")
    ax.set_xlabel("Overall pick")
    ax.set_ylabel("CUMYr5 (AV, first five seasons)")
    ax.set_xlim(0, 263)
    ax.set_ylim(-6, 90)
    ax.annotate("average", xy=(60, smooth.loc[60]), xytext=(75, 32), color=INK2, fontsize=9,
                arrowprops={"arrowstyle": "-", "color": AXIS})
    ax.text(262, 88, "Classes 2012-2021, n = %d" % len(c), ha="right", va="top", color=MUTED, fontsize=8)
    save(fig, "fig1_cumyr5_vs_pick.png")


def fig_by_round(d):
    """Mean cumulative AV by season and round: every round keeps rising, in order."""
    c = d[d["year_drafted"] <= 2021]
    m = c.groupby("round")[[f"CUMYr{k}" for k in range(1, 6)]].mean()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(1, 6)
    for r in m.index:
        color = BLUE if r == 1 else (ORANGE if r == 7 else AXIS)
        lw = 2 if r in (1, 7) else 1.4
        ax.plot(x, m.loc[r].values, color=color, lw=lw, marker="o", ms=4)
        ax.text(5.08, m.loc[r].iloc[-1], f"Round {r}", va="center", fontsize=8,
                color=INK if r in (1, 7) else INK2)
    ax.set_title("Cumulative AV by season: rounds stay in draft order")
    ax.set_xticks(x, [f"Yr{k}" for k in x])
    ax.set_xlabel("Seasons since draft")
    ax.set_ylabel("Mean cumulative AV")
    ax.set_xlim(0.8, 5.6)
    save(fig, "fig2_cumulative_av_by_round.png")


def fig_cv_error(metrics):
    """Out-of-sample error of the main models as more seasons are observed."""
    show = [("pick_only", "Pick only", MUTED), ("gam_schuckers", "GAM s(pick)+s(CUMYrk)", BLUE),
            ("gam_plus", "GAM + latest AV + games", AQUA)]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharex=True)
    for ax, col, title in [(axes[0], "MAE", "Mean absolute error"), (axes[1], "RMSE", "Root mean squared error")]:
        for key, label, color in show:
            t = metrics[metrics["model"] == key].sort_values("k")
            ax.plot(t["k"], t[col], color=color, marker="o", ms=5, label=label)
        ax.set_title(title)
        ax.set_xticks([1, 2, 3, 4], ["1\n(2025)", "2\n(2024)", "3\n(2023)", "4\n(2022)"])
        ax.set_xlabel("Seasons observed (class it applies to)")
        ax.set_ylim(0, None)
    axes[0].set_ylabel("AV points")
    axes[1].legend(frameon=False, fontsize=8, labelcolor=INK2, loc="lower left")
    fig.suptitle("Leave-one-class-out error predicting CUMYr5 (2012-2021)", x=0.01, ha="left",
                 fontsize=12, fontweight="bold", color=INK)
    fig.tight_layout()
    save(fig, "fig3_cv_error_by_k.png")


def fig_pred_vs_actual(pred):
    """Predicted vs actual CUMYr5 for the professor's GAM, one panel per k."""
    g = pred[pred["model"] == "gam_schuckers"]
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.6), sharex=True, sharey=True)
    for k, ax in zip(range(1, 5), axes):
        t = g[g["k"] == k]
        ax.scatter(t["pred"], t["actual"], s=5, color=BLUE, alpha=0.3, linewidths=0)
        ax.plot([0, 90], [0, 90], color=INK2, lw=1, ls="--")
        mae = (t["pred"] - t["actual"]).abs().mean()
        ax.set_title(f"k = {k}   MAE {mae:.1f}", fontsize=10)
        ax.set_xlabel("Predicted CUMYr5")
    axes[0].set_ylabel("Actual CUMYr5")
    axes[0].set_xlim(-2, 90)
    axes[0].set_ylim(-6, 90)
    fig.suptitle("GAM s(pick) + s(CUMYrk): predicted vs actual, out of sample (dashed = perfect)",
                 x=0.01, ha="left", fontsize=12, fontweight="bold", color=INK)
    fig.tight_layout()
    save(fig, "fig4_predicted_vs_actual.png")


def fig_error_spread(pred):
    """Error spread grows with the prediction: why constant-width intervals miscover."""
    g = pred[pred["model"] == "gam_schuckers"].copy()
    bins = [-1, 3, 8, 15, 25, 100]
    labels = ["0-3", "3-8", "8-15", "15-25", "25+"]
    g["bucket"] = pd.cut(g["pred"], bins, labels=labels)
    sd = g.groupby(["k", "bucket"], observed=True)["err"].std().unstack("bucket")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    colors = {1: BLUE, 2: ORANGE, 3: AQUA, 4: YELLOW}
    x = np.arange(len(labels))
    for k in range(1, 5):
        ax.plot(x, sd.loc[k].values, color=colors[k], marker="o", ms=5)
        ax.text(x[-1] + 0.08, sd.loc[k].iloc[-1], f"k = {k}", va="center", fontsize=9, color=INK2)
    ax.set_xticks(x, labels)
    ax.set_xlim(-0.2, len(labels) - 0.5)
    ax.set_ylim(0, None)
    ax.set_xlabel("Predicted CUMYr5")
    ax.set_ylabel("SD of prediction error (AV)")
    ax.set_title("Prediction error grows with the prediction (GAM, out of sample)")
    save(fig, "fig5_error_spread_by_prediction.png")


def fig_step5_calibration(cal):
    """Step 5: interval coverage by prediction size, against the target."""
    labels = ["0-3", "3-8", "8-15", "15-25", "25+"]
    colors = {1: BLUE, 2: ORANGE, 3: AQUA, 4: YELLOW}
    x = np.arange(len(labels))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharex=True)
    for ax, w in zip(axes, (80, 95)):
        ax.axhline(w / 100, color=INK2, lw=1, ls="--")
        ax.text(len(labels) - 0.55, w / 100, "target", va="bottom", ha="right", fontsize=8, color=INK2)
        for k in range(1, 5):
            row = cal[(cal["k"] == k) & (cal["width"] == w)][labels].iloc[0].to_numpy()
            ax.plot(x, row, color=colors[k], marker="o", ms=5, label=f"k = {k}")
        ax.set_title(f"{w}% intervals")
        ax.set_xticks(x, labels)
        ax.set_xlabel("Predicted CUMYr5")
        ax.set_ylim(0.7 if w == 80 else 0.88, 1.0)
    axes[0].set_ylabel("Share of players inside interval")
    axes[1].legend(frameon=False, fontsize=8, labelcolor=INK2, loc="lower left")
    fig.suptitle("Step 5 imputation intervals: coverage on 2012-2021 (each class imputed from the other nine)",
                 x=0.01, ha="left", fontsize=12, fontweight="bold", color=INK)
    fig.tight_layout()
    save(fig, "fig6_imputation_coverage.png")


def fig_step5_rounds(d, imp):
    """Step 5: imputed round averages for 2022-2025 with 80% intervals, vs 2012-2021."""
    ref = d[d["year_drafted"] <= 2021].groupby("round")["CUMYr5"].mean()
    cols = [c for c in imp.columns if c.startswith("imp_")]
    rounds = np.arange(1, 8)
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.8), sharey=True)
    for ax, (year, k) in zip(axes, [(2022, 4), (2023, 3), (2024, 2), (2025, 1)]):
        t = imp[imp["year_drafted"] == year]
        means = t.groupby("round")[cols].mean()          # round x imputation
        mid = means.mean(axis=1)
        lo, hi = means.quantile(0.10, axis=1), means.quantile(0.90, axis=1)
        ax.plot(rounds, ref.values, color=MUTED, lw=1.5, marker="o", ms=3, label="2012-2021 actual mean")
        ax.errorbar(rounds, mid, yerr=[mid - lo, hi - mid], fmt="o", color=BLUE, ms=5,
                    elinewidth=2, capsize=0, label=f"{year} imputed mean, 80% interval")
        ax.set_title(f"{year} class ({k} season{'s' if k > 1 else ''} seen)", fontsize=10)
        ax.set_xticks(rounds)
        ax.set_xlabel("Round")
        ax.legend(frameon=False, fontsize=7.5, labelcolor=INK2, loc="upper right")
    axes[0].set_ylabel("Mean CUMYr5")
    fig.suptitle("Imputed CUMYr5 by round: uncertainty grows as fewer seasons are observed",
                 x=0.01, ha="left", fontsize=12, fontweight="bold", color=INK)
    fig.tight_layout()
    save(fig, "fig7_imputed_classes_by_round.png")


BAND95, BAND80 = "#cde2fb", "#86b6ef"     # sequential blue steps 100 and 250


def fig_value_chart(chart, imp):
    """Step 6: the bounded value chart in AV."""
    cols = [c for c in imp.columns if c.startswith("imp_")]
    pick_mean = imp.assign(v=imp[cols].mean(axis=1)).groupby("pick")["v"].mean()
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.scatter(pick_mean.index, pick_mean.values, s=7, color=MUTED, alpha=0.6, linewidths=0,
               label="average CUMYr5 at each pick (14 classes)", zorder=1)
    ax.fill_between(chart["pick"], chart["value_lo95"], chart["value_hi95"], color=BAND95, lw=0,
                    label="95% credible band", zorder=2)
    ax.fill_between(chart["pick"], chart["value_lo80"], chart["value_hi80"], color=BAND80, lw=0,
                    label="80% credible band", zorder=3)
    ax.plot(chart["pick"], chart["value_mean"], color=BLUE, label="posterior mean value", zorder=4)
    ax.set_title("Bounded draft pick value chart (Bayesian monotone regression, 2012-2025 drafts)")
    ax.set_xlabel("Overall pick")
    ax.set_ylabel("Expected CUMYr5 (AV, first five seasons)")
    ax.set_xlim(0, 263)
    ax.set_ylim(0, None)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK2, loc="upper right")
    save(fig, "fig8_value_chart.png")


def fig_relative_chart(chart):
    """Step 6: value relative to the first pick, as trade charts are usually read."""
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.fill_between(chart["pick"], chart["rel_lo95"], chart["rel_hi95"], color=BAND95, lw=0, label="95% credible band")
    ax.fill_between(chart["pick"], chart["rel_lo80"], chart["rel_hi80"], color=BAND80, lw=0, label="80% credible band")
    ax.plot(chart["pick"], chart["rel_mean"], color=BLUE, label="posterior mean")
    for b in range(32, 225, 32):
        ax.axvline(b + 0.5, color=AXIS, lw=0.8, ls=":")
    for p in (32, 64, 100):
        r = chart.loc[chart["pick"] == p].iloc[0]
        ax.annotate(f"pick {p}: {r['rel_mean']:.0f}  ({r['rel_lo95']:.0f}-{r['rel_hi95']:.0f})",
                    xy=(p, r["rel_mean"]), xytext=(p + 18, r["rel_mean"] + 12), fontsize=8, color=INK2,
                    arrowprops={"arrowstyle": "-", "color": AXIS})
    ax.set_title("Pick value relative to the first pick (pick 1 = 100)")
    ax.set_xlabel("Overall pick (dotted lines: every 32 picks)")
    ax.set_ylabel("Relative value")
    ax.set_xlim(0, 263)
    ax.set_ylim(0, 105)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK2, loc="upper right")
    save(fig, "fig9_relative_value_chart.png")


def fig_chart_comparison(chart, chart_obs):
    """Step 6: all classes with imputation vs complete classes only."""
    fig, ax = plt.subplots(figsize=(9, 5))
    for tab, color, label in [(chart_obs, ORANGE, "2012-2021 only (complete careers)"),
                              (chart, BLUE, "2012-2025 (2022-2025 imputed)")]:
        ax.fill_between(tab["pick"], tab["value_lo95"], tab["value_hi95"], color=color, alpha=0.18, lw=0)
        ax.plot(tab["pick"], tab["value_mean"], color=color, label=f"{label}: mean and 95% band")
    ax.set_title("Adding the imputed 2022-2025 classes barely moves the chart")
    ax.set_xlabel("Overall pick")
    ax.set_ylabel("Expected CUMYr5 (AV)")
    ax.set_xlim(0, 263)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK2, loc="upper right")
    save(fig, "fig10_value_chart_comparison.png")


MAGENTA = "#e87ba4"


def fig_model_comparison(curves, acc):
    """Step 7: the Bayesian chart against other curve-fitting methods."""
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.fill_between(curves["pick"], curves["bayes_lo95"], curves["bayes_hi95"], color=BAND95, lw=0,
                    label="Bayesian 95% band")
    series = [("isotonic", "Isotonic regression", ORANGE, 1.4), ("xgboost_mono", "XGBoost, monotone", AQUA, 1.4),
              ("spline_32", "Spline, knot every 32 picks", YELLOW, 1.4),
              ("spline_mono", "Monotone penalized spline", MAGENTA, 1.4),
              ("bayes_monotone", "Bayesian monotone (this project)", BLUE, 2.2)]
    for key, label, color, lw in series:
        mae = acc.loc[key, "MAE"]
        ax.plot(curves["pick"], curves[key], color=color, lw=lw, label=f"{label} (MAE {mae:.2f})")
    ax.set_title("Monotone curves agree after the first few picks; only the Bayesian one has bounds")
    ax.set_xlabel("Overall pick")
    ax.set_ylabel("Expected CUMYr5 (AV)")
    ax.set_xlim(0, 263)
    ax.set_ylim(0, 56)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK2, loc="upper right",
              title="out-of-sample MAE, 2012-2021", title_fontsize=8)
    save(fig, "fig11_model_comparison.png")


def fig_vs_jimmy_johnson(curves):
    """Step 7: relative value (pick 1 = 100) against the traditional Jimmy Johnson chart."""
    c = curves[curves["pick"] <= 224]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.fill_between(c["pick"], c["bayes_rel_lo95"], c["bayes_rel_hi95"], color=BAND95, lw=0,
                    label="Bayesian chart, 95% band")
    ax.plot(c["pick"], c["bayes_monotone_rel"], color=BLUE, label="Bayesian chart (performance-based)")
    ax.plot(c["pick"], c["jj_rel"], color=ORANGE, label="Jimmy Johnson chart (1990s trade chart)")
    for p in (32, 64):
        r = c[c["pick"] == p].iloc[0]
        ax.annotate(f"pick {p}: {r['bayes_monotone_rel']:.0f} vs {r['jj_rel']:.0f}",
                    xy=(p, r["bayes_monotone_rel"]), xytext=(p + 20, r["bayes_monotone_rel"] + 14),
                    fontsize=8, color=INK2, arrowprops={"arrowstyle": "-", "color": AXIS})
    ax.set_title("The traditional chart values later picks far below what they produce")
    ax.set_xlabel("Overall pick")
    ax.set_ylabel("Value relative to pick 1 (= 100)")
    ax.set_xlim(0, 225)
    ax.set_ylim(0, 105)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK2, loc="upper right")
    save(fig, "fig12_vs_jimmy_johnson.png")


def main():
    d = pd.read_csv(DATA)
    fig_value_by_pick(d)
    fig_by_round(d)
    metrics = pd.read_csv(CV / "impute_cv_metrics.csv")
    pred = pd.read_csv(CV / "impute_cv_predictions.csv")
    fig_cv_error(metrics)
    fig_pred_vs_actual(pred)
    fig_error_spread(pred)
    cal_path = CV / "step5_calibration_player.csv"
    imp_path = ROOT / "data" / "processed" / "cumyr5_imputed_2012_2025.csv"
    if cal_path.exists() and imp_path.exists():
        fig_step5_calibration(pd.read_csv(cal_path))
        fig_step5_rounds(d, pd.read_csv(imp_path))
    chart_path = CV / "step6_value_chart.csv"
    if chart_path.exists():
        chart = pd.read_csv(chart_path)
        fig_value_chart(chart, pd.read_csv(imp_path))
        fig_relative_chart(chart)
        fig_chart_comparison(chart, pd.read_csv(CV / "step6_value_chart_2012_2021.csv"))
    curves_path = CV / "step7_model_curves.csv"
    if curves_path.exists():
        curves = pd.read_csv(curves_path)
        fig_model_comparison(curves, pd.read_csv(CV / "step7_cv_accuracy.csv", index_col="model"))
        fig_vs_jimmy_johnson(curves)


if __name__ == "__main__":
    main()
