"""Step 3: build the final dataset: every player drafted 2012-2025 with yearly and cumulative AV and games.

Inputs:
  data/processed/draft_picks_2012_2025.csv   (from step1_draft.py)
  data/raw/stathead/<YEAR>_p*.xls / _r*.xls  (Stathead Player Season Finder exports)
Outputs:
  data/processed/nfl_draft_cumulative_2012_2025.csv
  data/processed/nfl_draft_cumulative_2012_2025.xlsx   (sheet "All" + one sheet per year)
  data/processed/by_year/<YEAR>_NFL_Draft_Cumulative_AV.csv
  docs/spot_check_players.csv                           (players verified by hand on PFR; written once)

Rules (see docs/METHODS.md):
  - Season k (1..5) is the season where season year - draft year = k - 1.
  - A played season (year <= LAST_SEASON) with no row for the player -> AV = 0, G = 0.
  - A season not yet played -> blank (NaN).
  - Stathead gives one row per player-season, already totalled across teams for
    traded players (Team shows e.g. "CHI,BAL"), so nothing is averaged.
  - Supplemental-draft picks appear in Stathead but not in the main draft list; they
    are dropped.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from step2_stathead import class_files, read_export  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"
DOCS = ROOT / "docs"
LAST_SEASON = 2025
YEARS = range(2012, 2026)
K = range(1, 6)
GAMES_MISMATCH_KNOWN = {"WillKe05", "KeyeTh00"}


def load_seasons() -> pd.DataFrame:
    frames = []
    for year in YEARS:
        files = class_files(year)
        if not files:
            raise FileNotFoundError(f"No Stathead files for {year}")
        for f in files:
            t = read_export(f)
            # A few rows are roster seasons with 0 games and a blank AV; that is AV 0.
            blank_av = pd.to_numeric(t["AV"], errors="coerce").isna()
            if (blank_av & (pd.to_numeric(t["G"]) > 0)).any():
                raise ValueError(f"{f.name}: blank AV in a season with games played")
            t.loc[blank_av, "AV"] = 0
            frames.append(pd.DataFrame({
                "player_id": t["player_id"],
                "sh_draft_year": t["Draft Year"].astype(int),
                "sh_pick": t["Pick"].astype(int),
                "season": t["Season"].astype(int),
                "season_team": t["Team"],
                "G": pd.to_numeric(t["G"]).astype(int),
                "AV": pd.to_numeric(t["AV"]).astype(int),
            }))
    s = pd.concat(frames, ignore_index=True)
    dups = s.duplicated(["player_id", "season"])
    if dups.any():
        raise ValueError(f"Duplicate player-seasons:\n{s[dups]}")
    return s


def build() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    draft = pd.read_csv(PROCESSED / "draft_picks_2012_2025.csv")
    draft["player_id"] = draft["url"].str.extract(r"/players/./([^/]+)\.htm", expand=False)
    seasons = load_seasons()

    # Drop rows for players who are not in the main draft list (supplemental picks).
    known = seasons.merge(draft[["player_id", "year_drafted", "pick"]], on="player_id", how="left")
    extra = known[known["year_drafted"].isna()]
    seasons = known[known["year_drafted"].notna()].copy()
    mismatch = seasons[(seasons["year_drafted"] != seasons["sh_draft_year"]) | (seasons["pick"] != seasons["sh_pick"])]
    if len(mismatch):
        raise ValueError(f"Stathead draft year/pick disagrees with draft list:\n{mismatch}")

    seasons["k"] = seasons["season"] - seasons["year_drafted"].astype(int) + 1
    av = seasons.pivot(index="player_id", columns="k", values="AV")
    g = seasons.pivot(index="player_id", columns="k", values="G")

    df = draft.copy()
    for k in K:
        df[f"AV{k}"] = df["player_id"].map(av[k]) if k in av else np.nan
        df[f"G{k}"] = df["player_id"].map(g[k]) if k in g else np.nan
        played = df["year_drafted"] + k - 1 <= LAST_SEASON
        no_link = df["player_id"].isna()
        for col in (f"AV{k}", f"G{k}"):
            df.loc[played & ~no_link, col] = df.loc[played & ~no_link, col].fillna(0)
            df.loc[~played, col] = np.nan
    av_cols, g_cols = [f"AV{k}" for k in K], [f"G{k}" for k in K]
    cum_av = df[av_cols].cumsum(axis=1, skipna=False)
    cum_g = df[g_cols].cumsum(axis=1, skipna=False)
    for k in K:
        df[f"CUMYr{k}"] = cum_av[f"AV{k}"]
        df[f"CUMG{k}"] = cum_g[f"G{k}"]
    df["seasons_observed"] = np.minimum(5, LAST_SEASON - df["year_drafted"] + 1)

    cols = (["name", "url", "year_drafted", "round", "pick", "team", "position", "college"]
            + av_cols + g_cols + [f"CUMYr{k}" for k in K] + [f"CUMG{k}" for k in K] + ["seasons_observed"])
    return df[cols + ["player_id", "last_year", "career_g"]], seasons, extra


def check(df: pd.DataFrame) -> list[str]:
    problems = []
    for year, grp in df.groupby("year_drafted"):
        if sorted(grp["pick"]) != list(range(1, grp["pick"].max() + 1)):
            problems.append(f"{year}: picks not 1..max")
        n = min(5, LAST_SEASON - year + 1)
        filled = [f"CUMYr{k}" for k in range(1, n + 1)]
        blank = [f"CUMYr{k}" for k in range(n + 1, 6)]
        if grp[filled].isna().any().any():
            problems.append(f"{year}: blanks in CUMYr1-{n}")
        if blank and grp[blank].notna().any().any():
            problems.append(f"{year}: values in {blank[0]}-CUMYr5, which should be blank")
    # CUMYr may only decrease where PFR assigned a negative AV (a few poor QB seasons).
    cum = df[[f"CUMYr{k}" for k in K]].to_numpy()
    av = df[[f"AV{k}" for k in K[1:]]].to_numpy()
    if ((np.diff(cum, axis=1) < 0) & ~(av < 0)).any():
        problems.append("some CUMYr values decrease without a negative-AV season")
    if (df[[f"G{k}" for k in K]] > 17).any().any():
        problems.append("a season has more than 17 games")
    # Players whose career ended inside the window: career games on the draft page
    # must equal our cumulative games through their last season.
    ended = df[df["last_year"].notna() & (df["last_year"] <= np.minimum(df["year_drafted"] + 4, LAST_SEASON))]
    k_last = (ended["last_year"] - ended["year_drafted"] + 1).astype(int)
    ours = [ended.loc[i, f"CUMG{k}"] for i, k in k_last.items()]
    bad = ended[np.array(ours) != ended["career_g"]]
    # Known: two players whose draft-page career G is higher than the regular-season
    # total. Hand-checked 2026-09-24: Kerwynn Williams' PFR season table sums to 39 G,
    # same as ours (draft page says 40). Thakarius Keyes is the same pattern. Anything
    # beyond these is a real problem.
    bad = bad[~bad["player_id"].isin(GAMES_MISMATCH_KNOWN)]
    if len(bad):
        problems.append(f"{len(bad)} players' career games (draft page) != our cumulative games: "
                        f"{bad['name'].head(10).tolist()}")
    return problems


def same_values(a: pd.DataFrame, b: pd.DataFrame) -> bool:
    """True if two tables hold the same values (Excel reads whole numbers as int, CSV as float)."""
    try:
        pd.testing.assert_frame_equal(a, b, check_dtype=False)
        return True
    except AssertionError:
        return False


def main() -> None:
    df, seasons, extra = build()
    problems = check(df)

    PROCESSED.mkdir(parents=True, exist_ok=True)
    out = df.drop(columns=["player_id", "last_year", "career_g"])
    out.to_csv(PROCESSED / "nfl_draft_cumulative_2012_2025.csv", index=False)
    by_year = PROCESSED / "by_year"
    by_year.mkdir(exist_ok=True)
    for year, grp in out.groupby("year_drafted"):
        grp.to_csv(by_year / f"{year}_NFL_Draft_Cumulative_AV.csv", index=False)
    # Excel files embed a timestamp, so only rewrite the workbook when the data changed.
    xlsx = PROCESSED / "nfl_draft_cumulative_2012_2025.xlsx"
    if not xlsx.exists() or not same_values(pd.read_excel(xlsx, sheet_name="All"),
                                            pd.read_csv(PROCESSED / "nfl_draft_cumulative_2012_2025.csv")):
        with pd.ExcelWriter(xlsx) as xl:
            out.to_excel(xl, sheet_name="All", index=False)
            for year, grp in out.groupby("year_drafted"):
                grp.to_excel(xl, sheet_name=str(year), index=False)

    # Players to check by hand: traded seasons, never played, stars, late picks.
    traded = seasons[seasons["season_team"].str.contains(",", na=False)]["player_id"].drop_duplicates()
    never = df[(df["seasons_observed"] >= 5) & (df["CUMG5"] == 0)]["player_id"]
    picks = pd.concat([
        traded.sample(4, random_state=1),
        never.sample(2, random_state=1),
        df[df["year_drafted"] == 2017].nsmallest(1, "pick")["player_id"],
        df[df["name"] == "Gareon Conley"]["player_id"],
        df[df["player_id"].isin(GAMES_MISMATCH_KNOWN)]["player_id"],
        df[df["name"] == "Jared Goff"]["player_id"],  # negative AV rookie season
        df[(df["year_drafted"] == 2025)].sample(1, random_state=1)["player_id"],
        df[(df["round"] == 7) & (df["year_drafted"] <= 2021)].sample(1, random_state=2)["player_id"],
    ]).drop_duplicates()
    spot = df[df["player_id"].isin(picks)][["name", "url", "year_drafted", "pick"]
                                           + [f"AV{k}" for k in K] + [f"G{k}" for k in K]]
    spot.insert(4, "why", spot.index.map(lambda i: "traded season" if df.loc[i, "player_id"] in set(traded)
                                          else "never played" if df.loc[i, "player_id"] in set(never) else "other"))
    spot["matches_pfr (fill Y/N)"] = ""
    # One-time list for the hand check; results are in docs/spot_check_results.md. Don't overwrite.
    if not (DOCS / "spot_check_players.csv").exists():
        spot.to_csv(DOCS / "spot_check_players.csv", index=False)

    # Summary
    print(f"Players: {len(df)} | player-seasons used: {len(seasons)}")
    print(f"Dropped supplemental-draft players: {sorted(extra['player_id'].unique())}")
    played_draft_page = df["career_g"].notna()
    no_rows = df[played_draft_page & ~df["player_id"].isin(seasons["player_id"])]
    print("\nPlayed per draft page but no season in our window (should be later debuts):")
    print(no_rows[["year_drafted", "pick", "name", "last_year", "career_g"]].to_string(index=False))

    obs = df[df["seasons_observed"] >= 1].copy()
    last_cumg = obs.apply(lambda r: r[f"CUMG{int(r['seasons_observed'])}"], axis=1)
    summary = df.groupby("year_drafted").agg(players=("name", "size"), seasons_observed=("seasons_observed", "first"))
    summary["pct_no_games_so_far"] = (last_cumg == 0).groupby(obs["year_drafted"]).mean().mul(100).round(1)
    print("\n", summary.to_string())
    complete = df[df["year_drafted"] <= 2021]
    print("\nMean CUMYr5 by round, 2012-2021:")
    print(complete.groupby("round")["CUMYr5"].agg(["mean", "median", "size"]).round(1).to_string())
    print("\nChecks:", "all passed" if not problems else "")
    for p in problems:
        print("  PROBLEM:", p)


if __name__ == "__main__":
    main()
