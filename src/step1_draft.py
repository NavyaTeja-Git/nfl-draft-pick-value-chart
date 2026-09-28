"""Step 1: combine the per-year draft tables into one checked draft list.

Input:  data/raw/draft/<YEAR>.csv, 2012-2025 (see data/raw/draft/README.md)
Output: data/processed/draft_picks_2012_2025.csv

Stops with an error if any year has missing, duplicate or out-of-order picks.
"""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw" / "draft"
OUT = ROOT / "data" / "processed" / "draft_picks_2012_2025.csv"
YEARS = range(2012, 2026)


def load() -> pd.DataFrame:
    frames = []
    for year in YEARS:
        df = pd.read_csv(RAW / f"{year}.csv")
        df.insert(0, "year_drafted", year)
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    df["has_url"] = df["url"].notna()
    return df


def check(df: pd.DataFrame) -> None:
    problems = []
    for year, g in df.groupby("year_drafted"):
        picks = g["pick"]
        if picks.duplicated().any():
            problems.append(f"{year}: duplicate picks {sorted(picks[picks.duplicated()])}")
        missing = set(range(1, picks.max() + 1)) - set(picks)
        if missing:
            problems.append(f"{year}: missing picks {sorted(missing)}")
        if not g.sort_values("pick")["round"].is_monotonic_increasing:
            problems.append(f"{year}: rounds out of order")
        if not 253 <= len(g) <= 262:
            problems.append(f"{year}: unusual pick count {len(g)}")
    dup_urls = df["url"].dropna().duplicated()
    if dup_urls.any():
        problems.append(f"same player URL in more than one row: {df.loc[dup_urls[dup_urls].index, 'url'].tolist()}")
    if problems:
        raise ValueError("Draft data failed checks:\n  " + "\n  ".join(problems))


def main() -> None:
    df = load()
    check(df)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)

    summary = df.groupby("year_drafted").agg(
        picks=("pick", "size"),
        no_profile_link=("has_url", lambda s: int((~s).sum())),
        never_played=("career_g", lambda s: int(s.isna().sum())),
    )
    print(summary.to_string())
    print(f"\nAll checks passed. {len(df)} players -> {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
