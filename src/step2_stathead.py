"""Step 2: move one draft class's Stathead exports from Downloads into the project and check them.

This step is manual: the exports are downloaded by hand from Stathead (see
data/raw/stathead/CHECKLIST.md). run_all.py does not run it.

Usage:  py src/step2_stathead.py 2018

Takes every sportsref_download*.xls in ~/Downloads, checks that it belongs to
that draft class, and moves it into data/raw/stathead/ as:
  <YEAR>_r<ROUND>.xls  if the file holds a single draft round (one query per round)
  <YEAR>_p<N>.xls      otherwise (paged query, used for 2012-2017)
Refuses to move anything if a file belongs to another class.

Why rounds: Stathead pages hold 200 rows and tied rows can swap order between
page requests, so a paged query can show one row twice and skip another. A
single round's first five seasons fit on one page, so nothing can be skipped.
"""

import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DOWNLOADS = Path.home() / "Downloads"
STATHEAD = ROOT / "data" / "raw" / "stathead"
PAGE_ROWS = 200


def read_export(path: Path) -> pd.DataFrame:
    """Read one Stathead 'Excel workbook' export (an HTML table) with player IDs."""
    raw = path.read_text(encoding="utf-8")
    ids = re.findall(r'data-append-csv="([^"]+)"', raw)
    t = pd.read_html(path)[0]
    t = t[t["Rk"] != "Rk"].reset_index(drop=True)  # drop repeated header rows
    if len(ids) != len(t):
        raise ValueError(f"{path.name}: {len(t)} rows but {len(ids)} player IDs")
    t["player_id"] = ids
    return t


def class_files(year: int) -> list[Path]:
    """The files that make up a class: round files if any exist, else page files."""
    rounds = sorted(STATHEAD.glob(f"{year}_r*.xls"))
    return rounds or sorted(STATHEAD.glob(f"{year}_p*.xls"),
                            key=lambda p: int(p.stem.split("_p")[1]))


def main(year: int) -> None:
    new = sorted(DOWNLOADS.glob("sportsref_download*.xls"), key=lambda p: p.stat().st_mtime)
    targets = []
    for f in new:
        t = read_export(f)
        years = set(t["Draft Year"].astype(int))
        if years != {year}:
            sys.exit(f"{f.name} has draft years {sorted(years)}, not {year}. Nothing moved.")
        rounds = set(t["Round"].astype(int))
        if len(rounds) == 1:
            dest = STATHEAD / f"{year}_r{rounds.pop()}.xls"
        else:
            n = len(list(STATHEAD.glob(f"{year}_p*.xls"))) + 1 + sum(d.stem.startswith(f"{year}_p") for d in targets)
            dest = STATHEAD / f"{year}_p{n}.xls"
        if dest.exists() or dest in targets:
            sys.exit(f"{f.name} would overwrite {dest.name} (exported twice?). Nothing moved.")
        targets.append(dest)
    for f, dest in zip(new, targets):
        f.rename(dest)
    if not new:
        print("No sportsref_download*.xls files in Downloads.")

    files = class_files(year)
    d = pd.concat([read_export(f).assign(file=f.name) for f in files], ignore_index=True)
    seasons = sorted(d["Season"].astype(int).unique())
    sizes = d.groupby("file", sort=False).size()

    print(f"{year}: {len(files)} files {sizes.to_dict()}")
    print(f"  player-seasons: {len(d)} | players: {d['player_id'].nunique()} | seasons: {seasons[0]}-{seasons[-1]}")
    ok = True
    dups = int(d.duplicated(["player_id", "Season"]).sum())
    if dups:
        ok = False
        print(f"  PROBLEM: {dups} duplicate player-seasons (pages shifted; the same number of rows is missing)")
    if not set(seasons) <= set(range(year, min(year + 4, 2025) + 1)):
        ok = False
        print(f"  PROBLEM: seasons outside {year}-{min(year + 4, 2025)}")
    if files[0].stem.startswith(f"{year}_r"):
        missing = sorted(set(range(1, 8)) - {int(f.stem.split("_r")[1]) for f in files})
        if missing:
            ok = False
            print(f"  STILL NEEDED: rounds {missing}")
        full = [n for n, s in sizes.items() if s >= PAGE_ROWS]
        if full:
            ok = False
            print(f"  PROBLEM: {full} have {PAGE_ROWS} rows, so the round did not fit on one page")
    elif sizes.iloc[-1] == PAGE_ROWS:
        ok = False
        print("  NOTE: last page has exactly 200 rows, so there may be another page.")
    if ok:
        print("  OK: looks complete.")


if __name__ == "__main__":
    main(int(sys.argv[1]))
