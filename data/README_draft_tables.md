# Draft tables, 2012-2025

One CSV per draft year, collected 2026-09-24 from
`https://www.pro-football-reference.com/years/<YEAR>/draft.htm`.

PFR now serves a Cloudflare challenge to scripts, so these were read from the
`drafts` table of each page as loaded in a normal browser (one page every
~8 seconds), not downloaded by `src/fetch.py`.

Columns (from the table's `data-stat` attributes):

| column | PFR data-stat | meaning |
|---|---|---|
| round | draft_round | draft round |
| pick | draft_pick | overall pick |
| team | team | drafting team |
| name | player | player name |
| url | link on player name | PFR profile URL |
| position | pos | position listed at draft |
| age | age | age at draft |
| last_year | year_max | last season played (blank = never played) |
| career_av | career_av | career AV, all teams, as of collection date |
| draft_av | draft_av | AV accrued for drafting team |
| career_g | g | career games (blank = never played) |
| college | college_id | college |

`career_av` and `career_g` are career totals, kept only for cross-checking.
The analysis uses per-season AV and games from player pages.
