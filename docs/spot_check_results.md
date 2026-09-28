# Hand spot-check against Pro-Football-Reference player pages

Checked 2026-09-24 by Navya Pathuri. For each player, G and AV for each of the
first five seasons in `nfl_draft_cumulative_2012_2025.csv` were compared with
the regular-season table on the player's PFR page (the "2TM" total row for
seasons with two teams).

Result: **11 of 11 checked players match exactly, 0 errors.**

| Player | Drafted | Case tested | Result |
|---|---|---|---|
| Kerwynn Williams | 2013 #230 | traded 2013 (IND, SDG) | match |
| John Hughes | 2012 #87 | traded 2016 (CLE, TAM) | match |
| Charles Johnson | 2013 #216 | missed seasons 2013 and 2017 filled as 0 | match |
| Marc Anthony | 2013 #247 | never played (all 0) | match |
| Jeremy Gallon | 2014 #244 | never played (all 0) | match |
| MyCole Pruitt | 2015 #143 | traded 2016 (MIN, CHI) | match |
| Marqui Christian | 2016 #167 | traded 2020 (CHI, NYJ) | match |
| Gareon Conley | 2017 #24 | traded 2019 (OAK, HOU); old data showed 2.67 | match (AV 4) |
| Donovan Peoples-Jones | 2020 #187 | traded 2023 (CLE, DET) | match |
| Thakarius Keyes | 2020 #237 | traded 2021 (CHI, IND) | match |
| Devin Neal | 2025 #184 | rookie; seasons 2-5 blank | match |
| Jared Goff | 2016 #1 | negative AV rookie season | not checked |
| Myles Garrett | 2017 #1 | star | not checked |

Notes:
- Gareon Conley 2019: PFR shows OAK AV 2, HOU AV 2, 2TM AV 4. The old dataset's 2.67
  is (2 + 2 + 4) / 3, the average of all three rows. The new dataset has 4.
- Kerwynn Williams and Thakarius Keyes: the season tables sum to 39 and 13 games,
  equal to ours. The career summary box and draft page show 40 and 18; that count
  includes something the regular-season table does not (likely playoff games).
  The dataset uses regular-season games, like AV.

Decision (2026-09-24): negative AV seasons are kept as PFR reports them (5 seasons,
e.g. Jared Goff 2016 = -2), so CUMYr can dip in those seasons.
