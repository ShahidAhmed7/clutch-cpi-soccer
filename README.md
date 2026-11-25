# Measuring Clutch Performance in Soccer When the Result Is on the Line

Code, data and results for our paper submitted to the Ubi-Media 2027 Special Session on Sport Data Analysis (Springer CCIS). CPI is reported in league points per 90 minutes and as season totals, for players with at least 900 minutes.

**Authors:** Kazi Shahid Ahmed Galib, Shihab Shahriar, Tarvir Anjum Aditto

The next goal can go either way, so every moment in a match has two prices: what a team gains if it scores next, and what it loses if it concedes next. The **Clutch Performance Index (CPI)** values each on-ball action in league points by weighting its effect on scoring and on conceding by those two prices. Summed over a season, this gives a ledger that can be checked against the league table, and a rating that puts attackers, defenders and keepers on one scale.

## Key results (2017/18, five top European leagues)

<!-- results:start -->
| Result | Value |
|---|---|
| Real results, one goal up after 80': cost of conceding, gain from scoring | 1.58, 0.23 league points |
| Model stakes in the same moments | 1.56, 0.23 |
| Season ledger: net clutch value against real points | slope 1.09, r .98 |
| Same player, high vs low stakes: pass completion, forward progress | -1.0 points, +0.53 m |
| Coaches, one goal up: odds of being taken off per SD of goal-prevention rating | .88 (0.83 to 0.94) |
| Coaches, one goal down: same, per SD of attacking rating | .84 (0.78 to 0.89) |
| Split-half reliability of CPI: forwards, midfielders, defenders, keepers | .32, .48, .41, .17 |
<!-- results:end -->

Every number in the paper comes from `results/`; `results/claims.json` lists them. CPI describes who delivered in a season. It is less repeatable than unweighted action value, and it is not a claim about a stable clutch trait.

## Reproduce

Python 3.12. `requirements.txt` pins every package.

```bash
git clone https://github.com/ShahidAhmed7/clutch-cpi-soccer && cd clutch-cpi-soccer
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python run_all.py --quick   # from the stored per-action data: ratings, tests, figures, paper numbers (minutes on a laptop)
python run_all.py           # from the raw events; Stage 3 uses a GPU if present (about 25 minutes on a Colab T4)
pytest tests                # data contracts: scores match official results, own goals counted, geometry correct
```

Build the paper from `paper/` with `tectonic cpi_ccis.tex` (Springer CCIS/LNCS format, llncs v2.24 included), or upload the folder to Overleaf. Fig. 1 is `paper/figures/method.pdf`, a vector conversion of `paper/figures/method.svg`.

## Run CPI on another league

`run_league.py` computes CPI for any league, season or set of matches and writes a report with tables and figures. It uses the same code as the paper after a provider-specific loading step.

```bash
python run_league.py train                     # once: fit the models on all five 2017/18 leagues and save them in models/
python run_league.py fetch --competition "1. Bundesliga" --season 2023/2024     # Hudl StatsBomb open data, no account needed
python run_league.py run --data data_in/statsbomb --competition "1. Bundesliga" --season 2023/2024
```

Results go to `runs/<name>/`: `report.md`, `player_ratings.csv`, `action_values.parquet`, `top_clutch_actions.csv`, `goalkeepers.csv`, `figures/` and `run_info.json`.

| `--provider` | Data | Conversion to actions |
|---|---|---|
| `statsbomb` | Hudl StatsBomb Open Data or API export (Events v4, Lineups v2, Matches v3; data version 1.1.0): the open-data folder layout, or a flat folder of `events/` and `lineups/` | socceraction's StatsBomb converter |
| `wyscout-v2` | Wyscout v2 (legacy API, and the public 2017/18 release): `events_<x>.json` + `matches_<x>.json`, or one API file per match | socceraction's Wyscout converter, as in the paper |
| `wyscout-v3` | Wyscout API v3 (current): one file per match from `matches/<id>/events?fetch=teams,players,match,substitutions,formations` | kloppy 3.19, then socceraction's kloppy converter (experimental) |
| `kloppy:<loader>` | Anything kloppy reads (`opta`, `statsperform`, `sportec`, `datafactory`, `metrica`, `impect`, ...): a CSV with one row per match listing the loader's file arguments | kloppy, then socceraction's kloppy converter (experimental) |
| `tables` | Your own tables as CSV or parquet (actions in SPADL, matches, players, teams, appearances; optional shots and red cards); columns in `cpi/league/bundle.py` | none needed |

Without `--provider` the format is detected from the files. Options: `--models auto|pretrained|retrain`, `--min-minutes`, `--boot`, `--out`, `--save-bundle`.

**Models.** By default the action values come from models fitted on the five 2017/18 leagues, and the report shows their accuracy on the new data. For a full league season of 300 or more matches, `--models auto` retrains them on that data with 5-fold cross-fitting by match and refits the stakes model.

With these pretrained models, data from another provider is adjusted in two ways, and the report states both:
- **Carries.** StatsBomb records every carry, so its native carries are replaced by socceraction's standard dribbles, as for Wyscout data. On Bundesliga 2023/24 this brought the scoring model to near-perfect calibration.
- **Recalibration.** Predicted scoring, conceding and post-shot xG chances get a two-parameter logistic recalibration, cross-fitted by match. On Bundesliga 2023/24 this left ROC-AUC unchanged (scoring .78, conceding .82, against .77 and .75 in the paper's held-out leagues). Post-shot xG then predicted 109.9 goals against 110 scored.

`--keep-carries`, `--no-harmonize` and `--no-recalibrate` turn these off.

**Coverage.** Open-data club seasons usually cover one team's matches (for example, Bayer Leverkusen in the 2023/24 Bundesliga). The report then rates that team's players only, and skips analyses that need a whole league: the season ledger, repeatability across teams, and the substitution analysis.

**Notes.**
- Providers define events differently, so CPI values are comparable within a dataset more than across providers.
- socceraction 1.5.3's kloppy converter fails on kloppy 3.16 and later, and its orientation now flips the second half. `cpi/league/kloppy_spadl.py` corrects both.

## Layout

| Path | Contents |
|---|---|
| `cpi/` | The method: events, match state, labels, value rule, stakes, keepers, ratings |
| `cpi/league/`, `run_league.py` | CPI for other leagues and providers: loaders, saved models, engine, report |
| `models/` | Models fitted on all five 2017/18 leagues, used by `run_league.py` |
| `pipeline/` | One script per stage, listed below |
| `tests/` | Data contracts and tests against known answers |
| `WyScout/` | Raw public data (CC BY 4.0), unchanged |
| `data/` | Stored per-action values and match tables, used by `--quick` |
| `results/` | Every table behind the paper |
| `paper/` | LaTeX source, figures, and numbers generated from `results/` |

| Stage | Does |
|---|---|
| `s1_actions` | Raw events to checked actions with the reference converter; score with own goals, red cards, minutes |
| `s2_features` | Reference game-state features in the acting team's direction, and labels |
| `s3_action_values` | CatBoost pair, leave one league out; resumable, and reuses a fold only if its inputs are unchanged |
| `s4_stakes` | One outcome model on a team-by-frame panel; `W`, `L+`, `L-` |
| `s5_values` | Clutch value of each action; keeper points saved from held-out post-shot xG |
| `s6_ratings` | CPI per 90, intervals, shrinkage, reliability, discrimination |
| `s7_validation` | Real results against the stakes; the match and season ledger |
| `s8_pressure` | Same player at high and low stakes |
| `s9_coaches` | Do substitutions agree with the ratings? |
| `s10_data_choices` | Effect of each data choice, with one change at a time |
| `s11_stories` | Biggest moments, goals against points, lead protectors, keepers, margins |
| `s12_sensitivity` | Seed, booster, label window, team strength in the stakes, minutes threshold |
| `s13_figures`, `s14_claims` | Figures and the paper's numbers |

An earlier version of this work, submitted as an abstract, is at commit `e76eb8f`. The method and results here replace it.

## Data and licences

Raw data: Pappalardo et al., *A public data set of spatio-temporal match events in soccer competitions*, Scientific Data 6, 236 (2019), https://doi.org/10.1038/s41597-019-0247-7, CC BY 4.0. See [`DATA_LICENSE.md`](DATA_LICENSE.md). Code: MIT ([`LICENSE`](LICENSE)). Conversion, features and the value rule use [socceraction](https://github.com/ML-KULeuven/socceraction); other providers are read with [kloppy](https://github.com/PySport/kloppy).

Data you analyse with `run_league.py` keeps its own licence. Hudl StatsBomb open data (https://github.com/hudl/open-data) asks that published work names StatsBomb as the data source and uses their logo. Downloaded data (`data_in/`) and run outputs (`runs/`) are not tracked.
