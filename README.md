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

`run_league.py` computes CPI for any league, season or set of matches. You give it event data from StatsBomb, Wyscout, Opta or another provider, and it writes a report with player ratings, the most valuable actions, goalkeeper ratings and figures. It uses the same method as the paper. The fitted models are included in `models/`, so no training is needed.

### 1. Set up (once)

```bash
git clone https://github.com/ShahidAhmed7/clutch-cpi-soccer && cd clutch-cpi-soccer
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt                         # Python 3.12
```

### 2. Get data

**Free data (Hudl StatsBomb open data, no account needed).** Download a season by its name:

```bash
python run_league.py fetch --competition "1. Bundesliga" --season 2023/2024
python run_league.py fetch --competition "La Liga" --season 2019/2020
```

The files are saved in `data_in/statsbomb/`. If the name is not in the open data, the command prints every competition and season that is.

**Your own data.** Put the files in one folder and pass that folder to `--data` in step 3. The format is detected automatically.

| Your data | Put in the folder | `--provider` |
|---|---|---|
| Hudl StatsBomb (open data or API export) | `events/<match_id>.json` and `lineups/<match_id>.json`, plus `matches/<competition>/<season>.json` or one `matches.json` | `statsbomb` |
| Wyscout v3, the current API | one JSON file per match from `matches/<id>/events?fetch=teams,players,match,substitutions,formations` | `wyscout-v3` |
| Wyscout v2, the legacy API or the 2017/18 public release | `events_<name>.json` and `matches_<name>.json` (plus `players.json`, `teams.json`), or one API file per match | `wyscout-v2` |
| Opta, Stats Perform, Sportec/DFL, Metrica, Impect, DataFactory | the provider's files, plus a CSV listing each match's files (example below) | `kloppy:opta`, `kloppy:sportec`, ... |
| Any other source, already converted | `actions`, `matches`, `players`, `teams`, `appearances` as CSV or parquet; columns are listed in `cpi/league/bundle.py` | `tables` |

For Opta, the CSV has one row per match with kloppy's file arguments. Paths are relative to the CSV:

```csv
f7_data,f24_data
f7/match1.xml,f24/match1.xml
f7/match2.xml,f24/match2.xml
```

### 3. Run

```bash
python run_league.py run --data data_in/statsbomb --competition "1. Bundesliga" --season 2023/2024
```

More examples:

```bash
python run_league.py run --data data_in/statsbomb --competition "La Liga" --season 2019/2020
python run_league.py run --data WyScout --competition Spain                        # the paper's raw data, one league
python run_league.py run --data my_wyscout_v3_matches/ --name "Serie A 2023-24"
python run_league.py run --data opta/matches.csv --provider kloppy:opta --name "Eredivisie 2022-23"
python run_league.py list --data data_in/statsbomb                                 # what a data folder contains
```

A one-team season takes about 15 seconds. A full league season with retrained models takes about 10 minutes. The terminal shows each step and ends with a summary:

```text
provider: statsbomb
StatsBomb: 34 matches in data_in/statsbomb/data
recalibrated: scoring a=0.11 b=1.07; conceding a=-1.64 b=0.78
15 players with at least 900 minutes (focus team only)
1. Bundesliga 2023/2024: 34 matches, 80,213 actions, models pretrained, scoring AUC 0.776, 15 rated players, 11s
open runs/1_Bundesliga_2023_2024/report.md
```

### 4. Read the results

Everything goes to `runs/<name>/`. Start with `report.md`.

| File | Contents |
|---|---|
| `report.md` | The full report: data checks, model accuracy, what a goal is worth, player ratings by position, top actions, goalkeepers, and what was not run and why |
| `player_ratings.csv` | Rated players: CPI per 90 with a 95% interval, adjusted rating, attacking and prevention parts, VAEP, minutes, goals |
| `all_players_per90.csv` | Every player with minutes, including those below the rating threshold |
| `top_clutch_actions.csv`, `top_preventive_actions.csv` | The most valuable actions and the most valuable defensive actions, with match, minute and score |
| `goalkeepers.csv` | Shots faced, post-shot xG, goals prevented and points saved |
| `goals_vs_value.csv`, `goal_values_by_context.csv` | Goals against their value in league points, and goals by match situation |
| `action_values.parquet` | Every action with its probabilities, stakes and values |
| `figures/` | Stakes over the match, top players with intervals, goals against value, and the season ledger for full leagues (PNG and PDF) |
| `run_info.json` | Every number in the report, for use in other code |

### Options

| Option | Default | Meaning |
|---|---|---|
| `--name` | from the data | Report title and output folder name |
| `--out` | `runs/<name>/` | Output folder |
| `--provider` | detected | Data format, see step 2 |
| `--competition`, `--season` | | Which season to use when the folder has several (StatsBomb: name or id; Wyscout v2: file name, e.g. `Spain`) |
| `--models` | `auto` | `pretrained` uses the saved models; `retrain` fits new ones on your data (needs a full league season); `auto` retrains for full leagues of 300+ matches |
| `--min-minutes` | 900 | Minutes a player needs to be rated (lowered automatically for short datasets) |
| `--boot` | 1000 | Match resamples for the 95% intervals |
| `--workers` | 6 | Parallel processes |
| `--save-bundle` | off | Also save the converted tables, to rerun later with `--provider tables` |
| `--keep-carries`, `--no-harmonize`, `--no-recalibrate` | off | Turn off the adjustments for other providers (below) |

### Good to know

- **Free club data usually covers one team.** StatsBomb's open club seasons cover one team's matches, for example Bayer Leverkusen in the 2023/24 Bundesliga. The report then rates that team's players only, and skips what needs a whole league: the season ledger, stakes against real results, and repeatability. With a full league season, everything runs.
- **Other providers are adjusted to the models.** The saved models were fitted on Wyscout data. StatsBomb records every carry, so carries are replaced by the standard dribbles used for Wyscout. Predicted chances are then recalibrated on your data, cross-fitted by match. On Bundesliga 2023/24, this put predicted and observed scoring rates level, and post-shot xG predicted 110 goals against 110 scored. The report states every adjustment it made.
- **Experimental routes.** Wyscout v3 and the kloppy providers are converted through kloppy, a route socceraction validates only for StatsBomb. Ratings are comparable within one dataset more than across providers.
- **Refitting the models.** `python run_league.py train` refits the saved models. It needs the `work/` files from `python run_all.py`.

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
