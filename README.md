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
| Season ledger: net clutch value against real points | slope 1.09, r .975 |
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

## Layout

| Path | Contents |
|---|---|
| `cpi/` | The method: events, match state, labels, value rule, stakes, keepers, ratings |
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

Raw data: Pappalardo et al., *A public data set of spatio-temporal match events in soccer competitions*, Scientific Data 6, 236 (2019), https://doi.org/10.1038/s41597-019-0247-7, CC BY 4.0. See [`DATA_LICENSE.md`](DATA_LICENSE.md). Code: MIT ([`LICENSE`](LICENSE)). Conversion, features and the value rule use [socceraction](https://github.com/ML-KULeuven/socceraction).
