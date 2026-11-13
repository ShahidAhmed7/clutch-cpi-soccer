"""Stage 10. How much do silent data choices move the results? Same booster everywhere, one choice changed at a time.

    python pipeline/s10_data_choices.py

Rule fixed in advance: a choice matters if within-position rank correlation with the method falls below 0.90, or top-10
overlap below 8 of 10, or held-out AUC moves by more than 0.01.
"""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cpi import config as C, ratings, raw, stakes  # noqa: E402

RUNS = [('method', 'full_hgb_s42', 'full', []), ('mirrored_coordinates', 'mirrored_hgb_s42', 'mirrored', []),
        ('no_own_goals', 'no_own_goals_hgb_s42', 'no_own_goals', ['--tables', 'stakes_tables_no_own.parquet']),
        ('no_phase_break', 'full_hgb_s42', 'full', ['--no-phase-break']), ('no_set_piece_baselines', 'full_hgb_s42', 'full', ['--no-baselines']),
        ('no_phase_break_no_baselines', 'full_hgb_s42', 'full', ['--no-phase-break', '--no-baselines'])]


def main():
    matches = pd.read_parquet(C.WORK / 'matches.parquet')
    reds = pd.read_parquet(C.WORK / 'reds.parquet')
    acts = pd.concat([pd.read_parquet(C.WORK / f'actions_{lg}.parquet', columns=['game_id', 'team_id', 'goal', 'own_goal', 'elapsed']) for lg in C.LEAGUES], ignore_index=True)
    panel = stakes.build_panel(acts.assign(own_goal=False), matches, reds)
    tabs = [stakes.stake_table(stakes.fit_rate_model(panel[panel['league'] != lg])).assign(held_out=lg) for lg in C.LEAGUES]
    pd.concat(tabs, ignore_index=True).to_parquet(C.WORK / 'stakes_tables_no_own.parquet', index=False)

    appearances = pd.read_parquet(C.WORK / 'appearances.parquet')
    players = raw.load_players().set_index('player_id')
    ratings_by = {}
    for name, tag, variant, extra in RUNS:
        out = f'check_{name}'
        subprocess.run([sys.executable, str(C.ROOT / 'pipeline/s5_values.py'), '--tag', tag, '--variant', variant, '--out', out] + extra, check=True, capture_output=True)
        v = pd.read_parquet(C.WORK / f'values_{out}.parquet', columns=['game_id', 'team_id', 'player_id', 'type_name', 'V', 'attack', 'defend', 'value', 'V_scoring_only', 'goal', 'own_goal', 'assist'])
        s = pd.read_parquet(C.WORK / f'keeper_shots_{out}.parquet')
        pl = ratings.per90(ratings.player_match_table(v, s, appearances))
        ratings_by[name] = pl[pl['minutes'] >= C.MIN_MINUTES].join(players[['position', 'name']])
        print('done', name, flush=True)
    base = ratings_by['method']
    auc = {r: pd.read_csv(C.WORK / f'metrics_{t}.csv') for r, t in (('method', 'full_hgb_s42'), ('mirrored_coordinates', 'mirrored_hgb_s42'), ('no_own_goals', 'no_own_goals_hgb_s42'))}
    rows = []
    for name, tab in ratings_by.items():
        row = {'choice': name}
        a = auc.get(name, auc['method']).groupby('target')['auc'].mean()
        row['auc_scores'], row['auc_concedes'] = a['scores'], a['concedes']
        for p in C.POSITIONS:
            x = base[base['position'] == p]
            y = tab.reindex(x.index)
            row[f'rho_{p}'] = spearmanr(x['V_per90'], y['V_per90'], nan_policy='omit').statistic
            row[f'top10_{p}'] = len(set(x['V_per90'].nlargest(10).index) & set(y['V_per90'].nlargest(10).index))
            row[f'mean_{p}'] = y['V_per90'].mean()
        rows.append(row)
    res = pd.DataFrame(rows)
    res.to_csv(C.RESULTS / 'data_choices.csv', index=False)
    print(res.round(3).T.to_string())
    print('stage 10 done')


if __name__ == '__main__':
    main()
