"""Stage 12. How much do the rankings depend on modelling choices the intervals do not cover?

    python pipeline/s12_sensitivity.py

Compares per-90 CPI within position across seed, booster, label window and team strength in the stakes, and shows the
effect of the minutes threshold. The seed and booster rows are the noise floor for reading results/data_choices.csv.
"""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cpi import config as C, ratings, raw, stakes  # noqa: E402

BASE = 'full_catboost_s42'


def per90(tag, appearances, players):
    v = pd.read_parquet(C.WORK / f'values_{tag}.parquet', columns=['game_id', 'team_id', 'player_id', 'type_name', 'V', 'attack', 'defend', 'value', 'V_scoring_only', 'goal', 'own_goal', 'assist'])
    s = pd.read_parquet(C.WORK / f'keeper_shots_{tag}.parquet')
    pl = ratings.per90(ratings.player_match_table(v, s, appearances))
    return pl[pl['minutes'] >= C.MIN_MINUTES].join(players[['position', 'name']])


def compare(a, b, label, base_label):
    row = {'comparison': label, 'against': base_label}
    for p in C.POSITIONS:
        x = a[a['position'] == p]
        y = b.reindex(x.index)
        row[f'rho_{p}'] = spearmanr(x['V_per90'], y['V_per90'], nan_policy='omit').statistic
        row[f'top10_{p}'] = len(set(x['V_per90'].nlargest(10).index) & set(y['V_per90'].nlargest(10).index))
    return row


def elo_weighted(appearances, players):
    """Team strength in the stakes: the red-card slot of the outcome model carries the Elo gap, in 100-point steps from -2 to +2."""
    panel = pd.read_parquet(C.WORK / 'panel.parquet')
    elo = pd.read_csv(C.WORK / 'elo_matches.csv')
    long = pd.concat([elo[['game_id', 'home_team_id', 'elo_home_pre', 'elo_away_pre']].set_axis(['game_id', 'team_id', 'e', 'o'], axis=1),
                      elo[['game_id', 'away_team_id', 'elo_away_pre', 'elo_home_pre']].set_axis(['game_id', 'team_id', 'e', 'o'], axis=1)])
    long['elo'] = ((long['e'] - long['o']) / 100).round().clip(-2, 2).astype(int)
    p = panel.drop(columns='adv').merge(long[['game_id', 'team_id', 'elo']], on=['game_id', 'team_id']).rename(columns={'elo': 'adv'})
    v = pd.read_parquet(C.WORK / f'values_{BASE}.parquet')
    v = v.merge(long[['game_id', 'team_id', 'elo']], on=['game_id', 'team_id'], how='left')
    Lp, Lm = np.zeros(len(v)), np.zeros(len(v))
    for lg in C.LEAGUES:
        spec = ('base', 'red')
        table = stakes.stake_table(stakes.fit_rate_model(p[p['league'] != lg], spec), spec)
        m = (v['league'] == lg).values
        t = stakes.attach(v[m], table, adv='elo')
        Lp[m], Lm[m] = t['L_plus'].values, t['L_minus'].values
    v['attack'], v['defend'] = v['dP_scores'] * Lp, -v['dP_concedes'] * Lm
    v['V'] = v['attack'] + v['defend']
    s = pd.read_parquet(C.WORK / f'keeper_shots_{BASE}.parquet')
    pl = ratings.per90(ratings.player_match_table(v, s, appearances))
    return pl[pl['minutes'] >= C.MIN_MINUTES].join(players[['position', 'name']])


def main():
    appearances = pd.read_parquet(C.WORK / 'appearances.parquet')
    players = raw.load_players().set_index('player_id')
    for tag in ('full_catboost_s43', 'full_hgb_s42_k5', 'full_hgb_s42_k15'):
        if not (C.WORK / f'values_{tag}.parquet').exists():
            subprocess.run([sys.executable, str(C.ROOT / 'pipeline/s5_values.py'), '--tag', tag], check=True, capture_output=True)
    base, hgb = per90(BASE, appearances, players), per90('check_method', appearances, players)
    rows = [compare(base, per90('full_catboost_s43', appearances, players), 'another seed', 'CatBoost seed 42'),
            compare(base, hgb, 'another booster', 'CatBoost seed 42'),
            compare(hgb, per90('full_hgb_s42_k5', appearances, players), 'window of 5 actions', 'same booster, window 10'),
            compare(hgb, per90('full_hgb_s42_k15', appearances, players), 'window of 15 actions', 'same booster, window 10'),
            compare(base, elo_weighted(appearances, players), 'team strength in the stakes', 'CatBoost seed 42')]
    res = pd.DataFrame(rows)
    res.to_csv(C.RESULTS / 'sensitivity.csv', index=False)
    print(res.round(3).T.to_string())
    r = base.copy()
    rows = []
    for th in (1350, 1800):
        keep = r[r['minutes'] >= th]
        over = {p: len(set(r[r['position'] == p]['V_per90'].nlargest(10).index) & set(keep[keep['position'] == p]['V_per90'].nlargest(10).index)) for p in C.POSITIONS}
        rows.append({'minutes': th, **over})
        print(f'minutes threshold {th}: top-10 overlap with the 900 list', over)
    pd.DataFrame(rows).to_csv(C.RESULTS / 'minutes_threshold.csv', index=False)
    print('stage 12 done')


if __name__ == '__main__':
    main()
