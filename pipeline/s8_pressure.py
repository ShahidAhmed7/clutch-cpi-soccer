"""Stage 8. What happens to a player when the result goes on the line?

    python pipeline/s8_pressure.py [--tag full_catboost_s42]

Same player, same match, same action type from the same part of the pitch, at high against low stakes. Fixed effects for
player-by-match and for action type by pitch zone; controls for the score and for match time. So stakes are identified by the
score-by-time interaction: a goal moves the stakes suddenly for the same player, net of what each score and each minute does on its own.

Primary tests (fixed in the plan before any result): pass completion, action value, forward progress of passes,
and the home and age interactions on pass completion. Standard errors clustered by match. Holm correction across them.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cpi import config as C  # noqa: E402

ZONE_X, ZONE_Y = 8, 5


def demean(columns, groups, iters=30, tol=1e-7):
    """Sweep out several sets of fixed effects by alternating projections."""
    X = columns.copy()
    codes = [pd.factorize(g)[0] for g in groups]
    sizes = [np.bincount(c).astype(float) for c in codes]
    for _ in range(iters):
        before = X.copy()
        for c, s in zip(codes, sizes):
            for j in range(X.shape[1]):
                X[:, j] -= (np.bincount(c, weights=X[:, j]) / s)[c]
        if np.abs(X - before).max() < tol:
            break
    return X


def cluster_ols(y, X, clusters):
    """OLS on demeaned data with match-clustered standard errors."""
    XtX_inv = np.linalg.inv(X.T @ X)
    beta = XtX_inv @ (X.T @ y)
    e = y - X @ beta
    codes = pd.factorize(clusters)[0]
    score = np.zeros((codes.max() + 1, X.shape[1]))
    for j in range(X.shape[1]):
        score[:, j] = np.bincount(codes, weights=X[:, j] * e)
    return beta, np.sqrt(np.diag(XtX_inv @ (score.T @ score) @ XtX_inv))


def fit(d, outcome, extra=None):
    """Effect of high stakes against low stakes (the bottom 20% is the reference)."""
    names = ['high', 'mid'] + (extra or [])
    controls = [c for c in d.columns if c.startswith(('g_', 'bin_'))]
    data = d[[outcome] + names + controls].values.astype(float)
    data = demean(data, [d['pg'].values, d['cell'].values])
    y, X = data[:, 0], data[:, 1:]
    beta, se = cluster_ols(y, X, d['game_id'].values)
    return {n: {'est': beta[i], 'se': se[i], 'z': beta[i] / se[i]} for i, n in enumerate(names)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', default='full_catboost_s42')
    args = ap.parse_args()
    cols = ['game_id', 'player_id', 'minute', 'type_name', 'result_name', 'is_home', 'goal_diff', 'L_plus', 'L_minus', 'value', 'start_x', 'start_y', 'end_x', 'end_y']
    v = pd.read_parquet(C.WORK / f'values_{args.tag}.parquet', columns=cols)
    v = v[v['player_id'] > 0].reset_index(drop=True)
    stakes = v['L_plus'] + v['L_minus']
    hi, lo = stakes.quantile(1 - C.HIGH_STAKES_SHARE), stakes.quantile(C.HIGH_STAKES_SHARE)
    v['high'], v['low'] = (stakes >= hi).astype(float), (stakes <= lo).astype(float)
    v['mid'] = 1.0 - v['high'] - v['low']

    # Attacking direction of the acting team: the converter's frame has the away team attacking towards x = 0
    side = np.where(v['is_home'], 1.0, -1.0)
    x0 = np.where(v['is_home'], v['start_x'], 105 - v['start_x'])
    y0 = np.where(v['is_home'], v['start_y'], 68 - v['start_y'])
    v['progress'] = (v['end_x'] - v['start_x']) * side
    v['zone'] = np.clip((x0 // (105 / ZONE_X)).astype(int), 0, ZONE_X - 1) * ZONE_Y + np.clip((y0 // (68 / ZONE_Y)).astype(int), 0, ZONE_Y - 1)
    v['cell'] = v['type_name'].astype(str) + '_' + v['zone'].astype(str)
    v['pg'] = v['player_id'].astype(str) + '_' + v['game_id'].astype(str)
    v['completed'] = (v['result_name'] == 'success').astype(float)
    for g in (-3, -2, -1, 1, 2, 3):
        v[f'g_{g}'] = (v['goal_diff'].clip(-3, 3) == g).astype(float)
    bins = (v['minute'] // 5).clip(upper=18).astype(int)
    for b in range(1, 19):
        v[f'bin_{b}'] = (bins == b).astype(float)

    births = pd.read_json(C.RAW / 'players.json')[['wyId', 'birthDate']].drop_duplicates('wyId').set_index('wyId')['birthDate']
    age = (pd.Timestamp('2018-01-01') - pd.to_datetime(v['player_id'].map(births), errors='coerce')).dt.days / 365.25
    v['age'] = (age - 26) / 5                     # centred at 26, per five years
    v['home_num'] = v['is_home'].astype(float)
    v['high_x_home'], v['mid_x_home'] = v['high'] * v['home_num'], v['mid'] * v['home_num']
    v['high_x_age'], v['mid_x_age'] = v['high'] * v['age'], v['mid'] * v['age']

    passes = v[v['type_name'] == 'pass'].reset_index(drop=True)
    print(f'{len(v):,} actions, {len(passes):,} open-play passes; high-stakes cut {hi:.3f}, low-stakes cut {lo:.3f}; pass completion {passes["completed"].mean():.3f}', flush=True)
    out = {
        'a_pass_completion': fit(passes, 'completed'),
        'b_action_value': fit(v, 'value'),
        'c_pass_progress_m': fit(passes, 'progress'),
        'd_home_moderates_completion': fit(passes, 'completed', ['high_x_home', 'mid_x_home']),
        'e_age_moderates_completion': fit(passes[passes['age'].notna()].reset_index(drop=True), 'completed', ['high_x_age', 'mid_x_age']),
    }
    tests = {'a_pass_completion': 'high', 'b_action_value': 'high', 'c_pass_progress_m': 'high', 'd_home_moderates_completion': 'high_x_home', 'e_age_moderates_completion': 'high_x_age'}
    rows = [{'test': k, 'term': t, **out[k][t], 'p': 2 * norm.sf(abs(out[k][t]['z']))} for k, t in tests.items()]
    res = pd.DataFrame(rows).sort_values('p').reset_index(drop=True)
    res['p_holm'] = (res['p'] * (len(res) - res.index)).cummax().clip(upper=1)
    res.to_csv(C.RESULTS / 'q1_pressure_primary.csv', index=False)
    print(res.round(5).to_string(index=False))

    # Exploratory, reported as such. 1) Nerves or choice: completion among passes with the same start and end zones.
    x1 = np.where(passes['is_home'], passes['end_x'], 105 - passes['end_x'])
    y1 = np.where(passes['is_home'], passes['end_y'], 68 - passes['end_y'])
    end_zone = np.clip((x1 // (105 / ZONE_X)).astype(int), 0, ZONE_X - 1) * ZONE_Y + np.clip((y1 // (68 / ZONE_Y)).astype(int), 0, ZONE_Y - 1)
    same_route = passes.assign(cell=passes['zone'].astype(str) + '_' + pd.Series(end_zone).astype(str))
    explore = {'completion_same_route': fit(same_route, 'completed')}
    # 2) Which side of the stakes is high: scoring would gain more (L+ > L-) or conceding would cost more (L- > L+)
    for name, d in (('pass', passes), ('all', v)):
        big_plus = d['high'] * (d['L_plus'] > d['L_minus']).astype(float)
        big_minus = d['high'] * (d['L_minus'] >= d['L_plus']).astype(float)
        e = d.assign(high=big_plus, side_minus=big_minus)
        if name == 'pass':
            explore['completion_by_side'] = fit(e, 'completed', ['side_minus'])
            explore['progress_by_side'] = fit(e, 'progress', ['side_minus'])
        else:
            explore['value_by_side'] = fit(e, 'value', ['side_minus'])
        explore[f'share_high_gain_side_{name}'] = float(big_plus.sum() / d['high'].sum())
    print(json.dumps({k: ({a: {x: round(y, 5) for x, y in b.items()} for a, b in w.items() if a in ('high', 'side_minus')} if isinstance(w, dict) else round(w, 3)) for k, w in explore.items()}, indent=0).replace('\n', ' '))
    out['exploratory'] = explore
    (C.RESULTS / 'q1_pressure_all.json').write_text(json.dumps({'cuts': {'high': hi, 'low': lo}, 'base_completion': passes['completed'].mean(), 'n_actions': len(v), 'n_passes': len(passes), 'fits': out}, indent=1, default=float))
    print('stage 8 done')


if __name__ == '__main__':
    main()
