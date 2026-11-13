"""Stage 7. Tests that can fail: real results against the stakes, and whether the ledger adds up.

    python pipeline/s7_validation.py [--tag full_catboost_s42]

T1 real final results: what one goal was worth after the 80th minute (no outcome model involved)
T2 stakes against real stakes in every minute-by-score cell: error, slope, rank; scoring-only pressure alongside
T3 the ledger: summed action values against real points, per match and per season
T4 goal-prevention credit under each weighting
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cpi import config as C  # noqa: E402

G = [-3, -2, -1, 1, 2, 3]


def real_points(d, w):
    """Weighted least squares of final points on score (level = 0), Elo gap and home."""
    X = np.column_stack([np.ones(len(d))] + [(d['g3'].values == k).astype(float) for k in G] + [d['elo_gap'].values, d['home'].values])
    Xw = X * w[:, None]
    beta = np.linalg.lstsq(Xw.T @ X, Xw.T @ d['pts'].values.astype(float), rcond=None)[0]
    return {0: 0.0, **dict(zip(G, beta[1:7]))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', default='full_catboost_s42')
    args = ap.parse_args()
    claims = {}
    panel = pd.read_parquet(C.WORK / 'panel.parquet')
    elo = pd.read_csv(C.WORK / 'elo_matches.csv')
    long = pd.concat([elo[['game_id', 'home_team_id', 'elo_home_pre', 'elo_away_pre']].set_axis(['game_id', 'team_id', 'e', 'o'], axis=1),
                      elo[['game_id', 'away_team_id', 'elo_away_pre', 'elo_home_pre']].set_axis(['game_id', 'team_id', 'e', 'o'], axis=1)])
    long['elo_gap'] = (long['e'] - long['o']) / 100
    d = panel.merge(long[['game_id', 'team_id', 'elo_gap']], on=['game_id', 'team_id'])
    d['g3'] = d['g'].clip(-3, 3)
    d['minute'] = d['frame'] * C.SECS_PER_FRAME / 60
    games = d['game_id'].unique()
    gi = pd.Series(np.arange(len(games)), index=games)
    d['gi'] = gi.loc[d['game_id']].values
    rng = np.random.default_rng(C.SEED)
    draw = lambda: np.bincount(rng.integers(0, len(games), len(games)), minlength=len(games)).astype(float)

    # T1
    late = d[d['minute'] >= 80].reset_index(drop=True)
    sits = {'leading_by_1': 1, 'level': 0, 'trailing_by_1': -1}

    def t1(wg):
        w = wg[late['gi'].values]
        p = real_points(late, w)
        return {n: (p[g + 1] - p[g], p[g] - p[g - 1]) for n, g in sits.items()}
    point, boot = t1(np.ones(len(games))), [t1(draw()) for _ in range(C.N_BOOT)]
    rows = []
    for n, g in sits.items():
        m = late['g3'] == g
        diff = np.array([b[n][1] - b[n][0] for b in boot])
        rows.append({'situation': n, 'real_gain': point[n][0], 'real_cost': point[n][1], 'model_gain': late.loc[m, 'L_plus'].mean(), 'model_cost': late.loc[m, 'L_minus'].mean(),
                     'cost_minus_gain': point[n][1] - point[n][0], 'ci_low': np.percentile(diff, 2.5), 'ci_high': np.percentile(diff, 97.5)})
    t1t = pd.DataFrame(rows)
    t1t.to_csv(C.RESULTS / 't1_real_stakes.csv', index=False)
    print(t1t.round(3).to_string(index=False), flush=True)
    claims['t1'] = t1t.set_index('situation').to_dict('index')

    # T2
    d['bin'] = (d['minute'] // 10).clip(upper=8).astype(int)
    counts = d.groupby(['bin', 'g3']).size()
    cells = {(b, g) for b in range(9) for g in (-2, -1, 0, 1, 2) if all(counts.get((b, k), 0) >= 500 for k in (g - 1, g, g + 1))}
    parts = {b: x.reset_index(drop=True) for b, x in d.groupby('bin')}

    def t2(wg):
        rows = []
        for b, x in parts.items():
            w = wg[x['gi'].values]
            p = real_points(x, w)
            for g in (-2, -1, 0, 1, 2):
                m = (x['g3'].values == g)
                if (b, g) in cells and w[m].sum() > 0:
                    rows.append({'bin': b, 'g': g, 'real_gain': p[g + 1] - p[g], 'real_cost': p[g] - p[g - 1], 'model_gain': np.average(x['L_plus'].values[m], weights=w[m]),
                                 'model_cost': np.average(x['L_minus'].values[m], weights=w[m]), 'scoring_only': np.average(x['scoring_only'].values[m], weights=w[m])})
        c = pd.DataFrame(rows)
        c['real_total'], c['model_total'] = c['real_gain'] + c['real_cost'], c['model_gain'] + c['model_cost']
        return c
    obs = t2(np.ones(len(games)))
    obs.to_csv(C.RESULTS / 't2_cells.csv', index=False)
    pairs = {'L_minus_vs_real_cost': ('model_cost', 'real_cost'), 'scoring_only_vs_real_cost': ('scoring_only', 'real_cost'), 'L_plus_vs_real_gain': ('model_gain', 'real_gain'),
             'scoring_only_vs_real_gain': ('scoring_only', 'real_gain'), 'total_vs_real_total': ('model_total', 'real_total'), 'scoring_only_vs_real_total': ('scoring_only', 'real_total')}
    sp = lambda c: {k: spearmanr(c[a], c[b]).statistic for k, (a, b) in pairs.items()}
    boots = pd.DataFrame([sp(t2(draw())) for _ in range(300)])
    rows = []
    for k, (a, b) in pairs.items():
        row = {'pair': k, 'spearman': spearmanr(obs[a], obs[b]).statistic, 'ci_low': boots[k].quantile(.025), 'ci_high': boots[k].quantile(.975)}
        if a.startswith('model'):
            e = obs[a] - obs[b]
            row.update(mae=e.abs().mean(), max_abs=e.abs().max(), slope=np.polyfit(obs[a], obs[b], 1)[0], intercept=np.polyfit(obs[a], obs[b], 1)[1])
        rows.append(row)
    t2t = pd.DataFrame(rows)
    t2t.to_csv(C.RESULTS / 't2_validity.csv', index=False)
    print(len(obs), 'cells'); print(t2t.round(3).to_string(index=False), flush=True)
    claims['t2'] = {'cells': len(obs), **t2t.set_index('pair').to_dict('index')}

    # T3 the ledger
    v = pd.read_parquet(C.WORK / f'values_{args.tag}.parquet', columns=['game_id', 'league', 'team_id', 'V', 'value', 'goal', 'own_goal', 'dP_concedes', 'L_minus', 'scoring_only', 'player_id', 'goal_diff', 'minute'])
    matches = pd.read_parquet(C.WORK / 'matches.parquet')
    tm = v.groupby(['game_id', 'team_id'])[['V', 'value']].sum().reset_index()
    sc = pd.concat([matches[['game_id', 'home_team_id', 'away_team_id', 'home_score', 'away_score']].set_axis(['game_id', 'team_id', 'opp', 'gf', 'ga'], axis=1),
                    matches[['game_id', 'away_team_id', 'home_team_id', 'away_score', 'home_score']].set_axis(['game_id', 'team_id', 'opp', 'gf', 'ga'], axis=1)])
    tm = tm.merge(sc, on=['game_id', 'team_id'])
    tm = tm.merge(tm[['game_id', 'team_id', 'V', 'value']].rename(columns={'team_id': 'opp', 'V': 'V_opp', 'value': 'value_opp'}), on=['game_id', 'opp'])
    tm['pts'] = np.select([tm['gf'] > tm['ga'], tm['gf'] == tm['ga']], [3, 1], 0)
    tm['net_V'], tm['net_value'], tm['gd'] = tm['V'] - tm['V_opp'], tm['value'] - tm['value_opp'], tm['gf'] - tm['ga']
    led = {}
    for x, name in (('net_V', 'net clutch value'), ('net_value', 'net plain VAEP'), ('gd', 'goal difference')):
        slope, intercept = np.polyfit(tm[x], tm['pts'], 1)
        led[x] = {'slope': slope, 'intercept': intercept, 'pearson': pearsonr(tm[x], tm['pts'])[0], 'spearman': spearmanr(tm[x], tm['pts']).statistic}
    season = tm.groupby('team_id').agg(league=('game_id', lambda s: matches.set_index('game_id').loc[s.iloc[0], 'league']), games=('pts', 'size'), pts=('pts', 'sum'), net_V=('net_V', 'sum'), net_value=('net_value', 'sum'), gd=('gd', 'sum'))
    season['pred_pts'] = 1.378 * season['games'] + season['net_V']
    season.to_csv(C.RESULTS / 't3_season_ledger.csv')
    sled = {}
    for x in ('net_V', 'net_value', 'gd'):
        slope, intercept = np.polyfit(season[x], season['pts'], 1)
        sled[x] = {'slope': slope, 'pearson': pearsonr(season[x], season['pts'])[0]}
    print('match ledger', {k: {a: round(b, 3) for a, b in w.items()} for k, w in led.items()})
    print('season ledger', {k: {a: round(b, 3) for a, b in w.items()} for k, w in sled.items()}, flush=True)
    claims['t3'] = {'match': led, 'season': sled, 'team_seasons': len(season)}

    # T4 credit for goal-saving actions (those that lower the risk of conceding) while ahead, each weight scaled to average 1
    w_so = v['scoring_only'] / v['scoring_only'].mean()
    w_lm = v['L_minus'] / v['L_minus'].mean()
    saving = (-v['dP_concedes']).clip(lower=0)
    players = pd.read_csv(C.RESULTS / 'ratings.csv', usecols=['player_id', 'position']).set_index('player_id')['position']
    pos = v['player_id'].map(players)
    ahead = v['goal_diff'] >= 1
    credit = pd.DataFrame({'unweighted': saving[ahead].groupby(pos[ahead]).sum(), 'scoring_only': (saving * w_so)[ahead].groupby(pos[ahead]).sum(), 'two_sided': (saving * w_lm)[ahead].groupby(pos[ahead]).sum()})
    credit['ratio'] = credit['two_sided'] / credit['scoring_only']
    late_lead = (v['goal_diff'] == 1) & (v['minute'] >= 80)
    credit.to_csv(C.RESULTS / 't4_credit.csv')
    print(credit.round(2).to_string())
    claims['t4'] = {**credit.to_dict('index'), 'weight_late_lead': {'two_sided': float(w_lm[late_lead].mean()), 'scoring_only': float(w_so[late_lead].mean())}}
    print('late one-goal lead, weight on a goal-saving action:', claims['t4']['weight_late_lead'])
    (C.RESULTS / 'claims_validation.json').write_text(json.dumps(claims, indent=1, default=float))
    print('stage 7 done')


if __name__ == '__main__':
    main()
