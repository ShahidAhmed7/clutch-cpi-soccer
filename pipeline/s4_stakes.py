"""Stage 4. The outcome model and the stakes of every moment, leave one league out.

    python pipeline/s4_stakes.py

Compares model specifications on held-out leagues, then writes for the chosen one:
    work/stakes_tables.parquet   W, P(win), P(draw), L+, L- and the scoring-only index, per fold and state
    work/panel.parquet           team-by-frame panel with held-out forecasts
    results/stakes_spec_comparison.csv
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cpi import config as C, stakes  # noqa: E402

SPECS = {'base': ('base',), 'base+red': ('base', 'red'), 'base+red+gxt': ('base', 'red', 'gxt')}


def evaluate(panel, spec):
    rows, tables, scored = [], [], []
    for held in C.LEAGUES:
        train, test = panel[panel['league'] != held], panel[panel['league'] == held]
        model = stakes.fit_rate_model(train, spec)
        table = stakes.stake_table(model, spec)
        t = stakes.attach(test, table, g='g', n='n_left', home='home', adv='adv')
        p = model.predict_proba(stakes.design(test['g'], test['n_left'], test['home'], test['adv'], spec))[:, 1]
        rps = stakes.ranked_probability_score(t['p_win'].values, t['p_draw'].values, test['pts'].values)
        rows.append({'held_out': held, 'log_loss': log_loss(test['scores_again'], p), 'auc': roc_auc_score(test['scores_again'], p),
                     'rps': rps.mean(), 'bias_points': (t['W'].values - test['pts'].values).mean()})
        tables.append(table.assign(held_out=held))
        scored.append(test.assign(W=t['W'].values, p_win=t['p_win'].values, p_draw=t['p_draw'].values, L_plus=t['L_plus'].values,
                                  L_minus=t['L_minus'].values, scoring_only=t['scoring_only'].values))
    return pd.DataFrame(rows), pd.concat(tables, ignore_index=True), pd.concat(scored, ignore_index=True)


def cell_errors(s):
    """Absolute error of the stakes against real final points, by 10-minute bin and score, adjusted for Elo gap and home."""
    elo = pd.read_csv(C.WORK / 'elo_matches.csv')
    long = pd.concat([elo[['game_id', 'home_team_id', 'elo_home_pre', 'elo_away_pre']].set_axis(['game_id', 'team_id', 'e', 'o'], axis=1),
                      elo[['game_id', 'away_team_id', 'elo_away_pre', 'elo_home_pre']].set_axis(['game_id', 'team_id', 'e', 'o'], axis=1)])
    long['elo_gap'] = (long['e'] - long['o']) / 100
    d = s.merge(long[['game_id', 'team_id', 'elo_gap']], on=['game_id', 'team_id'])
    d['minute_bin'] = (d['frame'] * 54 // 600).clip(upper=8)
    d['g3'] = d['g'].clip(-3, 3)
    out = []
    for b, x in d.groupby('minute_bin'):
        G = [-3, -2, -1, 1, 2, 3]
        X = np.column_stack([np.ones(len(x))] + [(x['g3'] == k).astype(float) for k in G] + [x['elo_gap'], x['home']])
        beta = np.linalg.lstsq(X, x['pts'].values.astype(float), rcond=None)[0]
        real = {0: 0.0, **dict(zip(G, beta[1:7]))}
        for g in (-2, -1, 0, 1, 2):
            m = x['g3'] == g
            if m.sum() >= 2000 and (x['g3'] == g + 1).sum() >= 2000 and (x['g3'] == g - 1).sum() >= 2000:
                out.append({'minute_bin': b, 'g': g, 'real_gain': real[g + 1] - real[g], 'real_cost': real[g] - real[g - 1],
                            'model_gain': x.loc[m, 'L_plus'].mean(), 'model_cost': x.loc[m, 'L_minus'].mean()})
    return pd.DataFrame(out)


def main():
    C.RESULTS.mkdir(exist_ok=True)
    matches = pd.read_parquet(C.WORK / 'matches.parquet')
    reds = pd.read_parquet(C.WORK / 'reds.parquet')
    actions = pd.concat([pd.read_parquet(C.WORK / f'actions_{lg}.parquet', columns=['game_id', 'team_id', 'goal', 'own_goal', 'elapsed'])
                         for lg in C.LEAGUES], ignore_index=True)
    panel = stakes.build_panel(actions, matches, reds)
    print('panel', panel.shape, flush=True)
    summary, keep = [], {}
    for name, spec in SPECS.items():
        rows, tables, scored = evaluate(panel, spec)
        e = cell_errors(scored)
        line = {'spec': name, **rows.drop(columns='held_out').mean().to_dict(), 'cells': len(e),
                'mae_cost': (e['model_cost'] - e['real_cost']).abs().mean(), 'mae_gain': (e['model_gain'] - e['real_gain']).abs().mean()}
        summary.append(line)
        keep[name] = (tables, scored)
        print({k: (round(v, 5) if isinstance(v, float) else v) for k, v in line.items()}, flush=True)
    pd.DataFrame(summary).to_csv(C.RESULTS / 'stakes_spec_comparison.csv', index=False)

    # Rule fixed in advance: stay with the simplest model unless a richer one clearly improves held-out forecasts (RPS by more than 0.5%).
    base_rps = summary[0]['rps']
    chosen = 'base'
    for line in summary[1:]:
        if line['rps'] < base_rps * 0.995 and line['rps'] < next(x['rps'] for x in summary if x['spec'] == chosen) * 0.995:
            chosen = line['spec']
    print('chosen spec:', chosen, flush=True)
    tables, scored = keep[chosen]
    tables.to_parquet(C.WORK / 'stakes_tables.parquet', index=False)
    scored.to_parquet(C.WORK / 'panel.parquet', index=False)
    (C.WORK / 'stakes_spec.json').write_text(json.dumps({'chosen': chosen, 'spec': list(SPECS[chosen])}))
    print('stage 4 done')


if __name__ == '__main__':
    main()
