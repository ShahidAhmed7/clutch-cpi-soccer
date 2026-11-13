"""Stage 6. CPI per player, leaderboards with uncertainty, reliability, and what drives the rating.

    python pipeline/s6_ratings.py [--tag full_catboost_s42]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cpi import config as C, ratings, raw  # noqa: E402


def decomposition(values, qualified, position, minutes):
    """Each player's CPI split into: ability at average stakes, the mix of moments faced, and value concentrated in higher-stakes moments."""
    v = values[values['player_id'].isin(qualified) & (values['type_name'] != 'keeper_save')]
    lp, lm = v['L_plus'].mean(), v['L_minus'].mean()
    g = v.assign(defend_raw=-v['dP_concedes']).groupby('player_id')
    t = g.agg(ds=('dP_scores', 'sum'), dd=('defend_raw', 'sum'), V=('V', 'sum'), lp=('L_plus', 'mean'), lm=('L_minus', 'mean'))
    t['ability'] = (t['ds'] * lp + t['dd'] * lm) / minutes.reindex(t.index) * 90
    t['exposure'] = (t['ds'] * (t['lp'] - lp) + t['dd'] * (t['lm'] - lm)) / minutes.reindex(t.index) * 90
    t['cpi'] = t['V'] / minutes.reindex(t.index) * 90
    t['delivery'] = t['cpi'] - t['ability'] - t['exposure']
    rows = []
    for p in C.POSITIONS:
        x = t[position.reindex(t.index) == p]
        rows.append({'position': p, **{k: x[k].cov(x['cpi']) / x['cpi'].var() for k in ('ability', 'exposure', 'delivery')}})
    return pd.DataFrame(rows), t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', default='full_catboost_s42')
    args = ap.parse_args()
    C.RESULTS.mkdir(exist_ok=True)
    out_dir = C.RESULTS if args.tag == 'full_catboost_s42' else C.WORK / f'results_{args.tag}'
    out_dir.mkdir(exist_ok=True)

    values = pd.read_parquet(C.WORK / f'values_{args.tag}.parquet')
    shots = pd.read_parquet(C.WORK / f'keeper_shots_{args.tag}.parquet')
    matches = pd.read_parquet(C.WORK / 'matches.parquet')
    appearances = pd.read_parquet(C.WORK / 'appearances.parquet')
    players, teams = raw.load_players(), raw.load_teams()

    pm = ratings.player_match_table(values, shots, appearances)
    pl = ratings.per90(pm).join(players.set_index('player_id')[['name', 'position']])
    pl['team'] = pl['team_id'].map(teams)
    pl['league'] = pl['team_id'].map(pd.concat([matches.set_index('home_team_id')['league']]).groupby(level=0).first())
    q = pl[(pl['minutes'] >= C.MIN_MINUTES) & pl['position'].isin(C.POSITIONS)].copy()
    ids = q.index.values
    print(f'{len(q):,} qualified players of {len(pl):,}', q['position'].value_counts().to_dict(), flush=True)

    draws = ratings.bootstrap(pm, ids)
    q['CPI'] = q['V_per90']
    q['ci_low'] = np.nanpercentile(draws['V'], 2.5, axis=0)
    q['ci_high'] = np.nanpercentile(draws['V'], 97.5, axis=0)
    disc = []
    for p in C.POSITIONS:
        m = (q['position'] == p).values
        post, sd, tau2, D = ratings.shrink(q.loc[m, 'V_per90'].values, draws['V'][:, m])
        q.loc[m, 'CPI_shrunk'], q.loc[m, 'CPI_sd'] = post, sd
        q.loc[m, 'P_top10'] = ratings.top_k_probability(post, sd)
        row = {'position': p, 'D_CPI': D}
        for k, name in (('value', 'D_VAEP'), ('attack', 'D_attack'), ('defend', 'D_defend')):
            row[name] = ratings.shrink(q.loc[m, f'{k}_per90'].values, draws[k][:, m])[3]
        if p == 'goalkeeper':
            row['D_points_saved'] = ratings.shrink(q.loc[m, 'points_saved_per90'].values, draws['points_saved'][:, m])[3]
        disc.append(row)
    q['rank'] = q.groupby('position')['CPI'].rank(ascending=False)
    q['rank_shrunk'] = q.groupby('position')['CPI_shrunk'].rank(ascending=False)
    q['rank_vaep'] = q.groupby('position')['value_per90'].rank(ascending=False)
    q['rank_ga'] = q.groupby('position')['ga_per90'].rank(ascending=False)
    q['rank_np'] = q.groupby('position')['V_np_per90'].rank(ascending=False)
    q.sort_values(['position', 'rank']).to_csv(out_dir / 'ratings.csv')

    rel = ratings.split_half(pm, ids, q['position'], ['V', 'value', 'V_scoring_only', 'attack', 'defend', 'points_saved', 'goals'], matches)
    rel.to_csv(out_dir / 'reliability.csv', index=False)
    pd.DataFrame(disc).to_csv(out_dir / 'discrimination.csv', index=False)
    dec, parts = decomposition(values, ids, q['position'], q['minutes'])
    # Does value concentrated in higher-stakes moments repeat? Split-half correlation of the delivery term.
    half = values.merge(matches[['game_id', 'gameweek']], on='game_id')
    half_minutes = appearances.merge(matches[['game_id', 'gameweek']], on='game_id')
    terms = {}
    for name, m in (('odd', 1), ('even', 0)):
        t_mins = half_minutes[half_minutes['gameweek'] % 2 == m].groupby('player_id')['minutes'].sum()
        terms[name] = decomposition(half[half['gameweek'] % 2 == m], ids, q['position'], t_mins)[1].assign(minutes=t_mins)
    both = terms['odd'][['delivery', 'minutes']].join(terms['even'][['delivery', 'minutes']], lsuffix='_o', rsuffix='_e', how='inner')
    both = both[(both['minutes_o'] >= 450) & (both['minutes_e'] >= 450)]
    dec['delivery_rho'] = [spearmanr(x['delivery_o'], x['delivery_e']).statistic for x in (both[q['position'].reindex(both.index) == p] for p in dec['position'])]
    dec.to_csv(out_dir / 'decomposition.csv', index=False)

    pos = q.groupby('position').agg(players=('CPI', 'size'), mean_cpi=('CPI', 'mean'), median_cpi=('CPI', 'median'), share_positive=('CPI', lambda s: (s > 0).mean()),
                                    mean_attack=('attack_per90', 'mean'), mean_defend=('defend_per90', 'mean'), mean_saved=('points_saved_per90', 'mean'))
    pos.to_csv(out_dir / 'position_summary.csv')
    print(pos.round(3).to_string())
    print(rel.round(3).to_string(index=False))
    print(pd.DataFrame(disc).round(3).to_string(index=False))
    print(dec.round(3).to_string(index=False))
    corr = {p: {'vaep': spearmanr(x['CPI'], x['value_per90']).statistic, 'ga': spearmanr(x['CPI'], x['ga_per90']).statistic} for p, x in q.groupby('position')}
    print('within-position rank correlation with CPI:', {k: {a: round(b, 3) for a, b in v.items()} for k, v in corr.items()})
    for p in C.POSITIONS:
        t = q[q['position'] == p].nsmallest(8, 'rank')
        print(f'\nTop {p}s'); print(t[['name', 'team', 'minutes', 'goals', 'penalties', 'CPI', 'ci_low', 'ci_high', 'CPI_shrunk', 'P_top10', 'rank_vaep', 'rank_ga', 'V']].round(3).to_string(index=False))
    (out_dir / 'ratings_summary.json').write_text(json.dumps({'qualified': len(q), 'rank_correlation': corr}, indent=1))
    print('stage 6 done')


if __name__ == '__main__':
    main()
