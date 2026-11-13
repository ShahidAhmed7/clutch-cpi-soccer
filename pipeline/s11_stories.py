"""Stage 11. The season in league points: the biggest moments, goals against points, lead protectors, keepers, and who decided what.

    python pipeline/s11_stories.py [--tag full_catboost_s42]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cpi import config as C, raw  # noqa: E402

pd.set_option('display.width', 220)


def league_tables(matches):
    h = matches[['league', 'home_team_id', 'home_score', 'away_score']].set_axis(['league', 'team_id', 'gf', 'ga'], axis=1)
    a = matches[['league', 'away_team_id', 'away_score', 'home_score']].set_axis(['league', 'team_id', 'gf', 'ga'], axis=1)
    t = pd.concat([h, a])
    t['pts'] = np.select([t['gf'] > t['ga'], t['gf'] == t['ga']], [3, 1], 0)
    t = t.groupby(['league', 'team_id']).agg(pts=('pts', 'sum'), gf=('gf', 'sum'), ga=('ga', 'sum')).reset_index()
    t['gd'] = t['gf'] - t['ga']
    t = t.sort_values(['league', 'pts', 'gd', 'gf'], ascending=[True, False, False, False])
    t['place'] = t.groupby('league').cumcount() + 1
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', default='full_catboost_s42')
    args = ap.parse_args()
    v = pd.read_parquet(C.WORK / f'values_{args.tag}.parquet')
    matches = pd.read_parquet(C.WORK / 'matches.parquet')
    q = pd.read_csv(C.RESULTS / 'ratings.csv').set_index('player_id')
    players, teams = raw.load_players().set_index('player_id'), raw.load_teams()
    label = matches.set_index('game_id')['label']
    show = lambda d: d.assign(name=d['player_id'].map(players['name']), team=d['team_id'].map(teams), match=d['game_id'].map(label), minute=d['minute'].round(0),
                              score=d['goals_team'].astype(int).astype(str) + '-' + d['goals_opp'].astype(int).astype(str))[
        ['name', 'team', 'match', 'minute', 'score', 'type_name', 'result_name', 'L_plus', 'L_minus', 'dP_scores', 'dP_concedes', 'V']]
    named = v[v['player_id'] > 0]
    best, worst = show(named.nlargest(15, 'V')), show(named.nsmallest(15, 'V'))
    best.to_csv(C.RESULTS / 'moments_best.csv', index=False)
    worst.to_csv(C.RESULTS / 'moments_worst.csv', index=False)
    print('MOST VALUABLE'); print(best.head(10).round(2).to_string(index=False))
    print('\nMOST COSTLY'); print(worst.head(10).round(2).to_string(index=False))
    nongoal = show(named[~named['goal'] & (named['goal_diff'] == 1) & (named['minute'] >= 80) & named['type_name'].isin(['clearance', 'interception', 'tackle', 'keeper_save', 'keeper_claim', 'keeper_punch'])].nlargest(8, 'defend').assign(V=lambda d: d['defend']))
    nongoal.to_csv(C.RESULTS / 'moments_lead_saving.csv', index=False)
    print('\nBIGGEST DEFENSIVE ACTIONS, ONE UP AFTER 80'); print(nongoal.round(2).to_string(index=False))

    # Goals against points
    g = named[named['goal'] & (named['type_name'] != 'shot_penalty')]
    pts = g.groupby('player_id').agg(np_goals=('V', 'size'), goal_points=('V', 'sum'), mean_stake=('L_plus', 'mean'), decided=('L_plus', lambda s: (s < 0.1).mean()))
    pts['points_per_goal'] = pts['goal_points'] / pts['np_goals']
    pts = pts.join(players[['name']]).join(q[['team', 'position', 'minutes', 'goals', 'V', 'CPI']])
    pts = pts[pts['np_goals'] >= 8].sort_values('goal_points', ascending=False)
    pts.to_csv(C.RESULTS / 'goals_vs_points.csv')
    print('\nNON-PENALTY GOALS AGAINST POINTS, top by goal points'); print(pts.head(12)[['name', 'team', 'np_goals', 'goal_points', 'points_per_goal', 'decided', 'V']].round(2).to_string(index=False))
    print('\nhighest points per goal'); print(pts[pts['np_goals'] >= 10].nlargest(6, 'points_per_goal')[['name', 'team', 'np_goals', 'goal_points', 'points_per_goal', 'decided']].round(2).to_string(index=False))
    print('\nlowest points per goal'); print(pts[pts['np_goals'] >= 10].nsmallest(6, 'points_per_goal')[['name', 'team', 'np_goals', 'goal_points', 'points_per_goal', 'decided']].round(2).to_string(index=False))

    # Season total
    tot = q.sort_values('V', ascending=False)
    print('\nMOST LEAGUE POINTS ADDED, SEASON'); print(tot.head(12)[['name', 'team', 'position', 'minutes', 'goals', 'penalties', 'V', 'CPI', 'ci_low', 'ci_high']].round(2).to_string(index=False))

    # Lead protectors: value while one goal up after the hour
    late = named[(named['goal_diff'] == 1) & (named['minute'] >= 60)].groupby('player_id').agg(n_lead=('V', 'size'), lead_V=('V', 'sum'), lead_defend=('defend', 'sum'), lead_value=('value', 'sum'),
                                                                                               saving=('dP_concedes', lambda s: (-s).clip(lower=0).sum()))
    lp = late.join(q[['name', 'team', 'position', 'minutes', 'CPI', 'rank', 'rank_vaep', 'V_scoring_only_per90', 'value_per90']], how='inner')
    lp['per100'] = lp['lead_V'] / lp['n_lead'] * 100
    lp = lp[lp['n_lead'] >= 100]
    lp.sort_values('lead_V', ascending=False).to_csv(C.RESULTS / 'lead_protectors.csv')
    for p in ('defender', 'midfielder'):
        print(f'\nLEAD PROTECTORS {p}s: value while one up after 60'); print(lp[lp['position'] == p].nlargest(6, 'lead_V')[['name', 'team', 'n_lead', 'lead_V', 'per100', 'rank', 'rank_vaep']].round(2).to_string(index=False))

    # Who rises when both sides are priced, against one-sided weighting
    q['rank_so'] = q.groupby('position')['V_scoring_only_per90'].rank(ascending=False)
    q['rise'] = q['rank_so'] - q['rank']
    q.sort_values('rise', ascending=False)[['name', 'team', 'position', 'minutes', 'rank', 'rank_so', 'rank_vaep', 'CPI', 'attack_per90', 'defend_per90']].to_csv(C.RESULTS / 'risers.csv')
    for p in ('defender', 'midfielder'):
        print(f'\nRISERS {p}s, scoring-only rank to CPI rank, into the top 50'); print(q[(q['position'] == p) & (q['rank'] <= 50) & (q['minutes'] >= 1500)].nlargest(5, 'rise')[['name', 'team', 'rank_so', 'rank', 'rank_vaep', 'CPI']].round(2).to_string(index=False))

    # Keepers
    k = q[q['position'] == 'goalkeeper'].sort_values('points_saved', ascending=False)
    k[['name', 'team', 'minutes', 'shots_faced', 'goals_prevented', 'points_saved', 'CPI', 'attack_per90', 'rank']].to_csv(C.RESULTS / 'keepers.csv')
    print('\nKEEPERS: league points saved above an average keeper'); print(k[k['minutes'] >= 1800].head(8)[['name', 'team', 'shots_faced', 'goals_prevented', 'points_saved', 'CPI']].round(2).to_string(index=False))
    print('bottom'); print(k[k['minutes'] >= 1800].tail(4)[['name', 'team', 'shots_faced', 'goals_prevented', 'points_saved', 'CPI']].round(2).to_string(index=False))

    # Who decided what: points above the average player of the position, against the margin at the top and bottom of each table
    table = league_tables(matches)
    table.to_csv(C.RESULTS / 'league_tables.csv', index=False)
    q['paa'] = (q['CPI'] - q.groupby('position')['CPI'].transform('mean')) * q['minutes'] / 90
    rows = []
    for lg, t in table.groupby('league'):
        t = t.sort_values('place')
        pts_by = t.set_index('place')['pts']
        last_safe = len(t) - 3
        for place, cut, what in ((1, pts_by[1] - pts_by[2], 'title'), (last_safe, pts_by[last_safe] - pts_by[last_safe + 1], 'stayed up')):
            tid = t[t['place'] == place]['team_id'].iloc[0]
            star = q[q['team_id'] == tid].nlargest(1, 'paa').iloc[0]
            rows.append({'league': lg, 'team': teams.get(tid), 'outcome': what, 'margin': cut, 'player': star['name'], 'points_above_average': star['paa'], 'player_points': star['V']})
    deciders = pd.DataFrame(rows)
    deciders.to_csv(C.RESULTS / 'deciders.csv', index=False)
    print('\nMARGINS AND THE TEAM\'S TOP PLAYER'); print(deciders.round(1).to_string(index=False))
    q.groupby('team_id').apply(lambda d: d.nlargest(1, 'V')[['name', 'V']].iloc[0])
    print('stage 11 done')


if __name__ == '__main__':
    main()
