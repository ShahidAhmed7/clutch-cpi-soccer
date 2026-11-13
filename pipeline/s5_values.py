"""Stage 5. Clutch value of every action, and points saved by goalkeepers.

    V = dP_score * L+ - dP_concede * L-

    python pipeline/s5_values.py [--tag full_catboost_s42] [--variant full]

Writes work/values_<tag>.parquet and work/keeper_shots_<tag>.parquet, plus results/psxg_metrics.csv for the paper's tag.
Options reproduce single data choices for the bug impact check: --no-phase-break, --no-baselines.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cpi import config as C, keepers, stakes, state, values  # noqa: E402

KEEP = ['game_id', 'league', 'team_id', 'player_id', 'period_id', 'time_seconds', 'minute', 'type_name', 'result_name', 'bodypart_name',
        'original_event_id', 'goal', 'own_goal', 'assist', 'is_home', 'goal_diff', 'goals_team', 'goals_opp', 'n_left', 'man_adv',
        'start_x', 'start_y', 'end_x', 'end_y']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', default='full_catboost_s42')
    ap.add_argument('--variant', default='full', help='feature variant the predictions belong to')
    ap.add_argument('--no-phase-break', action='store_true')
    ap.add_argument('--no-baselines', action='store_true')
    ap.add_argument('--out', default=None)
    ap.add_argument('--tables', default='stakes_tables.parquet')
    args = ap.parse_args()
    out_tag = args.out or args.tag
    count_own = args.variant != 'no_own_goals'

    matches = pd.read_parquet(C.WORK / 'matches.parquet')
    reds = pd.read_parquet(C.WORK / 'reds.parquet')
    tables = pd.read_parquet(C.WORK / args.tables)
    parts = []
    for league in C.LEAGUES:
        a = pd.read_parquet(C.WORK / f'actions_{league}.parquet')
        if not count_own:
            a = state.add_state(a.drop(columns=['is_home', 'goals_team', 'goals_opp', 'goal_diff', 'elapsed', 't_frame', 'n_left', 'minute', 'man_adv']),
                                matches, reds, count_own_goals=False)
        p = pd.read_parquet(C.WORK / f'preds_{args.tag}_{league}.parquet')
        assert len(a) == len(p), f'{league}: predictions and actions are not aligned'
        ds, dc = values.channels(a, p['p_scores'].values, p['p_concedes'].values, phase_break=not args.no_phase_break,
                                 set_piece_baselines=not args.no_baselines, count_own_goals=count_own)
        t = stakes.attach(a, tables[tables['held_out'] == league])
        d = a[KEEP].copy()
        d['p_scores'], d['p_concedes'] = p['p_scores'].values, p['p_concedes'].values
        d['dP_scores'], d['dP_concedes'] = ds.astype('float32'), dc.astype('float32')
        for c in ('W', 'p_win', 'p_draw', 'L_plus', 'L_minus', 'scoring_only'):
            d[c] = t[c].values.astype('float32')
        parts.append(d)
    df = pd.concat(parts, ignore_index=True)
    assert df[['dP_scores', 'dP_concedes', 'L_plus', 'L_minus']].notna().all().all(), 'every action must have a value and stakes'
    df['value'] = df['dP_scores'] - df['dP_concedes']                 # plain VAEP
    df['attack'] = df['dP_scores'] * df['L_plus']                     # league points from the scoring channel
    df['defend'] = -df['dP_concedes'] * df['L_minus']                 # league points from the conceding channel
    df['V'] = df['attack'] + df['defend']
    df['V_scoring_only'] = df['value'] * df['scoring_only']           # one weight on both channels
    df.to_parquet(C.WORK / f'values_{out_tag}.parquet', index=False)
    print(f'{out_tag}: {len(df):,} actions, mean V {df["V"].mean():.5f}', flush=True)

    # Keepers: post-shot xG on held-out leagues, weighted by what conceding would cost at that moment
    shots = pd.read_parquet(C.WORK / 'shots.parquet')
    scored, metrics = keepers.post_shot_xg(shots)
    home = dict(zip(matches['game_id'], matches['home_team_id']))
    away = dict(zip(matches['game_id'], matches['away_team_id']))
    scored['defending_team_id'] = np.where(scored['team_id'].values == scored['game_id'].map(home).values,
                                           scored['game_id'].map(away).values, scored['game_id'].map(home).values)
    state_cols = df.loc[df['original_event_id'].notna() & df['type_name'].str.contains('shot|corner|freekick'),
                        ['original_event_id', 'goal_diff', 'n_left', 'is_home', 'man_adv', 'V', 'attack']].drop_duplicates('original_event_id')
    scored = scored.merge(state_cols, left_on='event_id', right_on='original_event_id', how='inner')
    faced = pd.DataFrame({'g': -scored['goal_diff'].values, 'n_left': scored['n_left'].values, 'home': (~scored['is_home'].astype(bool)).astype(int).values,
                          'adv': -scored['man_adv'].values, 'league': scored['league'].values})
    scored['L_minus_keeper'] = np.nan
    for league in C.LEAGUES:
        m = (faced['league'] == league).values
        t = stakes.attach(faced[m], tables[tables['held_out'] == league], g='g', n='n_left', home='home', adv='adv')
        scored.loc[m, 'L_minus_keeper'] = t['L_minus'].values
    appearances = pd.read_parquet(C.WORK / 'appearances.parquet')
    scored['keeper_id'] = keepers.keeper_on_pitch(scored, appearances)
    scored['goals_prevented'] = scored['psxg'] - scored['goal'].astype(float)
    scored['points_saved'] = scored['goals_prevented'] * scored['L_minus_keeper']
    scored.to_parquet(C.WORK / f'keeper_shots_{out_tag}.parquet', index=False)
    if out_tag == 'full_catboost_s42':
        C.RESULTS.mkdir(exist_ok=True)
        metrics.to_csv(C.RESULTS / 'psxg_metrics.csv', index=False)
    print('post-shot xG, held out:', metrics[['auc', 'brier']].mean().round(4).to_dict(), '| goals', metrics['goals'].sum(), 'predicted', round(metrics['predicted'].sum(), 1),
          '| shots with a keeper', int((scored['keeper_id'] > 0).sum()), 'of', len(scored), flush=True)
    print('stage 5 done')


if __name__ == '__main__':
    main()
