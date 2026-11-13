"""Stage 2. Game-state features and labels, per league.

    python pipeline/s2_features.py [--variant full|mirrored|no_own_goals]

Variants exist only for the trust experiment:
    full          the method used in the paper
    mirrored      away actions stay in the home team's direction, so distances to goal are wrong for them
    no_own_goals  a shots-only score and label, which is what the reference label code gives on Wyscout data
"""
import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cpi import config as C, labels, state  # noqa: E402
from cpi.features import game_features  # noqa: E402


def one_game(args):
    actions, home_team_id, acting_frame = args
    return game_features(actions, home_team_id, acting_frame)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--variant', default='full', choices=['full', 'mirrored', 'no_own_goals'])
    args = ap.parse_args()
    count_own = args.variant != 'no_own_goals'
    acting_frame = args.variant != 'mirrored'

    matches = pd.read_parquet(C.WORK / 'matches.parquet')
    home = dict(zip(matches['game_id'], matches['home_team_id']))
    reds = pd.read_parquet(C.WORK / 'reds.parquet')
    for league in C.LEAGUES:
        actions = pd.read_parquet(C.WORK / f'actions_{league}.parquet')
        if not count_own:
            actions = state.add_state(actions.drop(columns=['is_home', 'goals_team', 'goals_opp', 'goal_diff', 'elapsed', 't_frame', 'n_left', 'minute', 'man_adv']),
                                      matches, reds, count_own_goals=False)
        jobs = [(g, home[gid], acting_frame) for gid, g in actions.groupby('game_id', sort=False)]
        with ProcessPoolExecutor(max_workers=6) as pool:
            parts = list(pool.map(one_game, jobs, chunksize=20))
        X = pd.concat(parts, ignore_index=True)
        X['y_scores'], X['y_concedes'] = labels.build_labels(actions, count_own_goals=count_own)
        X['game_id'] = actions['game_id'].values
        assert len(X) == len(actions), 'features and actions must stay row-aligned'
        X.to_parquet(C.WORK / f'features_{args.variant}_{league}.parquet', index=False, compression='zstd')
        print(f"{args.variant} {league:8s} {X.shape}  y_scores {X['y_scores'].mean():.4f}  y_concedes {X['y_concedes'].mean():.4f}", flush=True)
    print('stage 2 done')


if __name__ == '__main__':
    main()
