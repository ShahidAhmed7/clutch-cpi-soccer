"""Stage 1. Raw Wyscout events to a checked action table with match state, per league.

    python pipeline/s1_actions.py

Writes to work/: actions_<league>.parquet, matches.parquet, shots.parquet, reds.parquet, appearances.parquet
"""
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cpi import config as C, raw, state  # noqa: E402

warnings.filterwarnings('ignore')


def convert_game(args):
    from socceraction import spadl
    from socceraction.spadl import wyscout
    game_events, home_team_id = args
    actions = wyscout.convert_to_actions(game_events, home_team_id)
    return spadl.add_names(actions)


def main():
    C.WORK.mkdir(exist_ok=True)
    players = raw.load_players()
    tables = {k: [] for k in ('matches', 'shots', 'reds', 'appearances')}
    for league in C.LEAGUES:
        events, matches = raw.load_events(league), raw.load_matches(league)
        mt = raw.match_table(matches, league)
        goals, own_goals, assists, reds, shots = raw.event_extras(events)
        ev = raw.reference_events(events)
        del events
        home = dict(zip(mt['game_id'], mt['home_team_id']))
        jobs = [(g, home[gid]) for gid, g in ev.groupby('game_id')]
        del ev
        with ProcessPoolExecutor(max_workers=6) as pool:
            parts = list(pool.map(convert_game, jobs, chunksize=20))
        actions = pd.concat(parts, ignore_index=True)
        del jobs, parts

        actions['league'] = league
        # the converter inserts interception rows that share the id of the event they precede, so ids alone count twice
        inserted = actions['type_name'] == 'interception'
        actions['goal'] = actions['original_event_id'].isin(goals) & ~inserted
        actions['own_goal'] = actions['result_name'] == 'owngoal'
        actions['assist'] = actions['original_event_id'].isin(assists) & ~inserted
        assert actions['goal'].sum() == len(goals) and actions['own_goal'].sum() == len(own_goals), 'goal events lost in conversion'
        # game order: the converter numbers actions within a game
        actions = actions.sort_values(['game_id', 'action_id'], kind='stable').reset_index(drop=True)
        actions = state.add_state(actions, mt, reds)
        for c in ('start_x', 'start_y', 'end_x', 'end_y', 'time_seconds', 'elapsed', 'minute'):
            actions[c] = actions[c].astype('float32')
        actions.to_parquet(C.WORK / f'actions_{league}.parquet', index=False)

        check = state.final_score_check(actions, mt)
        print(f"{league:8s} {len(mt):4d} matches  {len(actions):8,d} actions  goals {int(actions['goal'].sum()):5d}  "
              f"own goals {int(actions['own_goal'].sum()):3d}  score match {(check['official'] == check['made']).mean():.4f}", flush=True)
        tables['matches'].append(mt)
        tables['shots'].append(shots.assign(league=league))
        tables['reds'].append(reds.assign(league=league))
        tables['appearances'].append(raw.appearances(matches, reds, players))
    for name, parts in tables.items():
        pd.concat(parts, ignore_index=True).to_parquet(C.WORK / f'{name}.parquet', index=False)
    print('stage 1 done')


if __name__ == '__main__':
    main()
