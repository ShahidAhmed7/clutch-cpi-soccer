"""Wyscout API v3 event data (the current Wyscout/Hudl API; spec version 2026-07-03).

Accepted layout: one JSON file per match, as returned by
    GET /v3/matches/<wyId>/events?fetch=teams,players,match,substitutions,formations
i.e. {"events": [...], "match": {...}, "teams": {...}, "players": {...}, ...}. Files without "match" still load, but the
first team in the file is then taken as the home team.

Events are read with kloppy (3.19) and converted to SPADL with socceraction's kloppy converter (see kloppy_spadl.py).
Goals, own goals, assists, red cards, goal-mouth zones, lineups and substitution minutes come from the raw file.
Coordinates in v3 are percentages from each acting team's side (own goal at x = 0, attack towards x = 100).
"""
import io
import json
from pathlib import Path

import pandas as pd

from ... import config as C, raw
from ..bundle import Bundle, finish
from ..kloppy_spadl import convert

PERIODS = {'1H': 1, '2H': 2, '1E': 3, '2E': 4, 'E1': 3, 'E2': 4}
PERIOD_START = {1: 0, 2: 45 * 60, 3: 90 * 60, 4: 105 * 60}
ZONES = {'gtl': 'high_left', 'gt': 'high_center', 'gtr': 'high_right', 'gl': 'mid_left', 'gc': 'mid_center', 'gr': 'mid_right',
         'glb': 'low_left', 'gb': 'low_center', 'gbr': 'low_right'}


def _home_first(d):
    """kloppy treats the first entry of "teams" as the home team, so put the home side first."""
    teams = d.get('teams') or {}
    side = {tid: td.get('side') for tid, td in ((d.get('match') or {}).get('teamsData') or {}).items()}
    if len(teams) == 2 and side:
        d['teams'] = dict(sorted(teams.items(), key=lambda kv: 0 if side.get(kv[0]) == 'home' else 1))
    return d


def _one_match(path):
    from kloppy import wyscout
    d = _home_first(json.loads(Path(path).read_text(encoding='utf-8')))
    m = d.get('match') or {}
    gid = int(m.get('wyId') or d['events'][0]['matchId'])
    ds = wyscout.load(event_data=io.BytesIO(json.dumps(d).encode()), data_version='V3')
    actions = convert(ds, gid)

    goals, assists, reds, shots = set(), set(), [], []
    for e in d['events']:
        period = PERIODS.get(e.get('matchPeriod'))
        if period is None:                                  # penalty shoot-out
            continue
        prim, sec = e['type']['primary'], set(e['type'].get('secondary') or [])
        t_sec = e.get('minute', 0) * 60 + e.get('second', 0) - PERIOD_START[period]
        if 'assist' in sec:
            assists.add(str(e['id']))
        inf = e.get('infraction') or {}
        if inf.get('redCard') or 'red_card' in sec:
            reds.append({'game_id': gid, 'team_id': e['team']['id'], 'player_id': (e.get('player') or {}).get('id', 0), 'period_id': period, 'time_seconds': t_sec})
        s = e.get('shot')
        if s and prim != 'own_goal':
            if s.get('isGoal'):
                goals.add(str(e['id']))
            loc = e.get('location') or {}
            gz = s.get('goalZone')
            shots.append({'event_id': str(e['id']), 'game_id': gid, 'team_id': e['team']['id'], 'player_id': (e.get('player') or {}).get('id', 0),
                          'period_id': period, 'time_seconds': t_sec, 'x': loc.get('x', 0) * 1.05, 'y': loc.get('y', 0) * 0.68,
                          'kind': 'penalty' if prim == 'penalty' or 'penalty' in sec else 'free_kick' if prim == 'free_kick' or 'free_kick_shot' in sec else 'open',
                          'head': s.get('bodyPart') == 'head_or_other', 'blocked': gz == 'bc', 'goal': bool(s.get('isGoal')), 'zone': ZONES.get(gz)})
    actions['original_event_id'] = actions['original_event_id'].astype(str)
    actions['goal'] = actions['original_event_id'].isin(goals) & actions['type_name'].str.startswith('shot')
    actions['own_goal'] = actions['result_name'] == 'owngoal'
    actions['assist'] = actions['original_event_id'].isin(assists)

    players = []
    for plist in (d.get('players') or {}).values():
        for p in plist:
            players.append(p.get('player', p))
    teams = [{'team_id': int(k), 'team': (v.get('team', v) or {}).get('name')} for k, v in (d.get('teams') or {}).items()]
    return gid, m, actions, shots, reds, players, teams


def load(path, name=None, workers=6, log=print):
    from concurrent.futures import ProcessPoolExecutor
    files = sorted(Path(path).glob('*.json')) if Path(path).is_dir() else [Path(path)]
    if not files:
        raise FileNotFoundError(f'{path}: no JSON files')
    log(f'Wyscout v3: {len(files)} match files')
    with ProcessPoolExecutor(max_workers=workers) as pool:
        parts = list(pool.map(_one_match, files))
    notes = ['Wyscout v3 is converted through kloppy, a route socceraction validates only for StatsBomb: treat values as experimental.']
    matches_raw, rows = [], []
    for gid, m, *_ in parts:
        td = m.get('teamsData') or {}
        if len(td) != 2:
            notes.append(f'match {gid}: no "match" block, the first team in the file was taken as home and the score was rebuilt from goals')
            continue
        home = next(int(k) for k, v in td.items() if v.get('side') == 'home')
        away = next(int(k) for k, v in td.items() if v.get('side') == 'away')
        rows.append({'game_id': gid, 'home_team_id': home, 'away_team_id': away, 'home_score': int(td[str(home)]['score']),
                     'away_score': int(td[str(away)]['score']), 'date': m.get('dateutc'), 'gameweek': m.get('gameweek'),
                     'competition': m.get('competitionId'), 'season': m.get('seasonId'), 'label': m.get('label')})
        matches_raw.append(m)
    actions = pd.concat([p[2] for p in parts], ignore_index=True)
    shots = pd.DataFrame([s for p in parts for s in p[3]])
    reds = pd.DataFrame([r for p in parts for r in p[4]])
    players_raw = {p['wyId']: p for part in parts for p in part[5]}
    players = pd.DataFrame({'player_id': [int(k) for k in players_raw],
                            'position': [C.ROLE_TO_POSITION.get((p.get('role') or {}).get('name'), 'unknown') for p in players_raw.values()],
                            'name': [p.get('shortName') or f"{p.get('firstName', '')} {p.get('lastName', '')}".strip() for p in players_raw.values()]})
    teams = pd.DataFrame([t for p in parts for t in p[6]]).drop_duplicates('team_id')
    mt = pd.DataFrame(rows)
    names = teams.set_index('team_id')['team']
    mt['home_team'], mt['away_team'] = mt['home_team_id'].map(names), mt['away_team_id'].map(names)
    apps = raw.appearances(matches_raw, reds if len(reds) else pd.DataFrame(columns=['game_id', 'player_id', 'period_id', 'time_seconds']), players)
    b = Bundle(name=name or Path(path).name, provider='wyscout_v3', actions=actions[actions['game_id'].isin(mt['game_id'])], matches=mt,
               players=players, teams=teams, appearances=apps, reds=reds if len(reds) else None, shots=shots, notes=notes)
    return finish(b)
