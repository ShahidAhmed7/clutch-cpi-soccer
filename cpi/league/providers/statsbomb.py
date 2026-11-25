"""Hudl StatsBomb event data (Open Data and the StatsBomb API export, Events v4 / Lineups v2 / Matches v3, data version 1.1.0).

Accepted layouts
    1. The open-data repository layout (https://github.com/hudl/open-data):
         <root>/competitions.json, <root>/matches/<competition_id>/<season_id>.json, <root>/events/<match_id>.json,
         <root>/lineups/<match_id>.json      (<root> may also be the folder that contains data/)
    2. A flat folder of one season: events/<match_id>.json and lineups/<match_id>.json, plus matches.json (a list of match
       objects as in the open data). Without matches.json, home and away teams are taken from the event order.

Actions use socceraction's StatsBomb converter. Goals, own goals, assists, red cards, minutes and goal-mouth zones are read
from the raw files.
"""
import json
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from ..bundle import Bundle, finish

OPEN_DATA = 'https://raw.githubusercontent.com/hudl/open-data/master/data'
ON_TARGET = {'Goal', 'Saved', 'Saved To Post'}
GOAL_Y = (36.0, 44.0)           # StatsBomb goal posts on the 120 x 80 pitch
GOAL_Z = 2.67                   # crossbar height in StatsBomb units (yards)
# Wyscout's goal-mouth grid is not in equal thirds: its centre column holds 48% of on-target shots and its top row 12%.
# The bands below reproduce those shares on StatsBomb end locations (633 on-target shots, Bundesliga 2023/24 and La Liga
# 2019/20): centre = middle 60% of the goal width, low = bottom third, high = top quarter of the height.
CENTRE_FRACTION, LOW_FRACTION, HIGH_FRACTION = 0.6, 1 / 3, 0.75


def position_group(name):
    if not isinstance(name, str):
        return 'unknown'
    if name == 'Goalkeeper':
        return 'goalkeeper'
    if 'Back' in name:
        return 'defender'
    if 'Midfield' in name:
        return 'midfielder'
    if 'Wing' in name or 'Forward' in name or 'Striker' in name:
        return 'forward'
    return 'unknown'


def _clock(mmss):
    if not mmss:
        return None
    m, s = mmss.split(':')
    return int(m) + int(s) / 60


def _seconds(ts):
    h, m, s = ts.split(':')
    return int(h) * 3600 + int(m) * 60 + float(s)


def goal_zone(end_location):
    """One of the nine on-target zones used by the post-shot xG model (Wyscout's goal-mouth grid)."""
    if not end_location or len(end_location) < 2:
        return None
    y = end_location[1]
    mid, half = sum(GOAL_Y) / 2, (GOAL_Y[1] - GOAL_Y[0]) / 2
    side = 'center' if abs(y - mid) < CENTRE_FRACTION * half else 'left' if y < mid else 'right'
    z = end_location[2] if len(end_location) > 2 and end_location[2] is not None else None
    height = 'mid' if z is None else 'low' if z < LOW_FRACTION * GOAL_Z else 'high' if z > HIGH_FRACTION * GOAL_Z else 'mid'
    return f'{height}_{side}'


def find_root(path):
    path = Path(path)
    for cand in (path, path / 'data'):
        if (cand / 'events').is_dir() and (cand / 'lineups').is_dir():
            return cand
    raise FileNotFoundError(f'{path}: no events/ and lineups/ folders found (StatsBomb layout)')


def list_seasons(path):
    root = find_root(path)
    comps = json.loads((root / 'competitions.json').read_text()) if (root / 'competitions.json').exists() else []
    rows = [{'competition_id': c['competition_id'], 'season_id': c['season_id'], 'competition': c['competition_name'], 'season': c['season_name'],
             'country': c['country_name'], 'available': (root / 'matches' / str(c['competition_id']) / f"{c['season_id']}.json").exists()} for c in comps]
    return pd.DataFrame(rows)


def _pick(root, competition, season):
    """Match objects for the requested competition and season (names or ids), or for the flat layout."""
    if (root / 'matches.json').exists():
        return json.loads((root / 'matches.json').read_text())
    if not (root / 'matches').is_dir():
        return None
    seasons = list_seasons(root)
    if competition is None and season is None:
        files = list((root / 'matches').glob('*/*.json'))
        if len(files) == 1:
            return json.loads(files[0].read_text())
        raise ValueError('Several competitions or seasons found; choose one with --competition and --season. Available:\n'
                         + seasons[seasons['available']].to_string(index=False))
    m = seasons
    if competition is not None:
        m = m[(m['competition_id'].astype(str) == str(competition)) | (m['competition'].str.lower() == str(competition).lower())]
    if season is not None:
        m = m[(m['season_id'].astype(str) == str(season)) | (m['season'].str.replace('-', '/') == str(season).replace('-', '/'))]
    m = m[m['available']]
    if len(m) != 1:
        raise ValueError(f'competition={competition!r}, season={season!r} matched {len(m)} seasons. Available:\n'
                         + seasons[seasons['available']].to_string(index=False))
    r = m.iloc[0]
    return json.loads((root / 'matches' / str(r['competition_id']) / f"{r['season_id']}.json").read_text())


def _matches_from_events(root):
    """Flat layout without matches.json: teams from the lineups, home team from the first Starting XI, score from goals."""
    out = []
    for f in sorted((root / 'events').glob('*.json')):
        ev = json.loads(f.read_text())
        xi = [e['team'] for e in ev if e['type']['name'] == 'Starting XI']
        if len(xi) < 2:
            continue
        home, away = xi[0], xi[1]
        goals = {home['id']: 0, away['id']: 0}
        for e in ev:
            if e['type']['name'] == 'Shot' and e['shot']['outcome']['name'] == 'Goal' and e.get('period', 1) <= 4:
                goals[e['team']['id']] += 1
            elif e['type']['name'] == 'Own Goal Against':
                other = away['id'] if e['team']['id'] == home['id'] else home['id']
                goals[other] += 1
        out.append({'match_id': int(f.stem), 'home_team': {'home_team_id': home['id'], 'home_team_name': home['name']},
                    'away_team': {'away_team_id': away['id'], 'away_team_name': away['name']}, 'home_score': goals[home['id']],
                    'away_score': goals[away['id']], 'match_date': None, 'match_week': None, 'metadata': {}})
    return out


def _one_match(args):
    """Actions and extras of one match (runs in a worker process)."""
    import warnings
    warnings.filterwarnings('ignore')
    from socceraction import spadl
    from socceraction.data.statsbomb import StatsBombLoader
    from socceraction.spadl import statsbomb as sb_spadl
    root, m = args
    gid = int(m['match_id'])
    home_id, away_id = int(m['home_team']['home_team_id']), int(m['away_team']['away_team_id'])
    md = m.get('metadata') or {}
    events = StatsBombLoader(getter='local', root=str(root)).events(gid)
    xy = int(md['xy_fidelity_version']) if md.get('xy_fidelity_version') else None
    sf = int(md['shot_fidelity_version']) if md.get('shot_fidelity_version') else None
    actions = spadl.add_names(sb_spadl.convert_to_actions(events, home_id, xy_fidelity_version=xy, shot_fidelity_version=sf))

    raw = json.loads((root / 'events' / f'{gid}.json').read_text())
    goals, assists, reds, shots = set(), set(), [], []
    for e in raw:
        t = e['type']['name']
        if e.get('period', 1) > 4:                         # penalty shoot-outs are not match play
            continue
        if t == 'Shot':
            s = e['shot']
            outcome = s['outcome']['name']
            if outcome == 'Goal':
                goals.add(e['id'])
            loc = e.get('location') or [np.nan, np.nan]
            shots.append({'event_id': e['id'], 'game_id': gid, 'team_id': e['team']['id'], 'player_id': (e.get('player') or {}).get('id', 0),
                          'period_id': e['period'], 'time_seconds': _seconds(e['timestamp']),
                          'x': loc[0] * 105 / 120, 'y': loc[1] * 68 / 80,          # StatsBomb is already in the shooter's direction
                          'kind': {'Penalty': 'penalty', 'Free Kick': 'free_kick'}.get(s.get('type', {}).get('name'), 'open'),
                          'head': s.get('body_part', {}).get('name') == 'Head', 'blocked': outcome == 'Blocked', 'goal': outcome == 'Goal',
                          'zone': goal_zone(s.get('end_location')) if outcome in ON_TARGET else None})
        if t == 'Pass' and e['pass'].get('goal_assist'):
            assists.add(e['id'])
        card = ((e.get('foul_committed') or {}).get('card') or (e.get('bad_behaviour') or {}).get('card') or {}).get('name')
        if card in ('Red Card', 'Second Yellow'):
            reds.append({'game_id': gid, 'team_id': e['team']['id'], 'player_id': (e.get('player') or {}).get('id', 0),
                         'period_id': e['period'], 'time_seconds': _seconds(e['timestamp'])})
    actions['goal'] = actions['original_event_id'].isin(goals) & actions['type_name'].str.startswith('shot')
    actions['own_goal'] = actions['result_name'] == 'owngoal'
    actions['assist'] = actions['original_event_id'].isin(assists)
    kinds = {e['id']: e['type']['name'] for e in raw}
    actions['source_type'] = actions['original_event_id'].map(kinds).fillna('')     # e.g. 'Carry' for StatsBomb carries

    lineups = json.loads((root / 'lineups' / f'{gid}.json').read_text())
    apps, players = [], []
    for team in lineups:
        tid = int(team['team_id'])
        for p in team['lineup']:
            pos = p.get('positions') or []
            if not pos:
                continue                                   # named in the squad but did not play
            entry = _clock(pos[0]['from']) or 0.0
            last = pos[-1]
            exit_ = _clock(last['to']) if last.get('to') else 90.0
            entry, exit_ = min(entry, 90.0), min(exit_ if exit_ is not None else 90.0, 90.0)
            spans = [((_clock(q['to']) if q.get('to') else 90.0) or 90.0) - (_clock(q['from']) or 0.0) for q in pos]
            main = pos[int(np.argmax(spans))]['position']
            apps.append({'game_id': gid, 'team_id': tid, 'player_id': int(p['player_id']), 'entry': entry, 'exit': exit_,
                         'minutes': max(exit_ - entry, 0.0), 'starter': pos[0].get('start_reason') == 'Starting XI',
                         'is_keeper': any(q['position'] == 'Goalkeeper' for q in pos), 'position_name': main})
            players.append({'player_id': int(p['player_id']), 'name': p.get('player_nickname') or p['player_name']})
    return actions, pd.DataFrame(shots), pd.DataFrame(reds), pd.DataFrame(apps), pd.DataFrame(players)


def load(path, competition=None, season=None, name=None, workers=6, log=print):
    root = find_root(path)
    matches = _pick(root, competition, season)
    notes = []
    if matches is None:
        matches = _matches_from_events(root)
        notes.append('No matches file: home and away teams were taken from the order of the Starting XI events.')
    matches = [m for m in matches if (root / 'events' / f"{m['match_id']}.json").exists() and (root / 'lineups' / f"{m['match_id']}.json").exists()]
    if not matches:
        raise FileNotFoundError(f'{root}: no match has both an events and a lineups file')
    log(f'StatsBomb: {len(matches)} matches in {root}')
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=workers) as pool:
        parts = list(pool.map(_one_match, [(root, m) for m in matches], chunksize=2))
    actions = pd.concat([p[0] for p in parts], ignore_index=True)
    shots = pd.concat([p[1] for p in parts], ignore_index=True)
    reds = pd.concat([p[2] for p in parts if len(p[2])], ignore_index=True) if any(len(p[2]) for p in parts) else None
    apps = pd.concat([p[3] for p in parts], ignore_index=True)
    names = pd.concat([p[4] for p in parts], ignore_index=True).drop_duplicates('player_id')
    # season position: the role a player spent most minutes in
    pos = apps.assign(group=apps['position_name'].map(position_group)).groupby(['player_id', 'group'])['minutes'].sum().reset_index()
    pos = pos.sort_values('minutes').groupby('player_id').last()['group']
    players = names.assign(position=names['player_id'].map(pos).fillna('unknown'))
    mt = pd.DataFrame([{'game_id': int(m['match_id']), 'home_team_id': int(m['home_team']['home_team_id']), 'away_team_id': int(m['away_team']['away_team_id']),
                        'home_score': int(m['home_score']), 'away_score': int(m['away_score']), 'date': m.get('match_date'), 'gameweek': m.get('match_week'),
                        'competition': (m.get('competition') or {}).get('competition_name'), 'season': (m.get('season') or {}).get('season_name'),
                        'home_team': m['home_team']['home_team_name'], 'away_team': m['away_team']['away_team_name']} for m in matches])
    teams = pd.concat([mt[['home_team_id', 'home_team']].set_axis(['team_id', 'team'], axis=1),
                       mt[['away_team_id', 'away_team']].set_axis(['team_id', 'team'], axis=1)]).drop_duplicates('team_id')
    if name is None:
        name = f"{mt['competition'].iloc[0]} {mt['season'].iloc[0]}" if mt['competition'].notna().any() else Path(path).name
    b = Bundle(name=name, provider='statsbomb', actions=actions, matches=mt, players=players, teams=teams,
               appearances=apps.drop(columns='position_name'), reds=reds, shots=shots, notes=notes)
    return finish(b)


def fetch_open_data(competition, season, out, workers=8, log=print):
    """Download one season of Hudl StatsBomb open data into the repository layout under <out>/data. Existing files are kept."""
    out = Path(out) / 'data'
    (out / 'events').mkdir(parents=True, exist_ok=True)
    (out / 'lineups').mkdir(parents=True, exist_ok=True)

    def get(rel, dest):
        if dest.exists() and dest.stat().st_size > 0:
            return
        dest.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(f'{OPEN_DATA}/{rel}', timeout=60) as r:
            dest.write_bytes(r.read())

    get('competitions.json', out / 'competitions.json')
    seasons = list_seasons(out.parent)
    m = seasons
    m = m[(m['competition_id'].astype(str) == str(competition)) | (m['competition'].str.lower() == str(competition).lower())]
    m = m[(m['season_id'].astype(str) == str(season)) | (m['season'].str.replace('-', '/') == str(season).replace('-', '/'))]
    if len(m) != 1:
        raise ValueError(f'{competition!r} {season!r} is not in the open data. Available:\n' + seasons[['competition_id', 'season_id', 'competition', 'season']].to_string(index=False))
    cid, sid = int(m.iloc[0]['competition_id']), int(m.iloc[0]['season_id'])
    get(f'matches/{cid}/{sid}.json', out / 'matches' / str(cid) / f'{sid}.json')
    ids = [mm['match_id'] for mm in json.loads((out / 'matches' / str(cid) / f'{sid}.json').read_text())]
    log(f'downloading {len(ids)} matches of {m.iloc[0]["competition"]} {m.iloc[0]["season"]} ...')
    jobs = [(f'{kind}/{i}.json', out / kind / f'{i}.json') for i in ids for kind in ('events', 'lineups')]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(lambda j: get(*j), jobs))
    log(f'saved to {out}')
    return out.parent, cid, sid
