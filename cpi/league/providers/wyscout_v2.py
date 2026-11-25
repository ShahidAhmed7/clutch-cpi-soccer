"""Wyscout v2 event data (the "legacy" API version and the public 2017/18 release of Pappalardo et al., 2019).

Accepted layouts
    1. The public release: events_<name>.json and matches_<name>.json (plain or inside events.zip / matches.zip), with
       players.json and teams.json. Choose the competition with --competition, e.g. England, Spain, World_Cup.
    2. One JSON file per match from the v2 API, as returned by matches/<id>/events?fetch=teams,players,match:
       {"events": [...], "match": {...}, "players": {...}, "teams": {...}}.

Actions use socceraction's Wyscout converter, exactly as in the paper's pipeline (pipeline/s1_actions.py).
"""
import json
import zipfile
from pathlib import Path

import pandas as pd

from ... import config as C, raw
from ..bundle import Bundle, finish


def _read(folder, filename, zipname):
    folder = Path(folder)
    if (folder / filename).exists():
        return json.loads((folder / filename).read_text(encoding='utf-8'))
    if (folder / zipname).exists():
        with zipfile.ZipFile(folder / zipname) as z:
            if filename in z.namelist():
                return json.loads(z.read(filename))
    return None


def list_competitions(folder):
    folder = Path(folder)
    names = {p.stem.replace('matches_', '') for p in folder.glob('matches_*.json')}
    if (folder / 'matches.zip').exists():
        with zipfile.ZipFile(folder / 'matches.zip') as z:
            names |= {Path(n).stem.replace('matches_', '') for n in z.namelist() if Path(n).name.startswith('matches_')}
    return sorted(names)


def _players_table(players):
    return pd.DataFrame({
        'player_id': [int(p['wyId']) for p in players],
        'position': [C.ROLE_TO_POSITION.get((p.get('role') or {}).get('name'), 'unknown') for p in players],
        'name': [raw._decode(p.get('shortName') or f"{p.get('firstName', '')} {p.get('lastName', '')}".strip()) for p in players],
    }).drop_duplicates('player_id')


def convert_game(args):
    """socceraction's Wyscout converter for one match (worker process)."""
    import warnings
    warnings.filterwarnings('ignore')
    from socceraction import spadl
    from socceraction.spadl import wyscout
    game_events, home_team_id = args
    return spadl.add_names(wyscout.convert_to_actions(game_events, home_team_id))


def build(events, matches_raw, players_raw, teams_raw, name, notes=None, workers=6, log=print):
    """The paper's stage 1 for any v2 events and matches."""
    from concurrent.futures import ProcessPoolExecutor
    players = _players_table(players_raw)
    mt = raw.match_table(matches_raw, name)
    goals, own_goals, assists, reds, shots = raw.event_extras(events)
    ev = raw.reference_events(events)
    home = dict(zip(mt['game_id'], mt['home_team_id']))
    jobs = [(g, home[gid]) for gid, g in ev.groupby('game_id') if gid in home]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        actions = pd.concat(list(pool.map(convert_game, jobs, chunksize=10)), ignore_index=True)
    inserted = actions['type_name'] == 'interception'          # rows the converter adds share the id of the next event
    actions['goal'] = actions['original_event_id'].isin(goals) & ~inserted
    actions['own_goal'] = actions['result_name'] == 'owngoal'
    actions['assist'] = actions['original_event_id'].isin(assists) & ~inserted
    teams = pd.DataFrame({'team_id': [int(t['wyId']) for t in teams_raw], 'team': [raw._decode(t.get('name')) for t in teams_raw]}) if teams_raw else \
        pd.DataFrame({'team_id': pd.concat([mt['home_team_id'], mt['away_team_id']]).unique(), 'team': None})
    by_id = {int(m['wyId']): m for m in matches_raw}
    mt['home_team'] = mt['home_team_id'].map(teams.set_index('team_id')['team'])
    mt['away_team'] = mt['away_team_id'].map(teams.set_index('team_id')['team'])
    mt['competition'] = name
    if 'date' not in mt:
        mt['date'] = mt['game_id'].map(lambda g: by_id[g].get('dateutc'))
    apps = raw.appearances(matches_raw, reds, players)
    shots = shots.assign(event_id=shots['event_id'].astype(str)) if len(shots) else shots
    b = Bundle(name=name, provider='wyscout_v2', actions=actions, matches=mt, players=players, teams=teams, appearances=apps,
               reds=reds if len(reds) else None, shots=shots, notes=notes or [])
    return finish(b)


def load(path, competition=None, name=None, workers=6, log=print):
    path = Path(path)
    per_match = sorted(p for p in path.glob('*.json') if not p.name.startswith(('events_', 'matches_', 'players', 'teams', 'competitions')))
    if competition is None and per_match:
        return load_per_match(per_match, name or path.name, workers, log)
    if competition is None:
        comps = list_competitions(path)
        if len(comps) != 1:
            raise ValueError(f'Choose one with --competition. Available: {comps}')
        competition = comps[0]
    events = _read(path, f'events_{competition}.json', 'events.zip')
    matches = _read(path, f'matches_{competition}.json', 'matches.zip')
    if events is None or matches is None:
        raise FileNotFoundError(f'{path}: events_{competition}.json or matches_{competition}.json not found. Available: {list_competitions(path)}')
    players = _read(path, 'players.json', 'players.zip') or []
    teams = _read(path, 'teams.json', 'teams.zip') or []
    log(f'Wyscout v2: {competition}, {len(matches)} matches, {len(events):,} events')
    return build(events, matches, players, teams, name or competition, workers=workers, log=log)


def load_per_match(files, name, workers=6, log=print):
    events, matches, players, teams = [], [], {}, {}
    for f in files:
        d = json.loads(Path(f).read_text(encoding='utf-8'))
        if 'events' not in d or 'match' not in d:
            continue
        events += d['events']
        matches.append(d['match'])
        for plist in (d.get('players') or {}).values():
            for p in plist:
                p = p.get('player', p)
                players[p['wyId']] = p
        for t in (d.get('teams') or {}).values():
            t = t.get('team', t)
            teams[t['wyId']] = t
    if not matches:
        raise FileNotFoundError('No per-match Wyscout v2 files with "events" and "match" keys found')
    log(f'Wyscout v2 per-match files: {len(matches)} matches, {len(events):,} events')
    return build(events, matches, list(players.values()), list(teams.values()), name, workers=workers, log=log)
