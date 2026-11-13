"""Read the public Wyscout files and pull out what the action table does not keep."""
import json
import zipfile

import numpy as np
import pandas as pd

from . import config as C

PERIODS = {'1H': 1, '2H': 2, 'E1': 3, 'E2': 4, 'P': 5}


def load_events(league):
    with zipfile.ZipFile(C.RAW / 'events.zip') as z, z.open(f'events_{league}.json') as f:
        return json.load(f)


def load_matches(league):
    with zipfile.ZipFile(C.RAW / 'matches.zip') as z, z.open(f'matches_{league}.json') as f:
        return json.load(f)


def load_players():
    """One row per player: position group and a readable name."""
    players = json.loads((C.RAW / 'players.json').read_text(encoding='utf-8'))
    return pd.DataFrame({
        'player_id': [int(p['wyId']) for p in players],
        'position': [C.ROLE_TO_POSITION.get(p.get('role', {}).get('name'), 'unknown') for p in players],
        'name': [_decode(p.get('shortName') or f"{p.get('firstName', '')} {p.get('lastName', '')}".strip()) for p in players],
    }).drop_duplicates('player_id')


def load_teams():
    teams = json.loads((C.RAW / 'teams.json').read_text(encoding='utf-8'))
    return pd.Series({int(t['wyId']): _decode(t.get('name')) for t in teams}, name='team')


def _decode(s):
    """Wyscout stores some accented names as literal escapes."""
    if isinstance(s, str) and '\\u' in s:
        try:
            return s.encode('latin-1').decode('unicode_escape')
        except (UnicodeEncodeError, UnicodeDecodeError):
            return s
    return s


def match_table(matches, league):
    rows = []
    for m in matches:
        side = {info['side']: (int(tid), int(info['score'])) for tid, info in m['teamsData'].items()}
        rows.append({'game_id': int(m['wyId']), 'league': league, 'gameweek': int(m.get('gameweek') or 0),
                     'date': pd.to_datetime(m['dateutc']), 'label': _decode(m.get('label')),
                     'home_team_id': side['home'][0], 'away_team_id': side['away'][0],
                     'home_score': side['home'][1], 'away_score': side['away'][1]})
    return pd.DataFrame(rows)


def reference_events(events):
    """Rename raw events to the columns the socceraction Wyscout converter expects."""
    df = pd.DataFrame(events).rename(columns={
        'id': 'event_id', 'eventId': 'type_id', 'subEventId': 'subtype_id', 'eventName': 'type_name',
        'subEventName': 'subtype_name', 'playerId': 'player_id', 'matchId': 'game_id', 'teamId': 'team_id',
        'matchPeriod': 'period_id', 'eventSec': 'milliseconds'})
    df['milliseconds'] = df['milliseconds'] * 1000
    df['period_id'] = df['period_id'].map(PERIODS)
    df['subtype_id'] = pd.to_numeric(df['subtype_id'], errors='coerce').fillna(0).astype(int)
    return df


def tag_ids(event):
    return {t['id'] for t in event.get('tags') or []}


def event_extras(events):
    """Facts the action table loses: goal and assist events, red cards, and every shot with its goal-mouth zone."""
    goals, own_goals, assists, reds, shots = set(), set(), set(), [], []
    for e in events:
        tags = tag_ids(e)
        if C.TAG_GOAL in tags and e['eventId'] != C.EVENT_SAVE:   # on a save event, tag 101 means goal conceded
            goals.add(e['id'])
        if C.TAG_OWN_GOAL in tags:
            own_goals.add(e['id'])
        if C.TAG_ASSIST in tags:
            assists.add(e['id'])
        if tags & {C.TAG_RED, C.TAG_SECOND_YELLOW}:
            reds.append({'game_id': e['matchId'], 'team_id': e['teamId'], 'player_id': e['playerId'],
                         'period_id': PERIODS[e['matchPeriod']], 'time_seconds': e['eventSec']})
        if e['eventId'] == C.EVENT_SHOT or e['subEventId'] in (C.SUB_FREE_KICK_SHOT, C.SUB_PENALTY):
            zone = next((C.GOAL_MOUTH_TAGS[t] for t in tags if t in C.GOAL_MOUTH_TAGS), None)
            pos = e['positions'][0]
            shots.append({'event_id': e['id'], 'game_id': e['matchId'], 'team_id': e['teamId'], 'player_id': e['playerId'],
                          'period_id': PERIODS[e['matchPeriod']], 'time_seconds': e['eventSec'],
                          'x': pos['x'] * 1.05, 'y': pos['y'] * 0.68,          # raw is already in the shooter's direction
                          'kind': 'penalty' if e['subEventId'] == C.SUB_PENALTY else 'free_kick' if e['subEventId'] == C.SUB_FREE_KICK_SHOT else 'open',
                          'head': 403 in tags, 'blocked': 2101 in tags, 'goal': C.TAG_GOAL in tags, 'zone': zone})
    return goals, own_goals, assists, pd.DataFrame(reds), pd.DataFrame(shots)


def appearances(matches, reds, players):
    """One row per player and match: minutes played and whether the player is a goalkeeper."""
    pos = players.set_index('player_id')['position']
    red_minute = {}
    if len(reds):
        for r in reds.itertuples():
            red_minute[(r.game_id, r.player_id)] = (r.period_id - 1) * 45 + r.time_seconds / 60
    rows = []
    for m in matches:
        gid = int(m['wyId'])
        for tid, info in m['teamsData'].items():
            form = info.get('formation') or {}
            subs = [s for s in (form.get('substitutions') or []) if isinstance(s, dict)]
            sub_in = {int(s['playerIn']): int(s['minute']) for s in subs}
            sub_out = {int(s['playerOut']): int(s['minute']) for s in subs}
            starters = {int(p['playerId']) for p in form.get('lineup') or []}
            for pid in starters | set(sub_in):
                entry = 0 if pid in starters else sub_in[pid]
                leave = min(sub_out.get(pid, 90), red_minute.get((gid, pid), 90))
                rows.append({'game_id': gid, 'team_id': int(tid), 'player_id': pid, 'entry': entry, 'exit': leave,
                             'minutes': float(np.clip(leave - entry, 0, 90)), 'starter': pid in starters,
                             'is_keeper': pos.get(pid) == 'goalkeeper'})
    return pd.DataFrame(rows).drop_duplicates(['game_id', 'player_id'])
