"""The provider-independent tables every league run is built from.

Each provider adapter (StatsBomb, Wyscout v2/v3, kloppy, or your own tables) produces one Bundle. Everything after that
(features, action values, stakes, ratings, keepers, report) is the same code as the paper.

Tables and required columns
---------------------------
actions      SPADL actions in match order: game_id, period_id, time_seconds, team_id, player_id, start_x, start_y, end_x, end_y,
             type_id, result_id, bodypart_id, action_id, type_name, result_name, bodypart_name, original_event_id,
             plus goal (bool), own_goal (bool), assist (bool). Coordinates in metres on a 105 x 68 pitch, all actions in the
             home team's direction (the SPADL convention).
matches      game_id, home_team_id, away_team_id, home_score, away_score; optional date, gameweek, competition, season.
players      player_id, name, position (forward, midfielder, defender or goalkeeper).
teams        team_id, team.
appearances  game_id, team_id, player_id, entry, exit (match minutes), minutes, starter, is_keeper.
reds         game_id, team_id, player_id, period_id, time_seconds (red cards and second yellows; may be empty).
shots        event_id, game_id, team_id, player_id, period_id, time_seconds, x, y (shooter's direction, 105 x 68, y from the top
             touchline), kind (open, free_kick, penalty), head, blocked, goal, zone (on-target goal-mouth zone or missing).
"""
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .. import config as C
from ..keepers import ZONES

POSITIONS = set(C.POSITIONS)
REQUIRED = {
    'actions': ['game_id', 'period_id', 'time_seconds', 'team_id', 'player_id', 'start_x', 'start_y', 'end_x', 'end_y', 'type_id', 'result_id',
                'bodypart_id', 'type_name', 'result_name', 'bodypart_name', 'goal', 'own_goal'],
    'matches': ['game_id', 'home_team_id', 'away_team_id', 'home_score', 'away_score'],
    'players': ['player_id', 'name', 'position'],
    'teams': ['team_id', 'team'],
    'appearances': ['game_id', 'team_id', 'player_id', 'minutes', 'is_keeper'],
}
OPTIONAL = {
    'reds': ['game_id', 'team_id', 'player_id', 'period_id', 'time_seconds'],
    'shots': ['event_id', 'game_id', 'team_id', 'player_id', 'period_id', 'time_seconds', 'x', 'y', 'kind', 'head', 'goal', 'zone'],
}
TABLES = list(REQUIRED) + list(OPTIONAL)


@dataclass
class Bundle:
    name: str
    provider: str
    actions: pd.DataFrame
    matches: pd.DataFrame
    players: pd.DataFrame
    teams: pd.DataFrame
    appearances: pd.DataFrame
    reds: pd.DataFrame = field(default_factory=lambda: pd.DataFrame(columns=OPTIONAL['reds']))
    shots: pd.DataFrame = field(default_factory=lambda: pd.DataFrame(columns=OPTIONAL['shots']))
    notes: list = field(default_factory=list)          # provider caveats shown in the report

    def save(self, folder):
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        for t in TABLES:
            df = getattr(self, t).copy()
            if 'original_event_id' in df:
                df['original_event_id'] = df['original_event_id'].astype(str)
            if 'event_id' in df:
                df['event_id'] = df['event_id'].astype(str)
            df.to_parquet(folder / f'{t}.parquet', index=False)
        (folder / 'bundle.json').write_text(json.dumps({'name': self.name, 'provider': self.provider, 'notes': self.notes}, indent=1))


def read_table(folder, name):
    """A table from parquet or CSV, or None when absent."""
    folder = Path(folder)
    for ext, reader in (('parquet', pd.read_parquet), ('csv', pd.read_csv)):
        p = folder / f'{name}.{ext}'
        if p.exists():
            return reader(p)
    return None


def load_folder(folder, name=None):
    """A bundle saved by Bundle.save, or one you built yourself as CSV/parquet files with the columns above."""
    folder = Path(folder)
    info = json.loads((folder / 'bundle.json').read_text()) if (folder / 'bundle.json').exists() else {}
    tables = {}
    for t in TABLES:
        df = read_table(folder, t)
        if df is None and t in REQUIRED:
            raise FileNotFoundError(f'{folder}: missing {t}.parquet or {t}.csv')
        if df is not None:
            tables[t] = df
    b = Bundle(name=name or info.get('name', folder.name), provider=info.get('provider', 'tables'), notes=info.get('notes', []), **tables)
    return finish(b)


def _as_int_ids(b):
    """Integer ids everywhere (providers such as StatsBomb use integers already; others may use strings)."""
    maps = {}
    for col, tables in (('game_id', ['actions', 'matches', 'appearances', 'reds', 'shots']),
                        ('team_id', ['actions', 'teams', 'appearances', 'reds', 'shots']),
                        ('player_id', ['actions', 'players', 'appearances', 'reds', 'shots'])):
        series = [getattr(b, t)[col] for t in tables if col in getattr(b, t) and len(getattr(b, t))]
        series += [b.matches['home_team_id'], b.matches['away_team_id']] if col == 'team_id' else []
        values = pd.concat([s.astype(object) for s in series], ignore_index=True).dropna()
        if pd.to_numeric(values, errors='coerce').notna().all():
            continue
        codes = {v: i + 1 for i, v in enumerate(pd.unique(values.astype(str)))}
        maps[col] = codes
        for t in tables:
            df = getattr(b, t)
            if col in df:
                df[col] = df[col].map(lambda v: codes.get(str(v), 0) if pd.notna(v) else 0)
        if col == 'team_id':
            for c in ('home_team_id', 'away_team_id'):
                b.matches[c] = b.matches[c].astype(str).map(codes)
    if maps:
        b.notes.append('Non-numeric ids were replaced by integer codes; names are kept in the players and teams tables.')
    return b


def finish(b):
    """Types, ordering and the columns later stages expect; raises a clear error on anything missing."""
    for t, cols in REQUIRED.items():
        missing = [c for c in cols if c not in getattr(b, t)]
        if missing:
            raise ValueError(f'{t} is missing columns: {missing}')
    for t, cols in OPTIONAL.items():
        df = getattr(b, t)
        if df is None or not len(df):
            setattr(b, t, pd.DataFrame(columns=cols))
    b = _as_int_ids(b)
    a = b.actions
    for c in ('game_id', 'team_id', 'period_id', 'type_id', 'result_id', 'bodypart_id'):
        a[c] = pd.to_numeric(a[c]).astype('int64')
    a['player_id'] = pd.to_numeric(a['player_id'], errors='coerce').fillna(0).astype('int64')
    for c in ('goal', 'own_goal', 'assist'):
        a[c] = a[c].fillna(False).astype(bool) if c in a else False
    if 'original_event_id' not in a:
        a['original_event_id'] = np.arange(len(a)).astype(str)
    a['original_event_id'] = a['original_event_id'].astype(str)
    if 'action_id' not in a:
        a['action_id'] = a.groupby('game_id').cumcount()
    a = a.sort_values(['game_id', 'period_id', 'action_id'], kind='stable').reset_index(drop=True)
    a['action_id'] = a.groupby('game_id').cumcount()
    for c in ('start_x', 'start_y', 'end_x', 'end_y', 'time_seconds'):
        a[c] = pd.to_numeric(a[c]).astype('float64')
    b.actions = a

    m = b.matches
    for c in ('game_id', 'home_team_id', 'away_team_id', 'home_score', 'away_score'):
        m[c] = pd.to_numeric(m[c]).astype('int64')
    if 'gameweek' not in m or m['gameweek'].isna().all():
        order = m.sort_values('date' if 'date' in m else 'game_id')['game_id']
        m['gameweek'] = m['game_id'].map({g: i + 1 for i, g in enumerate(order)})
    m['gameweek'] = pd.to_numeric(m['gameweek'], errors='coerce').fillna(0).astype(int)
    m['league'] = b.name
    b.matches = m.drop_duplicates('game_id').reset_index(drop=True)

    p = b.players.drop_duplicates('player_id').copy()
    p['player_id'] = pd.to_numeric(p['player_id']).astype('int64')
    p['position'] = p['position'].where(p['position'].isin(POSITIONS), 'unknown')
    b.players = p
    t = b.teams.drop_duplicates('team_id').copy()
    t['team_id'] = pd.to_numeric(t['team_id']).astype('int64')
    b.teams = t

    ap = b.appearances.copy()
    for c in ('game_id', 'team_id', 'player_id'):
        ap[c] = pd.to_numeric(ap[c]).astype('int64')
    ap['minutes'] = pd.to_numeric(ap['minutes']).clip(0, 90).astype(float)
    if 'entry' not in ap:
        ap['entry'] = np.where(ap.get('starter', True), 0.0, 90.0 - ap['minutes'])
    if 'exit' not in ap:
        ap['exit'] = ap['entry'] + ap['minutes']
    ap['starter'] = ap['starter'].astype(bool) if 'starter' in ap else ap['entry'] <= 0
    ap['is_keeper'] = ap['is_keeper'].astype(bool)
    b.appearances = ap.drop_duplicates(['game_id', 'player_id']).reset_index(drop=True)

    s = b.shots.copy()
    if len(s):
        for c in ('game_id', 'team_id', 'period_id'):
            s[c] = pd.to_numeric(s[c]).astype('int64')
        s['player_id'] = pd.to_numeric(s['player_id'], errors='coerce').fillna(0).astype('int64')
        s['event_id'] = s['event_id'].astype(str)
        for c in ('head', 'goal'):
            s[c] = s[c].fillna(False).astype(bool)
        s['blocked'] = s['blocked'].fillna(False).astype(bool) if 'blocked' in s else False
        s['zone'] = s['zone'].where(s['zone'].isin(ZONES))
        s['league'] = b.name
    b.shots = s
    r = b.reds.copy()
    if len(r):
        for c in ('game_id', 'team_id', 'player_id', 'period_id'):
            r[c] = pd.to_numeric(r[c], errors='coerce').fillna(0).astype('int64')
    b.reds = r
    return b


def checks(b):
    """Data checks for the report: scores rebuilt from the events, minutes, positions, coordinates."""
    from ..state import final_score_check
    out = {}
    fc = final_score_check(b.actions, b.matches)
    out['score_match_share'] = float((fc['official'] == fc['made']).mean())
    out['score_mismatches'] = fc[fc['official'] != fc['made']].to_dict('records')[:10]
    out['matches'] = int(len(b.matches))
    out['actions'] = int(len(b.actions))
    out['goals'] = int(b.actions['goal'].sum())
    out['own_goals'] = int(b.actions['own_goal'].sum())
    out['official_goals'] = int(b.matches['home_score'].sum() + b.matches['away_score'].sum())
    out['players_with_minutes'] = int((b.appearances.groupby('player_id')['minutes'].sum() > 0).sum())
    out['unknown_position_share'] = float((b.players['position'] == 'unknown').mean()) if len(b.players) else 0.0
    out['actions_without_player'] = float((b.actions['player_id'] <= 0).mean())
    inside = b.actions['start_x'].between(-1, 106) & b.actions['start_y'].between(-1, 69)
    out['coordinates_on_pitch_share'] = float(inside.mean())
    out['on_target_shots_with_zone'] = int(b.shots['zone'].notna().sum()) if len(b.shots) else 0
    out['red_cards'] = int(len(b.reds))
    # How much of each team's season the data covers: a full league has every team in about the same number of matches.
    apps = pd.concat([b.matches[['game_id', 'home_team_id']].set_axis(['game_id', 'team_id'], axis=1),
                      b.matches[['game_id', 'away_team_id']].set_axis(['game_id', 'team_id'], axis=1)])
    per_team = apps.groupby('team_id').size().sort_values(ascending=False)
    out['teams'] = int(len(per_team))
    out['matches_per_team_max'] = int(per_team.iloc[0]) if len(per_team) else 0
    out['matches_per_team_median'] = float(per_team.median()) if len(per_team) else 0.0
    out['focus_team_id'] = int(per_team.index[0]) if len(per_team) and per_team.iloc[0] >= 0.9 * len(b.matches) and per_team.median() < 0.5 * per_team.iloc[0] else None
    out['full_league'] = bool(len(per_team) >= 8 and per_team.median() >= 0.8 * per_team.iloc[0] and len(b.matches) >= 0.8 * len(per_team) * per_team.iloc[0] / 2)
    return out
