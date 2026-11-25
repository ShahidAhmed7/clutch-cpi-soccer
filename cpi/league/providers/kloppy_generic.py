"""Any event-data provider kloppy can read (Opta F24/F7, Stats Perform MA1/MA3, Sportec/DFL, DataFactory, Metrica, Impect, ...).

Two ways in:
  - Python:  bundle = from_datasets([kloppy EventDataset, ...], name='My league')   (load the files with kloppy yourself)
  - CLI:     --provider kloppy:<loader> --data matches.csv
             a CSV manifest with one row per match whose columns are the kloppy loader's file arguments, e.g. for Opta
                 f7_data,f24_data
                 f7/123.xml,f24/123.xml
             paths are relative to the manifest. Loaders: opta, sportec, statsperform, datafactory, metrica, impect,
             statsbomb, wyscout.

Goals, own goals, red cards, minutes (from lineups and substitutions), positions and goal-mouth zones come from kloppy's
common data model. Conversion to SPADL uses socceraction's kloppy converter (see kloppy_spadl.py); results for providers
other than StatsBomb are experimental.
"""
import importlib
from pathlib import Path

import numpy as np
import pandas as pd

from ..bundle import Bundle, finish
from ..kloppy_spadl import convert

LOADERS = {'opta': 'load', 'sportec': 'load_event', 'statsperform': 'load_event', 'datafactory': 'load', 'metrica': 'load_event',
           'impect': 'load', 'statsbomb': 'load', 'wyscout': 'load'}
GOAL_HALF_WIDTH, GOAL_HEIGHT = 7.32 / 2, 2.44


def _group(position):
    from kloppy.domain import PositionType
    if position is None:
        return 'unknown'
    for parent, name in ((PositionType.Goalkeeper, 'goalkeeper'), (PositionType.Defender, 'defender'),
                         (PositionType.Midfielder, 'midfielder'), (PositionType.Attacker, 'forward')):
        if position == parent or position.is_subtype_of(parent):
            return 'forward' if name == 'midfielder' and position.name in ('LeftWing', 'RightWing') else name
    return 'unknown'


def _minute(event):
    p = event.period.id
    return (p - 1) * 45 + event.timestamp.total_seconds() / 60 if p <= 2 else 90.0


def _zone(result_coordinates):
    if result_coordinates is None or result_coordinates.y is None:
        return None
    dy = result_coordinates.y - 34.0                    # metres from the goal centre, y measured from the top touchline
    if abs(dy) > GOAL_HALF_WIDTH:
        return None
    from .statsbomb import CENTRE_FRACTION, HIGH_FRACTION, LOW_FRACTION       # Wyscout's grid, see statsbomb.py
    side = 'center' if abs(dy) < CENTRE_FRACTION * GOAL_HALF_WIDTH else 'left' if dy < 0 else 'right'
    z = getattr(result_coordinates, 'z', None)
    height = 'mid' if z is None else 'low' if z < LOW_FRACTION * GOAL_HEIGHT else 'high' if z > HIGH_FRACTION * GOAL_HEIGHT else 'mid'
    return f'{height}_{side}'


def _shooter_frame(ds):
    """The same events with every action in its own team's attacking direction, metres, y from the top touchline."""
    from kloppy.domain import CustomCoordinateSystem, Dimension, MetricPitchDimensions, Orientation, Origin, VerticalOrientation
    cs = CustomCoordinateSystem(origin=Origin.TOP_LEFT, vertical_orientation=VerticalOrientation.TOP_TO_BOTTOM,
                                pitch_dimensions=MetricPitchDimensions(x_dim=Dimension(0, 105), y_dim=Dimension(0, 68), pitch_length=105, pitch_width=68, standardized=True))
    return ds.transform(to_orientation=Orientation.ACTION_EXECUTING_TEAM, to_coordinate_system=cs)


def one_dataset(ds, game_id=None):
    from kloppy.domain import BodyPart, BodyPartQualifier, CardType, EventType, Ground, SetPieceQualifier, SetPieceType, ShotResult
    md = ds.metadata
    gid = game_id if game_id is not None else md.game_id
    home = next(t for t in md.teams if t.ground == Ground.HOME)
    away = next(t for t in md.teams if t.ground == Ground.AWAY)
    actions = convert(ds, gid)

    goals, reds, shots, subs_on, subs_off = set(), [], [], {}, {}
    for e, f in zip(ds.events, _shooter_frame(ds).events):
        if e.period.id > 4:
            continue
        t = e.event_type
        if t == EventType.SHOT and e.result != ShotResult.OWN_GOAL:
            if e.result == ShotResult.GOAL:
                goals.add(str(e.event_id))
            sp = e.get_qualifier_value(SetPieceQualifier)
            bp = e.get_qualifier_value(BodyPartQualifier)
            on_target = e.result in (ShotResult.GOAL, ShotResult.SAVED)
            shots.append({'event_id': str(e.event_id), 'game_id': gid, 'team_id': e.team.team_id, 'player_id': e.player.player_id if e.player else 0,
                          'period_id': e.period.id, 'time_seconds': e.timestamp.total_seconds(),
                          'x': f.coordinates.x if f.coordinates else np.nan, 'y': f.coordinates.y if f.coordinates else np.nan,
                          'kind': 'penalty' if sp == SetPieceType.PENALTY else 'free_kick' if sp == SetPieceType.FREE_KICK else 'open',
                          'head': bp == BodyPart.HEAD, 'blocked': e.result == ShotResult.BLOCKED, 'goal': e.result == ShotResult.GOAL,
                          'zone': _zone(f.result_coordinates) if on_target else None})
        elif t == EventType.CARD and e.card_type in (CardType.RED, CardType.SECOND_YELLOW) and e.player:
            reds.append({'game_id': gid, 'team_id': e.team.team_id, 'player_id': e.player.player_id, 'period_id': e.period.id,
                         'time_seconds': e.timestamp.total_seconds()})
            subs_off[e.player.player_id] = min(subs_off.get(e.player.player_id, 90.0), _minute(e))
        elif t == EventType.SUBSTITUTION and e.player:
            subs_off[e.player.player_id] = min(subs_off.get(e.player.player_id, 90.0), _minute(e))
            if e.replacement_player:
                subs_on[e.replacement_player.player_id] = _minute(e)
        elif t == EventType.PLAYER_OFF and e.player:
            subs_off[e.player.player_id] = min(subs_off.get(e.player.player_id, 90.0), _minute(e))
        elif t == EventType.PLAYER_ON and e.player:
            subs_on.setdefault(e.player.player_id, _minute(e))
    actions['original_event_id'] = actions['original_event_id'].astype(str)
    actions['goal'] = actions['original_event_id'].isin(goals) & actions['type_name'].str.startswith('shot')
    actions['own_goal'] = actions['result_name'] == 'owngoal'
    actions['assist'] = False

    apps, players = [], []
    for team in (home, away):
        for p in team.players:
            if not (p.starting or p.player_id in subs_on):
                continue
            entry = 0.0 if p.starting else min(subs_on[p.player_id], 90.0)
            exit_ = min(subs_off.get(p.player_id, 90.0), 90.0)
            pos = p.starting_position
            if pos is None and getattr(p, 'positions', None) is not None and len(p.positions.items):
                pos = list(p.positions.items.values())[0]
            apps.append({'game_id': gid, 'team_id': team.team_id, 'player_id': p.player_id, 'entry': entry, 'exit': exit_,
                         'minutes': max(exit_ - entry, 0.0), 'starter': bool(p.starting), 'is_keeper': _group(pos) == 'goalkeeper'})
            players.append({'player_id': p.player_id, 'name': p.name or p.full_name, 'position': _group(pos)})
    score = md.score
    if score is not None:
        hs, as_ = score.home, score.away
    else:
        own = actions[actions['own_goal']]
        hs = int(actions[actions['goal'] & (actions['team_id'] == home.team_id)].shape[0] + (own['team_id'] == away.team_id).sum())
        as_ = int(actions[actions['goal'] & (actions['team_id'] == away.team_id)].shape[0] + (own['team_id'] == home.team_id).sum())
    match = {'game_id': gid, 'home_team_id': home.team_id, 'away_team_id': away.team_id, 'home_score': int(hs), 'away_score': int(as_),
             'date': str(md.date) if getattr(md, 'date', None) else None, 'gameweek': getattr(md, 'game_week', None),
             'home_team': home.name, 'away_team': away.name}
    return actions, match, shots, reds, apps, players


def from_datasets(datasets, name, provider='kloppy', game_ids=None):
    parts = [one_dataset(ds, None if game_ids is None else game_ids[i]) for i, ds in enumerate(datasets)]
    return _bundle(parts, name, provider, datasets[0].metadata.provider if datasets else None)


def _bundle(parts, name, provider, source):
    actions = pd.concat([p[0] for p in parts], ignore_index=True)
    matches = pd.DataFrame([p[1] for p in parts])
    shots = pd.DataFrame([s for p in parts for s in p[2]])
    reds = pd.DataFrame([r for p in parts for r in p[3]])
    apps = pd.DataFrame([a for p in parts for a in p[4]])
    plist = pd.DataFrame([x for p in parts for x in p[5]])
    # season position: the group a player started in most often
    pos = plist.groupby('player_id')['position'].agg(lambda s: s[s != 'unknown'].mode().iloc[0] if (s != 'unknown').any() else 'unknown')
    players = plist.drop_duplicates('player_id').drop(columns='position').assign(position=lambda d: d['player_id'].map(pos))
    teams = pd.concat([matches[['home_team_id', 'home_team']].set_axis(['team_id', 'team'], axis=1),
                       matches[['away_team_id', 'away_team']].set_axis(['team_id', 'team'], axis=1)]).drop_duplicates('team_id')
    notes = [] if str(source).lower().endswith('statsbomb') else \
        [f'{source} data is converted through kloppy, a route socceraction validates only for StatsBomb: treat values as experimental.']
    b = Bundle(name=name, provider=provider, actions=actions, matches=matches, players=players, teams=teams, appearances=apps,
               reds=reds if len(reds) else None, shots=shots, notes=notes)
    return finish(b)


def _load_row(args):
    import warnings
    warnings.filterwarnings('ignore')
    loader, kwargs, gid = args
    mod = importlib.import_module(f'kloppy.{loader}')
    ds = getattr(mod, LOADERS[loader])(**kwargs)
    return one_dataset(ds, gid)


def load_manifest(manifest, loader, name=None, workers=6, log=print):
    from concurrent.futures import ProcessPoolExecutor
    if loader not in LOADERS:
        raise ValueError(f'unknown kloppy loader {loader!r}; choose from {sorted(LOADERS)}')
    manifest = Path(manifest)
    rows = pd.read_csv(manifest)
    base = manifest.parent
    jobs = []
    for i, r in rows.iterrows():
        kwargs = {k: str((base / v).resolve()) if isinstance(v, str) and (base / v).exists() else v
                  for k, v in r.items() if k != 'game_id' and pd.notna(v)}
        jobs.append((loader, kwargs, int(r['game_id']) if 'game_id' in r and pd.notna(r['game_id']) else i + 1))
    log(f'kloppy:{loader}: {len(jobs)} matches from {manifest}')
    with ProcessPoolExecutor(max_workers=workers) as pool:
        parts = list(pool.map(_load_row, jobs))
    return _bundle(parts, name or manifest.stem, f'kloppy:{loader}', loader)
