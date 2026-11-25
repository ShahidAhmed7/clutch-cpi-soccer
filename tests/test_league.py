"""Contracts for running CPI on other leagues and providers (run_league.py).

Unit tests use small hand-made inputs. Tests on downloaded data (data_in/) or on the trained models (models/) are
skipped when those are absent.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cpi.league import bundle as B
from cpi.league.providers import statsbomb, wyscout_v3

ROOT = Path(__file__).resolve().parents[1]


def tiny_bundle(team_ids=(10, 20), player_ids=(1, 2)):
    actions = pd.DataFrame({
        'game_id': ['g1'] * 4, 'period_id': [1, 1, 2, 2], 'time_seconds': [1.0, 2.0, 1.0, 2.0], 'team_id': [team_ids[0], team_ids[1], team_ids[0], team_ids[1]],
        'player_id': [player_ids[0], player_ids[1], player_ids[0], player_ids[1]], 'start_x': [50.0, 60, 90, 20], 'start_y': [34.0] * 4,
        'end_x': [60.0, 70, 100, 10], 'end_y': [34.0] * 4, 'type_id': [0, 0, 11, 0], 'result_id': [1, 1, 1, 1], 'bodypart_id': [0] * 4,
        'type_name': ['pass', 'pass', 'shot', 'pass'], 'result_name': ['success'] * 4, 'bodypart_name': ['foot'] * 4,
        'goal': [False, False, True, False], 'own_goal': [False] * 4})
    matches = pd.DataFrame({'game_id': ['g1'], 'home_team_id': [team_ids[0]], 'away_team_id': [team_ids[1]], 'home_score': [1], 'away_score': [0]})
    players = pd.DataFrame({'player_id': list(player_ids), 'name': ['A', 'B'], 'position': ['forward', 'Striker']})
    teams = pd.DataFrame({'team_id': list(team_ids), 'team': ['Home', 'Away']})
    apps = pd.DataFrame({'game_id': ['g1', 'g1'], 'team_id': list(team_ids), 'player_id': list(player_ids), 'minutes': [90, 95], 'is_keeper': [False, False]})
    return B.Bundle('toy', 'test', actions, matches, players, teams, apps)


def test_string_ids_become_integers_consistently():
    b = B.finish(tiny_bundle(team_ids=('home-fc', 'away-fc'), player_ids=('p-a', 'p-b')))
    assert b.actions['team_id'].dtype == 'int64' and b.actions['player_id'].dtype == 'int64'
    assert set(b.actions['team_id']) == {int(b.matches['home_team_id'][0]), int(b.matches['away_team_id'][0])}
    assert set(b.appearances['player_id']) == set(b.players['player_id'])
    assert any('integer codes' in n for n in b.notes)


def test_finish_cleans_positions_minutes_and_order():
    b = B.finish(tiny_bundle())
    assert b.players.set_index('player_id').loc[2, 'position'] == 'unknown'      # not one of the four groups
    assert b.appearances['minutes'].max() == 90                                   # capped to a 90-minute match
    assert b.actions['action_id'].tolist() == [0, 1, 2, 3]


def test_checks_rebuild_the_score():
    c = B.checks(B.finish(tiny_bundle()))
    assert c['score_match_share'] == 1.0 and c['goals'] == 1 and c['official_goals'] == 1


def test_focus_team_and_full_league_detection():
    rows = [{'game_id': i, 'home_team_id': 1 if i % 2 else 100 + i, 'away_team_id': 100 + i if i % 2 else 1, 'home_score': 0, 'away_score': 0} for i in range(30)]
    b = tiny_bundle()
    b.matches = pd.DataFrame(rows)
    b.actions = b.actions.assign(game_id=0, goal=False)
    b = B.finish(b)
    c = B.checks(b)
    assert c['focus_team_id'] == 1 and not c['full_league']


@pytest.mark.parametrize('loc,zone', [([120, 37, 0.2], 'low_left'), ([120, 40, 1.3], 'mid_center'), ([120, 43.5, 2.5], 'high_right'), ([120, 40], 'mid_center'), ([120, 38.0, 0.5], 'low_center')])
def test_statsbomb_goal_mouth_zones(loc, zone):
    assert statsbomb.goal_zone(loc) == zone


@pytest.mark.parametrize('name,group', [('Goalkeeper', 'goalkeeper'), ('Left Wing Back', 'defender'), ('Right Center Back', 'defender'),
                                        ('Center Defensive Midfield', 'midfielder'), ('Left Attacking Midfield', 'midfielder'),
                                        ('Right Wing', 'forward'), ('Striker', 'forward'), ('Secondary Striker', 'forward'), (None, 'unknown')])
def test_statsbomb_position_groups(name, group):
    assert statsbomb.position_group(name) == group


def test_wyscout_v3_home_team_goes_first_for_kloppy():
    d = {'teams': {'7': {'team': {'name': 'Away'}}, '3': {'team': {'name': 'Home'}}},
         'match': {'teamsData': {'7': {'side': 'away'}, '3': {'side': 'home'}}}}
    assert list(wyscout_v3._home_first(d)['teams']) == ['3', '7']


def test_saved_bundle_round_trip(tmp_path):
    b = B.finish(tiny_bundle())
    b.save(tmp_path / 'bundle')
    back = B.load_folder(tmp_path / 'bundle')
    assert len(back.actions) == len(b.actions) and back.actions['goal'].sum() == 1 and back.name == 'toy'


V3 = ROOT / 'data_in' / 'wyscout_v3_sample'


@pytest.mark.skipif(not V3.exists(), reason='needs a Wyscout v3 match file in data_in/wyscout_v3_sample')
def test_kloppy_route_keeps_home_attacking_right_in_both_halves():
    """SPADL keeps the home team attacking towards x = 105 all match; kloppy's HOME_AWAY would flip the second half."""
    b = wyscout_v3.load(V3, workers=1, log=lambda s: None)
    a = b.actions
    shots = a[a['type_name'].str.startswith('shot')]
    home = int(b.matches['home_team_id'].iloc[0])
    for period in (1, 2):
        s = shots[shots['period_id'] == period]
        assert s.loc[s['team_id'] == home, 'start_x'].mean() > 60
        assert s.loc[s['team_id'] != home, 'start_x'].mean() < 45
    assert B.checks(b)['score_match_share'] == 1.0


@pytest.mark.skipif(not (ROOT / 'models' / 'meta.json').exists(), reason='needs trained models (python run_league.py train)')
def test_pretrained_models_use_the_paper_features():
    from cpi.league import models
    m = models.load()
    assert len(m['meta']['feature_columns']) == 64
    assert set(m['stakes_table'].columns) >= {'g', 'n_left', 'home', 'adv', 'L_plus', 'L_minus', 'W'}
    late = m['stakes_table'].query('g == 1 and n_left <= 10 and adv == 0')
    assert (late['L_minus'] > late['L_plus']).all()                              # protecting a late lead: conceding costs more
