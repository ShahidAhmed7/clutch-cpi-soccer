"""Data contracts. Each test states something that must hold before any result can be trusted.

The tests on real data need Stage 1 output in work/ and are skipped without it.
"""
import numpy as np
import pandas as pd
import pytest

from cpi import config as C, labels, state, values


def toy_actions(rows):
    df = pd.DataFrame(rows, columns=['game_id', 'team_id', 'period_id', 'time_seconds', 'type_name', 'goal', 'own_goal'])
    return df


TOY_MATCH = pd.DataFrame({'game_id': [1], 'home_team_id': [10], 'away_team_id': [20], 'home_score': [1], 'away_score': [1]})


def test_own_goal_moves_the_score_for_the_other_team():
    df = toy_actions([(1, 10, 1, 1.0, 'pass', False, False),
                      (1, 20, 1, 2.0, 'bad_touch', False, True),     # away own goal: home leads 1-0
                      (1, 10, 1, 3.0, 'pass', False, False),
                      (1, 20, 1, 4.0, 'pass', False, False)])
    out = state.add_state(df, TOY_MATCH)
    assert out['goal_diff'].tolist() == [0, 0, 1, -1]


def test_shots_only_tracker_misses_the_own_goal():
    df = toy_actions([(1, 20, 1, 2.0, 'bad_touch', False, True), (1, 10, 1, 3.0, 'pass', False, False)])
    assert state.add_state(df, TOY_MATCH, count_own_goals=False)['goal_diff'].tolist() == [0, 0]


def test_labels_treat_an_own_goal_as_conceding_and_as_scoring_for_the_opponent():
    df = toy_actions([(1, 10, 1, 1.0, 'pass', False, False),
                      (1, 20, 1, 2.0, 'pass', False, False),
                      (1, 20, 1, 3.0, 'bad_touch', False, True)])
    y_s, y_c = labels.build_labels(df)
    assert y_s.tolist() == [1, 0, 0]     # team 10 benefits from the own goal
    assert y_c.tolist() == [0, 1, 1]     # team 20 concedes, on the own goal itself as well


def test_labels_window_includes_the_action_and_stops_after_ten():
    rows = [(1, 10, 1, float(i), 'pass', False, False) for i in range(12)]
    rows[10] = (1, 10, 1, 10.0, 'shot', True, False)
    y_s, y_c = labels.build_labels(toy_actions(rows))
    assert y_s.tolist() == [0] + [1] * 10 + [0]    # actions 1..10 see the goal at 10, action 0 does not
    assert y_c.sum() == 0


def test_labels_do_not_cross_matches():
    df = toy_actions([(1, 10, 2, 90.0, 'pass', False, False), (2, 10, 1, 1.0, 'shot', True, False)])
    assert labels.build_labels(df)[0].tolist() == [0, 1]


def test_value_rule_swaps_channels_when_possession_changes():
    df = toy_actions([(1, 10, 1, 1.0, 'pass', False, False), (1, 20, 1, 2.0, 'pass', False, False)])
    ds, dc = values.channels(df, [0.2, 0.05], [0.01, 0.03])
    assert ds.tolist() == pytest.approx([0.0, 0.05 - 0.01])
    assert dc.tolist() == pytest.approx([0.0, 0.03 - 0.2])


def test_value_rule_resets_after_goal_and_phase_break_and_uses_penalty_baseline():
    df = toy_actions([(1, 10, 1, 1.0, 'shot', True, False),
                      (1, 20, 1, 2.0, 'pass', False, False),       # after a goal
                      (1, 20, 1, 40.0, 'pass', False, False),      # 38 s later
                      (1, 20, 1, 41.0, 'shot_penalty', False, False)])
    ds, dc = values.channels(df, [0.9, 0.1, 0.2, 0.95], [0.1, 0.02, 0.03, 0.01])
    assert ds[1:].tolist() == pytest.approx([0.1, 0.2, 0.95 - C.PENALTY_BASELINE])
    assert dc[1:3].tolist() == pytest.approx([0.02, 0.03])


def test_value_rule_matches_the_reference_on_a_game_without_own_goals():
    pytest.importorskip('socceraction')
    from socceraction.vaep import formula
    rng = np.random.default_rng(0)
    n = 500
    df = pd.DataFrame({'game_id': 1, 'team_id': rng.choice([10, 20], n), 'period_id': 1,
                       'time_seconds': np.cumsum(rng.exponential(4, n)), 'type_name': rng.choice(['pass', 'cross', 'corner_short', 'shot', 'shot_penalty'], n, p=[.8, .1, .03, .06, .01]),
                       'own_goal': False})
    df['result_name'] = np.where(df['type_name'].str.contains('shot') & (rng.random(n) < .2), 'success', 'fail')
    df['goal'] = df['type_name'].str.contains('shot') & (df['result_name'] == 'success')
    p_s, p_c = pd.Series(rng.random(n)), pd.Series(rng.random(n))
    ref = formula.value(df, p_s, p_c)
    ds, dc = values.channels(df, p_s.values, p_c.values)
    assert ds == pytest.approx(ref['offensive_value'].values)
    assert -dc == pytest.approx(ref['defensive_value'].values)


# ---- real data -----------------------------------------------------------------------------------------------------
work = pytest.mark.skipif(not (C.WORK / 'actions_England.parquet').exists(), reason='run pipeline/s1_actions.py first')


@pytest.fixture(scope='module')
def actions():
    cols = ['game_id', 'team_id', 'type_name', 'result_name', 'goal', 'own_goal', 'is_home', 'goal_diff', 'start_x', 'player_id', 'minute', 'period_id', 'elapsed']
    return pd.concat([pd.read_parquet(C.WORK / f'actions_{lg}.parquet', columns=cols) for lg in C.LEAGUES], ignore_index=True)


@pytest.fixture(scope='module')
def matches():
    return pd.read_parquet(C.WORK / 'matches.parquet')


@work
def test_reconstructed_scores_match_official_results(actions, matches):
    check = state.final_score_check(actions, matches)
    assert (check['official'] == check['made']).mean() >= 0.998
    assert check['made'].sum() == check['official'].sum() or abs(check['made'].sum() - check['official'].sum()) <= 4


@work
def test_all_153_own_goals_survive_conversion(actions):
    assert actions['own_goal'].sum() == 153


@work
def test_every_goal_flag_is_a_shot_or_a_direct_set_piece(actions):
    kinds = actions.loc[actions['goal'], 'type_name'].unique()
    assert set(kinds) <= {'shot', 'shot_penalty', 'shot_freekick', 'corner_crossed', 'corner_short', 'freekick_crossed', 'freekick_short'}


@work
def test_action_table_is_in_the_absolute_frame_before_features(actions):
    # converter frame: home attacks towards x = 105, away towards x = 0. Features must mirror this back.
    shots = actions[actions['type_name'] == 'shot']
    assert shots[shots['is_home']]['start_x'].median() > 80
    assert shots[~shots['is_home']]['start_x'].median() < 25


@work
def test_corners_and_free_kicks_keep_their_real_names(actions):
    counts = actions['type_name'].value_counts()
    assert counts[['corner_crossed', 'corner_short']].sum() < 20000     # 18,181 corner events in the raw data
    assert counts[['freekick_crossed', 'freekick_short']].sum() > 40000


@work
def test_clock_never_runs_back_within_a_match(actions):
    diffs = actions.groupby('game_id')['elapsed'].diff().dropna()
    assert diffs.min() >= -1.0


@pytest.mark.skipif(not (C.WORK / 'features_full_England.parquet').exists(), reason='run pipeline/s2_features.py first')
def test_features_put_every_shooter_near_the_goal_they_attack():
    X = pd.read_parquet(C.WORK / 'features_full_England.parquet', columns=['actiontype_a0', 'start_dist_to_goal_a0', 'start_x_a0'])
    a = pd.read_parquet(C.WORK / 'actions_England.parquet', columns=['is_home', 'type_name'])
    shots = (a['type_name'] == 'shot').values
    for side in (True, False):
        m = shots & (a['is_home'].values == side)
        assert X.loc[m, 'start_dist_to_goal_a0'].median() < 25
        assert X.loc[m, 'start_x_a0'].median() > 80


def _pipeline_module(name):
    import importlib
    import sys
    sys.path.insert(0, str(C.ROOT / 'pipeline'))
    return importlib.import_module(name)


def test_fixed_effects_regression_recovers_a_known_effect_despite_confounding():
    s8 = _pipeline_module('s8_pressure')
    rng = np.random.default_rng(1)
    n = 60000
    player, cell = rng.integers(0, 500, n), rng.integers(0, 40, n)
    high = (rng.random(n) < 0.2 + 0.4 * (player % 2)).astype(float)       # stakes correlate with who is playing
    y = -0.03 * high + 0.5 * (player % 2) + 0.01 * cell + rng.normal(0, 0.2, n)
    data = s8.demean(np.column_stack([y, high]), [player, cell])
    beta, se = s8.cluster_ols(data[:, 0], data[:, 1:], rng.integers(0, 1000, n))
    assert abs(beta[0] + 0.03) < 3 * se[0]


def test_conditional_logit_matches_statsmodels_and_recovers_the_truth():
    from statsmodels.discrete.conditional_models import ConditionalLogit
    s9 = _pipeline_module('s9_coaches')
    rng = np.random.default_rng(2)
    groups = np.repeat(np.arange(3000), 8)
    X = rng.normal(size=(len(groups), 2))
    u = X @ np.array([0.8, -0.5]) + rng.gumbel(size=len(groups))
    chosen = np.zeros(len(groups))
    chosen[pd.Series(u).groupby(groups).idxmax().values] = 1
    mine = s9.conditional_logit(X, chosen, groups)
    theirs = ConditionalLogit(chosen, X, groups=groups).fit(disp=0).params
    assert mine == pytest.approx(theirs, abs=1e-4)
    assert mine == pytest.approx([0.8, -0.5], abs=0.1)


def test_doubling_a_decisions_weight_equals_counting_it_twice():
    s9 = _pipeline_module('s9_coaches')
    rng = np.random.default_rng(3)
    groups = np.repeat(np.arange(400), 5)
    X = rng.normal(size=(2000, 2))
    chosen = np.zeros(2000)
    chosen[np.arange(0, 2000, 5) + rng.integers(0, 5, 400)] = 1
    w = np.where(groups < 100, 2.0, 1.0)
    weighted = s9.conditional_logit(X, chosen, groups, w)
    keep_twice = np.where(groups < 100)[0]
    X2 = np.vstack([X, X[keep_twice]])
    c2 = np.concatenate([chosen, chosen[keep_twice]])
    g2 = np.concatenate([groups, groups[keep_twice] + 400])
    assert weighted == pytest.approx(s9.conditional_logit(X2, c2, g2), abs=1e-6)
