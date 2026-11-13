"""What each action did to its team's chances of scoring and conceding.

The rule is the reference VAEP value rule (socceraction.vaep.formula), with one change: a play also ends at an own goal.
  - possession changed: the previous state is read from the other team's side
  - more than 10 s since the previous action, or the previous action was a goal: the previous chances count as 0
  - penalties and corners are measured against fixed baselines
"""
import numpy as np

from . import config as C


def _prev(a):
    return np.concatenate([a[:1], a[:-1]])


def channels(df, p_s, p_c, phase_break=True, set_piece_baselines=True, count_own_goals=True):
    """Return (dP_scores, dP_concedes) for actions in match order."""
    game, team = df['game_id'].values, df['team_id'].values
    new_game = game != _prev(game)
    new_game[0] = True
    same_team = (team == _prev(team)) | new_game

    ended = df['goal'].values.astype(bool)
    if count_own_goals:
        ended = ended | df['own_goal'].values.astype(bool)
    after_goal = _prev(ended) & ~new_game
    after_goal[0] = False

    p_s, p_c = np.asarray(p_s, dtype=float), np.asarray(p_c, dtype=float)
    prev_s, prev_c = _prev(p_s), _prev(p_c)
    base_s = np.where(same_team, prev_s, prev_c)
    base_c = np.where(same_team, prev_c, prev_s)
    base_s = np.where(new_game, p_s, base_s)     # first action of a match: no change, as in the reference
    base_c = np.where(new_game, p_c, base_c)

    reset = after_goal.copy()
    if phase_break:
        t = df['time_seconds'].values.astype(float)
        reset |= (np.abs(t - _prev(t)) > C.PHASE_BREAK_SECONDS) & ~new_game
    base_s = np.where(reset, 0.0, base_s)
    base_c = np.where(reset, 0.0, base_c)

    if set_piece_baselines:
        kinds = df['type_name'].values
        base_s = np.where(kinds == 'shot_penalty', C.PENALTY_BASELINE, base_s)
        base_s = np.where((kinds == 'corner_crossed') | (kinds == 'corner_short'), C.CORNER_BASELINE, base_s)
    return p_s - base_s, p_c - base_c
