"""Game-state features for the action-value models: the reference VAEP features plus a score that counts own goals.

Every state is the action and its two predecessors, all seen in the attacking direction of the acting team. Reference
function `play_left_to_right` does that mirroring, and every distance and angle to goal depends on it.
"""
import warnings

import numpy as np
import pandas as pd
from socceraction import spadl
from socceraction.vaep import features as fs

from . import config as C

warnings.filterwarnings('ignore')

REFERENCE = [fs.actiontype, fs.result, fs.bodypart, fs.time, fs.startlocation, fs.endlocation,
             fs.startpolar, fs.endpolar, fs.movement, fs.team, fs.time_delta, fs.space_delta]
SCORE = ['goals_team', 'goals_opp', 'goal_diff']
VOCABULARY = {'actiontype': spadl.config.actiontypes, 'result': spadl.config.results, 'bodypart': spadl.config.bodyparts}
CATEGORICAL = C.CATEGORICAL


def game_features(actions, home_team_id, acting_frame=True):
    """Features of one match. `acting_frame=False` leaves away actions in the home team's direction."""
    states = fs.gamestates(actions, nb_prev_actions=3)
    if acting_frame:
        states = fs.play_left_to_right(states, home_team_id)
    out = pd.concat([fn(states) for fn in REFERENCE], axis=1)
    for c in CATEGORICAL:                               # fixed codes, the same in every match
        codes = out[c].cat.set_categories(VOCABULARY[c.rsplit('_', 1)[0]]).cat.codes
        assert codes.min() >= 0, f'unknown category in {c}'
        out[c] = codes.astype(np.int16)
    for c in out.columns.difference(CATEGORICAL):
        out[c] = out[c].astype('float32')
    for c in SCORE:
        out[c] = actions[c].values.astype(np.int16)
    return out.reset_index(drop=True)
