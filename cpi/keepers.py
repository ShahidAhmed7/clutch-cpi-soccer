"""Goalkeeper shot-stopping in league points.

An action-value model cannot see a save: a shot's own result is already in its state, so the shooter is debited for the
miss and the keeper's next action inherits nothing. So keepers get an explicit term. For each on-target shot faced,

    points saved = (post-shot xG - goal) * L-

where post-shot xG is the chance that an average keeper concedes this shot, given where it was hit from and where in the
goal mouth it was going, and L- is what conceding would cost the keeper's team at that moment.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import brier_score_loss, roc_auc_score

from . import config as C

ZONES = list(C.GOAL_MOUTH_TAGS.values())
KINDS = ['open', 'free_kick', 'penalty']
FEATURES = ['x', 'y', 'dist', 'angle', 'zone_code', 'head', 'kind_code']


def shot_features(shots):
    d = shots.copy()
    goal_x, goal_y, width = 105.0, 34.0, 7.32
    dx = np.maximum(goal_x - d['x'].values, 1e-3)
    dy = goal_y - d['y'].values
    d['dist'] = np.hypot(dx, dy)
    d['angle'] = np.arctan2(width * dx, dx ** 2 + dy ** 2 - (width / 2) ** 2)
    d['zone_code'] = pd.Categorical(d['zone'], categories=ZONES).codes
    d['kind_code'] = pd.Categorical(d['kind'], categories=KINDS).codes
    d['head'] = d['head'].astype(int)
    return d


def post_shot_xg(shots):
    """Held-out P(goal) for on-target shots: each league is scored by a model of the other four."""
    d = shot_features(shots[shots['zone'].notna()].reset_index(drop=True))
    d['psxg'] = np.nan
    rows = []
    for held in C.LEAGUES:
        te = d['league'] == held
        m = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_depth=4, min_samples_leaf=30,
                                           categorical_features=[FEATURES.index('zone_code'), FEATURES.index('kind_code')], random_state=C.SEED)
        m.fit(d.loc[~te, FEATURES], d.loc[~te, 'goal'].astype(int))
        d.loc[te, 'psxg'] = m.predict_proba(d.loc[te, FEATURES])[:, 1]
        y = d.loc[te, 'goal'].astype(int)
        rows.append({'held_out': held, 'auc': roc_auc_score(y, d.loc[te, 'psxg']), 'brier': brier_score_loss(y, d.loc[te, 'psxg']),
                     'goals': int(y.sum()), 'predicted': float(d.loc[te, 'psxg'].sum()), 'shots': int(te.sum())})
    return d, pd.DataFrame(rows)


def keeper_on_pitch(shots, appearances):
    """The defending team's goalkeeper at the minute of each shot."""
    keepers = appearances[appearances['is_keeper']]
    by_team = {k: g for k, g in keepers.groupby(['game_id', 'team_id'])}
    out = np.zeros(len(shots), dtype=np.int64)
    games = shots['game_id'].values
    defending = shots['defending_team_id'].values
    minutes = ((shots['period_id'].values - 1) * 45 + shots['time_seconds'].values / 60)
    for i in range(len(shots)):
        g = by_team.get((games[i], defending[i]))
        if g is None:
            continue
        on = g[(g['entry'] <= minutes[i]) & (g['exit'] >= minutes[i])]
        out[i] = int((on if len(on) else g.sort_values('minutes', ascending=False)).iloc[0]['player_id'])
    return out
