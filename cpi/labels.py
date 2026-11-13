"""Labels for the action-value models.

Same window as the reference VAEP labels: the action and the next nine. Unlike the reference code, own goals count
wherever they occur. The Wyscout converter stores them as `bad_touch`, and the reference labels only look for own goals
on shots, so there they vanish.
"""
import numpy as np
import pandas as pd

from . import config as C


def build_labels(df, window=C.WINDOW, count_own_goals=True):
    """y_scores: the acting team scores inside the window. y_concedes: it concedes inside the window."""
    n = len(df)
    y_s, y_c = np.zeros(n, dtype=np.int8), np.zeros(n, dtype=np.int8)
    goal = df['goal'].values.astype(bool)
    own = df['own_goal'].values.astype(bool) if count_own_goals else np.zeros(n, bool)
    team = df['team_id'].values
    for ix in pd.Series(np.arange(n)).groupby(df['game_id'].values).indices.values():
        g, o, t = goal[ix], own[ix], team[ix]
        m = len(ix)
        s, c = g.copy(), o.copy()             # the action itself: a goal scores, an own goal concedes
        for k in range(1, min(window, m)):
            g_k, o_k = np.zeros(m, bool), np.zeros(m, bool)
            g_k[:m - k], o_k[:m - k] = g[k:], o[k:]
            t_k = np.full(m, t[-1])
            t_k[:m - k] = t[k:]
            same = t_k == t
            s |= (g_k & same) | (o_k & ~same)
            c |= (g_k & ~same) | (o_k & same)
        y_s[ix], y_c[ix] = s, c
    return y_s, y_c
