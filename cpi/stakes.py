"""Stakes of every moment in league points.

One outcome model, fitted on a panel with one row per team and time frame, so possession does not weight it. A single
function f(g, n, home, adv) gives P(team scores at least once more); the opponent's chance is the same function seen from
its side, so both teams always share one coherent forecast. Team strength stays out on purpose: the stakes multiply action
values that already reflect player quality (Kharrat et al., 2020).

    theta = 1 - (1 - f)^(1/n)                      per-frame scoring rate
    future goals ~ independent Binomial(n, theta)  for each team
    W = 3 P(win) + P(draw)                         expected league points
    L+ = W(g + 1) - W(g),  L- = W(g) - W(g - 1)    same scoring rates: a goal changes the score, not the teams
"""
import numpy as np
import pandas as pd
from scipy.stats import binom
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from . import config as C
from .state import elapsed_seconds

G_RANGE = np.arange(-8, 9)
ADV_RANGE = np.arange(-2, 3)
N_RANGE = np.arange(1, C.T_FRAMES + 1)


def build_panel(actions, matches, reds):
    """One row per team and frame: goal difference, frames left, home flag, red-card balance, and what happened next."""
    a = actions[actions['goal'] | actions['own_goal']]
    home = dict(zip(matches['game_id'], matches['home_team_id']))
    goal_times = {g: {} for g in matches['game_id']}
    for r in a.itertuples():
        scorer_is_home = (r.team_id == home[r.game_id]) == bool(r.goal)
        goal_times[r.game_id].setdefault(scorer_is_home, []).append(r.elapsed)
    red_times = {}
    if len(reds):
        el = elapsed_seconds(reds['period_id'], reds['time_seconds'])
        for r, e in zip(reds.itertuples(), el):
            red_times.setdefault((r.game_id, r.team_id == home.get(r.game_id)), []).append(e)

    times = np.arange(C.T_FRAMES) * C.SECS_PER_FRAME
    n_left = C.T_FRAMES - np.arange(C.T_FRAMES)
    rows = []
    for m in matches.itertuples():
        gh = np.sort(goal_times[m.game_id].get(True, []))
        ga = np.sort(goal_times[m.game_id].get(False, []))
        hb, ab = np.searchsorted(gh, times), np.searchsorted(ga, times)      # goals strictly before the frame
        rh = np.searchsorted(np.sort(red_times.get((m.game_id, True), [])), times)
        ra = np.searchsorted(np.sort(red_times.get((m.game_id, False), [])), times)
        pts_h = 3 if m.home_score > m.away_score else 1 if m.home_score == m.away_score else 0
        pts_a = 3 if m.away_score > m.home_score else 1 if m.home_score == m.away_score else 0
        for is_home, g, more, adv, pts, team in ((1, hb - ab, len(gh) - hb > 0, ra - rh, pts_h, m.home_team_id),
                                                  (0, ab - hb, len(ga) - ab > 0, rh - ra, pts_a, m.away_team_id)):
            rows.append(pd.DataFrame({'game_id': m.game_id, 'league': m.league, 'team_id': team, 'frame': np.arange(C.T_FRAMES),
                                      'n_left': n_left, 'g': g, 'home': is_home, 'adv': adv, 'scores_again': more.astype(int), 'pts': pts}))
    return pd.concat(rows, ignore_index=True)


def design(g, n, home, adv, spec):
    g, n, home, adv = [np.asarray(v, dtype=float) for v in (g, n, home, adv)]
    cols = [g, n, n ** 2, home]
    if 'red' in spec:
        cols.append(adv)
    if 'gxt' in spec:
        cols += [g * n, g * n ** 2]
    return np.column_stack(cols)


def fit_rate_model(panel, spec=('base',)):
    model = Pipeline([('scale', StandardScaler()), ('lr', LogisticRegression(C=1.0, max_iter=1000, random_state=C.SEED))])
    return model.fit(design(panel['g'], panel['n_left'], panel['home'], panel['adv'], spec), panel['scores_again'])


def stake_table(model, spec=('base',), max_goals=C.MAX_GOALS, shared_rates=True):
    """W, P(win), P(draw), L+, L- and the scoring-only index for every state.

    `shared_rates=False` evaluates W(g +- 1) with the rates of the state g +- 1, used only as a sensitivity check.
    """
    grid = pd.MultiIndex.from_product([G_RANGE, N_RANGE, [0, 1], ADV_RANGE], names=['g', 'n_left', 'home', 'adv']).to_frame(index=False)

    def rate(g, n, home, adv):
        p = np.clip(model.predict_proba(design(g, n, home, adv, spec))[:, 1], 1e-9, 1 - 1e-9)
        return 1 - (1 - p) ** (1 / np.asarray(n, dtype=float))

    def outcome(th_team, th_opp, n, g):
        win, draw = np.zeros(len(g)), np.zeros(len(g))
        ks = np.arange(max_goals + 1)
        for t in np.unique(n):
            m = n == t
            a = binom.pmf(ks[None, :], t, th_team[m][:, None])
            b = binom.pmf(ks[None, :], t, th_opp[m][:, None])
            d = np.subtract.outer(ks, ks)                               # team goals minus opponent goals
            joint = a[:, :, None] * b[:, None, :]
            final = g[m][:, None, None] + d[None, :, :]
            win[m] = (joint * (final > 0)).sum(axis=(1, 2))
            draw[m] = (joint * (final == 0)).sum(axis=(1, 2))
        return win, draw

    g, n, home, adv = (grid[c].values for c in ('g', 'n_left', 'home', 'adv'))
    th_t, th_o = rate(g, n, home, adv), rate(-g, n, 1 - home, -adv)
    w0, d0 = outcome(th_t, th_o, n, g)
    if shared_rates:
        w_up, d_up = outcome(th_t, th_o, n, g + 1)
        w_dn, d_dn = outcome(th_t, th_o, n, g - 1)
    else:
        w_up, d_up = outcome(rate(g + 1, n, home, adv), rate(-g - 1, n, 1 - home, -adv), n, g + 1)
        w_dn, d_dn = outcome(rate(g - 1, n, home, adv), rate(-g + 1, n, 1 - home, -adv), n, g - 1)
    grid['p_win'], grid['p_draw'] = w0, d0
    grid['W'] = 3 * w0 + d0
    grid['L_plus'] = (3 * w_up + d_up) - grid['W']
    grid['L_minus'] = grid['W'] - (3 * w_dn + d_dn)
    grid['scoring_only'] = np.abs(w_up - w0) + np.abs(d_up - d0)      # the in-game pressure of Bransen et al. (2019)
    return grid


def attach(df, table, g='goal_diff', n='n_left', home='is_home', adv='man_adv'):
    """Look up the stakes of each row. States outside the grid are clipped to its edge."""
    key = pd.DataFrame({'g': np.clip(df[g].values, G_RANGE[0], G_RANGE[-1]), 'n_left': df[n].values,
                        'home': df[home].values.astype(int), 'adv': np.clip(df[adv].values, ADV_RANGE[0], ADV_RANGE[-1])})
    return key.merge(table, on=['g', 'n_left', 'home', 'adv'], how='left', validate='many_to_one')


def ranked_probability_score(p_win, p_draw, pts):
    """Ranked probability score of a win/draw/loss forecast against the final result."""
    return 0.5 * (((1 - p_win - p_draw) - (pts == 0)) ** 2 + (p_win - (pts == 3)) ** 2)
