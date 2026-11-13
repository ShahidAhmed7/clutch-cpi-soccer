"""Player ratings in league points per 90 minutes, with uncertainty.

CPI        = (sum of V over the season) / minutes * 90
Goalkeepers: the V of their save events is replaced by points saved, because a save's action value says nothing about the shot.
"""
import numpy as np
import pandas as pd
from scipy import sparse

from . import config as C


def player_match_table(values, keeper_shots, appearances):
    """One row per player and match: minutes and the sums every rating is built from."""
    v = values[values['player_id'] > 0]                    # events with no player stay in the sequence but earn no credit
    is_save = v['type_name'] == 'keeper_save'
    is_pen = v['type_name'] == 'shot_penalty'
    v = v.assign(V_cpi=np.where(is_save, 0.0, v['V']),
                 attack_np=np.where(is_pen, 0.0, v['attack']), V_np=np.where(is_save | is_pen, 0.0, v['V']))
    pm = v.groupby(['player_id', 'game_id'], sort=False).agg(
        team_id=('team_id', 'first'), n=('V', 'size'), V_raw=('V', 'sum'), V=('V_cpi', 'sum'), V_np=('V_np', 'sum'), attack=('attack', 'sum'),
        defend=('defend', 'sum'), value=('value', 'sum'), V_scoring_only=('V_scoring_only', 'sum'), goals=('goal', 'sum'),
        assists=('assist', 'sum'), own_goals=('own_goal', 'sum'), penalties=('type_name', lambda s: (s == 'shot_penalty').sum())).reset_index()
    saved = keeper_shots[keeper_shots['keeper_id'] > 0].groupby(['keeper_id', 'game_id']).agg(
        points_saved=('points_saved', 'sum'), goals_prevented=('goals_prevented', 'sum'), shots_faced=('goal', 'size')).reset_index().rename(columns={'keeper_id': 'player_id'})
    pm = pm.merge(saved, on=['player_id', 'game_id'], how='outer').merge(appearances[['player_id', 'game_id', 'minutes']], on=['player_id', 'game_id'], how='left')
    fill = [c for c in pm.columns if c not in ('player_id', 'game_id', 'team_id', 'minutes')]
    pm[fill] = pm[fill].fillna(0.0)
    pm['minutes'] = pm['minutes'].fillna(0.0)
    pm['V'] = pm['V'] + pm['points_saved']            # CPI numerator: on-ball value plus shot-stopping
    pm['V_np'] = pm['V_np'] + pm['points_saved']
    return pm


METRICS = ['V', 'V_np', 'V_raw', 'attack', 'defend', 'value', 'V_scoring_only', 'points_saved', 'goals_prevented']


def per90(pm):
    g = pm.groupby('player_id')
    out = g[METRICS + ['n', 'minutes', 'goals', 'assists', 'own_goals', 'penalties', 'shots_faced']].sum()
    out['matches'] = g.size()
    out['team_id'] = pm.sort_values('minutes').groupby('player_id')['team_id'].last()
    for m in METRICS:
        out[f'{m}_per90'] = out[m] / out['minutes'].replace(0, np.nan) * 90
    out['ga_per90'] = (out['goals'] + out['assists']) / out['minutes'].replace(0, np.nan) * 90
    return out


def bootstrap(pm, players, metrics=('V', 'value', 'attack', 'defend', 'points_saved'), n_boot=C.N_BOOT, seed=C.SEED):
    """Resample whole matches. Returns {metric: array (draws, players)} of per-90 ratings."""
    pm = pm[pm['player_id'].isin(players)]
    games = np.sort(pm['game_id'].unique())
    gi = pd.Series(np.arange(len(games)), index=games).loc[pm['game_id']].values
    pi = pd.Series(np.arange(len(players)), index=players).loc[pm['player_id']].values
    mats = {k: sparse.csr_matrix((pm[k].values.astype(float), (pi, gi)), shape=(len(players), len(games))) for k in set(metrics) | {'minutes'}}
    rng = np.random.default_rng(seed)
    out = {k: np.empty((n_boot, len(players))) for k in metrics}
    for b in range(n_boot):
        w = np.bincount(rng.integers(0, len(games), len(games)), minlength=len(games)).astype(float)
        mins = mats['minutes'] @ w
        mins[mins == 0] = np.nan
        for k in metrics:
            out[k][b] = mats[k] @ w / mins * 90
    return out


def shrink(obs, draws):
    """Empirical Bayes within one position: normal prior, sampling variance from the match bootstrap.

    Returns posterior mean, posterior sd, the prior variance and the discrimination D = 1 - E[sampling variance] / between-player variance.
    """
    se2 = np.nanvar(draws, axis=0, ddof=1)
    total = np.nanvar(obs, ddof=1)
    tau2 = max(total - np.nanmean(se2), 1e-9)
    w = tau2 / (tau2 + se2)
    mean = np.average(obs, weights=1 / (tau2 + se2))
    post = w * obs + (1 - w) * mean
    return post, np.sqrt(w * se2), tau2, 1 - np.nanmean(se2) / total


def top_k_probability(post, post_sd, k=10, n=20000, seed=C.SEED):
    """Chance of being in the true top k of the position, from the shrunken posterior."""
    rng = np.random.default_rng(seed)
    draws = rng.normal(post, post_sd, size=(n, len(post)))
    ranks = (-draws).argsort(axis=1).argsort(axis=1)
    return (ranks < k).mean(axis=0)


def split_half(pm, players, position, metrics, matches, sb=True):
    """Odd against even gameweeks, per position; Spearman-Brown corrected. Players need 450 minutes in each half."""
    from scipy.stats import spearmanr
    d = pm[pm['player_id'].isin(players)].merge(matches[['game_id', 'gameweek']], on='game_id')
    d = d.assign(half=np.where(d['gameweek'] % 2 == 1, 'odd', 'even'))
    g = d.groupby(['player_id', 'half'])[list(metrics) + ['minutes']].sum()
    wide = g.unstack('half')
    wide = wide[(wide[('minutes', 'odd')] >= 450) & (wide[('minutes', 'even')] >= 450)]
    rows = []
    for p in C.POSITIONS:
        x = wide[position.reindex(wide.index) == p]
        row = {'position': p, 'players': len(x)}
        for m in metrics:
            a = x[(m, 'odd')] / x[('minutes', 'odd')]
            b = x[(m, 'even')] / x[('minutes', 'even')]
            rho = spearmanr(a, b).statistic
            row[m] = 2 * rho / (1 + rho) if sb and rho > -1 else rho
        rows.append(row)
    return pd.DataFrame(rows)
