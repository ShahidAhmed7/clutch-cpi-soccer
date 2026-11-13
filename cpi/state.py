"""Match state at each action: the score (own goals included), the clock and the red-card balance."""
import numpy as np
import pandas as pd

from . import config as C


def elapsed_seconds(period_id, time_seconds):
    """Seconds into a 90-minute match. Stoppage time stays in the last moment of its half, so the clock never runs back."""
    return (np.asarray(period_id) - 1) * C.HALF_SECONDS + np.minimum(np.asarray(time_seconds, dtype=float), C.HALF_SECONDS - 0.1)


def add_state(actions, matches, reds=None, count_own_goals=True):
    """Add the score before each action from the acting team's side, the time frame and the red-card balance.

    `actions` must be in match order and hold the boolean columns `goal` and `own_goal`.
    `count_own_goals=False` reproduces what a shots-only score tracker would see.
    """
    df = actions.merge(matches[['game_id', 'home_team_id']], on='game_id', how='left', validate='many_to_one')
    home = (df['team_id'] == df['home_team_id']).values
    own = df['own_goal'].values if count_own_goals else np.zeros(len(df), bool)
    goal = df['goal'].values

    home_scores = (goal & home) | (own & ~home)
    away_scores = (goal & ~home) | (own & home)
    games = df['game_id'].values
    home_before = pd.Series(home_scores.astype(int)).groupby(games).cumsum().values - home_scores
    away_before = pd.Series(away_scores.astype(int)).groupby(games).cumsum().values - away_scores

    df['is_home'] = home
    df['goals_team'] = np.where(home, home_before, away_before)
    df['goals_opp'] = np.where(home, away_before, home_before)
    df['goal_diff'] = df['goals_team'] - df['goals_opp']
    df['elapsed'] = elapsed_seconds(df['period_id'], df['time_seconds'])
    df['t_frame'] = (df['elapsed'] / C.SECS_PER_FRAME).astype(int).clip(upper=C.T_FRAMES - 1)
    df['n_left'] = C.T_FRAMES - df['t_frame']
    df['minute'] = (df['period_id'] - 1) * 45 + df['time_seconds'] / 60
    df['man_adv'] = man_advantage(df, reds) if reds is not None else 0
    return df.drop(columns='home_team_id')


def man_advantage(df, reds):
    """Red cards shown to the opponent minus red cards shown to the acting team, before each action."""
    out = np.zeros(len(df), dtype=np.int8)
    if reds is None or not len(reds):
        return out
    reds = reds.assign(elapsed=elapsed_seconds(reds['period_id'], reds['time_seconds']))
    index = pd.Series(np.arange(len(df))).groupby(df['game_id'].values).indices
    elapsed, teams = df['elapsed'].values, df['team_id'].values
    for r in reds.itertuples():
        ix = index.get(r.game_id)
        if ix is None:
            continue
        after = ix[elapsed[ix] > r.elapsed]
        out[after] += np.where(teams[after] == r.team_id, -1, 1).astype(np.int8)
    return out


def final_score_check(actions, matches):
    """Reconstructed goals against official scores, one row per team and match."""
    home = dict(zip(matches['game_id'], matches['home_team_id']))
    away = dict(zip(matches['game_id'], matches['away_team_id']))
    a = actions[actions['goal'] | actions['own_goal']]
    other = np.where(a['team_id'].values == a['game_id'].map(home).values, a['game_id'].map(away).values, a['game_id'].map(home).values)
    credited = np.where(a['goal'].values, a['team_id'].values, other)
    made = pd.DataFrame({'game_id': a['game_id'].values, 'team_id': credited}).groupby(['game_id', 'team_id']).size().rename('made')
    official = pd.concat([
        matches[['game_id', 'home_team_id', 'home_score']].set_axis(['game_id', 'team_id', 'official'], axis=1),
        matches[['game_id', 'away_team_id', 'away_score']].set_axis(['game_id', 'team_id', 'official'], axis=1)])
    out = official.merge(made.reset_index(), on=['game_id', 'team_id'], how='left').fillna({'made': 0})
    out['made'] = out['made'].astype(int)
    return out
