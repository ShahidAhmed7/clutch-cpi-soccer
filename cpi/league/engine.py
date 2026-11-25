"""From a Bundle to action values, stakes, keeper points saved and player ratings, with the paper's code.

Model choice (--models)
    pretrained  the models fitted on all five 2017/18 leagues (models/). Works for any amount of data, including one
                team's matches or a single match. Accuracy on the new data is measured and reported.
    retrain     new action-value models on this data with 5-fold cross-fitting by match, so no action is valued by a
                model that saw its match. Needs a full season of a league (about 300+ matches) to be reliable.
                The stakes model is refitted only for a full league; otherwise the 2017/18 stakes are used.
    auto        retrain for a full league of at least 300 matches, pretrained otherwise.

Pretrained models on another provider's data (tested on Hudl StatsBomb, Bundesliga 2023/24)
    - harmonize(): StatsBomb records every carry, so its actions are finer than Wyscout's and a 10-action window covers
      less play. Replacing native carries with socceraction's synthetic dribbles (as for Wyscout) brought the scoring
      model to near-perfect calibration (logistic recalibration intercept 0.11, slope 1.07).
    - recalibrate(): the conceding model still over-predicted (0.69% against 0.36% observed), because an opponent's attack
      spans more recorded actions. A two-parameter logistic recalibration per model, cross-fitted by match (no match
      calibrates its own predictions), fixes the level and leaves the ranking (ROC-AUC) unchanged.
"""
import warnings
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from .. import config as C, keepers, labels, ratings, stakes, state, values
from ..features import game_features
from . import bundle as B, models as M

warnings.filterwarnings('ignore')
KEEP = ['game_id', 'team_id', 'player_id', 'period_id', 'time_seconds', 'minute', 'type_name', 'result_name', 'bodypart_name', 'original_event_id',
        'goal', 'own_goal', 'assist', 'is_home', 'goal_diff', 'goals_team', 'goals_opp', 'n_left', 'man_adv', 'start_x', 'start_y', 'end_x', 'end_y']
RETRAIN_MIN_MATCHES = 300


def harmonize(actions, drop_carries=True):
    """Bring another provider's actions to the conventions the 2017/18 Wyscout models were trained on.

    StatsBomb records every carry, so a 10-action window covers less play than in Wyscout data. Native carries are
    removed and socceraction's synthetic dribbles (3-60 m, within 10 s, as added for Wyscout) are inserted instead.
    (Body parts need no change: the reference features already treat left and right foot as one.)
    Returns the actions and a list of what changed.
    """
    from socceraction import spadl
    from socceraction.spadl.base import _add_dribbles
    done = []
    a = actions.copy()
    if drop_carries and 'source_type' in a and (a['source_type'] == 'Carry').any():
        n0 = len(a)
        a = a[~((a['source_type'] == 'Carry') & (a['type_name'] == 'dribble'))]
        parts = []
        for _, g in a.groupby('game_id', sort=False):
            g = g.sort_values(['period_id', 'action_id'], kind='stable').reset_index(drop=True)
            g['action_id'] = np.arange(len(g))
            parts.append(_add_dribbles(g[[c for c in g.columns if c not in ('type_name', 'result_name', 'bodypart_name')]]))
        a = spadl.add_names(pd.concat(parts, ignore_index=True))
        for c in ('goal', 'own_goal', 'assist'):
            a[c] = a[c].fillna(False).astype(bool)
        a['source_type'] = a['source_type'].fillna('synthetic dribble')
        done.append(f'native carries replaced by synthetic dribbles ({n0:,} to {len(a):,} actions)')
    return a.reset_index(drop=True), done


def recalibrate(p, y, games, folds=5, min_events=30):
    """Logistic recalibration logit(p') = a + b logit(p), cross-fitted by match. Returns (p', (a, b)) or (p, None)."""
    from sklearn.linear_model import LogisticRegression
    y = np.asarray(y)
    if y.sum() < min_events or len(pd.unique(games)) < 2 * folds:
        return p, None
    z = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6))).reshape(-1, 1)
    uniq = np.array(sorted(pd.unique(games)))
    fold_of = dict(zip(uniq, np.random.default_rng(C.SEED).permutation(len(uniq)) % folds))
    fold = np.array([fold_of[g] for g in games])
    out = np.zeros(len(p))
    for k in range(folds):
        m = LogisticRegression(C=1e6).fit(z[fold != k], y[fold != k])
        out[fold == k] = m.predict_proba(z[fold == k])[:, 1]
    full = LogisticRegression(C=1e6).fit(z, y)
    return out, (float(full.intercept_[0]), float(full.coef_[0][0]))


def _features(args):
    a, home = args
    return game_features(a, home)


def build_features(actions, matches, workers=6):
    home = dict(zip(matches['game_id'], matches['home_team_id']))
    jobs = [(g, home[gid]) for gid, g in actions.groupby('game_id', sort=False)]
    if workers > 1 and len(jobs) > 4:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            parts = list(pool.map(_features, jobs, chunksize=4))
    else:
        parts = [_features(j) for j in jobs]
    X = pd.concat(parts, ignore_index=True)
    assert len(X) == len(actions), 'features and actions must stay row-aligned'
    return X


def accuracy(y, p):
    y = np.asarray(y)
    if y.min() == y.max():
        return {'auc': None, 'brier': float(brier_score_loss(y, p)), 'rate': float(y.mean()), 'mean_pred': float(np.mean(p)), 'n': int(len(y))}
    return {'auc': float(roc_auc_score(y, p)), 'brier': float(brier_score_loss(y, p)), 'log_loss': float(log_loss(y, p, labels=[0, 1])),
            'rate': float(y.mean()), 'mean_pred': float(np.mean(p)), 'n': int(len(y))}


def calibration(y, p, bins=10):
    """Observed rate against predicted probability in deciles of the prediction."""
    d = pd.DataFrame({'y': y, 'p': p})
    d['bin'] = pd.qcut(d['p'].rank(method='first'), bins, labels=False)
    return d.groupby('bin').agg(predicted=('p', 'mean'), observed=('y', 'mean'), n=('y', 'size')).reset_index()


def crossfit_vaep(X, y_s, y_c, games, cols, cats, folds=5, log=print):
    """Out-of-fold predictions: every match is scored by models trained on the other folds."""
    from catboost import CatBoostClassifier, Pool
    rng = np.random.default_rng(C.SEED)
    uniq = np.array(sorted(pd.unique(games)))
    fold_of = dict(zip(uniq, rng.permutation(len(uniq)) % folds))
    f = np.array([fold_of[g] for g in games])
    preds = {'scores': np.zeros(len(X)), 'concedes': np.zeros(len(X))}
    for k in range(folds):
        test = f == k
        train_games = uniq[[fold_of[g] != k for g in uniq]]
        val_games = set(rng.choice(train_games, max(1, int(len(train_games) * 0.15)), replace=False))
        val = np.array([g in val_games for g in games]) & ~test
        tr = ~test & ~val
        for target, y in (('scores', y_s), ('concedes', y_c)):
            m = CatBoostClassifier(**M.CATBOOST, random_seed=C.SEED, task_type='CPU', thread_count=-1, verbose=0, use_best_model=True, allow_writing_files=False)
            m.fit(Pool(X.loc[tr, cols], y[tr], cat_features=cats), eval_set=Pool(X.loc[val, cols], y[val], cat_features=cats))
            preds[target][test] = m.predict_proba(X.loc[test, cols])[:, 1]
        log(f'  fold {k + 1}/{folds} done')
    return preds['scores'], preds['concedes']


def run(b, models='auto', min_minutes=None, n_boot=C.N_BOOT, workers=6, harmonize_actions=True, keep_carries=False,
        recalibrate_models=True, log=print):
    """Everything the report needs, as a dict of tables."""
    out = {'bundle': b, 'checks': B.checks(b)}
    ck = out['checks']
    mode = models
    if mode == 'auto':
        mode = 'retrain' if ck['full_league'] and ck['matches'] >= RETRAIN_MIN_MATCHES else 'pretrained'
    out['mode'] = mode

    acts, out['harmonized'] = b.actions, []
    if mode == 'pretrained' and harmonize_actions and b.provider != 'wyscout_v2':
        acts, out['harmonized'] = harmonize(b.actions, drop_carries=not keep_carries)
        for h in out['harmonized']:
            log('harmonized: ' + h)
    actions = state.add_state(acts, b.matches, b.reds if len(b.reds) else None)
    log(f'features for {len(actions):,} actions in {ck["matches"]} matches ...')
    X = build_features(actions, b.matches, workers)
    y_s, y_c = labels.build_labels(actions)
    pre = M.load()
    cols = pre['meta']['feature_columns']
    missing = [c for c in cols if c not in X]
    if missing:
        raise RuntimeError(f'features missing for the pretrained models: {missing}')
    if mode == 'retrain':
        log('retraining action-value models with 5-fold cross-fitting by match ...')
        p_s, p_c = crossfit_vaep(X, y_s, y_c, actions['game_id'].values, cols, pre['meta']['categorical'], log=log)
    else:
        p_s = pre['vaep']['scores'].predict_proba(X[cols])[:, 1]
        p_c = pre['vaep']['concedes'].predict_proba(X[cols])[:, 1]
    out['model_accuracy_raw'] = {'scores': accuracy(y_s, p_s), 'concedes': accuracy(y_c, p_c)}
    out['recalibration'] = None
    if mode == 'pretrained' and recalibrate_models:
        games = actions['game_id'].values
        p_s2, par_s = recalibrate(p_s, y_s, games)
        p_c2, par_c = recalibrate(p_c, y_c, games)
        if par_s and par_c:
            p_s, p_c = p_s2, p_c2
            out['recalibration'] = {'scores': par_s, 'concedes': par_c}
            log(f'recalibrated: scoring a={par_s[0]:.2f} b={par_s[1]:.2f}; conceding a={par_c[0]:.2f} b={par_c[1]:.2f}')
        else:
            log('not recalibrated: too few goals or matches')
    out['model_accuracy'] = {'scores': accuracy(y_s, p_s), 'concedes': accuracy(y_c, p_c),
                             'paper_held_out_auc': pre['meta'].get('paper_held_out_auc')}
    out['calibration'] = {'scores': calibration(y_s, p_s), 'concedes': calibration(y_c, p_c)}

    # stakes: refit only on a full league, otherwise the 2017/18 outcome model
    table = pre['stakes_table']
    out['stakes_source'] = 'pretrained (2017/18 five leagues)'
    if mode == 'retrain' and ck['full_league']:
        panel = stakes.build_panel(actions, b.matches, b.reds)
        spec = tuple(pre['meta'].get('stakes_spec', ['base']))
        table = stakes.stake_table(stakes.fit_rate_model(panel, spec), spec)
        out['stakes_source'] = f'refitted on this data ({len(panel):,} team-frames)'
    out['stakes_table'] = table

    ds, dc = values.channels(actions, p_s, p_c)
    t = stakes.attach(actions, table)
    v = actions[KEEP].copy()
    v['p_scores'], v['p_concedes'], v['dP_scores'], v['dP_concedes'] = p_s, p_c, ds, dc
    for c in ('W', 'p_win', 'p_draw', 'L_plus', 'L_minus', 'scoring_only'):
        v[c] = t[c].values
    v['value'] = v['dP_scores'] - v['dP_concedes']
    v['attack'] = v['dP_scores'] * v['L_plus']
    v['defend'] = -v['dP_concedes'] * v['L_minus']
    v['V'] = v['attack'] + v['defend']
    v['V_scoring_only'] = v['value'] * v['scoring_only']
    out['values'] = v

    # goalkeepers: post-shot xG weighted by what conceding would cost at that moment
    shots = b.shots[b.shots['zone'].notna()].reset_index(drop=True) if len(b.shots) else b.shots
    ks = pd.DataFrame(columns=['keeper_id', 'game_id', 'points_saved', 'goals_prevented', 'goal'])
    if len(shots):
        d = keepers.shot_features(shots)
        d['psxg'] = pre['psxg'].predict_proba(d[keepers.FEATURES])[:, 1]
        raw_goals_pred = float(d['psxg'].sum())
        out['psxg_recalibration'] = None
        if mode == 'pretrained' and recalibrate_models and len(d) >= 100:
            p2, par = recalibrate(d['psxg'].values, d['goal'].astype(int).values, d['game_id'].values)
            if par:
                d['psxg'], out['psxg_recalibration'] = p2, par
        home = dict(zip(b.matches['game_id'], b.matches['home_team_id']))
        away = dict(zip(b.matches['game_id'], b.matches['away_team_id']))
        d['defending_team_id'] = np.where(d['team_id'].values == d['game_id'].map(home).values, d['game_id'].map(away).values, d['game_id'].map(home).values)
        sc = v.loc[v['type_name'].str.contains('shot|corner|freekick'), ['original_event_id', 'goal_diff', 'n_left', 'is_home', 'man_adv']].drop_duplicates('original_event_id')
        d = d.merge(sc, left_on='event_id', right_on='original_event_id', how='inner')
        faced = pd.DataFrame({'g': -d['goal_diff'].values, 'n_left': d['n_left'].values, 'home': (~d['is_home'].astype(bool)).astype(int).values, 'adv': -d['man_adv'].values})
        d['L_minus_keeper'] = stakes.attach(faced, table, g='g', n='n_left', home='home', adv='adv')['L_minus'].values
        d['keeper_id'] = keepers.keeper_on_pitch(d, b.appearances)
        d['goals_prevented'] = d['psxg'] - d['goal'].astype(float)
        d['points_saved'] = d['goals_prevented'] * d['L_minus_keeper']
        ks = d
        out['psxg_check'] = {'shots': int(len(d)), 'goals': int(d['goal'].sum()), 'predicted': float(d['psxg'].sum()), 'predicted_raw': raw_goals_pred,
                             'auc': float(roc_auc_score(d['goal'], d['psxg'])) if d['goal'].nunique() == 2 else None}
    out['keeper_shots'] = ks

    # player ratings with match-bootstrap intervals and Empirical-Bayes adjustment within position
    pm = ratings.player_match_table(v, ks, b.appearances)
    pl = ratings.per90(pm).join(b.players.set_index('player_id')[['name', 'position']])
    pl['team'] = pl['team_id'].map(b.teams.set_index('team_id')['team'])
    if min_minutes is None:
        top = pl['minutes'].max() if len(pl) else 0
        min_minutes = C.MIN_MINUTES if top >= 2 * C.MIN_MINUTES else max(round(0.4 * top), 1)
    out['min_minutes'] = min_minutes
    focus = ck.get('focus_team_id')
    pl['qualified'] = (pl['minutes'] >= min_minutes) & pl['position'].isin(C.POSITIONS)
    if focus is not None:
        pl['qualified'] &= pl['team_id'] == focus           # opponents appear in one or two matches only
    q = pl[pl['qualified']].copy()
    log(f'{len(q)} players with at least {min_minutes:.0f} minutes' + (' (focus team only)' if focus is not None else ''))
    if len(q):
        ids = q.index.values
        draws = ratings.bootstrap(pm, ids, n_boot=n_boot)
        q['CPI'] = q['V_per90']
        q['ci_low'] = np.nanpercentile(draws['V'], 2.5, axis=0)
        q['ci_high'] = np.nanpercentile(draws['V'], 97.5, axis=0)
        for p in C.POSITIONS:
            m = (q['position'] == p).values
            if m.sum() >= 10:                                  # Empirical Bayes needs enough players to estimate the spread
                post, sd, _, _ = ratings.shrink(q.loc[m, 'V_per90'].values, draws['V'][:, m])
                q.loc[m, 'CPI_adjusted'], q.loc[m, 'CPI_sd'] = post, sd
                q.loc[m, 'P_top10'] = ratings.top_k_probability(post, sd, k=min(10, max(1, m.sum() // 2)))
        q['rank'] = q.groupby('position')['CPI'].rank(ascending=False)
        q['rank_vaep'] = q.groupby('position')['value_per90'].rank(ascending=False)
    out['player_match'] = pm
    out['players_all'] = pl
    out['ratings'] = q
    return out
