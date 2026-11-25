"""Pretrained CPI models, fitted once on all five 2017/18 leagues and saved, so any new league can be valued.

    models/vaep_scores.cbm, models/vaep_concedes.cbm   action-value models (same settings as the paper, all leagues)
    models/stakes_table.parquet                        L+, L-, W for every (goal difference, frames left, home, red cards)
    models/post_shot_xg.joblib                         post-shot xG for on-target shots (goal-mouth zone, location, type)
    models/meta.json                                   settings, versions and held-out accuracy from the paper's folds

The paper's own numbers come from leave-one-league-out models (pipeline/s3, s4, s5). These final models use every league,
which is the standard way to deploy a model once its held-out accuracy is known.
"""
import json
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from .. import config as C, keepers, stakes

MODEL_DIR = C.ROOT / 'models'
TARGETS = ('scores', 'concedes')
CATBOOST = dict(iterations=1000, learning_rate=0.05, depth=6, l2_leaf_reg=3.0, eval_metric='Logloss', early_stopping_rounds=50)
PSXG = dict(max_iter=200, learning_rate=0.05, max_depth=4, min_samples_leaf=30)


def _feature_columns(frame):
    return [c for c in frame.columns if c not in ('y_scores', 'y_concedes', 'game_id')]


def train_vaep(log=print):
    """Both action-value models on all leagues, with early stopping on 15% of matches (as in the paper)."""
    from catboost import CatBoostClassifier, Pool
    frames = [pd.read_parquet(C.WORK / f'features_full_{lg}.parquet') for lg in C.LEAGUES]
    data = pd.concat(frames, ignore_index=True)
    del frames
    cols = _feature_columns(data)
    cats = [c for c in C.CATEGORICAL if c in cols]
    rng = np.random.default_rng(C.SEED)
    games = data['game_id'].unique()
    val = data['game_id'].isin(rng.choice(games, int(len(games) * 0.15), replace=False)).values
    out = {'feature_columns': cols, 'categorical': cats, 'n_train_actions': int((~val).sum()), 'n_val_actions': int(val.sum())}
    for target in TARGETS:
        t0 = time.time()
        y = data[f'y_{target}'].values
        m = CatBoostClassifier(**CATBOOST, random_seed=C.SEED, task_type='CPU', thread_count=-1, verbose=0, use_best_model=True, allow_writing_files=False)
        m.fit(Pool(data.loc[~val, cols], y[~val], cat_features=cats), eval_set=Pool(data.loc[val, cols], y[val], cat_features=cats))
        m.save_model(str(MODEL_DIR / f'vaep_{target}.cbm'))
        out[f'best_iteration_{target}'] = int(m.get_best_iteration())
        log(f'vaep {target}: best iteration {out[f"best_iteration_{target}"]}, {time.time() - t0:.0f}s')
    return out


def train_stakes(log=print):
    """Outcome model on the full team-by-frame panel, turned into the lookup table of stakes."""
    spec = tuple(json.loads((C.WORK / 'stakes_spec.json').read_text())['spec'])
    panel = pd.read_parquet(C.WORK / 'panel.parquet')
    model = stakes.fit_rate_model(panel, spec)
    table = stakes.stake_table(model, spec)
    table.to_parquet(MODEL_DIR / 'stakes_table.parquet', index=False)
    log(f'stakes: spec {spec}, {len(table):,} states')
    return {'stakes_spec': list(spec), 'stakes_panel_rows': int(len(panel))}


def train_psxg(log=print):
    import joblib
    from sklearn.ensemble import HistGradientBoostingClassifier
    shots = pd.read_parquet(C.WORK / 'shots.parquet')
    d = keepers.shot_features(shots[shots['zone'].notna()].reset_index(drop=True))
    m = HistGradientBoostingClassifier(**PSXG, categorical_features=[keepers.FEATURES.index('zone_code'), keepers.FEATURES.index('kind_code')], random_state=C.SEED)
    m.fit(d[keepers.FEATURES], d['goal'].astype(int))
    joblib.dump(m, MODEL_DIR / 'post_shot_xg.joblib')
    log(f'post-shot xG: {len(d):,} on-target shots')
    return {'psxg_shots': int(len(d))}


def train_all(log=print):
    import catboost
    import sklearn
    MODEL_DIR.mkdir(exist_ok=True)
    meta = {'trained_utc': datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M'), 'training_data': 'Wyscout public event data 2017/18, '
            'top divisions of England, France, Germany, Italy and Spain (Pappalardo et al., 2019)', 'catboost': catboost.__version__,
            'sklearn': sklearn.__version__, 'catboost_settings': CATBOOST, 'psxg_settings': PSXG, 'window': C.WINDOW, 'seed': C.SEED}
    meta.update(train_stakes(log))
    meta.update(train_psxg(log))
    meta.update(train_vaep(log))
    held = pd.read_csv(C.WORK / 'metrics_full_catboost_s42.csv').drop_duplicates(['held_out', 'target'], keep='last')
    meta['paper_held_out_auc'] = held.groupby('target')['auc'].mean().round(4).to_dict()
    (MODEL_DIR / 'meta.json').write_text(json.dumps(meta, indent=1))
    log('models saved to ' + str(MODEL_DIR))
    return meta


def load():
    """Everything needed to value a new dataset. Trains first if the models are missing."""
    import joblib
    from catboost import CatBoostClassifier
    if not (MODEL_DIR / 'meta.json').exists():
        raise FileNotFoundError('No pretrained models in models/. Run:  python run_league.py train')
    meta = json.loads((MODEL_DIR / 'meta.json').read_text())
    vaep = {}
    for target in TARGETS:
        m = CatBoostClassifier()
        m.load_model(str(MODEL_DIR / f'vaep_{target}.cbm'))
        vaep[target] = m
    return {'vaep': vaep, 'stakes_table': pd.read_parquet(MODEL_DIR / 'stakes_table.parquet'),
            'psxg': joblib.load(MODEL_DIR / 'post_shot_xg.joblib'), 'meta': meta}
