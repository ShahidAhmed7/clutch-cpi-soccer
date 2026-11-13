"""Stage 3. Leave-one-league-out action-value models: P(team scores) and P(team concedes) in the window.

    python pipeline/s3_action_values.py --model catboost            # paper results, GPU if present
    python pipeline/s3_action_values.py --model hgb --variant mirrored   # sensitivity and trust experiment

A finished fold is reused only if its input file, model settings and seed are unchanged, so results can never go stale
silently. Each fold writes work/preds_<tag>_<league>.parquet and one line per target to work/metrics_<tag>.csv.
"""
import argparse
import gc
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cpi import config as C  # noqa: E402

TARGETS = ('scores', 'concedes')
SETTINGS = {
    'catboost': dict(iterations=1000, learning_rate=0.05, depth=6, l2_leaf_reg=3.0, eval_metric='Logloss', early_stopping_rounds=50),
    'hgb': dict(max_iter=300, learning_rate=0.08, max_depth=6, l2_regularization=3.0),
}


def has_gpu():
    try:
        subprocess.check_output('nvidia-smi', shell=True, stderr=subprocess.DEVNULL)
        return True
    except subprocess.CalledProcessError:
        return False


def fit_predict(model, Xtr, ytr, Xva, yva, Xte, cats, seed):
    if model == 'catboost':
        from catboost import CatBoostClassifier, Pool
        task = 'GPU' if has_gpu() else 'CPU'
        m = CatBoostClassifier(**SETTINGS['catboost'], random_seed=seed, task_type=task, verbose=0, use_best_model=True,
                               **({'devices': '0'} if task == 'GPU' else {}))
        m.fit(Pool(Xtr, ytr, cat_features=cats), eval_set=Pool(Xva, yva, cat_features=cats))
        return m.predict_proba(Xte)[:, 1], int(m.get_best_iteration())
    from sklearn.ensemble import HistGradientBoostingClassifier
    m = HistGradientBoostingClassifier(**SETTINGS['hgb'], categorical_features=[c in cats for c in Xtr.columns],
                                       random_state=seed, early_stopping=False)
    m.fit(pd.concat([Xtr, Xva]), np.concatenate([ytr, yva]))
    return m.predict_proba(Xte)[:, 1], SETTINGS['hgb']['max_iter']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='catboost', choices=list(SETTINGS))
    ap.add_argument('--variant', default='full')
    ap.add_argument('--seed', type=int, default=C.SEED)
    ap.add_argument('--drop', default='', help='comma-separated substring of feature names to leave out')
    ap.add_argument('--window', type=int, default=C.WINDOW, help='label window, for the sensitivity check')
    args = ap.parse_args()
    tag = f'{args.variant}_{args.model}_s{args.seed}' + (f'_k{args.window}' if args.window != C.WINDOW else '')

    paths = {lg: C.WORK / f'features_{args.variant}_{lg}.parquet' for lg in C.LEAGUES}
    signature = hashlib.sha256(json.dumps({'settings': SETTINGS[args.model], 'seed': args.seed, 'drop': args.drop, 'window': args.window,
                                           'inputs': {lg: paths[lg].stat().st_size for lg in C.LEAGUES}}, sort_keys=True).encode()).hexdigest()[:16]

    frames = {lg: pd.read_parquet(paths[lg]) for lg in C.LEAGUES}
    if args.window != C.WINDOW:
        from cpi.labels import build_labels
        for lg in C.LEAGUES:
            a = pd.read_parquet(C.WORK / f'actions_{lg}.parquet', columns=['game_id', 'team_id', 'goal', 'own_goal'])
            frames[lg]['y_scores'], frames[lg]['y_concedes'] = build_labels(a, window=args.window, count_own_goals=args.variant != 'no_own_goals')
    feature_cols = [c for c in frames[C.LEAGUES[0]].columns if c not in ('y_scores', 'y_concedes', 'game_id')
                    and not any(d and d in c for d in args.drop.split(','))]
    cats = [c for c in C.CATEGORICAL if c in feature_cols]
    metrics_path = C.WORK / f'metrics_{tag}.csv'

    for held in C.LEAGUES:
        out = C.WORK / f'preds_{tag}_{held}.parquet'
        sig_path = out.with_suffix('.sig')
        if out.exists() and sig_path.exists() and sig_path.read_text() == signature:
            print(f'{held}: already done with this input and settings, reused', flush=True)
            continue
        t0 = time.time()
        rng = np.random.default_rng(args.seed)
        train = pd.concat([frames[lg] for lg in C.LEAGUES if lg != held], ignore_index=True)
        games = train['game_id'].unique()
        val_games = rng.choice(games, int(len(games) * 0.15), replace=False)
        val = train['game_id'].isin(val_games).values
        test = frames[held]
        preds, rows = {}, []
        for target in TARGETS:
            y = f'y_{target}'
            p, best = fit_predict(args.model, train.loc[~val, feature_cols], train.loc[~val, y].values,
                                  train.loc[val, feature_cols], train.loc[val, y].values, test[feature_cols], cats, args.seed)
            preds[f'p_{target}'] = p.astype('float32')
            yt = test[y].values
            rows.append({'held_out': held, 'target': target, 'best_iter': best, 'auc': roc_auc_score(yt, p), 'brier': brier_score_loss(yt, p),
                         'log_loss': log_loss(yt, p), 'rate': yt.mean(), 'mean_pred': float(p.mean()), 'n': len(yt), 'seconds': round(time.time() - t0)})
            print(f"{held} {target}: AUC {rows[-1]['auc']:.4f} Brier {rows[-1]['brier']:.5f} best_iter {best} ({rows[-1]['seconds']}s)", flush=True)
        pd.DataFrame(preds).to_parquet(out, index=False)
        sig_path.write_text(signature)
        pd.DataFrame(rows).to_csv(metrics_path, mode='a', header=not metrics_path.exists(), index=False)
        del train, test, preds
        gc.collect()
    print('stage 3 done', tag)


if __name__ == '__main__':
    main()
