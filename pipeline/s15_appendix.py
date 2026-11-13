"""Stage 15. What CPI changes relative to VAEP, a case study, worked examples and the calculation details for the appendix.

    python pipeline/s15_appendix.py

Writes results/extra_claims.json (merged into the paper's numbers by s14) and paper/tables/*.tex.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binom, spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cpi import config as C, ratings, stakes  # noqa: E402

R, T = C.RESULTS, C.ROOT / 'paper' / 'tables'
X = {}
nz = lambda x: (f'{x:.2f}'.replace('0.', '.', 1) if 0 <= x < 1 else f'{x:.2f}'.replace('-0.', '$-$.', 1) if -1 < x < 0 else f'{x:.2f}'.replace('-', '$-$'))
f2 = lambda x: f'{x:.2f}'.replace('-', '$-$')


def ece(y, p, bins=20):
    q = pd.qcut(p, bins, labels=False, duplicates='drop')
    d = pd.DataFrame({'y': y, 'p': p, 'q': q}).groupby('q').agg(y=('y', 'mean'), p=('p', 'mean'), n=('y', 'size'))
    return float((d['n'] * (d['y'] - d['p']).abs()).sum() / d['n'].sum())


def main():
    T.mkdir(parents=True, exist_ok=True)
    v = pd.read_parquet(C.WORK / 'values_full_catboost_s42.parquet')
    q = pd.read_csv(R / 'ratings.csv').set_index('player_id')
    shots = pd.read_parquet(C.WORK / 'keeper_shots_full_catboost_s42.parquet')
    appearances = pd.read_parquet(C.WORK / 'appearances.parquet')
    matches = pd.read_parquet(C.WORK / 'matches.parquet')

    # 1. What CPI changes relative to VAEP --------------------------------------------------------------------------
    g = v[v['goal'] & (v['type_name'] != 'shot_penalty')]
    rows = []
    for label, m in (("Level, 85$'$+", (g['goal_diff'] == 0) & (g['minute'] >= 85)), ("One down, 85$'$+", (g['goal_diff'] == -1) & (g['minute'] >= 85)),
                     ("Level, first 15$'$", (g['goal_diff'] == 0) & (g['minute'] < 15)), ("One up, 85$'$+", (g['goal_diff'] == 1) & (g['minute'] >= 85)),
                     ("Three or more up, 60$'$+", (g['goal_diff'] >= 3) & (g['minute'] >= 60))):
        rows.append(f"{label} & {int(m.sum())} & {nz(g.loc[m, 'value'].mean())} & {nz(g.loc[m, 'V'].mean())} \\\\")
    (T / 'goal_values.tex').write_text('\n'.join(rows))
    X['goalLevelLateV'] = nz(g.loc[(g['goal_diff'] == 0) & (g['minute'] >= 85), 'V'].mean())
    X['goalUpLateV'] = nz(g.loc[(g['goal_diff'] == 1) & (g['minute'] >= 85), 'V'].mean())
    X['goalVaepLow'], X['goalVaepHigh'] = nz(g.groupby(pd.cut(g['minute'], [0, 15, 85, 200]))['value'].mean().min()), nz(g['value'].mean())
    st = v['L_plus'] + v['L_minus']
    band = np.select([st >= st.quantile(0.8), st <= st.quantile(0.2)], ['top', 'bottom'], 'mid')
    share = pd.DataFrame({'vaep': v['value'].abs().groupby(band).sum(), 'cpi': v['V'].abs().groupby(band).sum()})
    share = share / share.sum()
    X['shareLowVaep'], X['shareLowCpi'] = str(round(share.loc['bottom', 'vaep'] * 100)), str(round(share.loc['bottom', 'cpi'] * 100))
    X['shareTopVaep'], X['shareTopCpi'] = str(round(share.loc['top', 'vaep'] * 100)), str(round(share.loc['top', 'cpi'] * 100))
    shared, moves = [], []
    for p in ('forward', 'midfielder', 'defender'):
        x = q[q['position'] == p]
        shared.append(len(set(x['V_per90'].nlargest(10).index) & set(x['value_per90'].nlargest(10).index)))
        moves.append((x['rank'] - x['rank_vaep']).abs().median())
    X['sharedTopLow'], X['sharedTopHigh'] = str(min(shared)), str(max(shared))
    X['moveLow'], X['moveHigh'] = str(int(min(moves))), str(int(max(moves)))
    q['move'] = q['rank_vaep'] - q['rank']
    rows = []
    for p, lab in (('forward', 'Fwd'), ('midfielder', 'Mid'), ('defender', 'Def')):
        x = q[(q['position'] == p) & (q['minutes'] >= 1500)]
        for _, r in pd.concat([x.nsmallest(1, 'move'), x.nlargest(1, 'move')]).iterrows():
            rows.append(f"{r['name']} & {lab} & {r['team']} & {int(r['rank_vaep'])} & {int(r['rank'])} \\\\")
    (T / 'movers.tex').write_text('\n'.join(rows))
    # does the part beyond VAEP repeat?
    pm = ratings.player_match_table(v, shots, appearances)
    h = pm.merge(matches[['game_id', 'gameweek']], on='game_id')
    h['half'] = np.where(h['gameweek'] % 2 == 1, 'o', 'e')
    w = h.groupby(['player_id', 'half'])[['V', 'value', 'minutes']].sum().unstack('half')
    w = w[(w[('minutes', 'o')] >= 450) & (w[('minutes', 'e')] >= 450)]
    beyond = []
    for p in ('forward', 'midfielder', 'defender'):
        x = w[q['position'].reindex(w.index) == p]
        res = {}
        for k in 'oe':
            a, b = x[('V', k)] / x[('minutes', k)] * 90, x[('value', k)] / x[('minutes', k)] * 90
            res[k] = a - np.polyval(np.polyfit(b, a, 1), b)
        beyond.append(spearmanr(res['o'], res['e']).statistic)
    X['beyondLow'], X['beyondHigh'] = nz(min(beyond)), nz(max(beyond))

    # 2. Case study: Messi and Ronaldo ---------------------------------------------------------------------------------
    def parts(pid, minutes):
        d = v[v['player_id'] == pid]
        kind = np.select([d['goal'], d['type_name'].str.contains('shot'), d['type_name'].isin(['pass', 'cross']) | d['assist'],
                          d['type_name'].isin(['dribble', 'take_on'])], ['Goals', 'Missed shots', 'Passes and crosses', 'Dribbles and take-ons'], 'Other actions')
        return d['V'].groupby(kind).sum() / minutes * 90, len(d) / minutes * 90
    me, cr = q[q['name'] == 'L. Messi'].iloc[0], q[q['name'] == 'Cristiano Ronaldo'].iloc[0]
    pm_, am = parts(me.name, me['minutes'])
    pr_, ar = parts(cr.name, cr['minutes'])
    rows = [f"{k} & {f2(pm_.get(k, 0))} & {f2(pr_.get(k, 0))} & {f2(pm_.get(k, 0) - pr_.get(k, 0))} \\\\" for k in ('Goals', 'Passes and crosses', 'Missed shots', 'Dribbles and take-ons', 'Other actions')]
    rows.append(f"\\midrule Total CPI & {f2(me['CPI'])} & {f2(cr['CPI'])} & {f2(me['CPI'] - cr['CPI'])} \\\\")
    (T / 'case.tex').write_text('\n'.join(rows))
    for key, src in (('Goals', 'Goals'), ('Pass', 'Passes and crosses'), ('Miss', 'Missed shots')):
        X[f'case{key}M'], X[f'case{key}R'] = f2(abs(pm_.get(src, 0))), f2(abs(pr_.get(src, 0)))
    X['ronaldoRank'], X['ronaldoCPI'], X['ronaldoShrunkRank'] = str(int(cr['rank'])), nz(cr['CPI']), str(int(cr['rank_shrunk']))
    X['nForwards'] = str(int((q['position'] == 'forward').sum()))
    X['messiActions'], X['ronaldoActions'] = f'{am:.0f}', f'{ar:.0f}'
    draws = ratings.bootstrap(pm, np.array([me.name, cr.name]), metrics=('V',), n_boot=2000)['V']
    diff = draws[:, 0] - draws[:, 1]
    X['caseAhead'] = f'{np.nanmean(diff > 0) * 100:.0f}'
    X['caseLow'], X['caseHigh'] = nz(np.nanpercentile(diff, 2.5)), nz(np.nanpercentile(diff, 97.5))

    # 3. Worked examples -------------------------------------------------------------------------------------------------
    best = pd.read_csv(R / 'moments_best.csv')
    label = matches.set_index('game_id')['label']
    def example(row_mask, tag):
        r = v[row_mask].iloc[0]
        i = v.index.get_loc(v[row_mask].index[0])
        prev = v.iloc[i - 1]
        same = prev['team_id'] == r['team_id']
        X[f'{tag}Match'] = label[r['game_id']].split(',')[0].replace(' - ', ' v ')
        X[f'{tag}Minute'] = f"{r['minute']:.0f}"
        X[f'{tag}Score'] = f"{int(r['goals_team'])}--{int(r['goals_opp'])}"
        X[f'{tag}Ps'], X[f'{tag}Pc'] = f"{r['p_scores']:.3f}", f"{r['p_concedes']:.3f}"
        X[f'{tag}PrevPs'] = f"{(prev['p_scores'] if same else prev['p_concedes']):.3f}"
        X[f'{tag}PrevPc'] = f"{(prev['p_concedes'] if same else prev['p_scores']):.3f}"
        X[f'{tag}dPs'], X[f'{tag}dPc'] = f"{r['dP_scores']:.3f}".replace('-', '$-$'), f"{r['dP_concedes']:.3f}".replace('-', '$-$')
        X[f'{tag}Lp'], X[f'{tag}Lm'] = f"{r['L_plus']:.3f}", f"{r['L_minus']:.3f}"
        X[f'{tag}V'] = f"{r['V']:.3f}".replace('-', '$-$')
        X[f'{tag}Vaep'] = f"{r['value']:.3f}".replace('-', '$-$')
        X[f'{tag}Type'] = r['type_name'].replace('_', ' ')
        X[f'{tag}Player'] = r['player_id']
    names = pd.read_json(C.RAW / 'players.json').drop_duplicates('wyId').set_index('wyId')['shortName']
    gm = [k for k, s in label.items() if s.startswith('Crystal Palace - Manchester United')][0]
    example((v['game_id'] == gm) & v['goal'] & (v['minute'] >= 89), 'exA')
    X['exAPlayer'] = names.get(X['exAPlayer'], '').encode('latin-1', 'ignore').decode('unicode_escape', 'ignore') if '\\u' in str(names.get(X['exAPlayer'], '')) else names.get(X['exAPlayer'], '')
    gl = [k for k, s in label.items() if s.startswith('Lille - Toulouse')][0]
    example((v['game_id'] == gl) & (v['type_name'] == 'clearance') & (v['minute'] >= 91) & (v['goal_diff'] == 1), 'exB')
    X['exBPlayer'] = names.get(X['exBPlayer'], '')

    # stakes worked example: one goal up, 85th minute, home team (held-out fold of England)
    panel = pd.read_parquet(C.WORK / 'panel.parquet')
    model = stakes.fit_rate_model(panel[panel['league'] != 'England'])
    n = C.T_FRAMES - int(85 * 60 / C.SECS_PER_FRAME)
    f_t = model.predict_proba(stakes.design([1], [n], [1], [0], ('base',)))[0, 1]
    f_o = model.predict_proba(stakes.design([-1], [n], [0], [0], ('base',)))[0, 1]
    th_t, th_o = 1 - (1 - f_t) ** (1 / n), 1 - (1 - f_o) ** (1 / n)
    ks = np.arange(C.MAX_GOALS + 1)
    a, b = binom.pmf(ks, n, th_t), binom.pmf(ks, n, th_o)
    out = {}
    for s_ in (0, 1, -1):
        diffg = 1 + s_ + np.subtract.outer(ks, ks)
        joint = np.outer(a, b)
        out[s_] = (joint[diffg > 0].sum(), joint[diffg == 0].sum())
    W = {s_: 3 * w + d for s_, (w, d) in out.items()}
    X.update({'wxN': str(n), 'wxFt': f'{f_t:.3f}', 'wxFo': f'{f_o:.3f}', 'wxTt': f'{th_t:.4f}', 'wxTo': f'{th_o:.4f}',
              'wxWin': f'{out[0][0]:.3f}', 'wxDraw': f'{out[0][1]:.3f}', 'wxW': f'{W[0]:.3f}', 'wxWup': f'{W[1]:.3f}', 'wxWdown': f'{W[-1]:.3f}',
              'wxLp': f'{W[1] - W[0]:.3f}', 'wxLm': f'{W[0] - W[-1]:.3f}', 'wxPgoal': f'{1 - (1 - th_t) ** n:.3f}'})
    coef = model.named_steps['lr'].coef_[0] / model.named_steps['scale'].scale_
    intercept = model.named_steps['lr'].intercept_[0] - (model.named_steps['scale'].mean_ * coef).sum()
    X.update({'lrBzero': f'{intercept:.3f}', 'lrBg': f'{coef[0]:.3f}', 'lrBn': f'{coef[1]:.4f}', 'lrBnn': f'{coef[2]:.6f}', 'lrBh': f'{coef[3]:.3f}'})

    # 4. Model details and calibration --------------------------------------------------------------------------------
    met = pd.read_csv(C.WORK / 'metrics_full_catboost_s42.csv')
    feats = pd.read_parquet(C.WORK / 'features_full_England.parquet').columns if (C.WORK / 'features_full_England.parquet').exists() else []
    X['nFeatures'] = str(len([c for c in feats if c not in ('y_scores', 'y_concedes', 'game_id')]) or 64)
    rows, eces = [], []
    for lg in C.LEAGUES:
        lab = pd.read_parquet(C.WORK / f'features_full_{lg}.parquet', columns=['y_scores', 'y_concedes']) if (C.WORK / f'features_full_{lg}.parquet').exists() else None
        sub = v[v['league'] == lg]
        e_s = ece(lab['y_scores'].values, sub['p_scores'].values) if lab is not None else np.nan
        e_c = ece(lab['y_concedes'].values, sub['p_concedes'].values) if lab is not None else np.nan
        m_s = met[(met['held_out'] == lg) & (met['target'] == 'scores')].iloc[0]
        m_c = met[(met['held_out'] == lg) & (met['target'] == 'concedes')].iloc[0]
        rows.append(f"{lg} & {int(m_s['n']):,} & {m_s['auc']:.3f} & {m_s['brier']:.4f} & {e_s * 1000:.2f} & {int(m_s['best_iter'])} & {m_c['auc']:.3f} & {m_c['brier']:.4f} & {e_c * 1000:.2f} & {int(m_c['best_iter'])} \\\\")
        eces += [e_s, e_c]
    (T / 'folds.tex').write_text('\n'.join(rows))
    X['eceMax'] = f'{np.nanmax(eces) * 1000:.2f}'
    ps = pd.read_csv(R / 'psxg_metrics.csv')
    X['psxgGoals'], X['psxgPred'], X['psxgShots'] = f"{int(ps['goals'].sum()):,}", f"{ps['predicted'].sum():,.0f}", f"{int(ps['shots'].sum()):,}"
    X['psxgBrier'] = f"{ps['brier'].mean():.3f}"
    spec = pd.read_csv(R / 'stakes_spec_comparison.csv')
    rows = [f"{r['spec'].replace('base+red+gxt', '+ red cards, score $\\times$ time').replace('base+red', '+ red cards').replace('base', 'Score, time, home')} & {r['log_loss']:.4f} & {r['auc']:.4f} & {r['rps']:.4f} & {r['mae_cost']:.3f} \\\\" for _, r in spec.iterrows()]
    (T / 'stake_specs.tex').write_text('\n'.join(rows))
    q1 = json.loads((R / 'q1_pressure_all.json').read_text())
    X['qPlayersMatches'] = f"{v[v['player_id'] > 0].groupby(['player_id', 'game_id']).ngroups:,}"
    q2 = json.loads((R / 'q2_summary.json').read_text())
    X['qOffUp'], X['qOffDown'] = f"{q2['protecting_off']:,}", f"{q2['chasing_off']:,}"
    (R / 'extra_claims.json').write_text(json.dumps(X, indent=1, ensure_ascii=False, default=str))
    print(len(X), 'extra numbers;', json.dumps({k: X[k] for k in ('beyondLow', 'beyondHigh', 'sharedTopLow', 'sharedTopHigh', 'caseAhead', 'eceMax', 'wxLp', 'wxLm', 'exAMatch', 'exAV', 'exBMatch', 'exBV')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
