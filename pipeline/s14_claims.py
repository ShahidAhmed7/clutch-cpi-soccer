"""Stage 14. Turn results/ into paper/numbers.tex (one macro per number) and the generated tables, so no number in the paper is typed by hand.

    python pipeline/s14_claims.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cpi import config as C, state  # noqa: E402

R, P = C.RESULTS, C.ROOT / 'paper'
N = {}
f1, f2, f3 = (lambda x: f'{x:.1f}'), (lambda x: f'{x:.2f}'), (lambda x: f'{x:.3f}')
nz = lambda x: f2(x).replace('0.', '.', 1) if abs(x) < 1 and x >= 0 else f2(x).replace('-0.', '$-$.', 1) if abs(x) < 1 else f2(x)


def latex_accents(text):
    """Encode player-name accents unsupported by the paper's T1 font in XeTeX."""
    return text.replace('Š', r'\v{S}').replace('ć', r"\'{c}")


def main():
    (P / 'tables').mkdir(parents=True, exist_ok=True)
    val = json.loads((R / 'claims_validation.json').read_text())
    acts = pd.read_parquet(C.WORK / 'values_full_catboost_s42.parquet', columns=['game_id', 'team_id', 'goal', 'own_goal'])
    matches = pd.read_parquet(C.WORK / 'matches.parquet')
    check = state.final_score_check(acts, matches)
    N['scoreMatch'] = f"{(check['official'] == check['made']).mean() * 100:.2f}"
    N['nMatches'] = f'{len(matches):,}'
    N['nActions'] = f'{len(acts) / 1e6:.2f} million'
    N['nActionsExact'] = f'{len(acts):,}'
    N['nGoals'] = f"{int(acts['goal'].sum()):,}"
    N['nOwnGoals'] = str(int(acts['own_goal'].sum()))
    m = pd.read_csv(C.WORK / 'metrics_full_catboost_s42.csv').groupby('target')[['auc', 'brier']].mean()
    N['aucScores'], N['aucConcedes'] = nz(m.loc['scores', 'auc']), nz(m.loc['concedes', 'auc'])
    N['psxgAuc'] = nz(pd.read_csv(R / 'psxg_metrics.csv')['auc'].mean())

    # stakes
    spec = pd.read_csv(R / 'stakes_spec_comparison.csv').set_index('spec')
    N['stakesRps'], N['stakesBias'] = f3(spec.loc['base', 'rps']), f3(abs(spec.loc['base', 'bias_points']))
    t1 = val['t1']['leading_by_1']
    N['realCost'], N['realGain'], N['modelCost'], N['modelGain'] = f2(t1['real_cost']), f2(t1['real_gain']), f2(t1['model_cost']), f2(t1['model_gain'])
    N['costGainDiff'], N['costGainLow'], N['costGainHigh'] = f2(t1['cost_minus_gain']), f2(t1['ci_low']), f2(t1['ci_high'])
    lv = val['t1']['level']
    N['levelGain'], N['levelCost'], N['levelModelGain'], N['levelModelCost'] = f2(lv['real_gain']), f2(lv['real_cost']), f2(lv['model_gain']), f2(lv['model_cost'])
    tr = val['t1']['trailing_by_1']
    N['downGain'], N['downCost'], N['downModelGain'], N['downModelCost'] = f2(tr['real_gain']), f2(tr['real_cost']), f2(tr['model_gain']), f2(tr['model_cost'])
    t2 = val['t2']
    N['nCells'] = str(t2['cells'])
    a = t2['L_minus_vs_real_cost']
    N['rhoCost'], N['rhoCostLow'], N['rhoCostHigh'], N['maeCost'], N['slopeCost'] = nz(a['spearman']), nz(a['ci_low']), nz(a['ci_high']), nz(a['mae']), nz(a['slope'])
    b = t2['L_plus_vs_real_gain']
    N['rhoGain'], N['maeGain'] = nz(b['spearman']), nz(b['mae'])
    c = t2['total_vs_real_total']; d = t2['scoring_only_vs_real_total']
    N['rhoTotal'], N['rhoTotalLow'], N['rhoTotalHigh'] = nz(c['spearman']), nz(c['ci_low']), nz(c['ci_high'])
    N['rhoTotalSO'], N['rhoTotalSOLow'], N['rhoTotalSOHigh'] = nz(d['spearman']), nz(d['ci_low']), nz(d['ci_high'])
    N['rhoCostSO'] = nz(t2['scoring_only_vs_real_cost']['spearman'])
    N['rhoGainSO'] = nz(t2['scoring_only_vs_real_gain']['spearman'])
    cells = pd.read_csv(R / 't2_cells.csv').set_index(['bin', 'g'])
    pairs = [(r.real_cost, cells.loc[(b_, g - 1), 'scoring_only'], r.model_cost) for (b_, g), r in cells.iterrows() if (b_, g - 1) in cells.index]
    pairs = np.array(pairs)
    N['rhoShiftSO'], N['rhoShiftL'], N['nShiftCells'] = nz(spearmanr(pairs[:, 0], pairs[:, 1]).statistic), nz(spearmanr(pairs[:, 0], pairs[:, 2]).statistic), str(len(pairs))
    panel = pd.read_parquet(C.WORK / 'panel.parquet', columns=['frame', 'home', 'p_win', 'p_draw', 'W', 'pts'])
    k = panel[panel['frame'] == 0].groupby('home')[['p_win', 'W', 'pts']].mean()
    N['kickHomeW'], N['kickHomeReal'], N['kickAwayW'], N['kickAwayReal'] = f2(k.loc[1, 'W']), f2(k.loc[1, 'pts']), f2(k.loc[0, 'W']), f2(k.loc[0, 'pts'])

    t3 = val['t3']
    N['ledgerMatchSlope'], N['ledgerMatchR'] = f2(t3['match']['net_V']['slope']), nz(t3['match']['net_V']['pearson'])
    N['ledgerSeasonSlope'], N['ledgerSeasonR'] = f2(t3['season']['net_V']['slope']), f3(t3['season']['net_V']['pearson']).lstrip('0')
    N['ledgerSeasonRGd'], N['ledgerSeasonRVaep'] = f3(t3['season']['gd']['pearson']).lstrip('0'), f3(t3['season']['net_value']['pearson']).lstrip('0')
    N['ledgerMatchRGd'], N['ledgerMatchRVaep'] = nz(t3['match']['gd']['pearson']), nz(t3['match']['net_value']['pearson'])
    N['ledgerTeams'] = str(t3['team_seasons'])
    credit = pd.read_csv(R / 't4_credit.csv')
    N['creditLow'], N['creditHigh'] = f1(credit['ratio'].min()), f1(credit['ratio'].max())
    w = val['t4']['weight_late_lead']
    N['weightTwo'], N['weightSO'], N['weightRatio'] = f2(w['two_sided']), f2(w['scoring_only']), f1(w['two_sided'] / w['scoring_only'])

    # ratings
    q = pd.read_csv(R / 'ratings.csv')
    N['nQualified'] = f'{len(q):,}'
    pos = q.groupby('position')['CPI'].mean()
    for a_, b_ in (('forward', 'F'), ('midfielder', 'M'), ('defender', 'D'), ('goalkeeper', 'G')):
        N[f'meanCPI{b_}'] = nz(pos[a_])
    named = lambda s: q[q['name'] == s].iloc[0]
    me = named('L. Messi')
    N['messiCPI'], N['messiLow'], N['messiHigh'], N['messiTotal'] = nz(me['CPI']), nz(me['ci_low']), nz(me['ci_high']), f1(me['V'])
    rel = pd.read_csv(R / 'reliability.csv').set_index('position')
    dis = pd.read_csv(R / 'discrimination.csv').set_index('position')
    dec = pd.read_csv(R / 'decomposition.csv').set_index('position')
    for a_, b_ in (('forward', 'F'), ('midfielder', 'M'), ('defender', 'D'), ('goalkeeper', 'G')):
        N[f'relCPI{b_}'], N[f'relVaep{b_}'] = nz(rel.loc[a_, 'V']), nz(rel.loc[a_, 'value'])
        N[f'disCPI{b_}'] = nz(dis.loc[a_, 'D_CPI'])
    N['relSaved'], N['disSaved'] = nz(rel.loc['goalkeeper', 'points_saved']), nz(dis.loc['goalkeeper', 'D_points_saved'])
    of = dec.drop('goalkeeper')
    N['abilityLow'], N['abilityHigh'] = str(round(of['ability'].min() * 100)), str(round(of['ability'].max() * 100))
    N['deliveryLow'], N['deliveryHigh'] = str(round(of['delivery'].min() * 100)), str(round(of['delivery'].max() * 100))
    N['exposureMax'] = str(round(of['exposure'].abs().max() * 100))
    N['deliveryRhoLow'], N['deliveryRhoHigh'] = nz(of['delivery_rho'].min()), nz(of['delivery_rho'].max())

    rows = []
    for p_, lab in (('forward', 'Forwards'), ('midfielder', 'Midfielders'), ('defender', 'Defenders'), ('goalkeeper', 'Goalkeepers')):
        t = q[q['position'] == p_].nsmallest(3, 'rank')
        for i, r in enumerate(t.itertuples()):
            rows.append(f"{lab if i == 0 else ''} & {r.name} & {nz(r.CPI)} ({nz(r.ci_low)}, {nz(r.ci_high)}) & {f1(r.V)} & {nz(r.CPI_shrunk)} & {r.P_top10 * 100:.0f}\\% & {int(r.rank_vaep)} \\\\")
        rows.append('\\midrule')
    (P / 'tables' / 'leaders.tex').write_text('\n'.join(rows[:-1]))

    summary = json.loads((R / 'ratings_summary.json').read_text())['rank_correlation']
    for a_, b_ in (('forward', 'F'), ('midfielder', 'M'), ('defender', 'D'), ('goalkeeper', 'G')):
        N[f'corrVaep{b_}'], N[f'corrGa{b_}'] = nz(summary[a_]['vaep']), nz(summary[a_]['ga'])
        N[f'relSO{b_}'] = nz(rel.loc[a_, 'V_scoring_only'])
        if a_ != 'goalkeeper':
            N[f'relGoals{b_}'] = nz(rel.loc[a_, 'goals'])
    N['relDefendD'], N['disDefendD'] = nz(rel.loc['defender', 'defend']), nz(dis.loc['defender', 'D_defend'])

    # lead protectors and risers: selection rules are stated in the paper
    lp = pd.read_csv(R / 'lead_protectors.csv')
    lpd = lp[lp['position'] == 'defender'].nlargest(3, 'lead_V')
    N['leadDefenders'] = ', '.join(f"{r['name']} ({r['lead_V']:.2f})" for _, r in lpd.iterrows())
    q2 = q.assign(rank_so=q.groupby('position')['V_scoring_only_per90'].rank(ascending=False))
    q2['rise'] = q2['rank_so'] - q2['rank']
    ri = q2[(q2['position'] == 'defender') & (q2['rank'] <= 50) & (q2['minutes'] >= 1500)].nlargest(3, 'rise')
    N['riseDefenders'] = ', '.join(f"{r['name']} ({int(r['rank_so'])} to {int(r['rank'])})" for _, r in ri.iterrows())

    # stories
    g = pd.read_csv(R / 'goals_vs_points.csv')
    pick = lambda s: g[g['name'] == s].iloc[0]
    for key, s in (('Moreno', 'Gerard Moreno'), ('Ronaldo', 'Cristiano Ronaldo'), ('Aubameyang', 'P. Aubameyang'), ('Zapata', 'D. Zapata'), ('Aguero', 'S. Agüero'), ('Salah', 'Mohamed Salah')):
        r = pick(s)
        N[f'goals{key}'], N[f'gpts{key}'], N[f'ppg{key}'] = str(int(r['np_goals'])), f1(r['goal_points']), nz(r['points_per_goal'])
        N[f'decided{key}'] = str(round(r['decided'] * 100))
    best, worst = pd.read_csv(R / 'moments_best.csv'), pd.read_csv(R / 'moments_worst.csv')
    moment = lambda r: latex_accents(f"{r['name']} & {r['match'].split(',')[0].replace(' - ', ' v ')} {r['match'].split(', ')[-1].replace(' - ', '--')} & {r['minute']:.0f}$'$ & {r['score'].replace('-', '--')} & {'+' if r['V'] > 0 else '$-$'}{abs(r['V']):.2f} \\\\")
    (P / 'tables' / 'moments.tex').write_text('\\multicolumn{5}{@{}l}{\\textit{Most valuable}} \\\\\n' + '\n'.join(moment(r) for _, r in best.head(4).iterrows()) + '\n\\midrule\n\\multicolumn{5}{@{}l}{\\textit{Most costly}} \\\\\n' + '\n'.join(moment(r) for _, r in worst.head(4).iterrows()))
    N['costlyAllOwn'] = str(int((worst['result_name'] == 'owngoal').cumprod().sum()))
    sk = worst[worst['name'].str.contains('Škriniar')].iloc[0]
    N['skriniarCost'], N['skriniarMinute'] = f2(abs(sk['V'])), f"{sk['minute']:.0f}"
    k = pd.read_csv(R / 'keepers.csv')
    N['medianShots'] = str(int(k.loc[k['minutes'] >= 2700, 'shots_faced'].median()))
    kk = lambda s: k[k['name'] == s].iloc[0]
    for key, s in (('Gea', 'David de Gea'), ('Oblak', 'J. Oblak'), ('Alisson', 'Alisson'), ('Pope', 'N. Pope'), ('Cech', 'P. Čech')):
        r = kk(s)
        N[f'saved{key}'], N[f'prevented{key}'] = f1(r['points_saved']) if r['points_saved'] > 0 else '$-$' + f1(abs(r['points_saved'])), f1(r['goals_prevented']) if r['goals_prevented'] > 0 else '$-$' + f1(abs(r['goals_prevented']))
    de = pd.read_csv(R / 'deciders.csv')
    dd = lambda t: de[de['team'] == t].iloc[0]
    for key, t in (('Barca', 'Barcelona'), ('Juve', 'Juventus'), ('Lille', 'Lille')):
        r = dd(t)
        N[f'margin{key}'], N[f'paa{key}'] = str(int(r['margin'])), f1(r['points_above_average'])

    # Q1
    pr = pd.read_csv(R / 'q1_pressure_primary.csv').set_index('test')
    allq = json.loads((R / 'q1_pressure_all.json').read_text())
    ci = lambda e, s, sc=1, dp=2: f"{e * sc:+.{dp}f} ({(e - 1.96 * s) * sc:+.{dp}f}, {(e + 1.96 * s) * sc:+.{dp}f})".replace('-', '$-$')
    N['cutHigh'], N['cutLow'] = f2(allq['cuts']['high']), f2(allq['cuts']['low'])
    N['baseCompletion'] = f"{allq['base_completion'] * 100:.1f}"
    N['nPasses'] = f"{allq['n_passes'] / 1e6:.2f} million"
    a_, b_, c_ = pr.loc['a_pass_completion'], pr.loc['b_action_value'], pr.loc['c_pass_progress_m']
    N['qCompletion'], N['qProgress'] = f"{abs(a_['est']) * 100:.1f}", f2(c_['est'])
    v = pd.read_parquet(C.WORK / 'values_full_catboost_s42.parquet', columns=['player_id', 'value'])
    mean_value = v.loc[v['player_id'] > 0, 'value'].mean()
    N['qValuePct'] = str(round(abs(b_['est']) / mean_value * 100))
    N['meanValue'] = f"{mean_value * 1000:.1f}"
    ex = allq['fits']['exploratory']
    N['xSideCompletion'], N['xSideProgress'] = f"{abs(ex['completion_by_side']['side_minus']['est']) * 100:.1f}", f2(ex['progress_by_side']['side_minus']['est'])
    N['xRoute'] = f"{abs(ex['completion_same_route']['high']['est']) * 100:.1f}"
    rows = [f"Pass completion, pp & {ci(a_['est'], a_['se'], 100)} & {a_['p_holm']:.3f} \\\\", f"Action value, $\\times 10^{{-3}}$ & {ci(b_['est'], b_['se'], 1000)} & {b_['p_holm']:.3f} \\\\", f"Pass forward progress, m & {ci(c_['est'], c_['se'])} & {c_['p_holm']:.3f} \\\\"]
    d_, e_ = pr.loc['d_home_moderates_completion'], pr.loc['e_age_moderates_completion']
    rows += [f"Completion: home $\\times$ stakes & {ci(d_['est'], d_['se'], 100)} & {d_['p_holm']:.2f} \\\\", f"Completion: age (per 5 y) $\\times$ stakes & {ci(e_['est'], e_['se'], 100)} & {e_['p_holm']:.2f} \\\\", '\\midrule', '\\multicolumn{3}{@{}l}{\\textit{Exploratory: which side of the stakes is larger}} \\\\']
    for key, lab, sc in (('completion_by_side', 'Completion, pp', 100), ('progress_by_side', 'Progress, m', 1), ('value_by_side', 'Value, $\\times 10^{-3}$', 1000)):
        h, s_ = ex[key]['high'], ex[key]['side_minus']
        rows.append(f"{lab}: scoring side & {ci(h['est'], h['se'], sc)} & -- \\\\")
        rows.append(f"{lab}: conceding side & {ci(s_['est'], s_['se'], sc)} & -- \\\\")
    (P / 'tables' / 'pressure.tex').write_text('\n'.join(rows))

    # Q2
    co = pd.read_csv(R / 'q2_coaches.csv')
    s = json.loads((R / 'q2_summary.json').read_text())
    N['nTakenOff'], N['nBroughtOn'] = f"{s['taken_off_decisions']:,}", f"{s['brought_on_decisions']:,}"
    get = lambda d, st, t: co[(co['decision'] == d) & (co['state'] == st) & (co['term'] == t)].iloc[0]
    for key, (d_, st, t) in {'OffUpDef': ('taken_off', 'protecting', 'z_defend'), 'OffUpAtt': ('taken_off', 'protecting', 'z_attack'), 'OffDownDef': ('taken_off', 'chasing', 'z_defend'), 'OffDownAtt': ('taken_off', 'chasing', 'z_attack'),
                             'OnUpDef': ('brought_on', 'protecting', 'z_defend'), 'OnUpAtt': ('brought_on', 'protecting', 'z_attack'), 'OnDownDef': ('brought_on', 'chasing', 'z_defend'), 'OnDownAtt': ('brought_on', 'chasing', 'z_attack')}.items():
        r = get(d_, st, t)
        N[f'or{key}'], N[f'or{key}Low'], N[f'or{key}High'] = nz(np.exp(r['log_odds'])) if np.exp(r['log_odds']) < 1 else f2(np.exp(r['log_odds'])), f2(np.exp(r['ci_low'])), f2(np.exp(r['ci_high']))
    rp = pd.read_csv(R / 'q2_replacements.csv').set_index('case')
    u, dn = rp.loc['protecting, goal prevention'], rp.loc['chasing, attacking']
    N['pickUp'], N['pickUpChance'], N['pickDown'], N['pickDownChance'] = f1(u['picked_best'] * 100), f1(u['chance'] * 100), f1(dn['picked_best'] * 100), f1(dn['chance'] * 100)
    N['pickUpN'], N['pickDownN'] = str(int(u['decisions'])), str(int(dn['decisions']))

    # sensitivity and data choices, once available
    if (R / 'sensitivity.csv').exists():
        sens = pd.read_csv(R / 'sensitivity.csv')
        rows = [f"{r.comparison} & {nz(r.rho_forward)} & {nz(r.rho_midfielder)} & {nz(r.rho_defender)} & {nz(r.rho_goalkeeper)} & {r.top10_forward}, {r.top10_midfielder}, {r.top10_defender}, {r.top10_goalkeeper} \\\\" for r in sens.itertuples()]
        (P / 'tables' / 'sensitivity.tex').write_text('\n'.join(rows))
    dc = pd.read_csv(R / 'data_choices.csv').set_index('choice')
    names = {'mirrored_coordinates': 'Away actions left mirrored', 'no_set_piece_baselines': 'No penalty, corner baselines'}
    rows = []
    for k_, lab in names.items():
        r = dc.loc[k_]
        rows.append(f"{lab} & {nz(r.rho_forward)} & {nz(r.rho_midfielder)} & {nz(r.rho_defender)} & {nz(r.rho_goalkeeper)} & {int(r.top10_forward)}, {int(r.top10_midfielder)}, {int(r.top10_defender)}, {int(r.top10_goalkeeper)} \\\\")
    (P / 'tables' / 'choices.tex').write_text('\n'.join(rows))
    N['mirrorAuc'], N['baseAuc'] = nz(dc.loc['mirrored_coordinates', 'auc_scores']), nz(dc.loc['method', 'auc_scores'])
    N['mirrorKeeper'], N['baseKeeper'] = nz(dc.loc['mirrored_coordinates', 'mean_goalkeeper']), nz(dc.loc['method', 'mean_goalkeeper'])
    N['baseForward'], N['noBaseForward'] = nz(dc.loc['method', 'mean_forward']), nz(dc.loc['no_set_piece_baselines', 'mean_forward'])
    N['noBaseTopMid'] = str(int(dc.loc['no_set_piece_baselines', 'top10_midfielder']))
    sens = pd.read_csv(R / 'sensitivity.csv')
    rho_cols = [f'rho_{p_}' for p_ in C.POSITIONS]
    top_cols = [f'top10_{p_}' for p_ in C.POSITIONS]
    N['sensRhoMin'], N['sensTopMin'] = nz(sens[rho_cols].min().min()), str(int(sens[top_cols].min().min()))
    N['strengthRhoMin'] = nz(sens[sens['comparison'] == 'team strength in the stakes'][rho_cols].min(axis=1).iloc[0])
    th = pd.read_csv(R / 'minutes_threshold.csv').set_index('minutes')
    N['thresholdLow'], N['thresholdHigh'] = str(int(th.loc[1800, C.POSITIONS].min())), str(int(th.loc[1800, C.POSITIONS].max()))

    extra = R / 'extra_claims.json'
    if extra.exists():
        N.update(json.loads(extra.read_text()))
    tables = {'tabLeaders': 'leaders', 'tabMoments': 'moments', 'tabPressure': 'pressure', 'tabSensitivity': 'sensitivity', 'tabChoices': 'choices',
              'tabGoalValues': 'goal_values', 'tabMovers': 'movers', 'tabCase': 'case', 'tabFolds': 'folds', 'tabStakeSpecs': 'stake_specs'}
    macros = [f'\\newcommand{{\\{k_}}}{{{v_}}}' for k_, v_ in N.items()]
    for mac, name in tables.items():
        path = P / 'tables' / f'{name}.tex'
        body = path.read_text() if path.exists() and path.read_text().strip() else '\\multicolumn{6}{@{}l}{pending} \\\\'
        macros.append(f'\\newcommand{{\\{mac}}}{{%\n{body}\n}}')
    (P / 'numbers.tex').write_text(latex_accents('\n'.join(macros) + '\n'))
    readme = C.ROOT / 'README.md'
    if readme.exists():
        text = readme.read_text(encoding='utf-8')
        block = (f"| Real results, one goal up after 80': cost of conceding, gain from scoring | {N['realCost']}, {N['realGain']} league points |\n"
                 f"| Model stakes in the same moments | {N['modelCost']}, {N['modelGain']} |\n"
                 f"| Season ledger: net clutch value against real points | slope {N['ledgerSeasonSlope']}, r {N['ledgerSeasonR']} |\n"
                 f"| Same player, high vs low stakes: pass completion, forward progress | -{N['qCompletion']} points, +{N['qProgress']} m |\n"
                 f"| Coaches, one goal up: odds of being taken off per SD of goal-prevention rating | {N['orOffUpDef']} ({N['orOffUpDefLow']} to {N['orOffUpDefHigh']}) |\n"
                 f"| Coaches, one goal down: same, per SD of attacking rating | {N['orOffDownAtt']} ({N['orOffDownAttLow']} to {N['orOffDownAttHigh']}) |\n"
                 f"| Split-half reliability of CPI: forwards, midfielders, defenders, keepers | {N['relCPIF']}, {N['relCPIM']}, {N['relCPID']}, {N['relCPIG']} |\n")
        a_, b_ = '<!-- results:start -->', '<!-- results:end -->'
        if a_ in text:
            readme.write_text(text[:text.index(a_) + len(a_)] + '\n| Result | Value |\n|---|---|\n' + block + text[text.index(b_):], encoding='utf-8')
    (R / 'claims.json').write_text(json.dumps(N, indent=1, ensure_ascii=False))
    print(len(N), 'numbers written')


if __name__ == '__main__':
    main()
