"""Stage 9. Do coaches agree with CPI?

    python pipeline/s9_coaches.py [--tag full_catboost_s42]

Every substitution from minute 60 on with the team one goal up (protecting) or one goal down (chasing). Who is taken off, among the
outfield players on the pitch? Who comes on, among the outfield players on the bench? Each player's attacking and goal-prevention
ratings come only from his other matches, z-scored within position, and need 450 minutes outside the match being decided.

Conditional logit within each decision, with position, minutes on the pitch and a yellow-card flag as controls for who comes off.
Inference: match bootstrap, 500 draws.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cpi import config as C, raw, ratings  # noqa: E402

MIN_OTHER_MINUTES = 450
FIRST_MINUTE = 60
BOOT = 500


def conditional_logit(X, chosen, groups, weights=None, iters=40):
    """Newton solver for a logit with one choice per group. `weights` holds one bootstrap weight per row."""
    k = X.shape[1]
    beta = np.zeros(k)
    weights = np.ones(len(X)) if weights is None else weights
    n_groups = groups.max() + 1
    for _ in range(iters):
        u = X @ beta
        umax = np.full(n_groups, -np.inf)
        np.maximum.at(umax, groups, u)
        exp_u = np.exp(u - umax[groups])
        denom = np.bincount(groups, weights=exp_u, minlength=n_groups)
        p = exp_u / denom[groups]
        mean_x = np.column_stack([np.bincount(groups, weights=p * X[:, j], minlength=n_groups) for j in range(k)])
        diff = X - mean_x[groups]
        grad = ((weights * chosen)[:, None] * diff).sum(axis=0)
        H = (diff * (weights * p)[:, None]).T @ diff          # all rows of a decision share its match's weight
        step = np.linalg.solve(H + 1e-8 * np.eye(k), grad)
        beta += step
        if np.abs(step).max() < 1e-7:
            break
    return beta


def bootstrap_logit(d, cols, draws=BOOT):
    """Point estimate and match-bootstrap interval. Decisions are resampled through their match."""
    groups = pd.factorize(d['decision'])[0]
    X, chosen = d[cols].values.astype(float), d['chosen'].values.astype(float)
    point = conditional_logit(X, chosen, groups)
    games = d['game_id'].unique()
    gi = pd.Series(np.arange(len(games)), index=games).loc[d['game_id']].values
    rng = np.random.default_rng(C.SEED)
    out = np.empty((draws, len(cols)))
    for b in range(draws):
        w = np.bincount(rng.integers(0, len(games), len(games)), minlength=len(games)).astype(float)[gi]
        keep = w > 0
        out[b] = conditional_logit(X[keep], chosen[keep], pd.factorize(groups[keep])[0], w[keep])
    return point, out


def decisions(matches_raw, reds, goal_clock, players):
    """One row per candidate per substitution, with the state of the match at that minute."""
    position = players.set_index('player_id')['position']
    rows_off, rows_on = [], []
    decision = 0
    for m in matches_raw:
        gid = int(m['wyId'])
        for tid, info in m['teamsData'].items():
            tid = int(tid)
            form = info.get('formation') or {}
            subs = [s for s in (form.get('substitutions') or []) if isinstance(s, dict)]
            lineup = form.get('lineup') or []
            bench = form.get('bench') or []
            yellow = {int(p['playerId']): int(p.get('yellowCards') or 0) for p in lineup + bench}
            on = {int(p['playerId']): 0.0 for p in lineup}
            unused = {int(p['playerId']) for p in bench}
            subs = sorted(subs, key=lambda s: int(s['minute']))
            red_off = {r.player_id: (r.period_id - 1) * 45 + r.time_seconds / 60 for r in reds[(reds['game_id'] == gid) & (reds['team_id'] == tid)].itertuples()}
            for s in subs:
                minute, pin, pout = int(s['minute']), int(s['playerIn']), int(s['playerOut'])
                for pid, when in red_off.items():
                    if when <= minute:
                        on.pop(pid, None)
                if minute >= FIRST_MINUTE:
                    g = goal_clock(gid, tid, minute)
                    if g in (1, -1) and pout in on:
                        decision += 1
                        for pid, entered in on.items():
                            if position.get(pid) in ('forward', 'midfielder', 'defender'):
                                rows_off.append({'decision': decision, 'game_id': gid, 'team_id': tid, 'player_id': pid, 'minute': minute, 'g': g, 'chosen': float(pid == pout),
                                                 'on_pitch': minute - entered, 'yellow': float(yellow.get(pid, 0) > 0)})
                        for pid in unused:
                            if position.get(pid) in ('forward', 'midfielder', 'defender'):
                                rows_on.append({'decision': decision, 'game_id': gid, 'team_id': tid, 'player_id': pid, 'minute': minute, 'g': g, 'chosen': float(pid == pin)})
                on.pop(pout, None)
                on[pin] = float(minute)
                unused.discard(pin)
    return pd.DataFrame(rows_off), pd.DataFrame(rows_on)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', default='full_catboost_s42')
    args = ap.parse_args()
    values = pd.read_parquet(C.WORK / f'values_{args.tag}.parquet', columns=['game_id', 'team_id', 'player_id', 'type_name', 'goal', 'own_goal', 'attack', 'defend', 'V', 'value', 'V_scoring_only', 'assist', 'minute'])
    shots = pd.read_parquet(C.WORK / f'keeper_shots_{args.tag}.parquet')
    appearances = pd.read_parquet(C.WORK / 'appearances.parquet')
    reds = pd.read_parquet(C.WORK / 'reds.parquet')
    matches = pd.read_parquet(C.WORK / 'matches.parquet')
    players, teams = raw.load_players(), raw.load_teams()

    home = dict(zip(matches['game_id'], matches['home_team_id']))
    away = dict(zip(matches['game_id'], matches['away_team_id']))
    g = values[values['goal'] | values['own_goal']]
    scorer = np.where(g['goal'].values, g['team_id'].values,
                      np.where(g['team_id'].values == g['game_id'].map(home).values, g['game_id'].map(away).values, g['game_id'].map(home).values))
    clock = {}
    for gid, t, mn in zip(g['game_id'].values, scorer, g['minute'].values):
        clock.setdefault(gid, []).append((t, mn))

    def goal_clock(gid, tid, minute):
        return sum((1 if t == tid else -1) for t, mn in clock.get(gid, []) if mn <= minute)

    all_matches = [m for lg in C.LEAGUES for m in raw.load_matches(lg)]
    off, on = decisions(all_matches, reds, goal_clock, players)

    pm = ratings.player_match_table(values, shots, appearances)
    totals = pm.groupby('player_id')[['attack', 'defend', 'minutes', 'V', 'value']].sum()
    position = players.set_index('player_id')['position']

    def attach(d):
        one = pm.set_index(['player_id', 'game_id'])[['attack', 'defend', 'minutes', 'V', 'value']]
        d = d.join(one, on=['player_id', 'game_id'], rsuffix='_m').fillna({'attack': 0.0, 'defend': 0.0, 'minutes': 0.0, 'V': 0.0, 'value': 0.0})
        t = totals.reindex(d['player_id'].values)
        mins = t['minutes'].values - d['minutes'].values
        for k in ('attack', 'defend', 'V', 'value'):
            d[f'r_{k}'] = (t[k].values - d[k].values) / np.where(mins > 0, mins, np.nan) * 90
        d['other_minutes'] = mins
        d['position'] = d['player_id'].map(position)
        d = d[d['other_minutes'] >= MIN_OTHER_MINUTES].copy()
        for k in ('attack', 'defend', 'V', 'value'):
            d[f'z_{k}'] = d.groupby('position')[f'r_{k}'].transform(lambda s: (s - s.mean()) / s.std())
        for p in ('forward', 'midfielder'):
            d[f'pos_{p}'] = (d['position'] == p).astype(float)
        # keep decisions where the chosen player survived and there is a real choice
        ok = d.groupby('decision')['chosen'].transform('sum').eq(1) & d.groupby('decision')['chosen'].transform('size').ge(2)
        return d[ok].reset_index(drop=True)

    off, on = attach(off), attach(on)
    off['on_pitch'] = off['on_pitch'] / 30
    print(f"taken off: {off['decision'].nunique():,} decisions, protecting {off[off['g'] == 1]['decision'].nunique():,}, chasing {off[off['g'] == -1]['decision'].nunique():,}", flush=True)
    print(f"brought on: {on['decision'].nunique():,} decisions, protecting {on[on['g'] == 1]['decision'].nunique():,}, chasing {on[on['g'] == -1]['decision'].nunique():,}", flush=True)

    results = {}
    off_cols = ['z_defend', 'z_attack', 'pos_forward', 'pos_midfielder', 'on_pitch', 'yellow']
    on_cols = ['z_defend', 'z_attack', 'pos_forward', 'pos_midfielder']
    rows = []
    for table, cols, name in ((off, off_cols, 'taken_off'), (on, on_cols, 'brought_on')):
        boots = {}
        for g_val, label in ((1, 'protecting'), (-1, 'chasing')):
            point, draws = bootstrap_logit(table[table['g'] == g_val], cols)
            boots[label] = (point, draws)
            for i, c in enumerate(cols):
                rows.append({'decision': name, 'state': label, 'term': c, 'log_odds': point[i], 'odds_ratio': float(np.exp(point[i])),
                             'ci_low': np.percentile(draws[:, i], 2.5), 'ci_high': np.percentile(draws[:, i], 97.5), 'p': min(1.0, 2 * min((draws[:, i] <= 0).mean(), (draws[:, i] >= 0).mean()))})
        for c in ('z_defend', 'z_attack'):
            i = cols.index(c)
            d = boots['protecting'][1][:, i] - boots['chasing'][1][:, i]
            rows.append({'decision': name, 'state': 'protecting_minus_chasing', 'term': c, 'log_odds': boots['protecting'][0][i] - boots['chasing'][0][i],
                         'ci_low': np.percentile(d, 2.5), 'ci_high': np.percentile(d, 97.5), 'p': min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean()))})
    res = pd.DataFrame(rows)
    res.to_csv(C.RESULTS / 'q2_coaches.csv', index=False)
    print(res[res['term'].isin(['z_defend', 'z_attack'])].round(3).to_string(index=False), flush=True)

    # Trusted by the rating, taken off by the coach: chance of coming off when protecting, against the chance expected from position and time on the pitch
    prot = off[off['g'] == 1].copy()
    groups = pd.factorize(prot['decision'])[0]
    base_cols = ['pos_forward', 'pos_midfielder', 'on_pitch', 'yellow']
    beta = conditional_logit(prot[base_cols].values.astype(float), prot['chosen'].values.astype(float), groups)
    u = prot[base_cols].values.astype(float) @ beta
    ex = np.exp(u)
    prot['expected'] = ex / np.bincount(groups, weights=ex)[groups]
    table = prot.groupby('player_id').agg(decisions=('chosen', 'size'), taken_off=('chosen', 'sum'), expected=('expected', 'sum'), z_defend=('z_defend', 'mean'), z_attack=('z_attack', 'mean'), r_defend=('r_defend', 'mean'))
    table['pulled_vs_expected'] = table['taken_off'] - table['expected']
    table = table.join(players.set_index('player_id')[['name', 'position']])
    table['team'] = pm.sort_values('minutes').groupby('player_id')['team_id'].last().reindex(table.index).map(teams)
    table[table['decisions'] >= 15].sort_values('z_defend', ascending=False).to_csv(C.RESULTS / 'q2_trusted_by_rating.csv')
    show = table[(table['decisions'] >= 15) & (table['z_defend'] >= 1)].sort_values('pulled_vs_expected', ascending=False).head(10)
    print(show[['name', 'team', 'position', 'decisions', 'taken_off', 'expected', 'z_defend']].round(2).to_string())
    # Exploratory, reported as such. 1) Do the stakes help? Same test with the channels unweighted.
    raw_v = pd.read_parquet(C.WORK / f'values_{args.tag}.parquet', columns=['player_id', 'game_id', 'dP_scores', 'dP_concedes'])
    raw_v = raw_v[raw_v['player_id'] > 0].assign(u_attack=lambda x: x['dP_scores'], u_defend=lambda x: -x['dP_concedes']).groupby(['player_id', 'game_id'])[['u_attack', 'u_defend']].sum()
    raw_tot = raw_v.groupby('player_id').sum()
    compare = []
    for table, cols, name in ((off, off_cols, 'taken_off'), (on, on_cols, 'brought_on')):
        t = table.join(raw_v, on=['player_id', 'game_id']).fillna({'u_attack': 0.0, 'u_defend': 0.0})
        tot = raw_tot.reindex(t['player_id'].values)
        for k in ('u_attack', 'u_defend'):
            t[f'z_{k}'] = (tot[k].values - t[k].values) / t['other_minutes'].values * 90
            t[f'z_{k}'] = t.groupby('position')[f'z_{k}'].transform(lambda s: (s - s.mean()) / s.std())
        for g_val, label in ((1, 'protecting'), (-1, 'chasing')):
            x = t[t['g'] == g_val]
            gr = pd.factorize(x['decision'])[0]
            rows_ = {}
            for nm, focus in (('stakes', ['z_defend', 'z_attack']), ('unweighted', ['z_u_defend', 'z_u_attack'])):
                use = focus + [c for c in cols if c not in ('z_defend', 'z_attack')]
                X = x[use].values.astype(float)
                b = conditional_logit(X, x['chosen'].values.astype(float), gr)
                eta = X @ b
                big = np.full(gr.max() + 1, -np.inf)
                np.maximum.at(big, gr, eta)
                ex_ = np.exp(eta - big[gr])
                rows_[nm] = float((x['chosen'].values * (eta - big[gr] - np.log(np.bincount(gr, weights=ex_)[gr]))).sum())
            compare.append({'decision': name, 'state': label, 'loglik_stakes': rows_['stakes'], 'loglik_unweighted': rows_['unweighted'], 'gain': rows_['stakes'] - rows_['unweighted']})
    compare = pd.DataFrame(compare)
    compare.to_csv(C.RESULTS / 'q2_stakes_vs_unweighted.csv', index=False)
    print(compare.round(2).to_string(index=False))

    # 2) Replacements: how often does the coach pick the highest-rated bench player of the position he sends on?
    picks = []
    for g_val, rating, label in ((1, 'z_defend', 'protecting, goal prevention'), (-1, 'z_attack', 'chasing, attacking'), (1, 'z_attack', 'protecting, attacking'), (-1, 'z_defend', 'chasing, goal prevention')):
        x = on[on['g'] == g_val]
        chosen = x[x['chosen'] == 1][['decision', 'position']].rename(columns={'position': 'p_chosen'})
        same = x.merge(chosen, on='decision')
        same = same[same['position'] == same['p_chosen']]
        n = same.groupby('decision')['chosen'].transform('size')
        same = same[n >= 2]
        best = same.loc[same.groupby('decision')[rating].idxmax()]
        k = same.groupby('decision').size()
        z_gap = (same.groupby('decision')[rating].max() - same[same['chosen'] == 1].set_index('decision')[rating])
        picks.append({'case': label, 'decisions': len(k), 'picked_best': float(best['chosen'].mean()), 'chance': float((1 / k).mean()), 'mean_gap_to_best_z': float(z_gap.mean())})
    picks = pd.DataFrame(picks)
    picks.to_csv(C.RESULTS / 'q2_replacements.csv', index=False)
    print(picks.round(3).to_string(index=False))
    corr = off[['z_defend', 'z_attack']].corr().iloc[0, 1]
    print('correlation of the two ratings within position:', round(corr, 3))
    (C.RESULTS / 'q2_summary.json').write_text(json.dumps({'taken_off_decisions': int(off['decision'].nunique()), 'brought_on_decisions': int(on['decision'].nunique()),
                                                            'protecting_off': int(off[off['g'] == 1]['decision'].nunique()), 'chasing_off': int(off[off['g'] == -1]['decision'].nunique())}, indent=1))
    print('stage 9 done')


if __name__ == '__main__':
    main()
