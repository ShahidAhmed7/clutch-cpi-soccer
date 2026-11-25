"""Tables, figures and a readable report for one league run. Analyses that the data cannot support are skipped and listed.

What runs when
    always                        model accuracy on this data, stakes, ratings with intervals, top clutch moments,
                                  goal values by match context, goals against their clutch value, goalkeepers
    20+ matches                   match-level ledger: net clutch value against points earned
    a full league                 season ledger against the table, split-half repeatability
"""
import json
from datetime import datetime, timezone

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import pearsonr, spearmanr  # noqa: E402

from .. import config as C, ratings  # noqa: E402

BLUE, ORANGE, GREY, INK, INK2, GRID = '#2a78d6', '#eb6834', '#8a8984', '#0b0b0b', '#52514e', '#e4e3df'
plt.rcParams.update({'font.size': 8, 'font.family': 'DejaVu Sans', 'axes.edgecolor': GRID, 'axes.linewidth': 0.8, 'xtick.color': INK2, 'ytick.color': INK2,
                     'axes.labelcolor': INK2, 'text.color': INK, 'axes.titlesize': 8.5, 'axes.titlelocation': 'left', 'xtick.major.size': 0,
                     'ytick.major.size': 0, 'legend.frameon': False, 'pdf.fonttype': 42})
CONTEXTS = [("Level, 85'+", lambda d: (d['goal_diff'] == 0) & (d['minute'] >= 85)),
            ("One down, 85'+", lambda d: (d['goal_diff'] == -1) & (d['minute'] >= 85)),
            ("Level, first 15'", lambda d: (d['goal_diff'] == 0) & (d['minute'] < 15)),
            ("One up, 85'+", lambda d: (d['goal_diff'] == 1) & (d['minute'] >= 85)),
            ("Three or more up, 60'+", lambda d: (d['goal_diff'] >= 3) & (d['minute'] >= 60)),
            ('All non-penalty goals', lambda d: np.ones(len(d), bool))]


def _clean(ax, grid='y'):
    for s in ('top', 'right', 'left'):
        ax.spines[s].set_visible(False)
    ax.grid(axis=grid, color=GRID, lw=0.7)
    ax.set_axisbelow(True)


def _save(fig, folder, name):
    fig.savefig(folder / f'{name}.png', dpi=200, bbox_inches='tight', pad_inches=0.03)
    fig.savefig(folder / f'{name}.pdf', bbox_inches='tight', pad_inches=0.03)
    plt.close(fig)


def _md(df, floatfmt=3):
    """A small DataFrame as a Markdown table, without extra dependencies."""
    d = df.copy()
    for c in d.columns:
        if pd.api.types.is_float_dtype(d[c]):
            d[c] = d[c].map(lambda x: '' if pd.isna(x) else f'{x:.{floatfmt}f}')
    cols = [str(c) for c in d.columns]
    lines = ['| ' + ' | '.join(cols) + ' |', '|' + '|'.join('---' for _ in cols) + '|']
    lines += ['| ' + ' | '.join(str(x) for x in row) + ' |' for row in d.astype(str).values]
    return '\n'.join(lines)


def stakes_at(table, minutes=(0, 30, 60, 75, 80, 85, 89)):
    """L+ and L- for one goal down, level and one goal up, averaged over home and away, at chosen minutes."""
    rows = []
    for mnt in minutes:
        n = C.T_FRAMES - min(int(mnt * 60 / C.SECS_PER_FRAME), C.T_FRAMES - 1)
        for g, label in ((-1, 'one down'), (0, 'level'), (1, 'one up')):
            t = table[(table['g'] == g) & (table['n_left'] == n) & (table['adv'] == 0)]
            rows.append({'minute': mnt, 'score': label, 'L_plus': t['L_plus'].mean(), 'L_minus': t['L_minus'].mean(), 'W': t['W'].mean()})
    return pd.DataFrame(rows)


def late_lead(table):
    """Mean stakes for a one-goal lead after the 80th minute (the paper's headline contrast)."""
    n = np.arange(1, C.T_FRAMES - int(80 * 60 / C.SECS_PER_FRAME) + 1)
    t = table[(table['g'] == 1) & table['n_left'].isin(n) & (table['adv'] == 0)]
    return float(t['L_minus'].mean()), float(t['L_plus'].mean())


def fig_stakes(table, folder):
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.1), sharey=True)
    mins = np.arange(0, 90)
    for ax, (g, title) in zip(axes, ((0, 'Level'), (1, 'One goal up'), (-1, 'One goal down'))):
        lp, lm = [], []
        for mnt in mins:
            n = C.T_FRAMES - int(mnt * 60 / C.SECS_PER_FRAME)
            t = table[(table['g'] == g) & (table['n_left'] == n) & (table['adv'] == 0)]
            lp.append(t['L_plus'].mean())
            lm.append(t['L_minus'].mean())
        ax.plot(mins, lp, color=BLUE, lw=1.6, label='L+, gain if the team scores next')
        ax.plot(mins, lm, color=ORANGE, lw=1.6, label='L-, cost if it concedes next')
        ax.set_title(title)
        ax.set_xlabel('Match minute')
        ax.set_xticks([0, 30, 60, 90])
        _clean(ax)
    axes[0].set_ylabel('League points')
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc='upper center', bbox_to_anchor=(0.5, 1.08), ncol=2, fontsize=7)
    _save(fig, folder, 'stakes_curves')


def goal_values(v):
    g = v[v['goal'] & (v['type_name'] != 'shot_penalty')]
    rows = [{'Goal scored when': lab, 'Count': int(f(g).sum()), 'VAEP': g.loc[f(g), 'value'].mean(), 'V (league points)': g.loc[f(g), 'V'].mean()}
            for lab, f in CONTEXTS]
    return pd.DataFrame(rows)


OUTPUTS = ['report.md', 'run_info.json', 'action_values.parquet', 'player_ratings.csv', 'all_players_per90.csv', 'stakes_by_minute.csv',
           'goal_values_by_context.csv', 'top_clutch_actions.csv', 'top_preventive_actions.csv', 'goals_vs_value.csv', 'goalkeepers.csv',
           'season_ledger.csv', 'observed_late_stakes.csv', 'repeatability.csv'] + \
          [f'{f}.{e}' for f in ('stakes_curves', 'top_players', 'goals_vs_value', 'season_ledger') for e in ('png', 'pdf')]
DEFENSIVE = ['clearance', 'interception', 'tackle', 'keeper_save', 'keeper_claim', 'keeper_punch', 'keeper_pick_up']


def top_moments(v, b, by='V', n=15, types=None):
    names = b.players.set_index('player_id')['name']
    teams = b.teams.set_index('team_id')['team']
    m = b.matches.set_index('game_id')
    t = (v[v['type_name'].isin(types)] if types else v).nlargest(n, by).copy()
    t['match'] = [f"{teams.get(m.loc[g, 'home_team_id'], m.loc[g, 'home_team_id'])} v {teams.get(m.loc[g, 'away_team_id'], m.loc[g, 'away_team_id'])}" for g in t['game_id']]
    t['player'] = t['player_id'].map(names)
    t['team'] = t['team_id'].map(teams)
    t['score_before'] = t['goals_team'].astype(int).astype(str) + '-' + t['goals_opp'].astype(int).astype(str)
    t['minute'] = t['minute'].round(0).astype(int)
    return t[['match', 'minute', 'team', 'player', 'type_name', 'result_name', 'score_before', 'V', 'value', 'attack', 'defend', 'L_plus', 'L_minus']].rename(
        columns={'type_name': 'action', 'result_name': 'result', 'value': 'VAEP'})


def goals_vs_value(v, b, min_goals=None):
    g = v[v['goal'] & (v['type_name'] != 'shot_penalty') & (v['player_id'] > 0)]
    t = g.groupby('player_id').agg(np_goals=('V', 'size'), goal_points=('V', 'sum'), goal_vaep=('value', 'sum'))
    t = t.join(b.players.set_index('player_id')[['name']]).join(g.groupby('player_id')['team_id'].last().map(b.teams.set_index('team_id')['team']).rename('team'))
    if min_goals is None:
        min_goals = 8 if (t['np_goals'] >= 8).sum() >= 10 else max(2, int(t['np_goals'].quantile(0.8)))
    t = t[t['np_goals'] >= min_goals].sort_values('goal_points', ascending=False)
    t['points_per_goal'] = t['goal_points'] / t['np_goals']
    return t, min_goals


def fig_goals(t, folder, label_n=6):
    if len(t) < 3:
        return False
    fig, ax = plt.subplots(figsize=(3.8, 2.0))
    ax.scatter(t['np_goals'], t['goal_points'], s=14, color=GREY, alpha=0.7, edgecolor='white', lw=0.4)
    mean = t['goal_points'].sum() / t['np_goals'].sum()
    xs = np.array([t['np_goals'].min(), t['np_goals'].max() + 1])
    ax.plot(xs, mean * xs, color=GREY, lw=1, ls='--')
    resid = t['goal_points'] - mean * t['np_goals']
    for pid in list(resid.nlargest(label_n // 2).index) + list(t['goal_points'].nlargest(label_n // 2).index):
        r = t.loc[pid]
        ax.scatter(r['np_goals'], r['goal_points'], s=20, color=BLUE, edgecolor='white', lw=0.5, zorder=3)
        ax.annotate(str(r['name']), (r['np_goals'], r['goal_points']), xytext=(3, 3), textcoords='offset points', fontsize=6.5, color=INK)
    ax.set_xlabel('Non-penalty goals')
    ax.set_ylabel('Season clutch value of goals\n(league points)')
    _clean(ax, 'both')
    _save(fig, folder, 'goals_vs_value')
    return True


def keepers_table(ks, b, pl):
    if not len(ks) or 'keeper_id' not in ks:
        return pd.DataFrame()
    k = ks[ks['keeper_id'] > 0].groupby('keeper_id').agg(shots_faced=('goal', 'size'), goals=('goal', 'sum'), post_shot_xg=('psxg', 'sum'),
                                                          goals_prevented=('goals_prevented', 'sum'), points_saved=('points_saved', 'sum'))
    k = k.join(b.players.set_index('player_id')[['name']]).join(pl[['team', 'minutes']])
    k['points_saved_per90'] = k['points_saved'] / k['minutes'].replace(0, np.nan) * 90
    return k.sort_values('points_saved', ascending=False)


def fig_leaders(q, folder, n=8):
    pos = [p for p in C.POSITIONS if (q['position'] == p).any()]
    if not pos:
        return False
    fig, axes = plt.subplots(1, len(pos), figsize=(1.9 * len(pos) + 0.6, 2.4), sharex=False)
    axes = np.atleast_1d(axes)
    for ax, p in zip(axes, pos):
        t = q[q['position'] == p].nsmallest(n, 'rank').iloc[::-1]
        y = np.arange(len(t))
        ax.errorbar(t['CPI'], y, xerr=[t['CPI'] - t['ci_low'], t['ci_high'] - t['CPI']], fmt='o', ms=3, color=BLUE, ecolor=GREY, elinewidth=0.9, capsize=0)
        ax.set_yticks(y)
        ax.set_yticklabels([str(s)[:18] for s in t['name']], fontsize=6.5)
        ax.axvline(0, color=GRID, lw=0.8)
        ax.set_title(p.capitalize() + 's')
        _clean(ax, 'x')
    axes[0].set_xlabel('CPI, league points per 90 (95% interval)')
    fig.tight_layout()
    _save(fig, folder, 'top_players')
    return True


def match_ledger(v, matches):
    tm = v.groupby(['game_id', 'team_id'])[['V', 'value']].sum().reset_index()
    sc = pd.concat([matches[['game_id', 'home_team_id', 'away_team_id', 'home_score', 'away_score']].set_axis(['game_id', 'team_id', 'opp', 'gf', 'ga'], axis=1),
                    matches[['game_id', 'away_team_id', 'home_team_id', 'away_score', 'home_score']].set_axis(['game_id', 'team_id', 'opp', 'gf', 'ga'], axis=1)])
    tm = tm.merge(sc, on=['game_id', 'team_id'])
    tm = tm.merge(tm[['game_id', 'team_id', 'V', 'value']].rename(columns={'team_id': 'opp', 'V': 'V_opp', 'value': 'value_opp'}), on=['game_id', 'opp'])
    tm['pts'] = np.select([tm['gf'] > tm['ga'], tm['gf'] == tm['ga']], [3, 1], 0)
    tm['net_V'], tm['net_value'], tm['gd'] = tm['V'] - tm['V_opp'], tm['value'] - tm['value_opp'], tm['gf'] - tm['ga']
    return tm


def observed_late_stakes(b, v, table, n_boot=200):
    """The paper's test against real results: after the 80th minute, what scoring and conceding did to final points,
    from adjacent score states, adjusted for pre-match Elo difference and home advantage (weighted least squares)."""
    from .. import elo, stakes, state
    acts = v[['game_id', 'team_id', 'goal', 'own_goal']].assign(elapsed=state.elapsed_seconds(v['period_id'], v['time_seconds']))
    panel = stakes.build_panel(acts, b.matches, b.reds)
    e = elo.pre_match_elo(b.matches.assign(date=b.matches['date'].fillna('') if 'date' in b.matches else ''))
    long = pd.concat([e[['game_id', 'home_team_id', 'elo_home_pre', 'elo_away_pre']].set_axis(['game_id', 'team_id', 'e', 'o'], axis=1),
                      e[['game_id', 'away_team_id', 'elo_away_pre', 'elo_home_pre']].set_axis(['game_id', 'team_id', 'e', 'o'], axis=1)])
    long['elo_gap'] = (long['e'] - long['o']) / 100
    d = panel.merge(long[['game_id', 'team_id', 'elo_gap']], on=['game_id', 'team_id'])
    d = d[d['frame'] * C.SECS_PER_FRAME / 60 >= 80].reset_index(drop=True)
    d['g3'] = d['g'].clip(-3, 3)
    t = stakes.attach(d, table, g='g', n='n_left', home='home', adv='adv')
    d['L_plus'], d['L_minus'] = t['L_plus'].values, t['L_minus'].values
    G = [-3, -2, -1, 1, 2, 3]
    games = d['game_id'].unique()
    gi = pd.Series(np.arange(len(games)), index=games).loc[d['game_id']].values

    def real(w):
        X = np.column_stack([np.ones(len(d))] + [(d['g3'].values == k).astype(float) for k in G] + [d['elo_gap'].values, d['home'].values])
        Xw = X * w[gi][:, None]
        beta = np.linalg.lstsq(Xw.T @ X, Xw.T @ d['pts'].values.astype(float), rcond=None)[0]
        return {0: 0.0, **dict(zip(G, beta[1:7]))}
    rng = np.random.default_rng(C.SEED)
    point = real(np.ones(len(games)))
    boots = [real(np.bincount(rng.integers(0, len(games), len(games)), minlength=len(games)).astype(float)) for _ in range(n_boot)]
    rows = []
    for name, g in (('one goal up', 1), ('level', 0), ('one goal down', -1)):
        m = d['g3'] == g
        diff = np.array([(p[g] - p[g - 1]) - (p[g + 1] - p[g]) for p in boots])
        rows.append({'after 80 minutes': name, 'observed gain': point[g + 1] - point[g], 'observed cost': point[g] - point[g - 1],
                     'model gain': d.loc[m, 'L_plus'].mean(), 'model cost': d.loc[m, 'L_minus'].mean(),
                     'cost - gain': (point[g] - point[g - 1]) - (point[g + 1] - point[g]), 'ci low': np.percentile(diff, 2.5), 'ci high': np.percentile(diff, 97.5)})
    return pd.DataFrame(rows)


def fig_ledger(season, folder):
    fig, ax = plt.subplots(figsize=(3.8, 2.0))
    ax.scatter(season['net_V'], season['pts'], s=16, color=BLUE, edgecolor='white', lw=0.5)
    slope, icept = np.polyfit(season['net_V'], season['pts'], 1)
    xs = np.linspace(season['net_V'].min(), season['net_V'].max(), 10)
    ax.plot(xs, icept + slope * xs, color=GREY, lw=1, ls='--')
    for _, r in season.nlargest(2, 'pts').iterrows():
        ax.annotate(str(r['team']), (r['net_V'], r['pts']), xytext=(3, -8), textcoords='offset points', fontsize=6.5)
    ax.set_xlabel('Season net clutch value (league points)')
    ax.set_ylabel('Earned league points')
    _clean(ax, 'both')
    _save(fig, folder, 'season_ledger')


def write(res, folder, data_path=None, log=print):
    """All outputs of one run into <folder>."""
    folder.mkdir(parents=True, exist_ok=True)
    figs = folder / 'figures'
    figs.mkdir(exist_ok=True)
    for name in OUTPUTS:                                       # only this tool's own files from an earlier run of the same name
        for old in (folder / name, figs / name):
            if old.is_file():
                old.unlink()
    b, ck, v, q = res['bundle'], res['checks'], res['values'], res['ratings']
    skipped, lines = [], []

    v.to_parquet(folder / 'action_values.parquet', index=False)
    keep = ['name', 'team', 'position', 'minutes', 'matches', 'goals', 'assists', 'CPI', 'ci_low', 'ci_high', 'CPI_adjusted', 'P_top10', 'rank',
            'attack_per90', 'defend_per90', 'points_saved_per90', 'value_per90', 'rank_vaep', 'V', 'penalties']
    if len(q):
        q.sort_values(['position', 'rank'])[[c for c in keep if c in q]].rename(columns={'defend_per90': 'prevention_per90', 'value_per90': 'VAEP_per90',
                                                                                         'V': 'season_clutch_value'}).to_csv(folder / 'player_ratings.csv')
    res['players_all'].to_csv(folder / 'all_players_per90.csv')
    acc = res['model_accuracy']
    fig_stakes(res['stakes_table'], figs)
    st = stakes_at(res['stakes_table'])
    st.to_csv(folder / 'stakes_by_minute.csv', index=False)
    lm, lp = late_lead(res['stakes_table'])
    gv = goal_values(v)
    gv.to_csv(folder / 'goal_values_by_context.csv', index=False)
    top = top_moments(v, b)
    top.to_csv(folder / 'top_clutch_actions.csv', index=False)
    prev = top_moments(v, b, by='defend', n=10, types=DEFENSIVE)
    prev.to_csv(folder / 'top_preventive_actions.csv', index=False)
    gvv, min_goals = goals_vs_value(v, b)
    gvv.to_csv(folder / 'goals_vs_value.csv')
    has_goals_fig = fig_goals(gvv, figs)
    kt = keepers_table(res['keeper_shots'], b, res['players_all'])
    if len(kt):
        kt.to_csv(folder / 'goalkeepers.csv')
    has_leaders = len(q) and fig_leaders(q, figs)

    # ledgers and repeatability, only when the data covers enough
    tm = match_ledger(v, b.matches)
    match_led, season_led, rel = None, None, None
    focus = ck.get('focus_team_id')
    tm_focus = tm[tm['team_id'] == focus] if focus is not None else tm
    if len(tm_focus) >= 20:
        match_led = {'matches': int(len(tm_focus)), 'pearson_net_V_pts': float(pearsonr(tm_focus['net_V'], tm_focus['pts'])[0]),
                     'pearson_gd_pts': float(pearsonr(tm_focus['gd'], tm_focus['pts'])[0])}
    else:
        skipped.append('match ledger (needs 20+ matches)')
    if ck['full_league']:
        season = tm.groupby('team_id').agg(games=('pts', 'size'), pts=('pts', 'sum'), net_V=('net_V', 'sum'), gd=('gd', 'sum')).reset_index()
        season['team'] = season['team_id'].map(b.teams.set_index('team_id')['team'])
        season.to_csv(folder / 'season_ledger.csv', index=False)
        slope = float(np.polyfit(season['net_V'], season['pts'], 1)[0])
        season_led = {'teams': int(len(season)), 'pearson': float(pearsonr(season['net_V'], season['pts'])[0]), 'slope': slope}
        fig_ledger(season, figs)
    else:
        skipped.append('season ledger against the league table (needs a full league)')
    observed = None
    if ck['full_league'] and ck['matches'] >= 150:
        observed = observed_late_stakes(b, v, res['stakes_table'])
        observed.to_csv(folder / 'observed_late_stakes.csv', index=False)
    else:
        skipped.append('stakes against real results (needs a full league of 150+ matches)')
    if len(q) >= 10 and b.matches['gameweek'].nunique() >= 10:
        rel = ratings.split_half(res['player_match'], q.index.values, q['position'], ['V', 'value'], b.matches)
        rel = rel[rel['players'] >= 15]
        if len(rel):
            rel.to_csv(folder / 'repeatability.csv', index=False)
        else:
            rel = None
            skipped.append('split-half repeatability (fewer than 15 players per position with 450+ minutes in each half)')
    else:
        skipped.append('split-half repeatability (needs 10+ rated players and 10+ gameweeks)')
    skipped.append('substitution analysis (needs a full league season; run the paper pipeline for that)')

    # report
    when = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')
    teams = b.teams.set_index('team_id')['team']
    lines += [f'# Clutch Performance Index: {b.name}', '', f'Generated {when} from `{data_path or "?"}` (provider: {b.provider}).', '']
    lines += ['## Data', '', f"- {ck['matches']} matches, {ck['actions']:,} actions, {ck['teams']} teams, {ck['players_with_minutes']} players with minutes.",
              f"- Goals rebuilt from events: {ck['goals']} + {ck['own_goals']} own goals against {ck['official_goals']} in the official scores; "
              f"scores match for {ck['score_match_share']:.1%} of team-match records.",
              f"- Coverage: " + (f"one team's matches ({teams.get(focus, focus)}); ratings are shown for that team only." if focus is not None else
                                 'a full league season.' if ck['full_league'] else 'a partial set of matches.'),
              f"- On-target shots with a goal-mouth zone (for keepers): {ck['on_target_shots_with_zone']}; red cards: {ck['red_cards']}."]
    for n in b.notes:
        lines.append(f'- Note: {n}')
    lines += ['', '## Models', '', f"- Action values: **{res['mode']}**" + (' (fitted on the five 2017/18 leagues, 2.3M actions)' if res['mode'] == 'pretrained' else
                                                                         ' (5-fold cross-fitting by match on this data)') + '.',
              f"- Stakes: {res['stakes_source']}.",
              *([f"- Matched to the training data's conventions: {'; '.join(res['harmonized'])}."] if res.get('harmonized') else []),
              *([f"- Recalibrated on this data (logistic, cross-fitted by match): scoring a = {res['recalibration']['scores'][0]:.2f}, "
                 f"b = {res['recalibration']['scores'][1]:.2f}; conceding a = {res['recalibration']['concedes'][0]:.2f}, b = {res['recalibration']['concedes'][1]:.2f}. "
                 f"Before recalibration the mean predicted chances were {res['model_accuracy_raw']['scores']['mean_pred']:.4f} (scoring) and "
                 f"{res['model_accuracy_raw']['concedes']['mean_pred']:.4f} (conceding)."] if res.get('recalibration') else []),
              f"- Accuracy on this data: scoring ROC-AUC {acc['scores']['auc']:.3f}, conceding ROC-AUC {acc['concedes']['auc']:.3f} "
              f"(paper, held-out leagues: {acc['paper_held_out_auc'].get('scores', float('nan')):.3f} and {acc['paper_held_out_auc'].get('concedes', float('nan')):.3f})."
              if acc['scores']['auc'] is not None else '- Accuracy: too few goals in this data to measure.',
              f"- Mean predicted scoring chance {acc['scores']['mean_pred']:.4f} against an observed rate of {acc['scores']['rate']:.4f}; "
              f"conceding {acc['concedes']['mean_pred']:.4f} against {acc['concedes']['rate']:.4f}."]
    if 'psxg_check' in res and res['psxg_check']['auc'] is not None:
        p = res['psxg_check']
        lines.append(f"- Post-shot xG on {p['shots']} on-target shots: ROC-AUC {p['auc']:.3f}, {p['predicted']:.1f} goals predicted against {p['goals']} scored"
                     + (f" (before recalibration {p['predicted_raw']:.1f})." if res.get('psxg_recalibration') else '.'))
    lines += ['', '## What a goal is worth', '', f'For a one-goal lead after the 80th minute, conceding costs {lm:.2f} league points on average and scoring gains '
              f'{lp:.2f} ({res["stakes_source"]}).', '', '![stakes](figures/stakes_curves.png)', '',
              *(['### Against real results', '', 'From final points in adjacent score states after the 80th minute, adjusted for pre-match Elo '
                 'difference and home advantage; 95% match-bootstrap interval for cost minus gain.', '', _md(observed, 2), ''] if observed is not None else []),
              '### Goals by match context', '',
              'VAEP is on a goal-probability scale; V is in league points (this data).', '', _md(gv, 2), '']
    if len(q):
        lines += [f'## Player ratings (at least {res["min_minutes"]:.0f} minutes)', '',
                  'CPI is league points per 90 minutes with a 95% match-bootstrap interval. Adjusted is the Empirical-Bayes rating within '
                  'position, shown where a position has 10 or more rated players.', '']
        if has_leaders:
            lines += ['![leaders](figures/top_players.png)', '']
        for p in C.POSITIONS:
            t = q[q['position'] == p].nsmallest(10, 'rank')
            if not len(t):
                continue
            cols = ['name', 'team', 'minutes', 'goals', 'CPI', 'ci_low', 'ci_high', 'CPI_adjusted', 'attack_per90', 'defend_per90', 'value_per90']
            cols = [c for c in cols if c in t]
            show = t[cols].rename(columns={'attack_per90': 'attack/90', 'defend_per90': 'prevention/90', 'value_per90': 'VAEP/90', 'CPI_adjusted': 'adjusted'})
            show['minutes'] = show['minutes'].round(0).astype(int)
            lines += [f'### {p.capitalize()}s', '', _md(show), '']
    else:
        lines += ['## Player ratings', '', f'No player reached {res["min_minutes"]:.0f} minutes.', '']
    lines += ['## Most valuable actions', '', _md(top.head(10), 3), '', '### Most valuable defensive actions', '',
              'Clearances, interceptions, tackles and goalkeeper actions, ranked by their goal-prevention value.', '', _md(prev.head(5), 3), '']
    if has_goals_fig:
        lines += [f'## Goals and what they were worth (players with {min_goals}+ non-penalty goals)', '', '![goals](figures/goals_vs_value.png)', '',
                  _md(gvv.head(10)[['name', 'team', 'np_goals', 'goal_points', 'points_per_goal']], 2), '']
    if len(kt):
        lines += ['## Goalkeepers', '', 'Points saved = (post-shot xG - goal) x cost of conceding at that moment.', '',
                  _md(kt.head(10)[['name', 'team', 'shots_faced', 'goals', 'post_shot_xg', 'goals_prevented', 'points_saved']].round(2), 2), '']
    if match_led:
        lines += ['## Does the ledger add up?', '', f"Across {match_led['matches']} team-matches, net clutch value correlates with points earned at "
                  f"r = {match_led['pearson_net_V_pts']:.2f} (goal difference: {match_led['pearson_gd_pts']:.2f})."]
        if season_led:
            lines += [f"Across {season_led['teams']} teams, season net clutch value correlates with league points at r = {season_led['pearson']:.3f} "
                      f"(slope {season_led['slope']:.2f}).", '', '![ledger](figures/season_ledger.png)']
        lines.append('')
    if rel is not None:
        lines += ['## Repeatability', '', 'Odd against even gameweeks, Spearman-Brown corrected (players with 450+ minutes in each half).', '',
                  _md(rel.rename(columns={'V': 'CPI', 'value': 'VAEP'}), 2), '']
    lines += ['## Not run on this data', ''] + [f'- {s}' for s in skipped] + ['', '## Files', '',
              '- `player_ratings.csv`: rated players with intervals and components; `all_players_per90.csv`: everyone with minutes',
              '- `action_values.parquet`: every action with its probabilities, stakes and values (V, attack, defend, VAEP)',
              '- `top_clutch_actions.csv`, `top_preventive_actions.csv`, `goal_values_by_context.csv`, `goals_vs_value.csv`, `goalkeepers.csv`',
              '- `stakes_by_minute.csv`, `run_info.json`, `figures/` (PNG and PDF)', '']
    (folder / 'report.md').write_text('\n'.join(lines))

    info = {'name': b.name, 'provider': b.provider, 'data': str(data_path), 'generated_utc': when, 'mode': res['mode'], 'stakes': res['stakes_source'],
            'harmonized': res.get('harmonized', []), 'recalibration': res.get('recalibration'), 'model_accuracy_raw': res.get('model_accuracy_raw'),
            'min_minutes': res['min_minutes'], 'checks': {k: v_ for k, v_ in ck.items() if k != 'score_mismatches'}, 'model_accuracy': acc,
            'late_one_goal_lead': {'cost_of_conceding': lm, 'gain_from_scoring': lp}, 'match_ledger': match_led, 'season_ledger': season_led,
            'observed_late_stakes': observed.to_dict('records') if observed is not None else None,
            'psxg_check': res.get('psxg_check'), 'notes': b.notes, 'skipped': skipped}
    (folder / 'run_info.json').write_text(json.dumps(info, indent=1, default=lambda o: int(o) if isinstance(o, np.integer) else float(o) if isinstance(o, np.floating) else str(o)))
    log(f'report written to {folder / "report.md"}')
    return info
