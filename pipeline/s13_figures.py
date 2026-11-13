"""Stage 13. Figures for the paper, written to paper/figures/. Every number comes from results/ or work/.

    python pipeline/s13_figures.py
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cpi import config as C  # noqa: E402

OUT = C.ROOT / 'paper' / 'figures'
BLUE, ORANGE, GREY = '#2a78d6', '#eb6834', '#8a8984'
INK, INK2, GRID = '#0b0b0b', '#52514e', '#e4e3df'
plt.rcParams.update({'font.size': 8, 'font.family': 'DejaVu Sans', 'axes.edgecolor': GRID, 'axes.linewidth': 0.8, 'xtick.color': INK2, 'ytick.color': INK2,
                     'axes.labelcolor': INK2, 'text.color': INK, 'axes.titlesize': 8.5, 'axes.titlelocation': 'left', 'xtick.major.size': 0, 'ytick.major.size': 0,
                     'legend.frameon': False, 'pdf.fonttype': 42})


def clean(ax, grid='y'):
    for s in ('top', 'right', 'left'):
        ax.spines[s].set_visible(False)
    ax.grid(axis=grid, color=GRID, lw=0.7)
    ax.set_axisbelow(True)


def save(fig, name):
    fig.savefig(OUT / f'{name}.pdf', bbox_inches='tight', pad_inches=0.02)
    fig.savefig(OUT / f'{name}.png', dpi=220, bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)


def fig_stakes():
    p = pd.read_parquet(C.WORK / 'panel.parquet', columns=['frame', 'g', 'L_plus', 'L_minus'])
    p['minute'] = p['frame'] * C.SECS_PER_FRAME / 60
    cells = pd.read_csv(C.RESULTS / 't2_cells.csv')
    fig, axes = plt.subplots(1, 3, figsize=(4.8, 1.65), sharey=True)
    for ax, (g, title) in zip(axes, ((0, 'Level'), (1, 'One goal up'), (-1, 'One goal down'))):
        m = p[p['g'] == g].groupby('minute')[['L_plus', 'L_minus']].mean()
        m = m[m.index <= 90]
        ax.plot(m.index, m['L_plus'], color=BLUE, lw=1.6, solid_capstyle='round')
        ax.plot(m.index, m['L_minus'], color=ORANGE, lw=1.6, solid_capstyle='round')
        c = cells[cells['g'] == g]
        x = c['bin'] * 10 + 5
        ax.scatter(x, c['real_gain'], s=17, color=BLUE, edgecolor='white', lw=0.6, zorder=3)
        ax.scatter(x, c['real_cost'], s=17, color=ORANGE, edgecolor='white', lw=0.6, zorder=3, marker='s')
        ax.set_title(title)
        ax.set_xlabel('Match minute')
        ax.set_xticks([0, 30, 60, 90])
        ax.set_xlim(0, 92)
        clean(ax)
    axes[0].set_ylabel('League points')
    axes[0].set_ylim(-0.1, 1.9)
    h = [plt.Line2D([], [], color=BLUE, lw=1.6, label='$L^{+}$, gain if the team scores next (model)'),
         plt.Line2D([], [], color=ORANGE, lw=1.6, label='$L^{-}$, cost if it concedes next (model)'),
         plt.Line2D([], [], marker='o', color=BLUE, ls='', ms=4, label='real gain'), plt.Line2D([], [], marker='s', color=ORANGE, ls='', ms=4, label='real cost')]
    fig.legend(handles=h, loc='upper center', ncol=2, bbox_to_anchor=(0.5, 1.2), fontsize=6.8, handlelength=1.6, columnspacing=1.2)
    save(fig, 'stakes')


def fig_ledger(ax):
    s = pd.read_csv(C.RESULTS / 't3_season_ledger.csv')
    teams = pd.read_csv(C.RESULTS / 'ratings.csv', usecols=['team_id', 'team']).drop_duplicates('team_id').set_index('team_id')['team']
    ax.scatter(s['net_V'], s['pts'], s=15, color=BLUE, edgecolor='white', lw=0.5, alpha=0.9)
    b = np.polyfit(s['net_V'], s['pts'], 1)
    xs = np.array([s['net_V'].min(), s['net_V'].max()])
    ax.plot(xs, np.polyval(b, xs), color=GREY, lw=1, ls='--')
    top = s.nlargest(2, 'pts').index.tolist() + s.nsmallest(1, 'pts').index.tolist()
    for i in top:
        ax.annotate(teams.get(s.loc[i, 'team_id'], ''), (s.loc[i, 'net_V'], s.loc[i, 'pts']), xytext=(-5, 0) if s.loc[i, 'pts'] > 60 else (6, -1), textcoords='offset points',
                    ha='right' if s.loc[i, 'pts'] > 60 else 'left', fontsize=6, color=INK2)
    ax.set_xlabel('Season net clutch value (league points)')
    ax.set_ylabel('Real league points')
    ax.set_title('(a) Season ledger, one dot per team', fontsize=7.5)
    clean(ax, 'both')


def fig_goals(ax):
    g = pd.read_csv(C.RESULTS / 'goals_vs_points.csv')
    ax.scatter(g['np_goals'], g['goal_points'], s=13, color=GREY, alpha=0.55, edgecolor='white', lw=0.4)
    mean = g['goal_points'].sum() / g['np_goals'].sum()
    xs = np.array([8, g['np_goals'].max() + 1])
    ax.plot(xs, mean * xs, color=GREY, lw=1, ls='--')
    ax.text(33, 3.0, f'dashed: {mean:.2f} per goal', fontsize=6, color=INK2, ha='right')
    label = {'Gerard Moreno': (-3, 3, 'right'), 'Cristiano Ronaldo': (4, -7, 'left'), 'Mohamed Salah': (-4, 2, 'right'), 'L. Messi': (2, -8, 'center'), 'P. Aubameyang': (4, -3, 'left'),
             'S. Agüero': (-4, -3, 'right'), 'D. Zapata': (4, 3, 'left'), 'Maxi Gómez': (-3, 3, 'right'), 'H. Kane': (0, 5, 'center')}
    for name, (dx, dy, ha) in label.items():
        r = g[g['name'] == name]
        if len(r):
            r = r.iloc[0]
            ax.scatter(r['np_goals'], r['goal_points'], s=20, color=BLUE, edgecolor='white', lw=0.5, zorder=3)
            ax.annotate(name.replace('Mohamed ', '').replace('Cristiano ', 'C. '), (r['np_goals'], r['goal_points']), xytext=(dx, dy), textcoords='offset points', fontsize=5.8, ha=ha, color=INK)
    ax.set_xlabel('Non-penalty goals')
    ax.set_ylabel('Points from those goals')
    ax.set_title('(b) Same goals, different points', fontsize=7.5)
    clean(ax, 'both')


def fig_ledger_goals():
    fig, axes = plt.subplots(1, 2, figsize=(4.8, 2.0))
    fig_ledger(axes[0])
    fig_goals(axes[1])
    plt.tight_layout(w_pad=1.2)
    save(fig, 'ledger_goals')


def fig_coaches():
    r = pd.read_csv(C.RESULTS / 'q2_coaches.csv')
    r = r[r['state'].isin(['protecting', 'chasing']) & r['term'].isin(['z_defend', 'z_attack'])]
    fig, axes = plt.subplots(1, 2, figsize=(4.8, 1.6), sharey=True)
    order = [('protecting', 'z_defend'), ('protecting', 'z_attack'), ('chasing', 'z_defend'), ('chasing', 'z_attack')]
    labels = ['One up, goal prevention', 'One up, attacking', 'One down, goal prevention', 'One down, attacking']
    for ax, (dec, title) in zip(axes, (('taken_off', 'Taken off'), ('brought_on', 'Brought on'))):
        for i, (st, term) in enumerate(order):
            x = r[(r['decision'] == dec) & (r['state'] == st) & (r['term'] == term)].iloc[0]
            colour = ORANGE if term == 'z_defend' else BLUE
            y = len(order) - 1 - i
            ax.plot(np.exp([x['ci_low'], x['ci_high']]), [y, y], color=colour, lw=1.4, solid_capstyle='round')
            ax.scatter(np.exp(x['log_odds']), y, s=20, color=colour, edgecolor='white', lw=0.5, zorder=3, marker='s' if term == 'z_defend' else 'o')
        ax.axvline(1, color=GREY, lw=0.8)
        ax.set_title(title)
        ax.set_xlim(0.72, 1.38)
        ax.set_xticks([0.8, 1.0, 1.2])
        clean(ax, 'x')
    axes[0].set_yticks(range(len(order)))
    axes[0].set_yticklabels(labels[::-1], fontsize=7)
    fig.supxlabel('Odds ratio per one SD of rating', fontsize=7.5, color=INK2, y=-0.08, x=0.62)
    save(fig, 'coaches')


if __name__ == '__main__':
    OUT.mkdir(parents=True, exist_ok=True)
    fig_stakes()
    fig_ledger_goals()
    fig_coaches()
    print('figures done')
