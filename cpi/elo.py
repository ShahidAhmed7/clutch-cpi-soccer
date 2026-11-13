"""Pre-match Elo, used only to adjust real results in the validation tests. It never enters the stakes or the values."""
from collections import defaultdict

import numpy as np
import pandas as pd

K, HOME, START = 20.0, 100.0, 1500.0


def expected(r_team, r_opp, home):
    return 1.0 / (1.0 + 10 ** ((r_opp - r_team - (HOME if home else -HOME)) / 400.0))


def pre_match_elo(matches):
    """Walk each league in date order and record both teams' ratings before each match."""
    parts = []
    for _, league in matches.groupby('league'):
        ms = league.sort_values(['date', 'gameweek'], kind='stable').reset_index(drop=True)
        ratings = defaultdict(lambda: START)
        h_pre, a_pre = np.zeros(len(ms)), np.zeros(len(ms))
        for i, m in enumerate(ms.itertuples()):
            rh, ra = ratings[m.home_team_id], ratings[m.away_team_id]
            h_pre[i], a_pre[i] = rh, ra
            s = 1.0 if m.home_score > m.away_score else 0.0 if m.home_score < m.away_score else 0.5
            ratings[m.home_team_id] = rh + K * (s - expected(rh, ra, True))
            ratings[m.away_team_id] = ra + K * ((1 - s) - expected(ra, rh, False))
        parts.append(ms.assign(elo_home_pre=h_pre, elo_away_pre=a_pre))
    return pd.concat(parts, ignore_index=True)
