"""Per-game ESPN-vs-market alignment flags from alignment.parquet (last-print cross-correlation, real print time).

align_ok: whole-game lag in [-20, 30] s with peak correlation >= 0.15, and each half either in [-20, 30] s or weakly
identified (peak < 0.2). lag_used: the whole-game lag for ok games, else the season median. ESPN information is used
only from wc + lag_used + L (L = info delay) so the bot never acts before the market has seen the play.
"""
import numpy as np
import pandas as pd
from common import DL, load_games

A = pd.read_parquet(DL / 'alignment.parquet')
g = load_games()
A = A.merge(g[['game_id', 'season']], on='game_id', how='right')
ok_game = A.lag_local.between(-20, 30) & (A.peak_local >= 0.15)
ok_h1 = A.lag_q1.between(-20, 30) | (A.peak_q1 < 0.2)
ok_h2 = A.lag_q2.between(-20, 30) | (A.peak_q2 < 0.2)
A['align_ok'] = ok_game & ok_h1 & ok_h2
med = A[A.align_ok].groupby('season').lag_local.median()
A['lag_used'] = np.where(A.align_ok, A.lag_local, A.season.map(med))
A[['game_id', 'season', 'lag_local', 'peak_local', 'lag_q1', 'peak_q1', 'lag_q2', 'peak_q2', 'align_ok', 'lag_used']].to_parquet(DL / 'alignment_flags.parquet', index=False)
print(A.groupby('season').agg(games=('game_id', 'size'), ok=('align_ok', 'sum'), lag_med=('lag_used', 'median')))
