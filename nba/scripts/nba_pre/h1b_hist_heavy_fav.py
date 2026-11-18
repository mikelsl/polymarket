"""H1b - independent check of the heavy-favourite lead on seasons BEFORE the Polymarket data (2022-23, 2023-24),
using ESPN closing sportsbook moneylines (power de-vig), plus the same statistic for 2024-25 / 2025-26 / 2026-27
(ESPN odds of the core games) for comparison. Polymarket's pre-game price tracks the book close within ~1c
(h2_info.csv), so a stable book bias at the extremes would carry over.
Output: h1b_hist_heavy_fav.csv (favourite de-vigged prob buckets x period: n, mean prob, win rate, z)
"""
import numpy as np
import pandas as pd

from common_pre import DATA, RES, american_to_prob, devig_power

h = pd.read_parquet(DATA / 'espn_hist_odds.parquet')
h = h[h.ml_home.notna() & h.ml_away.notna() & ~h.provider.fillna('').str.contains('Live')]
prio = ['DraftKings', 'ESPN BET', 'Caesars Sportsbook (New Jersey)', 'MGM', 'BetMGM', 'FanDuel']
h['prio'] = h.provider.map({p: i for i, p in enumerate(prio)}).fillna(99)
h = h.sort_values(['event', 'prio']).drop_duplicates('event')
h['p_home'] = devig_power(american_to_prob(h.ml_home.astype(float)), american_to_prob(h.ml_away.astype(float)))
h['y_home'] = (h.home_score > h.away_score).astype(float)
h['month'] = h.date.str[5:7]
h['period'] = h.season
p = pd.read_parquet(DATA / 'ml_panel.parquet')
q = pd.DataFrame({'p_home': np.where(p.home_idx == 0, p.pbook0_close_pow, 1 - p.pbook0_close_pow),
                  'y_home': np.where(p.home_idx == 0, p.y0, 1 - p.y0), 'month': p.month.str[5:7],
                  'period': p.season + ' (PM era)', 'provider': p.book_src,
                  'pm_home': np.where(p.home_idx == 0, p.mid0_s0, 1 - p.mid0_s0)})
a = pd.concat([h[['p_home', 'y_home', 'month', 'period', 'provider']], q], ignore_index=True).dropna(subset=['p_home'])
a['pf'] = np.maximum(a.p_home, 1 - a.p_home)
a['yf'] = np.where(a.p_home >= 0.5, a.y_home, 1 - a.y_home)
a['pmf'] = np.where(a.p_home >= 0.5, a.pm_home, 1 - a.pm_home)
a['late'] = np.where(a.month.isin(['03', '04']), 'Mar-Apr', 'other months')
rows = []
for lo, hi in ((0.80, 0.85), (0.85, 0.90), (0.90, 1.0), (0.93, 1.0)):
    for (per, late), d in a[(a.pf >= lo) & (a.pf < hi)].groupby(['period', 'late']):
        n = len(d)
        rows.append(dict(band=f'{lo}-{hi}', period=per, months=late, n=n, book_prob=d.pf.mean(), win=d.yf.mean(),
                         pm_price=d.pmf.mean(), z=(d.yf.mean() - d.pf.mean()) / np.sqrt(d.pf.mean() * (1 - d.pf.mean()) / n)))
    for per, d in a[(a.pf >= lo) & (a.pf < hi)].groupby('period'):
        n = len(d)
        rows.append(dict(band=f'{lo}-{hi}', period=per, months='all', n=n, book_prob=d.pf.mean(), win=d.yf.mean(),
                         pm_price=d.pmf.mean(), z=(d.yf.mean() - d.pf.mean()) / np.sqrt(d.pf.mean() * (1 - d.pf.mean()) / n)))
out = pd.DataFrame(rows).sort_values(['band', 'period', 'months'])
out.round(4).to_csv(RES / 'h1b_hist_heavy_fav.csv', index=False)
pd.set_option('display.width', 200)
print(h.provider.value_counts().head(6).to_dict())
print(out[out.band.isin(['0.9-1.0', '0.85-0.9'])].round(3).to_string(index=False))
