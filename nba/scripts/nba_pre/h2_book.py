"""H2 - Polymarket moneyline vs sportsbook (ESPN odds: ESPN BET 2024-25 and early 2025-26, DraftKings after).

1. Information: logistic y_home ~ logit(PM) + logit(book) and Brier scores, PM at several times vs book open / close.
2. Value rule: at entry time E buy the Polymarket token whose ask is below the de-vigged book probability by >= d
   (one trade per game: the side with the larger edge). Fee 0.05 p (1-p), 10 USDC, hold to resolution.
   Timing (look-ahead control):
     close@m1 : PM at tip - 1 min vs book CLOSE (close is the last pre-game line; minimal look-ahead)
     close@s0 : PM at the scheduled start vs book close (look-ahead: the close can move until the real tip, ~11 min)
     open@s0, open@m60, open@h3 : PM vs book OPEN (no look-ahead if the open was posted before E; the open time is
                not in the data - ESPN opens are normally posted the day before / morning of the game)
   De-vig: multiplicative (main) and power. Also the mirrored rule (buy the side where PM is ABOVE the book).
Outputs: h2_info.csv, h2_gap_dist.csv, h2_value_rules.csv
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm

from common_pre import DATA, RES, ask_of_mid, fmt, summarize, trade_pnl

p = pd.read_parquet(DATA / 'ml_panel.parquet')
p['pmid0_h3x'] = p.pmid0_h3
lg = lambda x: np.log(np.clip(x, 0.005, 0.995) / (1 - np.clip(x, 0.005, 0.995)))  # noqa: E731

# ---------------- 1. information ----------------
info = []
for pm_col, bk_col in [('mid0_m1', 'pbook0_close'), ('mid0_s0', 'pbook0_close'), ('mid0_m60', 'pbook0_close'),
                       ('mid0_m1', 'pbook0_open'), ('mid0_s0', 'pbook0_open'), ('mid0_m300', 'pbook0_open'),
                       ('pmid0_h24', 'pbook0_open')]:
    d = p.dropna(subset=[pm_col, bk_col])
    for sp, x in [('train', d[d.split == 'train']), ('test', d[d.split == 'test']), ('all', d)]:
        X = pd.DataFrame({'pm': lg(x[pm_col]), 'book': lg(x[bk_col])})
        m = sm.Logit(x.y0, sm.add_constant(X)).fit(disp=0)
        info.append(dict(pm=pm_col, book=bk_col, split=sp, n=len(x), b_pm=m.params['pm'], se_pm=m.bse['pm'],
                         b_book=m.params['book'], se_book=m.bse['book'],
                         brier_pm=np.mean((x[pm_col] - x.y0) ** 2), brier_book=np.mean((x[bk_col] - x.y0) ** 2),
                         mean_gap=np.mean(x[pm_col] - x[bk_col]), mad_gap=np.mean(np.abs(x[pm_col] - x[bk_col]))))
pd.DataFrame(info).pipe(fmt, 4).to_csv(RES / 'h2_info.csv', index=False)

# ---------------- 2. value rules ----------------
setups = {'close@m1': ('mid0_m1', 'pbook0_close'), 'close@s0': ('mid0_s0', 'pbook0_close'),
          'open@s0': ('mid0_s0', 'pbook0_open'), 'open@m60': ('mid0_m60', 'pbook0_open'),
          'open@m300': ('mid0_m300', 'pbook0_open')}
rows, gaps = [], []
for name, (pm_col, bk_col) in setups.items():
    for dv in ('', '_pow'):
        d = p.dropna(subset=[pm_col, bk_col + dv]).copy()
        mid0, fair0 = d[pm_col].to_numpy(), d[bk_col + dv].to_numpy()
        a0, a1 = ask_of_mid(mid0), ask_of_mid(1 - mid0)
        e0, e1 = fair0 - a0, (1 - fair0) - a1        # edge of buying token 0 / token 1 at the ask
        for mirror in (False, True):
            if mirror:  # buy the side the book likes LESS than Polymarket (i.e. trust Polymarket)
                e0m, e1m = (a0 - 0.01) - fair0, (a1 - 0.01) - (1 - fair0)
                take1 = e1m > e0m
                edge = np.where(take1, e1m, e0m)
            else:
                take1 = e1 > e0
                edge = np.where(take1, e1, e0)
            t = pd.DataFrame({'game_id': d.game_id, 'split': d.split, 'season': d.season, 'edge': edge,
                              'price': np.where(take1, a1, a0), 'y': np.where(take1, 1 - d.y0, d.y0)})
            t['pnl'], t['cost'], t['gross'] = trade_pnl(t.price, t.y)
            if not mirror and dv == '':
                gaps.append(t.groupby('split').edge.describe(percentiles=[.5, .9, .95, .99]).assign(setup=name))
            for thr in (-0.01, 0.0, 0.005, 0.01, 0.015, 0.02, 0.03, 0.04):
                s = summarize(t[t.edge >= thr], by='split', B=1000)
                rows.append(s.assign(setup=name, devig=dv or '_mult', mirror=mirror, min_edge=thr))
pd.concat(gaps).reset_index().pipe(fmt, 4).to_csv(RES / 'h2_gap_dist.csv', index=False)
R = pd.concat(rows)
R = R[['setup', 'devig', 'mirror', 'min_edge', 'split', 'n', 'games', 'avg_price', 'win', 'pnl', 'pnl_per_trade',
       'roi', 'roi_lo', 'roi_hi', 'gross_pnl', 'gross_roi']]
R.pipe(fmt, 4).to_csv(RES / 'h2_value_rules.csv', index=False)
print('variants:', R.groupby(['setup', 'devig', 'mirror', 'min_edge']).ngroups)
