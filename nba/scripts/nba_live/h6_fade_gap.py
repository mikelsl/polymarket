"""H6c: fade large in-play departures from the pre-game price (anchoring / early overreaction).
Rule: in window W (by ESPN remaining time, 15 s info delay + game lag), the first panel time where |pre-game price - mid| >= G
-> buy the side whose price FELL relative to pre-game (toward the pre-game view) at the next ask print (rt >= t + 2 s),
hold to resolution (also: exit after 10 min). One trade per game per rule. All games except print-gap games.
Grid: W in {Q1 first 6m, Q1, Q2, Q3, Q4 (>2m)} x G in {0.15, 0.20, 0.25, 0.30} (20 variants x 2 exits)."""
import numpy as np, pandas as pd
from features import load_panel, pnl_hold, pnl_gross_hold, exit_px, pnl_exit2
from common import summarize, RL, DL
p = load_panel(L=15, align_filter=False)
p = p[p.mid.between(0.03, 0.97)].sort_values(['game_id', 't']).copy()
p['grp'] = np.where(p.season == '2024-25', 'tr24', np.where(p.split == 'train', 'tr25', 'test'))
p['gap'] = p.pre_ph - p.mid
W = {'Q1a(0-6m)': (2520, 2881), 'Q1': (2160, 2881), 'Q2': (1440, 2160), 'Q3': (720, 1440), 'Q4(>2m)': (120, 720)}
T = []
for wn, (lo, hi) in W.items():
    for G in (0.15, 0.20, 0.25, 0.30):
        x = p[(~p.ot) & (p.rem >= lo) & (p.rem < hi) & (p.gap.abs() >= G)].groupby('game_id').head(1).copy()
        bh = (x.gap > 0).values          # home price fell below pre-game -> buy home
        px = np.where(bh, x.buyH_px, x.buyA_px)
        win = np.where(bh, x.home_win, 1 - x.home_win)
        o10, f10, _ = exit_px(x, bh, 600)
        T.append(pd.DataFrame({'window': wn, 'G': G, 'game_id': x.game_id.values, 'grp': x.grp.values, 'split': x.split.values, 'month': x.month.values,
                               'season': x.season.values, 'gap': x.gap.abs().values, 'mid_own': np.where(bh, x.mid, 1 - x.mid), 'px': px, 'win': win,
                               'pnl_hold': pnl_hold(px, win), 'gross_hold': pnl_gross_hold(px, win), 'pnl_10m': pnl_exit2(px, o10, f10)}))
T = pd.concat(T, ignore_index=True); T = T[T.px.notna()]
T.to_parquet(DL / 'h6_fade_gap_trades.parquet', index=False)
rows = []
for (wn, G, grp), d in T.groupby(['window', 'G', 'grp']):
    s = summarize(d, 'pnl_hold', n_boot=1000); s2 = summarize(d, 'pnl_10m', n_boot=500)
    rows.append(dict(window=wn, G=G, grp=grp, n=len(d), px=d.px.mean(), win=d.win.mean(), gross=d.gross_hold.mean(), pnl_hold=s.pnl_mean[0], lo=s.ci_lo[0], hi=s.ci_hi[0],
                     pnl_10m=s2.pnl_mean[0], lo10=s2.ci_lo[0], hi10=s2.ci_hi[0]))
O = pd.DataFrame(rows); O.to_csv(RL / 'h6_fade_gap.csv', index=False)
pd.set_option('display.width', 250)
print(O.round(3).to_string())
