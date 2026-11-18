"""H1 rules: trade the side the state model (EMP, fitted on train outcomes) prefers when it deviates from the market mid.

Decision rows: 30-s panel, aligned games, ESPN info delay 15 s + game lag, market stable (|mid - mid 20 s ago| <= 2c), no
known score change in the last 30 s, 0.05 <= mid <= 0.95. Signal dev = p_model - mid. If dev >= theta buy home at the next
ask-side print (rt >= t+2 s), if dev <= -theta buy away at 1 - (next bid-side print). First trigger per game per rule.
Exits: hold to resolution / sell at the first opposite print >= t+300 s / >= t+600 s. 0.05 fee per taker leg.
Grid (fixed before looking at test): model {EMP, BM12.34} x theta {0.03, 0.05, 0.08, 0.12} x window {all, Q1-Q2, Q3-Q4(>2m), last 6m} x exit {hold, 5m, 10m}.
"""
import sys
import numpy as np, pandas as pd
LINFO = int(sys.argv[1]) if len(sys.argv) > 1 else 15
SFX = '' if LINFO == 15 else f'_L{LINFO}'
from features import load_panel, p_emp_rows, pnl_hold, pnl_exit2, exit_px, pnl_gross_hold
from common import summarize, RL, DL
p = load_panel(L=LINFO, align_filter=True)
p = p[p.mid.between(0.05, 0.95) & ((p.mid - p.mid_m20).abs() <= 0.02) & (p.since_score >= 30)].copy()
p['grp'] = np.where(p.season == '2024-25', 'tr24', np.where(p.split == 'train', 'tr25', 'test'))
p['p_emp'] = p_emp_rows(p.m.values, p.elapsed.values, p.mu.values)
p = p.sort_values(['game_id', 't'])
W = {'all': (0, 99999), 'Q1-Q2': (1440, 99999), 'Q3-Q4(>2m)': (120, 1440), 'last6m': (0, 360)}
trades = []
for model in ['p_emp', 'p_bm']:
    dev = p[model] - p.mid
    for th in [0.03, 0.05, 0.08, 0.12]:
        for wn, (lo, hi) in W.items():
            inwin = (p.rem >= lo) & (p.rem < hi) if wn != 'all' else np.ones(len(p), bool)
            if wn == 'last6m':
                inwin = inwin & ~p.ot
            x = p[(dev.abs() >= th) & inwin].groupby('game_id').head(1).copy()
            buyH = (x[model] - x.mid) > 0
            px = np.where(buyH, x.buyH_px, x.buyA_px)
            win = np.where(buyH, x.home_win, 1 - x.home_win)
            o5, f5, s5 = exit_px(x, buyH.values, 300)
            o10, f10, s10 = exit_px(x, buyH.values, 600)
            T = pd.DataFrame({'model': model, 'theta': th, 'window': wn, 'game_id': x.game_id.values, 'grp': x.grp.values, 'split': x.split.values,
                              'month': x.month.values, 'px': px, 'win': win, 'dev': (x[model] - x.mid).abs().values,
                              'pnl_hold': pnl_hold(px, win), 'gross_hold': pnl_gross_hold(px, win), 'pnl_5m': pnl_exit2(px, o5, f5), 'pnl_10m': pnl_exit2(px, o10, f10),
                              'src5': s5, 'src10': s10})
            trades.append(T[T.px.notna()])
T = pd.concat(trades, ignore_index=True)
T.to_parquet(DL / f'h1_trades{SFX}.parquet', index=False)
rows = []
for (model, th, wn, grp), d in T.groupby(['model', 'theta', 'window', 'grp']):
    r = dict(model=model, theta=th, window=wn, grp=grp, n=len(d), px=d.px.mean(), win=d.win.mean(), gross_hold=d.gross_hold.mean())
    for ex in ['pnl_hold', 'pnl_5m', 'pnl_10m']:
        s = summarize(d[d[ex].notna()], ex, n_boot=500)
        r[ex] = s.pnl_mean[0]; r[ex + '_lo'] = s.ci_lo[0]; r[ex + '_hi'] = s.ci_hi[0]
    rows.append(r)
R = pd.DataFrame(rows); R.to_csv(RL / f'h1_rules{SFX}.csv', index=False)
pd.set_option('display.width', 250)
for ex in ['pnl_hold', 'pnl_5m', 'pnl_10m']:
    V = R.pivot_table(index=['model', 'theta', 'window'], columns='grp', values=ex)
    print('\n==', ex, '(USDC per 10 USDC trade)'); print(V.round(3).to_string())
print('\nn per cell'); print(R.pivot_table(index=['model', 'theta', 'window'], columns='grp', values='n').to_string())
