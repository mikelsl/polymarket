"""H2: calibration of in-play moneyline prices by price bucket x time remaining (pure price rules, hold to resolution).

A) descriptive: every 30-s panel row in play; favourite = side with observed mid >= 0.5. Win rate vs mid and vs the
   executable ask (first ask-side print >= t+2 s, real time), P&L per 10 USDC after the 0.05 fee (and gross),
   bootstrap CI clustered by game. Also the underdog side.
B) strategies: first row per game inside (time window x favourite-price band) -> buy that side at the next ask print,
   hold to resolution. Grid fixed in advance (listed below); train/test separately.
Outputs: results/nba_live/h2_desc.csv, h2_rules.csv, h2_trades.parquet (data/nba_live)
"""
import numpy as np
import pandas as pd
from common import RL, DL, summarize, fee_ps
from features import load_panel, pnl_hold, pnl_gross_hold

p = load_panel(L=15, align_filter=False)
p = p[p.mid.notna()].copy()
p['fav_home'] = p.mid >= 0.5
p['fav_mid'] = np.where(p.fav_home, p.mid, 1 - p.mid)
p['fav_px'] = np.where(p.fav_home, p.buyH_px, p.buyA_px)
p['dog_px'] = np.where(p.fav_home, p.buyA_px, p.buyH_px)
p['fav_win'] = np.where(p.fav_home, p.home_win, 1 - p.home_win)
PB = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.97, 0.99, 1.0001]
p['pb'] = pd.cut(p.fav_mid, PB, right=False).astype(str)
p['pnl_fav'] = pnl_hold(p.fav_px, p.fav_win)
p['pnl_fav_gross'] = pnl_gross_hold(p.fav_px, p.fav_win)
p['pnl_dog'] = pnl_hold(p.dog_px, 1 - p.fav_win)
p['res_mid'] = p.fav_win - p.fav_mid
ok = p.fav_px.between(0.01, 0.995)
rows = []
for sp in ['train', 'test']:
    d = p[(p.split == sp) & ok]
    for (tb, pb), x in d.groupby(['tb', 'pb'], observed=True):
        if len(x) < 50:
            continue
        s = summarize(x, 'pnl_fav', n_boot=500)
        sg = summarize(x, 'pnl_fav_gross', n_boot=200)
        sr = summarize(x, 'res_mid', n_boot=500)
        dg = x[x.dog_px.between(0.005, 0.99)]
        sd = summarize(dg, 'pnl_dog', n_boot=200) if len(dg) else None
        rows.append(dict(split=sp, tb=tb, pb=pb, rows=len(x), games=x.game_id.nunique(), fav_mid=x.fav_mid.mean(), fav_ask=x.fav_px.mean(),
                         win=x.fav_win.mean(), res_mid=sr.pnl_mean[0], res_lo=sr.ci_lo[0], res_hi=sr.ci_hi[0],
                         fav_pnl=s.pnl_mean[0], fav_lo=s.ci_lo[0], fav_hi=s.ci_hi[0], fav_gross=sg.pnl_mean[0],
                         dog_pnl=(sd.pnl_mean[0] if sd is not None and len(sd) else np.nan)))
D = pd.DataFrame(rows)
D.to_csv(RL / 'h2_desc.csv', index=False)
pd.set_option('display.width', 250)
print(D.round(4).to_string())

# ---- B) pre-registered first-trigger rules ----
WINDOWS = {'Q1': (2160, 2880), 'Q2': (1440, 2160), 'Q3': (720, 1440), 'Q4a(6-12m)': (360, 720), 'Q4b(2-6m)': (120, 360), 'Q4c(0-2m)': (0, 120)}
BANDS = [(0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 0.95), (0.95, 0.98), (0.98, 0.995)]
trades = []
q = p[ok & ~p.ot].sort_values(['game_id', 't'])
for wn, (lo, hi) in WINDOWS.items():
    w = q[(q.rem >= lo) & (q.rem < hi)]
    for blo, bhi in BANDS:
        for side in ['fav', 'dog']:
            if side == 'fav':
                x = w[(w.fav_mid >= blo) & (w.fav_mid < bhi)]
            else:   # underdog bought when the favourite is in the band
                x = w[(w.fav_mid >= blo) & (w.fav_mid < bhi) & w.dog_px.between(0.005, 0.99)]
            x = x.groupby('game_id').head(1)
            px = x.fav_px if side == 'fav' else x.dog_px
            win = x.fav_win if side == 'fav' else 1 - x.fav_win
            trades.append(pd.DataFrame({'rule': f'{side}|{wn}|{blo:.3f}-{bhi:.3f}', 'side': side, 'window': wn, 'band': f'{blo:.3f}-{bhi:.3f}',
                                        'game_id': x.game_id.values, 'split': x.split.values, 'season': x.season.values, 'month': x.month.values,
                                        't': x.t.values, 'px': px.values, 'win': win.values, 'pnl': pnl_hold(px.values, win.values),
                                        'pnl_gross': pnl_gross_hold(px.values, win.values)}))
T = pd.concat(trades, ignore_index=True)
T.to_parquet(DL / 'h2_trades.parquet', index=False)
R = summarize(T, 'pnl', by=['rule', 'split'], n_boot=1000)
Rg = T.groupby(['rule', 'split']).agg(px=('px', 'mean'), win=('win', 'mean'), gross=('pnl_gross', 'mean')).reset_index()
R = R.merge(Rg, on=['rule', 'split'])
R.to_csv(RL / 'h2_rules.csv', index=False)
W = R.pivot(index='rule', columns='split', values=['n', 'px', 'win', 'pnl_mean', 'ci_lo', 'ci_hi'])
print(W.round(3).to_string())
print('variants', T.rule.nunique())
