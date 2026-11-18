"""H4 - Polymarket spread / total lines (2025-26 and 2026-27 core games) vs the sportsbook and vs outcomes.

Side convention: spreads = token 'home covers L' (home margin + L > 0), totals = token 'over L'.
Model: margin ~ N(mu, s), mu = -book home spread; total ~ N(mu_T, s_T), mu_T = book total. s, s_T fitted on TRAIN
line outcomes by maximum likelihood (all lines with a price at the entry time).
Parts
  A calib     : PM price buckets of both sides of every line vs win rate (look for tail / longshot bias)
  B value     : buy the side (of any line) whose model prob - ask >= d; one trade per game and kind (largest edge)
                book CLOSE with entry tip - 1 min (minimal look-ahead), book OPEN with entry at sched / tip - 60
  C over/under: the line nearest 0.5 at tip - 60 (decision-time choice); buy over / buy under at tip - 1
  D tails     : buy the expensive side (price >= x) of far lines, hold to resolution (fee is small at extremes)
  E ML vs spread: PM moneyline vs the moneyline implied by the PM spread line nearest 0.5 (same time)
Fees 0.05 p (1-p), 10 USDC per trade, ask = next tick above the mid. Outputs h4_*.csv.
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.stats import norm

from common_pre import DATA, RES, ask_of_mid, fmt, summarize, trade_pnl

L = pd.read_parquet(DATA / 'lines_panel.parquet')
L = L[L.y_side.notna() & (L.created_at < L.tip)].copy()       # lines listed before the real tip only
L = L[L.n_chg_2h >= 1]          # drop lines whose minute quote never moved in the 2 h before tip (stale / default 0.5)
L['mu'] = np.where(L.kind == 'spreads', -L.spread_home_close, L.total_close)
L['mu_open'] = np.where(L.kind == 'spreads', -L.spread_home_open, L.total_open)
# spreads: P(home covers L) = Phi((mu + L)/s); totals: P(over L) = Phi((mu - L)/s)
L['z'] = np.where(L.kind == 'spreads', L.mu + L.L, L.mu - L.L)
L['z_open'] = np.where(L.kind == 'spreads', L.mu_open + L.L, L.mu_open - L.L)

# ---------------- sigma fit on train ----------------
# Game-level residuals of the final margin / total around the book CLOSE line, train games of all seasons
# (2024-25 + 2025-26 to 2026-01-31). The line-outcome MLE is also reported (weakly identified: most priced lines sit
# within 2 points of the book line).
gp = pd.read_parquet(DATA / 'ml_panel.parquet')
gp = gp[gp.split == 'train']
res_m = (gp.espn_home_score - gp.espn_away_score) + gp.spread_home_close
res_t = (gp.espn_home_score + gp.espn_away_score) - gp.total_close
sig = {'spreads': float(np.nanstd(res_m)), 'totals': float(np.nanstd(res_t))}
mle = {}
for kind in ('spreads', 'totals'):
    x = L[(L.kind == kind) & (L.split == 'train') & L.z.notna() & L.mid_side_m1.notna()]   # priced pre-game
    f = lambda s: -np.sum(x.y_side * norm.logcdf(x.z / s) + (1 - x.y_side) * norm.logcdf(-x.z / s))  # noqa: E731
    mle[kind] = minimize_scalar(f, bounds=(5, 40), method='bounded').x
print('sigma (train game residual SD):', {k: round(v, 2) for k, v in sig.items()},
      'line-outcome MLE:', {k: round(v, 2) for k, v in mle.items()})
pd.DataFrame({'sigma_used': sig, 'sigma_line_mle': mle}).to_csv(RES / 'h4_sigma.csv')
L['s'] = L.kind.map(sig)
L['q'] = norm.cdf(L.z / L.s)
L['q_open'] = norm.cdf(L.z_open / L.s)

# both sides of every line as 'tokens'
# Pricing: PRICE_MODE 'print' (main) = buy at max(next tick above the minute mid, last print-revealed ask of that
# token in the 15 min before entry), only lines with such a print (real, recent liquidity); 'mid' = tick above mid,
# lines with >= 1 print in the 2 h before tip. Labels without prints (m5, m120) fall back to 'mid'.
PRICE_MODE = 'print'


def tokens(E, mode=None):
    mode = mode or PRICE_MODE
    rows = []
    has_pq = f'pask_side_{E}' in L.columns
    for side in (1, 0):
        t = L[['game_id', 'market_id', 'kind', 'split', 'season', 'month', 'L', 'mu', 'q', 'q_open',
               'n_prints_2h', f'mid_side_{E}']].copy()
        t['side'] = side
        t['mid'] = t[f'mid_side_{E}'] if side else 1 - t[f'mid_side_{E}']
        t['qm'] = t.q if side else 1 - t.q
        t['qm_open'] = t.q_open if side else 1 - t.q_open
        t['y'] = L.y_side if side else 1 - L.y_side
        if has_pq:
            t['pask'] = L[f'pask_side_{E}'] if side else 1 - L[f'pbid_side_{E}']
        else:
            t['pask'] = np.nan
        rows.append(t.drop(columns=[f'mid_side_{E}']))
    t = pd.concat(rows, ignore_index=True).dropna(subset=['mid'])
    t = t[(t.mid > 0.005) & (t.mid < 0.995) & (t.n_prints_2h >= 1)]
    t['price'] = ask_of_mid(t.mid)
    if mode == 'print' and has_pq:
        t = t[t.pask.notna()]
        t['price'] = np.maximum(t.price, t.pask)
    t = t[t.price < 0.999]
    t['pnl'], t['cost'], t['gross'] = trade_pnl(t.price, t.y)
    return t


tk = tokens('m1')
# ---------------- A calibration ----------------
bins = [0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1]
tk['bucket'] = pd.cut(tk.mid, bins).astype(str)
A = tk.groupby(['kind', 'split', 'bucket']).agg(n=('y', 'size'), games=('game_id', 'nunique'), price=('mid', 'mean'),
                                                model=('qm', 'mean'), win=('y', 'mean')).reset_index()
A['win_minus_price'] = A.win - A.price
A['model_minus_price'] = A.model - A.price
A.pipe(fmt, 4).to_csv(RES / 'h4_calib.csv', index=False)

# ---------------- B value vs book ----------------
rows = []
for E, qcol in (('m1', 'qm'), ('m60', 'qm_open'), ('s0', 'qm_open'), ('m1', 'qm_open')):
    t = tokens(E)
    t['edge'] = t[qcol] - t.price
    for kind in ('spreads', 'totals'):
        for scope in ('near', 'all'):
            x = t[t.kind == kind]
            if scope == 'near':          # line within 3 points of the book line (home handicap = -mu)
                x = x[(x.L + x.mu).abs() <= 3] if kind == 'spreads' else x[(x.L - x.mu).abs() <= 3]
            best = x.sort_values('edge', ascending=False).drop_duplicates('game_id')
            for d in (0.0, 0.01, 0.02, 0.03, 0.05):
                s = summarize(best[best.edge >= d], by='split', B=500)
                rows.append(s.assign(entry=E, book=('close' if qcol == 'qm' else 'open'), kind=kind, scope=scope,
                                     min_edge=d))
B = pd.concat(rows)[['entry', 'book', 'kind', 'scope', 'min_edge', 'split', 'n', 'avg_price', 'win', 'pnl', 'roi',
                     'roi_lo', 'roi_hi', 'gross_roi']]
B.pipe(fmt, 4).to_csv(RES / 'h4_value.csv', index=False)

# ---------------- C over / under and home / away cover on the decision-time line ----------------
rows = []
t1 = tokens('m1')
for kind in ('totals', 'spreads'):
    x = L[(L.kind == kind) & L.mid_side_m60.notna() & (L.n_prints_2h >= 1)].copy()
    x['dist'] = (x.mid_side_m60 - 0.5).abs()
    sel = x.sort_values('dist').drop_duplicates('game_id')[['market_id']]
    for side in (1, 0):
        t = t1[t1.market_id.isin(sel.market_id) & (t1.side == side)]
        name = {('totals', 1): 'over', ('totals', 0): 'under', ('spreads', 1): 'home_cover',
                ('spreads', 0): 'away_cover'}[(kind, side)]
        rows.append(summarize(t, by='split', B=500).assign(rule=name))
        rows.append(summarize(t, by='season', B=500).rename(columns={'season': 'split'}).assign(rule=name))
C = pd.concat(rows)[['rule', 'split', 'n', 'avg_price', 'win', 'pnl', 'roi', 'roi_lo', 'roi_hi', 'gross_roi']]
C.pipe(fmt, 4).to_csv(RES / 'h4_overunder.csv', index=False)

# ---------------- D tails: buy the expensive side ----------------
rows = []
for E in ('m1', 's0'):
    t = tokens(E)
    for kind in ('spreads', 'totals', 'both'):
        x = t if kind == 'both' else t[t.kind == kind]
        for lo, hi in ((0.80, 0.90), (0.90, 0.95), (0.95, 0.99), (0.85, 0.99), (0.01, 0.10), (0.10, 0.20)):
            s = summarize(x[(x.mid >= lo) & (x.mid < hi)], by='split', B=500)
            rows.append(s.assign(entry=E, kind=kind, band=f'{lo}-{hi}'))
D = pd.concat(rows)[['entry', 'kind', 'band', 'split', 'n', 'games', 'avg_price', 'win', 'pnl', 'roi', 'roi_lo',
                     'roi_hi', 'gross_roi']]
D.pipe(fmt, 4).to_csv(RES / 'h4_tails.csv', index=False)

# ---------------- E moneyline vs PM spread ----------------
rows = []
x = L[(L.kind == 'spreads') & L.mid_side_m1.notna() & (L.n_prints_2h >= 1)].copy()
x['dist'] = (x.mid_side_m1 - 0.5).abs()
sel = x.sort_values('dist').drop_duplicates('game_id')
sel = sel[sel.dist <= 0.15]
s_sp = sig['spreads']
sel['mu_pm'] = s_sp * norm.ppf(sel.mid_side_m1.clip(0.01, 0.99)) - sel.L
pan = pd.read_parquet(DATA / 'ml_panel.parquet')[['game_id', 'home_idx', 'mid0_m1', 'y0']]
sel = sel.drop(columns=['mid0_m1'], errors='ignore').merge(pan, on='game_id', suffixes=('', '_p'))
sel['p_home_ml'] = np.where(sel.home_idx == 0, sel.mid0_m1, 1 - sel.mid0_m1)
sel['y_home'] = np.where(sel.home_idx == 0, sel.y0, 1 - sel.y0)
tr = sel[sel.split == 'train']
f = lambda s: np.mean((norm.cdf(tr.mu_pm / s) - tr.p_home_ml) ** 2)  # noqa: E731
s_ml = minimize_scalar(f, bounds=(6, 30), method='bounded').x
sel['p_home_impl'] = norm.cdf(sel.mu_pm / s_ml)
for side in (1, 0):
    t = sel[['game_id', 'split', 'season']].copy()
    t['mid'] = sel.p_home_ml if side else 1 - sel.p_home_ml
    t['impl'] = sel.p_home_impl if side else 1 - sel.p_home_impl
    t['y'] = sel.y_home if side else 1 - sel.y_home
    t['price'] = ask_of_mid(t.mid)
    t['edge'] = t.impl - t.price
    t['pnl'], t['cost'], t['gross'] = trade_pnl(t.price, t.y)
    rows.append(t)
t = pd.concat(rows).sort_values('edge', ascending=False).drop_duplicates('game_id')
out = []
for d in (0.0, 0.01, 0.02, 0.03):
    out.append(summarize(t[t.edge >= d], by='split', B=500).assign(min_edge=d))
E_ = pd.concat(out)
E_['sigma_ml'] = s_ml
E_.pipe(fmt, 4).to_csv(RES / 'h4_ml_vs_spread.csv', index=False)
print('sigma_ml', round(s_ml, 2), 'mean |impl - ml|', round(float(np.mean(np.abs(sel.p_home_impl - sel.p_home_ml))), 4))

# ---------------- F ladder consistency / static arbitrage ----------------
# For two lines of one game with L1 < L2 (spreads: home handicap; totals: line, using 'under' so that the covered set
# grows with L), P(side covers L1) <= P(side covers L2). If bid(L1) - ask(L2) exceeds both taker fees, buying
# 'covers L2' + 'not covers L1' pays >= 1 for less than 1. Also the moneyline must sit between the nearest spread
# lines below / above 0. Uses minute mids (bid/ask = tick below/above): an upper bound on what was available.
from common_pre import bid_of_mid, fee  # noqa: E402

arb = []
for E in ('m1', 'm60', 's0'):
    x = L[L[f'mid_side_{E}'].notna() & (L.n_prints_2h >= 3)].copy()
    sp_ = x.kind == 'spreads'
    # P(cover) increasing in L: spreads 'home covers', totals 'under'; conservative quotes (worse of mid / print)
    mid = np.where(sp_, x[f'mid_side_{E}'], 1 - x[f'mid_side_{E}'])
    pa_ = np.where(sp_, x[f'pask_side_{E}'], 1 - x[f'pbid_side_{E}'])
    pb_ = np.where(sp_, x[f'pbid_side_{E}'], 1 - x[f'pask_side_{E}'])
    # executable quotes = print-revealed ask / bid in the 15 min before E, kept only when consistent with the minute
    # mid (within 3c); minute mids alone are not executable on thin lines (a 0.01/0.99 book has a 0.50 'mid')
    x['ask'] = np.where(np.abs(pa_ - mid) <= 0.03, np.fmax(pa_, ask_of_mid(mid)), np.nan)
    x['bid'] = np.where(np.abs(pb_ - mid) <= 0.03, np.fmin(pb_, bid_of_mid(mid)), np.nan)
    x['pc'] = mid
    for (gid, kind), d in x.groupby(['game_id', 'kind']):
        d = d.sort_values('L')
        Ls, bb, aa, pc = d.L.to_numpy(), d.bid.to_numpy(), d.ask.to_numpy(), d.pc.to_numpy()
        for i in range(len(d)):
            for j in range(i + 1, len(d)):
                if Ls[j] <= Ls[i] or not (bb[i] - aa[j] > 0):
                    continue
                net = bb[i] - aa[j] - fee(aa[j]) - fee(1 - bb[i])
                arb.append(dict(entry=E, game_id=gid, kind=kind, split=d.split.iloc[0], L1=Ls[i], L2=Ls[j],
                                p1=pc[i], p2=pc[j], gross=bb[i] - aa[j], net=net))
A_ = pd.DataFrame(arb)
A_.to_csv(RES / 'h4_ladder_violations.csv', index=False)
if len(A_):
    print(A_.groupby(['entry', 'kind']).agg(n=('net', 'size'), games=('game_id', 'nunique'),
                                             net_pos=('net', lambda s: (s > 0).sum()),
                                             max_net=('net', 'max')).to_string())
