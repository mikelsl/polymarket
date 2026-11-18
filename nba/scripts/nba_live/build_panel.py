"""Decision-point panel: every STEP seconds of real time from tip+60 s to the last ESPN play of each core game.

For decision time t (real time; print times already shifted rt = ts - 4 s):
  ESPN state known at t - L - lag_game (L = 15 s, also 45 s; lag_game = per-game market-vs-ESPN lag, alignment_flags): period, clock, elapsed, scores, seconds since last score change
  market observed at t: last ask-side / bid-side print (home terms) at rt <= t, mid, staleness; mids at t-60/-180/-300
  fills: first ask-side print at rt >= t + 2 (buy home) and first bid-side print at rt >= t + 2 (buy away = 1 - ph), <= 60 s
  exits: first bid-side / ask-side print at rt >= t + h (h = 60, 180, 300, 600 s); fair mid ("pm") at t + h
  outcome: home_win
Output: data/nba_live/panel.parquet
"""
import sys
import time
import numpy as np
import pandas as pd
from common import DL, load_games, load_prints, load_states, game_slices, first_at_or_after, last_at_or_before

STEP = float(sys.argv[1]) if len(sys.argv) > 1 else 30.0
LS = (15.0, 30.0, 45.0)
HS = (60, 180, 300, 600)
REACT = 2.0

t0 = time.time()
g = load_games()
P = load_prints()
S = load_states()
psl, ssl = game_slices(P), game_slices(S)
AL = pd.read_parquet(DL / 'alignment_flags.parquet').set_index('game_id')
out = []
for gi, r in enumerate(g.itertuples(index=False)):
    gid = r.game_id
    if gid not in psl or gid not in ssl or np.isnan(r.tip):
        continue
    a, b = psl[gid]
    rt, ph, dd, sz = P.rt.values[a:b], P.ph.values[a:b], P.d.values[a:b], P['size'].values[a:b]
    A_t, A_p = rt[dd == 1], ph[dd == 1]
    B_t, B_p = rt[dd == -1], ph[dd == -1]
    a2, b2 = ssl[gid]
    wc = S.wc.values[a2:b2]
    kp, kc, ke = S.k_period.values[a2:b2], S.k_clock_s.values[a2:b2], S.k_elapsed_s.values[a2:b2]
    kh, ka = S.k_home_score.values[a2:b2].astype(np.int32), S.k_away_score.values[a2:b2].astype(np.int32)
    # time of last score change (in wallclock order of known states)
    tot = kh + ka
    chg = np.r_[True, tot[1:] != tot[:-1]]
    last_chg_t = np.where(chg, wc, np.nan)
    last_chg_t = pd.Series(last_chg_t).ffill().values
    lagg = float(AL.lag_used.get(gid, 0.0)) if gid in AL.index else 0.0
    grid = np.arange(r.tip + 60.0, r.end_wc + 1.0, STEP)
    if len(grid) == 0:
        continue
    D = {'game_id': np.full(len(grid), gid, dtype=object), 't': grid}
    for L in LS:
        ix = np.searchsorted(wc, grid - L - lagg, side='right') - 1
        ok = ix >= 0
        ix = np.clip(ix, 0, None)
        sfx = f'_L{int(L)}'
        D['period' + sfx] = np.where(ok, kp[ix], 0).astype(np.int16)
        D['clock' + sfx] = np.where(ok, kc[ix], 720.0).astype(np.float32)
        D['elapsed' + sfx] = np.where(ok, ke[ix], 0.0).astype(np.float32)
        D['hs' + sfx] = np.where(ok, kh[ix], 0).astype(np.int16)
        D['as' + sfx] = np.where(ok, ka[ix], 0).astype(np.int16)
        D['since_score' + sfx] = np.where(ok, grid - lagg - last_chg_t[ix], np.nan).astype(np.float32)
    # market observed at t
    ta, pa = last_at_or_before(A_t, A_p, grid, within=180)
    tb, pb = last_at_or_before(B_t, B_p, grid, within=180)
    D['ask_last'] = pa; D['bid_last'] = pb
    D['ask_age'] = grid - ta; D['bid_age'] = grid - tb
    mid = np.where(np.isnan(pa) | np.isnan(pb), np.where(np.isnan(pa), pb, pa), (pa + pb) / 2)
    D['mid'] = mid
    for back in (20, 60, 180, 300):
        ta2, pa2 = last_at_or_before(A_t, A_p, grid - back, within=180)
        tb2, pb2 = last_at_or_before(B_t, B_p, grid - back, within=180)
        D[f'mid_m{back}'] = np.where(np.isnan(pa2) | np.isnan(pb2), np.where(np.isnan(pa2), pb2, pa2), (pa2 + pb2) / 2)
    # fills
    fa_t, fa_p = first_at_or_after(A_t, A_p, grid + REACT, within=60)
    fb_t, fb_p = first_at_or_after(B_t, B_p, grid + REACT, within=60)
    D['buyH_px'] = fa_p; D['buyH_wait'] = fa_t - grid
    D['buyA_px'] = 1 - fb_p; D['buyA_wait'] = fb_t - grid
    # print volume (shares) in the 60 s after t+2 at ask side / bid side, for depth checks
    for h in HS:
        xb_t, xb_p = first_at_or_after(B_t, B_p, grid + h, within=120)   # sell home at a bid-side print
        xa_t, xa_p = first_at_or_after(A_t, A_p, grid + h, within=120)   # sell away = ask-side print in home terms
        D[f'sellH_px_{h}'] = xb_p
        D[f'sellA_px_{h}'] = 1 - xa_p
        # fair mid at t+h: mean of last mid before and first mid after
        la_t, la_p = last_at_or_before(A_t, A_p, grid + h, within=180)
        lb_t, lb_p = last_at_or_before(B_t, B_p, grid + h, within=180)
        m_before = np.where(np.isnan(la_p) | np.isnan(lb_p), np.where(np.isnan(la_p), lb_p, la_p), (la_p + lb_p) / 2)
        m_after = np.where(np.isnan(xa_p) | np.isnan(xb_p), np.where(np.isnan(xa_p), xb_p, xa_p), (xa_p + xb_p) / 2)
        D[f'pm_{h}'] = np.where(np.isnan(m_after), m_before, np.where(np.isnan(m_before), m_after, (m_before + m_after) / 2))
    D['home_win'] = np.full(len(grid), r.home_win, dtype=np.int8)
    out.append(pd.DataFrame(D))
    if gi % 250 == 0:
        print(gi, round(time.time() - t0, 1), flush=True)
panel = pd.concat(out, ignore_index=True)
for c in panel.columns:
    if panel[c].dtype == np.float64 and c not in ('t',):
        panel[c] = panel[c].astype(np.float32)
name = 'panel.parquet' if STEP == 30 else f'panel_{int(STEP)}s.parquet'
panel.to_parquet(DL / name, index=False)
print('rows', len(panel), 'games', panel.game_id.nunique(), round(time.time() - t0, 1), flush=True)
