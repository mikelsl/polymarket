"""H3: after a large in-play price swing, does the moneyline keep going (momentum) or revert (overshoot)?

Signal (market-only, no ESPN timing): on a 5-s grid from tip to the last ESPN play, mid(t) = mean of the last ask-side and
last bid-side print (real time, each <= 120 s old). Swing = mid(t) - mid(t - W). Signal when |swing| >= X and
0.10 <= mid(t) <= 0.90; cooldown 300 s per game and (W, X).
Entry D seconds after the signal (D = 10, 30) at the first print on the side we buy (rt >= t + D, within 60 s):
  follow = buy the side that rose; fade = buy the side that fell.
Exit: first opposite-side print at rt >= entry + h (h = 60, 180, 300, 600 s; resolution if the game is over by then;
fair mid - 1c if no print within 120 s), or hold to resolution. 0.05 fee per taker leg; gross also kept.
Also the descriptive drift: signed change of the print mid from entry time to entry + h.
Output: data/nba_live/h3_events.parquet, results/nba_live/h3_summary.csv
"""
import numpy as np
import pandas as pd
from common import DL, RL, load_games, load_prints, game_slices, first_at_or_after, last_at_or_before, fee_ps, summarize, STAKE

STEP = 5.0
WS = (60, 180, 300)
XS = (0.05, 0.10, 0.15)
DS = (10, 30)
HS = (60, 180, 300, 600)
COOL = 300.0

g = load_games()
P = load_prints()
psl = game_slices(P)
AL = pd.read_parquet(DL / 'alignment_flags.parquet').set_index('game_id')
rows = []
for r in g.itertuples(index=False):
    gid = r.game_id
    if gid not in psl or np.isnan(r.tip) or r.flag_print_gap_in_play:
        continue
    a, b = psl[gid]
    rt, ph, dd = P.rt.values[a:b], P.ph.values[a:b], P.d.values[a:b]
    A_t, A_p, B_t, B_p = rt[dd == 1], ph[dd == 1], rt[dd == -1], ph[dd == -1]
    grid = np.arange(r.tip, r.end_wc, STEP)
    if len(grid) < 100:
        continue

    def mid_at(q):
        _, pa = last_at_or_before(A_t, A_p, q, within=120)
        _, pb = last_at_or_before(B_t, B_p, q, within=120)
        return (pa + pb) / 2

    m_now = mid_at(grid)
    for W in WS:
        m_back = mid_at(grid - W)
        sw = m_now - m_back
        for X in XS:
            cand = np.flatnonzero((np.abs(sw) >= X) & (m_now >= 0.10) & (m_now <= 0.90) & (grid - W >= r.tip))
            last = -1e18
            for i in cand:
                t = grid[i]
                if t - last < COOL:
                    continue
                last = t
                up_home = sw[i] > 0
                for D in DS:
                    for mode in ('follow', 'fade'):
                        buy_home = up_home if mode == 'follow' else (not up_home)
                        if buy_home:
                            te, pe = first_at_or_after(A_t, A_p, np.array([t + D]), within=60)
                            own = pe[0]
                        else:
                            te, pe = first_at_or_after(B_t, B_p, np.array([t + D]), within=60)
                            own = 1 - pe[0]
                        if np.isnan(own) or not (0.02 < own < 0.98):
                            continue
                        te = te[0]
                        win = r.home_win if buy_home else 1 - r.home_win
                        sh = STAKE / own
                        fin = sh * fee_ps(own)
                        rec = dict(game_id=gid, split=r.split, season=r.season, month=r.month, W=W, X=X, D=D, mode=mode, t=t,
                                   swing=abs(sw[i]), mid=m_now[i], own_in=own, win=win, align_ok=bool(AL.align_ok.get(gid, False)),
                                   q4=bool(t > r.end_wc - 900), pnl_hold=sh * (win - own) - fin, gross_hold=sh * (win - own))
                        for h in HS:
                            q = te + h
                            if q >= r.end_wc:
                                out_px, f_out = float(win), False
                            else:
                                if buy_home:
                                    _, xp = first_at_or_after(B_t, B_p, np.array([q]), within=120); xo = xp[0]
                                else:
                                    _, xp = first_at_or_after(A_t, A_p, np.array([q]), within=120); xo = 1 - xp[0]
                                if np.isnan(xo):
                                    mm = mid_at(np.array([q]))[0]
                                    xo = (mm if buy_home else 1 - mm) - 0.01
                                out_px, f_out = xo, True
                            rec[f'pnl_{h}'] = sh * (out_px - own) - fin - (sh * fee_ps(out_px) if f_out else 0.0)
                            rec[f'gross_{h}'] = sh * (out_px - own)
                            # drift of the fair mid (signed for the bought side), c per share
                            mq = mid_at(np.array([q]))[0] if q < r.end_wc else float(r.home_win)
                            m0 = mid_at(np.array([te]))[0]
                            rec[f'drift_{h}'] = (mq - m0) * (1 if buy_home else -1)
                        rows.append(rec)
E = pd.DataFrame(rows)
E.to_parquet(DL / 'h3_events.parquet', index=False)
print('events', len(E), flush=True)
out = []
for key, d in E.groupby(['W', 'X', 'D', 'mode', 'split']):
    rr = dict(zip(['W', 'X', 'D', 'mode', 'split'], key))
    rr['n'] = len(d); rr['games'] = d.game_id.nunique()
    for c in ['pnl_60', 'pnl_180', 'pnl_300', 'pnl_600', 'pnl_hold']:
        s = summarize(d[d[c].notna()], c, n_boot=300)
        rr[c] = s.pnl_mean[0]; rr[c + '_lo'] = s.ci_lo[0]; rr[c + '_hi'] = s.ci_hi[0]
    for c in ['gross_60', 'gross_180', 'gross_300', 'gross_600', 'gross_hold', 'drift_60', 'drift_180', 'drift_300', 'drift_600']:
        rr[c] = d[c].mean()
    out.append(rr)
O = pd.DataFrame(out)
O.to_csv(RL / 'h3_summary.csv', index=False)
pd.set_option('display.width', 250)
print(O[['W', 'X', 'D', 'mode', 'split', 'n', 'drift_60', 'drift_300', 'drift_600', 'gross_300', 'pnl_300', 'pnl_300_lo', 'pnl_300_hi', 'pnl_hold', 'pnl_hold_lo', 'pnl_hold_hi']].round(3).to_string())
