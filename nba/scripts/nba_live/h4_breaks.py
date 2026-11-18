"""H4: price behaviour during breaks in play (quarter breaks, halftime, timeouts, other long stoppages).

Breaks from ESPN plays (aligned games only; ESPN time shifted by the per-game market lag):
  qbreak: 'End Period' of periods 1 and 3; half: end of period 2; ot: end of period >= 4 when tied (break before OT)
  timeout: 'Full Timeout' / 'Short Timeout' plays in periods 1-4 that are followed by >= 60 s without plays
  stoppage: any other gap >= 90 s between consecutive plays (reviews, injuries, free-throw delays) in periods 1-4
Break start s = aligned wallclock of the play that starts the break; break end e = aligned wallclock of the next play.
Entry at s + 20 s (first print on our side), exit at min(s + planned, e - 10 s) where planned = 100 s (qbreak, timeout,
stoppage), 600 s (half), 60 s (ot); exit at the first opposite print (fair mid - 1c if none). Also hold to resolution.
Sides: leader (by score at s), trailer, mover (side whose price rose over the 180 s before s), fader (side whose price fell),
pre-game favourite. Descriptive drift: signed print-mid change from s+20 to the exit time.
Output: data/nba_live/h4_breaks.parquet, results/nba_live/h4_summary.csv
"""
import numpy as np
import pandas as pd
from common import DL, RL, load_games, load_prints, load_states, game_slices, first_at_or_after, last_at_or_before, fee_ps, summarize, STAKE

g = load_games()
P = load_prints(); S = load_states()
psl, ssl = game_slices(P), game_slices(S)
AL = pd.read_parquet(DL / 'alignment_flags.parquet').set_index('game_id')
PLAN = {'qbreak': 100.0, 'half': 600.0, 'ot': 60.0, 'timeout': 100.0, 'stoppage': 100.0}
rows = []
for r in g.itertuples(index=False):
    gid = r.game_id
    if gid not in psl or gid not in ssl or r.flag_print_gap_in_play or not bool(AL.align_ok.get(gid, False)):
        continue
    lag = float(AL.lag_used[gid])
    a, b = psl[gid]
    rt, ph, dd = P.rt.values[a:b], P.ph.values[a:b], P.d.values[a:b]
    A_t, A_p, B_t, B_p = rt[dd == 1], ph[dd == 1], rt[dd == -1], ph[dd == -1]
    a2, b2 = ssl[gid]
    st = S.iloc[a2:b2]
    wc = st.wc.values + lag
    ptype = st.play_type_id.values.astype(str)
    per = st.k_period.values
    hs, as_ = st.k_home_score.values, st.k_away_score.values

    def mid_at(q):
        _, pa = last_at_or_before(A_t, A_p, q, within=120)
        _, pb = last_at_or_before(B_t, B_p, q, within=120)
        return (pa + pb) / 2

    gaps = np.r_[np.diff(wc), np.nan]
    used = set()
    brk = []
    for i in range(len(wc) - 1):
        kind = None
        if ptype[i] == '412':
            pp = st.period.values[i]
            if pp in (1, 3): kind = 'qbreak'
            elif pp == 2: kind = 'half'
            elif pp >= 4 and hs[i] == as_[i]: kind = 'ot'
        elif ptype[i] in ('16', '17') and per[i] <= 4 and gaps[i] >= 60:
            kind = 'timeout'
        elif per[i] <= 4 and gaps[i] >= 90 and ptype[i] not in ('412', '402'):
            kind = 'stoppage'
        if kind:
            brk.append((kind, i))
    for kind, i in brk:
        s, e = wc[i], wc[i + 1]
        if e - s < 40:
            continue
        t_in = s + 20.0
        t_out = min(s + PLAN[kind], e - 10.0)
        if t_out - t_in < 15:
            continue
        m_in = mid_at(np.array([t_in]))[0]
        if np.isnan(m_in) or not (0.05 <= m_in <= 0.95):
            continue
        m_back = mid_at(np.array([s - 180.0]))[0]
        m_out_fair = mid_at(np.array([t_out]))[0]
        lead_home = hs[i] > as_[i] if hs[i] != as_[i] else None
        sides = {'leader': lead_home, 'trailer': (None if lead_home is None else not lead_home),
                 'mover': (None if np.isnan(m_back) or m_in == m_back else m_in > m_back),
                 'fader': (None if np.isnan(m_back) or m_in == m_back else m_in < m_back),
                 'prefav': r.pre_ph >= 0.5}
        for side, buy_home in sides.items():
            if buy_home is None:
                continue
            if buy_home:
                te, pe = first_at_or_after(A_t, A_p, np.array([t_in]), within=60); own = pe[0]
                _, xp = first_at_or_after(B_t, B_p, np.array([t_out]), within=30); xo = xp[0]
            else:
                te, pe = first_at_or_after(B_t, B_p, np.array([t_in]), within=60); own = 1 - pe[0]
                _, xp = first_at_or_after(A_t, A_p, np.array([t_out]), within=30); xo = 1 - xp[0]
            if np.isnan(own) or not (0.02 < own < 0.98) or te[0] >= t_out:
                continue
            src = 'print'
            if np.isnan(xo):
                xo = (m_out_fair if buy_home else 1 - m_out_fair) - 0.01; src = 'mid-1c'
            win = r.home_win if buy_home else 1 - r.home_win
            sh = STAKE / own
            rows.append(dict(game_id=gid, split=r.split, season=r.season, month=r.month, kind=kind, side=side, period=int(per[i]),
                             s=s, dur=e - s, own_in=own, own_out=xo, exit_src=src, win=win,
                             drift=(m_out_fair - m_in) * (1 if buy_home else -1),
                             pnl_break=sh * (xo - own) - sh * fee_ps(own) - sh * fee_ps(xo), gross_break=sh * (xo - own),
                             pnl_hold=sh * (win - own) - sh * fee_ps(own), gross_hold=sh * (win - own)))
B = pd.DataFrame(rows)
B.to_parquet(DL / 'h4_breaks.parquet', index=False)
print('break x side rows', len(B), B.groupby('kind').game_id.size().to_dict(), flush=True)
out = []
for key, d in B.groupby(['kind', 'side', 'split']):
    rr = dict(zip(['kind', 'side', 'split'], key)); rr['n'] = len(d); rr['games'] = d.game_id.nunique()
    rr['dur_med'] = d.dur.median(); rr['drift_c'] = 100 * d.drift.mean()
    s = summarize(d.assign(dc=100 * d.drift), 'dc', n_boot=300); rr['drift_lo'] = s.ci_lo[0]; rr['drift_hi'] = s.ci_hi[0]
    for c in ['pnl_break', 'pnl_hold']:
        s = summarize(d, c, n_boot=300); rr[c] = s.pnl_mean[0]; rr[c + '_lo'] = s.ci_lo[0]; rr[c + '_hi'] = s.ci_hi[0]
    rr['gross_break'] = d.gross_break.mean(); rr['gross_hold'] = d.gross_hold.mean()
    out.append(rr)
O = pd.DataFrame(out); O.to_csv(RL / 'h4_summary.csv', index=False)
pd.set_option('display.width', 250)
print(O.round(3).to_string())
