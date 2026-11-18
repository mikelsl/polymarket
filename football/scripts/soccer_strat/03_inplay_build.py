"""In-play dataset from 1X2 taker prints + ESPN events (analysis-ready matches).

Outputs (data/soccer_strat/):
  events_located.parquet : goals and red cards with the print-located jump time (`ev_ts`)
  snaps.parquet          : one row per match x role x check time (every 60 s of play) with state and print prices
  event_trades.parquet   : per goal/red card x role x entry lag: entry asks/bids and exit prices at several horizons
Price conventions (YES scale): ask = YES price of a print where the taker gained YES (BUY YES / SELL NO);
bid = YES price of a print where the taker gained NO. Buy YES at ask, buy NO at 1 - bid, sell YES at bid,
sell NO at 1 - ask. Entries use the FIRST print at/after the decision time (never minute history).
"""
import sys

sys.path.insert(0, 'polymarket/research/auto-20261104/scripts/soccer_strat')
import numpy as np
import pandas as pd

from lib import DS, SD, ROLES, load_matches, load_trades, log

LAGS = [10, 30, 60, 120, 300]          # entry delay after the located jump (s)
HORIZONS = [120, 300, 600, 1200, 1800]  # holding time after entry (s); 'res' = to resolution
ENTRY_WIN = 60                        # entry print must come within 60 s of the decision time
EXIT_WIN = 180


class Book:
    """Print arrays of one 1X2 market (YES scale)."""

    def __init__(self, g):
        b = g.taker_buys_yes.values
        self.ts_a, self.px_a, self.sz_a = g.ts.values[b], g.yes_price.values[b], g['size'].values[b]
        self.ts_b, self.px_b, self.sz_b = g.ts.values[~b], g.yes_price.values[~b], g['size'].values[~b]
        self.ts, self.px = g.ts.values, g.yes_price.values

    @staticmethod
    def _next(ts, px, sz, t, win):
        i = np.searchsorted(ts, t, 'left')
        ok = (i < len(ts))
        i2 = np.minimum(i, len(ts) - 1)
        good = ok & (ts[i2] - t <= win) if len(ts) else np.zeros_like(t, dtype=bool)
        if not len(ts):
            return np.full(len(t), np.nan), np.full(len(t), np.nan), np.full(len(t), np.nan)
        return (np.where(good, px[i2], np.nan), np.where(good, ts[i2], np.nan), np.where(good, sz[i2], np.nan))

    @staticmethod
    def _last(ts, px, t, win):
        if not len(ts):
            return np.full(len(t), np.nan)
        i = np.searchsorted(ts, t, 'right') - 1
        i2 = np.maximum(i, 0)
        good = (i >= 0) & (t - ts[i2] <= win)
        return np.where(good, px[i2], np.nan)

    def next_ask(self, t, win=ENTRY_WIN):
        return self._next(self.ts_a, self.px_a, self.sz_a, np.asarray(t, float), win)

    def next_bid(self, t, win=ENTRY_WIN):
        return self._next(self.ts_b, self.px_b, self.sz_b, np.asarray(t, float), win)

    def last_ask(self, t, win=120):
        return self._last(self.ts_a, self.px_a, np.asarray(t, float), win)

    def last_bid(self, t, win=120):
        return self._last(self.ts_b, self.px_b, np.asarray(t, float), win)

    def median(self, a, b):
        i, j = np.searchsorted(self.ts, a, 'left'), np.searchsorted(self.ts, b, 'left')
        return float(np.median(self.px[i:j])) if j > i else np.nan


def locate(book, wc, up):
    """Jump time of a goal/red card in the prints of `book` (YES price moves up if `up`)."""
    pre = book.median(wc - 240, wc - 60)
    post = book.median(wc + 90, wc + 240)
    if np.isnan(pre):
        i = np.searchsorted(book.ts, wc - 60, 'left') - 1
        pre = book.px[i] if i >= 0 else np.nan
    if np.isnan(pre) or np.isnan(post):
        return wc, pre, post, 'no_prints'
    mv = (post - pre) if up else (pre - post)
    if mv < 0.02:
        return wc, pre, post, 'small_move'
    thr = pre + 0.5 * (post - pre)
    i, j = np.searchsorted(book.ts, wc - 150, 'left'), np.searchsorted(book.ts, wc + 150, 'right')
    seg_t, seg_p = book.ts[i:j], book.px[i:j]
    hit = np.where(seg_p >= thr)[0] if up else np.where(seg_p <= thr)[0]
    if len(hit) == 0:
        return wc, pre, post, 'no_cross'
    return int(seg_t[hit[0]]), pre, post, 'located'


def main():
    m = load_matches()
    m = m.dropna(subset=['kickoff_wall_ts', 'halftime_wall_ts', 'second_half_wall_ts', 'fulltime_wall_ts'])
    log('matches with period wallclocks', len(m))
    ev = pd.read_parquet(f'{SD}/events.parquet')
    ev = ev[ev.match_id.isin(m.match_id) & (ev.scoring_play.fillna(False) | ev.is_red_card.fillna(False))
            & ev.wallclock_ts.notna() & (ev.source == 'keyEvents')].copy()
    ev['kind'] = np.where(ev.scoring_play.fillna(False), 'goal', 'red')
    ev = ev.sort_values(['match_id', 'wallclock_ts', 'seq'])
    tr = load_trades(filters=[('match_id', 'in', list(m.match_id)), ('t_rel_kickoff_s', '>=', -1200),
                              ('t_rel_kickoff_s', '<=', 12000)])
    tr = tr.sort_values(['match_id', 'role', 'ts'], kind='mergesort')
    log('in-play trades', len(tr))
    trg = {k: g for k, g in tr.groupby('match_id', sort=False)}
    evg = {k: g for k, g in ev.groupby('match_id', sort=False)}
    res = m.set_index('match_id')[['res_home', 'res_draw', 'res_away']]
    E, S, T = [], [], []
    for n, row in enumerate(m.itertuples(index=False)):
        mid = row.match_id
        g = trg.get(mid)
        if g is None:
            continue
        books = {r: Book(gg) for r, gg in g.groupby('role', sort=False)}
        if len(books) < 3:
            continue
        kw, ht, sh, ft = int(row.kickoff_wall_ts), int(row.halftime_wall_ts), int(row.second_half_wall_ts), int(row.fulltime_wall_ts)
        # ---- locate events
        evs = []
        for e in (evg.get(mid).itertuples(index=False) if mid in evg else []):
            side = e.team_side_pm
            if side not in ('home', 'away'):
                continue
            up = e.kind == 'goal'
            jt, pre, post, how = locate(books[side], int(e.wallclock_ts), up)
            evs.append(dict(match_id=mid, kind=e.kind, side=side, wc=int(e.wallclock_ts), ev_ts=jt, pre=pre, post=post,
                            how=how, period=e.period, minute=e.minute, added=e.added_minute, seq=e.seq,
                            goal_kind=e.goal_kind))
        evs.sort(key=lambda d: (d['ev_ts'], d['seq']))
        hg = ag = hr = ar = 0
        for d in evs:
            if d['kind'] == 'goal':
                d['h_before'], d['a_before'] = hg, ag
                hg += d['side'] == 'home'
                ag += d['side'] == 'away'
                d['h_after'], d['a_after'] = hg, ag
            else:
                hr += d['side'] == 'home'
                ar += d['side'] == 'away'
            d['hr_after'], d['ar_after'] = hr, ar
        E.extend(evs)
        ev_t = np.array([d['ev_ts'] for d in evs], dtype=float)
        ev_hg = np.array([d['h_after'] if d['kind'] == 'goal' else np.nan for d in evs], dtype=float)
        # cumulative state arrays
        cg_h = np.cumsum([1 if (d['kind'] == 'goal' and d['side'] == 'home') else 0 for d in evs]) if evs else np.array([])
        cg_a = np.cumsum([1 if (d['kind'] == 'goal' and d['side'] == 'away') else 0 for d in evs]) if evs else np.array([])
        cr_h = np.cumsum([1 if (d['kind'] == 'red' and d['side'] == 'home') else 0 for d in evs]) if evs else np.array([])
        cr_a = np.cumsum([1 if (d['kind'] == 'red' and d['side'] == 'away') else 0 for d in evs]) if evs else np.array([])

        def state(t):
            t = np.asarray(t, float)
            k = np.searchsorted(ev_t, t, 'right')  # events with ev_ts <= t
            z = lambda c: np.where(k > 0, c[np.maximum(k - 1, 0)], 0) if len(c) else np.zeros(len(t))
            last = np.where(k > 0, ev_t[np.maximum(k - 1, 0)], np.nan) if len(ev_t) else np.full(len(t), np.nan)
            return z(cg_h), z(cg_a), z(cr_h), z(cr_a), t - last

        # ---- periodic snapshots (every 60 s of play)
        t1 = np.arange(kw + 60, ht, 60)
        t2 = np.arange(sh + 60, ft, 60)
        tt = np.concatenate([t1, t2]).astype(float)
        half = np.concatenate([np.ones(len(t1)), 2 * np.ones(len(t2))])
        el = np.concatenate([(t1 - kw) / 60.0, (t2 - sh) / 60.0])
        gh, ga, rh, ra, since = state(tt)
        for r in ROLES:
            b = books[r]
            la, lb = b.last_ask(tt), b.last_bid(tt)
            na, nat, nas = b.next_ask(tt + 2)
            nb, nbt, nbs = b.next_bid(tt + 2)
            S.append(pd.DataFrame(dict(match_id=mid, role=r, t=tt.astype(np.int64), half=half.astype(np.int8), el=el,
                                       gh=gh, ga=ga, rh=rh, ra=ra, since_ev=since, last_ask=la, last_bid=lb,
                                       ask=na, ask_ts=nat, ask_size=nas, bid=nb, bid_ts=nbt, bid_size=nbs,
                                       y=int(res.at[mid, f'res_{r}']))))
        # ---- event trades
        for k_e, d in enumerate(evs):
            for L in LAGS:
                te = np.array([d['ev_ts'] + L], float)
                for r in ROLES:
                    b = books[r]
                    a, ats, asz = b.next_ask(te)
                    bb, bts, bsz = b.next_bid(te)
                    rec = dict(match_id=mid, ev_idx=k_e, kind=d['kind'], side=d['side'], ev_ts=d['ev_ts'], lag=L,
                               role=r, ask=a[0], ask_ts=ats[0], ask_size=asz[0], bid=bb[0], bid_ts=bts[0],
                               bid_size=bsz[0], y=int(res.at[mid, f'res_{r}']), ft=ft,
                               pre_mid=b.median(d['ev_ts'] - 240, d['ev_ts'] - 60))
                    for H in HORIZONS:
                        tx = np.array([d['ev_ts'] + L + H], float)
                        if tx[0] >= ft - 30:
                            rec[f'xb_{H}'] = np.nan
                            rec[f'xa_{H}'] = np.nan
                            rec[f'xres_{H}'] = 1
                        else:
                            rec[f'xb_{H}'] = b.next_bid(tx, EXIT_WIN)[0][0]
                            rec[f'xa_{H}'] = b.next_ask(tx, EXIT_WIN)[0][0]
                            rec[f'xres_{H}'] = 0
                    T.append(rec)
        if n % 200 == 0:
            log(n, mid, len(E), sum(len(s) for s in S), len(T))
    E = pd.DataFrame(E)
    E.to_parquet(f'{DS}/events_located.parquet', index=False)
    S = pd.concat(S, ignore_index=True)
    S.to_parquet(f'{DS}/snaps.parquet', index=False)
    T = pd.DataFrame(T)
    T.to_parquet(f'{DS}/event_trades.parquet', index=False)
    log('events', len(E), E.groupby(['kind', 'how']).size().to_dict())
    log('snaps', len(S), 'event trades', len(T))


if __name__ == '__main__':
    main()
