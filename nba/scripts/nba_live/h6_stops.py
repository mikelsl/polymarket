"""H6a: validate the live bot's exit rules on the full sample (moneyline all seasons; decision-time spread line 2025-26+).

Live rules (RUNSTATE 2026-11-03): enter early, then (a) stop: second half (ESPN period >= 3), best bid <= 0.20 and mid < 0.50
-> sell all; (b) Q4 take-profit: from period 4, best bid >= 0.50 -> sell all; else hold to resolution.
Population: every token whose entry price (first ask-side print at real time >= tip + 600 s) is in [0.25, 0.50].
Triggers are detected on bid-side prints of the token (a taker sold at <= 0.20 / >= 0.50); the sale fills at the next
bid-side print >= trigger + 2 s. Value of a rule at its trigger (per share) = sale - fee(sale) - final payoff
(positive = selling beats holding). Strategy P&L per 10 USDC for hold / stop / tp / both. Period starts = aligned ESPN
times (per-game lag). Output: data/nba_live/h6_stops_<market>.parquet, results/nba_live/h6_stops.csv
"""
import sys
import numpy as np
import pandas as pd
from common import DL, RL, ROOT, LAG, load_games, load_prints, load_states, game_slices, first_at_or_after, last_at_or_before, fee_ps, summarize, STAKE

g = load_games().set_index('game_id')
S = load_states(); ssl = game_slices(S)
AL = pd.read_parquet(DL / 'alignment_flags.parquet').set_index('game_id')


def period_starts(gid):
    a, b = ssl[gid]
    st = S.iloc[a:b]
    lag = float(AL.lag_used.get(gid, 0.0))
    out = {}
    for q in (3, 4):
        w = st.wc.values[st.k_period.values >= q]
        out[q] = (w.min() + lag) if len(w) else np.nan
    return out


def simulate(tok_prints, market):
    """tok_prints: dict gid -> list of (side_name, win, ask_t, ask_p, bid_t, bid_p) in the token's own price terms"""
    rows = []
    for gid, toks in tok_prints.items():
        r = g.loc[gid]
        if gid not in ssl or np.isnan(r.tip):
            continue
        ps = period_starts(gid)
        for name, win, A_t, A_p, B_t, B_p in toks:
            te, pe = first_at_or_after(A_t, A_p, np.array([r.tip + 600.0]), within=600)
            px = pe[0]
            if np.isnan(px) or not (0.25 <= px <= 0.50):
                continue
            sh = STAKE / px
            base = dict(game_id=gid, market=market, token=name, split=r.split, season=r.season, month=r.month, px=px, win=win)
            hold = sh * (win - px) - sh * fee_ps(px)
            res = {}
            for rule, q, cond in (('stop', 3, lambda b: b <= 0.20), ('tp', 4, lambda b: b >= 0.50)):
                if np.isnan(ps[q]):
                    res[rule] = None; continue
                k = np.searchsorted(B_t, max(ps[q], te[0]), side='left')
                idx = np.flatnonzero(cond(B_p[k:]))
                trig = None
                for j in idx:
                    t_tr = B_t[k + j]
                    if rule == 'stop':
                        # mid < 0.5 check with the last ask print
                        _, la = last_at_or_before(A_t, A_p, np.array([t_tr]), within=300)
                        if not np.isnan(la[0]) and (la[0] + B_p[k + j]) / 2 >= 0.5:
                            continue
                    trig = t_tr; break
                if trig is None:
                    res[rule] = None; continue
                ts_, sp = first_at_or_after(B_t, B_p, np.array([trig + 2.0]), within=120)
                sale = sp[0] if not np.isnan(sp[0]) else B_p[np.searchsorted(B_t, trig)]
                res[rule] = (trig, sale)
            rec = dict(base, pnl_hold=hold)
            for rule in ('stop', 'tp'):
                if res[rule] is None:
                    rec[f'{rule}_trig'] = False; rec[f'pnl_{rule}'] = hold; rec[f'{rule}_value'] = np.nan; rec[f'{rule}_sale'] = np.nan
                else:
                    t_tr, sale = res[rule]
                    rec[f'{rule}_trig'] = True; rec[f'{rule}_sale'] = sale
                    rec[f'pnl_{rule}'] = sh * (sale - px) - sh * fee_ps(px) - sh * fee_ps(sale)
                    rec[f'{rule}_value'] = sale - fee_ps(sale) - win      # per share, vs holding
            # both: whichever trigger comes first
            cands = [(res[k][0], res[k][1]) for k in ('stop', 'tp') if res[k] is not None]
            if cands:
                t_tr, sale = min(cands)
                rec['pnl_both'] = sh * (sale - px) - sh * fee_ps(px) - sh * fee_ps(sale)
            else:
                rec['pnl_both'] = hold
            rows.append(rec)
    return pd.DataFrame(rows)


def ml_tokens():
    P = load_prints(); psl = game_slices(P)
    out = {}
    for gid, (a, b) in psl.items():
        if gid not in g.index or g.loc[gid, 'flag_print_gap_in_play']:
            continue
        rt, ph, dd = P.rt.values[a:b], P.ph.values[a:b], P.d.values[a:b]
        hw = int(g.loc[gid, 'home_win'])
        At, Ap, Bt, Bp = rt[dd == 1], ph[dd == 1], rt[dd == -1], ph[dd == -1]
        out[gid] = [('home', hw, At, Ap, Bt, Bp), ('away', 1 - hw, Bt, 1 - Bp, At, 1 - Ap)]
    return out


def spread_tokens():
    sel = pd.read_parquet(DL / 'lines_selected.parquet')
    sel = sel[sel.sel_kind == 'spread']
    out = {}
    for r in sel.itertuples(index=False):
        f1 = ROOT / 'data' / 'nba_data' / 'work' / 'trades_parts' / f'{r.market_id}.parquet'
        f2 = DL / 'trades_lines' / f'{r.market_id}.parquet'
        f = f1 if f1.exists() else f2
        if not f.exists() or r.game_id not in g.index or g.loc[r.game_id, 'flag_print_gap_in_play'] or pd.isna(r.winner_index) or r.winner_index < 0:
            continue
        x = pd.read_parquet(f, columns=['ts', 'token_index', 'side', 'price'])
        x = x[x.token_index >= 0].sort_values('ts')
        rt = x.ts.values.astype(float) - LAG
        tok0 = x.token_index.values == 0
        buy = x.side.values == 'BUY'
        p0 = np.where(tok0, x.price.values, 1 - x.price.values)
        d = np.where(tok0 == buy, 1, -1)
        w0 = int(r.winner_index == 0)
        At, Ap, Bt, Bp = rt[d == 1], p0[d == 1], rt[d == -1], p0[d == -1]
        out[r.game_id] = [('t0', w0, At, Ap, Bt, Bp), ('t1', 1 - w0, Bt, 1 - Bp, At, 1 - Ap)]
    return out


if __name__ == '__main__':
    which = sys.argv[1:] or ['ml', 'spread']
    allr = []
    for mk in which:
        toks = ml_tokens() if mk == 'ml' else spread_tokens()
        D = simulate(toks, mk)
        D.to_parquet(DL / f'h6_stops_{mk}.parquet', index=False)
        allr.append(D)
        print(mk, 'positions', len(D), 'games', D.game_id.nunique(), flush=True)
    D = pd.concat(allr, ignore_index=True)
    out = []
    for (mk, sp), d in D.groupby(['market', 'split']):
        r = dict(market=mk, split=sp, n=len(d), games=d.game_id.nunique(), px=d.px.mean(), win=d.win.mean())
        for c in ['pnl_hold', 'pnl_stop', 'pnl_tp', 'pnl_both']:
            s = summarize(d, c, n_boot=500); r[c] = s.pnl_mean[0]; r[c + '_lo'] = s.ci_lo[0]; r[c + '_hi'] = s.ci_hi[0]
        for rule in ['stop', 'tp']:
            t = d[d[f'{rule}_trig']]
            r[f'{rule}_n'] = len(t); r[f'{rule}_sale'] = t[f'{rule}_sale'].mean(); r[f'{rule}_winrate'] = t.win.mean()
            s = summarize(t.assign(v=100 * t[f'{rule}_value']), 'v', n_boot=500)
            r[f'{rule}_value_c'] = s.pnl_mean[0] if len(s) else np.nan; r[f'{rule}_value_lo'] = s.ci_lo[0] if len(s) else np.nan; r[f'{rule}_value_hi'] = s.ci_hi[0] if len(s) else np.nan
            # paired difference vs hold per position
            dd = d.assign(diff=d[f'pnl_{rule}'] - d.pnl_hold)
            s2 = summarize(dd, 'diff', n_boot=500); r[f'{rule}_minus_hold'] = s2.pnl_mean[0]; r[f'{rule}_mh_lo'] = s2.ci_lo[0]; r[f'{rule}_mh_hi'] = s2.ci_hi[0]
        dd = d.assign(diff=d.pnl_both - d.pnl_hold)
        s2 = summarize(dd, 'diff', n_boot=500); r['both_minus_hold'] = s2.pnl_mean[0]; r['both_mh_lo'] = s2.ci_lo[0]; r['both_mh_hi'] = s2.ci_hi[0]
        out.append(r)
    O = pd.DataFrame(out)
    O.to_csv(RL / 'h6_stops.csv', index=False)
    pd.set_option('display.width', 250)
    print(O.round(3).T.to_string())
