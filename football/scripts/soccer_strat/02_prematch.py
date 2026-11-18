"""H1: Polymarket 1X2 pre-match prices vs bookmaker fair value (calibration, value rules, lag/convergence).

Inputs: data/soccer_strat/prematch_snap.parquet (01_prematch_snapshots.py), matches + odds.
Outputs (results/soccer_strat/): h1_*.csv
Entry prices are taker prints (ask_yes for YES, 1 - bid_yes for NO). Fee = 0.05 p (1-p) per share on every fill.
"""
import sys

sys.path.insert(0, 'polymarket/research/auto-20261104/scripts/soccer_strat')
import itertools

import numpy as np
import pandas as pd

from lib import DS, RS, ROLES, STAKE, fee_per_share, load_matches, log, pretty, summarize, cluster_boot

pd.set_option('display.width', 250)
pd.set_option('display.max_columns', 40)
pd.set_option('display.max_rows', 400)


def long_table():
    m = load_matches()
    s = pd.read_parquet(f'{DS}/prematch_snap.parquet')
    keep = ['match_id', 'league', 'period', 'train', 'month', 'kickoff_ts', 'kickoff_wall_ts', 'fee_rate_1x2']
    rows = []
    for r in ROLES:
        d = m[keep].copy()
        d['role'] = r
        for c in ['pin_c', 'bfe_c', 'avg_c', 'bfx_c', 'fair_c', 'pin_o', 'bfe_o', 'avg_o', 'bfx_o', 'fair_o', 'b365_c']:
            d[c] = m[f'{c}_{r}']
        d['y'] = m[f'res_{r}']
        rows.append(d)
    L = pd.concat(rows, ignore_index=True)
    s = s.drop(columns=['kickoff_wall_ts'])
    X = s.merge(L, on=['match_id', 'role'], how='inner')
    X['mid'] = X.mid_med.fillna(X.last_px)
    return m, X


def trade_pnl(price, y, stake=STAKE):
    sh = stake / price
    return sh * (y - price) - sh * fee_per_share(price)


def calibration(X):
    """Accuracy of PM mid (normalised over the 3 roles) vs bookmaker fair on the same matches at T-5m."""
    out = []
    for off in [-3600, -300, 0]:
        d = X[X.offset_s == off].copy()
        w = d.pivot_table(index=['match_id', 'period'], columns='role', values=['mid', 'y', 'pin_c', 'bfe_c', 'avg_c'])
        w = w.dropna(subset=[('mid', r) for r in ROLES])
        for per in ['train', 'test']:
            ww = w[w.index.get_level_values(1) == per]
            for src in ['mid', 'pin_c', 'bfe_c', 'avg_c']:
                cols = [(src, r) for r in ROLES]
                ok = ww[cols].notna().all(axis=1) & ww[[('pin_c', r) for r in ROLES]].notna().all(axis=1) \
                    if per == 'train' else ww[cols].notna().all(axis=1) & ww[[('bfe_c', r) for r in ROLES]].notna().all(axis=1)
                P = ww.loc[ok, cols].values.astype(float)
                P = P / P.sum(axis=1, keepdims=True)
                Y = ww.loc[ok, [('y', r) for r in ROLES]].values.astype(float)
                brier = np.mean(np.sum((P - Y) ** 2, axis=1))
                ll = -np.mean(np.log(np.clip(np.sum(P * Y, axis=1), 1e-6, 1)))
                out.append(dict(offset_s=off, period=per, source=src, n=int(ok.sum()), brier=brier, logloss=ll,
                                pm_overround=float(np.nanmean(ww.loc[ok, [('mid', r) for r in ROLES]].sum(axis=1)))))
    return pd.DataFrame(out)


def bias_tables(X):
    d = X[(X.offset_s == -300) & X.mid.notna()].copy()
    d['bucket'] = pd.cut(d.mid, [0, .05, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.0])
    d['ask_ok'] = d.ask_yes.notna()
    d['pnl_yes'] = np.where(d.ask_yes.notna(), trade_pnl(d.ask_yes.clip(0.001, 0.999), d.y), np.nan)
    d['pnl_no'] = np.where(d.bid_yes.notna(), trade_pnl((1 - d.bid_yes).clip(0.001, 0.999), 1 - d.y), np.nan)
    d['spread'] = d.last_ask - d.last_bid
    g = d.groupby(['period', 'role', 'bucket'], observed=True).agg(
        n=('y', 'size'), freq=('y', 'mean'), pm_mid=('mid', 'mean'), fair_bfx=('bfx_c', 'mean'),
        fair_pin=('pin_c', 'mean'), mid_minus_bfx=('mid', lambda v: np.nan),
        roi_buy_yes=('pnl_yes', lambda v: v.sum() / (STAKE * v.notna().sum()) if v.notna().sum() else np.nan),
        roi_buy_no=('pnl_no', lambda v: v.sum() / (STAKE * v.notna().sum()) if v.notna().sum() else np.nan))
    g['mid_minus_bfx'] = g.pm_mid - g.fair_bfx
    d['gap_bfx'] = d.mid - d.bfx_c
    d['gap_pin'] = d.mid - d.pin_c
    d['gap_avg'] = d.mid - d.avg_c
    byrole = d.groupby(['period', 'role']).agg(n=('y', 'size'), freq=('y', 'mean'), pm_mid=('mid', 'mean'),
                                               gap_bfx=('gap_bfx', 'mean'), gap_pin=('gap_pin', 'mean'),
                                               gap_avg=('gap_avg', 'mean'), abs_gap_bfx=('gap_bfx', lambda v: v.abs().mean()),
                                               med_spread=('spread', 'median'))
    byleague = d.groupby(['period', 'league', 'role']).agg(n=('y', 'size'), gap_bfx=('gap_bfx', 'mean'),
                                                           abs_gap_bfx=('gap_bfx', lambda v: v.abs().mean()))
    return g.reset_index(), byrole.reset_index(), byleague.reset_index(), d


def value_rules(X):
    """Buy YES if fair - ask - fee >= d, buy NO if bid - fair - fee >= d; hold to resolution."""
    rows, trades = [], []
    for off, bench in itertools.product([-3600, -600, -300, -60], ['bfx_c', 'avg_c', 'pin_c', 'b365_c']):
        d = X[(X.offset_s == off)].copy()
        fy = d[bench]
        e_yes = fy - d.ask_yes - fee_per_share(d.ask_yes)
        e_no = d.bid_yes - fy - fee_per_share(d.bid_yes)
        for dd in [0.0, 0.005, 0.01, 0.02, 0.03, 0.05]:
            for side in ['yes', 'no', 'both']:
                parts = []
                if side in ('yes', 'both'):
                    s = d[(e_yes >= dd) & d.ask_yes.between(0.02, 0.98)].copy()
                    s['price'] = s.ask_yes
                    s['yy'] = s.y
                    s['edge'] = e_yes[s.index]
                    s['side'] = 'yes'
                    parts.append(s)
                if side in ('no', 'both'):
                    s = d[(e_no >= dd) & d.bid_yes.between(0.02, 0.98)].copy()
                    s['price'] = 1 - s.bid_yes
                    s['yy'] = 1 - s.y
                    s['edge'] = e_no[s.index]
                    s['side'] = 'no'
                    parts.append(s)
                t = pd.concat(parts)
                if len(t) == 0:
                    continue
                t['stake'] = STAKE
                t['pnl'] = trade_pnl(t.price, t.yy)
                for per in ['train', 'test']:
                    tt = t[t.period == per]
                    if len(tt) == 0:
                        continue
                    r = cluster_boot(tt)
                    rows.append(dict(offset_s=off, bench=bench, d=dd, side=side, period=per, mean_edge=tt.edge.mean()) | r)
                if side == 'both':
                    t['offset_s'], t['bench'], t['d'] = off, bench, dd
                    trades.append(t[['match_id', 'role', 'side', 'period', 'league', 'month', 'offset_s', 'bench', 'd',
                                     'price', 'yy', 'edge', 'pnl', 'stake', 'ask_size', 'bid_size']])
    return pd.DataFrame(rows), pd.concat(trades, ignore_index=True)


def lag_tests(X):
    """Does PM lag bookmakers? Signal known at t (opening odds) vs PM at t; exit at T-5m or hold."""
    W = X.pivot_table(index=['match_id', 'role'], columns='offset_s',
                      values=['mid', 'ask_yes', 'bid_yes'], aggfunc='first')
    W.columns = [f'{a}|{b}' for a, b in W.columns]
    meta = X[X.offset_s == -300].set_index(['match_id', 'role'])[
        ['period', 'league', 'month', 'y', 'bfx_o', 'bfx_c', 'pin_o', 'pin_c', 'avg_o', 'avg_c', 'bfe_o', 'bfe_c']]
    W = W.join(meta, how='inner')
    reg, rules = [], []
    for x in [-86400, -21600, -10800, -7200, -3600, -1800]:
        mx = W[f'mid|{x}']
        mc = W[f'mid|-300']
        for per in ['train', 'test']:
            sel = (W.period == per) & mx.notna() & mc.notna()
            for sig in ['bfx_o', 'pin_o', 'bfx_c', 'pin_c']:
                ok = sel & W[sig].notna()
                if ok.sum() < 50:
                    continue
                g = (W.loc[ok, sig] - mx[ok]).values
                dm = (mc[ok] - mx[ok]).values
                b = np.polyfit(g, dm, 1)
                reg.append(dict(offset_s=x, period=per, signal=sig, n=int(ok.sum()), slope=b[0], intercept=b[1],
                                corr=np.corrcoef(g, dm)[0, 1], mean_abs_gap=np.abs(g).mean(),
                                mean_abs_pm_move=np.abs(dm).mean(),
                                pm_x_abs_err_vs_close=np.abs(mx[ok] - W.loc[ok, 'bfx_c']).mean(),
                                sig_abs_err_vs_close=np.abs(W.loc[ok, sig] - W.loc[ok, 'bfx_c']).mean()))
        # convergence trades with the opening signal (known before t if t is after the collection time)
        for sig, dd in itertools.product(['bfx_o', 'pin_o', 'avg_o'], [0.0, 0.01, 0.02, 0.03, 0.05]):
            ay, by = W[f'ask_yes|{x}'], W[f'bid_yes|{x}']
            ex_b, ex_a = W['bid_yes|-300'], W['ask_yes|-300']
            fy = W[sig]
            e_yes = fy - ay - fee_per_share(ay)
            e_no = by - fy - fee_per_share(by)
            for side in ['yes', 'no']:
                if side == 'yes':
                    ok = (e_yes >= dd) & ay.between(0.02, 0.98)
                    pin = ay[ok]
                    pout = ex_b[ok]               # sell YES at the bid at T-5m
                    yy = W.y[ok]
                else:
                    ok = (e_no >= dd) & by.between(0.02, 0.98)
                    pin = 1 - by[ok]
                    pout = 1 - ex_a[ok]           # sell NO at 1 - ask at T-5m
                    yy = 1 - W.y[ok]
                sh = STAKE / pin
                t = pd.DataFrame({'match_id': W.index.get_level_values(0)[ok], 'period': W.period[ok].values,
                                  'stake': STAKE,
                                  'pnl_hold': (sh * (yy - pin) - sh * fee_per_share(pin)).values,
                                  'pnl_conv': (sh * (pout - pin) - sh * fee_per_share(pin) - sh * fee_per_share(pout)).values})
                for per in ['train', 'test']:
                    tt = t[t.period == per]
                    if len(tt) < 10:
                        continue
                    r1 = cluster_boot(tt.rename(columns={'pnl_hold': 'pnl'}))
                    tc = tt.dropna(subset=['pnl_conv']).rename(columns={'pnl_conv': 'pnl'})
                    r2 = cluster_boot(tc)
                    rules.append(dict(offset_s=x, signal=sig, d=dd, side=side, period=per, n=r1['n'],
                                      roi_hold=r1['roi'], ci_hold=r1['ci_roi'], n_conv=r2['n'], roi_conv=r2['roi'],
                                      ci_conv=r2['ci_roi'], pnl_conv_per_trade=r2['pnl_per_trade']))
    return pd.DataFrame(reg), pd.DataFrame(rules)


def opening_time(X):
    """When does the PM mid best match the bookmaker opening snapshot (proxy for its collection time)?"""
    d = X[X.mid.notna() & X.bfe_o.notna()].copy()
    d['err'] = (d.mid - d.bfe_o).abs()
    e = d.groupby(['match_id', 'offset_s']).err.mean().reset_index()
    best = e.loc[e.groupby('match_id').err.idxmin()]
    km = X.drop_duplicates('match_id').set_index('match_id').kickoff_ts
    best = best.assign(dow=pd.to_datetime(best.match_id.map(km), unit='s', utc=True).dt.day_name())
    return e.groupby('offset_s').err.mean().reset_index(), best.groupby(['offset_s']).size().reset_index(name='n_best')


def main():
    m, X = long_table()
    if len(sys.argv) > 1 and sys.argv[1] == 'lag':
        reg, rules = lag_tests(X)
        reg.to_csv(f'{RS}/h1_lag_regression.csv', index=False)
        rules.to_csv(f'{RS}/h1_lag_rules.csv', index=False)
        e, best = opening_time(X)
        e.to_csv(f'{RS}/h1_opening_time.csv', index=False)
        best.to_csv(f'{RS}/h1_opening_time_best.csv', index=False)
        return
    log('rows', len(X), 'matches', X.match_id.nunique())
    cal = calibration(X)
    cal.to_csv(f'{RS}/h1_calibration.csv', index=False)
    print(cal.round(4).to_string())
    g, byrole, byleague, d = bias_tables(X)
    g.to_csv(f'{RS}/h1_buckets.csv', index=False)
    byrole.to_csv(f'{RS}/h1_gap_by_role.csv', index=False)
    byleague.to_csv(f'{RS}/h1_gap_by_league.csv', index=False)
    print(byrole.round(4).to_string())
    print(g.round(4).to_string())
    vr, vt = value_rules(X)
    vr.to_csv(f'{RS}/h1_value_rules.csv', index=False)
    vt.to_parquet(f'{DS}/h1_value_trades.parquet', index=False)
    print(pretty(vr[(vr.side == 'both')]).to_string())
    reg, rules = lag_tests(X)
    reg.to_csv(f'{RS}/h1_lag_regression.csv', index=False)
    rules.to_csv(f'{RS}/h1_lag_rules.csv', index=False)
    print(reg.round(4).to_string())
    r2 = rules.copy()
    for c in ['ci_hold', 'ci_conv']:
        r2[c] = r2[c].map(lambda t: f'[{t[0]:+.3f},{t[1]:+.3f}]')
    print(r2.round(4).to_string())
    e, best = opening_time(X)
    e.to_csv(f'{RS}/h1_opening_time.csv', index=False)
    print(e.round(4).to_string())
    print(best.to_string())


if __name__ == '__main__':
    main()
