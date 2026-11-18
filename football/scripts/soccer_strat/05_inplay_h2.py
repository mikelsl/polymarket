"""H2: in-play market vs goal model, state calibration and pre-registered state rules (print entries, hold to
resolution, fee 0.05 p(1-p) per share on entry).

Input: data/soccer_strat/snaps_model.parquet (04_model.py). Snapshots every 60 s of play; state uses events with
located time <= t; entry = first print at/after t+2 s (ask for YES, 1 - bid for NO) within 60 s.
Outputs: results/soccer_strat/h2_*.csv
"""
import sys

sys.path.insert(0, 'polymarket/research/auto-20261104/scripts/soccer_strat')
import numpy as np
import pandas as pd

from lib import DS, RS, STAKE, cluster_boot, fee_per_share, load_matches, log

MB = [0, 15, 30, 45, 60, 70, 80, 200]
MBL = ['00-15', '15-30', '30-45+', '45-60', '60-70', '70-80', '80-90+']


def load():
    m = load_matches()
    lam = pd.read_parquet(f'{DS}/lambdas.parquet')
    bad = set(lam.match_id[lam.err > 0.01])
    S = pd.read_parquet(f'{DS}/snaps_model.parquet')
    S = S[~S.match_id.isin(bad)]
    S = S.merge(m[['match_id', 'period', 'league', 'month', 'bfx_c_home', 'bfx_c_away']], on='match_id')
    S['minute'] = np.where(S.half == 1, S.el, 45 + S.el)
    S['mb'] = pd.cut(S.minute, MB, labels=MBL, right=False)
    S['mid'] = np.where(S.last_ask.notna() & S.last_bid.notna() & (S.last_ask >= S.last_bid - 0.01),
                        (S.last_ask + S.last_bid) / 2, np.nan)
    gd_home = S.gh - S.ga
    S['gd_team'] = np.select([S.role == 'home', S.role == 'away'], [gd_home, -gd_home], np.abs(gd_home))
    S['fav'] = np.select([S.role == 'home', S.role == 'away'], [S.bfx_c_home > S.bfx_c_away, S.bfx_c_away > S.bfx_c_home], False)
    S['red'] = (S.rh + S.ra) > 0
    S['quiet'] = S.since_ev.isna() | (S.since_ev >= 120)
    return S


def pnl_rows(t, side):
    if side == 'yes':
        p, yy = t.ask, t.y
    else:
        p, yy = 1 - t.bid, 1 - t.y
    sh = STAKE / p
    t = t.assign(price=p, stake=STAKE, pnl=sh * (yy - p) - sh * fee_per_share(p), usdc_at_px=np.where(side == 'yes', t.ask_size * t.ask, t.bid_size * (1 - t.bid)))
    return t


def evaluate(S, mask, side, name, first=True, rows=None, extra=None):
    px_ok = S.ask.between(0.03, 0.97) if side == 'yes' else S.bid.between(0.03, 0.97)
    t = S[mask & px_ok & S.quiet]
    if first:
        t = t.sort_values('t').groupby(['match_id', 'role'], sort=False).head(1)
    t = pnl_rows(t, side)
    for per in ['train', 'test']:
        tt = t[t.period == per]
        r = cluster_boot(tt) if len(tt) else dict(n=0)
        d = dict(rule=name, side=side, period=per) | (extra or {}) | r
        if len(tt):
            d['mean_px'] = tt.price.mean()
            d['share_depth10'] = float((tt.usdc_at_px >= 10).mean())
        rows.append(d)
    return t


def main():
    S = load()
    log('snaps', len(S), 'matches', S.match_id.nunique())
    Q = S[S.quiet & S.mid.notna()].copy()
    # ---- A. calibration of market mid and model by state (one obs per match-role-minute bucket)
    A = Q.sort_values('t').groupby(['match_id', 'role', 'mb'], observed=True).head(1)
    A['state'] = np.where(A.role == 'draw', 'draw|gd' + A.gd_team.clip(0, 2).astype(int).astype(str),
                          'team|gd' + A.gd_team.clip(-2, 2).astype(int).astype(str))
    def ll(p, y):
        p = np.clip(p, 1e-4, 1 - 1e-4)
        return -(y * np.log(p) + (1 - y) * np.log(1 - p))
    A['ll_mkt'], A['ll_mod'] = ll(A.mid, A.y), ll(A.model, A.y)
    cal = A.groupby(['period', 'mb', 'state'], observed=True).agg(
        n=('y', 'size'), freq=('y', 'mean'), mid=('mid', 'mean'), model=('model', 'mean'),
        mid_minus_model=('mid', lambda v: np.nan), ll_mkt=('ll_mkt', 'mean'), ll_mod=('ll_mod', 'mean')).reset_index()
    cal['mid_minus_model'] = cal.mid - cal.model
    cal['freq_minus_mid'] = cal.freq - cal.mid
    cal.to_csv(f'{RS}/h2_state_calibration.csv', index=False)
    tot = A.groupby(['period', 'mb'], observed=True).agg(n=('y', 'size'), ll_mkt=('ll_mkt', 'mean'), ll_mod=('ll_mod', 'mean'),
                                                          abs_gap=('mid', lambda v: np.nan)).reset_index()
    A['abs_gap'] = (A.mid - A.model).abs()
    tot['abs_gap'] = A.groupby(['period', 'mb'], observed=True).abs_gap.mean().values
    tot.to_csv(f'{RS}/h2_logloss_by_minute.csv', index=False)
    print(tot.round(4).to_string())
    big = cal[(cal.n >= 150)].copy()
    big['z'] = big.freq_minus_mid / np.sqrt(big.mid * (1 - big.mid) / big.n)
    print(big[big.period == 'train'].sort_values('z')[['mb', 'state', 'n', 'freq', 'mid', 'model', 'z']].round(3).head(6).to_string())
    print(big[big.period == 'train'].sort_values('z')[['mb', 'state', 'n', 'freq', 'mid', 'model', 'z']].round(3).tail(6).to_string())
    # ---- B. rules
    rows = []
    fee_y, fee_n = fee_per_share(S.ask), fee_per_share(S.bid)
    e_yes = S.model - S.ask - fee_y
    e_no = S.bid - S.model - fee_n
    n_var = 0
    for d in [0.02, 0.03, 0.05, 0.08]:
        for mlo, mhi in [(0, 200), (0, 60), (60, 200)]:
            mm = S.minute.between(mlo, mhi, inclusive='left')
            evaluate(S, (e_yes >= d) & mm, 'yes', 'model_edge', rows=rows, extra=dict(d=d, mins=f'{mlo}-{mhi}'))
            evaluate(S, (e_no >= d) & mm, 'no', 'model_edge', rows=rows, extra=dict(d=d, mins=f'{mlo}-{mhi}'))
            n_var += 2
    # current-score outcome late (prior project's idea), with prints
    gdh = S.gh - S.ga
    cur = np.select([gdh > 0, gdh < 0], [S.role == 'home', S.role == 'away'], S.role == 'draw')
    for M in [60, 70, 75, 80, 85]:
        for lo, hi in [(0.03, 0.97), (0.5, 0.9), (0.9, 0.97)]:
            band_y = S.ask.between(lo, hi)
            evaluate(S, cur & (S.minute >= M) & band_y, 'yes', 'current_score', rows=rows, extra=dict(M=M, band=f'{lo}-{hi}'))
            evaluate(S, cur & (S.minute >= M) & (1 - S.bid).between(1 - hi, 1 - lo), 'no', 'current_score', rows=rows,
                     extra=dict(M=M, band=f'{lo}-{hi}'))
            n_var += 2
    # draw at level scores late; favourite trailing by one
    for M in [60, 70, 80]:
        for g0 in [0, 1]:
            lvl = (S.role == 'draw') & (gdh == 0) & (S.minute >= M) & ((S.gh == 0) if g0 == 0 else (S.gh >= 1))
            evaluate(S, lvl, 'yes', 'draw_level', rows=rows, extra=dict(M=M, score='0-0' if g0 == 0 else 'n-n'))
            evaluate(S, lvl, 'no', 'draw_level', rows=rows, extra=dict(M=M, score='0-0' if g0 == 0 else 'n-n'))
            n_var += 2
    for lo, hi in [(15, 45), (45, 70), (70, 85)]:
        ft = S.fav & (S.gd_team == -1) & S.minute.between(lo, hi, inclusive='left') & S.role.isin(['home', 'away'])
        evaluate(S, ft, 'yes', 'fav_trailing1', rows=rows, extra=dict(mins=f'{lo}-{hi}'))
        evaluate(S, ft, 'no', 'fav_trailing1', rows=rows, extra=dict(mins=f'{lo}-{hi}'))
        n_var += 2
    R = pd.DataFrame(rows)
    R['ci_roi'] = R.ci_roi.map(lambda t: f'[{t[0]:+.3f},{t[1]:+.3f}]' if isinstance(t, tuple) else '')
    R.to_csv(f'{RS}/h2_rules.csv', index=False)
    log('variants (rule x side x params)', n_var)
    cols = ['rule', 'side', 'period', 'd', 'mins', 'M', 'band', 'score', 'n', 'matches', 'roi', 'ci_roi', 'mean_px', 'share_depth10']
    cols = [c for c in cols if c in R]
    keys = [c for c in ['rule', 'side', 'd', 'mins', 'M', 'band', 'score'] if c in R]
    R2 = R.copy()
    for k in keys:
        R2[k] = R2[k].astype(object).where(R2[k].notna(), '')
    W = R2.pivot_table(index=keys, columns='period',
                      values=['n', 'roi'], aggfunc='first', dropna=False)
    W.columns = [f'{a}_{b}' for a, b in W.columns]
    W = W.dropna(subset=['n_train'])
    W.to_csv(f'{RS}/h2_rules_wide.csv')
    print(W.sort_values('roi_train', ascending=False).round(3).head(15).to_string())


if __name__ == '__main__':
    main()
