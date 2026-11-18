"""Shared helpers for workstream nba_live (in-game NBA pricing research). Read-only research.

Conventions
- All times are epoch seconds (float). Print times are converted to REAL fill time: rt = ts - LAG (Data API stamps lag ~4 s).
- Prices are expressed for the HOME team token (ph). d=+1: ask-side print (taker bought home / sold away),
  d=-1: bid-side print (taker sold home / bought away). Buying home fills at an ask-side print price, buying away fills at
  1 - (bid-side print price).
- Fee: today's taker fee 0.05 * p * (1-p) per share on every taker leg, for all periods.
"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
NBA = ROOT / 'data' / 'nba_data'
DL = ROOT / 'data' / 'nba_live'
RL = ROOT / 'results' / 'nba_live'
DL.mkdir(parents=True, exist_ok=True)
RL.mkdir(parents=True, exist_ok=True)

FEE = 0.05
LAG = 4.0
SPLIT_DATE = '2026-02-01'
STAKE = 10.0


def fee_ps(p):
    p = np.asarray(p, dtype=float)
    return FEE * p * (1 - p)


def epoch(s):
    """tz-aware datetime Series -> epoch seconds float"""
    s = pd.to_datetime(s, utc=True)
    return (s.astype('int64') / 1e6).astype(float) if str(s.dtype).endswith('[us, UTC]') else s.map(lambda x: x.timestamp() if pd.notna(x) else np.nan)


def to_epoch(s):
    s = pd.to_datetime(s, utc=True)
    out = np.full(len(s), np.nan)
    ok = s.notna().values
    out[ok] = np.array([x.timestamp() for x in s[ok]])
    return out


def load_games():
    return pd.read_parquet(DL / 'games.parquet')


def load_prints(role='moneyline'):
    return pd.read_parquet(DL / f'prints_{role}.parquet')


def load_states():
    return pd.read_parquet(DL / 'states.parquet')


def game_slices(df, key='game_id'):
    """df sorted by key: dict key -> (start, end) row slice"""
    k = df[key].values
    if len(k) == 0:
        return {}
    chg = np.r_[0, np.flatnonzero(k[1:] != k[:-1]) + 1, len(k)]
    return {k[chg[i]]: (chg[i], chg[i + 1]) for i in range(len(chg) - 1)}


def first_at_or_after(t_arr, v_arr, q, within=60.0):
    """vectorised: for each q, the first (t, v) with t >= q and t - q <= within; nan if none"""
    q = np.asarray(q, dtype=float)
    i = np.searchsorted(t_arr, q, side='left')
    ok = i < len(t_arr)
    ii = np.where(ok, i, 0)
    tt = np.where(ok, t_arr[ii] if len(t_arr) else np.nan, np.nan)
    vv = np.where(ok, v_arr[ii] if len(t_arr) else np.nan, np.nan)
    bad = (tt - q) > within
    tt[bad] = np.nan
    vv[bad] = np.nan
    return tt, vv


def last_at_or_before(t_arr, v_arr, q, within=120.0):
    q = np.asarray(q, dtype=float)
    i = np.searchsorted(t_arr, q, side='right') - 1
    ok = i >= 0
    ii = np.where(ok, i, 0)
    tt = np.where(ok, t_arr[ii] if len(t_arr) else np.nan, np.nan)
    vv = np.where(ok, v_arr[ii] if len(t_arr) else np.nan, np.nan)
    bad = (q - tt) > within
    tt[bad] = np.nan
    vv[bad] = np.nan
    return tt, vv


def boot_ci(values, groups, n_boot=2000, seed=0, stat='mean', weights=None):
    """Cluster bootstrap (resample groups) 95% CI of the mean (or sum per trade) of `values`."""
    v = np.asarray(values, dtype=float)
    g = np.asarray(groups)
    ok = ~np.isnan(v)
    v, g = v[ok], g[ok]
    if len(v) == 0:
        return np.nan, np.nan, np.nan
    codes, uniq = pd.factorize(g)
    G = len(uniq)
    s = np.bincount(codes, weights=v, minlength=G)
    c = np.bincount(codes, minlength=G).astype(float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, G, size=(n_boot, G))
    S = s[idx].sum(1)
    C = c[idx].sum(1)
    m = S / C
    return v.mean(), np.nanpercentile(m, 2.5), np.nanpercentile(m, 97.5)


def summarize(df, pnl_col, group_col='game_id', by=None, stake=STAKE, n_boot=2000):
    """per-trade P&L summary with clustered bootstrap CI. Returns DataFrame."""
    rows = []
    groups = [((), df)] if by is None else df.groupby(by, observed=True)
    for key, d in groups:
        v = d[pnl_col].values.astype(float)
        ok = ~np.isnan(v)
        if ok.sum() == 0:
            continue
        m, lo, hi = boot_ci(v[ok], d[group_col].values[ok], n_boot=n_boot)
        r = dict(n=int(ok.sum()), games=int(d.loc[ok, group_col].nunique()), pnl_total=float(v[ok].sum()),
                 pnl_mean=m, ci_lo=lo, ci_hi=hi, roi=m / stake, win_rate=float((v[ok] > 0).mean()))
        if by is not None:
            key = key if isinstance(key, tuple) else (key,)
            for b, k in zip(by if isinstance(by, list) else [by], key):
                r[b] = k
        rows.append(r)
    out = pd.DataFrame(rows)
    if by is not None and len(out):
        cols = (by if isinstance(by, list) else [by])
        out = out[cols + [c for c in out.columns if c not in cols]]
    return out


def norm_cdf(x):
    from scipy.special import ndtr
    return ndtr(x)


def norm_ppf(p):
    from scipy.special import ndtri
    return ndtri(p)
