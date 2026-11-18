"""Shared helpers for the soccer_strat workstream (read-only research; never places orders).

Run scripts as `.venv/bin/python -I scripts/soccer_strat/<script>.py` from the project root.
Every script inserts this directory into sys.path explicitly.
"""
import datetime as dt
import os
import sys

import numpy as np
import pandas as pd

ROOT = 'polymarket/research/auto-20261104'
SD = f'{ROOT}/data/soccer_data'
DS = f'{ROOT}/data/soccer_strat'
RS = f'{ROOT}/results/soccer_strat'
os.makedirs(DS, exist_ok=True)
os.makedirs(RS, exist_ok=True)

SPLIT_TS = int(dt.datetime(2026, 2, 1, tzinfo=dt.timezone.utc).timestamp())  # train: kickoff < SPLIT_TS
FEE_RATE = 0.05  # today's taker fee, applied to every period
STAKE = 10.0     # USDC per trade
ROLES = ['home', 'draw', 'away']


def log(*a):
    print(dt.datetime.now().strftime('%H:%M:%S'), *a, flush=True)


def fee_per_share(p, rate=FEE_RATE):
    p = np.asarray(p, dtype=float)
    return rate * p * (1.0 - p)


# ----------------------------------------------------------------------------------------------
# matches + odds
# ----------------------------------------------------------------------------------------------

def devig_prop(o):
    """Proportional de-vig of an (n, k) array of decimal odds -> probabilities."""
    q = 1.0 / np.asarray(o, dtype=float)
    return q / q.sum(axis=1, keepdims=True)


def devig_power(o, iters=60):
    """Power method: p_i = q_i^k with k such that sum p_i = 1 (handles favourite-longshot margin)."""
    q = 1.0 / np.asarray(o, dtype=float)
    out = np.full_like(q, np.nan)
    ok = np.all(np.isfinite(q), axis=1) & np.all(q > 0, axis=1)
    lo = np.full(q.shape[0], 0.5)
    hi = np.full(q.shape[0], 3.0)
    for _ in range(iters):
        k = 0.5 * (lo + hi)
        s = np.nansum(np.where(ok[:, None], q, 0.5) ** k[:, None], axis=1)
        hi = np.where(s < 1, k, hi)
        lo = np.where(s >= 1, k, lo)
    k = 0.5 * (lo + hi)
    out[ok] = (q[ok] ** k[ok, None])
    out[ok] = out[ok] / out[ok].sum(axis=1, keepdims=True)
    return out


def load_matches(ready_only=True):
    m = pd.read_parquet(f'{SD}/matches.parquet')
    if ready_only:
        m = m[m.analysis_ready].copy()
    od = pd.read_parquet(f'{SD}/odds.parquet')
    od = od.drop(columns=['league', 'season', 'match_id', 'home_team', 'away_team'])
    m = m.merge(od, on='espn_event_id', how='left', suffixes=('', '_fd'))
    m['train'] = m.kickoff_ts < SPLIT_TS
    m['period'] = np.where(m.train, 'train', 'test')
    m['month'] = pd.to_datetime(m.kickoff_ts, unit='s', utc=True).dt.strftime('%Y-%m')
    # football-data columns are in ESPN orientation; Polymarket orientation is reversed when swapped
    sw = m.pm_orientation_swapped.fillna(False).astype(bool).values
    for pre in ['PS', 'PSC', 'Avg', 'AvgC', 'BFE', 'BFEC', 'B365', 'B365C', 'Max', 'MaxC']:
        h, a = m[pre + 'H'].values.copy(), m[pre + 'A'].values.copy()
        m[pre + 'H'] = np.where(sw, a, h)
        m[pre + 'A'] = np.where(sw, h, a)
    # fair probabilities, Polymarket orientation
    for pre, name in [('PSC', 'pin_c'), ('PS', 'pin_o'), ('BFEC', 'bfe_c'), ('BFE', 'bfe_o'), ('AvgC', 'avg_c'),
                      ('Avg', 'avg_o'), ('B365C', 'b365_c')]:
        o = m[[pre + 'H', pre + 'D', pre + 'A']].values
        p = devig_power(o) if pre.startswith(('Avg', 'B365')) else devig_prop(o)
        for i, r in enumerate(ROLES):
            m[f'{name}_{r}'] = p[:, i]
    for pre, name in [('PC', 'pin_c'), ('P', 'pin_o'), ('BFEC', 'bfe_c'), ('BFE', 'bfe_o'), ('AvgC', 'avg_c'),
                      ('Avg', 'avg_o')]:
        o = m[[pre + '_over25', pre + '_under25']].values.astype(float)
        o = np.where(o > 1.0, o, np.nan)
        p = devig_power(o) if pre.startswith('Avg') else devig_prop(o)
        m[f'{name}_over'] = p[:, 0]
    # best available sharp close: Pinnacle, else Betfair exchange, else market average (power de-vig)
    for r in ROLES + ['over']:
        m[f'fair_c_{r}'] = m[f'pin_c_{r}'].fillna(m[f'bfe_c_{r}']).fillna(m[f'avg_c_{r}'])
        m[f'fair_o_{r}'] = m[f'pin_o_{r}'].fillna(m[f'bfe_o_{r}']).fillna(m[f'avg_o_{r}'])
        m[f'bfx_c_{r}'] = m[f'bfe_c_{r}'].fillna(m[f'avg_c_{r}'])  # deployable in both periods
        m[f'bfx_o_{r}'] = m[f'bfe_o_{r}'].fillna(m[f'avg_o_{r}'])
    m['res_home'] = (m.pm_result == 'H').astype(int)
    m['res_draw'] = (m.pm_result == 'D').astype(int)
    m['res_away'] = (m.pm_result == 'A').astype(int)
    m['total_goals'] = m.home_score + m.away_score
    m['res_over'] = (m.total_goals > 2.5).astype(int)
    m['res_btts'] = ((m.home_score > 0) & (m.away_score > 0)).astype(int)
    return m.reset_index(drop=True)


def load_trades(cols=None, filters=None):
    base = ['match_id', 'role', 'outcome_index', 'side', 'price', 'size', 'ts', 't_rel_kickoff_s', 'yes_price',
            'taker_buys_yes']
    cols = cols or base
    tr = pd.read_parquet(f'{SD}/trades.parquet', columns=cols, filters=filters)
    return tr


# ----------------------------------------------------------------------------------------------
# statistics
# ----------------------------------------------------------------------------------------------

def cluster_boot(df, pnl='pnl', stake='stake', cluster='match_id', n_boot=2000, seed=7):
    """Bootstrap (resample clusters with replacement) 95% CI of mean P&L per trade, total P&L and ROI."""
    if len(df) == 0:
        return dict(n=0, matches=0, pnl=0.0, pnl_per_trade=np.nan, roi=np.nan, ci_trade=(np.nan, np.nan),
                    ci_roi=(np.nan, np.nan), ci_total=(np.nan, np.nan))
    g = df.groupby(cluster).agg(p=(pnl, 'sum'), s=(stake, 'sum'), n=(pnl, 'size'))
    P, S, N = g.p.values, g.s.values, g.n.values.astype(float)
    rng = np.random.default_rng(seed)
    k = len(g)
    W = rng.multinomial(k, np.full(k, 1.0 / k), size=n_boot).astype(float)
    tp, ts, tn = W @ P, W @ S, W @ N
    per = tp / tn
    roi = tp / ts
    q = lambda x: (float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5)))
    return dict(n=int(N.sum()), matches=k, pnl=float(P.sum()), pnl_per_trade=float(P.sum() / N.sum()),
                roi=float(P.sum() / S.sum()), ci_trade=q(per), ci_roi=q(roi), ci_total=q(tp))


def summarize(df, by=None, **kw):
    rows = []
    if by is None:
        r = cluster_boot(df, **kw)
        return pd.DataFrame([r])
    for key, sub in df.groupby(by, observed=True):
        r = cluster_boot(sub, **kw)
        if not isinstance(key, tuple):
            key = (key,)
        rows.append(dict(zip(by if isinstance(by, list) else [by], key)) | r)
    return pd.DataFrame(rows)


def fmt_ci(t):
    return f'[{t[0]:+.3f},{t[1]:+.3f}]'


def pretty(df):
    d = df.copy()
    for c in ['ci_trade', 'ci_roi', 'ci_total']:
        if c in d:
            d[c] = d[c].map(fmt_ci)
    for c in ['pnl', 'pnl_per_trade', 'roi']:
        if c in d:
            d[c] = d[c].round(4)
    return d
