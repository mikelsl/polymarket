"""Shared helpers for the nba_pre workstream (NBA pre-game strategy research). Read-only research.

Conventions used by every nba_pre script
- Fee: today's taker fee 0.05 x p x (1 - p) per share, applied to ALL periods (gross = no fee also reported).
- Stake: 10 USDC of shares per trade (shares = 10 / price); the fee is paid on top (cost = 10 + fee).
- Executable prices from the CLOB minute mid p: ask = next 1c tick strictly above p, bid = next tick strictly below
  (p on a half tick, as on >= 82% of pre-game minutes -> +-0.5c; p on a tick -> +-1c).
- Train = game date (ET) before 2026-02-01; test = 2026-02-01 .. 2026-11-02 (untouched until parameters are fixed).
- Bootstrap: 2,000 resamples of games (clustered by game), percentile 95% CI of total P&L and ROI.
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / 'data' / 'nba_data'
DATA = ROOT / 'data' / 'nba_pre'
RES = ROOT / 'results' / 'nba_pre'
DATA.mkdir(parents=True, exist_ok=True)
RES.mkdir(parents=True, exist_ok=True)
FEE = 0.05
STAKE = 10.0
TEST_START = '2026-02-01'


def fee(p, rate=FEE):
    return rate * p * (1 - p)


def ask_of_mid(p, tick=0.01):
    return np.clip((np.floor(np.round(np.asarray(p, float) / tick, 4)) + 1) * tick, 0.001, 0.999)


def bid_of_mid(p, tick=0.01):
    return np.clip((np.ceil(np.round(np.asarray(p, float) / tick, 4)) - 1) * tick, 0.001, 0.999)


def american_to_prob(a):
    a = np.asarray(a, float)
    return np.where(a < 0, -a / (-a + 100.0), 100.0 / (a + 100.0))


def devig_mult(i1, i2):
    s = i1 + i2
    return i1 / s


def devig_power(i1, i2, iters=60):
    """k such that i1^k + i2^k = 1 (bisection); returns fair prob of outcome 1."""
    i1 = np.asarray(i1, float)
    i2 = np.asarray(i2, float)
    lo, hi = np.full(i1.shape, 0.5), np.full(i1.shape, 3.0)
    for _ in range(iters):
        k = (lo + hi) / 2
        f = i1 ** k + i2 ** k - 1
        lo = np.where(f > 0, k, lo)
        hi = np.where(f > 0, hi, k)
    k = (lo + hi) / 2
    return i1 ** k


def split_of(date_et):
    return np.where(pd.Series(date_et).astype(str).values < TEST_START, 'train', 'test')


def core_games():
    """Core games with the fields every nba_pre script needs."""
    g = pd.read_parquet(SRC / 'games.parquet')
    g = g[g.is_core_game].copy()
    g['sched'] = g.espn_sched_utc.fillna(g.game_start_utc)
    tip = g.espn_tipoff_utc_fixed
    ok = (tip - g.sched).dt.total_seconds().between(-5 * 60, 45 * 60)
    g['tip'] = tip.where(ok, g.sched + pd.Timedelta(minutes=11))
    g['tip_source'] = np.where(ok, 'espn', 'sched+11m')
    g['home_idx'] = np.where(g.home_away_swapped_vs_espn.fillna(False), 0, 1).astype(int)
    g['y0'] = (g.ml_winner_index == 0).astype(int)
    g['split'] = split_of(g.game_date_et)
    g['month'] = g.game_date_et.astype(str).str[:7]
    return g.reset_index(drop=True)


def trade_pnl(price, y, stake=STAKE, rate=FEE):
    """Buy `stake` USDC of a token at `price`, hold to resolution y (1/0). Returns pnl, cost, gross pnl."""
    price = np.asarray(price, float)
    sh = stake / price
    f = sh * fee(price, rate)
    gross = sh * (np.asarray(y, float) - price)
    return gross - f, stake + f, gross


def boot(df, pnl='pnl', cost='cost', cluster='game_id', B=2000, seed=7):
    """Cluster bootstrap (by game) of total P&L and ROI. Returns dict of point estimates and 95% CIs."""
    if len(df) == 0:
        return dict(n=0)
    a = df.groupby(cluster)[[pnl, cost]].sum().to_numpy()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), size=(B, len(a)))
    s = a[idx].sum(axis=1)
    tot, roi = s[:, 0], s[:, 0] / s[:, 1]
    return dict(pnl=a[:, 0].sum(), roi=a[:, 0].sum() / a[:, 1].sum(),
                pnl_lo=np.percentile(tot, 2.5), pnl_hi=np.percentile(tot, 97.5),
                roi_lo=np.percentile(roi, 2.5), roi_hi=np.percentile(roi, 97.5),
                p_pos=float((tot > 0).mean()))


def summarize(df, by=None, pnl='pnl', cost='cost', gross='gross', B=2000):
    """Table of n, games, avg price, win rate, P&L after fee (10 USDC stakes), ROI with cluster-bootstrap CI, gross."""
    rows = []
    groups = [((), df)] if by is None else df.groupby(by, observed=True)
    for k, d in groups:
        b = boot(d, pnl, cost, B=B)
        if not b.get('n', 1):
            continue
        r = dict(zip(by if isinstance(by, list) else [by], k if isinstance(k, tuple) else (k,))) if by else {}
        r.update(n=len(d), games=d.game_id.nunique(), avg_price=d.price.mean(), win=d.y.mean(),
                 pnl=b['pnl'], pnl_per_trade=b['pnl'] / len(d), roi=b['roi'], roi_lo=b['roi_lo'],
                 roi_hi=b['roi_hi'], pnl_lo=b['pnl_lo'], pnl_hi=b['pnl_hi'],
                 gross_pnl=d[gross].sum(), gross_roi=d[gross].sum() / (STAKE * len(d)))
        rows.append(r)
    return pd.DataFrame(rows)


def fmt(df, nd=3):
    out = df.copy()
    for c in out.columns:
        if out[c].dtype.kind == 'f':
            out[c] = out[c].round(nd)
    return out
