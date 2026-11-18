"""Panel features shared by the hypothesis scripts."""
import json
import numpy as np
import pandas as pd
from scipy.special import ndtr, ndtri
from common import DL, load_games, fee_ps, STAKE

SIG0 = 12.34


def rem_seconds(elapsed):
    el = np.asarray(elapsed, dtype=float)
    reg = np.clip(2880.0 - el, 0, None)
    ot = el >= 2880
    ot_rem = 300.0 - ((el - 2880.0) % 300.0)
    return np.where(ot, ot_rem, reg), ot


def p_bm(m, elapsed, mu, sig=SIG0):
    rem, ot = rem_seconds(elapsed)
    tau = rem / 2880.0
    s = sig * np.sqrt(np.maximum(tau, 1e-9))
    mean = m + mu * tau
    p_gt = 1 - ndtr((0.5 - mean) / s)
    p_tie = ndtr((0.5 - mean) / s) - ndtr((-0.5 - mean) / s)
    q = ndtr(mu * (300 / 2880) / (sig * np.sqrt(300 / 2880)))
    return np.clip(p_gt + p_tie * q, 1e-6, 1 - 1e-6)


TB_EDGES = [-1, 60, 180, 360, 720, 1440, 2160, 2881]
TB_LABELS = ['0-1m', '1-3m', '3-6m', '6-12m', 'Q3(12-24m)', 'Q2(24-36m)', 'Q1(36-48m)']


def time_bucket(elapsed):
    rem, ot = rem_seconds(elapsed)
    b = pd.cut(rem, TB_EDGES, labels=TB_LABELS)
    b = b.astype(object)
    b[ot] = 'OT'
    return b


def load_panel(L=15, align_filter=True):
    p = pd.read_parquet(DL / 'panel.parquet')
    g = load_games()
    p = p.merge(g[['game_id', 'season', 'split', 'month', 'game_date_et', 'pre_ph', 'flag_print_gap_in_play', 'ml_fee_type', 'end_wc']], on='game_id')
    sfx = f'_L{L}'
    p['period'] = p['period' + sfx]; p['elapsed'] = p['elapsed' + sfx].astype(float)
    p['m'] = (p['hs' + sfx] - p['as' + sfx]).astype(float)
    p['since_score'] = p['since_score' + sfx]
    p['rem'], p['ot'] = rem_seconds(p.elapsed.values)
    p['tb'] = time_bucket(p.elapsed.values)
    p['mu'] = SIG0 * ndtri(p.pre_ph.clip(0.02, 0.98))
    p['p_bm'] = p_bm(p.m.values, p.elapsed.values, p.mu.values)
    p = p[~p.flag_print_gap_in_play]
    if align_filter:
        A = pd.read_parquet(DL / 'alignment_flags.parquet')
        p = p.merge(A[['game_id', 'align_ok']], on='game_id', how='left')
        p = p[p.align_ok.fillna(False)]
    return p


def pnl_hold(px, win, stake=STAKE):
    """buy at px (taker), hold to resolution: USDC P&L per `stake` USDC"""
    px = np.asarray(px, float)
    sh = stake / px
    return sh * (np.asarray(win, float) - px) - sh * fee_ps(px)


def pnl_exit(px_in, px_out, stake=STAKE):
    px_in = np.asarray(px_in, float); px_out = np.asarray(px_out, float)
    sh = stake / px_in
    return sh * (px_out - px_in) - sh * fee_ps(px_in) - sh * fee_ps(px_out)


def pnl_gross_hold(px, win, stake=STAKE):
    px = np.asarray(px, float)
    return stake / px * (np.asarray(win, float) - px)


def load_model_params():
    return json.load(open(DL / 'model_params.json'))


def p_emp_rows(m, elapsed, mu, P=None):
    """empirical bucket-logit model (fit_model.py, train only); OT -> BM with sigma_ot"""
    P = P or load_model_params()
    edges = P['EDGES']
    el = np.asarray(elapsed, float)
    rem_reg = np.clip(2880.0 - el, 0, None)
    bk = np.searchsorted(edges, rem_reg, side='right') - 1
    sq = np.sqrt(np.clip(rem_reg, 10, None) / 2880.0)
    th = np.array([P['emp'].get(str(int(k)), [np.nan] * 3) for k in bk])
    z = th[:, 0] + th[:, 1] * np.asarray(m, float) / sq / 10 + th[:, 2] * np.asarray(mu, float) * sq / 10
    pe = 1 / (1 + np.exp(-z))
    ot = el >= 2880
    pb = p_bm(m, elapsed, mu, sig=P['SIG_OT'])
    return np.where(ot, pb, pe)


def exit_px(p, buy_home, h, half_spread=0.01):
    """exit price for a position bought at decision row t, sold h seconds later.
    - if t + h is after the last ESPN play (game over): resolution value (0/1), no exit fee (returned flag fee=False)
    - else the first opposite-side print at rt >= t + h (within 120 s)
    - else (no print): fair mid at t + h minus `half_spread` (counted, flagged)
    Returns (price, fee_flag, source) arrays for the position's own token."""
    bh = np.asarray(buy_home, bool)
    over = (p.t.values + h) >= p.end_wc.values
    win_own = np.where(bh, p.home_win.values, 1 - p.home_win.values).astype(float)
    sell = np.where(bh, p[f'sellH_px_{h}'].values, p[f'sellA_px_{h}'].values)
    pm = p[f'pm_{h}'].values
    pm_own = np.where(bh, pm, 1 - pm) - half_spread
    px = np.where(over, win_own, np.where(np.isnan(sell), pm_own, sell))
    src = np.where(over, 'resolved', np.where(np.isnan(sell), 'mid-1c', 'print'))
    return px, ~over, src


def pnl_exit2(px_in, px_out, fee_out, stake=STAKE):
    px_in = np.asarray(px_in, float); px_out = np.asarray(px_out, float)
    sh = stake / px_in
    return sh * (px_out - px_in) - sh * fee_ps(px_in) - np.where(fee_out, sh * fee_ps(px_out), 0.0)
