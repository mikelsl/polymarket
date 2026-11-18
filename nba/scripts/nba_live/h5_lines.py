"""H5: in-play totals and spreads (2025-26+), decision-time line = line closest to the sportsbook close (pre-game info).

Models fitted on TRAIN games only (ESPN scores + sportsbook close; 2024-25 and 2025-26 to 2026-01-31), per remaining-time
bucket (60-s game-clock checkpoints):
  totals : remaining points R = a + b * B*tau + c * D,   D = points so far - B*(1-tau)  (pace deviation), B = book close total
  spreads: remaining margin X = a + b * mu*tau + c * m,  m = current home margin, mu = -book home spread close
  residual SD per bucket; P(over) = 1 - Phi((line - C - E[R]) / sd), P(home covers) = Phi((m + E[X] + home_hcp) / sd).
Market: decision-time line prints (token 0 = Over / outcome 0 team), real time rt = ts - 4 s. 30-s grid from tip+60 s to the
last play; ESPN state at t - 15 s - game lag (aligned games only); market mid from last ask/bid prints; fills at the first
print >= t + 2 s on our side (within 60 s).
Tests: (a) outcome residual (win - mid) by pace deviation (totals overreaction to early pace) and by model-minus-market
deviation; (b) rules: first trigger per game with |model - mid| >= theta (0.05, 0.08, 0.12) in windows (Q1-Q2, Q3-Q4>2m),
buy the model side at the ask, hold to resolution. 0.05 fee.
Outputs: results/nba_live/h5_*.csv, data/nba_live/h5_panel_<kind>.parquet
"""
import sys
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.special import ndtr
from common import DL, RL, ROOT, NBA, LAG, load_games, load_states, game_slices, first_at_or_after, last_at_or_before, summarize, fee_ps
from features import pnl_hold, pnl_gross_hold

KIND = sys.argv[1] if len(sys.argv) > 1 else 'total'
LINFO = float(sys.argv[2]) if len(sys.argv) > 2 else 15.0      # ESPN information delay (s) on top of the per-game lag
SFX = '' if LINFO == 15 else f'_L{int(LINFO)}'
g = load_games()
AL = pd.read_parquet(DL / 'alignment_flags.parquet').set_index('game_id')
EDGES = [0, 60, 180, 360, 720, 1080, 1440, 1800, 2160, 2520, 2881]

# ---------- model fit on train (ESPN only) ----------
pbp = pd.read_parquet(NBA / 'espn_pbp.parquet', columns=['espn_event_id', 'seq', 'elapsed_s', 'home_score', 'away_score'])
bad = set(g.loc[g.flag_final_score_pbp_mismatch, 'espn_event_id'])
gg = g[~g.espn_event_id.isin(bad)].dropna(subset=['book_total_close', 'book_spread_home_close'])
pbp = pbp.merge(gg[['espn_event_id', 'game_id', 'split', 'book_total_close', 'book_spread_home_close', 'espn_home_score', 'espn_away_score']], on='espn_event_id')
pbp = pbp.sort_values(['game_id', 'elapsed_s', 'seq'])
rows = []
for gid, d in pbp.groupby('game_id', sort=False):
    el = d.elapsed_s.values.astype(float)
    cps = np.arange(0, 2880, 60.0)
    ix = np.clip(np.searchsorted(el, cps, side='right') - 1, 0, None)
    r0 = d.iloc[0]
    rows.append(pd.DataFrame({'game_id': gid, 'split': r0.split, 'B': r0.book_total_close, 'mu': -r0.book_spread_home_close, 'el': cps,
                              'hs': d.home_score.values[ix], 'as': d.away_score.values[ix],
                              'TF': r0.espn_home_score + r0.espn_away_score, 'MF': r0.espn_home_score - r0.espn_away_score}))
X = pd.concat(rows, ignore_index=True)
X['tau'] = (2880 - X.el) / 2880
X['C'] = X.hs + X['as']; X['m'] = X.hs - X['as']
X['D'] = X.C - X.B * (1 - X.tau)
X['R'] = X.TF - X.C; X['XM'] = X.MF - X.m
X['rem'] = 2880 - X.el
X['bk'] = np.searchsorted(EDGES, X.rem, side='right') - 1
params = {}
fitrows = []
for bk, d in X[X.split == 'train'].groupby('bk'):
    if KIND == 'total':
        F = sm.add_constant(pd.DataFrame({'Bt': d.B * d.tau, 'D': d.D}))
        mm = sm.OLS(d.R, F).fit()
    else:
        F = sm.add_constant(pd.DataFrame({'mut': d.mu * d.tau, 'm': d.m}))
        mm = sm.OLS(d.XM, F).fit()
    params[bk] = (mm.params.values, float(np.sqrt(mm.scale)))
    fitrows.append(dict(kind=KIND, bucket_rem_from=EDGES[bk], n=len(d), coef=[round(v, 4) for v in mm.params.values], resid_sd=np.sqrt(mm.scale)))
pd.DataFrame(fitrows).to_csv(RL / f'h5_fit_{KIND}.csv', index=False)
print(pd.DataFrame(fitrows).to_string(), flush=True)


def model_prob(C_or_m, el, B_or_mu, line_or_hcp):
    rem = np.clip(2880 - el, 0, None)
    tau = rem / 2880
    bk = np.clip(np.searchsorted(EDGES, rem, side='right') - 1, min(params), max(params))
    co = np.array([params[k][0] for k in bk]); sd = np.array([params[k][1] for k in bk])
    if KIND == 'total':
        D = C_or_m - B_or_mu * (1 - tau)
        ER = co[:, 0] + co[:, 1] * B_or_mu * tau + co[:, 2] * D
        return 1 - ndtr((line_or_hcp - C_or_m - ER) / sd)          # P(final total > line); lines are x.5
    ER = co[:, 0] + co[:, 1] * B_or_mu * tau + co[:, 2] * C_or_m
    return ndtr((C_or_m + ER + line_or_hcp) / sd)                     # P(home margin + home handicap > 0)


# ---------- market panel ----------
sel = pd.read_parquet(DL / 'lines_selected.parquet')
sel = sel[sel.sel_kind == KIND].merge(g[['game_id', 'split', 'season', 'month', 'tip', 'end_wc', 'flag_print_gap_in_play', 'book_total_close', 'book_spread_home_close',
                                         'espn_home_score', 'espn_away_score']], on='game_id')
S = load_states(); ssl = game_slices(S)
G0 = pd.read_parquet(NBA / 'games.parquet', columns=['game_id', 'espn_home_name'])
sel = sel.merge(G0, on='game_id')
out = []
for r in sel.itertuples(index=False):
    if r.flag_print_gap_in_play or r.game_id not in ssl or not bool(AL.align_ok.get(r.game_id, False)) or pd.isna(r.winner_index) or r.winner_index < 0:
        continue
    f1 = ROOT / 'data' / 'nba_data' / 'work' / 'trades_parts' / f'{r.market_id}.parquet'
    f2 = DL / 'trades_lines' / f'{r.market_id}.parquet'
    f = f1 if f1.exists() else f2
    if not f.exists():
        continue
    x = pd.read_parquet(f, columns=['ts', 'token_index', 'side', 'price'])
    x = x[x.token_index >= 0].sort_values('ts')
    rt = x.ts.values.astype(float) - LAG
    tok0 = x.token_index.values == 0
    buy = x.side.values == 'BUY'
    p0 = np.where(tok0, x.price.values, 1 - x.price.values)
    d = np.where(tok0 == buy, 1, -1)
    At, Ap, Bt, Bp = rt[d == 1], p0[d == 1], rt[d == -1], p0[d == -1]
    if len(At) + len(Bt) < 20:
        continue
    lag = float(AL.lag_used[r.game_id])
    a2, b2 = ssl[r.game_id]
    wc = S.wc.values[a2:b2]
    kh, ka, ke = S.k_home_score.values[a2:b2], S.k_away_score.values[a2:b2], S.k_elapsed_s.values[a2:b2]
    grid = np.arange(r.tip + 60.0, r.end_wc, 30.0)
    ix = np.searchsorted(wc, grid - LINFO - lag, side='right') - 1
    ok = ix >= 0
    ix = np.clip(ix, 0, None)
    hs = np.where(ok, kh[ix], 0); as_ = np.where(ok, ka[ix], 0); el = np.where(ok, ke[ix], 0).astype(float)
    _, pa = last_at_or_before(At, Ap, grid, within=300)
    _, pb = last_at_or_before(Bt, Bp, grid, within=300)
    mid = (pa + pb) / 2
    ta_, fa = first_at_or_after(At, Ap, grid + 2, within=60)      # buy token0
    tb_, fb = first_at_or_after(Bt, Bp, grid + 2, within=60)      # buy token1 = 1 - bid-side print
    if KIND == 'total':
        line = r.total_line
        pm = model_prob(hs + as_, el, r.book_total_close, line)
        win0 = int(r.espn_home_score + r.espn_away_score > line)
        Dv = (hs + as_) - r.book_total_close * (el / 2880)
        pm_naive = np.full(len(grid), np.nan); t0_home = True
    else:
        t0_home = r.team0 == r.espn_home_name
        hcp_home = r.home_hcp
        ph_cover = model_prob((hs - as_).astype(float), el, -r.book_spread_home_close, hcp_home)
        pm = ph_cover if t0_home else 1 - ph_cover
        win0 = int(r.winner_index == 0)
        Dv = np.where(t0_home, 1, -1) * ((hs - as_) - (-r.book_spread_home_close) * (el / 2880))
        # naive Brownian margin model (sigma 12.34 per 48 min, drift = -book spread, no mean reversion)
        tau_n = np.clip(2880 - el, 10, None) / 2880
        ph_naive = ndtr(((hs - as_) + (-r.book_spread_home_close) * tau_n + hcp_home) / (12.34 * np.sqrt(tau_n)))
        pm_naive = ph_naive if t0_home else 1 - ph_naive
    out.append(pd.DataFrame({'game_id': r.game_id, 'split': r.split, 'season': r.season, 'month': r.month, 't': grid, 'el': el,
                             'rem': np.clip(2880 - el, 0, None), 'ot': el >= 2880, 'mid': mid, 'ask0': fa, 'ask1': 1 - fb, 'wait0': ta_ - grid, 'wait1': tb_ - grid,
                             'pm': pm, 'win0': win0,
                             'dev_pace': Dv, 'line_dist': r.dist, 'pm_naive': pm_naive, 'm_home': (hs - as_).astype(float),
                             't0_home': bool(t0_home), 'line0': (r.handicap0 if KIND == 'spread' else r.total_line),
                             'mu_home': -r.book_spread_home_close, 'B': r.book_total_close}))
Pn = pd.concat(out, ignore_index=True)
Pn.to_parquet(DL / f'h5_panel_{KIND}{SFX}.parquet', index=False)
Pn = Pn[Pn.mid.between(0.03, 0.97) & ~Pn.ot]
print('panel rows', len(Pn), 'games', Pn.game_id.nunique(), Pn.groupby('split').game_id.nunique().to_dict(), flush=True)
ll = lambda q, y: -(y * np.log(np.clip(q, 1e-4, 1 - 1e-4)) + (1 - y) * np.log(1 - np.clip(q, 1e-4, 1 - 1e-4)))
for sp, d in Pn.groupby('split'):
    print(sp, 'log loss market %.4f model %.4f' % (ll(d.mid, d.win0).mean(), ll(d.pm, d.win0).mean()), flush=True)
Pn['res'] = Pn.win0 - Pn.mid
Pn['dev'] = Pn.pm - Pn.mid
rows = []
for sp, d0 in Pn.groupby('split'):
    for (lo, hi, nm) in [(2160, 2881, 'Q1'), (1440, 2160, 'Q2'), (720, 1440, 'Q3'), (0, 720, 'Q4')]:
        d = d0[(d0.rem >= lo) & (d0.rem < hi)]
        mm = sm.OLS(d.res, sm.add_constant(d[['dev_pace']] / 10)).fit(cov_type='cluster', cov_kwds={'groups': pd.factorize(d.game_id)[0]})
        m2 = sm.OLS(d.res, sm.add_constant(d[['dev']])).fit(cov_type='cluster', cov_kwds={'groups': pd.factorize(d.game_id)[0]})
        rows.append(dict(kind=KIND, split=sp, q=nm, rows=len(d), games=d.game_id.nunique(), slope_pace_per10=mm.params['dev_pace'], se_pace=mm.bse['dev_pace'],
                         slope_dev=m2.params['dev'], se_dev=m2.bse['dev']))
R1 = pd.DataFrame(rows); R1.to_csv(RL / f'h5_resid_{KIND}{SFX}.csv', index=False)
print(R1.round(4).to_string(), flush=True)
# rules
Pn = Pn.sort_values(['game_id', 't'])
T = []
for th in (0.05, 0.08, 0.12):
    for wn, (lo, hi) in {'Q1-Q2': (1440, 2881), 'Q3-Q4(>2m)': (120, 1440)}.items():
        x = Pn[(Pn.dev.abs() >= th) & (Pn.rem >= lo) & (Pn.rem < hi)].groupby('game_id').head(1)
        b0 = (x.dev > 0).values
        px = np.where(b0, x.ask0, x.ask1)
        win = np.where(b0, x.win0, 1 - x.win0)
        T.append(pd.DataFrame({'theta': th, 'window': wn, 'game_id': x.game_id.values, 'split': x.split.values, 'px': px, 'win': win,
                               'pnl': pnl_hold(px, win), 'gross': pnl_gross_hold(px, win)}))
    # pace-fade rule (totals overreaction): after Q1 (rem < 2160) when |pace dev| >= 8 pts, buy the side against the pace
for pth in (6.0, 10.0):
    for wn, (lo, hi) in {'Q2': (1440, 2160), 'Q3': (720, 1440)}.items():
        x = Pn[(Pn.dev_pace.abs() >= pth) & (Pn.rem >= lo) & (Pn.rem < hi)].groupby('game_id').head(1)
        b0 = (x.dev_pace < 0).values     # pace below expectation -> buy token0 (Over / outcome-0 team) = fade the pace
        px = np.where(b0, x.ask0, x.ask1)
        win = np.where(b0, x.win0, 1 - x.win0)
        T.append(pd.DataFrame({'theta': -pth, 'window': 'fadepace-' + wn, 'game_id': x.game_id.values, 'split': x.split.values, 'px': px, 'win': win,
                               'pnl': pnl_hold(px, win), 'gross': pnl_gross_hold(px, win)}))
T = pd.concat(T, ignore_index=True); T = T[T.px.between(0.02, 0.98)]
T.to_parquet(DL / f'h5_trades_{KIND}{SFX}.parquet', index=False)
R = summarize(T, 'pnl', by=['theta', 'window', 'split'], n_boot=1000)
R = R.merge(T.groupby(['theta', 'window', 'split']).agg(px=('px', 'mean'), win=('win', 'mean'), gross=('gross', 'mean')).reset_index(), on=['theta', 'window', 'split'])
R.to_csv(RL / f'h5_rules_{KIND}{SFX}.csv', index=False)
pd.set_option('display.width', 250)
print(R.round(3).to_string())
