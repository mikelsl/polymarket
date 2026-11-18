"""H1 - calibration of pre-game moneyline prices (favourite-longshot bias, home/away, by bucket, season, month).

Price: CLOB minute mid of the moneyline at the entry time E (main E = scheduled start `s0`, i.e. ~11 min before the
real tip; robustness E = tip - 1 min `m1`). Outcomes: Gamma resolution (agrees with ESPN on every core game).
Trades: buy 10 USDC at ask = next tick above the mid, hold to resolution, fee 0.05 p (1-p) per share.

Outputs (results/nba_pre/):
  h1_calib_buckets.csv   favourite-price buckets: n, mean price, win rate, win - price, z, by split
  h1_logit.csv           logistic fits y_home ~ a + b logit(p_home) per split/season (b > 1 = longshots overpriced)
  h1_residual_by.csv     mean (y - p) of the home token by season / month / game type
  h1_strategies.csv      every pre-registered rule, train and test, P&L after fee with cluster bootstrap CI
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm

from common_pre import DATA, RES, ask_of_mid, fmt, summarize, trade_pnl

p = pd.read_parquet(DATA / 'ml_panel.parquet')
out_b, out_l, out_r, out_s = [], [], [], []
for E in ('s0', 'm1', 'm60'):
    d = p.dropna(subset=[f'mid0_{E}']).copy()
    d['p0'] = d[f'mid0_{E}']
    d['p_home'] = np.where(d.home_idx == 0, d.p0, 1 - d.p0)
    d['y_home'] = np.where(d.home_idx == 0, d.y0, 1 - d.y0)
    d['p_fav'] = np.maximum(d.p0, 1 - d.p0)
    d['y_fav'] = np.where(d.p0 >= 0.5, d.y0, 1 - d.y0)
    d['fav_home'] = (d.p_home >= 0.5)
    # ---------------- buckets ----------------
    bins = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0001]
    d['bucket'] = pd.cut(d.p_fav, bins, right=False)
    for sp in ('train', 'test', 'all'):
        x = d if sp == 'all' else d[d.split == sp]
        b = x.groupby('bucket', observed=True).agg(n=('y_fav', 'size'), price=('p_fav', 'mean'), win=('y_fav', 'mean'))
        b['diff'] = b.win - b.price
        b['se'] = np.sqrt(b.price * (1 - b.price) / b.n)
        b['z'] = b['diff'] / b.se
        b = b.reset_index().assign(entry=E, split=sp)
        b['bucket'] = b.bucket.astype(str)
        out_b.append(b)
    # ---------------- logistic fits ----------------
    d['lp'] = np.log(d.p_home.clip(0.005, 0.995) / (1 - d.p_home.clip(0.005, 0.995)))
    for key, x in [('train', d[d.split == 'train']), ('test', d[d.split == 'test']), ('all', d)] + \
                  [(s, x) for s, x in d.groupby('season')]:
        m = sm.Logit(x.y_home, sm.add_constant(x[['lp']])).fit(disp=0)
        out_l.append(dict(entry=E, sample=key, n=len(x), a_home=m.params['const'], a_se=m.bse['const'],
                          b_slope=m.params['lp'], b_se=m.bse['lp'],
                          brier=np.mean((x.p_home - x.y_home) ** 2),
                          logloss=-np.mean(x.y_home * np.log(x.p_home.clip(1e-3, 1)) +
                                           (1 - x.y_home) * np.log((1 - x.p_home).clip(1e-3, 1)))))
    # ---------------- residuals ----------------
    d['res_home'] = d.y_home - d.p_home
    for by in ('season', 'month', 'game_type', 'split'):
        r = d.groupby(by).agg(n=('res_home', 'size'), p_home=('p_home', 'mean'), y_home=('y_home', 'mean'),
                              res=('res_home', 'mean'), res_sd=('res_home', 'std')).reset_index()
        r['z'] = r.res / (r.res_sd / np.sqrt(r.n))
        out_r.append(r.rename(columns={by: 'level'}).assign(by=by, entry=E))
    # ---------------- strategies ----------------
    toks = []
    for ti in (0, 1):
        t = d[['game_id', 'split', 'season', 'month', 'home_idx', 'p0', 'y0']].copy()
        t['tok'] = ti
        t['mid'] = np.where(ti == 0, t.p0, 1 - t.p0)
        t['y'] = np.where(ti == 0, t.y0, 1 - t.y0)
        t['home'] = (t.home_idx == ti)
        toks.append(t)
    t = pd.concat(toks, ignore_index=True)
    t['price'] = ask_of_mid(t.mid)
    t['pnl'], t['cost'], t['gross'] = trade_pnl(t.price, t.y)
    rules = {}
    for x in (0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95):
        rules[f'fav_mid>={x}'] = t.mid >= x if x > 0.5 else t.mid > 0.5
    for x in (0.5, 0.4, 0.3, 0.2, 0.15, 0.1):
        rules[f'dog_mid<{x}'] = t.mid < x
    for lo, hi in ((0.05, 0.2), (0.2, 0.35), (0.35, 0.5), (0.5, 0.65), (0.65, 0.8), (0.8, 0.95)):
        rules[f'band_{lo}-{hi}'] = (t.mid >= lo) & (t.mid < hi)
    rules['home_all'] = t.home
    rules['away_all'] = ~t.home
    rules['home_fav'] = t.home & (t.mid > 0.5)
    rules['home_dog'] = t.home & (t.mid < 0.5)
    rules['away_fav'] = ~t.home & (t.mid > 0.5)
    rules['away_dog'] = ~t.home & (t.mid < 0.5)
    for name, mask in rules.items():
        s = summarize(t[mask], by='split', B=1000)
        out_s.append(s.assign(rule=name, entry=E))

pd.concat(out_b).pipe(fmt, 4).to_csv(RES / 'h1_calib_buckets.csv', index=False)
pd.DataFrame(out_l).pipe(fmt, 4).to_csv(RES / 'h1_logit.csv', index=False)
pd.concat(out_r).pipe(fmt, 4).to_csv(RES / 'h1_residual_by.csv', index=False)
S = pd.concat(out_s)
S = S[['entry', 'rule', 'split', 'n', 'games', 'avg_price', 'win', 'pnl', 'pnl_per_trade', 'roi', 'roi_lo', 'roi_hi',
       'gross_pnl', 'gross_roi']]
S.pipe(fmt, 4).to_csv(RES / 'h1_strategies.csv', index=False)
print('variants (rule x entry):', S.groupby(['entry', 'rule']).ngroups)

# ---------------- heavy favourites (follow-up of the train-chosen threshold fav >= 0.90) ----------------
from scipy.stats import beta as _beta  # noqa: E402

hv = []
for E in ('s0', 'm1', 'm60', 'm300', 'm5_print'):
    if E == 'm5_print':     # entry at the last print-revealed ask in the 10 min before tip - 5 min
        d = p.dropna(subset=['mid0_m5']).copy()
        m0 = d.mid0_m5
        fav1 = m0 < 0.5
        price = np.where(fav1, 1 - d.pbid0_m5, d.pask0_m5)
    else:
        d = p.dropna(subset=[f'mid0_{E}']).copy()
        m0 = d[f'mid0_{E}']
        fav1 = m0 < 0.5
        price = ask_of_mid(np.where(fav1, 1 - m0, m0))
    pf = np.where(fav1, 1 - m0, m0)
    yf = np.where(fav1, 1 - d.y0, d.y0)
    pb = np.where(fav1, 1 - d.pbook0_close_pow, d.pbook0_close_pow)
    for lo in (0.85, 0.90, 0.93):
        for sp in ('train', 'test', 'all', '2024-25', '2025-26', '2026-27'):
            s = (pf >= lo) & ~np.isnan(price)
            s &= (d.split == sp).to_numpy() if sp in ('train', 'test') else \
                ((d.season == sp).to_numpy() if sp != 'all' else True)
            if s.sum() == 0:
                continue
            pnl, cost, gross = trade_pnl(price[s], yf[s])
            k, n = int(yf[s].sum()), int(s.sum())
            hv.append(dict(entry=E, min_fav=lo, sample=sp, n=n, wins=k, win=k / n, mid=pf[s].mean(),
                           ask=np.mean(price[s]), book_pow_close=np.nanmean(pb[s]),
                           win_lo95=_beta.ppf(0.025, k, n - k + 1) if k > 0 else 0.0,
                           pnl=pnl.sum(), roi=pnl.sum() / cost.sum(), gross_roi=gross.sum() / (10 * n)))
pd.DataFrame(hv).pipe(fmt, 4).to_csv(RES / 'h1_heavy_fav.csv', index=False)
