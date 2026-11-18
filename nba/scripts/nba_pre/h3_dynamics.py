"""H3 - moneyline price dynamics before tip: information arrival, drift, momentum / reversal, best entry time,
overnight vs game day, convergence to the sportsbook.

Prices of token 0 at time points before the real tip (ESPN first play):
  h24 h12 h6 h3 : median print mid-proxy in the hour before tip - k hours (prints; mid-proxy = ask - 0.5c / bid + 0.5c)
  m300 .. m1    : CLOB minute mid at tip - k minutes
Trades
  to_tip  : buy at ask(t) (next tick above the mid), sell at bid(m1) (next tick below the tip mid); fees on both legs
  to_res  : buy at ask(t), hold to resolution
Rules (one trade per game, the token selected by the rule; thresholds chosen on train):
  drift_fav / drift_dog : buy the favourite / underdog at t, exit at tip
  mom(t1,t2,x)  : if token 0 moved >= x between t1 and t2, buy the token that rose (momentum) at t2
  rev(t1,t2,x)  : ... buy the token that fell (reversal)
  conv_open(t,x): buy the token whose PM ask at t is >= x below the book OPEN fair prob (power de-vig)
Outputs: h3_accuracy.csv, h3_drift.csv, h3_autocorr.csv, h3_rules.csv
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm

from common_pre import DATA, RES, STAKE, ask_of_mid, bid_of_mid, fee, fmt, summarize

p = pd.read_parquet(DATA / 'ml_panel.parquet')
T = {'h24': 'pmid0_h24', 'h12': 'pmid0_h12', 'h6': 'pmid0_h6', 'h3': 'pmid0_h3', 'm300': 'mid0_m300',
     'm180': 'mid0_m180', 'm120': 'mid0_m120', 'm60': 'mid0_m60', 'm30': 'mid0_m30', 'm15': 'mid0_m15',
     'm5': 'mid0_m5', 'm1': 'mid0_m1'}
order = list(T)

# ---------------- accuracy by time ----------------
acc = []
for k, c in list(T.items()) + [('book_open', 'pbook0_open_pow'), ('book_close', 'pbook0_close_pow')]:
    for sp in ('train', 'test'):
        x = p[(p.split == sp)].dropna(subset=[c, 'mid0_m1', 'pmid0_h24'])     # common sample: games with all prices
        pr = x[c].clip(0.005, 0.995)
        acc.append(dict(time=k, split=sp, n=len(x), brier=np.mean((pr - x.y0) ** 2),
                        logloss=-np.mean(x.y0 * np.log(pr) + (1 - x.y0) * np.log(1 - pr)),
                        mad_vs_tip=np.mean(np.abs(x[c] - x.mid0_m1))))
pd.DataFrame(acc).pipe(fmt, 5).to_csv(RES / 'h3_accuracy.csv', index=False)


def leg(mid_t, mid_tip, y, exit_):
    """per-share P&L of buying a token at ask(mid_t): exit at bid(mid_tip) or hold to resolution."""
    a = ask_of_mid(mid_t)
    if exit_ == 'to_tip':
        b = bid_of_mid(mid_tip)
        g = b - a
        n = g - fee(a) - fee(b)
    else:
        g = y - a
        n = g - fee(a)
    return a, g, n


def trades(mask, tok1, t, exit_):
    d = p[mask].copy()
    tok1 = np.asarray(tok1)[np.asarray(mask)]
    m0 = d[T[t]].to_numpy()
    mid_t = np.where(tok1, 1 - m0, m0)
    mid_tip = np.where(tok1, 1 - d.mid0_m1.to_numpy(), d.mid0_m1.to_numpy())
    y = np.where(tok1, 1 - d.y0, d.y0)
    a, g, n = leg(mid_t, mid_tip, y, exit_)
    sh = STAKE / a
    out = pd.DataFrame({'game_id': d.game_id, 'split': d.split, 'price': a, 'y': y if exit_ == 'to_res' else
                        (mid_tip > mid_t).astype(int), 'pnl': sh * n, 'gross': sh * g, 'cost': STAKE + sh * fee(a)})
    return out


# ---------------- drift by time: favourite at t ----------------
dr = []
for t in order[:-1]:
    x = p.dropna(subset=[T[t], 'mid0_m1'])
    fav1 = x[T[t]] < 0.5
    pf_t = np.where(fav1, 1 - x[T[t]], x[T[t]])
    pf_tip = np.where(fav1, 1 - x.mid0_m1, x.mid0_m1)
    yf = np.where(fav1, 1 - x.y0, x.y0)
    for sp in ('train', 'test'):
        s = (x.split == sp).to_numpy()
        ch = pf_tip[s] - pf_t[s]
        dr.append(dict(time=t, split=sp, n=int(s.sum()), fav_price=pf_t[s].mean(), mean_change_to_tip=ch.mean(),
                       sd_change=ch.std(), z=ch.mean() / (ch.std() / np.sqrt(s.sum())),
                       abs_change=np.abs(ch).mean(), share_move_ge_2c=(np.abs(ch) >= 0.02).mean(),
                       fav_win=yf[s].mean(), res_at_t=(yf[s] - pf_t[s]).mean(), res_at_tip=(yf[s] - pf_tip[s]).mean()))
pd.DataFrame(dr).pipe(fmt, 4).to_csv(RES / 'h3_drift.csv', index=False)

# ---------------- autocorrelation of moves ----------------
ac = []
pairs = [('h24', 'h12'), ('h12', 'h6'), ('h24', 'h6'), ('h6', 'h3'), ('h6', 'm120'), ('m300', 'm120'),
         ('m300', 'm60'), ('m120', 'm60'), ('m60', 'm30'), ('m60', 'm15'), ('m30', 'm15'), ('m30', 'm5'),
         ('m15', 'm5')]
for t1, t2 in pairs:
    x = p.dropna(subset=[T[t1], T[t2], 'mid0_m1'])
    for sp in ('train', 'test'):
        z = x[x.split == sp]
        mv, fut = z[T[t2]] - z[T[t1]], z.mid0_m1 - z[T[t2]]
        res = z.y0 - z[T[t2]]
        m = sm.OLS(fut, sm.add_constant(mv)).fit()
        m2 = sm.OLS(res, sm.add_constant(mv)).fit()
        ac.append(dict(t1=t1, t2=t2, split=sp, n=len(z), sd_move=mv.std(), b_future_move=m.params.iloc[1],
                       t_future=m.tvalues.iloc[1], b_outcome_resid=m2.params.iloc[1], t_outcome=m2.tvalues.iloc[1]))
pd.DataFrame(ac).pipe(fmt, 4).to_csv(RES / 'h3_autocorr.csv', index=False)

# ---------------- rules ----------------
rows = []
for t in order[:-1]:
    x = p[T[t]]
    ok = x.notna() & p.mid0_m1.notna()
    for exit_ in ('to_tip', 'to_res'):
        for name, tok1 in (('fav', x < 0.5), ('dog', x >= 0.5)):
            rows.append(summarize(trades(ok, tok1, t, exit_), by='split', B=500)
                        .assign(rule=f'drift_{name}', t1='', t2=t, x=np.nan, exit=exit_))
for t1, t2 in pairs:
    mv = p[T[t2]] - p[T[t1]]
    ok0 = mv.notna() & p.mid0_m1.notna()
    for x_ in (0.01, 0.02, 0.03, 0.05):
        ok = ok0 & (mv.abs() >= x_)
        for kind in ('mom', 'rev'):
            tok1 = (mv < 0) if kind == 'mom' else (mv > 0)
            for exit_ in ('to_tip', 'to_res'):
                rows.append(summarize(trades(ok, tok1, t2, exit_), by='split', B=500)
                            .assign(rule=kind, t1=t1, t2=t2, x=x_, exit=exit_))
for t in ('h24', 'h12', 'h6', 'h3', 'm300', 'm120', 'm60'):
    m0 = p[T[t]]
    fair0 = p.pbook0_open_pow
    e0, e1 = fair0 - ask_of_mid(m0), (1 - fair0) - ask_of_mid(1 - m0)
    tok1 = e1 > e0
    edge = np.where(tok1, e1, e0)
    for x_ in (0.0, 0.01, 0.02, 0.03, 0.05):
        ok = m0.notna() & fair0.notna() & p.mid0_m1.notna() & (edge >= x_)
        for exit_ in ('to_tip', 'to_res'):
            rows.append(summarize(trades(ok, tok1, t, exit_), by='split', B=500)
                        .assign(rule='conv_open', t1='', t2=t, x=x_, exit=exit_))
R = pd.concat(rows)
R = R[['rule', 't1', 't2', 'x', 'exit', 'split', 'n', 'avg_price', 'win', 'pnl', 'pnl_per_trade', 'roi', 'roi_lo',
       'roi_hi', 'gross_pnl', 'gross_roi']]
R.pipe(fmt, 4).to_csv(RES / 'h3_rules.csv', index=False)
print('variants:', R.groupby(['rule', 't1', 't2', 'x', 'exit'], dropna=False).ngroups)
