"""H5 - team- and situation-specific biases of the pre-game moneyline (price at the scheduled start, s0).

Long format: two rows per game (each team's token). Residual = outcome - mid price.
1. OLS of the residual on situation dummies (own / opponent back-to-back, rest advantage, home, playoffs,
   popular team, popular opponent), SE clustered by game; train and test separately.
2. Team persistence: per-team mean residual in train vs test (and 2024-25 vs 2025-26), Spearman correlation.
3. Rules (buy at ask, hold to resolution, fee 0.05 p (1-p)): opponent on a back-to-back while the team is rested,
   fade popular teams, rested home team, playoffs favourites/underdogs, and 'best train teams' / 'worst train teams'.
Outputs: h5_regression.csv, h5_team_persistence.csv, h5_rules.csv
"""
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import spearmanr

from common_pre import DATA, RES, ask_of_mid, fmt, summarize, trade_pnl

p = pd.read_parquet(DATA / 'ml_panel.parquet').dropna(subset=['mid0_s0'])
POP = {'LAL', 'GS', 'BOS', 'NY'}
rows = []
for side in ('home', 'away'):
    opp = 'away' if side == 'home' else 'home'
    t = pd.DataFrame({'game_id': p.game_id, 'split': p.split, 'season': p.season, 'month': p.month,
                      'team': p[f'espn_{side}_abbr'], 'opp': p[f'espn_{opp}_abbr'], 'is_home': side == 'home',
                      'rest': p[f'{side}_rest'], 'opp_rest': p[f'{opp}_rest'],
                      'playoffs': p.game_type.isin(['playoffs', 'play_in'])})
    tok0 = (p.home_idx == 0) == (side == 'home')
    t['mid'] = np.where(tok0, p.mid0_s0, 1 - p.mid0_s0)
    t['y'] = np.where(tok0, p.y0, 1 - p.y0)
    rows.append(t)
t = pd.concat(rows, ignore_index=True)
t['res'] = t.y - t.mid
t['b2b'] = t.rest.eq(1).astype(int)
t['opp_b2b'] = t.opp_rest.eq(1).astype(int)
t['rest_adv'] = (t.rest.clip(upper=4) - t.opp_rest.clip(upper=4)).fillna(0)
t['popular'] = t.team.isin(POP).astype(int)
t['opp_popular'] = t.opp.isin(POP).astype(int)
t['fav'] = (t.mid > 0.5).astype(int)
t['home'] = t.is_home.astype(int)
t['po'] = t.playoffs.astype(int)

reg = []
hm = t[t.home == 1].copy()          # one row per game (home team's token); residuals of the two tokens sum to 0
hm['rest_diff'] = hm.rest_adv
for sp in ('train', 'test', 'all'):
    x = hm if sp == 'all' else hm[hm.split == sp]
    m = smf.ols('res ~ b2b + opp_b2b + rest_diff + po + popular + opp_popular + fav', data=x).fit(cov_type='HC1')
    for k in m.params.index:
        reg.append(dict(split=sp, n=len(x), term=k, coef=m.params[k], se=m.bse[k], t=m.tvalues[k]))
pd.DataFrame(reg).pipe(fmt, 4).to_csv(RES / 'h5_regression.csv', index=False)

tm = t.groupby(['team', 'split']).res.agg(['mean', 'size']).unstack()
ts = t[t.season.isin(['2024-25', '2025-26'])].groupby(['team', 'season']).res.mean().unstack()
pers = dict(spearman_train_test=spearmanr(tm[('mean', 'train')], tm[('mean', 'test')], nan_policy='omit')[0],
            spearman_2425_2526=spearmanr(ts['2024-25'], ts['2025-26'], nan_policy='omit')[0])
tm.columns = [f'{a}_{b}' for a, b in tm.columns]
tm = tm.join(ts.add_prefix('res_'))
tm.reset_index().pipe(fmt, 4).to_csv(RES / 'h5_team_persistence.csv', index=False)
print(pers)

t['price'] = ask_of_mid(t.mid)
t['pnl'], t['cost'], t['gross'] = trade_pnl(t.price, t.y)
best = tm['mean_train'].sort_values()
rules = {'opp_b2b_rested': (t.opp_b2b == 1) & (t.b2b == 0), 'own_b2b_opp_rested': (t.b2b == 1) & (t.opp_b2b == 0),
         'rest_adv>=2': t.rest_adv >= 2, 'fade_popular(buy opp)': t.opp_popular == 1, 'buy_popular': t.popular == 1,
         'playoff_fav': (t.po == 1) & (t.fav == 1), 'playoff_dog': (t.po == 1) & (t.fav == 0),
         'home_rested_vs_b2b': (t.home == 1) & (t.opp_b2b == 1) & (t.b2b == 0),
         'top5_train_teams': t.team.isin(best.index[-5:]), 'bottom5_train_teams': t.team.isin(best.index[:5])}
out = []
for k, msk in rules.items():
    out.append(summarize(t[msk], by='split', B=1000).assign(rule=k))
R = pd.concat(out)[['rule', 'split', 'n', 'games', 'avg_price', 'win', 'pnl', 'roi', 'roi_lo', 'roi_hi', 'gross_roi']]
R.pipe(fmt, 4).to_csv(RES / 'h5_rules.csv', index=False)
pd.Series(pers).to_csv(RES / 'h5_persistence_summary.csv')
print(R.pipe(fmt, 3).to_string(index=False))
print(pd.DataFrame(reg).query("term!='Intercept'").pipe(fmt, 3).to_string(index=False))
