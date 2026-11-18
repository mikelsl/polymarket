"""Build the spread / total line panel (2025-26 and 2026-27 core games) -> data/nba_pre/lines_panel.parquet.

One row per traded full-game spread or total line:
  kind, market_id, L (spreads: handicap of the HOME team, home covers iff home margin + L > 0; totals: line),
  mid_cover_* / mid_over_* : CLOB token mid (trailing 5-point median) of 'home covers' / 'over' at
      m1, m5, m60, m120 = tip - k minutes, s0 = scheduled start
  y_cover / y_over : resolution of that token (Gamma; NaN if 50-50 / unresolved)
  game-level: book spread / total open & close (home perspective), ESPN final margin and total points,
  vol_final (look-ahead, only for reference), rank_created_in_kind, created_at
"""
import numpy as np
import pandas as pd

from common_pre import DATA, SRC, core_games

g = core_games()
g = g[g.season.isin(['2025-26', '2026-27'])].copy()
g['home_nick'] = np.where(g.home_idx == 1, g.ml_outcome_1, g.ml_outcome_0)
g['margin'] = g.espn_home_score - g.espn_away_score
g['tot_pts'] = g.espn_home_score + g.espn_away_score
pan = pd.read_parquet(DATA / 'ml_panel.parquet')
book = ['spread_home_open', 'spread_home_close', 'total_open', 'total_close', 'spread_home_close_odds',
        'spread_away_close_odds', 'over_close_odds', 'under_close_odds', 'mid0_s0', 'mid0_m1', 'mid0_m60',
        'pbook0_close_pow', 'pbook0_open_pow']
g = g.merge(pan[['game_id'] + book], on='game_id', how='left')

m = pd.read_parquet(SRC / 'markets.parquet')
m = m[m.kind.isin(['spreads', 'totals']) & m.event_role.isin(['main', 'v2']) & (m.volume_effective.fillna(0) > 0)]
m = m.merge(g[['game_id', 'season', 'split', 'month', 'sched', 'tip', 'home_nick', 'home_idx', 'margin', 'tot_pts',
               'y0'] + book], on='game_id')
sp = m.kind == 'spreads'
m['home_is_team0'] = np.where(sp, m.team0 == m.home_nick, np.nan)
m['L'] = np.where(sp, np.where(m.team0 == m.home_nick, m.handicap0, -m.handicap0), m.total_line)
wi = m.winner_index.astype('float')
y0 = np.where(wi.isin([0, 1]), (wi == 0).astype(float), np.nan)
# token "home covers" (spreads) / "over" (totals): token 0 if team0 is home / outcome_0 is Over
flip = np.where(sp, m.team0 != m.home_nick, m.outcome_0.str.lower() != 'over')
m['flip'] = flip.astype(bool)
m['y_side'] = np.where(m.flip, 1 - y0, y0)

lp = pd.read_parquet(DATA / 'line_prices.parquet')
lp = lp[lp.market_id.isin(m.market_id)].sort_values(['market_id', 't'])
# activity in the 2 h before tip: number of minute-to-minute price changes (0 = stale / default quote)
_w = lp.merge(m[['market_id', 'tip']], on='market_id')
_tt = _w.tip.astype('int64') // 10**6
_w = _w[(_w.t <= _tt) & (_w.t > _tt - 7200)]
nchg = _w.groupby('market_id').p0.apply(lambda s: int((s.diff().abs() > 1e-9).sum()))
m['n_chg_2h'] = m.market_id.map(nchg).fillna(0).astype(int)
lp['p0'] = lp.groupby('market_id', sort=False).p0.transform(lambda s: s.rolling(5, min_periods=1).median())
lp = lp.sort_values('t')
tip_ts = m.tip.astype('int64') // 10**6
sched_ts = m.sched.astype('int64') // 10**6
q = []
for lab, tq in (('m1', tip_ts - 60), ('m5', tip_ts - 300), ('m60', tip_ts - 3600), ('m120', tip_ts - 7200),
                ('s0', sched_ts)):
    q.append(pd.DataFrame({'market_id': m.market_id.values, 'label': lab, 'tq': tq.values}))
q = pd.concat(q).sort_values('tq')
mm = pd.merge_asof(q, lp, left_on='tq', right_on='t', by='market_id', direction='backward', tolerance=900)
w = mm.pivot(index='market_id', columns='label', values='p0')
m = m.merge(w.add_prefix('p0_'), left_on='market_id', right_index=True, how='left')
for lab in ('m1', 'm5', 'm60', 'm120', 's0'):
    m[f'mid_side_{lab}'] = np.where(m.flip, 1 - m[f'p0_{lab}'], m[f'p0_{lab}'])
# ---------------- prints: pre-tip activity and print-revealed quotes of the side token ----------------
trs = []
for name in ('spread_trades', 'total_trades'):
    if (DATA / f'{name}.parquet').exists():
        trs.append(pd.read_parquet(DATA / f'{name}.parquet'))
tr = pd.concat(trs, ignore_index=True)
tr = tr.merge(m[['market_id', 'flip']].assign(tip_ts=tip_ts.values, side_tok=m.flip.astype(int).values),
              on='market_id')
tr = tr[tr.ts < tr.tip_ts]
own = tr.token_index == tr.side_tok
tr['is_ask'] = (own & (tr.side == 'BUY')) | (~own & (tr.side == 'SELL'))
tr['px'] = np.where(own, tr.price, 1 - tr.price)
pre2 = tr[tr.ts >= tr.tip_ts - 7200].groupby('market_id').agg(n_prints_2h=('ts', 'size'))
m['n_prints_2h'] = m.market_id.map(pre2.n_prints_2h).fillna(0).astype(int)
m['vol_pre_tip'] = m.market_id.map(tr.groupby('market_id')['size'].sum()).fillna(0.0)
m['prints_pulled'] = m.market_id.isin(set(pd.concat(trs).market_id))
tr = tr.sort_values('ts')
for lab in ('m1', 'm60', 's0'):
    qq = q[q.label == lab][['market_id', 'tq']].sort_values('tq')
    for nm, sub in (('pask', tr[tr.is_ask]), ('pbid', tr[~tr.is_ask])):
        x = pd.merge_asof(qq, sub[['market_id', 'ts', 'px']], left_on='tq', right_on='ts', by='market_id',
                          direction='backward', tolerance=900)
        m[f'{nm}_side_{lab}'] = m.market_id.map(x.set_index('market_id').px)

keep = ['game_id', 'market_id', 'kind', 'season', 'split', 'month', 'sched', 'tip', 'L', 'flip', 'y_side', 'team0',
        'handicap0', 'total_line', 'home_is_team0', 'margin', 'tot_pts', 'y0', 'volume_effective',
        'rank_created_in_kind', 'rank_volume_in_kind', 'role', 'n_chg_2h', 'created_at', 'question', 'token_0', 'token_1'] + \
       book + [f'mid_side_{x}' for x in ('m1', 'm5', 'm60', 'm120', 's0')] + \
       ['prints_pulled', 'n_prints_2h', 'vol_pre_tip'] + [f'{a}_side_{b}' for a in ('pask', 'pbid') for b in ('m1', 'm60', 's0')]
m = m[keep].rename(columns={'y0': 'ml_y0'})
m.to_parquet(DATA / 'lines_panel.parquet', index=False)
print(m.groupby(['season', 'kind']).agg(lines=('market_id', 'size'), games=('game_id', 'nunique'),
                                         with_m60=('mid_side_m60', lambda s: s.notna().sum()),
                                         with_s0=('mid_side_s0', lambda s: s.notna().sum())).to_string())
