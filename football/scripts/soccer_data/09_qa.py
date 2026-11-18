"""Step 9: QA metrics for REPORT.md -> results/soccer_data/qa_*.csv and qa_summary.json.

- coverage by league x season x month (ESPN fixtures, Polymarket events, link, O/U 2.5, BTTS, prices, trades, odds)
- trades completeness: sum of taker print sizes vs Gamma `volume` per 1X2 market
- price history gaps, ESPN kickoff wallclock vs schedule, key-event goals vs final score
- timestamp precision: ESPN goal wallclock vs first sustained jump in the scoring team's 1X2 taker prints
Usage: .venv/bin/python -I scripts/soccer_data/09_qa.py
"""
import sys
sys.path.insert(0, 'polymarket/research/auto-20261104/scripts/soccer_data')
import json
import os
import subprocess

import numpy as np
import pandas as pd

from common import DATA, RES, log


def du(path):
    try:
        return subprocess.run(['du', '-sk', path], capture_output=True, text=True).stdout.split()[0]
    except Exception:
        return None


def _series(tr, ids):
    tr = tr[tr.match_id.isin(ids)].sort_values('ts', kind='stable')
    return {k: (v.ts.to_numpy(), v.yes_price.to_numpy()) for k, v in tr.groupby(['match_id', 'role'], observed=True)}


def _goals(m, ev):
    g = ev[ev.scoring_play & ev.wallclock_ts.notna() & ev.team_side_pm.isin(['home', 'away'])]
    return g[g.match_id.isin(m.match_id)]


def goal_lag(m, ev, tr, half=5, win=300, min_step=0.05):
    """Per goal (wallclock W): among the scoring team's win-market prints in [W-win, W+win] (YES scale), find the split
    with the largest local step = median(next `half` prints) - median(previous `half` prints). If the step is >= min_step,
    lag = time of the first post-split print - W. Local medians make this robust to slow pre-goal drift."""
    g = _goals(m, ev)
    byk = _series(tr, g.match_id.unique())
    out = []
    for r in g.itertuples():
        k = byk.get((r.match_id, r.team_side_pm))
        W = int(r.wallclock_ts)
        base = dict(match_id=r.match_id, seq=r.seq, W=W, minute=r.minute, added=r.added_minute)
        if k is None:
            out.append(dict(base, status='no_prints'))
            continue
        ts, yp = k
        sel = np.where((ts >= W - win) & (ts <= W + win))[0]
        if len(sel) < 2 * half:
            out.append(dict(base, status='few_prints'))
            continue
        x, t = yp[sel], ts[sel]
        best, bk = -1.0, None
        for j in range(half, len(x) - half + 1):
            s = float(np.median(x[j:j + half]) - np.median(x[j - half:j]))
            if s > best:
                best, bk = s, j
        pre = float(np.median(x[bk - half:bk]))
        if best < min_step:
            out.append(dict(base, pre=pre, step=best, status='no_step_ge_0.05'))
            continue
        out.append(dict(base, pre=pre, step=best, lag_s=int(t[bk] - W), status='ok'))
    return pd.DataFrame(out)


def clock_alignment(m, ev, tr):
    """Whole-match clock-shift test (e.g. DST errors): for shifts s in [-5400, 5400] s (30 s grid), score(s) = sum over the
    match's goals of the scoring team's price rise p(W+s+90) - p(W+s-90) (last-print price). The best shift should be ~0;
    |best| > 300 s flags a shifted ESPN wallclock for that match."""
    g = _goals(m, ev)
    byk = _series(tr, g.match_id.unique())
    shifts = np.arange(-5400, 5401, 30)
    rows = []
    for mid, gg in g.groupby('match_id'):
        score, n = np.zeros(len(shifts)), 0
        for r in gg.itertuples():
            k = byk.get((mid, r.team_side_pm))
            if k is None or len(k[0]) < 20:
                continue
            ts, yp = k
            def at(tt):
                i = np.searchsorted(ts, tt, side='right') - 1
                return np.where(i >= 0, yp[np.clip(i, 0, None)], np.nan)
            t = int(r.wallclock_ts) + shifts
            score += np.nan_to_num(at(t + 90) - at(t - 90))
            n += 1
        if n:
            j = int(np.argmax(score))
            rows.append(dict(match_id=mid, n_goals=n, best_shift_s=int(shifts[j]), score_best=float(score[j]),
                             score_at_0=float(score[shifts == 0][0])))
    return pd.DataFrame(rows)


def outage_windows(m, tr, bucket=300, min_live=2, frac=0.05, min_expected=40):
    """Platform-wide in-play trading gaps. Each match's normal rate = its in-play 1X2 prints (kickoff..kickoff+115 min)
    per 5-minute bucket. A bucket is flagged when >= min_live matches are in play, the sum of their normal rates is
    >= min_expected, and the prints actually seen are < frac of that expectation. Consecutive buckets are merged."""
    k = dict(zip(m.match_id, m.kickoff_ts.astype('int64')))
    t = tr[tr.match_id.isin(k)][['match_id', 'ts']].copy()
    t['k'] = t.match_id.map(k)
    t = t[(t.ts >= t.k) & (t.ts < t.k + 6900)]
    t['b'] = t.ts // bucket * bucket
    rate = (t.groupby('match_id').size() / (6900 / bucket)).to_dict()
    typical = float(np.median(list(rate.values())))
    cnt = t.groupby('b').size()
    lo, hi = int(m.kickoff_ts.min()) // bucket * bucket, int(m.kickoff_ts.max()) // bucket * bucket + 7200
    km = sorted(k.items(), key=lambda kv: kv[1])
    ks = np.array([v for _, v in km])
    ids = [i for i, _ in km]
    rows = []
    for b in range(lo, hi, bucket):
        a, z = np.searchsorted(ks, b - 6900 + 1, side='left'), np.searchsorted(ks, b + bucket, side='left')
        live = ids[a:z]
        if len(live) < min_live:
            continue
        exp = sum(rate.get(i, 0.0) for i in live)
        p = int(cnt.get(b, 0))
        if exp >= min_expected and p < frac * exp:
            rows.append((b, len(live), p))
    win = []
    for b, n, p in rows:
        if win and b == win[-1]['end_ts']:
            w = win[-1]
            w.update(end_ts=b + bucket, max_live=max(w['max_live'], n), prints=w['prints'] + p, buckets=w['buckets'] + 1)
        else:
            win.append(dict(start_ts=b, end_ts=b + bucket, max_live=n, prints=p, buckets=1))
    w = pd.DataFrame(win)
    if len(w):
        w['start_utc'] = pd.to_datetime(w.start_ts, unit='s', utc=True)
        w['end_utc'] = pd.to_datetime(w.end_ts, unit='s', utc=True)
        w['minutes'] = (w.end_ts - w.start_ts) // 60
    return w, typical


def main():
    S = {}
    m_all = pd.read_parquet(f'{DATA}/matches.parquet')
    m = m_all
    mk = pd.read_parquet(f'{DATA}/markets.parquet')
    fx = pd.read_parquet(f'{DATA}/espn_fixtures.parquet')
    ev = pd.read_parquet(f'{DATA}/events.parquet')
    ph_q = pd.read_parquet(f'{DATA}/qa_price_tokens.parquet')
    od = pd.read_parquet(f'{DATA}/odds.parquet')

    # coverage by month
    fx = fx[fx.season.notna()].copy()
    fx['month'] = pd.to_datetime(fx.espn_date_ts, unit='s', utc=True).dt.strftime('%Y-%m')
    one = m[m.is_primary_event]
    tok_ok = ph_q.groupby('match_id').apply(lambda d: int((d.n_points > 0).sum()), include_groups=False)
    x12 = mk[mk.role.isin(['home', 'draw', 'away'])]
    tr_ok = x12.groupby('match_id').fetched.apply(lambda s: bool(s.fillna(False).all()) and len(s) == 3)
    one = one.assign(month=one.season_month, prices_3=one.match_id.map(tok_ok).fillna(0) >= 3,
                     trades_3=one.match_id.map(tr_ok).fillna(False), summary=one.espn_status.notna())
    a = fx.groupby(['league', 'season', 'month']).size().rename('espn_fixtures')
    b = one.groupby(['league', 'season', 'month']).agg(pm_events=('match_id', 'size'), linked=('espn_event_id', lambda s: int(s.notna().sum())),
                                                     ou25=('has_ou25', 'sum'), btts=('has_btts', 'sum'), prices_1x2=('prices_3', 'sum'),
                                                     trades_1x2=('trades_3', 'sum'), odds=('has_odds', 'sum'), espn_summary=('summary', 'sum'),
                                                     resolved_match=('resolution_check', lambda s: int((s == 'match').sum())))
    cov = pd.concat([a, b], axis=1).fillna(0).astype(int).reset_index()
    cov.to_csv(f'{RES}/qa_coverage_by_month.csv', index=False)
    cs = cov.groupby(['season', 'league']).sum(numeric_only=True).reset_index()
    cs.to_csv(f'{RES}/qa_coverage_by_league.csv', index=False)
    S['coverage_by_league'] = cs.to_dict('records')

    # trades completeness
    t = x12[x12.match_id.isin(m.match_id)].copy()
    S['markets_1x2'] = len(t)
    S['markets_trades_fetched'] = int(t.fetched.fillna(False).sum())
    t['ratio'] = t.taker_size_sum / t.volume
    S['trade_volume_ratio_quantiles'] = t.ratio.describe(percentiles=[.01, .05, .5, .95, .99]).round(4).to_dict()
    S['markets_volume_missing'] = int(t.volume.isna().sum())
    S['markets_zero_prints'] = int((t.n_taker_prints == 0).sum())
    bad = t[t.fetched.fillna(False) & ((t.ratio < 0.98) | (t.ratio > 1.02) | t.volume.isna())]
    bad[['match_id', 'role', 'condition_id', 'volume', 'taker_size_sum', 'ratio', 'n_taker_prints']].to_csv(f'{RES}/qa_trades_volume_outliers.csv', index=False)
    S['markets_ratio_outside_2pct'] = len(bad)

    # price history
    S['price_tokens'] = len(ph_q)
    S['price_tokens_zero_points'] = int((ph_q.n_points == 0).sum())
    S['price_points_per_token_quantiles'] = ph_q.n_points.describe(percentiles=[.01, .05, .5]).round(1).to_dict()
    S['price_max_gap_s_quantiles'] = ph_q.max_gap_s.describe(percentiles=[.5, .9, .99]).round(0).to_dict()
    S['price_tokens_by_basis'] = ph_q.window_basis.value_counts().to_dict()
    ph_q[ph_q.n_points < 60].to_csv(f'{RES}/qa_price_tokens_sparse.csv', index=False)

    # ESPN timing
    k = m.kickoff_wall_minus_sched_s.dropna().astype(float)
    S['kickoff_wall_minus_sched_s_quantiles'] = k.describe(percentiles=[.01, .05, .5, .95, .99]).round(0).to_dict()
    odd = m[m.kickoff_wall_minus_sched_s.abs() > 900]
    odd[['match_id', 'kickoff_utc', 'kickoff_wall_utc', 'kickoff_wall_minus_sched_s', 'pm_minus_espn_kickoff_s']].to_csv(f'{RES}/qa_kickoff_wallclock_outliers.csv', index=False)
    S['kickoff_wall_outliers_gt_15min'] = len(odd)
    S['ke_goals_inconsistent'] = m[m.espn_status.notna() & ~m.ke_goals_consistent.fillna(False)].match_id.tolist()
    S['espn_wallclock_available_false'] = int((m.wallclock_available == False).sum())  # noqa: E712
    S['event_types'] = ev.type.value_counts().to_dict()
    S['goal_events'] = int(ev.scoring_play.sum())
    S['goal_wallclock_missing'] = int((ev.scoring_play & ev.wallclock_ts.isna()).sum())
    S['red_cards'] = int(ev.is_red_card.sum())
    sec = ev.wallclock_ts.dropna().astype('int64') % 60
    S['wallclock_share_second_00'] = round(float((sec == 0).mean()), 4)
    # clock-derived vs wallclock consistency (goals in regular time, not added time)
    gg = ev[ev.scoring_play & ev.wallclock_ts.notna() & (ev.added_minute.fillna(0) == 0)].merge(
        m[['match_id', 'kickoff_wall_ts', 'second_half_wall_ts']], on='match_id')
    est = np.where(gg.period == 1, gg.kickoff_wall_ts + gg.clock_value_s, gg.second_half_wall_ts + gg.clock_value_s - 2700)
    d = (gg.wallclock_ts - est).astype(float)
    S['goal_wallclock_minus_clock_estimate_s_quantiles'] = d.describe(percentiles=[.05, .25, .5, .75, .95]).round(0).to_dict()

    # resolution
    S['resolution_check'] = m.resolution_check.value_counts().to_dict()
    S['resolution_non_match'] = m[m.resolution_check != 'match'][['match_id', 'resolution_check', 'pm_result', 'espn_result']].to_dict('records')
    fd = od[od.match_id.notna()].merge(m[['match_id', 'home_score', 'away_score', 'espn_result']], on='match_id')
    fd_bad = fd[(fd.FTHG != fd.home_score) | (fd.FTAG != fd.away_score)]
    S['fd_vs_espn_score_mismatch'] = fd_bad[['match_id', 'FTHG', 'FTAG', 'home_score', 'away_score']].to_dict('records')
    fdk = fd.fd_kickoff_ts.astype('float') - fd.match_id.map(dict(zip(m.match_id, m.kickoff_ts))).astype('float')
    S['fd_kickoff_minus_espn_s'] = fdk.value_counts().head(5).to_dict()

    # fees by month
    fee = x12.merge(m[['match_id', 'season_month']], on='match_id').groupby(['season_month', 'fee_rate'], dropna=False).size()
    S['fee_rate_by_month'] = {f'{a}|{b}': int(v) for (a, b), v in fee.items()}

    # goal lag vs prints
    if os.path.exists(f'{DATA}/trades.parquet'):
        tr = pd.read_parquet(f'{DATA}/trades.parquet', columns=['match_id', 'role', 'ts', 'yes_price'])
        gl = goal_lag(m[m.is_primary_event], ev, tr)
        gl.to_csv(f'{RES}/qa_goal_lag.csv', index=False)
        ok = gl[gl.status == 'ok']
        S['goal_lag_status'] = gl.status.value_counts().to_dict()
        S['goal_lag_s_quantiles'] = ok.lag_s.describe(percentiles=[.05, .1, .25, .5, .75, .9, .95]).round(0).to_dict()
        S['goal_lag_share_before_wallclock'] = round(float((ok.lag_s < 0).mean()), 3)
        S['goal_lag_share_abs_le_30s'] = round(float((ok.lag_s.abs() <= 30).mean()), 3)
        S['goal_lag_share_abs_le_60s'] = round(float((ok.lag_s.abs() <= 60).mean()), 3)
        S['goal_lag_n_matches'] = int(ok.match_id.nunique())
        ow, typical = outage_windows(m[m.is_primary_event], tr)
        ow.to_csv(f'{RES}/qa_trading_gaps.csv', index=False)
        S['trading_gap_typical_prints_per_live_match_5min'] = typical
        S['trading_gap_windows'] = ow[['start_utc', 'end_utc', 'minutes', 'max_live', 'prints']].astype(str).to_dict('records') if len(ow) else []
        ca = clock_alignment(m[m.is_primary_event], ev, tr)
        ca.to_csv(f'{RES}/qa_clock_shift_by_match.csv', index=False)
        S['clock_alignment_matches'] = len(ca)
        S['clock_alignment_best_shift_quantiles'] = ca.best_shift_s.describe(percentiles=[.01, .05, .5, .95, .99]).round(0).to_dict()
        S['clock_alignment_share_abs_le_120s'] = round(float((ca.best_shift_s.abs() <= 120).mean()), 3)
        # single-match and platform-wide in-play halts from matches.inplay_max_gap_s (08b_inplay_stats.py)
        if 'inplay_max_gap_s' in m.columns:
            mm_ = m[m.inplay_max_gap_s.notna()].copy()
            mm_['rate_per_min'] = mm_.inplay_prints / ((mm_.fulltime_wall_ts - mm_.kickoff_wall_ts) / 60)
            h = mm_[(mm_.inplay_max_gap_s >= 900) & (mm_.rate_per_min >= 2)].copy()
            h['gap_start_utc'] = pd.to_datetime(h.inplay_max_gap_start_ts, unit='s', utc=True)
            h['gap_end_utc'] = pd.to_datetime(h.inplay_max_gap_start_ts + h.inplay_max_gap_s, unit='s', utc=True)
            h = h.sort_values('gap_start_utc')[['match_id', 'gap_start_utc', 'gap_end_utc', 'inplay_max_gap_s', 'inplay_prints', 'rate_per_min']]
            h.to_csv(f'{RES}/qa_inplay_halts.csv', index=False)
            S['inplay_halts_n'] = len(h)
            S['inplay_halts_by_date'] = h.groupby(h.gap_start_utc.dt.strftime('%Y-%m-%d')).match_id.apply(list).to_dict()
            S['inplay_max_gap_s_quantiles'] = mm_.inplay_max_gap_s.astype(float).describe(percentiles=[.5, .9, .95, .99]).round(0).to_dict()
        strong = ca[(ca.best_shift_s.abs() > 300) & (ca.score_best - ca.score_at_0 >= 0.15) & (ca.score_at_0 <= 0.5 * ca.score_best)].copy()
        if 'inplay_max_gap_s' in m.columns:
            strong = strong.merge(m[['match_id', 'inplay_max_gap_s']], on='match_id', how='left')
        if len(ow):  # does the match's in-play window overlap a trading halt?
            kk = dict(zip(m.match_id, m.kickoff_ts.astype('int64')))
            strong['overlaps_trading_gap'] = [bool(((ow.start_ts < kk[i] + 6900) & (ow.end_ts > kk[i])).any()) for i in strong.match_id]
        strong.to_csv(f'{RES}/qa_clock_shift_strong.csv', index=False)
        S['clock_shift_gt_300s_any'] = int((ca.best_shift_s.abs() > 300).sum())
        S['clock_shift_strong'] = strong.to_dict('records')

    # cross-check with the existing 404-match dataset (read-only): polymarket/data/soccer/matches/*.json
    import glob
    old = glob.glob('polymarket/data/soccer/matches/*.json')
    mi = m.set_index('match_id')
    evg = ev[ev.scoring_play].groupby('match_id').wallclock_ts.apply(lambda s: sorted(int(x) for x in s.dropna()))
    won = mk[mk.role.isin(['home', 'draw', 'away'])].set_index('condition_id').resolved_outcome0.to_dict()
    cmp = dict(files=len(old), in_new=0, espn_id_same=0, score_same=0, goal_walls_same=0, kickoff_same=0, won_same=0, won_compared=0)
    diffs = []
    for f in old:
        o = json.load(open(f))
        if o['slug'] not in mi.index:
            diffs.append((o['slug'], 'not in new dataset'))
            continue
        cmp['in_new'] += 1
        r = mi.loc[o['slug']]
        e = o.get('espn') or {}
        if str(e.get('id')) == str(r.espn_event_id):
            cmp['espn_id_same'] += 1
        else:
            diffs.append((o['slug'], f"espn id old {e.get('id')} new {r.espn_event_id}"))
        try:
            if int(e.get('homeScore')) == int(r.espn_home_score) and int(e.get('awayScore')) == int(r.espn_away_score):
                cmp['score_same'] += 1
            else:
                diffs.append((o['slug'], f"score old {e.get('homeScore')}-{e.get('awayScore')} new {r.espn_home_score}-{r.espn_away_score}"))
        except Exception:
            diffs.append((o['slug'], 'score not comparable'))
        ow = sorted(int(pd.Timestamp(k['wall']).timestamp()) for k in e.get('keyEvents') or [] if k.get('scoring') and k.get('wall'))
        if ow == evg.get(o['slug'], []):
            cmp['goal_walls_same'] += 1
        else:
            diffs.append((o['slug'], f'goal wallclocks old {ow} new {evg.get(o["slug"], [])}'))
        if int(o['kick']) == int(r.pm_game_start_ts):
            cmp['kickoff_same'] += 1
        for x in o.get('markets') or []:
            if x.get('conditionId') in won and won[x['conditionId']] is not None:
                cmp['won_compared'] += 1
                cmp['won_same'] += int(bool(x.get('won')) == bool(won[x['conditionId']]))
    S['crosscheck_existing_404'] = cmp
    S['crosscheck_existing_404_diffs'] = diffs[:50]

    # sizes
    S['disk_kb'] = {p: du(f'{DATA}/{p}') for p in ['matches.parquet', 'markets.parquet', 'price_history.parquet', 'trades.parquet',
                                                   'events.parquet', 'odds.parquet', 'trades_parts', 'raw', '.']}
    S['rows'] = {}
    for p in ['matches', 'markets', 'price_history', 'trades', 'events', 'odds']:
        f = f'{DATA}/{p}.parquet'
        if os.path.exists(f):
            import pyarrow.parquet as pq
            S['rows'][p] = pq.ParquetFile(f).metadata.num_rows
    with open(f'{RES}/qa_summary.json', 'w') as f:
        json.dump(S, f, indent=1, default=str)
    log(json.dumps({k: S[k] for k in ['rows', 'resolution_check', 'markets_ratio_outside_2pct', 'price_tokens_zero_points']}, default=str))


if __name__ == '__main__':
    main()
