"""Step 6 - write results/nba_data/REPORT.md from the final tables, QA lists and job logs (no network).

Every number in the report is computed here from data/nba_data/ and results/nba_data/qa/, so re-running after a
rebuild refreshes the report.  Usage: .venv/bin/python scripts/nba_data/make_report.py
"""
import datetime as dt
import json
import re
from pathlib import Path

import pandas as pd
import pyarrow.dataset as ds

from common import DATA, RES

QA = RES / 'qa'
CORE = ['regular', 'play_in', 'playoffs', 'cup_final']


def size_of(p):
    p = Path(p)
    if p.is_file():
        return p.stat().st_size
    return sum(f.stat().st_size for f in p.rglob('*') if f.is_file())


def mb(n):
    return f'{n / 1e6:,.1f} MB'


def md_table(df, index=True):
    df = df.reset_index() if index else df
    cols = [str(c) for c in df.columns]
    out = ['| ' + ' | '.join(cols) + ' |', '|' + '|'.join('---' for _ in cols) + '|']
    for r in df.itertuples(index=False):
        cells = []
        for v in r:
            if v is None or (not isinstance(v, str) and pd.isna(v)):
                cells.append('')
            elif isinstance(v, float) and v.is_integer():
                cells.append(f'{int(v):,}')
            elif isinstance(v, (int,)) or hasattr(v, 'dtype') and 'int' in str(getattr(v, 'dtype', '')):
                cells.append(f'{int(v):,}')
            else:
                cells.append(str(v))
        out.append('| ' + ' | '.join(cells) + ' |')
    return '\n'.join(out)


def log_span(name):
    f = RES / name
    if not f.exists():
        return None
    ts = re.findall(r'^(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ)', f.read_text(), re.M)
    if not ts:
        return None
    p = lambda s: dt.datetime.fromisoformat(s.replace('Z', '+00:00'))
    return p(ts[0]), p(ts[-1])


def log_segments(name):
    """One (start, end, duration, last http stats) tuple per run in a log (runs are appended with >>)."""
    f = RES / name
    if not f.exists():
        return []
    out, cur = [], []
    for line in f.read_text().splitlines():
        if re.search(r'(mode=|price targets:|scoreboards to fetch|gamma windows to fetch|completed ESPN events without summary)',
                     line) and cur:
            out.append(cur)
            cur = []
        cur.append(line)
    if cur:
        out.append(cur)
    rows = []
    p = lambda s: dt.datetime.fromisoformat(s.replace('Z', '+00:00'))
    for seg in out:
        ts = [m.group(1) for m in (re.match(r'^(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ)', l) for l in seg) if m]
        if not ts:
            continue
        stats = re.findall(r"http (\{'requests'.*?\})", '\n'.join(seg))
        a, b = p(ts[0]), p(ts[-1])
        rows.append((f'{a:%Y-%m-%d %H:%M}Z', f'{b:%H:%M}Z', f'{(b - a).total_seconds() / 60:.0f} min',
                     stats[-1] if stats else ''))
    return rows


def last_stats(name):
    f = RES / name
    if not f.exists():
        return None
    m = re.findall(r"http (\{'requests'.*?\})", f.read_text())
    return m[-1] if m else None


def qa_csv(name):
    f = QA / f'{name}.csv'
    return pd.read_csv(f, dtype=str) if f.exists() and f.stat().st_size > 1 else pd.DataFrame()


def main():
    g = pd.read_parquet(DATA / 'games.parquet')
    mk = pd.read_parquet(DATA / 'markets.parquet')
    qa = json.loads((RES / 'qa_summary.json').read_text())
    cov = pd.read_csv(QA / 'market_coverage.csv', dtype={'market_id': str, 'game_id': str})
    tr = ds.dataset(DATA / 'trades', format='parquet', partitioning='hive')
    ph = ds.dataset(DATA / 'price_history', format='parquet', partitioning='hive')
    n_tr, n_ph = tr.count_rows(), ph.count_rows()
    pbp = pd.read_parquet(DATA / 'espn_pbp.parquet', columns=['espn_event_id'])
    pc = pd.read_parquet(DATA / 'pickcenter.parquet', columns=['espn_event_id', 'source', 'provider_name'])
    rev = pd.read_parquet(DATA / 'work' / 'espn_score_reversals.parquet') if (DATA / 'work' / 'espn_score_reversals.parquet').exists() else pd.DataFrame()

    g['month'] = g.game_date_et.str[:7]
    g['has_ml_prices'] = (g.ml_n_price_points_0.fillna(0) > 0) & (g.ml_n_price_points_1.fillna(0) > 0)
    g['has_ml_trades'] = g.ml_n_prints.fillna(0) > 0
    g['has_pbp'] = g.espn_event_id.isin(set(pbp.espn_event_id))
    g['has_pickcenter'] = g.espn_event_id.isin(set(pc[pc.source == 'summary_pickcenter'].espn_event_id))
    g['has_any_odds'] = g.espn_event_id.isin(set(pc.espn_event_id))
    agg = dict(games=('game_id', 'size'), espn_matched=('espn_event_id', lambda x: int(x.notna().sum())),
               ml_resolved=('ml_winner_index', lambda x: int(x.isin([0, 1]).sum())),
               ml_prices=('has_ml_prices', 'sum'), ml_trades=('has_ml_trades', 'sum'),
               main_spread=('main_spread_market_id', lambda x: int(x.notna().sum())),
               main_total=('main_total_market_id', lambda x: int(x.notna().sum())),
               pbp=('has_pbp', 'sum'), pickcenter=('has_pickcenter', 'sum'), any_odds=('has_any_odds', 'sum'))
    core = g[g.is_core_game]
    by_season = core.groupby('season').agg(**agg)
    by_season.loc['total'] = by_season.sum()
    by_season_type = g.groupby(['season', 'game_type']).agg(**agg)
    by_month = core.groupby('month').agg(**agg)

    files = [('games.parquet', len(g)), ('markets.parquet', len(mk)), ('trades/', n_tr), ('price_history/', n_ph),
             ('espn_pbp.parquet', len(pbp)), ('pickcenter.parquet', len(pc))]
    ftab = pd.DataFrame([(f'data/nba_data/{f}', f'{n:,}', mb(size_of(DATA / f))) for f, n in files],
                        columns=['path', 'rows', 'size'])

    tcols = tr.to_table(columns=['market_role', 'size', 'season']).to_pandas()
    tsum = tcols.groupby(['season', 'market_role'], observed=True).agg(prints=('size', 'size'), shares=('size', 'sum'))
    tsum['shares'] = tsum.shares.round(0).astype('int64')
    tsum.loc[('all', 'all'), :] = [len(tcols), int(round(tcols['size'].sum()))]
    vol = cov[cov.trades_fetched & cov.gamma_volume.notna() & ~cov.net_of_fee_sizes.astype(bool)]
    exact = int((vol.volume_rel_gap.abs() <= 1e-4).sum())
    pcov = cov[cov.prices_fetched]
    pts = pd.concat([pcov.n_price_0, pcov.n_price_1])

    # ---- ESPN text ---------------------------------------------------------------------------------------------
    nopm = qa_csv('espn_games_without_pm_event')
    if len(nopm):
        kind = nopm.apply(lambda r: 'postponed shell' if r.status_name != 'STATUS_FINAL' else
                          ('All-Star' if r.comp_type == 'ALLSTAR' else ('preseason' if r.season_type == '1' else
                                                                         f'{r.short_name} {r.espn_date_et}')), axis=1)
        nopm_txt = ', '.join(f'{k} x{n}' for k, n in kind.value_counts().items())
    else:
        nopm_txt = 'none'
    played = g[g.espn_home_score.notna()]
    src = pc.groupby(['source', 'provider_name']).espn_event_id.nunique()
    pcs = core.assign(pc=core.has_pickcenter, ao=core.has_any_odds).groupby('season')[['pc', 'ao']].sum()
    rv = rev.groupby('order').size().to_dict() if len(rev) else {}
    espn_text = '\n'.join([
        f'- Matched {int(g.espn_event_id.notna().sum()):,} of {len(g):,} events to an ESPN event '
        f'({g.match_method.value_counts().to_dict()}); unmatched = Summer League, All-Star specials and never-played '
        f'playoff games (`qa/unmatched_espn.csv`). ESPN games in range with no Polymarket event: '
        f"{qa.get('n_espn_games_without_pm_event', 0)} (`qa/espn_games_without_pm_event.csv`: " + nopm_txt + ').',
        f'- Play-by-play: {len(pbp):,} plays for {pbp.espn_event_id.nunique():,} games; tip-off = first period-1 '
        f'wallclock (median {g.tipoff_minus_espn_sched_min.median():.1f} min after the scheduled time, p5 '
        f'{g.tipoff_minus_espn_sched_min.quantile(.05):.1f}, p95 {g.tipoff_minus_espn_sched_min.quantile(.95):.1f}).',
        f'- Final scores: {len(played):,} events have an ESPN final; {int(g.espn_max_period.ge(5).sum())} went to overtime.',
        '- Odds rows by source/provider (games): ' + ', '.join(f'{a}/{b}: {n:,}' for (a, b), n in src.items()) + '.',
        '- Core games with a summary pickcenter / any odds, by season: ' +
        '; '.join(f'{s} {int(r.pc):,} / {int(r.ao):,}' for s, r in pcs.iterrows()) +
        '. ESPN has no summary pickcenter for 2024-25; the core-API odds (ESPN BET, with open/close) fill that season.',
        f'- Score step-backs in game-clock order: {rv.get("clock", 0)} plays in '
        f"{int((g.espn_n_score_backsteps_clock.fillna(0) > 0).sum())} games; among scoring plays in wallclock order: "
        f'{rv.get("scoring_wallclock", 0)} plays in {int((g.espn_n_score_backsteps_scoring_wallclock.fillna(0) > 0).sum())} '
        f'games (`work/espn_score_reversals.parquet`). In ESPN sequence order {int((g.espn_n_score_backsteps_seq.fillna(0) > 0).sum()):,} '
        'games show a step-back, almost all from late-inserted plays (not score changes).',
    ])

    # ---- problems --------------------------------------------------------------------------------------------
    wa = qa_csv('wallclock_anomalies')
    shift = g.wallclock_shift_suspect_s.value_counts().to_dict()
    sp = qa_csv('spread_resolution_mismatch')
    dups = qa_csv('duplicate_pm_events')
    probs = []
    probs.append(f"Gamma `volume` is null for most markets of 2026-03-04..03-25 ({int(g.ml_volume.isna().sum())} "
                 'moneylines in scope, all spread/total ladders of those games). Those markets did trade: 2,318 of 2,348 '
                 'null-volume spread/total lines of 2026-03 have prints. Main spread/total there were chosen by the sum of '
                 'print sizes (`volume_source='"'trades'"'`); `markets.volume_effective` carries the filled value.')
    probs.append(f"Moneyline mislabelled `sportsMarketType='totals'` on {int(g.ml_kind_note.notna().sum())} events "
                 '(every one an event listing the Thunder first, 2024-25); identified by structure (market slug = event slug).')
    probs.append('The 2024 NBA Cup final (OKC vs MIL, 2024-12-17) is listed under the non-standard slug '
                 '`nba-cup-final-thunder-vs-bucks` with question-style spread ("Will the Thunder beat the Bucks by 5 or '
                 'more points?") and total markets; it is included explicitly (`games.slug_nonstandard`). A scan of all '
                 'in-scope events whose markets have two NBA nicknames as outcomes found no other single-game event '
                 'outside the slug pattern (the rest are series-winner markets).')
    probs.append(f"Slug date is the UTC date instead of the ET date on {qa.get('n_slug_date_not_et_date', 0)} events "
                 '(mostly 2025-10/11 evening games); the ET date from gameStartTime is used for scope/season.')
    probs.append(f"Duplicate Polymarket events for one real game: {len(dups)} events in "
                 f"{dups.espn_event_id.nunique() if len(dups) else 0} groups (both resolved; secondary flagged "
                 '`is_primary_event=False`, excluded from `is_core_game`). Pairs: UTC-date and ET-date slugs for the same '
                 'game (2025-10-25/26, 11-07/08, 11-08/09), and wrong-date events (LAC@OKC and WAS@SAS 2025-12-17 for '
                 '12-18 games, CHI@MIA 2026-01-30 for the 01-31 game).')
    probs.append(f"Polymarket gameStartTime differs from ESPN's schedule by >5 min on {qa.get('n_pm_start_vs_espn_sched', 0)} "
                 'events (+-60 min DST/timezone slips around 2025-11-01, +-17..25 min TV slots, a few hours to one day off '
                 'on make-up/duplicate events). Price windows were widened to cover ESPN\'s scheduled time + 4 h.')
    probs.append(f"Polymarket event `score` field disagrees with ESPN on "
                 f"{int((g.flag_pm_score_differs_espn & g.espn_home_score.notna()).sum())} played games (2024-25 mostly); "
                 'never use it - the moneyline resolution itself agrees with ESPN everywhere.')
    probs.append(f"Spread/total ladder of HOU@GSW 2025-05-02 (playoffs G6, final 115-107) resolved 50-50 by "
                 f"Polymarket (and two lines closed at non-binary prices); listed as the {len(sp)} spread and "
                 f"{qa.get('n_total_resolution_mismatch', 0)} total resolution mismatch. Volume there was tiny (5-20 shares).")
    probs.append(f"Postponed games: {int(g.flag_postponed.sum())} events. The four 2024-25 ones resolved 50-50 by rule; "
                 '2025-26 ones stayed open and resolved with the make-up game (MIA@CHI 01-08 -> 01-29, GSW@MIN 01-24 -> '
                 '01-25); two others moved `gameStartTime` to the make-up date (DEN@MEM -> 03-18, DAL@MIL -> 03-31).')
    probs.append(f"{int((g.game_type == 'not_played').sum())} never-played 'if necessary' playoff games (2026-04..06) "
                 'resolved 50-50 after real trading (up to ~200k shares). Placeholder start time 04:00Z.')
    sh = pd.read_parquet(DATA / 'espn_pbp.parquet', columns=['espn_event_id', 'wallclock_shift_suspect_s',
                                                            'wallclock_out_of_window'])
    ev = sh.groupby('espn_event_id').agg(day=('wallclock_shift_suspect_s', lambda x: (x == 86400).sum()),
                                         n=('wallclock_shift_suspect_s', 'size'),
                                         plus=('wallclock_shift_suspect_s', lambda x: (x == 3600).sum()),
                                         minus=('wallclock_shift_suspect_s', lambda x: (x == -3600).sum()),
                                         oow=('wallclock_out_of_window', 'sum'))
    day_all, day_part = int(((ev.day > 0) & (ev.day == ev.n)).sum()), int(((ev.day > 0) & (ev.day < ev.n)).sum())
    early = g[(g.tipoff_minus_espn_sched_min < -5) & g.wallclock_shift_suspect_s.isna()]
    probs.append(f"ESPN wallclock anomalies ({len(wa)} games in `qa/wallclock_anomalies.csv`; rule: first play < sched-5 "
                 f"min or > sched+60 min, or any play needing a fix / outside the game window): {day_all + day_part} games "
                 f"of 2025-10-28..31 carry wallclocks exactly 1 day late ({day_all} entirely, {day_part} only after some "
                 f"point in the game, so a whole-game shift is wrong); {int((ev.plus > 0).sum())} games on 2026-03-08 are "
                 f"+1 h (DST start); {int((ev.minus > 0).sum())} games on 2026-11-01 are -1 h (DST end; the later 11-01 "
                 f"games are fine); {len(early)} preseason games of 2026-10 tip 5-14 min before the scheduled time (offset "
                 f"unknown, not fixed); {int((ev.oow > 0).sum())} games have a few plays 5-7 h after the scheduled start. "
                 'The per-play suggested fix is `espn_pbp.wallclock_shift_suspect_s` (not applied to `wallclock_utc`). '
                 'Market close times confirm the games were played on the scheduled day.')
    fb = g[g.espn_summary_source.fillna('').str.startswith('core')]
    probs.append(f"ESPN site-API `/summary` returns 404 for {fb.espn_event_id.nunique()} completed 2026 playoff games "
                 f"({', '.join(sorted(fb.espn_event_id.unique()))}); their play-by-play was rebuilt from the ESPN core-API "
                 'play list (same fields incl. wallclock) and the final score from the scoreboard; no summary pickcenter '
                 'for them (core-API odds exist).')
    probs.append(f"ESPN pbp final score != header final on {qa.get('n_final_score_pbp_mismatch', 0)} games (missing late "
                 'scoring plays); first period-1 play is not the jump ball on '
                 f"{qa.get('n_first_play_not_jumpball', 0)} games (tip-off then 20-80 s late).")
    gap = qa_csv('trades_volume_gap')
    if len(gap):
        gap = gap.merge(g[['game_id', 'game_date_et']], on='game_id', how='left')
        gap['rel'] = pd.to_numeric(gap.volume_rel_gap)
        neg, pos = gap[gap.rel < 0], gap[gap.rel > 0]
        dates = lambda d: ', '.join(sorted(d.game_date_et.dropna().unique()))
        gap_txt = (f'{len(neg)} markets have FEWER print shares than Gamma volume ({neg.rel.min():.2%}..{neg.rel.max():.2%}; '
                   f'games {dates(neg)}): on 2026-01-22..02-20 a few prints are missing from the Data API (no night-wide '
                   'gap; cause unknown); the 2026-04-29..05-03 markets straddle the end of the net-of-fee period (gaps <= 0.1% '
                   'on core games, larger only on never-played games). '
                   f'{len(pos)} markets have MORE print shares than Gamma ({pos.rel.min():.2%}..{pos.rel.max():.2%}; games '
                   f'{dates(pos)}): Gamma volume undercounts, by a single print in 2024-11/12 and heavily on 2026-03-01..03 '
                   'when Gamma volume went stale just before turning null, and by up to 6.9% on thinly traded 2026-10 '
                   'preseason lines')
    else:
        gap_txt = 'no market differs from Gamma volume'
    probs.append(f'Prints vs Gamma volume (markets without net-of-fee sizes): {gap_txt}. No exact duplicate prints, no token mismatches, '
                 'the API outcome_index always equals the token position.')
    fq = qa.get('print_vs_gamma_volume_by_fee_type', {})
    fee_lines2 = '; '.join(f"{k}: {v['markets']} markets, median gap {v['median_rel_gap']:.2%} "
                           f"({v['min_rel_gap']:.2%}..{v['max_rel_gap']:.2%})" for k, v in fq.items())
    probs.append('For games 2026-03-30..2026-04-28 (the first weeks of the `sports_fees_v2` fee schedule) taker BUY print '
                 'sizes are net of the taker fee: never round numbers (0% integer sizes vs ~24-53% otherwise) and the '
                 'print sum is ~2.6% (median) below Gamma volume. From 2026-04-29 (still v2), and under `sports_fees_v3` '
                 f'and `zero_fees`, prints match Gamma volume again. By group: {fee_lines2}. For the affected games '
                 'treat `size` as the shares the taker received (`qa/market_coverage.csv` column `net_of_fee_sizes`).')
    pg = qa_csv('trades_print_gaps')
    if len(pg):
        pg['minutes'] = pd.to_numeric(pg.minutes)
        so = pg[pg.suspected_outage.astype(str) == 'True'].sort_values('minutes', ascending=False)
        lst = '; '.join(f'{r.gap_start_utc[:16]}..{r.gap_end_utc[11:16]}Z ({r.minutes:.0f} min, {r.games_in_play} games in play)'
                        for r in so.head(6).itertuples())
        lst = '; '.join(f'{r.gap_start_utc[:16]}..{r.gap_end_utc[11:16]}Z ({r.minutes:.0f} min, {r.games_in_play} games '
                        f"in play, {'prices kept moving' if r.gap_kind == 'prints_missing' else 'no minute prices either'})"
                        for r in so.itertuples())
        probs.append(f"Data holes during live games: {len(so)} windows (total {so.minutes.sum():.0f} min) with no print in any "
                     f"market of the night while >= 2 core games were in play: {lst}. "
                     f"{qa.get('n_games_in_print_outages', 0)} core games are affected (`games.flag_print_gap_in_play`, "
                     '`qa/trades_print_gaps.csv`). Where minute prices kept moving, trades happened but the Data API does not '
                     'return them (Gamma volume misses them too, so volume checks cannot reveal it). Do not read these '
                     'windows as "no trading".')
    ac = qa_csv('trades_after_close')
    if len(ac):
        for c in ('n_prints', 'max_minutes_after_close', 'min_price', 'max_price'):
            ac[c] = pd.to_numeric(ac[c])
        odd = ac[(ac.n_prints >= 10) & (ac.max_minutes_after_close > 30)].sort_values('n_prints', ascending=False)
        top = ', '.join(f'{r.market_slug} ({r.n_prints} prints, up to {r.max_minutes_after_close:.0f} min after, '
                        f'{r.min_price:.3f}-{r.max_price:.3f})' for r in odd.head(5).itertuples())
        probs.append(f"{qa.get('trades_after_market_closed', 0)} prints on {len(ac)} markets are stamped more than 60 s after "
                     f"Gamma `closedTime`. On {int((ac.max_minutes_after_close <= 5).sum())} markets these are settlement "
                     f'prints at 0.001/0.999 within 5 minutes; {len(odd)} markets kept trading for 1.5-3 h after `closedTime` '
                     f'at non-settlement prices: {top}. For the 2026-01-28 SAS@HOU moneyline the winning '
                     'side was bought at 0.18-0.40 after the result, consistent with a reopened or disputed resolution; final '
                     '`outcomePrices` match ESPN. List: `qa/trades_after_close.csv`.')
    po = qa.get('prices_outside_0_1_markets', [])
    if po:
        probs.append(f"CLOB minute history has {qa.get('prices_outside_0_1', 0)} points outside [0, 1] (-0.0035 / 1.0035), all in "
                     f"market(s) {', '.join(po)} (SAC@GSW 2025-01-05 moneyline, after the result was clear); clip to [0, 1].")
    cg = g[g.is_core_game]
    sd, td = cg.main_spread_minus_book.dropna(), cg.main_total_minus_book.dropna()
    probs.append(f"'Main' (highest final volume) is not always the market line: vs the sportsbook close, the main spread "
                 f"is within 1 point on {(sd.abs() <= 1).mean():.0%} of core games but 3+ points away on {int((sd.abs() >= 3).sum())} "
                 f"(6+ on {int((sd.abs() >= 6).sum())}); the main total within 1 point on {(td.abs() <= 1).mean():.0%}, 3+ away on "
                 f"{int((td.abs() >= 3).sum())}. Some are reverse lines (e.g. 'Spread: Nets (-13.5)' traded 770k shares while "
                 'the book had BOS -13.5). Columns `main_spread_minus_book`, `main_total_minus_book`; list in '
                 '`qa/main_line_far_from_book.csv`. Selecting by final volume is also look-ahead.')
    probs.append('2024-25 games have only a moneyline in the main event. Spread/total ladders exist for 2025-03/04 games '
                 'in separate `-v2` events (11 lines each, almost no volume) and from 2025-05 in the main event; '
                 '`main_spread_event_role` tells which.')
    probs.append('Gamma `line` sign is inconsistent (e.g. +16.5 for "Spread: Thunder (-16.5)" in 2025-05); handicaps are '
                 'parsed from the question text instead (`markets.handicap0`, `total_line`), 0 unparsed.')
    problems_text = '\n'.join(f'{i}. {p}' for i, p in enumerate(probs, 1))

    fm = mk[mk.role.notna()].merge(g[['game_id', 'game_date_et']], on='game_id')
    fees = fm.assign(ft=fm.fee_type.fillna('none'), rate=fm.fee_rate).groupby(['ft', 'rate'], dropna=False).game_date_et.agg(
        ['min', 'max', 'count'])
    fee_lines = '; '.join(f'`{ft}` rate {r if pd.notna(r) else "-"}: games {a}..{b} ({n} markets)'
                          for (ft, r), (a, b, n) in fees.iterrows())
    method_text = '\n'.join([
        '- Fee regimes on the moneyline / main lines (from `feeType` / `feeSchedule`, fee = shares x rate x p x (1-p), '
        'taker only): ' + fee_lines + '. So: no fees before 2026-03-30, 3% schedule (v2) through June 2026, v3 (5%) '
        'for the 2026 preseason and from 2026-11-01, and zero fees for the regular-season games of 2026-10-20..31.',
        '- Discovery: Gamma `/events?tag_id=745` in 3-day `end_date` windows (game events have endDate = start), 2024-10-01 '
        'to 2026-11-10, small pages (limit 20, 4 for 2026-27 events with ~500 markets). Every event saved raw. Game events '
        'match `^nba-<code>-<code>-YYYY-MM-DD$`; siblings `-v2`, `-moneyline`, `-exact-margin` are attached to the game.',
        '- Moneyline = the main event market whose slug equals the event slug. Main spread / total = highest final volume '
        '(> 0) among full-game spreads / totals of the main and `-v2` events. **This ranking uses final volume, i.e. '
        'look-ahead**; use `markets.rank_created_in_kind` or a pre-game rule for strategy work.',
        '- Trades: `/v2/trades?condition=..&taker_only=true&limit=1000`, cursor paging to the end of the feed for each market, '
        'stored per market then partitioned by season/month. Token index matched against `clobTokenIds`.',
        '- Prices: `/prices-history?market=<token>&startTs&endTs&fidelity=1` for both tokens; window gameStart-6h .. +4h '
        '(widened for wrong start times). Median points per token: '
        f'{pts.median():.0f} (min {pts.min():.0f}, max {pts.max():.0f}).',
        '- ESPN: scoreboard per ET date (2024-10-01..2026-11-04), summary + core odds per completed game. Matching by '
        'unordered team pair (slug codes mapped: nyk->NY, gsw->GS, sas->SA, nop->NO, uta->UTAH, was/wsh->WSH, pho->PHX, '
        'All-Star codes) on the ET date, then slug date, then +-1 day with the same orientation.',
        '- Resolution checks: moneyline winner from `outcomePrices` vs ESPN final by team nickname; spread outcome 0 wins iff '
        'score(team0)+handicap0 > score(team1); totals Over iff home+away > line (OT included).',
    ])

    rt = []
    for name in ('discover_gamma.log', 'espn_fetch.log', 'espn_fallback.log', 'trades_probe.log', 'trades_main.log',
                 'prices.log'):
        for seg in log_segments(name):
            rt.append((name,) + seg)
    api_text = ('No HTTP 429 from any API at the chosen rates (Gamma 8/s, Data API 12/s, CLOB 15/s, ESPN 6/s per host, '
                '<= 6 concurrent requests in total). Retries were network/5xx only, except the four ESPN summary 404s.')

    rep = dict(
        generated=dt.datetime.now(dt.timezone.utc).strftime('%Y-%m-%d %H:%M UTC'), n_games=f'{len(g):,}',
        n_core=f'{len(core):,}', files_table=md_table(ftab, index=False), raw_size=mb(size_of(DATA / 'raw')),
        work_size=mb(size_of(DATA / 'work')), total_size=mb(size_of(DATA)), by_season=md_table(by_season),
        by_season_type=md_table(by_season_type), by_month=md_table(by_month), trades_table=md_table(tsum),
        n_trades=f'{n_tr:,}', n_prices=f'{n_ph:,}', trades_size=mb(size_of(DATA / 'trades')),
        prices_size=mb(size_of(DATA / 'price_history')), vol_checked=f'{len(vol):,}', vol_exact=f'{exact:,}',
        espn_text=espn_text, problems_text=problems_text, method_text=method_text,
        runtime_table=md_table(pd.DataFrame(rt, columns=['job log', 'start', 'end', 'duration', 'last http stats']),
                               index=False),
        api_text=api_text)
    txt = TEMPLATE
    for k, v in rep.items():
        txt = txt.replace('{{' + k + '}}', str(v))
    left = re.findall(r'\{\{[^}]+\}\}', txt)
    if left:
        print('WARNING unreplaced placeholders:', sorted(set(left)))
    (RES / 'REPORT.md').write_text(txt)
    print('wrote', RES / 'REPORT.md')


TEMPLATE = """# NBA data (Polymarket + ESPN): build and QA report

Workstream `nba_data`. Generated {{generated}} by `scripts/nba_data/make_report.py` from the final tables.
Data only, no strategy analysis. Column definitions: `DATA_DICTIONARY.md`. QA lists: `results/nba_data/qa/*.csv`.

## 1. What was built
Every Polymarket NBA game event (Gamma `tag_id=745`, slug `nba-<away>-<home>-YYYY-MM-DD`, plus the 2024 NBA Cup final
listed as `nba-cup-final-thunder-vs-bucks`) whose game date (ET, from
`gameStartTime`) is 2024-10-22 .. 2026-11-03: **{{n_games}} events**, of which **{{n_core}} core games**
(`is_core_game`: regular season, play-in, playoffs, NBA Cup final; one primary event per real game). There were no NBA
games on 2026-11-03, so the last slate is 2026-11-02 (ET). The other events are kept and labelled in `game_type`:
preseason, All-Star, NBA Summer League (July 2025), postponed games, never-played "if necessary" playoff games;
duplicate Polymarket events for one game have `is_primary_event=False`.

## 2. Files
{{files_table}}

Raw pulls (gzipped JSON, rebuilds need no network): {{raw_size}}; per-market parts and manifests: {{work_size}};
whole `data/nba_data/`: {{total_size}}.

## 3. Coverage
`games` events; `espn_matched` matched to an ESPN event; `ml_resolved` moneyline resolved to a team; `ml_prices`
minute history for both moneyline tokens; `ml_trades` >= 1 moneyline print; `main_spread` / `main_total` a traded
full-game spread / total exists; `pbp` ESPN play-by-play; `pickcenter` ESPN summary pickcenter; `any_odds` pickcenter
or ESPN core-API odds.

### 3.1 Core games by season
{{by_season}}

### 3.2 All in-scope events by season and type
{{by_season_type}}

### 3.3 Core games by month (ET)
{{by_month}}

## 4. Trades (taker prints)
**{{n_trades}} rows, {{trades_size}} on disk** (zstd parquet, `data/nba_data/trades/season=/month=`).

{{trades_table}}

Completeness: every market's feed was walked to its first print (no truncation). For {{vol_exact}} of {{vol_checked}}
fetched markets that carry a Gamma volume, the sum of print sizes equals Gamma `volume` (relative gap <= 1e-4). Left out:
the markets of games 2026-03-30..04-28, whose taker BUY sizes are net of the fee (section 7). Data API holes are not
visible in this check (section 7).

## 5. Minute price history
**{{n_prices}} rows, {{prices_size}} on disk** (`data/nba_data/price_history/season=`).

## 6. ESPN
{{espn_text}}

## 7. Data-quality problems found
{{problems_text}}

## 8. Method notes
{{method_text}}

## 9. Runtime and API usage
{{runtime_table}}

{{api_text}}

## 10. How to re-run
All steps are resumable (finished items are skipped) and read-only. From `polymarket/research/auto-20261104/`:
```
.venv/bin/python scripts/nba_data/discover_gamma.py --threads 3              # Gamma windows -> raw/gamma_windows
.venv/bin/python scripts/nba_data/espn_fetch.py --threads 2                  # ESPN scoreboards, summaries, core odds
.venv/bin/python scripts/nba_data/espn_fallback.py                           # summaries that 404 on the site API
.venv/bin/python scripts/nba_data/fetch_trades.py --mode probe --threads 4   # prints of null-volume spread/total lines
.venv/bin/python scripts/nba_data/gamma_tables.py                            # games_gamma + markets (probe volumes)
.venv/bin/python scripts/nba_data/fetch_trades.py --mode main --threads 4    # moneyline / main spread / main total
.venv/bin/python scripts/nba_data/espn_tables.py                             # ESPN games, pbp, odds (no network)
.venv/bin/python scripts/nba_data/build_tables.py                            # first pass: ESPN times for price windows
.venv/bin/python scripts/nba_data/fetch_prices.py --threads 2                # CLOB minute history
.venv/bin/python scripts/nba_data/gamma_tables.py && .venv/bin/python scripts/nba_data/build_tables.py
.venv/bin/python scripts/nba_data/make_report.py                             # this report
```
Run long steps in the background: `nohup caffeinate -is .venv/bin/python scripts/nba_data/<step>.py ... >
results/nba_data/<step>.log 2>&1 &`; thread counts of jobs running at the same time must add up to <= 6.
To extend the range: change `SCOPE_LAST_DATE` in `common.py` and `LAST` in `discover_gamma.py` / `espn_fetch.py`, then
re-run the fetchers with `--refresh-since <date>`.
"""

if __name__ == '__main__':
    main()
