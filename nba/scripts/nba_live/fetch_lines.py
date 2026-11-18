"""Fetch taker prints for decision-time spread/total lines (no look-ahead line choice), 2025-26+ core games.

Line choice per game uses pre-game information only: the full-game total (spread) line closest to the sportsbook closing
total (home spread) from ESPN; ties -> the earliest created market. Prints already pulled by nba_data
(data/nba_data/work/trades_parts/<market_id>.parquet) are reused; the rest are fetched from the Data API
/v2/trades?condition=..&taker_only=true (cursor paging to the first print) into data/nba_live/trades_lines/.
Polite: 2 threads, <= 4 requests/s from this job, exponential backoff, honours Retry-After. Resumable.
Output: data/nba_live/lines_selected.parquet (one row per game x kind) and the per-market parquet files.
"""
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import pandas as pd
import requests
from common import ROOT, DL, load_games

API = 'https://data-api.polymarket.com/v2/trades'
OUT = DL / 'trades_lines'
OUT.mkdir(exist_ok=True)
NBA_PARTS = ROOT / 'data' / 'nba_data' / 'work' / 'trades_parts'
RPS = 4.0
_lock = threading.Lock()
_next = [0.0]
STATS = {'req': 0, 'retry': 0, '429': 0}


def wait():
    with _lock:
        now = time.monotonic(); t = max(now, _next[0]); _next[0] = t + 1.0 / RPS
    if t > now:
        time.sleep(t - now)


def get(params, tries=8):
    sess = requests.Session()
    for k in range(tries):
        wait()
        try:
            r = sess.get(API, params=params, timeout=60)
            STATS['req'] += 1
            if r.status_code == 429:
                STATS['429'] += 1
                ra = float(r.headers.get('Retry-After', 5))
                with _lock:
                    _next[0] = max(_next[0], time.monotonic() + ra)
                continue
            if r.status_code >= 500:
                raise RuntimeError(r.status_code)
            r.raise_for_status()
            return r.json()
        except Exception:
            STATS['retry'] += 1
            time.sleep(min(60, 2 ** k))
    raise RuntimeError('failed ' + str(params))


def fetch(row):
    f = OUT / f"{row['market_id']}.parquet"
    if f.exists():
        return row['market_id'], -1
    toks = {row['token_0']: 0, row['token_1']: 1}
    recs, cur = [], None
    while True:
        params = dict(condition=row['condition_id'], taker_only='true', limit=1000)
        if cur:
            params['cursor'] = cur
        js = get(params)
        data = js.get('data') or []
        for r in data:
            recs.append((r.get('timestamp'), toks.get(str(r.get('token_id')), -1), r.get('side'), r.get('price'), r.get('size')))
        cur = (js.get('pagination') or {}).get('next_cursor')
        if not cur or not data:
            break
    df = pd.DataFrame(recs, columns=['ts', 'token_index', 'side', 'price', 'size']).astype({'ts': 'int64', 'token_index': 'int8', 'price': float, 'size': float})
    df.sort_values('ts', kind='stable').to_parquet(f, index=False)
    return row['market_id'], len(df)


def main():
    g = load_games()
    g = g[g.season != '2024-25']
    m = pd.read_parquet(ROOT / 'data' / 'nba_data' / 'markets.parquet',
                        columns=['game_id', 'market_id', 'condition_id', 'kind', 'event_role', 'total_line', 'handicap0', 'team0', 'team1',
                                 'token_0', 'token_1', 'created_at', 'winner_index', 'volume_effective', 'market_slug'])
    m = m[m.game_id.isin(g.game_id) & m.event_role.isin(['main', 'v2'])]
    sel = []
    gg = g.set_index('game_id')
    # totals
    t = m[m.kind == 'totals'].copy()
    t['dist'] = (t.total_line - t.game_id.map(gg.book_total_close)).abs()
    t = t.dropna(subset=['dist']).sort_values(['game_id', 'dist', 'created_at'])
    ts = t.groupby('game_id').head(1).assign(sel_kind='total')
    sel.append(ts)
    # spreads: home handicap of each market
    s = m[m.kind == 'spreads'].copy()
    home_name = s.game_id.map(gg.espn_home_abbr)  # abbr, need nickname: use ML outcomes
    G0 = pd.read_parquet(ROOT / 'data' / 'nba_data' / 'games.parquet', columns=['game_id', 'espn_home_name', 'espn_away_name'])
    s = s.merge(G0, on='game_id')
    s['team0_home'] = s.team0 == s.espn_home_name
    s['team1_home'] = s.team1 == s.espn_home_name
    s = s[s.team0_home | s.team1_home]
    s['home_hcp'] = np.where(s.team0_home, s.handicap0, -s.handicap0)
    s['dist'] = (s.home_hcp - s.game_id.map(gg.book_spread_home_close)).abs()
    s = s.dropna(subset=['dist']).sort_values(['game_id', 'dist', 'created_at'])
    ss = s.groupby('game_id').head(1).assign(sel_kind='spread')
    sel.append(ss)
    S = pd.concat(sel, ignore_index=True)
    S['have_nba_data'] = S.market_id.map(lambda x: (NBA_PARTS / f'{x}.parquet').exists())
    keep = ['game_id', 'sel_kind', 'market_id', 'condition_id', 'market_slug', 'total_line', 'handicap0', 'team0', 'team1', 'home_hcp', 'dist',
            'token_0', 'token_1', 'winner_index', 'volume_effective', 'have_nba_data']
    S[[c for c in keep if c in S.columns]].to_parquet(DL / 'lines_selected.parquet', index=False)
    todo = S[~S.have_nba_data & S.condition_id.notna()].to_dict('records')
    print('selected', S.groupby('sel_kind').size().to_dict(), 'already have', int(S.have_nba_data.sum()), 'to fetch', len(todo), flush=True)
    t0 = time.time(); done = 0; n = 0
    with ThreadPoolExecutor(2) as ex:
        futs = [ex.submit(fetch, r) for r in todo]
        for fu in as_completed(futs):
            try:
                _, k = fu.result(); n += max(k, 0)
            except Exception as e:
                print('FAILED', e, flush=True)
            done += 1
            if done % 100 == 0 or done == len(todo):
                print(f'{done}/{len(todo)} prints {n} {time.time() - t0:.0f}s {STATS}', flush=True)
    print('done', STATS, flush=True)


if __name__ == '__main__':
    main()
