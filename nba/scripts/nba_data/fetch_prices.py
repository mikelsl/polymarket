"""Step 4 - minute price history from CLOB /prices-history for BOTH tokens of the moneyline, main spread and main total
of every in-scope game, window [game_start_utc - 6h, game_start_utc + 4h], fidelity=1 (widened to also cover ESPN's
scheduled time when Polymarket's gameStartTime differs; run build_tables.py once before this step).

One part file per market: data/nba_data/work/price_parts/<market_id>.parquet (token_index, t, p) and a manifest
data/nba_data/work/price_manifest/<market_id>.json. Resumable (existing parts are skipped unless the window ended after
the previous pull, i.e. the game was still running). build_tables.py consolidates the parts.

Caution for users of this table (README): minute history lags around in-game events; use trade prints for any
event-time entry or exit.

Usage: .venv/bin/python scripts/nba_data/fetch_prices.py --threads 2
Threads: CLOB only; keep the total of concurrently running jobs <= 6.
"""
import argparse
import datetime as dt
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

from common import CLOB, DATA, STATS, WORK, get_json, log, write_parquet

PARTS = WORK / 'price_parts'
MANI = WORK / 'price_manifest'
BEFORE_S, AFTER_S = 6 * 3600, 4 * 3600


def fetch_market(row):
    mid = row['market_id']
    gst = int(pd.Timestamp(row['game_start_utc']).timestamp())
    # [start - 6h, start + 4h] around Polymarket gameStartTime, widened to also cover ESPN's scheduled time where it
    # differs (wrong/moved gameStartTime on ~45 events), so the played game is always inside. ESPN tip-off is not
    # used: its wallclock is shifted by +1 day / +-1 h on some dates.
    alt = [int(pd.Timestamp(x).timestamp()) for x in (row.get('espn_sched_utc'),) if x is not None and not pd.isna(x)]
    a, b = min([gst] + alt) - BEFORE_S, max([gst] + alt) + AFTER_S
    frames, man = [], dict(market_id=mid, window_start=a, window_end=b)
    for ti in (0, 1):
        tok = row[f'token_{ti}']
        js = get_json(f'{CLOB}/prices-history', dict(market=tok, startTs=a, endTs=b, fidelity=1))
        h = js.get('history') or []
        df = pd.DataFrame(h, columns=['t', 'p']) if h else pd.DataFrame(columns=['t', 'p'])
        df['token_index'] = ti
        frames.append(df)
        man[f'n_{ti}'] = len(df)
        man[f'first_t_{ti}'] = int(df.t.min()) if len(df) else None
        man[f'last_t_{ti}'] = int(df.t.max()) if len(df) else None
    df = pd.concat(frames, ignore_index=True).astype({'t': 'int64', 'p': 'float64', 'token_index': 'int8'})
    df = df.sort_values(['token_index', 't']).reset_index(drop=True)
    write_parquet(df[['token_index', 't', 'p']], PARTS / f'{mid}.parquet')
    man['fetched_at'] = int(time.time())
    (MANI / f'{mid}.json').write_text(json.dumps(man))
    return man


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--threads', type=int, default=2)
    args = ap.parse_args()
    PARTS.mkdir(parents=True, exist_ok=True)
    MANI.mkdir(parents=True, exist_ok=True)
    mk = pd.read_parquet(DATA / 'markets.parquet')
    g = pd.read_parquet(WORK / 'games_gamma.parquet')
    g = g[g.in_scope & g.game_start_utc.notna()][['game_id', 'game_start_utc']]
    if (DATA / 'games.parquet').exists():      # ESPN times from a previous build_tables.py run
        e = pd.read_parquet(DATA / 'games.parquet', columns=['game_id', 'espn_sched_utc'])
        g = g.merge(e, on='game_id', how='left')
    t = mk[mk.role.isin(['moneyline', 'main_spread', 'main_total'])].merge(g, on='game_id')
    todo = []
    for r in t.to_dict('records'):
        mf = MANI / f"{r['market_id']}.json"
        if mf.exists() and (PARTS / f"{r['market_id']}.parquet").exists():
            m = json.loads(mf.read_text())
            if m['fetched_at'] > m['window_end'] + 1800:   # pulled after the window had fully passed
                continue
        todo.append(r)
    log(f'price targets: {len(t)} markets ({2 * len(t)} tokens), to fetch: {len(todo)} (threads={args.threads})')
    done, pts, t0 = 0, 0, time.time()
    with ThreadPoolExecutor(args.threads) as ex:
        futs = {ex.submit(fetch_market, r): r for r in todo}
        for fu in as_completed(futs):
            r = futs[fu]
            try:
                m = fu.result()
                pts += m['n_0'] + m['n_1']
            except Exception as e:
                log(f"  FAILED {r['market_id']} {r['market_slug']}: {e}")
            done += 1
            if done % 250 == 0 or done == len(todo):
                el = time.time() - t0
                log(f'  {done}/{len(todo)} markets, {pts:,} points, {el / 60:.1f} min, '
                    f'eta {el / done * (len(todo) - done) / 60:.1f} min, http {STATS}')
    log('done')


if __name__ == '__main__':
    main()
