"""Step 3b: re-fetch minute price history for matches whose canonical kickoff (ESPN scheduled time, see matches.parquet)
differs from Polymarket gameStartTime by more than 30 minutes (postponed / re-dated matches), so the window
[kickoff - 6 h, kickoff + 2.5 h] covers the real match. Writes raw/prices/prices_raw_fix.jsonl (overwritten).
Run after 07_build_matches_events.py and after 03_prices.py has finished.
Usage: .venv/bin/python -I scripts/soccer_data/03b_prices_fix.py
"""
import sys
sys.path.insert(0, 'polymarket/research/auto-20261104/scripts/soccer_data')
import concurrent.futures as cf
import json

import pandas as pd

from common import CLOB, DATA, RAW, get_json, log

OUT = f'{RAW}/prices/prices_raw_fix.jsonl'
PRE, POST = 6 * 3600, int(2.5 * 3600)


def fetch(j):
    s, e = int(j['kickoff_ts']) - PRE, int(j['kickoff_ts']) + POST
    r = get_json(f'{CLOB}/prices-history', dict(market=j['token0_id'], startTs=s, endTs=e, fidelity=1))
    h = [[int(p['t']), float(p['p'])] for p in (r.get('history') or [])]
    return dict(match_id=j['match_id'], role=j['role'], condition_id=j['condition_id'], token_id=j['token0_id'],
                outcome=j['outcome0'], start_ts=s, end_ts=e, n=len(h), history=h, window_basis='espn_scheduled')


def main():
    m = pd.read_parquet(f'{DATA}/matches.parquet')
    mk = pd.read_parquet(f'{DATA}/markets.parquet')
    shifted = m[(m.kickoff_ts - m.pm_game_start_ts).abs() > 1800][['match_id', 'kickoff_ts']]
    jobs = mk.drop(columns=['kickoff_ts'], errors='ignore').merge(shifted, on='match_id')
    jobs = jobs[['match_id', 'role', 'condition_id', 'token0_id', 'outcome0', 'kickoff_ts']].to_dict('records')
    log(f'shifted matches {len(shifted)}, tokens {len(jobs)}')
    with cf.ThreadPoolExecutor(2) as ex:
        recs = list(ex.map(fetch, jobs))
    with open(OUT, 'w') as f:
        for r in recs:
            f.write(json.dumps(r) + '\n')
    log('written', len(recs), 'points', sum(r['n'] for r in recs))


if __name__ == '__main__':
    main()
