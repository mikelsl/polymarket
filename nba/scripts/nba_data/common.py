"""Shared helpers for the nba_data workstream: paths, polite HTTP with per-host rate limits, logging, small parsers.

Read-only research code. Nothing here signs, places or cancels orders.
Concurrency rule (README): at most 6 concurrent requests from this worker in total. Each job takes a fixed number of
threads (see the job docstrings); the jobs that run at the same time must add up to <= 6.
"""
import datetime as dt
import gzip
import json
import os
import random
import re
import sys
import threading
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[2]          # .../research/auto-20261104
DATA = ROOT / 'data' / 'nba_data'
RAW = DATA / 'raw'
WORK = DATA / 'work'
RES = ROOT / 'results' / 'nba_data'
for _p in (DATA, RAW, WORK, RES):
    _p.mkdir(parents=True, exist_ok=True)

# Scope (ET game dates, from the Polymarket slug): 2024-10-22 .. 2026-11-03 inclusive.
SCOPE_FIRST_DATE = '2024-10-22'
SCOPE_LAST_DATE = '2026-11-03'

GAMMA = 'https://gamma-api.polymarket.com'
DATA_API = 'https://data-api.polymarket.com/v2'
CLOB = 'https://clob.polymarket.com'
ESPN_SITE = 'https://site.api.espn.com/apis/site/v2/sports/basketball/nba'
ESPN_CORE = 'https://sports.core.api.espn.com/v2/sports/basketball/leagues/nba'

# Requests per second per host from this worker. The README budgets are for ALL workers together
# (/v2/trades 30/s, prices-history 100/s, Gamma ~30/s), so stay well below them.
HOST_RPS = {
    'gamma-api.polymarket.com': 8.0,
    'data-api.polymarket.com': 12.0,
    'clob.polymarket.com': 15.0,
    'site.api.espn.com': 6.0,
    'sports.core.api.espn.com': 6.0,
}


def log(*a):
    print(dt.datetime.now(dt.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'), *a, flush=True)


class _Bucket:
    def __init__(self, rps):
        self.interval = 1.0 / rps
        self.next_t = 0.0
        self.lock = threading.Lock()

    def wait(self):
        with self.lock:
            now = time.monotonic()
            t = max(now, self.next_t)
            self.next_t = t + self.interval
        if t > now:
            time.sleep(t - now)

    def penalize(self, secs):
        with self.lock:
            self.next_t = max(self.next_t, time.monotonic() + secs)


_buckets = {h: _Bucket(r) for h, r in HOST_RPS.items()}
_tls = threading.local()
STATS = {'requests': 0, 'retries': 0, 'http429': 0, 'errors': 0}
_stats_lock = threading.Lock()


def _bump(k, n=1):
    with _stats_lock:
        STATS[k] += n


def session():
    s = getattr(_tls, 's', None)
    if s is None:
        s = requests.Session()
        s.headers['User-Agent'] = 'Mozilla/5.0 (research; read-only)'
        _tls.s = s
    return s


class HTTPFail(RuntimeError):
    pass


def get_json(url, params=None, tries=8, timeout=60, allow_404=False):
    """GET with per-host rate limit, exponential backoff, Retry-After on 429. Returns parsed JSON (or None on 404
    when allow_404). Raises HTTPFail after `tries` attempts or on a 4xx other than 404/408/425/429."""
    host = re.sub(r'^https?://([^/]+)/.*$', r'\1', url)
    bucket = _buckets.get(host)
    last = None
    for i in range(tries):
        if bucket:
            bucket.wait()
        _bump('requests')
        try:
            r = session().get(url, params=params, timeout=timeout)
        except Exception as e:  # network error / timeout
            last = repr(e)
            _bump('retries')
            time.sleep(min(60, 2 ** i) + random.random())
            continue
        if r.status_code == 200:
            try:
                return r.json()
            except Exception as e:
                last = f'bad json {e!r} {r.text[:120]!r}'
                _bump('retries')
                time.sleep(min(60, 2 ** i) + random.random())
                continue
        if r.status_code == 404 and allow_404:
            return None
        if r.status_code == 429:
            _bump('http429')
            ra = r.headers.get('Retry-After')
            try:
                wait = float(ra) if ra else min(60, 2 ** (i + 1))
            except ValueError:
                wait = min(60, 2 ** (i + 1))
            if bucket:
                bucket.penalize(wait)
            time.sleep(wait + random.random())
            last = '429'
            continue
        if 400 <= r.status_code < 500 and r.status_code not in (404, 408, 425):
            _bump('errors')
            raise HTTPFail(f'{r.status_code} {r.url}: {r.text[:300]!r}')
        last = f'{r.status_code} {r.text[:200]!r}'
        _bump('retries')
        time.sleep(min(60, 2 ** i) + random.random())
    _bump('errors')
    raise HTTPFail(f'giving up after {tries} tries: {url} {params} last={last}')


def write_json_gz(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f'.tmp{os.getpid()}_{threading.get_ident()}')
    with gzip.open(tmp, 'wt', encoding='utf-8') as f:
        json.dump(obj, f, separators=(',', ':'))
    os.replace(tmp, path)


def read_json_gz(path):
    with gzip.open(path, 'rt', encoding='utf-8') as f:
        return json.load(f)


def write_parquet(df, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f'.tmp{os.getpid()}_{threading.get_ident()}')
    df.to_parquet(tmp, index=False, compression='zstd')
    os.replace(tmp, path)


def parse_ts(s):
    """Gamma/ESPN timestamps ('2024-10-22 23:30:00+00', '2024-10-22T23:30:00Z', '2024-10-22T23:30Z',
    '2024-10-21T20:19:55.5803Z') -> aware UTC datetime, or None."""
    if not s:
        return None
    s = str(s).strip().replace(' ', 'T')
    if s.endswith('Z'):
        s = s[:-1] + '+00:00'
    if re.search(r'[+-]\d\d$', s):
        s += ':00'
    m = re.match(r'^(\d{4}-\d\d-\d\dT\d\d:\d\d)(\+.*|-.*)?$', s)
    if m:  # minute precision, no seconds
        s = m.group(1) + ':00' + (m.group(2) or '+00:00')
    m = re.match(r'^(.*T\d\d:\d\d:\d\d)\.(\d+)(.*)$', s)
    if m:  # python < 3.11 style fraction normalisation (keep microseconds)
        s = m.group(1) + '.' + (m.group(2) + '000000')[:6] + m.group(3)
    try:
        d = dt.datetime.fromisoformat(s)
    except ValueError:
        return None
    if d.tzinfo is None:
        d = d.replace(tzinfo=dt.timezone.utc)
    return d.astimezone(dt.timezone.utc)


def epoch(s):
    d = parse_ts(s)
    return int(d.timestamp()) if d else None


def jload(x, default=None):
    if x is None:
        return default
    if isinstance(x, (list, dict)):
        return x
    try:
        return json.loads(x)
    except Exception:
        return default


GAME_SLUG_RE = re.compile(r'^nba-([a-z]{2,5})-([a-z]{2,5})-(\d{4}-\d{2}-\d{2})$')


def season_of(date_str):
    """NBA season label for an ET game date 'YYYY-MM-DD': Aug..Dec -> 'YYYY-(YY+1)', Jan..Jul -> '(YYYY-1)-YY'."""
    y, m = int(date_str[:4]), int(date_str[5:7])
    start = y if m >= 8 else y - 1
    return f'{start}-{str(start + 1)[2:]}'
