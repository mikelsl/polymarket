"""Shared helpers for the soccer_data workstream (read-only research; never places orders).

All scripts insert this directory into sys.path explicitly and are run with `python -I`.
"""
import datetime as dt
import gzip
import json
import os
import random
import sys
import threading
import time

import requests

ROOT = 'polymarket/research/auto-20261104'
DATA = f'{ROOT}/data/soccer_data'
RES = f'{ROOT}/results/soccer_data'
SCR = f'{ROOT}/scripts/soccer_data'
RAW = f'{DATA}/raw'

GAMMA = 'https://gamma-api.polymarket.com'
CLOB = 'https://clob.polymarket.com'
DATA_API = 'https://data-api.polymarket.com/v2'
ESPN = 'https://site.api.espn.com/apis/site/v2/sports/soccer'

# Polymarket slug prefix -> league metadata
LEAGUES = {
    'epl': dict(name='Premier League', series=10188, espn='eng.1', fd='E0'),
    'lal': dict(name='LaLiga', series=10193, espn='esp.1', fd='SP1'),
    'sea': dict(name='Serie A', series=10203, espn='ita.1', fd='I1'),
    'bun': dict(name='Bundesliga', series=10194, espn='ger.1', fd='D1'),
    'fl1': dict(name='Ligue 1', series=10195, espn='fra.1', fd='F1'),
}

SEASONS = {  # kickoff windows (UTC), inclusive start, exclusive end
    '2025-26': (dt.datetime(2025, 7, 1, tzinfo=dt.timezone.utc), dt.datetime(2026, 7, 1, tzinfo=dt.timezone.utc)),
    '2026-27': (dt.datetime(2026, 7, 1, tzinfo=dt.timezone.utc), dt.datetime(2026, 11, 4, tzinfo=dt.timezone.utc)),
}


def season_of(ts):
    """Season label for an epoch-second kickoff, or None if out of scope."""
    if ts is None:
        return None
    t = dt.datetime.fromtimestamp(ts, dt.timezone.utc)
    for s, (a, b) in SEASONS.items():
        if a <= t < b:
            return s
    return None


_print_lock = threading.Lock()


def log(*a):
    with _print_lock:
        print(dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), *a, flush=True)


def parse_ts(s):
    """Parse Gamma/ESPN time strings ('2025-08-15 19:00:00+00', '2025-08-15T19:00Z', ...) to epoch seconds."""
    if s is None or s == '':
        return None
    s = str(s).strip().replace(' ', 'T')
    if s.endswith('Z'):
        s = s[:-1] + '+00:00'
    if s.endswith('+00'):
        s = s + ':00'
    if 'T' in s and len(s.split('T')[1]) == 5:  # 'YYYY-MM-DDTHH:MM'
        s = s + ':00'
    try:
        t = dt.datetime.fromisoformat(s)
    except ValueError:
        # fractional seconds with odd precision, e.g. '2025-08-16T00:03:46.12345+00:00'
        import re
        s2 = re.sub(r'\.(\d+)', lambda m: '.' + (m.group(1) + '000000')[:6], s)
        t = dt.datetime.fromisoformat(s2)
    if t.tzinfo is None:
        t = t.replace(tzinfo=dt.timezone.utc)
    return int(t.timestamp())


def iso(ts):
    return None if ts is None else dt.datetime.fromtimestamp(ts, dt.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')


_tls = threading.local()


def session():
    s = getattr(_tls, 's', None)
    if s is None:
        s = requests.Session()
        s.headers['User-Agent'] = 'Mozilla/5.0 (read-only research)'
        _tls.s = s
    return s


class PermanentError(Exception):
    pass


STATS = {'req': 0, 'r429': 0, 'retry': 0}
_stats_lock = threading.Lock()


def _bump(k):
    with _stats_lock:
        STATS[k] += 1


def get_json(url, params=None, tries=8, timeout=45, ok404=False):
    """GET JSON with retry/backoff; honors Retry-After on 429. 400/404 are permanent (404 -> None if ok404)."""
    last = None
    for i in range(tries):
        try:
            _bump('req')
            r = session().get(url, params=params, timeout=timeout)
            if r.status_code == 200:
                return r.json()
            if r.status_code == 429:
                _bump('r429')
                ra = r.headers.get('Retry-After')
                wait = float(ra) if ra and ra.replace('.', '', 1).isdigit() else min(60, 2 ** i)
                log(f'429 on {url} -> wait {wait}s')
                time.sleep(wait + random.random())
                continue
            if r.status_code == 404 and ok404:
                return None
            if r.status_code in (400, 404):
                raise PermanentError(f'HTTP {r.status_code} {url} {params} {r.text[:300]}')
            last = RuntimeError(f'HTTP {r.status_code} {r.text[:200]}')
        except PermanentError:
            raise
        except Exception as e:  # network errors, JSON decode errors
            last = e
        _bump('retry')
        time.sleep(min(60, 1.5 * 2 ** i) + random.random())
    raise RuntimeError(f'fetch failed {url} {params}: {last!r}')


def get_text(url, tries=6, timeout=60):
    last = None
    for i in range(tries):
        try:
            r = session().get(url, timeout=timeout)
            if r.status_code == 200:
                return r.content
            if r.status_code == 404:
                return None
            last = RuntimeError(f'HTTP {r.status_code}')
        except Exception as e:
            last = e
        time.sleep(min(60, 2 * 2 ** i))
    raise RuntimeError(f'fetch failed {url}: {last!r}')


def write_json_gz(path, obj):
    tmp = path + '.tmp'
    with gzip.open(tmp, 'wt') as f:
        json.dump(obj, f)
    os.replace(tmp, path)


def read_json_gz(path):
    with gzip.open(path, 'rt') as f:
        return json.load(f)


def jl(x):
    """Gamma stores lists as JSON strings ('["Yes", "No"]'); return a Python list."""
    if x is None:
        return None
    if isinstance(x, list):
        return x
    try:
        return json.loads(x)
    except Exception:
        return None
