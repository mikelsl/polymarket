"""Step 2c - fallback for completed ESPN games whose site-API /summary returns 404 (seen for two 2026 playoff games).

Tries, in order:
  1. site.web.api.espn.com/apis/site/v2/sports/basketball/nba/summary?event=<id>  (same JSON as /summary)
  2. core API play list  sports.core.api.espn.com/v2/.../events/<id>/competitions/<id>/plays (paged), wrapped into a
     summary-like object {header from the scoreboard, plays in site-API shape, pickcenter: []}
The result is written to raw/espn_summary/<id>.json.gz with a top-level key `_fallback_source`.

Usage: .venv/bin/python scripts/nba_data/espn_fallback.py      (single-threaded, a handful of requests)
"""
import re

from common import ESPN_CORE, RAW, HTTPFail, get_json, log, read_json_gz, write_json_gz

SB_DIR, SUM_DIR = RAW / 'espn_scoreboard', RAW / 'espn_summary'


def core_plays(eid):
    plays, page = [], 1
    while True:
        js = get_json(f'{ESPN_CORE}/events/{eid}/competitions/{eid}/plays', dict(limit=1000, page=page), tries=4,
                      allow_404=True)
        if not js:
            return None
        for p in js.get('items') or []:
            tref = (p.get('team') or {}).get('$ref') or ''
            m = re.search(r'/teams/(\d+)', tref)
            plays.append(dict(id=p.get('id'), sequenceNumber=p.get('sequenceNumber'), type=p.get('type'),
                              text=p.get('text'), awayScore=p.get('awayScore'), homeScore=p.get('homeScore'),
                              period=p.get('period'), clock=p.get('clock'), scoringPlay=p.get('scoringPlay'),
                              scoreValue=p.get('scoreValue'), team={'id': m.group(1)} if m else None,
                              wallclock=p.get('wallclock')))
        if page >= (js.get('pageCount') or 1):
            return plays
        page += 1


def main():
    events = {}
    for f in sorted(SB_DIR.glob('*.json.gz')):
        for e in read_json_gz(f).get('events') or []:
            st = (((e.get('competitions') or [{}])[0].get('status') or {}).get('type') or {})
            if st.get('completed'):
                events[str(e['id'])] = e
    missing = [eid for eid in events if not (SUM_DIR / f'{eid}.json.gz').exists()]
    log(f'completed ESPN events without summary: {missing}')
    for eid in missing:
        try:
            js = get_json('https://site.web.api.espn.com/apis/site/v2/sports/basketball/nba/summary', dict(event=eid),
                          tries=3, allow_404=True)
        except HTTPFail as e:
            js = None
            log(f'  {eid} site.web failed: {e}')
        if js and js.get('plays'):
            js['_fallback_source'] = 'site.web.api summary'
            write_json_gz(SUM_DIR / f'{eid}.json.gz', js)
            log(f'  {eid}: site.web.api summary ok, {len(js["plays"])} plays')
            continue
        plays = core_plays(eid)
        if plays:
            e = events[eid]
            comp = (e.get('competitions') or [{}])[0]
            header = {'competitions': [{'date': e.get('date'), 'status': comp.get('status'),
                                        'competitors': comp.get('competitors')}]}
            write_json_gz(SUM_DIR / f'{eid}.json.gz', {'_fallback_source': 'core plays + scoreboard header',
                                                       'header': header, 'plays': plays, 'pickcenter': []})
            log(f'  {eid}: core plays ok, {len(plays)} plays')
        else:
            log(f'  {eid}: no fallback source has plays')


if __name__ == '__main__':
    main()
