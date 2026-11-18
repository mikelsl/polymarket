"""Step 2: build Polymarket match and market metadata tables from the raw Gamma events saved by 01_discover.py.

Outputs (data/soccer_data/):
  pm_matches.parquet  one row per main event (all discovered, with in_scope flag; kickoff <= 2026-11-03 UTC)
  markets.parquet     one row per market: 1X2 home/draw/away (+ O/U 2.5 'ou25' and BTTS 'btts' when listed), in-scope only
Usage: .venv/bin/python -I scripts/soccer_data/02_build_meta.py
"""
import sys
sys.path.insert(0, 'polymarket/research/auto-20261104/scripts/soccer_data')
import glob
import json
import os
import re

import pandas as pd

from common import DATA, LEAGUES, RAW, jl, log, parse_ts, read_json_gz, season_of

SLUG = re.compile(r'^(epl|lal|sea|bun|fl1)-([a-z0-9]+)-([a-z0-9]+)-(\d{4}-\d{2}-\d{2})$')


def fnum(x):
    try:
        return None if x is None or x == '' else float(x)
    except (TypeError, ValueError):
        return None


def market_row(match_id, role, x):
    toks = jl(x.get('clobTokenIds')) or []
    outs = jl(x.get('outcomes')) or []
    prices = [fnum(p) for p in (jl(x.get('outcomePrices')) or [])]
    fs = x.get('feeSchedule') or {}
    p0 = prices[0] if prices else None
    resolved = None
    if x.get('closed') and len(prices) == 2 and set(prices) <= {0.0, 1.0} and sum(prices) == 1.0:
        resolved = bool(p0 == 1.0)
    return dict(
        match_id=match_id, role=role, market_id=str(x.get('id')), condition_id=x.get('conditionId'),
        question=x.get('question'), group_item_title=x.get('groupItemTitle'), market_slug=x.get('slug'),
        sports_market_type=x.get('sportsMarketType'), line=fnum(x.get('line')),
        outcomes=json.dumps(outs), outcome0=outs[0] if outs else None, outcome1=outs[1] if len(outs) > 1 else None,
        clob_token_ids=json.dumps(toks), token0_id=toks[0] if toks else None, token1_id=toks[1] if len(toks) > 1 else None,
        volume=fnum(x.get('volume')), volume_clob=fnum(x.get('volumeClob')), closed=bool(x.get('closed')),
        active=x.get('active'), archived=x.get('archived'),
        outcome_prices=json.dumps(prices), outcome0_final_price=p0, resolved_outcome0=resolved,
        uma_resolution_status=x.get('umaResolutionStatus'), resolved_by=x.get('resolvedBy'),
        automatically_resolved=x.get('automaticallyResolved'),
        fees_enabled=bool(x.get('feesEnabled')), fee_schedule=json.dumps(x.get('feeSchedule'), sort_keys=True) if x.get('feeSchedule') else None,
        fee_rate=fnum(fs.get('rate')), fee_exponent=fnum(fs.get('exponent')), fee_taker_only=fs.get('takerOnly'),
        fee_rebate_rate=fnum(fs.get('rebateRate')), fee_type=x.get('feeType'),
        taker_base_fee=fnum(x.get('takerBaseFee')), maker_base_fee=fnum(x.get('makerBaseFee')),
        created_at_ts=parse_ts(x.get('createdAt')), start_date_ts=parse_ts(x.get('startDate')),
        end_date_ts=parse_ts(x.get('endDate')), closed_time_ts=parse_ts(x.get('closedTime')),
        game_start_time_ts=parse_ts(x.get('gameStartTime')),
        neg_risk=x.get('negRisk'), neg_risk_market_id=x.get('negRiskMarketID'),
        order_price_min_tick_size=fnum(x.get('orderPriceMinTickSize')), order_min_size=fnum(x.get('orderMinSize')),
        seconds_delay=fnum(x.get('secondsDelay')), last_trade_price=fnum(x.get('lastTradePrice')),
    )


def main():
    matches, markets = [], []
    for f in sorted(glob.glob(f'{RAW}/gamma/main/*.json.gz')):
        e = read_json_gz(f)
        slug = e['slug']
        lg, hc, ac, d = SLUG.match(slug).groups()
        parts = re.split(r'\s+vs\.?\s+', e.get('title') or '', maxsplit=1)
        home, away = (parts + [None, None])[:2]
        roles = {}
        for x in e.get('markets') or []:
            suf = (x.get('slug') or '')[len(slug) + 1:]
            role = 'home' if suf == hc else 'away' if suf == ac else 'draw' if suf == 'draw' else None
            if role is None:  # fallback on question text
                q = x.get('question') or ''
                role = 'draw' if 'draw' in q.lower() else 'home' if home and home in q else 'away' if away and away in q else 'unknown'
            roles.setdefault(role, x)
        gst = sorted({parse_ts(x.get('gameStartTime')) for x in roles.values() if x.get('gameStartTime')})
        kick = gst[0] if gst else parse_ts(e.get('endDate'))
        season = season_of(kick)
        more_path = f'{RAW}/gamma/more/{slug}-more-markets.json.gz'
        more = read_json_gz(more_path) if os.path.exists(more_path) else None
        ou = bt = None
        if more:
            for x in more.get('markets') or []:
                if x.get('sportsMarketType') == 'totals' and fnum(x.get('line')) == 2.5 and ou is None:
                    ou = x
                if x.get('sportsMarketType') == 'both_teams_to_score' and bt is None:
                    bt = x
        res = {r: market_row(slug, r, x) for r, x in roles.items()}
        won = [r for r in ('home', 'draw', 'away') if r in res and res[r]['resolved_outcome0'] is True]
        all_res = all(r in res and res[r]['resolved_outcome0'] is not None for r in ('home', 'draw', 'away'))
        matches.append(dict(
            match_id=slug, league=lg, league_name=LEAGUES[lg]['name'], season=season,
            in_scope=season is not None, pm_event_id=str(e['id']), pm_title=e.get('title'),
            home_team_pm=home, away_team_pm=away, home_code=hc, away_code=ac, slug_date=d,
            kickoff_ts=kick, kickoff_source='gameStartTime' if gst else 'event.endDate', n_gamestart_values=len(gst),
            event_end_date_ts=parse_ts(e.get('endDate')), event_start_date_ts=parse_ts(e.get('startDate')),
            event_created_at_ts=parse_ts(e.get('createdAt')), event_closed=bool(e.get('closed')),
            event_closed_time_ts=parse_ts(e.get('closedTime')), event_volume=fnum(e.get('volume')),
            event_score=e.get('score'), event_period=e.get('period'), event_finished_ts=parse_ts(e.get('finishedTimestamp')),
            pm_game_id=e.get('gameId'), neg_risk=e.get('negRisk'), series_slug=e.get('seriesSlug'),
            n_1x2=sum(r in res for r in ('home', 'draw', 'away')), unknown_roles=int('unknown' in res),
            vol_home=res.get('home', {}).get('volume'), vol_draw=res.get('draw', {}).get('volume'), vol_away=res.get('away', {}).get('volume'),
            has_more_markets=more is not None, has_ou25=ou is not None, has_btts=bt is not None,
            pm_result=won[0][0].upper() if len(won) == 1 else ('MULTI' if len(won) > 1 else ('NONE' if all_res else 'UNRESOLVED')),
            fees_enabled_1x2=any(res[r]['fees_enabled'] for r in res), fee_rate_1x2=max([res[r]['fee_rate'] or 0 for r in res] or [0]),
        ))
        if season is None:
            continue
        markets += list(res.values())
        if ou:
            markets.append(market_row(slug, 'ou25', ou))
        if bt:
            markets.append(market_row(slug, 'btts', bt))
    mdf = pd.DataFrame(matches)
    mdf['kickoff_utc'] = pd.to_datetime(mdf['kickoff_ts'], unit='s', utc=True)
    kdf = pd.DataFrame(markets)
    kdf = kdf.merge(mdf[['match_id', 'league', 'season', 'kickoff_ts']], on='match_id', how='left')
    for c in [c for c in kdf.columns if c.endswith('_ts')]:
        kdf[c] = kdf[c].astype('Int64')
    for c in [c for c in mdf.columns if c.endswith('_ts')]:
        mdf[c] = mdf[c].astype('Int64')
    mdf.to_parquet(f'{DATA}/pm_matches.parquet', index=False)
    kdf.to_parquet(f'{DATA}/markets.parquet', index=False)
    log('pm_matches', len(mdf), 'in scope', int(mdf.in_scope.sum()), 'markets', len(kdf), kdf.role.value_counts().to_dict())
    log(mdf[mdf.in_scope].groupby(['season', 'league']).size().to_dict())
    log('pm_result', mdf[mdf.in_scope].pm_result.value_counts().to_dict())


if __name__ == '__main__':
    main()
