"""Collect headline numbers of the nba_live workstream into results/nba_live/summary.json (from the CSV/parquet outputs)."""
import json
import numpy as np
import pandas as pd
from common import DL, RL
g = pd.read_parquet(DL / 'games.parquet')
A = pd.read_parquet(DL / 'alignment_flags.parquet')
out = {'games': g.groupby(['season', 'split']).size().rename('n').reset_index().to_dict('records'),
       'alignment': A.groupby('season').agg(games=('game_id', 'size'), align_ok=('align_ok', 'sum'), median_lag_s=('lag_used', 'median')).reset_index().to_dict('records'),
       'model_params': json.load(open(DL / 'model_params.json'))}
for name in ['best_of_train_t_tr', 'best_of_train_mean_tr', 'h6_stops', 'h4_summary', 'h3_summary', 'h1_rules', 'h2_rules', 'h5b_rules_spread', 'h5b_rules_total', 'h5c_rules', 'h1b_rules_ml', 'depth_by_price']:
    f = RL / f'{name}.csv'
    if f.exists():
        out[name] = pd.read_csv(f).replace({np.nan: None}).to_dict('records')
json.dump(out, open(RL / 'summary.json', 'w'), indent=1, default=str)
print('wrote', RL / 'summary.json')
