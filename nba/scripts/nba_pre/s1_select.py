"""Honest out-of-sample check per hypothesis family: pick the variant with the best TRAIN ROI (n_train >= 50,
and >= 100 for big families), then report its untouched TEST result. Also counts variants per family and how many
are positive with a 95% CI above zero in train / test.
Output: results/nba_pre/selection.csv
"""
import pandas as pd

from common_pre import RES

fam = {
    'H1 calibration rules': ('h1_strategies.csv', ['entry', 'rule'], 50),
    'H2 PM vs book value': ('h2_value_rules.csv', ['setup', 'devig', 'mirror', 'min_edge'], 50),
    'H3 dynamics rules': ('h3_rules.csv', ['rule', 't1', 't2', 'x', 'exit'], 50),
    'H4B line value vs book': ('h4_value.csv', ['entry', 'book', 'kind', 'scope', 'min_edge'], 50),
    'H4C over/under, cover': ('h4_overunder.csv', ['rule'], 50),
    'H4D line tails': ('h4_tails.csv', ['entry', 'kind', 'band'], 100),
    'H4E ML vs PM spread': ('h4_ml_vs_spread.csv', ['min_edge'], 50),
    'H5 situational rules': ('h5_rules.csv', ['rule'], 50),
}
rows = []
for name, (f, keys, nmin) in fam.items():
    try:
        d = pd.read_csv(RES / f)
    except FileNotFoundError:
        continue
    d = d[d.split.isin(['train', 'test'])]
    for k in keys:
        d[k] = d[k].astype(str).replace('nan', '')
    tr = d[d.split == 'train'].set_index(keys)
    te = d[d.split == 'test'].set_index(keys)
    nvar = len(tr)
    ok = tr[tr.n >= nmin]
    if not len(ok):
        continue
    best = ok.roi.idxmax()
    t, s = ok.loc[best], te.loc[best] if best in te.index else None
    rows.append(dict(family=name, variants=nvar, eligible=len(ok),
                     train_ci_above0=int((tr.roi_lo > 0).sum()),
                     test_ci_above0=int((te.roi_lo > 0).sum()) if len(te) else 0,
                     chosen=' | '.join(map(str, best if isinstance(best, tuple) else (best,))),
                     n_train=int(t.n), roi_train=t.roi, ci_train=f'[{t.roi_lo:.3f}, {t.roi_hi:.3f}]',
                     n_test=int(s.n) if s is not None else 0, roi_test=s.roi if s is not None else None,
                     ci_test=f'[{s.roi_lo:.3f}, {s.roi_hi:.3f}]' if s is not None else '',
                     pnl_test=s.pnl if s is not None else None))
out = pd.DataFrame(rows)
out.round(4).to_csv(RES / 'selection.csv', index=False)
pd.set_option('display.width', 250)
print(out.round(3).to_string(index=False))
