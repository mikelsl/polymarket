"""H1 descriptive: outcome residual vs market mid in specific game states (leading team / pre-game favourite view).
Rows: 30-s panel, aligned games, info delay 15 s + game lag; market stable (|mid - mid 20 s ago| <= 2c), no known score in
the last 30 s. Residual = win - mid of the side named in the state; CI clustered by game. Also the executable P&L of buying
that side at the next ask print and holding (USDC per 10 USDC, 0.05 fee)."""
import numpy as np, pandas as pd
from features import load_panel, pnl_hold
from common import summarize, RL
p = load_panel(L=15, align_filter=True)
p = p[p.mid.between(0.02, 0.98)].copy()
p['grp'] = np.where(p.season == '2024-25', 'tr24', np.where(p.split == 'train', 'tr25', 'test'))
p['clean'] = ((p.mid - p.mid_m20).abs() <= 0.02) & (p.since_score >= 30)
p = p[p.clean]
lead_home = p.m > 0
p['lead_mid'] = np.where(lead_home, p.mid, 1 - p.mid)
p['lead_win'] = np.where(lead_home, p.home_win, 1 - p.home_win)
p['lead_px'] = np.where(lead_home, p.buyH_px, p.buyA_px)
p['trail_px'] = np.where(lead_home, p.buyA_px, p.buyH_px)
fav_home = p.pre_ph >= 0.5
p['pf_mid'] = np.where(fav_home, p.mid, 1 - p.mid)
p['pf_win'] = np.where(fav_home, p.home_win, 1 - p.home_win)
p['pf_px'] = np.where(fav_home, p.buyH_px, p.buyA_px)
p['pf_p0'] = np.where(fav_home, p.pre_ph, 1 - p.pre_ph)
p['pf_margin'] = np.where(fav_home, p.m, -p.m)
am = p.m.abs(); rem = p.rem; reg = ~p.ot
S = {
 'early(0-6m) lead>=8: leader': (reg & (rem >= 2520) & (am >= 8), 'lead'),
 'early(0-6m) lead 4-7: leader': (reg & (rem >= 2520) & am.between(4, 7), 'lead'),
 'Q1-Q2 lead>=15: leader': (reg & (rem >= 1440) & (am >= 15), 'lead'),
 'Q3 lead>=15: leader': (reg & rem.between(720, 1440) & (am >= 15), 'lead'),
 'Q3 lead 8-14: leader': (reg & rem.between(720, 1440) & am.between(8, 14), 'lead'),
 'Q4(6-12m) lead>=12: leader': (reg & rem.between(360, 720) & (am >= 12), 'lead'),
 'Q4(2-6m) lead 4-9: leader': (reg & rem.between(120, 360) & am.between(4, 9), 'lead'),
 'Q4(0-2m) lead 1-3: leader': (reg & (rem < 120) & am.between(1, 3), 'lead'),
 'Q4(0-2m) lead 4-6: leader': (reg & (rem < 120) & am.between(4, 6), 'lead'),
 'Q4(0-2m) tied: home': (reg & (rem < 120) & (am == 0), 'home'),
 'Q2-Q3 pre-fav(p0>=.65) trails by >=10: pre-fav': (reg & rem.between(720, 2160) & (p.pf_p0 >= 0.65) & (p.pf_margin <= -10), 'pf'),
 'Q4 pre-fav(p0>=.65) trails by 5-12 (2-12m): pre-fav': (reg & rem.between(120, 720) & (p.pf_p0 >= 0.65) & p.pf_margin.between(-12, -5), 'pf'),
 'Q2-Q3 pre-dog(p0<=.35) leads by >=10: pre-fav': (reg & rem.between(720, 2160) & (p.pf_p0 >= 0.65) & (p.pf_margin >= -100) & (-p.pf_margin >= 10) , 'pf'),
 'OT (any): home': (p.ot, 'home'),
}
out = []
for name, (msk, side) in S.items():
    d = p[msk].copy()
    if side == 'lead':
        d['res'] = d.lead_win - d.lead_mid; d['pnl'] = pnl_hold(d.lead_px, d.lead_win); mid = d.lead_mid
    elif side == 'pf':
        d['res'] = d.pf_win - d.pf_mid; d['pnl'] = pnl_hold(d.pf_px, d.pf_win); mid = d.pf_mid
    else:
        d['res'] = d.home_win - d.mid; d['pnl'] = pnl_hold(d.buyH_px, d.home_win); mid = d.mid
    for gname, x in d.groupby('grp'):
        s = summarize(x, 'res', n_boot=300); s2 = summarize(x[x.pnl.notna()], 'pnl', n_boot=300)
        out.append(dict(state=name, grp=gname, rows=len(x), games=x.game_id.nunique(), mid=mid[x.index].mean(), win=(x.res + mid[x.index]).mean(),
                        res=s.pnl_mean[0], res_lo=s.ci_lo[0], res_hi=s.ci_hi[0], pnl=s2.pnl_mean[0], pnl_lo=s2.ci_lo[0], pnl_hi=s2.ci_hi[0]))
O = pd.DataFrame(out); O.to_csv(RL / 'h1_states.csv', index=False)
pd.set_option('display.width', 250)
print(O.round(3).to_string())
