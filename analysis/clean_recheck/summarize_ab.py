"""
analysis/clean_recheck/summarize_ab.py
Summarises settings_ab.py: action-level table per setting (seed 0), clamped-cone shares, radius ranges, last-10-epoch
loss terms. Applies the pre-registered rule: a setting "clearly beats" the base if rho (all actions, >= 3 held-out)
is >= 0.05 higher than the base and higher in >= 7 of 9 people.

Run (intention env): python analysis/clean_recheck/summarize_ab.py
Output: ab_summary.json
"""
import glob
import json
import pathlib

import numpy as np
import pandas as pd
from scipy.stats import t as tdist

HERE = pathlib.Path(__file__).resolve().parent
TAB = HERE / "tables"
BASE, EUC = "base K0.05 w1.0", "euclidean"


def paired(a, b):
    d = (a - b).dropna()
    h = tdist.ppf(.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
    return dict(mean=round(float(d.mean()), 3), ci95=[round(float(d.mean() - h), 3), round(float(d.mean() + h), 3)],
                wins=f"{int((d > 0).sum())}/{len(d)}")


def main():
    a = pd.read_csv(TAB / "ab_action_level.csv")
    a = a[a.seed == 0]
    models = list(dict.fromkeys(a.model))
    new = [m for m in models if m not in (BASE, EUC)]
    res = dict(action={}, rule={}, clamped={}, radius={}, losses={})
    for mh in (3, 5):
        for g in ["all", "0 uses", "1-9 uses", ">= 10 uses"]:
            d = a[(a.min_heldout == mh) & (a.group == g)]
            p = d.pivot(index="fold", columns="model", values="rho_radius")
            base = d.groupby("fold").rho_baseline.first()
            r = dict(n=int(d.groupby("fold").n.first().sum()), baseline=round(float(base.mean()), 3),
                     **{m: round(float(p[m].mean()), 3) for m in models})
            for m in new:
                r[f"{m} - base"] = paired(p[m], p[BASE])
                r[f"{m} - counting"] = paired(p[m], base)
                r[f"{m} - euclidean"] = paired(p[m], p[EUC])
            res["action"][f"heldout>={mh} | {g}"] = r
    for m in new:
        x = res["action"]["heldout>=3 | all"][f"{m} - base"]
        res["rule"][m] = dict(diff=x["mean"], wins=x["wins"],
                              clearly_beats=bool(x["mean"] >= 0.05 and int(x["wins"].split("/")[0]) >= 7))
    c = pd.read_csv(TAB / "ab_clamped.csv")
    for (s, k), g in c.groupby(["setting", "kind"]):
        res["clamped"][f"{s} | {k}"] = dict(median=round(float(g.share_clamped.median()), 4), min=round(float(g.share_clamped.min()), 4),
                                            max=round(float(g.share_clamped.max()), 4), median_sqrt_c_norm=round(float(g.median_sqrt_c_norm.median()), 4))
    rr = pd.read_csv(TAB / "ab_radius.csv")
    for s, g in rr.groupby("setting"):
        res["radius"][s] = g[["min", "p10", "median", "p90", "max"]].median().round(4).to_dict()
    for s in ("A", "B"):
        rows = []
        for f in sorted(glob.glob(str(HERE / "ab_runs" / s / "*_s0_epochs.csv"))):
            e = pd.read_csv(f)
            rows.append(e[e.epoch > 90].mean())
        e = pd.DataFrame(rows)
        cols = ["L_IF", "L_SF", "L_FI", "L_ent_IF", "L_ent_sub", "total", "share_pos_ent_IF", "share_pos_ent_sub",
                "share_clamped_full", "share_clamped_sub", "c", "alpha_s", "alpha_i", "tau"]
        res["losses"][s] = e[cols].mean().round(4).to_dict()
    json.dump(res, open(HERE / "ab_summary.json", "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
