"""
analysis/clean_recheck/summarize_A3.py
Setting A (K = 0.02, entailment weight 0.1) with 3 seeds vs the base clean setting (K = 0.05, weight 1.0) and
Euclidean, both also 3 seeds, and the counting baselines. Per person: mean over seeds; then over the 9 people.
Output: ab_A3_summary.json
"""
import json
import pathlib

import numpy as np
import pandas as pd
from scipy.stats import t as tdist

HERE = pathlib.Path(__file__).resolve().parent
TAB = HERE / "tables"
A, BASE, EUC = "A K0.02 w0.1", "base K0.05 w1.0", "euclidean"


def paired(a, b):
    d = (a - b).dropna()
    h = tdist.ppf(.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
    return dict(mean=round(float(d.mean()), 3), ci95=[round(float(d.mean() - h), 3), round(float(d.mean() + h), 3)],
                wins=f"{int((d > 0).sum())}/{len(d)}")


def main():
    a = pd.read_csv(TAB / "ab_action_level_A3.csv")
    res = {}
    for mh in (3, 5):
        for g in ["all", "0 uses", "1-9 uses", ">= 10 uses"]:
            d = a[(a.min_heldout == mh) & (a.group == g)]
            per = d.groupby(["fold", "model"]).rho_radius.mean().unstack()
            seed_sd = d.groupby(["fold", "model"]).rho_radius.std().unstack().mean()
            base = d.groupby("fold").rho_baseline.first()
            res[f"heldout>={mh} | {g}"] = dict(
                n=int(d.groupby("fold").n.first().sum()), counting=round(float(base.mean()), 3),
                **{m: round(float(per[m].mean()), 3) for m in (A, BASE, EUC)},
                seed_sd={m: round(float(seed_sd[m]), 3) for m in (A, BASE, EUC)},
                A_minus_base=paired(per[A], per[BASE]), A_minus_counting=paired(per[A], base),
                A_minus_euclidean=paired(per[A], per[EUC]), base_minus_counting=paired(per[BASE], base))
    r = pd.read_csv(TAB / "ab_radius_A3.csv")
    res["A_radius_by_seed"] = r.groupby("seed")[["min", "p10", "median", "p90", "max"]].median().round(3).to_dict("index")
    c = pd.read_csv(TAB / "ab_clamped_A3.csv")
    res["A_clamped"] = c.groupby("kind").share_clamped.agg(["median", "max"]).round(4).to_dict("index")
    json.dump(res, open(HERE / "ab_A3_summary.json", "w"), indent=1)
    for k, v in res.items():
        if k.startswith("heldout"):
            print(f"{k:25s} n={v['n']:5d} counting {v['counting']:.3f} A {v[A]:.3f} base {v[BASE]:.3f} euc {v[EUC]:.3f} | "
                  f"A-base {v['A_minus_base']} | A-count {v['A_minus_counting']} | A-euc {v['A_minus_euclidean']} | sd {v['seed_sd']}")
    print(res["A_radius_by_seed"]); print(res["A_clamped"])


if __name__ == "__main__":
    main()
