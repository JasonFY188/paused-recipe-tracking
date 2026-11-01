"""
analysis/clean_recheck/summarize.py
Summarises tables/action_level_runs.csv and tables/window_level_runs.csv (written by clean_recheck.py).
Per fold: mean over the 3 seeds. Then over the 9 folds: mean, sd, min, max. Paired comparisons are per fold
(radius minus counts, hyperbolic minus euclidean): mean difference, t-based 95% CI over folds, wins out of 9.

Run (intention env): python analysis/clean_recheck/summarize.py
Output: summary.json, tables/summary_*.csv (report.md is written by hand from these)
"""
import json
import pathlib

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, t as tdist

OUT = pathlib.Path(__file__).resolve().parent
TAB = OUT / "tables"


def paired(a, b):
    d = np.asarray(a) - np.asarray(b)
    d = d[~np.isnan(d)]
    n = len(d)
    half = tdist.ppf(.975, n - 1) * d.std(ddof=1) / np.sqrt(n) if n > 1 else np.nan
    return dict(mean_diff=round(float(d.mean()), 3), ci95=[round(float(d.mean() - half), 3), round(float(d.mean() + half), 3)],
                wins=f"{int((d > 0).sum())}/{n}")


def across(v):
    v = np.asarray(v, float)
    v = v[~np.isnan(v)]
    return dict(mean=round(float(v.mean()), 3), sd=round(float(v.std(ddof=1)), 3), min=round(float(v.min()), 3),
                max=round(float(v.max()), 3), n_folds=len(v))


def main():
    a = pd.read_csv(TAB / "action_level_runs.csv")
    w = pd.read_csv(TAB / "window_level_runs.csv")
    res = dict(n_runs=int(a.groupby(["fold", "geometry", "seed"]).ngroups), action={}, window={})

    keys = ["test", "group", "min_heldout"]
    for (test, group, mh), g in a.groupby(keys, sort=False):
        name = f"{test} | {group} | heldout>={mh}" if mh else f"{test} | {group}"
        pf = g.groupby(["fold", "geometry"]).agg(rho_radius=("rho_radius", "mean"), seed_sd=("rho_radius", "std"),
                                                 rho_counts=("rho_counts", "first"), rho_backoff=("rho_backoff", "first"),
                                                 rho_freq=("rho_radius_vs_log_uses", "mean"), n=("n", "first")).reset_index()
        hyp, euc = (pf[pf.geometry == x].set_index("fold").sort_index() for x in ("hyperbolic", "euclidean"))
        r = dict(n_actions_total=int(hyp.n.sum()), n_actions_per_fold=[int(hyp.n.min()), int(hyp.n.max())])
        for geom, t in (("hyperbolic", hyp), ("euclidean", euc)):
            r[geom] = dict(rho_radius=across(t.rho_radius), mean_seed_sd=round(float(t.seed_sd.mean()), 3),
                           rho_radius_vs_log_uses=across(t.rho_freq) if t.rho_freq.notna().any() else None)
        base = "rho_backoff" if group == "0 uses" else "rho_counts"
        if hyp[base].notna().any():
            r["baseline"] = dict(kind="back-off (noun, else verb)" if group == "0 uses" else "training counts",
                                 rho=across(hyp[base]))
            r["radius_minus_baseline"] = {geom: paired(t.rho_radius, t[base]) for geom, t in (("hyperbolic", hyp), ("euclidean", euc))}
        r["hyperbolic_minus_euclidean"] = paired(hyp.rho_radius, euc.loc[hyp.index].rho_radius)
        r["per_fold"] = pd.DataFrame(dict(n=hyp.n, hyp=hyp.rho_radius.round(3), euc=euc.rho_radius.round(3),
                                          baseline=hyp[base].round(3))).reset_index().to_dict("records")
        res["action"][name] = r

    for k, g in w.groupby("k"):
        pf = g.groupby(["fold", "geometry"]).rho_radius_vs_window_specificity.agg(["mean", "std"]).reset_index()
        hyp, euc = (pf[pf.geometry == x].set_index("fold").sort_index() for x in ("hyperbolic", "euclidean"))
        res["window"][f"k={k}"] = dict(n_windows_total=int(g.groupby("fold").n.first().sum()),
                                       hyperbolic=across(hyp["mean"]), euclidean=across(euc["mean"]),
                                       mean_seed_sd=dict(hyperbolic=round(float(hyp["std"].mean()), 3),
                                                         euclidean=round(float(euc["std"].mean()), 3)),
                                       hyperbolic_minus_euclidean=paired(hyp["mean"], euc.loc[hyp.index, "mean"]))

    # confound check: is the held-out "truth" just the number of held-out occurrences?
    conf = []
    for f in sorted(TAB.glob("actions_P0*.csv")):
        t = pd.read_csv(f)
        t = t[t.heldout_n >= 3]
        rad = t[[c for c in t.columns if c.startswith("radius_hyperbolic")]].mean(1)
        conf.append(dict(fold=t.fold.iloc[0], rho_truth_vs_heldout_n=spearmanr(t.heldout_specificity, t.heldout_n)[0],
                         rho_radius_vs_heldout_n=spearmanr(rad, t.heldout_n)[0],
                         rho_counts_vs_heldout_n=spearmanr(t.train_count_specificity, t.heldout_n, nan_policy="omit")[0]))
    res["confound_heldout_n"] = {k: round(float(np.mean([c[k] for c in conf])), 3) for k in conf[0] if k != "fold"}

    json.dump(res, open(OUT / "summary.json", "w", encoding="utf-8"), indent=1)
    for name, r in res["action"].items():
        line = f"{name:70s} n={r['n_actions_total']:5d} hyp {r['hyperbolic']['rho_radius']['mean']:+.3f} euc {r['euclidean']['rho_radius']['mean']:+.3f}"
        if "baseline" in r:
            line += f" base {r['baseline']['rho']['mean']:+.3f} | hyp-base {r['radius_minus_baseline']['hyperbolic']}"
        print(line)
    for k, r in res["window"].items():
        print(k, "hyp", r["hyperbolic"]["mean"], "euc", r["euclidean"]["mean"], r["hyperbolic_minus_euclidean"])
    print("confound:", res["confound_heldout_n"])


if __name__ == "__main__":
    main()
