"""
analysis/object_hierarchy/summarize.py
Tables and accuracy-specificity curves for feasibility.md from results/moments_<variant>.pkl (eval_baselines.py).
All numbers are pooled over the 9 held-out people (each moment counts once).
Output: results/summary_*.csv, figures/hedging_<variant>.png, summary.json
"""
import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
RES, FIG = HERE / "results", HERE / "figures"
THETAS = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
MODELS = ["global", "kitchen", "nb", "mlp"]
TREES = {"storage": ["fixture", "area"], "storage3d": ["fixture", "area"], "semantic": ["category"]}
COLORS = {"global": "#888888", "kitchen": "#2a78d6", "nb": "#eb6834", "mlp": "#2ca02c"}


def wilson(k, n, z=1.96):
    if n == 0:
        return [np.nan, np.nan]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(c - h, 3), round(c + h, 3)]


def level_table(R):
    rows = []
    for m in MODELS:
        g = R[R.model == m]
        r = dict(model=m, n=len(g), acc1=g.acc1.mean(), acc5=g.acc5.mean())
        for t, ls in TREES.items():
            for l in ls:
                r[f"{t}_{l}_acc1"] = g[f"{t}_{l}_acc1"].mean()
        rows.append(r)
    return pd.DataFrame(rows).round(3)


def wrong_table(R):
    rows = []
    for m in MODELS:
        g = R[(R.model == m) & ~R.acc1]
        for t, ls in TREES.items():
            for l in ls:
                rows.append(dict(model=m, tree=t, level=l, n_wrong=len(g), wrong_top1_in_right_node=g[f"{t}_{l}_wrong_in"].mean(),
                                 random_wrong_uniform=g[f"{t}_{l}_rand_uniform"].mean(), random_wrong_freq=g[f"{t}_{l}_rand_freq"].mean()))
    return pd.DataFrame(rows).round(3)


def split_table(R, col, order):
    rows = []
    for m in MODELS:
        for val in order:
            g = R[(R.model == m) & (R[col] == val)]
            if not len(g):
                continue
            rows.append(dict(model=m, **{col: val}, n=len(g), acc1=g.acc1.mean(), acc1_ci=wilson(g.acc1.sum(), len(g)),
                             acc5=g.acc5.mean(), storage_fixture_acc1=g.storage_fixture_acc1.mean(),
                             storage_area_acc1=g.storage_area_acc1.mean(), semantic_category_acc1=g.semantic_category_acc1.mean(),
                             wrong_in_fixture=g[~g.acc1].storage_fixture_wrong_in.mean(),
                             wrong_in_category=g[~g.acc1].semantic_category_wrong_in.mean()))
    return pd.DataFrame(rows).round(3)


def hedge_curves(R, name):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharey=True)
    curves = []
    for ax, t in zip(axes, TREES):
        for m in MODELS:
            g = R[R.model == m]
            xs = [1.0] + [g[f"{t}_hedge{th}_spec"].mean() for th in THETAS]
            ys = [g.acc1.mean()] + [g[f"{t}_hedge{th}_ok"].mean() for th in THETAS]
            ax.plot(xs, ys, "o-", color=COLORS[m], label=m, ms=4)
            for th, x, y in zip([None] + THETAS, xs, ys):
                curves.append(dict(tree=t, model=m, theta=th if th else "flat top-1", specificity=round(x, 3), accuracy=round(y, 3)))
                if m == "mlp" and th in (0.1, 0.3, 0.5, 0.9):
                    ax.annotate(f"θ={th}", (x, y), xytext=(4, 4), textcoords="offset points", fontsize=7, color=COLORS[m])
        ax.set(title=f"{t} tree", xlabel="mean specificity of the answer (1 = exact class, 0 = root)", xlim=(-0.02, 1.02), ylim=(0, 1.02))
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("accuracy (answer node contains the true object)")
    axes[0].legend(loc="lower left")
    fig.suptitle(f"Hedged next-new-object prediction ({name}): each point = one threshold θ; rightmost = flat top-1", x=0.01, ha="left")
    fig.tight_layout(); fig.savefig(FIG / f"hedging_{name}.png", dpi=130); plt.close(fig)
    return pd.DataFrame(curves)


def main():
    FIG.mkdir(exist_ok=True)
    out = {}
    for f in sorted(RES.glob("moments_*.pkl")):
        name = f.stem.replace("moments_", "")
        R = pd.read_pickle(f)
        lt, wt = level_table(R), wrong_table(R)
        lt.to_csv(RES / f"summary_levels_{name}.csv", index=False)
        wt.to_csv(RES / f"summary_wrong_{name}.csv", index=False)
        cv = hedge_curves(R, name); cv.to_csv(RES / f"summary_hedging_{name}.csv", index=False)
        R["freq_bucket"] = pd.cut(R.freq, [-1, 0, 9, 10 ** 9], labels=["0", "1-9", ">=10"]).astype(str)
        sp = {c: split_table(R, c, o) for c, o in (("freq_bucket", ["0", "1-9", ">=10"]), ("resume", [False, True]),
                                                    ("picked_from_storage", [False, True]),
                                                    ("home_kind", ["storage", "always out", "unknown"]))}
        for c, t in sp.items():
            t.to_csv(RES / f"summary_split_{c}_{name}.csv", index=False)
        out[name] = dict(levels=lt.to_dict("records"), n=int((R.model == "global").sum()))
        print(f"===== {name}  (n moments = {out[name]['n']})")
        print(lt.to_string(index=False))
        if name == "main":
            print(wt.to_string(index=False))
            for c, t in sp.items():
                print(t.to_string(index=False))
            print(cv[cv.model.isin(["mlp", "kitchen"])].to_string(index=False))
    json.dump(out, open(HERE / "summary.json", "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
