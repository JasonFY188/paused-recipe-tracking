"""
Draws docs/figures/result_summary.png from the committed aggregate numbers only (results/summary.json):
  left : at the middle of a pause, how often the paused recipe is called "paused" — local LLM vs a one-line rule
  right: the hyperbolic models against simple baselines on the same data (action radius, next-object top-1)

Usage:  python docs/make_figure.py
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "docs", "figures", "result_summary.png")
HYP, BASE = "#5b3f8c", "#9a9a9a"


def main():
    s = json.load(open(os.path.join(ROOT, "results", "summary.json"), encoding="utf-8"))
    a = s["llm_pause_pilot"]["A_middle_of_pause"]
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.4), gridspec_kw=dict(width_ratios=[1, 1.15]))

    bars = [("LLM, full history", a["llm_says_paused_full_history"], "#c0504d"),
            ("LLM, last 10 actions", a["llm_says_paused_last10"], "#c0504d"),
            ("one-line rule\n(uses true past labels)", a["rule_with_true_past_labels"], BASE)]
    l = ax[0]
    for i, (name, r, col) in enumerate(bars):
        lo, hi = r["ci95"]
        l.bar(i, 100 * r["rate"], color=col, width=0.6,
              yerr=[[100 * (r["rate"] - lo)], [100 * (hi - r["rate"])]], capsize=4)
        l.text(i, 100 * hi + 2, f"{100 * r['rate']:.1f}%", ha="center", fontweight="bold")
    l.set(xticks=range(3), xticklabels=[b[0] for b in bars], ylim=(0, 105),
          ylabel='recipe correctly called "paused" (%)',
          title=f"Middle of a pause ({a['cases']} cases): qwen3:8b vs a rule")
    l.grid(axis="y", alpha=0.3)

    r = s["action_radius_recheck"]
    t = s["object_tree"]
    groups = [("action radius vs held-out\nspecificity (Spearman ρ)",
               r["hyperbolic_setting_A_K0.02_w0.1_3seeds"], r["training_counts_baseline"], "training counts"),
              ("next new object,\nexact top-1 accuracy",
               t["exact_top1"]["hyperbolic_two_stage"], t["exact_top1"]["flat_softmax"], "flat softmax")]
    g = ax[1]
    x = np.arange(len(groups))
    for i, (name, hyp, base, base_name) in enumerate(groups):
        g.bar(i - 0.18, hyp, 0.34, color=HYP, label="hyperbolic model" if i == 0 else None)
        g.bar(i + 0.18, base, 0.34, color=BASE, label="simple baseline" if i == 0 else None)
        fmt = ".3f" if base < 0.2 else ".2f"
        g.text(i - 0.18, hyp + 0.01, f"{hyp:{fmt}}", ha="center")
        g.text(i + 0.18, base + 0.01, f"{base:{fmt}}\n({base_name})", ha="center", va="bottom", fontsize=8.5)
    g.set(xticks=x, xticklabels=[q[0] for q in groups], ylim=(0, 0.65),
          title="Hyperbolic geometry did not beat simple baselines")
    g.legend(loc="upper right", frameon=False)
    g.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, dpi=150)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
