"""
scripts/evaluate_intention.py
Paper-style evaluation (Kubota et al., arXiv:2604.04158, Sec. 5.4 / Table 1) of the intention models, leave-one-participant-out
(each of P01..P09 is the test participant once; the model never sees that person's kitchen or recipes).

Task 1 - intention retrieval (analogue of impression-based font retrieval):
  a query is a set of observed actions; the 12 intentions are ranked by distance to the query.
    mAP-single  every action seen in the test participant's data is a query; relevant = every intention of a
                test segment containing that action (paper: mAP-single, fonts containing the tag)
    mAP-multi   random 2-5-action subsets of test segments; relevant = intentions of test segments containing
                all of them (paper: mAP-multi)
    acc@first-k first k actions of a test segment, in the order they happened -> top-1 accuracy (online use)
    acc@full    the segment's whole action set -> top-1 accuracy
Task 2 - next actions (the end goal: predict what the person does next):
  query = first k actions of a test segment; every other action in the vocabulary is ranked.
    recall@10   share of the segment's remaining actions that are in the top 10
    AP          average precision of the remaining actions over the whole ranking
    next@10     is the very next new action in the top 10
  Model score for a candidate action a: how well {query + a} matches an intention the query already points
  to: log sum_k p(k | query) exp(-d(query + a, intention_k) / tau). "<model>+pop" adds log action popularity
  (the model re-ranks a frequency prior).

Methods (same frozen CLIP action features and set encoder everywhere; only the method changes):
  euclidean       contrastive only, Euclidean vectors (Impression-CLIP+-style baseline)
  hyp_no_cones    hyperbolic, contrastive only (entailment weight 0)
  paper_default   hyperbolic + entailment, paper / MERU settings (K 0.1, weight 0.1, uniform subsets)
  paper_tuned     hyperbolic + entailment (K 0.05, weight 1.0, small subsets more often)
  tuned_coarse    paper_tuned + coarse-to-fine entailment (plain action entails item-detailed action)
Count baselines: prior (intention frequency / action popularity), naive Bayes on present actions,
co-occurrence (next actions only).

Run in the `intention` env (≈1 h for all 9 folds; finished folds are cached and skipped on rerun):
  python scripts/evaluate_intention.py [--folds P01 P02] [--epochs 100]
Outputs: scripts/intention_eval/<name>/ fold_PXX.json, summary.csv, summary.png
"""
import argparse
import json
import os
import pathlib
import random
import subprocess
import sys
import time

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from hyperbolic_intention import ROOT, IntentionModel, clip_features, load_data, pad, train_model

FOLDS = [f"P0{i}" for i in range(1, 10)]
VARIANTS = {
    "euclidean": dict(geometry="euclidean", K=0.05, ent_weight=0.0, subset="harmonic", coarse=False),
    "hyp_no_cones": dict(geometry="hyperbolic", K=0.05, ent_weight=0.0, subset="harmonic", coarse=False),
    "paper_default": dict(geometry="hyperbolic", K=0.1, ent_weight=0.1, subset="uniform", coarse=False),
    "paper_tuned": dict(geometry="hyperbolic", K=0.05, ent_weight=1.0, subset="harmonic", coarse=False),
    "tuned_coarse": dict(geometry="hyperbolic", K=0.05, ent_weight=1.0, subset="harmonic", coarse=True),
}
PREFIX_KS = [1, 3, 5]
N_MULTI = 1000
TOP = 10


# ---------------------------------------------------------------- metrics
def average_precision(scores, relevant):
    """scores [n, m], relevant bool [n, m] -> AP per row (nan if a row has no relevant item)."""
    order = np.argsort(-scores, axis=1)
    rel = np.take_along_axis(relevant, order, axis=1)
    hits = np.cumsum(rel, axis=1)
    prec = hits / np.arange(1, rel.shape[1] + 1)
    n_rel = rel.sum(1)
    return np.where(n_rel > 0, (prec * rel).sum(1) / np.maximum(n_rel, 1), np.nan)


def top_hits(scores, relevant, k):
    idx = np.argsort(-scores, axis=1)[:, :k]
    return np.take_along_axis(relevant, idx, axis=1)


# ---------------------------------------------------------------- queries
def intent_queries(test, n_cls, rng):
    q = {}
    by_action = {}
    for s in test:
        for a in s["ids"]:
            by_action.setdefault(a, set()).add(s["y"])
    q["single"] = [([a], ys) for a, ys in by_action.items()]
    multi = []
    cands = [s for s in test if len(s["ids"]) >= 2]
    for _ in range(N_MULTI if cands else 0):
        s = cands[rng.integers(len(cands))]
        k = int(rng.integers(2, min(5, len(s["ids"])) + 1))
        sub = sorted(rng.choice(s["ids"], k, replace=False).tolist())
        multi.append((sub, {t["y"] for t in test if set(sub) <= set(t["ids"])}))
    q["multi"] = multi
    for k in PREFIX_KS:
        q[f"first{k}"] = [(s["order"][:k], {s["y"]}) for s in test if len(s["order"]) >= k]
    q["full"] = [(s["ids"], {s["y"]}) for s in test]
    return {name: (lists, np.array([[k in ys for k in range(n_cls)] for _, ys in lists_ys]))
            for name, lists_ys in q.items() if lists_ys for lists in [[l for l, _ in lists_ys]]}


def next_queries(test, k):
    out = []
    for s in test:
        if len(s["order"]) > k:
            out.append((s["order"][:k], set(s["order"][k:]), s["order"][k]))
    return out


# ---------------------------------------------------------------- scorers
class Counts:
    def __init__(self, train, n_vocab, n_cls, alpha=1.0):
        X = np.zeros((len(train), n_vocab))
        for i, s in enumerate(train):
            X[i, s["ids"]] = 1
        y = np.array([s["y"] for s in train])
        self.log_prior = np.log(np.bincount(y, minlength=n_cls) + alpha)
        self.log_prior -= np.log(np.exp(self.log_prior).sum())
        self.p_a_k = np.stack([(X[y == k].sum(0) + alpha) / ((y == k).sum() + 2 * alpha) for k in range(n_cls)])
        self.pop = np.log(X.sum(0) + 1)
        co = X.T @ X
        self.p_cooc = (co + 0.1) / (np.diag(co)[:, None] + 1)   # P(a | q) from segment co-occurrence

    def nb_intent(self, lists):
        """naive Bayes over the actions present in the query only (a partial observation)."""
        return np.stack([self.log_prior + np.log(self.p_a_k[:, l]).sum(1) for l in lists])


@torch.no_grad()
def embed(model, lists, device, bs=8192):
    out = []
    for i in range(0, len(lists), bs):
        out.append(model.embed_sets(*pad(lists[i:i + bs], device)))
    return torch.cat(out)


@torch.no_grad()
def model_intent_scores(model, lists, device):
    return (-model.pdist(embed(model, lists, device), model.embed_intents())).cpu().numpy()


@torch.no_grad()
def model_next_scores(model, queries, n_vocab, device):
    f, tau = model.embed_intents(), model.tau()
    out = np.zeros((len(queries), n_vocab), dtype=np.float32)
    for qi, (q, _, _) in enumerate(queries):
        log_belief = torch.log_softmax(-model.pdist(embed(model, [q], device), f) / tau, dim=1)   # [1, K]
        d = model.pdist(embed(model, [q + [a] for a in range(n_vocab)], device), f)                # [V, K]
        out[qi] = torch.logsumexp(log_belief - d / tau, dim=1).cpu().numpy()
    return out


def next_metrics(scores, queries):
    scores = scores.copy()
    rel = np.zeros_like(scores, dtype=bool)
    nxt = np.zeros_like(rel)
    for i, (q, future, n) in enumerate(queries):
        scores[i, q] = -np.inf
        rel[i, list(future - set(q))] = True
        nxt[i, n] = True
    top = top_hits(scores, rel, TOP)
    n_rel = rel.sum(1)
    ok = n_rel > 0
    return dict(recall10=float((top.sum(1)[ok] / n_rel[ok]).mean()), AP=float(np.nanmean(average_precision(scores, rel))),
                next10=float(top_hits(scores, nxt, TOP).any(1).mean()), n=int(len(queries)))


def intent_metrics(scores, relevant):
    top1 = top_hits(scores, relevant, 1)[:, 0]
    return dict(mAP=float(np.nanmean(average_precision(scores, relevant))), acc=float(top1.mean()), n=int(len(scores)))


# ---------------------------------------------------------------- one fold
def run_fold(fold, args, device, out_dir):
    path = out_dir / f"fold_{fold}.json"
    if path.exists():
        return json.load(open(path))
    t0 = time.time()
    data_name = f"intention_folds/{fold}"
    subprocess.run([sys.executable, str(ROOT / "scripts" / "build_intention_sets.py"), "--detail", "--min-count", "5",
                    "--test", fold, "--val", "none", "--out", data_name], check=True, capture_output=True)
    data = ROOT / "scripts" / data_name
    segs, vocab, intents = load_data(data)
    for s in segs:
        s["y"] = intents.index(s["intent"])
    train = [s for s in segs if s["split"] == "train"]
    test = [s for s in segs if s["split"] == "test"]
    plain_of = segs[0]["plain_of"]
    detail_pairs = [(plain_of[i], i) for i in range(len(vocab)) if plain_of[i] != i]
    feats = clip_features(vocab, device, data)
    rng = np.random.default_rng(0)
    iq = intent_queries(test, len(intents), rng)
    nq = {k: next_queries(test, k) for k in PREFIX_KS}
    counts = Counts(train, len(vocab), len(intents))
    res = dict(fold=fold, n_train=len(train), n_test=len(test), vocab=len(vocab), intents=intents, methods={})

    # count baselines
    res["methods"]["prior"] = dict(
        intent={n: intent_metrics(np.tile(counts.log_prior, (len(l), 1)), r) for n, (l, r) in iq.items()},
        next={k: next_metrics(np.tile(counts.pop, (len(q), 1)), q) for k, q in nq.items() if q})
    res["methods"]["naive_bayes"] = dict(
        intent={n: intent_metrics(counts.nb_intent(l), r) for n, (l, r) in iq.items()},
        next={k: next_metrics(np.stack([np.log(np.exp(counts.nb_intent([qq])[0] - counts.nb_intent([qq])[0].max())
                                               @ counts.p_a_k + 1e-12) for qq, _, _ in q]), q)
              for k, q in nq.items() if q})
    res["methods"]["cooccurrence"] = dict(
        next={k: next_metrics(np.stack([np.log(counts.p_cooc[qq].mean(0)) for qq, _, _ in q]), q)
              for k, q in nq.items() if q})

    for name, cfg in VARIANTS.items():
        random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
        model = IntentionModel(len(vocab), len(intents), args.dim, feats, cfg["K"], cfg["geometry"]).to(device)
        train_model(model, train, device, args.epochs, subset=cfg["subset"], ent_weight=cfg["ent_weight"],
                    coarse=cfg["coarse"], plain_of=plain_of, detail_pairs=detail_pairs)
        next_scores = {k: model_next_scores(model, q, len(vocab), device) for k, q in nq.items() if q}
        res["methods"][name] = dict(
            intent={n: intent_metrics(model_intent_scores(model, l, device), r) for n, (l, r) in iq.items()},
            next={k: next_metrics(sc, nq[k]) for k, sc in next_scores.items()})
        # the model as a re-ranker on top of action popularity: p(a | query) ~ p(a) * fit(query + a)
        res["methods"][name + "+pop"] = dict(
            next={k: next_metrics(sc + counts.pop[None], nq[k]) for k, sc in next_scores.items()})
        del model
        torch.cuda.empty_cache()
        print(f"  [{fold}] {name} done ({time.time() - t0:.0f}s)", flush=True)
    json.dump(res, open(path, "w"), indent=1)
    return res


# ---------------------------------------------------------------- summary
def summarize(results, out_dir):
    rows = []
    for r in results:
        for m, v in r["methods"].items():
            for n, x in v.get("intent", {}).items():
                for metric in ("mAP", "acc"):
                    rows.append(dict(fold=r["fold"], method=m, task="intent", query=n, metric=metric, value=x[metric], n=x["n"]))
            for k, x in v.get("next", {}).items():
                for metric in ("recall10", "AP", "next10"):
                    rows.append(dict(fold=r["fold"], method=m, task="next", query=f"first{k}", metric=metric, value=x[metric], n=x["n"]))
    df = pd.DataFrame(rows)
    # pooled over folds, weighted by the number of queries
    pooled = pd.DataFrame([dict(task=t, query=q, metric=m, method=me, mean=np.average(g.value, weights=g.n),
                                fold_std=g.value.std(), n=g.n.sum())
                           for (t, q, m, me), g in df.groupby(["task", "query", "metric", "method"])])
    pooled.to_csv(out_dir / "summary_long.csv", index=False)
    cols = [("intent", "single", "mAP"), ("intent", "multi", "mAP"), ("intent", "first1", "acc"), ("intent", "first3", "acc"),
            ("intent", "first5", "acc"), ("intent", "full", "acc"), ("next", "first1", "recall10"),
            ("next", "first3", "recall10"), ("next", "first3", "AP"), ("next", "first3", "next10"), ("next", "first5", "recall10")]
    table = pd.DataFrame({f"{t}:{q}:{m}": pooled[(pooled.task == t) & (pooled["query"] == q) & (pooled.metric == m)]
                          .set_index("method")["mean"] for t, q, m in cols})
    order = ["prior", "naive_bayes", "cooccurrence"] + list(VARIANTS) + [v + "+pop" for v in VARIANTS]
    table = table.reindex([m for m in order if m in table.index])
    table.to_csv(out_dir / "summary.csv")

    show = [("intent:single:mAP", "Intention mAP\nsingle-action queries"), ("intent:multi:mAP", "Intention mAP\n2–5-action queries"),
            ("intent:first3:acc", "Intention accuracy\nafter first 3 actions"), ("next:first3:recall10", "Next actions recall@10\nafter first 3 actions"),
            ("next:first3:next10", "Very next action in top 10\nafter first 3 actions")]
    fig, axes = plt.subplots(1, len(show), figsize=(3.3 * len(show), 3.8), sharey=True)
    for ax, (col, title) in zip(axes, show):
        v = table[col]
        colors = ["#eb6834" if m.startswith("tuned_coarse") else "#2a78d6" if m.split("+")[0] in VARIANTS else "#b9b8b2"
                  for m in v.index]
        ax.barh(range(len(v)), v.fillna(0).values, color=colors, height=0.7)
        for i, x in enumerate(v.values):
            if not np.isnan(x):
                ax.text(x, i, f" {x:.2f}", va="center", fontsize=8, color="#0b0b0b")
        ax.set_title(title, fontsize=9, loc="left", color="#0b0b0b")
        ax.set_yticks(range(len(v))); ax.set_yticklabels(v.index, fontsize=8)
        ax.invert_yaxis(); ax.set_xlim(0, max(1e-3, np.nanmax(v.values)) * 1.3)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    fig.suptitle(f"Leave-one-participant-out ({len(results)} folds): grey = count baselines, blue = embedding models, "
                 "orange = with coarse-to-fine", x=0.01, ha="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_dir / "summary.png", dpi=140, bbox_inches="tight")
    return table


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", nargs="+", default=FOLDS)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--name", default="lopo_detail")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = ROOT / "scripts" / "intention_eval" / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for fold in args.folds:
        print(f"fold {fold}", flush=True)
        results.append(run_fold(fold, args, device, out_dir))
    table = summarize(results, out_dir)
    pd.set_option("display.width", 250)
    print(table.round(3).to_string())
    print("saved", out_dir)


if __name__ == "__main__":
    main()
