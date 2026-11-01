"""
analysis/clean_recheck/settings_ab.py
Clean re-check with cone settings that should keep the cones active:
  A: K = 0.02, entailment weight 0.1  (the setting of the old rho = 0.59 run)
  B: K = 0.02, entailment weight 0.3
  base: K = 0.05, entailment weight 1.0 (the first clean re-check; its seed-0 models = diag_runs/)
Same data, LOPO, model and training loop as clean_recheck.py / diagnostics.py; only K and the weight change.
The Euclidean model has no cones, so K and the weight do not affect it: its existing runs (runs/*_euclidean_s*.npz)
are the Euclidean comparison.

Per run it saves the weights, learned c / alpha_s / alpha_i / tau, per-epoch loss terms and cone checks, the
single-action and window radii, and the share of clamped cones for single actions, subsets, full sets and intentions.

Rule fixed BEFORE seeing results ("clearly beats"): rho(radius, held-out truth), all actions, >= 3 held-out
occurrences, is at least 0.05 higher than the base setting's seed 0 AND higher in >= 7 of the 9 people.
Only then are seeds 1 and 2 run (--seeds 0 1 2).

Run (intention env): KMP_DUPLICATE_LIB_OK=TRUE python analysis/clean_recheck/settings_ab.py
        [--settings A B] [--seeds 0]
Outputs: analysis/clean_recheck/ab_runs/<setting>/<fold>_s<seed>.*, tables/ab_*.csv, ab_summary.json
"""
import argparse
import json
import math
import os
import pathlib
import random
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np
import pandas as pd
import torch

import clean_recheck as cr
import diagnostics as dg
from hyperbolic_intention import IntentionModel, pad, sample_pair

SETTINGS = {"A": dict(K=0.02, ent_weight=0.1), "B": dict(K=0.02, ent_weight=0.3), "base": dict(K=0.05, ent_weight=1.0)}
OUT = HERE / "ab_runs"


def train_logged(model, train, device, epochs, ent_weight):
    """diagnostics.train_logged with a chosen entailment weight (same steps and random draws)."""
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=cr.LR, weight_decay=0.01)
    train = list(train)
    rows = []
    for ep in range(1, epochs + 1):
        model.train()
        random.shuffle(train)
        parts = []
        for i in range(0, len(train), cr.BATCH):
            batch = train[i:i + cr.BATCH]
            pairs = [sample_pair(s["ids"], "harmonic", s.get("seq")) for s in batch]
            full, sub = pad([p[0] for p in pairs], device), pad([p[1] for p in pairs], device)
            y = torch.tensor([s["y"] for s in batch], device=device)
            loss, info = dg.diag_loss(model, *full, *sub, y, ent_weight)
            opt.zero_grad(); loss.backward(); opt.step()
            parts.append(info)
        p = pd.DataFrame(parts)
        row = dict(epoch=ep, **p[["L_IF", "L_SF", "L_FI", "L_ent_IF", "L_ent_sub", "total"]].mean().to_dict())
        n = p.n_pairs.sum()
        row.update(share_pos_ent_IF=p.pos_ent_IF.sum() / n, share_pos_ent_sub=p.pos_ent_sub.sum() / n,
                   share_clamped_full=p.clamped_full.sum() / n, share_clamped_sub=p.clamped_sub.sum() / n,
                   **dg.scalars(model))
        rows.append(row)
    model.eval()
    return pd.DataFrame(rows)


def run(setting, fold, seed, segs, vocab, feats, device):
    d = OUT / setting
    d.mkdir(parents=True, exist_ok=True)
    stem = d / f"{fold}_s{seed}"
    if (d / f"{fold}_s{seed}.npz").exists():
        return dict(np.load(d / f"{fold}_s{seed}.npz"))
    cfg = SETTINGS[setting]
    train = [dict(s) for s in segs if s["participant"] != fold]
    test = [s for s in segs if s["participant"] == fold]
    labels = sorted({s["label"] for s in train})
    for s in train:
        s["y"] = labels.index(s["label"])
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    model = IntentionModel(len(vocab), len(labels), cr.DIM, feats, cfg["K"], "hyperbolic").to(device)
    ep = train_logged(model, train, device, 100, cfg["ent_weight"])
    ep.to_csv(f"{stem}_epochs.csv", index=False)
    torch.save(model.state_dict(), f"{stem}_model.pt")
    sc = dg.scalars(model)
    json.dump(dict(setting=setting, fold=fold, seed=seed, labels=labels, ent_weight=cfg["ent_weight"], **sc),
              open(f"{stem}_params.json", "w"), indent=1)
    # clamped-cone shares for single actions, subsets (one harmonic subset per training block, own RNG), full sets, intentions
    rng = random.Random(seed)
    fulls = [s["ids"] for s in train]
    subs = []
    for ids in fulls:
        if len(ids) == 1:
            subs.append(list(ids)); continue
        sizes = range(1, len(ids))
        k = rng.choices(sizes, weights=[1 / k for k in sizes])[0]
        subs.append(rng.sample(ids, k))
    res = {}
    with torch.no_grad():
        pts = dict(single_actions=dg.points(model, [[i] for i in range(len(vocab))], device), subsets=dg.points(model, subs, device),
                   full_sets=dg.points(model, fulls, device), intention_points=model.embed_intents())
        for name, x in pts.items():
            s_ = (torch.sqrt(model.c()) * x.norm(dim=-1)).cpu().numpy()
            res[f"{name}_sqrt_c_norm"], res[f"{name}_radius"] = s_, model.dist0(x).cpu().numpy()
    res["action_radius"] = res["single_actions_radius"]
    for k, lists in cr.windows(test, cr.WINDOW_KS).items():
        res[f"window{k}_radius"] = cr.radii(model, lists, device)
    res["K"] = np.array(cfg["K"])
    np.savez_compressed(d / f"{fold}_s{seed}.npz", **res)
    last = ep[ep.epoch > 90].mean()
    print(f"  {setting} {fold} s{seed}: c {sc['c']:.3f} | clamped single {np.mean(res['single_actions_sqrt_c_norm'] <= 2 * cfg['K']):.3f} "
          f"| L_ent>0 {last.share_pos_ent_IF:.3f}/{last.share_pos_ent_sub:.3f}", flush=True)
    del model
    torch.cuda.empty_cache()
    return res


def fold_stats(fold, occ, segs, vocab, noun_of, verb_of, n_nouns, n_verbs):
    """the same truth / counts / back-off as clean_recheck.main"""
    tr, te = occ[occ.participant != fold], occ[occ.participant == fold]
    tr_labels, te_labels = sorted(tr.label.unique()), sorted(te.label.unique())
    uses_tr = np.bincount(tr.drop_duplicates("narr").aid, minlength=len(vocab))
    n_te = np.bincount(te.drop_duplicates("narr").aid, minlength=len(vocab))
    truth = cr.specificity(cr.label_counts(te, "aid", len(vocab), te_labels))
    spec_tr = cr.specificity(cr.label_counts(tr, "aid", len(vocab), tr_labels))
    spec_noun = cr.specificity(cr.label_counts(tr, "nidx", n_nouns, tr_labels))[noun_of]
    spec_verb = cr.specificity(cr.label_counts(tr, "vid", n_verbs, tr_labels))[verb_of]
    backoff = np.where(~np.isnan(spec_noun), spec_noun, spec_verb)
    return uses_tr, n_te, truth, spec_tr, backoff


def table_rows(tag, fold, seed, r, st):
    uses_tr, n_te, truth, spec_tr, backoff = st
    rows = []
    for mh in cr.MIN_HELDOUT:
        ok = n_te >= mh
        for gname, lo, hi in [("all", 0, 10 ** 9)] + cr.GROUPS:
            m = ok & (uses_tr >= lo) & (uses_tr <= hi)
            base = backoff if gname == "0 uses" else spec_tr
            mb = m & (uses_tr > 0) if gname == "all" else m
            rows.append(dict(model=tag, fold=fold, seed=seed, min_heldout=mh, group=gname, n=int(m.sum()),
                             rho_radius=cr.rho(r[m], truth[m]), rho_baseline=cr.rho(base[mb], truth[mb])))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--settings", nargs="+", default=["A", "B"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--suffix", default="", help="added to the output table names, e.g. _A3 for the 3-seed run of A")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    narr, blocks, segs, occ, vocab, info = cr.load_data()
    feats = cr.qwen3_features(vocab, device)
    verbs, nouns = sorted(narr.verb.unique()), sorted(narr.noun.unique())
    first = narr.drop_duplicates("aid").set_index("aid").reindex(range(len(vocab)))
    verb_of = first.verb.map({v: i for i, v in enumerate(verbs)}).values.astype(int)
    noun_of = first.noun.map({n: i for i, n in enumerate(nouns)}).values.astype(int)
    occ["vid"], occ["nidx"] = verb_of[occ.aid], noun_of[occ.aid]

    rows, clamp_rows, radius_rows = [], [], []
    for fold in cr.FOLDS:
        st = fold_stats(fold, occ, segs, vocab, noun_of, verb_of, len(nouns), len(verbs))
        for seed in args.seeds:
            # base setting (first clean re-check) and Euclidean, from saved runs
            rows += table_rows("base K0.05 w1.0", fold, seed, np.load(cr.RUNS / f"{fold}_hyperbolic_s{seed}.npz")["action_radius"], st)
            rows += table_rows("euclidean", fold, seed, np.load(cr.RUNS / f"{fold}_euclidean_s{seed}.npz")["action_radius"], st)
            for setting in args.settings:
                res = run(setting, fold, seed, segs, vocab, feats, device)
                K = float(res["K"])
                rows += table_rows(f"{setting} K{SETTINGS[setting]['K']} w{SETTINGS[setting]['ent_weight']}", fold, seed, res["action_radius"], st)
                for name in ("single_actions", "subsets", "full_sets", "intention_points"):
                    s_ = res[f"{name}_sqrt_c_norm"]
                    clamp_rows.append(dict(setting=setting, fold=fold, seed=seed, kind=name, n=len(s_),
                                           share_clamped=float((s_ <= 2 * K).mean()), median_sqrt_c_norm=float(np.median(s_))))
                r = res["action_radius"]
                radius_rows.append(dict(setting=setting, fold=fold, seed=seed, **dict(zip(["min", "p10", "median", "p90", "max"],
                                                                                         np.percentile(r, [0, 10, 50, 90, 100])))))
        cr.TAB.mkdir(exist_ok=True)
        pd.DataFrame(rows).to_csv(cr.TAB / f"ab_action_level{args.suffix}.csv", index=False)
        pd.DataFrame(clamp_rows).to_csv(cr.TAB / f"ab_clamped{args.suffix}.csv", index=False)
        pd.DataFrame(radius_rows).to_csv(cr.TAB / f"ab_radius{args.suffix}.csv", index=False)
        print(f"== {fold} done", flush=True)


if __name__ == "__main__":
    main()
