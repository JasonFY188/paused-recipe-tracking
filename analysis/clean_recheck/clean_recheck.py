"""
analysis/clean_recheck/clean_recheck.py
Clean re-check of the action-level specificity result (RESEARCH_CONTEXT.md, priority 1), with HD-EPIC labels only.

  input   main action of each narration = "<verb class key> <noun class key>" exactly as in the HD-EPIC class files
          (e.g. "take maker:coffee"); no item details, no rewriting, no minimum count (all 4,373 action types)
  labels  activity-block recipe id ("P01_R03"), or "non-recipe" for blocks without a recipe id
  split   leave-one-participant-out (9 folds) x seeds 0, 1, 2 x geometry (hyperbolic, euclidean)
  model   scripts/hyperbolic_intention.py (imported, not modified): frozen Qwen3-Embedding features -> set Transformer,
          paper losses, harmonic subsets, K = 0.05, entailment weight 1.0, 100 epochs, intention points random-initialised

Part 1 (action level): radius of each single action vs its specificity in the HELD-OUT person
        (1 - normalised entropy of p(label | action) over that person's occurrences), by training-use group
        (0 / 1-9 / >= 10); baselines: the same specificity from the TRAINING people's counts, and for unseen actions a
        back-off (noun of the pair, else its verb).
Part 2b (window level): radius of a window of k consecutive held-out actions vs the window's count-based specificity
        (1 - normalised entropy of the naive-Bayes posterior from training counts). 2a (AUROC) is skipped (user decision).

Run (intention env): KMP_DUPLICATE_LIB_OK=TRUE python analysis/clean_recheck/clean_recheck.py
        [--folds P01 ...] [--seeds 0 1 2] [--epochs 100]
Outputs (analysis/clean_recheck/): runs/<fold>_<geometry>_s<seed>.npz (resumable), tables/*.csv, results.json
"""
import argparse
import glob
import json
import math
import os
import pathlib
import random
import sys
import time

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr

from evaluate_intention import Counts
from hyperbolic_intention import QWEN3_TASK, IntentionModel, pad, train_model

ANN = ROOT / "annotations"
OUT = ROOT / "analysis" / "clean_recheck"
RUNS, TAB = OUT / "runs", OUT / "tables"
FOLDS = [f"P0{i}" for i in range(1, 10)]
VIDEO_ID_FIXES = {"P09-2024623-154547": "P09-20240623-154547"}     # typo in P09_recipe_timestamps.csv
NON_RECIPE = "non-recipe"
DIM, K_CONE, ENT_WEIGHT, LR, BATCH = 128, 0.05, 1.0, 1e-4, 32      # settings of the main model
WINDOW_KS = [1, 2, 3, 5, 8]                                         # window sizes of the old window check
MIN_HELDOUT = (3, 5)                                                # main setting, robustness check
GROUPS = [("0 uses", 0, 0), ("1-9 uses", 1, 9), (">= 10 uses", 10, 10 ** 9)]


# ------------------------------------------------------------------ data (HD-EPIC only)
def load_data():
    d = pd.read_pickle(ANN / "narrations-and-action-segments" / "HD_EPIC_Narrations.pkl")
    vc = pd.read_csv(ANN / "narrations-and-action-segments" / "HD_EPIC_verb_classes.csv")
    nc = pd.read_csv(ANN / "narrations-and-action-segments" / "HD_EPIC_noun_classes.csv")
    V, N = dict(zip(vc.id, vc.key)), dict(zip(nc.id, nc.key))
    ok = d.main_action_classes.apply(lambda m: len(m) == 1 and len(m[0]) == 2)
    n_no_main = int((~ok).sum())
    d = d[ok]
    narr = pd.DataFrame(dict(nid=d.unique_narration_id.values, video_id=d.video_id.values, participant=d.participant_id.values,
                             start=d.start_timestamp.astype(float).values,
                             verb=[V[m[0][0]] for m in d.main_action_classes], noun=[N[m[0][1]] for m in d.main_action_classes]))
    narr["action"] = narr.verb + " " + narr.noun
    narr = narr.sort_values(["video_id", "start"]).reset_index(drop=True)

    dur = pd.read_csv(ANN / "youtube-links" / "HD_EPIC_YouTube_URLs.csv").set_index("video_id").duration
    b = pd.concat([pd.read_csv(f, dtype=str) for f in sorted(glob.glob(str(ANN / "high-level" / "activities" / "*.csv")))],
                  ignore_index=True)
    b["video_id"] = b.video_id.replace(VIDEO_ID_FIXES)
    b["s"] = b.start_time.astype(float)
    b["e"] = pd.to_numeric(b.end_time, errors="coerce")
    b.loc[b.e.isna(), "e"] = b.video_id.map(dur)                       # "end" = end of the video
    rid = b.recipe_id.fillna("").str.strip()
    multi = rid.str.contains(",")
    dropped_multi = b[multi][["video_id", "recipe_id", "high_level_activity_label"]].to_dict("records")
    b, rid = b[~multi].copy(), rid[~multi]
    b["participant"] = b.video_id.str[:3]
    b["label"] = np.where(rid != "", b.participant + "_" + rid, NON_RECIPE)
    b = b.reset_index(drop=True)

    vocab = sorted(narr.action.unique())
    aidx = {a: i for i, a in enumerate(vocab)}
    narr["aid"] = narr.action.map(aidx)
    by_v = {v: g for v, g in narr.groupby("video_id")}
    segs, occ = [], []                                                  # occ = (block, narration) pairs
    for bi, r in b.iterrows():
        g = by_v.get(r.video_id)
        if g is None:
            continue
        g = g[(g.start >= r.s) & (g.start < r.e)]                       # narration start in [s, e)
        if not len(g):
            continue
        seq = g.aid.tolist()
        segs.append(dict(block=bi, participant=r.participant, label=r.label, seq=seq, ids=sorted(set(seq))))
        occ += [(bi, r.participant, r.label, n, a) for n, a in zip(g.index, g.aid)]
    occ = pd.DataFrame(occ, columns=["block", "participant", "label", "narr", "aid"])
    info = dict(n_narrations_with_main_action=len(narr), n_narrations_without_main_action=n_no_main,
                n_action_types=len(vocab), n_blocks=len(b) + len(dropped_multi), dropped_multi_recipe_blocks=dropped_multi,
                n_blocks_with_narrations=len(segs), n_broken_blocks=int((b.e < b.s).sum()),
                n_narrations_in_no_block=int(len(narr) - occ.narr.nunique()),
                n_narrations_in_2plus_blocks=int((occ.groupby("narr").size() > 1).sum()))
    return narr, b, segs, occ, vocab, info


@torch.no_grad()
def qwen3_features(vocab, device):
    """Qwen3-Embedding-0.6B of each class key exactly as given, with the main model's instruction prompt; cached."""
    cache = OUT / "cache" / "qwen3_keys.pt"
    if cache.exists():
        saved = torch.load(cache)
        if saved["vocab"] == vocab:
            return saved["feats"]
    from transformers import AutoModel, AutoTokenizer
    name = "Qwen/Qwen3-Embedding-0.6B"
    tok = AutoTokenizer.from_pretrained(name, padding_side="left")
    enc = AutoModel.from_pretrained(name, dtype=torch.float16).to(device).eval()
    prompts = [f"Instruct: {QWEN3_TASK}\nQuery: {a}" for a in vocab]
    out = []
    for i in range(0, len(prompts), 256):
        batch = tok(prompts[i:i + 256], padding=True, truncation=True, max_length=64, return_tensors="pt").to(device)
        out.append(enc(**batch).last_hidden_state[:, -1].float().cpu())
    feats = F.normalize(torch.cat(out), dim=-1)
    cache.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(vocab=vocab, feats=feats), cache)
    del enc
    torch.cuda.empty_cache()
    return feats


# ------------------------------------------------------------------ counting
def specificity(counts):
    """rows = items, cols = labels -> 1 - normalised entropy (nan where the row is empty)"""
    n = counts.sum(1, keepdims=True)
    p = np.divide(counts, n, out=np.zeros_like(counts, dtype=float), where=n > 0)
    h = -(np.where(p > 0, p * np.log(np.where(p > 0, p, 1)), 0)).sum(1)
    s = 1 - h / math.log(counts.shape[1])
    s[n[:, 0] == 0] = np.nan
    return s


def label_counts(o, key, n_keys, labels):
    li = {l: i for i, l in enumerate(labels)}
    C = np.zeros((n_keys, len(labels)))
    np.add.at(C, (o[key].values, o.label.map(li).values), 1)
    return C


def windows(segs, ks):
    out = {}
    for k in ks:
        out[k] = [list(dict.fromkeys(s["seq"][i:i + k])) for s in segs for i in range(max(len(s["seq"]) - k + 1, 0))]
    return out


# ------------------------------------------------------------------ one training run
@torch.no_grad()
def radii(model, lists, device, bs=4096):
    out = []
    for i in range(0, len(lists), bs):
        out.append(model.dist0(model.embed_sets(*pad(lists[i:i + bs], device))).cpu())
    return torch.cat(out).numpy()


def run_one(fold, geom, seed, segs, vocab, feats, epochs, device):
    path = RUNS / f"{fold}_{geom}_s{seed}.npz"
    if path.exists():
        return dict(np.load(path))
    train = [dict(s) for s in segs if s["participant"] != fold]
    test = [s for s in segs if s["participant"] == fold]
    labels = sorted({s["label"] for s in train})
    for s in train:
        s["y"] = labels.index(s["label"])
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    model = IntentionModel(len(vocab), len(labels), DIM, feats, K_CONE, geom).to(device)
    t0 = time.time()
    log = train_model(model, train, device, epochs, BATCH, LR, "harmonic", ENT_WEIGHT)
    model.eval()
    res = dict(action_radius=radii(model, [[i] for i in range(len(vocab))], device),
               final_loss=np.array(log[-1]["loss"]), seconds=np.array(time.time() - t0))
    for k, lists in windows(test, WINDOW_KS).items():
        res[f"window{k}_radius"] = radii(model, lists, device)
    RUNS.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **res)
    # always keep the weights and learned scalars, so diagnostics never need a retrain
    torch.save(model.state_dict(), RUNS / f"{fold}_{geom}_s{seed}_model.pt")
    json.dump(dict(labels=labels, c=model.c().item(), alpha_s=model.log_alpha_set.clamp(max=0).exp().item(),
                   alpha_i=model.log_alpha_int.clamp(max=0).exp().item(), tau=model.tau().item(), K=model.K,
                   train_log=log), open(RUNS / f"{fold}_{geom}_s{seed}_params.json", "w"), indent=1)
    del model
    torch.cuda.empty_cache()
    print(f"  {fold} {geom} seed {seed}: {res['seconds']:.0f}s, loss {res['final_loss']:.3f}", flush=True)
    return res


# ------------------------------------------------------------------ main
def rho(x, y):
    m = ~(np.isnan(x) | np.isnan(y))
    return float(spearmanr(x[m], y[m])[0]) if m.sum() >= 5 else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", nargs="+", default=FOLDS)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--geoms", nargs="+", default=["hyperbolic", "euclidean"])
    ap.add_argument("--epochs", type=int, default=100)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    TAB.mkdir(parents=True, exist_ok=True)

    narr, blocks, segs, occ, vocab, info = load_data()
    print(json.dumps({k: v for k, v in info.items() if k != "dropped_multi_recipe_blocks"}))
    feats = qwen3_features(vocab, device)
    verbs, nouns = sorted(narr.verb.unique()), sorted(narr.noun.unique())
    first = narr.drop_duplicates("aid").set_index("aid").reindex(range(len(vocab)))
    verb_of = first.verb.map({v: i for i, v in enumerate(verbs)}).values.astype(int)
    noun_of = first.noun.map({n: i for i, n in enumerate(nouns)}).values.astype(int)
    occ["vid"], occ["nidx"] = verb_of[occ.aid], noun_of[occ.aid]

    action_rows, window_rows, run_rows = [], [], []
    for fold in args.folds:
        tr, te = occ[occ.participant != fold], occ[occ.participant == fold]
        tr_labels, te_labels = sorted(tr.label.unique()), sorted(te.label.unique())
        # uses = narrations (an occurrence in two overlapping blocks counts once for the use count)
        uses_tr = np.bincount(tr.drop_duplicates("narr").aid, minlength=len(vocab))
        n_te = np.bincount(te.drop_duplicates("narr").aid, minlength=len(vocab))
        truth = specificity(label_counts(te, "aid", len(vocab), te_labels))
        spec_tr = specificity(label_counts(tr, "aid", len(vocab), tr_labels))
        spec_noun = specificity(label_counts(tr, "nidx", len(nouns), tr_labels))[noun_of]
        spec_verb = specificity(label_counts(tr, "vid", len(verbs), tr_labels))[verb_of]
        backoff = np.where(~np.isnan(spec_noun), spec_noun, spec_verb)
        backoff_src = np.where(~np.isnan(spec_noun), "noun", np.where(~np.isnan(spec_verb), "verb", "none"))
        # old-style metric: specificity from training-segment presence, actions in >= 5 training segments
        tr_segs = [s for s in segs if s["participant"] != fold]
        seg_counts = np.zeros((len(vocab), len(tr_labels)))
        for s in tr_segs:
            seg_counts[s["ids"], tr_labels.index(s["label"])] += 1
        spec_seg, n_seg = specificity(seg_counts), seg_counts.sum(1)

        te_segs = [s for s in segs if s["participant"] == fold]
        wins = windows(te_segs, WINDOW_KS)
        cnt = Counts([dict(ids=s["ids"], y=tr_labels.index(s["label"])) for s in tr_segs], len(vocab), len(tr_labels))
        win_spec = {}
        for k, lists in wins.items():
            z = cnt.nb_intent(lists)
            q = np.exp(z - z.max(1, keepdims=True)); q /= q.sum(1, keepdims=True)
            win_spec[k] = 1 - (-(q * np.log(q + 1e-12)).sum(1) / math.log(len(tr_labels)))

        rad = {}
        for geom in args.geoms:
            for seed in args.seeds:
                r = run_one(fold, geom, seed, segs, vocab, feats, args.epochs, device)
                rad[(geom, seed)] = r["action_radius"]
                run_rows.append(dict(fold=fold, geometry=geom, seed=seed, seconds=float(r["seconds"]), final_loss=float(r["final_loss"])))
                for k in WINDOW_KS:
                    window_rows.append(dict(fold=fold, geometry=geom, seed=seed, k=k, n=len(wins[k]),
                                            rho_radius_vs_window_specificity=rho(r[f"window{k}_radius"], win_spec[k])))

        tab = pd.DataFrame(dict(fold=fold, action=vocab, train_uses=uses_tr, heldout_n=n_te, heldout_specificity=truth,
                                train_count_specificity=spec_tr, backoff_specificity=backoff, backoff_source=backoff_src,
                                train_segments=n_seg, train_segment_specificity=spec_seg))
        for (geom, seed), r in rad.items():
            tab[f"radius_{geom}_s{seed}"] = r
        tab[tab.heldout_n > 0].to_csv(TAB / f"actions_{fold}.csv", index=False)

        for geom, seed in rad:
            r = rad[(geom, seed)]
            base = dict(fold=fold, geometry=geom, seed=seed)
            # old metric (radius vs TRAINING segment counts, >= 5 training segments), as in the 0.60 result
            m_old = n_seg >= 5
            action_rows.append(dict(base, test="old metric: vs training counts (>= 5 train segments)", group="all",
                                    min_heldout=0, n=int(m_old.sum()), rho_radius=rho(r[m_old], spec_seg[m_old]),
                                    rho_counts=np.nan, rho_radius_vs_log_uses=rho(r[m_old], np.log(n_seg[m_old]))))
            for mh in MIN_HELDOUT:
                ok = n_te >= mh
                for gname, lo, hi in [("all", 0, 10 ** 9)] + GROUPS:
                    m = ok & (uses_tr >= lo) & (uses_tr <= hi)
                    row = dict(base, test="vs held-out truth", group=gname, min_heldout=mh, n=int(m.sum()),
                               rho_radius=rho(r[m], truth[m]),
                               rho_counts=rho(spec_tr[m], truth[m]) if gname != "0 uses" else np.nan,
                               rho_backoff=rho(backoff[m], truth[m]) if gname == "0 uses" else np.nan,
                               n_backoff_noun=int((backoff_src[m] == "noun").sum()) if gname == "0 uses" else None,
                               n_backoff_verb=int((backoff_src[m] == "verb").sum()) if gname == "0 uses" else None,
                               n_backoff_none=int((backoff_src[m] == "none").sum()) if gname == "0 uses" else None,
                               rho_radius_vs_log_uses=rho(r[m], np.log1p(uses_tr[m])))
                    if gname == "all":   # counts can only score actions seen in training
                        row["rho_counts"] = rho(spec_tr[m & (uses_tr > 0)], truth[m & (uses_tr > 0)])
                        row["n_counts"] = int((m & (uses_tr > 0)).sum())
                    action_rows.append(row)
        pd.DataFrame(action_rows).to_csv(TAB / "action_level_runs.csv", index=False)
        pd.DataFrame(window_rows).to_csv(TAB / "window_level_runs.csv", index=False)
        pd.DataFrame(run_rows).to_csv(TAB / "runs.csv", index=False)
        print(f"== {fold} done", flush=True)

    json.dump(dict(data=info, settings=dict(dim=DIM, K=K_CONE, ent_weight=ENT_WEIGHT, lr=LR, batch=BATCH, epochs=args.epochs,
                                            subset="harmonic", seeds=args.seeds, window_ks=WINDOW_KS, min_heldout=MIN_HELDOUT,
                                            text="Qwen3-Embedding-0.6B, class key as given, main-model instruction prompt")),
              open(OUT / "results.json", "w", encoding="utf-8"), indent=1, default=str)


if __name__ == "__main__":
    main()
