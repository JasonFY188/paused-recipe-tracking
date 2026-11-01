"""
analysis/object_hierarchy/eval_baselines.py
Steps 3-4 of the feasibility check: flat baselines for the next new object, summed into tree levels, and the value of
hedging. LOPO over 9 people, seed 0, no LLM.

Models (probability over the training people's target classes):
  global    frequency of target classes in the training people
  kitchen   frequency in the test person's OTHER videos, smoothed with the global frequency (weight 1)
  nb        naive Bayes over the distinct past actions of the block (verb + noun class); an action unseen in training
            backs off to its noun and its verb; Laplace smoothing 1
  mlp       1 hidden layer (512, ReLU, dropout 0.2) on the mean Qwen3-Embedding of the past actions' class keys
            ("take maker:coffee"), soft targets, AdamW 1e-3, 15 epochs, batch 512; weights saved per fold
Trees: storage (class -> home fixture -> fixture type -> root), storage-3D (class -> home fixture -> 3D area -> root),
semantic (class -> noun category -> root). A class's home comes from the test video's map (other videos; or earlier
videos = causal); classes without a storage home go to the nodes "always out" / "unknown".
Hedging (Deng et al. 2012 style): answer at the deepest level whose best node has summed probability >= theta.
Specificity of an answer node = 1 - log(#classes under it) / log(#classes).

Run (intention env): KMP_DUPLICATE_LIB_OK=TRUE python analysis/object_hierarchy/eval_baselines.py
Outputs: analysis/object_hierarchy/results/*.csv, figures/*.png, models/*.pt
"""
import json
import math
import pathlib
import random
import sys
from collections import Counter, defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RES, FIG, MOD = HERE / "results", HERE / "figures", HERE / "models"
FOLDS = [f"P0{i}" for i in range(1, 10)]
SEED = 0
THETAS = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
MODELS = ["global", "kitchen", "nb", "mlp"]
# variants: (name, target column prefix, window, home map)
VARIANTS = [("main", "", 10, "home_main"), ("causal_map", "", 10, "home_causal"), ("class_new", "cls_", 10, "home_main"),
            ("W5", "", 5, "home_main"), ("W20", "", 20, "home_main")]


@torch.no_grad()
def qwen3(keys, device):
    cache = HERE / "data" / "qwen3_keys.pt"
    if cache.exists():
        s = torch.load(cache)
        if set(keys) <= set(s["keys"]):
            return {k: s["feats"][i] for i, k in enumerate(s["keys"])}
    sys.path.insert(0, str(ROOT / "scripts"))
    from hyperbolic_intention import QWEN3_TASK
    from transformers import AutoModel, AutoTokenizer
    name = "Qwen/Qwen3-Embedding-0.6B"
    tok = AutoTokenizer.from_pretrained(name, padding_side="left")
    enc = AutoModel.from_pretrained(name, dtype=torch.float16).to(device).eval()
    keys = sorted(keys)
    out = []
    for i in range(0, len(keys), 256):
        b = tok([f"Instruct: {QWEN3_TASK}\nQuery: {k}" for k in keys[i:i + 256]], padding=True, truncation=True,
                max_length=64, return_tensors="pt").to(device)
        out.append(enc(**b).last_hidden_state[:, -1].float().cpu())
    feats = F.normalize(torch.cat(out), dim=-1)
    torch.save(dict(keys=keys, feats=feats), cache)
    del enc; torch.cuda.empty_cache()
    return {k: feats[i] for i, k in enumerate(keys)}


def soft_targets(targets, cidx):
    rows, cols, vals = [], [], []
    for i, t in enumerate(targets):
        cs = [cidx[c] for c in t if c in cidx]
        for c in cs:
            rows.append(i); cols.append(c); vals.append(1 / len(cs))
    return sp.csr_matrix((vals, (rows, cols)), shape=(len(targets), len(cidx)))


def nb_features(df, fvocab=None):
    feats = []
    for past, vs, ns in zip(df.past, df.past_verbs, df.past_nouns):
        f = set()
        for a, v, n in zip(past, vs, ns):
            if fvocab is None or ("a:" + a) in fvocab:
                f.add("a:" + a)
            else:
                if ("n:" + n) in fvocab:
                    f.add("n:" + n)
                if ("v:" + v) in fvocab:
                    f.add("v:" + v)
        if fvocab is None:                                            # training: actions + their back-off features
            f |= {"n:" + n for n in ns} | {"v:" + v for v in vs}
        feats.append(f)
    return feats


def to_sparse(feats, fidx):
    rows, cols = [], []
    for i, f in enumerate(feats):
        for x in f:
            if x in fidx:
                rows.append(i); cols.append(fidx[x])
    return sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(feats), len(fidx)))


def nb_model(tr, te, Y, alpha=1.0):
    ftr = nb_features(tr)
    fvocab = sorted(set().union(*ftr))
    fidx = {f: i for i, f in enumerate(fvocab)}
    Xtr = to_sparse(ftr, fidx)
    nc = np.asarray(Y.sum(0)).ravel()
    ncf = (Xtr.T @ Y).toarray().T                                     # classes x features
    # multinomial NB (feature presence counted once per moment): p(f | c) = (n_cf + a) / (sum_f n_cf + a |F|)
    logp = np.log((ncf + alpha) / (ncf.sum(1, keepdims=True) + alpha * ncf.shape[1]))
    prior = np.log(nc + alpha) - np.log(nc.sum() + alpha * len(nc))
    Xte = to_sparse(nb_features(te, set(fvocab)), fidx)
    z = Xte @ logp.T + prior
    z = z - z.max(1, keepdims=True)
    p = np.exp(z)
    return p / p.sum(1, keepdims=True)


def mlp_model(tr, te, Y, emb, device, fold, tag):
    def X(df):
        return torch.stack([torch.stack([emb[a] for a in past]).mean(0) for past in df.past])
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
    xtr, xte = X(tr).to(device), X(te).to(device)
    ytr = torch.tensor(Y.toarray(), dtype=torch.float32, device=device)
    net = nn.Sequential(nn.Linear(xtr.shape[1], 512), nn.ReLU(), nn.Dropout(0.2), nn.Linear(512, Y.shape[1])).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    g = torch.Generator(device="cpu").manual_seed(SEED)
    for _ in range(15):
        net.train()
        for idx in torch.randperm(len(xtr), generator=g).split(512):
            loss = -(ytr[idx] * F.log_softmax(net(xtr[idx]), 1)).sum(1).mean()
            opt.zero_grad(); loss.backward(); opt.step()
    net.eval()
    MOD.mkdir(exist_ok=True)
    torch.save(net.state_dict(), MOD / f"mlp_{tag}_{fold}.pt")
    with torch.no_grad():
        return torch.softmax(net(xte), 1).cpu().numpy()


def evaluate(P, te, classes, tcol, home, area3d, category, freq_train, resume):
    """per-moment results for one model; P: n_test x n_classes. Class probabilities are summed into tree nodes."""
    nC = len(classes)
    typ = lambda h: h if h in ("always out", "unknown") else h.split("_", 1)[1].rsplit(".", 1)[0]
    a3 = lambda h: h if h in ("always out", "unknown") else area3d.get(h, "unknown")
    gfreq = np.array([freq_train.get(c, 0) for c in classes], float)
    out = []
    top = P.argmax(1)
    top5 = np.argsort(-P, 1)[:, :5]
    for v in te.video_id.unique():
        rows_v = np.where((te.video_id == v).values)[0]
        hm = home.get(v, {})
        node_names = {}
        nf = [hm.get(c, "unknown") for c in classes]
        node_names["storage"] = [("fixture", nf), ("area", [typ(h) for h in nf])]
        node_names["storage3d"] = [("fixture", nf), ("area", [a3(h) for h in nf])]
        node_names["semantic"] = [("category", [category.get(c, "other") for c in classes])]
        enc = {}
        for tname, levels in node_names.items():                      # node index per class + one-hot (classes x nodes)
            enc[tname] = []
            for lname, names in levels:
                uniq = sorted(set(names))
                ix = np.array([uniq.index(n) for n in names])
                oh = np.zeros((nC, len(uniq))); oh[np.arange(nC), ix] = 1
                enc[tname].append((lname, names, ix, oh, oh.sum(0)))
        Pv = P[rows_v]
        agg = {t: [Pv @ oh for (_, _, _, oh, _) in L] for t, L in enc.items()}
        for k, i in enumerate(rows_v):
            T = set(te[tcol].iloc[i])
            Tm = np.array([c in T for c in classes])
            truth_cls = {c for c in T}
            r = dict(i=int(i), video_id=v, resume=bool(resume[i]), picked_from_storage=bool(te["_storage"].iloc[i]),
                     freq=max([freq_train.get(c, 0) for c in T]),
                     home_kind="storage" if any(hm.get(c, "unknown") not in ("always out", "unknown") for c in T) else
                     ("always out" if any(hm.get(c) == "always out" for c in T) else "unknown"),
                     acc1=bool(Tm[top[i]]), acc5=bool(Tm[top5[i]].any()), top1_prob=float(P[i, top[i]]))
            wrong = not Tm[top[i]]
            others = ~Tm
            gw = gfreq[others]
            gw = gw / gw.sum() if gw.sum() else np.full(others.sum(), 1 / max(others.sum(), 1))
            for tname, L in enc.items():
                truths = []
                for (lname, names, ix, oh, size), A in zip(L, agg[tname]):
                    if tname != "semantic" and lname == "fixture":
                        truth = {hm.get(c, "unknown") for c in T}
                    elif tname == "storage" and lname == "area":
                        truth = {typ(hm.get(c, "unknown")) for c in T}
                    elif tname == "storage3d" and lname == "area":
                        truth = {a3(hm.get(c, "unknown")) for c in T}
                    else:
                        truth = {category.get(c, "other") for c in T}
                    uniq = sorted(set(names))
                    tmask = np.array([u in truth for u in uniq])
                    b = A[k].argmax()
                    r[f"{tname}_{lname}_acc1"] = bool(tmask[b])
                    if wrong:
                        inn = tmask[ix][others].astype(float)
                        r[f"{tname}_{lname}_wrong_in"] = bool(tmask[ix[top[i]]])
                        r[f"{tname}_{lname}_rand_uniform"] = float(inn.mean()) if len(inn) else np.nan
                        r[f"{tname}_{lname}_rand_freq"] = float((inn * gw).sum())
                    truths.append((A[k], tmask, size))
                for th in THETAS:                                     # hedging: deepest level with best node >= theta
                    if P[i, top[i]] >= th:
                        ok, spec = bool(Tm[top[i]]), 1.0
                    else:
                        ok, spec = True, 0.0
                        for A_k, tmask, size in truths:
                            b = A_k.argmax()
                            if A_k[b] >= th:
                                ok, spec = bool(tmask[b]), 1 - math.log(size[b]) / math.log(nC)
                                break
                    r[f"{tname}_hedge{th}_ok"] = ok
                    r[f"{tname}_hedge{th}_spec"] = spec
            out.append(r)
    return pd.DataFrame(out)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="+", default=[v[0] for v in VARIANTS])
    ap.add_argument("--folds", nargs="+", default=FOLDS)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    RES.mkdir(exist_ok=True); FIG.mkdir(exist_ok=True)
    D = pd.read_pickle(HERE / "data" / "objects.pkl")
    mom, pick, kind = D["moments"], D["pick"], D["kind"]
    rf = pd.read_pickle(HERE / "data" / "resume_flags.pkl")
    mom["resume"] = rf.resume.values
    nc = pd.read_csv(ROOT / "annotations" / "narrations-and-action-segments" / "HD_EPIC_noun_classes.csv")
    category = dict(zip(nc.key, nc.category))
    emb = qwen3(sorted({a for past in mom.past for a in past}), device)
    fetch_pick = pick[pick.fetch.apply(len) > 0]

    for vname, prefix, W, hmap in [v for v in VARIANTS if v[0] in args.variants]:
        tcol = f"{prefix}target_W{W}"
        m = mom[mom[tcol].notna()].copy()
        m["_storage"] = m[f"{prefix}storage_W{W}"]
        rows = []
        for fold in args.folds:
            tr, te = m[m.participant != fold], m[m.participant == fold]
            classes = sorted({c for t in tr[tcol] for c in t})
            cidx = {c: i for i, c in enumerate(classes)}
            Y = soft_targets(list(tr[tcol]), cidx)
            freq_train = Counter(c for cl in fetch_pick[fetch_pick.participant != fold].fetch for c in cl)
            gl = np.asarray(Y.sum(0)).ravel(); gl = gl / gl.sum()
            probs = {"global": np.tile(gl, (len(te), 1))}
            kp = np.zeros((len(te), len(classes)))
            for v in te.video_id.unique():                             # kitchen: the person's other videos
                other = m[(m.participant == fold) & (m.video_id != v)]
                cnt = np.asarray(soft_targets(list(other[tcol]), cidx).sum(0)).ravel()
                kp[(te.video_id == v).values] = (cnt + gl) / (cnt.sum() + 1)
            probs["kitchen"] = kp
            probs["nb"] = nb_model(tr, te, Y)
            probs["mlp"] = mlp_model(tr, te, Y, emb, device, fold, vname)
            for name, P in probs.items():
                ev = evaluate(P, te, classes, tcol, D[hmap], D["area3d"], category, freq_train, te.resume.values)
                ev["model"], ev["fold"] = name, fold
                rows.append(ev)
            print(f"  {vname} {fold}: n={len(te)} acc1 " + " ".join(f"{k} {rows[-4 + j].acc1.mean():.3f}" for j, k in enumerate(probs)), flush=True)
        R = pd.concat(rows, ignore_index=True)
        R.to_pickle(RES / f"moments_{vname}{args.tag}.pkl")
        print(f"== {vname} done", flush=True)


if __name__ == "__main__":
    main()
