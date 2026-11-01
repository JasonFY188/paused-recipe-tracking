"""
analysis/hyperbolic_tree/train_eval.py
Stage 2: map past actions onto the FIXED tree of tree_embed.py, and evaluate as in analysis/object_hierarchy.

Data     analysis/object_hierarchy main setting: next new physical object within 10 actions ("X of Y": X or Y counts),
         fetchable rule, other-videos storage map, LOPO over 9 people; input = mean Qwen3 embedding of the past action
         keys (the same input as its MLP); classes = the 221 tree leaves
Models   hyperbolic  MLP (1024 -> 256 -> 16, ReLU, dropout 0.2) -> expmap0 into the Poincare ball; Busemann loss to the
                     target's ideal prototype (leaf direction of Stage 1), the closer one for X / Y targets, plus the
                     penalty -(phi) log(1 - |z|^2), phi = 0.1 * dim (Ghadimi Atigh et al. 2021 form; phi not tuned)
         euclidean   same MLP, no map; prototypes = unit leaf directions (relative to the root); loss = squared distance
                     to the closer target prototype
         flat        same MLP with a 221-way softmax head, soft targets (the flat control on the same classes)
         AdamW lr 1e-3, weight decay 1e-4, 20 epochs, batch 512; seeds 0, 1, 2
Scores   hyperbolic -Busemann(z, p_c); euclidean -|z - p_c|; flat logits. Probabilities = softmax(score / tau), tau = one
         temperature per run chosen on the TRAINING moments (grid, min NLL of the target)
Hedging  A  sum class probabilities into tree nodes; answer at the deepest level whose best node >= theta
         B  (geometric; hyperbolic and euclidean) candidates = leaves whose logit is within delta of the best;
            answer = their lowest common ancestor in the tree (leaf / category / kind / root); confidence = radius of z
Saved    runs/<model>_s<seed>_<fold>.pkl (per-moment results), .pt (weights), .json (learned scalars, tau, loss log)
Time     --hours (default 7): stops cleanly before starting a run that would pass the limit; restart skips done runs

Run (intention env): KMP_DUPLICATE_LIB_OK=TRUE python analysis/hyperbolic_tree/train_eval.py
"""
import argparse
import json
import math
import pathlib
import subprocess
import sys
import time
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OH = ROOT / "analysis" / "object_hierarchy"
RUNS = HERE / "runs"
FOLDS = [f"P0{i}" for i in range(1, 10)]
THETAS = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
DELTAS = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
PHI_PER_DIM = 0.1
MAXN = 1 - 1e-4


def load():
    D = pd.read_pickle(OH / "data" / "objects.pkl")
    tree = pd.read_pickle(HERE / "data" / "tree.pkl")
    mom = D["moments"]
    rf = pd.read_pickle(OH / "data" / "resume_flags.pkl")
    mom = mom.assign(resume=rf.resume.values)
    m = mom[mom.target_W10.notna()].reset_index(drop=True)
    cache = torch.load(OH / "data" / "qwen3_keys.pt")
    emb = {k: cache["feats"][i] for i, k in enumerate(cache["keys"])}
    X = torch.stack([torch.stack([emb[a] for a in past]).mean(0) for past in m.past])
    return D, tree, m, X


class Enc(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim_in, 256), nn.ReLU(), nn.Dropout(0.2), nn.Linear(256, dim_out))

    def forward(self, x):
        return self.net(x)


def expmap0(v):
    n = v.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    z = torch.tanh(n) * v / n
    zn = z.norm(dim=-1, keepdim=True)
    return torch.where(zn > MAXN, z / zn * MAXN, z)


def scores(model, out, P):
    """class scores (higher = better) for outputs out [n, d] and prototypes P [C, d]"""
    if model == "hyperbolic":
        b = torch.log(((P[None] - out[:, None]) ** 2).sum(-1).clamp_min(1e-12)) - torch.log((1 - (out ** 2).sum(-1, keepdim=True)).clamp_min(1e-12))
        return -b
    if model == "euclidean":
        return -torch.cdist(out, P)
    return out


def train_run(model, seed, Xtr, Ttr, P, epochs, device):
    torch.manual_seed(seed); np.random.seed(seed)
    d = P.shape[1]
    enc = Enc(Xtr.shape[1], P.shape[0] if model == "flat" else d).to(device)
    opt = torch.optim.AdamW(enc.parameters(), lr=1e-3, weight_decay=1e-4)
    g = torch.Generator().manual_seed(seed)
    Xtr, Ttr = Xtr.to(device), Ttr.to(device)
    phi = PHI_PER_DIM * d
    log = []
    for ep in range(epochs):
        enc.train()
        tot = 0.0
        for idx in torch.randperm(len(Xtr), generator=g).split(512):
            v, T = enc(Xtr[idx]), Ttr[idx]
            if model == "hyperbolic":
                z = expmap0(v)
                dist2 = ((P[None] - z[:, None]) ** 2).sum(-1).clamp_min(1e-12)
                first = torch.log(dist2).masked_fill(~T, float("inf")).min(1).values
                loss = (first - (phi + 1) * torch.log((1 - (z ** 2).sum(-1)).clamp_min(1e-12))).mean()
            elif model == "euclidean":
                dist2 = ((P[None] - v[:, None]) ** 2).sum(-1)
                loss = dist2.masked_fill(~T, float("inf")).min(1).values.mean()
            else:
                tgt = T.float() / T.float().sum(1, keepdim=True)
                loss = -(tgt * F.log_softmax(v, 1)).sum(1).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(idx)
        log.append(round(tot / len(Xtr), 5))
    enc.eval()
    return enc, log


@torch.no_grad()
def forward(model, enc, X, P, device):
    v = enc(X.to(device))
    out = expmap0(v) if model == "hyperbolic" else v
    S = scores(model, out, P)
    if model == "hyperbolic":
        radius = 2 * torch.atanh(out.norm(dim=-1).clamp(max=MAXN))
    elif model == "euclidean":
        radius = out.norm(dim=-1)
    else:
        radius = torch.full((len(out),), float("nan"), device=device)
    return S.cpu(), radius.cpu().numpy()


def fit_tau(S, T, prior=None):
    best = (float("inf"), 1.0)
    for tau in np.exp(np.linspace(np.log(0.02), np.log(50), 60)):
        p = torch.softmax(S / tau + (0 if prior is None else prior), 1)
        nll = -torch.log((p * T).sum(1).clamp_min(1e-12)).mean().item()
        best = min(best, (nll, float(tau)))
    return best[1], best[0]


def evaluate(P, S_logit, T, m_te, leaves, tree, homes, freq, radius, geometric):
    """per-moment results. P probs [n, C] (numpy), T bool [n, C]"""
    n, C = P.shape
    cat_of = {l: tree["parent"][l] for l in leaves}
    kind_of = {l: tree["parent"][cat_of[l]] for l in leaves}
    def onehot(names):
        u = sorted(set(names)); ix = np.array([u.index(x) for x in names])
        M = np.zeros((C, len(u))); M[np.arange(C), ix] = 1
        return M
    Mcat, Mkind = onehot([cat_of[l] for l in leaves]), onehot([kind_of[l] for l in leaves])
    f = np.array([freq.get(l, 0) for l in leaves], float)
    top = P.argmax(1); top5 = np.argsort(-P, 1)[:, :5]
    R = pd.DataFrame(dict(acc1=T[np.arange(n), top], acc5=np.take_along_axis(T, top5, 1).any(1),
                          top1_prob=P[np.arange(n), top], radius=radius,
                          freq=[max(freq.get(c, 0) for c in t) for t in m_te.target_W10],
                          storage=m_te.storage_W10.values.astype(bool), resume=m_te.resume.values.astype(bool),
                          participant=m_te.participant.values))
    wrong = ~R.acc1.values
    others = ~T
    def level_stats(name, M, Pn, Tn, Mper=None):
        b = Pn.argmax(1)
        R[f"{name}_acc1"] = Tn[np.arange(n), b]
        inn = (Tn @ M.T > 0) if Mper is None else Mper                 # class j lies in a true node
        R[f"{name}_wrong_in"] = np.where(wrong, inn[np.arange(n), top], np.nan)
        fw = others * f
        R[f"{name}_rand_freq"] = np.where(wrong, (inn * fw).sum(1) / np.maximum(fw.sum(1), 1e-12), np.nan)
    Tcat, Tkind = (T @ Mcat) > 0, (T @ Mkind) > 0
    Pcat, Pkind = P @ Mcat, P @ Mkind
    level_stats("category", Mcat, Pcat, Tcat)
    level_stats("kind", Mkind, Pkind, Tkind)
    # storage tree per video (home fixture, fixture type)
    fix_ok, typ_ok = np.zeros(n, bool), np.zeros(n, bool)
    fix_win, fix_rf, typ_win, typ_rf = (np.full(n, np.nan) for _ in range(4))
    hedge_store = {th: (np.zeros(n, bool), np.zeros(n)) for th in THETAS}
    typ = lambda h: h if h in ("always out", "unknown") else h.split("_", 1)[1].rsplit(".", 1)[0]
    for v in m_te.video_id.unique():
        rows = np.where((m_te.video_id == v).values)[0]
        hm = homes.get(v, {})
        names_f = [hm.get(l, "unknown") for l in leaves]
        levels = []
        for names in (names_f, [typ(h) for h in names_f]):
            M = onehot(names)
            Pn, Tn = P[rows] @ M, (T[rows] @ M) > 0
            b = Pn.argmax(1)
            inn = (Tn @ M.T) > 0
            fw = others[rows] * f
            levels.append((Pn, Tn, M, b, inn, fw))
        (Pf, Tf, Mf, bf, innf, fwf), (Pt, Tt, Mt, bt, innt, fwt) = levels
        fix_ok[rows], typ_ok[rows] = Tf[np.arange(len(rows)), bf], Tt[np.arange(len(rows)), bt]
        w = wrong[rows]
        fix_win[rows] = np.where(w, innf[np.arange(len(rows)), top[rows]], np.nan)
        typ_win[rows] = np.where(w, innt[np.arange(len(rows)), top[rows]], np.nan)
        fix_rf[rows] = np.where(w, (innf * fwf).sum(1) / np.maximum(fwf.sum(1), 1e-12), np.nan)
        typ_rf[rows] = np.where(w, (innt * fwt).sum(1) / np.maximum(fwt.sum(1), 1e-12), np.nan)
        sizes = [Mf.sum(0), Mt.sum(0)]
        for th in THETAS:                                             # hedging A on the storage tree
            ok, spec = hedge_store[th]
            for k, i in enumerate(rows):
                if P[i, top[i]] >= th:
                    ok[i], spec[i] = T[i, top[i]], 1.0
                    continue
                ok[i], spec[i] = True, 0.0
                for (Pn, Tn), sz in zip(((Pf, Tf), (Pt, Tt)), sizes):
                    b = Pn[k].argmax()
                    if Pn[k, b] >= th:
                        ok[i], spec[i] = Tn[k, b], 1 - math.log(sz[b]) / math.log(C)
                        break
    R["fixture_acc1"], R["type_acc1"] = fix_ok, typ_ok
    R["fixture_wrong_in"], R["fixture_rand_freq"] = fix_win, fix_rf
    R["type_wrong_in"], R["type_rand_freq"] = typ_win, typ_rf
    for th in THETAS:
        R[f"storageA_{th}_ok"], R[f"storageA_{th}_spec"] = hedge_store[th]
    # hedging A on the semantic tree (leaf -> category -> kind -> root)
    scat, skind = Mcat.sum(0), Mkind.sum(0)
    for th in THETAS:
        leaf = P[np.arange(n), top] >= th
        bc, bk = Pcat.argmax(1), Pkind.argmax(1)
        catq = ~leaf & (Pcat[np.arange(n), bc] >= th)
        kindq = ~leaf & ~catq & (Pkind[np.arange(n), bk] >= th)
        ok = np.where(leaf, T[np.arange(n), top], np.where(catq, Tcat[np.arange(n), bc], np.where(kindq, Tkind[np.arange(n), bk], True)))
        spec = np.where(leaf, 1.0, np.where(catq, 1 - np.log(scat[bc]) / math.log(C), np.where(kindq, 1 - np.log(skind[bk]) / math.log(C), 0.0)))
        R[f"semanticA_{th}_ok"], R[f"semanticA_{th}_spec"] = ok, spec
    # hedging B (geometric): lowest common ancestor of the leaves within delta of the best logit
    if geometric:
        L = S_logit
        best = L.max(1, keepdims=True)
        for dl in DELTAS:
            cand = L >= best - dl
            ncand = cand.sum(1)
            ncat = ((cand.astype(float) @ Mcat) > 0).sum(1)
            nkind = ((cand.astype(float) @ Mkind) > 0).sum(1)
            catn = (cand.astype(float) @ Mcat).argmax(1)
            kindn = (cand.astype(float) @ Mkind).argmax(1)
            ok = np.where(ncand == 1, T[np.arange(n), top],
                          np.where(ncat == 1, Tcat[np.arange(n), catn], np.where(nkind == 1, Tkind[np.arange(n), kindn], True)))
            spec = np.where(ncand == 1, 1.0, np.where(ncat == 1, 1 - np.log(scat[catn]) / math.log(C),
                                                        np.where(nkind == 1, 1 - np.log(skind[kindn]) / math.log(C), 0.0)))
            R[f"semanticB_{dl}_ok"], R[f"semanticB_{dl}_spec"] = ok, spec
    return R


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", nargs="+", default=FOLDS)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--models", nargs="+", default=["hyperbolic", "euclidean", "flat"])
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--hours", type=float, default=7.0)
    ap.add_argument("--tag", default="")
    ap.add_argument("--no-report", action="store_true")
    args = ap.parse_args()
    t0 = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    RUNS.mkdir(exist_ok=True)
    D, tree, m, X = load()
    leaves = tree["leaves"]
    lidx = {l: i for i, l in enumerate(leaves)}
    T_all = np.zeros((len(m), len(leaves)), bool)
    for i, t in enumerate(m.target_W10):
        for c in t:
            T_all[i, lidx[c]] = True
    protos = {}
    for geom in ("hyperbolic", "euclidean"):
        Xg = torch.tensor(tree[geom]["X"], dtype=torch.float32)
        Pl = Xg[[tree["nodes"].index(l) for l in leaves]]
        protos[geom] = (Pl / Pl.norm(dim=-1, keepdim=True)).to(device)   # ideal points / unit directions
    protos["flat"] = protos["euclidean"]                               # unused by the flat head
    pick = D["pick"]
    fp = pick[pick.fetch.apply(len) > 0]
    status = dict(done=[], skipped_time=[])
    est = None
    for seed in args.seeds:
        for fold in args.folds:
            for model in args.models:
                name = f"{model}_s{seed}_{fold}{args.tag}"
                if (RUNS / f"{name}.pkl").exists():
                    continue
                el = time.time() - t0
                if est and el + est > args.hours * 3600:
                    status["skipped_time"].append(name)
                    continue
                ts = time.time()
                tr, te = (m.participant != fold).values, (m.participant == fold).values
                Ttr = torch.tensor(T_all[tr])
                enc, log = train_run(model, seed, X[tr], Ttr, protos[model], args.epochs, device)
                Str, _ = forward(model, enc, X[tr], protos[model], device)
                Ste, rad = forward(model, enc, X[te], protos[model], device)
                freq = Counter(c for cl in fp[fp.participant != fold].fetch for c in cl)
                # scoring: geometric models are evaluated with the training log prior added (main: pure Busemann /
                # distance scores collapse onto one "hub" class, see report) and without it ("_pure")
                cnt = Ttr.float().sum(0) + 1
                prior = torch.log(cnt / cnt.sum())
                variants = [(model, None)] if model == "flat" else [(model, prior), (f"{model}_pure", None)]
                params = dict(model=model, seed=seed, fold=fold, loss_log=log, epochs=args.epochs,
                              phi=PHI_PER_DIM * protos[model].shape[1] if model == "hyperbolic" else None)
                Rs = []
                for vname, pr in variants:
                    tau, nll = fit_tau(Str, Ttr.float(), pr)
                    L = Ste / tau + (0 if pr is None else pr)
                    Pte = torch.softmax(L, 1).numpy()
                    R = evaluate(Pte, L.numpy(), T_all[te], m[te].reset_index(drop=True), leaves, tree, D["home_main"],
                                 freq, rad, model != "flat")
                    R["model"], R["seed"], R["fold"] = vname, seed, fold
                    Rs.append(R)
                    params[f"tau_{vname}"], params[f"train_nll_{vname}"] = tau, nll
                R = pd.concat(Rs, ignore_index=True)
                R.to_pickle(RUNS / f"{name}.pkl")
                torch.save(enc.state_dict(), RUNS / f"{name}.pt")
                params["seconds"] = round(time.time() - ts, 1)
                json.dump(params, open(RUNS / f"{name}.json", "w"), indent=1)
                R = Rs[0]
                est = time.time() - ts if est is None else max(est, time.time() - ts)
                status["done"].append(name)
                print(f"{name}: acc1 {R.acc1.mean():.3f} acc5 {R.acc5.mean():.3f} cat {R.category_acc1.mean():.3f} "
                      f"fixture {R.fixture_acc1.mean():.3f} ({time.time() - ts:.0f}s)", flush=True)
                json.dump(dict(state="running", hours=round((time.time() - t0) / 3600, 2), **status),
                          open(HERE / "status.json", "w"), indent=1)
    status["state"] = "time limit reached" if status["skipped_time"] else "finished"
    status["hours"] = round((time.time() - t0) / 3600, 2)
    json.dump(status, open(HERE / "status.json", "w"), indent=1)
    if not args.no_report:
        subprocess.run([sys.executable, str(HERE / "report.py")], check=False)


if __name__ == "__main__":
    main()
