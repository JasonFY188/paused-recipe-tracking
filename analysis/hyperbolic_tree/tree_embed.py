"""
analysis/hyperbolic_tree/tree_embed.py
Stage 1: place the KNOWN object tree in space, then keep it fixed.

Tree    root -> kind (tool / ingredient) -> (kind, HD-EPIC noun category) -> noun class; leaves = all fetchable classes
        (kind tool or ingredient, not water / food / mixture; kinds from analysis/recipe_groups step 1 as confirmed)
Embed   Poincare ball (Nickel & Kiela 2017): for every ancestor-descendant pair (u, v) of the transitive closure,
        -log softmax of -d(u, v) against 50 sampled unrelated nodes; Riemannian SGD (lr 1.0, 50 burn-in epochs at
        lr 0.1), 1500 epochs (400 epochs / lr 0.3 / 10 negatives gave MAP 0.32 only), dimension 16, seed 0. Euclidean control: the same loss with Euclidean distance, plain SGD.
Checks  mean rank and MAP of reconstructing the closure; radius (distance from the origin) per tree level
Output  data/tree.pkl
"""
import pathlib
import random

import numpy as np
import pandas as pd
import torch

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DIM, EPOCHS, BURN, LR, NEG, SEED = 16, 1500, 50, 1.0, 50, 0
EPS = 1e-5


def build_tree():
    kinds = pd.read_csv(ROOT / "analysis" / "recipe_groups" / "tables" / "step1_proposed_kinds.csv")
    nc = pd.read_csv(ROOT / "annotations" / "narrations-and-action-segments" / "HD_EPIC_noun_classes.csv")
    cat = dict(zip(nc.key, nc.category))
    leaves = sorted(k for k, kd in zip(kinds.cls, kinds.kind)
                    if kd in ("tool", "ingredient") and k not in ("water", "food", "mixture"))
    kind = dict(zip(kinds.cls, kinds.kind))
    nodes = ["root", "kind:tool", "kind:ingredient"]
    parent = {"kind:tool": "root", "kind:ingredient": "root"}
    for l in leaves:
        c = f"cat:{kind[l]}/{cat[l]}"
        if c not in parent:
            nodes.append(c); parent[c] = f"kind:{kind[l]}"
    nodes += leaves
    for l in leaves:
        parent[l] = f"cat:{kind[l]}/{cat[l]}"
    level = {"root": 0}
    for n in nodes[1:]:
        level[n] = level[parent[n]] + 1
    anc = {n: [] for n in nodes}
    for n in nodes:
        p = parent.get(n)
        while p:
            anc[n].append(p); p = parent.get(p)
    return nodes, parent, level, anc, leaves


def pdist_poincare(u, v):
    sq = ((u - v) ** 2).sum(-1)
    a = (1 - (u ** 2).sum(-1)).clamp_min(EPS)
    b = (1 - (v ** 2).sum(-1)).clamp_min(EPS)
    return torch.acosh((1 + 2 * sq / (a * b)).clamp_min(1 + 1e-7))


def embed(nodes, anc, geometry):
    rng = np.random.default_rng(SEED); torch.manual_seed(SEED)
    idx = {n: i for i, n in enumerate(nodes)}
    # (descendant u, ancestor v) as in Nickel & Kiela's hypernymy pairs: negatives are sampled for the descendant, so
    # siblings are pushed apart and a node ends up closer to its ancestors than to unrelated nodes
    pairs = np.array([(idx[n], idx[a]) for n in nodes for a in anc[n]])
    related = {i: {i} for i in range(len(nodes))}
    for a, b in pairs:
        related[a].add(b); related[b].add(a)
    allowed = [np.array([w for w in range(len(nodes)) if w not in related[u]]) for u in range(len(nodes))]
    # the root is related to every node, so its pairs have no negatives: they are left out (the root stays at its
    # initial position next to the origin)
    pairs = np.array([p for p in pairs if len(allowed[p[0]])])
    X = (torch.rand(len(nodes), DIM) * 2 - 1) * 1e-3
    dist = pdist_poincare if geometry == "hyperbolic" else (lambda u, v: ((u - v) ** 2).sum(-1).clamp_min(1e-12).sqrt())
    for ep in range(EPOCHS):
        lr = LR / 10 if ep < BURN else LR
        perm = rng.permutation(len(pairs))
        for i in range(0, len(pairs), 64):
            b = pairs[perm[i:i + 64]]
            negs = np.stack([rng.choice(allowed[u], NEG) for u in b[:, 0]])
            cand = torch.tensor(np.concatenate([b[:, 1:2], negs], 1))  # [B, 1 + NEG], true descendant first
            X.requires_grad_(True)
            u = X[torch.tensor(b[:, 0])][:, None].expand(-1, cand.shape[1], -1)
            d = dist(u, X[cand])
            loss = torch.nn.functional.cross_entropy(-d, torch.zeros(len(b), dtype=torch.long))
            g, = torch.autograd.grad(loss, X)
            with torch.no_grad():
                if geometry == "hyperbolic":                         # Riemannian gradient, then project into the ball
                    g = g * ((1 - (X ** 2).sum(-1, keepdim=True)) ** 2 / 4)
                    X = X - lr * g
                    n = X.norm(dim=-1, keepdim=True)
                    X = torch.where(n >= 1 - EPS, X / n * (1 - EPS), X)
                else:
                    X = X - lr * g
            X = X.detach()
    return X, dist


def checks(X, dist, nodes, anc, level):
    idx = {n: i for i, n in enumerate(nodes)}
    ranks, aps = [], []
    with torch.no_grad():
        D = torch.stack([dist(X[i].expand(len(nodes), -1), X) for i in range(len(nodes))]).numpy()
    rel = {n: set(anc[n]) | {m for m in nodes if n in anc[m]} for n in nodes}
    for n in nodes:
        i = idx[n]
        pos = [idx[m] for m in rel[n]]
        if not pos:
            continue
        others = np.array([j for j in range(len(nodes)) if j != i and nodes[j] not in rel[n]], dtype=int)
        if not len(others):                                          # the root: related to every node
            continue
        for j in pos:
            ranks.append(1 + (D[i, others] < D[i, j]).sum())
        order = np.argsort(D[i])
        order = order[order != i]
        hits, prec = 0, []
        for k, j in enumerate(order, 1):
            if j in pos:
                hits += 1; prec.append(hits / k)
        aps.append(np.mean(prec))
    return dict(mean_rank=float(np.mean(ranks)), map=float(np.mean(aps)))


def main():
    (HERE / "data").mkdir(exist_ok=True)
    nodes, parent, level, anc, leaves = build_tree()
    out = dict(nodes=nodes, parent=parent, level=level, anc=anc, leaves=leaves, dim=DIM)
    for geom in ("hyperbolic", "euclidean"):
        X, dist = embed(nodes, anc, geom)
        if geom == "euclidean":
            X = X - X[nodes.index("root")]                           # directions are taken relative to the root
        ck = checks(X, dist, nodes, anc, level)
        if geom == "hyperbolic":
            r = torch.atanh(X.norm(dim=-1).clamp(max=1 - EPS)) * 2   # hyperbolic distance from the origin
        else:
            r = X.norm(dim=-1)
        ck["radius_by_level"] = {int(l): round(float(r[[i for i, n in enumerate(nodes) if level[n] == l]].mean()), 3)
                                 for l in sorted(set(level.values()))}
        ck["root_norm"] = round(float(X[0].norm()), 4)
        out[geom] = dict(X=X.numpy(), checks=ck)
        print(geom, ck)
    pd.to_pickle(out, HERE / "data" / "tree.pkl")
    print(f"nodes {len(nodes)} | leaves {len(leaves)} | categories {sum(n.startswith('cat:') for n in nodes)}")


if __name__ == "__main__":
    main()
