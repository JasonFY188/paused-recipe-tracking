"""
analysis/clean_recheck/diagnostics.py
Are the cone losses of the hyperbolic model actually active? Retrains the clean-recheck hyperbolic model (same data,
seed, settings) for each held-out person with seed 0, and this time keeps the weights, the learned scalars and every
loss term. Old code is imported unchanged.

Exactness: the training loop below repeats scripts/hyperbolic_intention.py train_model() step by step (same random
draws in the same order), and diag_loss() repeats loss_fn() with the same operations in the same order (soft targets
off, set->task cone on), only returning each term and per-pair cone checks as well. The retrained single-action radii
are compared with the saved ones in runs/<fold>_hyperbolic_s0.npz.

Run (intention env): KMP_DUPLICATE_LIB_OK=TRUE python analysis/clean_recheck/diagnostics.py
Outputs (analysis/clean_recheck/diag_runs/): <fold>_model.pt, <fold>_params.json, <fold>_epochs.csv, <fold>_points.npz;
         tables/diag_*.csv  (diagnostics.md is written by hand from these)
"""
import json
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
import torch.nn.functional as F

import clean_recheck as cr
from hyperbolic_intention import IntentionModel, entail_loss, exterior_angle, half_aperture, pad, sample_pair

OUT = HERE / "diag_runs"
SEED, GEOM, LAST = 0, "hyperbolic", 10


def scalars(model):
    return dict(c=model.c().item(), alpha_s=model.log_alpha_set.clamp(max=0).exp().item(),
                alpha_i=model.log_alpha_int.clamp(max=0).exp().item(), tau=model.tau().item(), K=model.K)


def diag_loss(model, full, full_mask, sub, sub_mask, y, ent_weight):
    """loss_fn() of hyperbolic_intention.py (soft_sub=None, set_task_cone=True), term by term."""
    c, tau = model.c(), model.tau()
    i_full, i_sub, f_all = model.embed_sets(full, full_mask), model.embed_sets(sub, sub_mask), model.embed_intents()
    l_if = F.cross_entropy(-model.pdist(i_full, f_all) / tau, y)
    l_sub_f = F.cross_entropy(-model.pdist(i_sub, f_all) / tau, y)
    f = f_all[y]
    logits = -model.pdist(f, i_full) / tau
    pos = y[:, None] == y[None]
    l_fi = -(torch.logsumexp(logits.masked_fill(~pos, -1e9), 1) - torch.logsumexp(logits, 1)).mean()
    l_cont = 0.25 * l_if + 0.25 * l_sub_f + 0.5 * l_fi
    l_ent_if = entail_loss(i_full, f, c, model.K)
    l_ent_sub = entail_loss(i_sub, i_full, c, model.K)
    total = l_cont + ent_weight * l_ent_if + ent_weight * l_ent_sub
    with torch.no_grad():
        v_if = exterior_angle(i_full, f, c) - half_aperture(i_full, c, model.K)
        v_sub = exterior_angle(i_sub, i_full, c) - half_aperture(i_sub, c, model.K)
        sc = torch.sqrt(c)
        info = dict(L_IF=l_if.item(), L_SF=l_sub_f.item(), L_FI=l_fi.item(), L_ent_IF=l_ent_if.item(),
                    L_ent_sub=l_ent_sub.item(), total=total.item(),
                    n_pairs=len(y), pos_ent_IF=int((v_if > 0).sum()), pos_ent_sub=int((v_sub > 0).sum()),
                    clamped_full=int((sc * i_full.norm(dim=-1) <= 2 * model.K).sum()),
                    clamped_sub=int((sc * i_sub.norm(dim=-1) <= 2 * model.K).sum()))
    return total, info


def train_logged(model, train, device, epochs):
    """train_model() of hyperbolic_intention.py with subset="harmonic", no coarse / hierarchy / soft targets."""
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
            loss, info = diag_loss(model, *full, *sub, y, cr.ENT_WEIGHT)
            opt.zero_grad(); loss.backward(); opt.step()
            parts.append(info)
        p = pd.DataFrame(parts)
        row = dict(epoch=ep, **p[["L_IF", "L_SF", "L_FI", "L_ent_IF", "L_ent_sub", "total"]].mean().to_dict())
        n = p.n_pairs.sum()
        row.update(share_pos_ent_IF=p.pos_ent_IF.sum() / n, share_pos_ent_sub=p.pos_ent_sub.sum() / n,
                   share_clamped_full=p.clamped_full.sum() / n, share_clamped_sub=p.clamped_sub.sum() / n,
                   **scalars(model))
        rows.append(row)
    model.eval()
    return pd.DataFrame(rows)


@torch.no_grad()
def points(model, lists, device, bs=2048):
    out = []
    for i in range(0, len(lists), bs):
        out.append(model.embed_sets(*pad(lists[i:i + bs], device)))
    return torch.cat(out)


@torch.no_grad()
def describe(name, x, model):
    c = model.c()
    s = (torch.sqrt(c) * x.norm(dim=-1)).cpu().numpy()
    r = model.dist0(x).cpu().numpy()
    q = lambda v: np.percentile(v, [0, 10, 50, 90, 100]).round(4).tolist()
    return dict(kind=name, n=len(s), sqrt_c_norm_min_p10_med_p90_max=q(s), share_clamped=float((s <= 2 * model.K).mean()),
                radius_min_p10_med_p90_max=q(r)), s, r


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    OUT.mkdir(exist_ok=True)
    narr, blocks, segs, occ, vocab, info = cr.load_data()
    feats = cr.qwen3_features(vocab, device)
    point_rows, scalar_rows, loss_rows, check_rows = [], [], [], []
    for fold in cr.FOLDS:
        # same as clean_recheck.run_one
        train = [dict(s) for s in segs if s["participant"] != fold]
        labels = sorted({s["label"] for s in train})
        for s in train:
            s["y"] = labels.index(s["label"])
        random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
        model = IntentionModel(len(vocab), len(labels), cr.DIM, feats, cr.K_CONE, GEOM).to(device)
        ep = train_logged(model, train, device, 100)
        ep.to_csv(OUT / f"{fold}_epochs.csv", index=False)
        torch.save(model.state_dict(), OUT / f"{fold}_model.pt")
        sc = scalars(model)
        json.dump(dict(fold=fold, seed=SEED, geometry=GEOM, labels=labels, **sc), open(OUT / f"{fold}_params.json", "w"), indent=1)

        # exactness check against the saved clean-recheck run
        r_new = cr.radii(model, [[i] for i in range(len(vocab))], device)
        r_old = np.load(cr.RUNS / f"{fold}_{GEOM}_s{SEED}.npz")["action_radius"]
        check_rows.append(dict(fold=fold, max_abs_diff=float(np.abs(r_new - r_old).max()),
                               spearman=float(cr.spearmanr(r_new, r_old)[0])))

        # points: single actions, full sets and one harmonic subset of every training block (own RNG), intentions
        rng = random.Random(SEED)
        fulls = [s["ids"] for s in train]
        subs = []
        for ids in fulls:
            if len(ids) == 1:
                subs.append(list(ids)); continue
            sizes = range(1, len(ids))
            k = rng.choices(sizes, weights=[1 / k for k in sizes])[0]
            subs.append(rng.sample(ids, k))
        with torch.no_grad():
            sets = dict(single_actions=points(model, [[i] for i in range(len(vocab))], device),
                        subsets=points(model, subs, device), full_sets=points(model, fulls, device),
                        intention_points=model.embed_intents())
        store = {}
        for name, x in sets.items():
            d, s, r = describe(name, x, model)
            point_rows.append(dict(fold=fold, **d))
            store[f"{name}_sqrt_c_norm"], store[f"{name}_radius"] = s, r
        np.savez_compressed(OUT / f"{fold}_points.npz", **store)
        scalar_rows.append(dict(fold=fold, **sc))
        last = ep[ep.epoch > ep.epoch.max() - LAST].mean().drop("epoch")
        loss_rows.append(dict(fold=fold, **last.round(5).to_dict()))
        print(f"{fold}: c {sc['c']:.3f} alpha_s {sc['alpha_s']:.4f} alpha_i {sc['alpha_i']:.4f} tau {sc['tau']:.4f} | "
              f"radius check max diff {check_rows[-1]['max_abs_diff']:.2e} | last-10 L_ent_IF {last.L_ent_IF:.4f} "
              f"L_ent_sub {last.L_ent_sub:.4f} share>0 {last.share_pos_ent_IF:.3f}/{last.share_pos_ent_sub:.3f}", flush=True)
        del model
        torch.cuda.empty_cache()

    cr.TAB.mkdir(exist_ok=True)
    pd.DataFrame(point_rows).to_csv(cr.TAB / "diag_points.csv", index=False)
    pd.DataFrame(scalar_rows).to_csv(cr.TAB / "diag_scalars.csv", index=False)
    pd.DataFrame(loss_rows).to_csv(cr.TAB / "diag_last10_losses.csv", index=False)
    pd.DataFrame(check_rows).to_csv(cr.TAB / "diag_reproduction_check.csv", index=False)


if __name__ == "__main__":
    main()
