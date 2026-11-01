"""
scripts/hyperbolic_intention.py
Hyperbolic co-embedding of intentions and action sets, following Kubota, Shiku & Uchida,
"Hierarchical Co-Embedding of Font Shapes and Impression Tags" (arXiv:2604.04158):

  font               -> intention (learnable point per intent class)
  impression tag set -> the set of actions performed in an intention segment
  tag subset         -> random subset of that action set (lower specificity)

Losses (as in the paper; --coarse adds a coarse-to-fine term, see coarse_loss):
  L = 1/4 L_cont(I->F) + 1/4 L_cont(I~->F) + 1/2 L_cont(F->I)
      + 0.1 L_ent(action set entails intention) + 0.1 L_ent(subset entails set)
Contrastive I->F is a softmax over all intent points (many segments share an intent);
F->I is multi-positive over the batch. Lorentz-model operations follow MERU (Desai et al. 2023).

Action encoder: frozen CLIP text features ("The action is {verb noun}.") -> Transformer with a
learnable aggregation token (paper setup), or --action-enc learned for per-action embeddings.

Evaluation (val / test participants, see build_intention_sets.py):
  - intent top-1 / macro accuracy from the full action set
  - online prefixes: first k distinct actions in temporal order -> accuracy + distance from origin
  - single-action distance from origin vs count-based specificity (1 - normalised entropy of
    P(intent | action) on train); Spearman correlation
  - baselines: majority class, Bernoulli naive Bayes on the same action sets

Run in the `intention` conda env:
  python scripts/hyperbolic_intention.py --action-enc clip
Outputs: scripts/intention_runs/<run name>/
"""
import argparse
import json
import math
import os
import pathlib
import random

# conda MKL numpy/pandas (needed: pip builds are blocked by Smart App Control) and pip torch
# each ship libiomp5md.dll; allow both to load
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA = ROOT / "scripts" / "intention_data"
RUNS = ROOT / "scripts" / "intention_runs"
MAX_SET = 64
PREFIX_KS = [1, 2, 3, 5, 8, 12, 20]


# ---------------------------------------------------------------- Lorentz model (MERU)
def time_comp(x, c):
    return torch.sqrt(1 / c + (x * x).sum(-1))


def expmap0(v, c):
    rc_norm = (torch.sqrt(c) * v.norm(dim=-1, keepdim=True)).clamp_min(1e-8)
    return torch.sinh(rc_norm.clamp(max=math.asinh(2 ** 15))) * v / rc_norm


def pairwise_dist(x, y, c):
    inner = x @ y.T - time_comp(x, c)[:, None] * time_comp(y, c)[None]
    return torch.acosh((-c * inner).clamp_min(1 + 1e-7)) / torch.sqrt(c)


def dist_origin(x, c):
    return torch.acosh((torch.sqrt(c) * time_comp(x, c)).clamp_min(1 + 1e-7)) / torch.sqrt(c)


def half_aperture(x, c, K=0.1):
    return torch.asin((2 * K / (torch.sqrt(c) * x.norm(dim=-1) + 1e-8)).clamp(-1 + 1e-6, 1 - 1e-6))


def exterior_angle(x, y, c):
    # exterior angle at x in the hyperbolic triangle o-x-y
    xt, yt = time_comp(x, c), time_comp(y, c)
    c_xyl = c * ((x * y).sum(-1) - xt * yt)
    num = yt + c_xyl * xt
    den = x.norm(dim=-1) * torch.sqrt((c_xyl ** 2 - 1).clamp_min(1e-8))
    return torch.acos((num / (den + 1e-8)).clamp(-1 + 1e-6, 1 - 1e-6))


def entail_loss(x, y, c, K=0.1):
    """y should lie inside the entailment cone of x."""
    return F.relu(exterior_angle(x, y, c) - half_aperture(x, c, K)).mean()


# ---------------------------------------------------------------- model
class UnitEncoder(nn.Module):
    """Action units with separate detail tokens ("way 2"): unit "take package (lentil)" = [ACT] take package +
    [DETAIL] lentil. Each token = projected frozen text feature + a learned type vector (0 = action, 1 = detail);
    a small Transformer reads the unit's own tokens, and the output at the action token is the unit vector.
    Units without details have only the action token, so the model knows when there is something to narrow by."""

    def __init__(self, token_feats, unit_tokens, dim, heads=4):
        super().__init__()
        self.tok = nn.Embedding.from_pretrained(token_feats, freeze=True)
        self.proj = nn.Linear(token_feats.shape[1], dim)
        self.type_emb = nn.Embedding(2, dim)
        self.register_buffer("unit_tokens", unit_tokens)          # [n_units, 1 + max_details], -1 = padding
        layer = nn.TransformerEncoderLayer(dim, heads, dim * 4, dropout=0.1, batch_first=True)
        self.inner = nn.TransformerEncoder(layer, 1, enable_nested_tensor=False)

    def forward(self):
        t, pad = self.unit_tokens.clamp_min(0), self.unit_tokens < 0
        types = torch.ones_like(t)
        types[:, 0] = 0
        h = self.proj(self.tok(t)) + self.type_emb(types)
        return self.inner(h, src_key_padding_mask=pad)[:, 0]   # [n_units, dim]


class SetEncoder(nn.Module):
    def __init__(self, n_vocab, dim, clip_feats=None, layers=2, heads=4, action_groups=None, units=None):
        super().__init__()
        if clip_feats is not None:
            self.tok = nn.Embedding.from_pretrained(clip_feats, freeze=True)
        else:
            self.tok = nn.Embedding(n_vocab, 512)
        self.proj = nn.Linear(self.tok.embedding_dim, dim)
        # optional action units with detail tokens: units = (token_feats, unit_tokens), see UnitEncoder
        self.unit_enc = UnitEncoder(*units, dim) if units is not None else None
        # optional action groups (e.g. "take containers" for "take cup"): a learned group vector is added to each
        # action, so rare actions share what is learned from the other actions of their group. Starts at zero.
        self.gemb = None
        if action_groups is not None:
            self.register_buffer("act_group", action_groups)
            self.gemb = nn.Embedding(int(action_groups.max()) + 1, dim)
            nn.init.zeros_(self.gemb.weight)
        self.agg = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        layer = nn.TransformerEncoderLayer(dim, heads, dim * 4, dropout=0.1, batch_first=True)
        self.tf = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.out = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim))

    def forward(self, ids, mask):
        h = self.unit_enc()[ids] if self.unit_enc is not None else self.proj(self.tok(ids))
        if self.gemb is not None:
            h = h + self.gemb(self.act_group[ids])
        h = torch.cat([self.agg.expand(len(h), -1, -1), h], 1)
        pad = torch.cat([torch.zeros(len(h), 1, dtype=torch.bool, device=h.device), ~mask], 1)
        return self.out(self.tf(h, src_key_padding_mask=pad)[:, 0])


class IntentionModel(nn.Module):
    def __init__(self, n_vocab, n_intent, dim, clip_feats=None, K=0.1, geometry="hyperbolic", n_groups=0,
                 intent_names=None, group_names=None, action_groups=None, units=None):
        super().__init__()
        # optional: intention / group points start from text embeddings of their descriptions (shared projection +
        # a small per-point offset), so rare intentions begin near related ones instead of at random
        self.register_buffer("intent_names", intent_names if intent_names is not None else torch.zeros(0))
        self.register_buffer("group_names", group_names if group_names is not None else torch.zeros(0))
        self.name_proj = nn.Linear(intent_names.shape[1], dim) if intent_names is not None else None
        # middle level of the intention hierarchy (e.g. "drink" above coffee / cold drink); see hierarchy_loss
        self.group = (nn.Parameter(torch.zeros(n_groups, dim) if group_names is not None else torch.randn(n_groups, dim))
                      if n_groups else None)
        self.K = K
        self.geometry = geometry  # "euclidean" = same encoders, plain vectors + Euclidean distance, no cones
        self.set_enc = SetEncoder(n_vocab, dim, clip_feats, action_groups=action_groups, units=units)
        self.intent = nn.Parameter(torch.zeros(n_intent, dim) if intent_names is not None else torch.randn(n_intent, dim))
        self.log_alpha_set = nn.Parameter(torch.tensor(-0.5 * math.log(dim)))
        self.log_alpha_int = nn.Parameter(torch.tensor(-0.5 * math.log(dim)))
        self.log_c = nn.Parameter(torch.tensor(0.0))
        self.log_tau = nn.Parameter(torch.tensor(math.log(0.07)))

    def c(self):
        return self.log_c.exp().clamp(0.1, 10.0)

    def tau(self):
        return self.log_tau.exp().clamp_min(0.01)

    @property
    def hyperbolic(self):
        return self.geometry == "hyperbolic"

    def embed_sets(self, ids, mask):
        v = self.set_enc(ids, mask) * self.log_alpha_set.clamp(max=0).exp()
        return expmap0(v, self.c()) if self.hyperbolic else v

    def embed_intents(self):
        v = self._from_names(self.intent_names, self.intent) * self.log_alpha_int.clamp(max=0).exp()
        return expmap0(v, self.c()) if self.hyperbolic else v

    def _from_names(self, names, offset):
        if self.name_proj is None or names.numel() == 0:
            return offset
        v = self.name_proj(names)
        return v / v.norm(dim=-1, keepdim=True) * math.sqrt(v.shape[1]) + offset

    def embed_groups(self):
        v = self._from_names(self.group_names, self.group) * self.log_alpha_int.clamp(max=0).exp()
        return expmap0(v, self.c()) if self.hyperbolic else v

    def pdist(self, x, y):
        return pairwise_dist(x, y, self.c()) if self.hyperbolic else torch.cdist(x, y)

    def dist0(self, x):
        return dist_origin(x, self.c()) if self.hyperbolic else x.norm(dim=-1)


def loss_fn(model, full, full_mask, sub, sub_mask, y, ent_weight=0.1, soft_sub=None, set_task_cone=True):
    # set_task_cone=False (Plan B): drop the cone "action set entails its task", which keeps every set more general
    # than (= nearer the origin than) its task; the radius is then set by radius_loss in evaluate_h3.py
    c, tau = model.c(), model.tau()
    i_full, i_sub, f_all = model.embed_sets(full, full_mask), model.embed_sets(sub, sub_mask), model.embed_intents()
    l_if = F.cross_entropy(-model.pdist(i_full, f_all) / tau, y)
    # soft_sub: optional target distributions for the partial sets (vague sets need not name one intention)
    l_sub_f = F.cross_entropy(-model.pdist(i_sub, f_all) / tau, soft_sub if soft_sub is not None else y)
    f = f_all[y]
    logits = -model.pdist(f, i_full) / tau
    pos = y[:, None] == y[None]
    l_fi = -(torch.logsumexp(logits.masked_fill(~pos, -1e9), 1) - torch.logsumexp(logits, 1)).mean()
    l_cont = 0.25 * l_if + 0.25 * l_sub_f + 0.5 * l_fi
    zero = torch.zeros((), device=y.device)
    l_ent_if = entail_loss(i_full, f, c, model.K) if model.hyperbolic and set_task_cone else zero
    l_ent_sub = entail_loss(i_sub, i_full, c, model.K) if model.hyperbolic else zero
    total = l_cont + ent_weight * l_ent_if + ent_weight * l_ent_sub
    return total, dict(cont=l_cont.item(), ent_if=l_ent_if.item(), ent_sub=l_ent_sub.item())


def hierarchy_loss(model, full, full_mask, sub, sub_mask, y, leaf_group, ent_weight):
    """Vague-to-specific extension (Suris et al. 2021 style): intention groups ("drink") sit between the
    origin and their intentions ("coffee", "cold drink"). Each group's cone must contain its intentions,
    and action sets (full and partial) are also pulled towards their group, so a vague partial set can be
    answered at group level. leaf_group[k] = group index of intention k, or -1 if it has no group."""
    tau = model.tau()
    g_all = model.embed_groups()
    yg = leaf_group[y]
    has = yg >= 0
    zero = torch.zeros((), device=y.device)
    l_group = zero
    if has.any():
        for ids, mask in ((full, full_mask), (sub, sub_mask)):
            x = model.embed_sets(ids, mask)[has]
            l_group = l_group + 0.5 * F.cross_entropy(-model.pdist(x, g_all) / tau, yg[has])
    leaves = (leaf_group >= 0).nonzero().squeeze(1)
    l_ent_hier = (entail_loss(g_all[leaf_group[leaves]], model.embed_intents()[leaves], model.c(), model.K)
                  if model.hyperbolic and len(leaves) else zero)
    return 0.25 * l_group + ent_weight * l_ent_hier, dict(group=l_group.item(), ent_hier=l_ent_hier.item())


def coarse_loss(model, full, full_mask, coarse, coarse_mask, differs, y, pair_plain, pair_detail, ent_weight):
    """Coarse-to-fine extension (not in the paper): removing the item details from an action set
    ("take package (lentil)" -> "take package") gives a lower-specificity version, used like the
    paper's tag subsets: it should entail the detailed set and still point to the intention.
    The same relation is also applied to single actions (plain action entails its detailed form)."""
    c, tau = model.c(), model.tau()
    i_full, i_coarse = model.embed_sets(full, full_mask), model.embed_sets(coarse, coarse_mask)
    l_coarse_f = F.cross_entropy(-model.pdist(i_coarse, model.embed_intents()) / tau, y)
    zero = torch.zeros((), device=y.device)
    l_ent_coarse = entail_loss(i_coarse[differs], i_full[differs], c, model.K) if differs.any() and model.hyperbolic else zero
    l_ent_single = zero
    if len(pair_plain) and model.hyperbolic:
        ones = torch.ones(len(pair_plain), 1, dtype=torch.bool, device=y.device)
        x_plain = model.embed_sets(pair_plain[:, None], ones)
        x_detail = model.embed_sets(pair_detail[:, None], ones)
        l_ent_single = entail_loss(x_plain, x_detail, c, model.K)
    total = 0.25 * l_coarse_f + ent_weight * (l_ent_coarse + l_ent_single)
    return total, dict(coarse_f=l_coarse_f.item(), ent_coarse=l_ent_coarse.item(), ent_single=l_ent_single.item())


# ---------------------------------------------------------------- data
def action_text(a):
    # "take maker:coffee" -> "take coffee maker"; "mix pan (butter, pepper)" -> "mix pan with butter, pepper"
    a, _, detail = a.partition(" (")
    verb, noun = a.split(" ", 1)
    text = f"{verb} {' '.join(reversed(noun.split(':')))}"
    if detail:
        text += " with " + ", ".join(" ".join(reversed(d.split(":"))) for d in detail.rstrip(")").split(", "))
    return text


QWEN3_TASK = "Given a kitchen action, identify the cooking or household task the person is doing"


def text_features(vocab, device, data=DATA, encoder="clip"):
    """Frozen text features for each action phrase, cached per dataset folder.
    clip   CLIP ViT-B/32 text tower, prompt "The action is {phrase}." (the paper's setup)
    qwen3  Qwen3-Embedding-0.6B, instruction prompt, last-token pooling (a sentence-embedding model)"""
    cache = data / f"{encoder}_action_feats.pt"
    if cache.exists():
        saved = torch.load(cache)
        if saved["vocab"] == vocab:
            return saved["feats"]
    if encoder == "clip":
        from transformers import CLIPTextModelWithProjection, CLIPTokenizer
        name = "openai/clip-vit-base-patch32"
        tok, enc = CLIPTokenizer.from_pretrained(name), CLIPTextModelWithProjection.from_pretrained(name).to(device).eval()
        prompts = [f"The action is {action_text(a)}." for a in vocab]
        with torch.no_grad():
            feats = enc(**tok(prompts, padding=True, return_tensors="pt").to(device)).text_embeds.float().cpu()
    elif encoder == "qwen3":
        from transformers import AutoModel, AutoTokenizer
        name = "Qwen/Qwen3-Embedding-0.6B"
        tok = AutoTokenizer.from_pretrained(name, padding_side="left")
        enc = AutoModel.from_pretrained(name, dtype=torch.float16).to(device).eval()
        prompts = [f"Instruct: {QWEN3_TASK}\nQuery: {action_text(a)}" for a in vocab]
        out = []
        with torch.no_grad():
            for i in range(0, len(prompts), 256):
                batch = tok(prompts[i:i + 256], padding=True, truncation=True, max_length=64, return_tensors="pt").to(device)
                out.append(enc(**batch).last_hidden_state[:, -1].float().cpu())  # left padding: last token = EOS
        feats = torch.cat(out)
    else:
        raise ValueError(encoder)
    feats = F.normalize(feats, dim=-1)
    torch.save(dict(vocab=vocab, feats=feats), cache)
    return feats


def unit_tokens(vocab, max_details=3):
    """Split units "verb noun (item, item)" into tokens: returns (token strings, kinds, [n_units, 1+max_details]
    index tensor with -1 padding). Kinds: "action" for plain actions, "item" for details."""
    plain = sorted({u.split(" (")[0] for u in vocab})
    items = sorted({i for u in vocab if " (" in u for i in u.split(" (", 1)[1].rstrip(")").split(", ")})
    tokens, kinds = plain + items, ["action"] * len(plain) + ["item"] * len(items)
    plain_index = {t: i for i, t in enumerate(plain)}
    item_index = {t: len(plain) + i for i, t in enumerate(items)}
    rows = []
    for u in vocab:
        p, _, rest = u.partition(" (")
        det = [item_index[i] for i in rest.rstrip(")").split(", ")] if rest else []
        rows.append(([plain_index[p]] + det + [-1] * max_details)[:1 + max_details])
    return tokens, kinds, torch.tensor(rows)


@torch.no_grad()
def token_features(tokens, kinds, device, data):
    """Frozen Qwen3 features for action tokens ("take coffee maker") and item tokens ("coffee maker"), cached."""
    cache = data / "qwen3_token_feats.pt"
    if cache.exists():
        saved = torch.load(cache)
        if saved["tokens"] == tokens:
            return saved["feats"]
    from transformers import AutoModel, AutoTokenizer
    name = "Qwen/Qwen3-Embedding-0.6B"
    tok = AutoTokenizer.from_pretrained(name, padding_side="left")
    enc = AutoModel.from_pretrained(name, dtype=torch.float16).to(device).eval()
    texts = [action_text(t) if k == "action" else " ".join(reversed(t.split(":"))) for t, k in zip(tokens, kinds)]
    prompts = [f"Instruct: {QWEN3_TASK}\nQuery: {x}" for x in texts]
    out = []
    for i in range(0, len(prompts), 256):
        batch = tok(prompts[i:i + 256], padding=True, truncation=True, max_length=64, return_tensors="pt").to(device)
        out.append(enc(**batch).last_hidden_state[:, -1].float().cpu())
    feats = F.normalize(torch.cat(out), dim=-1)
    torch.save(dict(tokens=tokens, feats=feats), cache)
    return feats


@torch.no_grad()
def name_features(texts, device):
    """Qwen3-Embedding of label descriptions (document style, no instruction), L2-normalised."""
    from transformers import AutoModel, AutoTokenizer
    name = "Qwen/Qwen3-Embedding-0.6B"
    tok = AutoTokenizer.from_pretrained(name, padding_side="left")
    enc = AutoModel.from_pretrained(name, dtype=torch.float16).to(device).eval()
    batch = tok(list(texts), padding=True, truncation=True, max_length=64, return_tensors="pt").to(device)
    return F.normalize(enc(**batch).last_hidden_state[:, -1].float(), dim=-1)


def clip_features(vocab, device, data=DATA):
    return text_features(vocab, device, data, "clip")


def load_data(data=DATA):
    segs = [json.loads(l) for l in open(data / "segments.jsonl", encoding="utf-8")]
    vocab = list(json.load(open(data / "vocab.json", encoding="utf-8")))
    idx = {a: i for i, a in enumerate(vocab)}
    intents = sorted({s["intent"] for s in segs if s["split"] == "train"})
    for s in segs:
        s["ids"] = [idx[a] for a in s["action_set"]]
        seen, order = set(), []
        for a in s["actions"]:
            if a["action"] in idx and a["action"] not in seen:
                seen.add(a["action"])
                order.append(idx[a["action"]])
        s["order"] = order
        s["seq"] = [idx[a["action"]] for a in s["actions"] if a["action"] in idx]  # time order, with repeats
    segs = [s for s in segs if s["intent"] in intents]
    # plain form of each action ("take package (lentil)" -> "take package"), itself if it has no detail
    plain = [idx.get(a.split(" (")[0], i) for i, a in enumerate(vocab)]
    for s in segs:
        s["plain_of"] = plain
    return segs, vocab, intents


def pad(lists, device):
    L = max(len(x) for x in lists)
    ids = torch.zeros(len(lists), L, dtype=torch.long)
    mask = torch.zeros(len(lists), L, dtype=torch.bool)
    for i, x in enumerate(lists):
        ids[i, :len(x)] = torch.tensor(x)
        mask[i, :len(x)] = True
    return ids.to(device), mask.to(device)


def sample_pair(ids, subset="uniform", seq=None, max_window=8):
    full = random.sample(ids, MAX_SET) if len(ids) > MAX_SET else list(ids)
    if len(full) == 1:
        return full, list(full)
    if subset == "window" and seq:  # consecutive actions as they happened; short windows more often
        widths = range(1, min(len(seq), max_window) + 1)
        w = random.choices(widths, weights=[1 / k for k in widths])[0]
        start = random.randrange(len(seq) - w + 1)
        return full, list(dict.fromkeys(seq[start:start + w]))
    sizes = range(1, len(full))
    if subset == "uniform":  # any subset size equally likely (the paper does not state its sampling)
        k = random.choice(sizes)
    else:  # "harmonic": P(k) ~ 1/k, so single actions and small sets are seen often
        k = random.choices(sizes, weights=[1 / k for k in sizes])[0]
    return full, random.sample(full, k)


# ---------------------------------------------------------------- evaluation
@torch.no_grad()
def embed_lists(model, lists, device, bs=256):
    out = []
    for i in range(0, len(lists), bs):
        out.append(model.embed_sets(*pad(lists[i:i + bs], device)))
    return torch.cat(out)


@torch.no_grad()
def predict(model, lists, device):
    x = embed_lists(model, lists, device)
    d = model.pdist(x, model.embed_intents())
    return d.argmin(1).cpu().numpy(), model.dist0(x).cpu().numpy()


def accuracy(pred, y, n_cls):
    acc = float((pred == y).mean())
    per = [float((pred[y == k] == k).mean()) for k in range(n_cls) if (y == k).any()]
    return acc, float(np.mean(per))


class NaiveBayes:
    """Bernoulli naive Bayes over action presence (count baseline)."""

    def __init__(self, train, n_vocab, n_cls, alpha=1.0):
        X = np.zeros((len(train), n_vocab))
        for i, s in enumerate(train):
            X[i, s["ids"]] = 1
        y = np.array([s["y"] for s in train])
        self.log_prior = np.log(np.bincount(y, minlength=n_cls) + alpha) - np.log(len(y) + alpha * n_cls)
        p = np.stack([(X[y == k].sum(0) + alpha) / ((y == k).sum() + 2 * alpha) for k in range(n_cls)])
        self.log_p, self.log_q = np.log(p), np.log(1 - p)

    def predict(self, lists):
        out = []
        for ids in lists:
            x = np.zeros(self.log_p.shape[1])
            x[ids] = 1
            out.append(np.argmax(self.log_prior + (x * self.log_p + (1 - x) * self.log_q).sum(1)))
        return np.array(out)


def evaluate(model, nb, segs, n_cls, device):
    y = np.array([s["y"] for s in segs])
    res = {}
    pred, d0 = predict(model, [s["ids"] for s in segs], device)
    res["full_set"] = dict(n=len(segs), acc_macro=accuracy(pred, y, n_cls), nb_acc_macro=accuracy(nb.predict([s["ids"] for s in segs]), y, n_cls),
                           mean_dist_origin=float(d0.mean()))
    res["prefix"] = {}
    for k in PREFIX_KS:
        sub = [s for s in segs if len(s["order"]) >= k]
        if not sub:
            continue
        lists, yk = [s["order"][:k] for s in sub], np.array([s["y"] for s in sub])
        pk, dk = predict(model, lists, device)
        res["prefix"][k] = dict(n=len(sub), acc_macro=accuracy(pk, yk, n_cls), nb_acc_macro=accuracy(nb.predict(lists), yk, n_cls),
                                mean_dist_origin=float(dk.mean()))
    return res


@torch.no_grad()
def detail_refinement(model, segs, plain_of, detail_pairs, device):
    """Adding item details: does the point move out, the cone narrow, and the detailed version
    fall inside the plain version's cone? For single actions and for whole segment sets."""
    if not detail_pairs or not model.hyperbolic:
        return None
    c, out = model.c(), {}
    set_pairs = [(sorted({plain_of[i] for i in s["ids"]}), s["ids"]) for s in segs
                 if any(plain_of[i] != i for i in s["ids"])]
    for level, plain, detail in (("single", [[a] for a, _ in detail_pairs], [[b] for _, b in detail_pairs]),
                                 ("set", [a for a, _ in set_pairs], [b for _, b in set_pairs])):
        xp, xd = embed_lists(model, plain, device), embed_lists(model, detail, device)
        ap, ad = half_aperture(xp, c, model.K), half_aperture(xd, c, model.K)
        out[level] = dict(n=len(xp), further_out=float((dist_origin(xd, c) > dist_origin(xp, c)).float().mean()),
                          narrower_cone=float((ad < ap - 1e-4).float().mean()),
                          inside_plain_cone=float((exterior_angle(xp, xd, c) <= ap).float().mean()),
                          mean_cone_deg=[float(np.degrees(ap.mean().item())), float(np.degrees(ad.mean().item()))])
    return out


def count_specificity(train, vocab, n_cls):
    counts = np.zeros((len(vocab), n_cls))
    for s in train:
        counts[s["ids"], s["y"]] += 1
    p = counts / counts.sum(1, keepdims=True)
    h = -(np.where(p > 0, p * np.log(p + 1e-12), 0)).sum(1)
    return 1 - h / math.log(n_cls), counts.sum(1)


# ---------------------------------------------------------------- training
def train_model(model, train, device, epochs=150, batch_size=32, lr=1e-4, subset="uniform", ent_weight=0.1,
                coarse=False, plain_of=None, detail_pairs=(), on_epoch=None, leaf_group=None, soft_fn=None):
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=0.01)
    train = list(train)
    log = []
    for ep in range(1, epochs + 1):
        model.train()
        random.shuffle(train)
        parts = []
        for i in range(0, len(train), batch_size):
            batch = train[i:i + batch_size]
            pairs = [sample_pair(s["ids"], subset, s.get("seq")) for s in batch]
            full, sub = pad([p[0] for p in pairs], device), pad([p[1] for p in pairs], device)
            y = torch.tensor([s["y"] for s in batch], device=device)
            soft = soft_fn([p[1] for p in pairs], y) if soft_fn else None
            loss, info = loss_fn(model, *full, *sub, y, ent_weight, soft)
            if coarse:
                coarse_lists = [sorted({plain_of[i] for i in p[0]}) for p in pairs]
                differs = torch.tensor([cl != sorted(p[0]) for cl, p in zip(coarse_lists, pairs)], device=device)
                pp = random.sample(detail_pairs, min(64, len(detail_pairs)))
                pair_plain = torch.tensor([a for a, _ in pp], device=device, dtype=torch.long)
                pair_detail = torch.tensor([b for _, b in pp], device=device, dtype=torch.long)
                l2, info2 = coarse_loss(model, *full, *pad(coarse_lists, device), differs, y, pair_plain, pair_detail,
                                        ent_weight)
                loss = loss + l2
                info.update(info2)
            if leaf_group is not None:
                l3, info3 = hierarchy_loss(model, *full, *sub, y, leaf_group, ent_weight)
                loss = loss + l3
                info.update(info3)
            opt.zero_grad(); loss.backward(); opt.step()
            parts.append(dict(loss=loss.item(), **info))
        row = dict(epoch=ep, **pd.DataFrame(parts).mean().round(4).to_dict(), c=round(model.c().item(), 3),
                   tau=round(model.tau().item(), 4))
        if on_epoch:
            on_epoch(row)
        log.append(row)
    model.eval()
    return log


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--action-enc", choices=["clip", "qwen3", "learned"], default="clip")
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--K", type=float, default=0.1, help="cone aperture constant (MERU default 0.1)")
    ap.add_argument("--subset", choices=["uniform", "harmonic", "window"], default="uniform",
                    help="lower-specificity sets: random subsets (uniform), small subsets more often, or consecutive windows")
    ap.add_argument("--ent-weight", type=float, default=0.1, help="weight of each entailment loss (paper: 0.1)")
    ap.add_argument("--coarse", action="store_true",
                    help="coarse-to-fine: plain actions/sets must entail their item-detailed versions (detail data only)")
    ap.add_argument("--geometry", choices=["hyperbolic", "euclidean"], default="hyperbolic")
    ap.add_argument("--name", default=None)
    ap.add_argument("--data", default="intention_data", help="dataset folder under scripts/ (e.g. intention_data_detail)")
    args = ap.parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    run = RUNS / (args.name or f"{args.action_enc}_seed{args.seed}")
    run.mkdir(parents=True, exist_ok=True)

    data = ROOT / "scripts" / args.data
    segs, vocab, intents = load_data(data)
    for s in segs:
        s["y"] = intents.index(s["intent"])
    train = [s for s in segs if s["split"] == "train"]
    val = [s for s in segs if s["split"] == "val"]
    plain_of = segs[0]["plain_of"]
    detail_pairs = [(plain_of[i], i) for i in range(len(vocab)) if plain_of[i] != i]
    test = [s for s in segs if s["split"] == "test"]
    print(f"train {len(train)} | val {len(val)} | test {len(test)} | vocab {len(vocab)} | intents {intents}")

    feats = text_features(vocab, device, data, args.action_enc) if args.action_enc != "learned" else None
    model = IntentionModel(len(vocab), len(intents), args.dim, feats, args.K, args.geometry).to(device)
    nb = NaiveBayes(train, len(vocab), len(intents))

    def on_epoch(row):
        if (row["epoch"] % 10 == 0 or row["epoch"] == args.epochs) and val:
            model.eval()
            row["val_acc"], row["val_macro"] = evaluate(model, nb, val, len(intents), device)["full_set"]["acc_macro"]
            print(row)

    log = train_model(model, train, device, args.epochs, args.batch_size, args.lr, args.subset, args.ent_weight,
                      args.coarse, plain_of, detail_pairs, on_epoch)
    pd.DataFrame(log).to_csv(run / "train_log.csv", index=False)

    model.eval()
    metrics = dict(args=vars(args), intents=intents,
                   majority_class=intents[np.bincount([s["y"] for s in train]).argmax()],
                   val=evaluate(model, nb, val, len(intents), device) if val else None,
                   test=evaluate(model, nb, test, len(intents), device))
    y_test = np.array([s["y"] for s in test])
    metrics["test"]["majority_acc"] = float((y_test == np.bincount([s["y"] for s in train]).argmax()).mean())

    # single-action specificity: hyperbolic distance from origin vs count-based
    _, d_single = predict(model, [[i] for i in range(len(vocab))], device)
    spec, n_train = count_specificity(train, vocab, len(intents))
    pred_single, _ = predict(model, [[i] for i in range(len(vocab))], device)
    sa = pd.DataFrame(dict(action=vocab, dist_origin=d_single, count_specificity=spec, n_train_segments=n_train,
                           nearest_intent=[intents[k] for k in pred_single])).sort_values("dist_origin")
    sa.to_csv(run / "action_specificity.csv", index=False)
    ok = sa.n_train_segments >= 5
    R = sa[ok][["dist_origin", "count_specificity"]].rank()
    logn = np.log(sa[ok].n_train_segments).rank()
    resid = lambda v: v - np.polyval(np.polyfit(logn, v, 1), logn)
    # partial correlation: frequent actions are both "less specific" by counting and pulled to the origin
    metrics["single_action_spearman"] = dict(n=int(ok.sum()), rho=float(R.dist_origin.corr(R.count_specificity)),
                                             rho_given_frequency=float(np.corrcoef(resid(R.dist_origin), resid(R.count_specificity))[0, 1]),
                                             rho_dist_frequency=float(R.dist_origin.corr(logn)))
    metrics["cone_saturated_share"] = float((np.degrees(half_aperture(embed_lists(model, [[i] for i in range(len(vocab))], device), model.c(), model.K).detach().cpu().numpy()) > 89).mean()) if model.hyperbolic else None
    metrics["detail_refinement"] = detail_refinement(model, segs, plain_of, detail_pairs, device)
    json.dump(metrics, open(run / "metrics.json", "w"), indent=1)
    torch.save(model.state_dict(), run / "model.pt")

    # plot: online prefix accuracy + distance from origin
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for split, style in (("val", "--"), ("test", "-")):
        if not metrics[split]:
            continue
        pr = metrics[split]["prefix"]
        ks = sorted(pr)
        ax[0].plot(ks, [pr[k]["acc_macro"][0] for k in ks], "o" + style, label=f"hyperbolic ({split})")
        ax[0].plot(ks, [pr[k]["nb_acc_macro"][0] for k in ks], "s" + style, alpha=0.6, label=f"naive Bayes ({split})")
        ax[1].plot(ks, [pr[k]["mean_dist_origin"] for k in ks], "o" + style, label=split)
    ax[0].set(xlabel="first k distinct actions observed", ylabel="intent top-1 accuracy", xscale="log")
    ax[1].set(xlabel="first k distinct actions observed", ylabel="mean distance from origin", xscale="log")
    for a in ax:
        a.legend(); a.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(run / "prefix_curves.png", dpi=120)

    t = metrics["test"]
    print(f"\nTEST full set: acc {t['full_set']['acc_macro'][0]:.3f} (macro {t['full_set']['acc_macro'][1]:.3f}) | "
          f"naive Bayes {t['full_set']['nb_acc_macro'][0]:.3f} | majority {t['majority_acc']:.3f}")
    for k, r in t["prefix"].items():
        print(f"  first {k:2d} actions (n={r['n']:3d}): acc {r['acc_macro'][0]:.3f}  NB {r['nb_acc_macro'][0]:.3f}  dist_origin {r['mean_dist_origin']:.3f}")
    sp = metrics["single_action_spearman"]
    print(f"single-action dist-from-origin vs count specificity: Spearman {sp['rho']:.3f}, given frequency {sp['rho_given_frequency']:.3f}, "
          f"dist vs frequency {sp['rho_dist_frequency']:.3f} (n={sp['n']}) | cones saturated at 90deg: {metrics['cone_saturated_share']}")
    dr = metrics["detail_refinement"]
    if dr:
        for level, r in dr.items():
            print(f"adding item details ({level}, n={r['n']}): further out {r['further_out']:.0%} | narrower cone "
                  f"{r['narrower_cone']:.0%} | inside plain cone {r['inside_plain_cone']:.0%} | mean cone "
                  f"{r['mean_cone_deg'][0]:.0f} -> {r['mean_cone_deg'][1]:.0f} deg")
    print("\nnearest origin:", sa.head(10)[["action", "dist_origin", "nearest_intent"]].to_string(index=False))
    print("\nfarthest:", sa.tail(10)[["action", "dist_origin", "nearest_intent"]].to_string(index=False))
    print(f"\nsaved to {run}")


if __name__ == "__main__":
    main()
