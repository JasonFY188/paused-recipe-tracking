"""
analysis/recipe_groups/recipe_groups.py
Steps 2-4 of the recipe-group task (report.md): item profiles per recipe, hierarchical grouping, comparison.
HD-EPIC data only, no LLM, no hyperbolic model. Every run is done WITH and WITHOUT the phrase corrections.

Decisions (who decided: see report.md):
  kinds            tables/step1_proposed_kinds.csv (Claude's proposal, accepted by the user; packaging = tool)
  corrections      CORRECTIONS below: user's rule (kind-changing, >= 30 mentions in recipe blocks, exact phrase) applied
                   to the 10 listed errors + "food bin"; target classes chosen by Claude
  fetchable        tool (V1, V2) or tool + ingredient (V3), never fixture / cleaning / other, never water / food / mixture
  recipe scope     narrations whose start, and movements whose start, lie in one of the recipe's activity blocks
  object names     strip digits; split "X of Y" into X and Y; each part matched to a noun class (see match_part)
  forms            raw counts | normalised = tf-idf (log(1 + count) * log(N / df)), then each person's mean removed
  grouping         cosine distance, average linkage, cut at k = 2, 3, 4, 5, 6, 8
  V3 source        whichever of V1 / V2 has the higher mean fetch F1 over both forms and all k (with corrections),
                   + the recipe's listed ingredients (count + 1)

Run (intention env): KMP_DUPLICATE_LIB_OK=TRUE python analysis/recipe_groups/recipe_groups.py
Outputs: analysis/recipe_groups/{corrected,uncorrected}/..., tables/, figures/
"""
import ast
import glob
import json
import pathlib
import re
from collections import Counter, defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from scipy.spatial.distance import pdist
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

ROOT = pathlib.Path(__file__).resolve().parents[2]
ANN = ROOT / "annotations"
OUT = ROOT / "analysis" / "recipe_groups"
TAB = OUT / "tables"
VIDEO_ID_FIXES = {"P09-2024623-154547": "P09-20240623-154547"}
KS = [2, 3, 4, 5, 6, 8]
SEED, N_RANDOM = 0, 50
NOT_FETCHABLE = {"water", "food", "mixture"}
# exact phrase (lower case, stripped) -> corrected noun class; user's rule: kind-changing, >= 30 mentions, exact phrase
CORRECTIONS = {
    "pieces": "food", "smaller pieces": "food", "piece": "food",
    "pressure cooker": "cooker:slow",
    "oven glove": "glove", "oven gloves": "glove",
    "slice": "food", "thin slices": "food",
    "food processor": "processor:food", "potato masher": "masher", "hand towel": "towel", "dish sponge": "sponge",
    "cheese grater": "grater", "potato peeler": "peeler:potato", "food bin": "bin",
}


# ------------------------------------------------------------------ loading
def load_blocks():
    dur = pd.read_csv(ANN / "youtube-links" / "HD_EPIC_YouTube_URLs.csv").set_index("video_id").duration
    b = pd.concat([pd.read_csv(f, dtype=str) for f in sorted(glob.glob(str(ANN / "high-level" / "activities" / "*.csv")))],
                  ignore_index=True)
    b["video_id"] = b.video_id.replace(VIDEO_ID_FIXES)
    b["s"] = b.start_time.astype(float)
    b["e"] = pd.to_numeric(b.end_time, errors="coerce")
    b.loc[b.e.isna(), "e"] = b.video_id.map(dur)
    b = b[b.recipe_id.notna() & (b.recipe_id.str.strip() != "")]
    rows = []
    for r in b.itertuples():
        for rid in r.recipe_id.split(","):
            rows.append(dict(recipe=f"{r.video_id[:3]}_{rid.strip()}", video_id=r.video_id, s=r.s, e=r.e))
    return pd.DataFrame(rows)


def noun_tables():
    nc = pd.read_csv(ANN / "narrations-and-action-segments" / "HD_EPIC_noun_classes.csv")
    key_of = dict(zip(nc.id, nc.key))
    phrase = {}
    conflicts = 0
    for k in nc.key:                                                   # class keys: "board:chopping" -> "chopping board"
        phrase.setdefault(" ".join(reversed(k.split(":"))), k)
    for k, inst in zip(nc.key, nc.instances):
        for i in ast.literal_eval(inst):
            p = " ".join(reversed(i.split(":"))) if ":" in i else i
            p = p.lower().strip()
            if p in phrase and phrase[p] != k:
                conflicts += 1
                continue                                               # first class (lowest id) keeps the phrase
            phrase[p] = k
    return key_of, phrase, conflicts


def singulars(w):
    out = [w]
    if w.endswith("ies"):
        out.append(w[:-3] + "y")
    if w.endswith("es"):
        out.append(w[:-2])
    if w.endswith("s"):
        out.append(w[:-1])
    return out


# fixed word lists for object names (Claude-made, listed in the report)
TRAILING_MODIFIERS = {"grey", "gray", "black", "white", "red", "blue", "green", "yellow", "pink", "purple", "brown",
                      "silver", "first", "second", "third", "fourth", "fifth", "one", "two", "three", "big", "small",
                      "large", "left", "right", "new", "old", "top", "bottom", "whole", "unknown", "together"}
QUANTITY_WORDS = {"piece", "pieces", "slice", "slices", "half", "halves", "bunch", "handful", "rest", "portion",
                  "portions", "pair", "bit", "bits", "roll", "set", "stack", "pile", "pinch", "some", "lot"}
SKIPPED = re.compile(r"^track\s*\(?\s*skipped\s*\)?$")


def make_matcher(phrase, corrections):
    """free-text object name -> list of (part, noun class or None).
    digits removed; HD-EPIC's "Track N (skipped)" placeholders -> no item; "X of Y" -> parts X and Y, but X is dropped
    when it is only a quantity word ("half of lemon" -> lemon); each part: trailing modifiers stripped ("frying pan
    grey" -> "frying pan"), then the corrections, then the longest word suffix found among the HD-EPIC class phrases
    ("wooden spoon", then "spoon"), with simple singular forms."""
    cache = {}

    def match_part(part):
        words = part.split()
        while len(words) > 1 and words[-1] in TRAILING_MODIFIERS:
            words = words[:-1]
        cands = [" ".join(words[i:]) for i in range(len(words))]
        for cand in cands:
            for c in singulars(cand):
                if c in corrections:
                    return corrections[c]
                if c in phrase:
                    return phrase[c]
        return None

    def match(name):
        if name in cache:
            return cache[name]
        n = re.sub(r"\d+", "", name.lower().replace("_", " "))
        n = re.sub(r"[^a-z' ]", " ", n)
        n = re.sub(r"\s+", " ", n).strip()
        if SKIPPED.match(n):
            cache[name] = []
            return []
        parts = [p.strip() for p in n.split(" of ") if p.strip()]
        if len(parts) > 1:
            parts = [p for i, p in enumerate(parts)
                     if i == len(parts) - 1 or not set(p.split()) <= QUANTITY_WORDS | TRAILING_MODIFIERS | {"a", "an", "the", "more", "another"}]
        cache[name] = [(p, match_part(p)) for p in parts]
        return cache[name]
    return match


# ------------------------------------------------------------------ item counts per recipe
def narration_counts(blocks, key_of, corrections):
    d = pd.read_pickle(ANN / "narrations-and-action-segments" / "HD_EPIC_Narrations.pkl")
    by_v = {v: g for v, g in d.groupby("video_id")}
    counts = defaultdict(Counter)
    n_corrected = Counter()
    seen = set()
    for r in blocks.itertuples():
        g = by_v.get(r.video_id)
        if g is None:
            continue
        g = g[(g.start_timestamp >= r.s) & (g.start_timestamp < r.e)]
        for idx, nouns, classes in zip(g.index, g.nouns, g.noun_classes):
            if (r.recipe, idx) in seen:                                 # overlapping blocks of the same recipe
                continue
            seen.add((r.recipe, idx))
            for p, c in zip(nouns, classes):
                k = key_of[c]
                pk = p.strip().lower()
                if pk in corrections:
                    n_corrected[(pk, k, corrections[pk])] += 1
                    k = corrections[pk]
                counts[r.recipe][k] += 1
    return counts, n_corrected


def movement_counts(blocks, match):
    a = json.load(open(ANN / "scene-and-object-movements" / "assoc_info.json"))
    by_v = {v: g for v, g in blocks.groupby("video_id")}
    counts = defaultdict(Counter)
    unmatched, n_tracks, n_in = Counter(), 0, 0
    for vid, assocs in a.items():
        g = by_v.get(vid)
        for v in assocs.values():
            for t in v["tracks"]:
                n_tracks += 1
                if g is None:
                    continue
                st = float(t["time_segment"][0])
                recs = set(g[(g.s <= st) & (st < g.e)].recipe)
                if not recs:
                    continue
                n_in += 1
                for part, k in match(v["name"]):
                    if k is None:
                        unmatched[(v["name"], part)] += 1
                        continue
                    for rec in recs:
                        counts[rec][k] += 1
    return counts, unmatched, n_tracks, n_in


def listed_ingredients(recipes_json, match, kind):
    out, unmatched = defaultdict(set), Counter()
    for rec, r in recipes_json.items():
        for cap in r["captures"]:
            for ing in cap["ingredients"].values():
                for part, k in match(ing["name"]):
                    if k is None:
                        unmatched[(ing["name"], part)] += 1
                    elif kind.get(k) == "ingredient":
                        out[rec].add(k)
    return out, unmatched


# ------------------------------------------------------------------ profiles, grouping, evaluation
def matrix(counts, recipes, items):
    return np.array([[counts[r].get(i, 0) for i in items] for r in recipes], float)


def normalise(X, person):
    df = (X > 0).sum(0)
    idf = np.log(len(X) / np.maximum(df, 1))
    T = np.log1p(X) * idf
    for p in set(person):
        T[person == p] -= T[person == p].mean(0)
    return T


def cluster(P):
    D = pdist(P, "cosine")
    D = np.nan_to_num(D, nan=1.0)                                      # empty profiles: distance 1 to all
    return linkage(D, "average")


def cut(Z, k):
    return fcluster(Z, k, "maxclust")


def ranked(lists_counts):
    """lists_counts: list of Counters (other recipes' fetchable item counts) -> items ranked by #recipes, total, name"""
    nrec, tot = Counter(), Counter()
    for c in lists_counts:
        for i, n in c.items():
            nrec[i] += 1
            tot[i] += n
    order = sorted(nrec, key=lambda i: (-nrec[i], -tot[i], i))
    return order, nrec


def fetch_scores(members, truth):
    """members: Counters of the other recipes used for the list; truth: set of items moved in the held-out recipe"""
    if not members:
        return dict(P=0.0, R=0.0, F1=0.0, P5=0.0, P10=0.0, listlen=0)
    order, nrec = ranked(members)
    lst = {i for i in order if nrec[i] >= 0.5 * len(members)}
    tp = len(lst & truth)
    P = tp / len(lst) if lst else 0.0
    R = tp / len(truth)
    F1 = 2 * P * R / (P + R) if P + R else 0.0
    return dict(P=P, R=R, F1=F1, P5=sum(i in truth for i in order[:5]) / 5, P10=sum(i in truth for i in order[:10]) / 10,
                listlen=len(lst))


def evaluate(labels, recipes, list_counts, truth_sets, rng):
    rows = []
    idx = {r: i for i, r in enumerate(recipes)}
    for r in recipes:
        truth = truth_sets.get(r, set())
        if not truth:
            continue
        others = [o for o in recipes if o != r]
        group = [o for o in others if labels[idx[o]] == labels[idx[r]]]
        g = fetch_scores([list_counts[o] for o in group], truth)
        gl = fetch_scores([list_counts[o] for o in others], truth)
        rand = [fetch_scores([list_counts[o] for o in rng.choice(others, len(group), replace=False)], truth)
                for _ in range(N_RANDOM)] if group else [fetch_scores([], truth)]
        rd = {k: float(np.mean([x[k] for x in rand])) for k in g}
        rows.append(dict(recipe=r, group_size=len(group) + 1, **{f"group_{k}": v for k, v in g.items()},
                         **{f"global_{k}": v for k, v in gl.items()}, **{f"random_{k}": v for k, v in rd.items()}))
    return pd.DataFrame(rows)


def stability(build_profile, recipes, person, k, form):
    full = cut(cluster(build_profile(recipes, form)), k)
    aris = []
    for p in sorted(set(person)):
        keep = [r for r, pp in zip(recipes, person) if pp != p]
        lab = cut(cluster(build_profile(keep, form)), k)
        aris.append(adjusted_rand_score(full[[recipes.index(r) for r in keep]], lab))
    return float(np.mean(aris))


def distinctive(labels, recipes, list_counts, n=8):
    pres = {r: set(list_counts[r]) for r in recipes}
    allshare = Counter(i for r in recipes for i in pres[r])
    out = {}
    for g in sorted(set(labels)):
        mem = [r for r, l in zip(recipes, labels) if l == g]
        share = Counter(i for r in mem for i in pres[r])
        score = {i: share[i] / len(mem) - allshare[i] / len(recipes) for i in share}
        out[int(g)] = [f"{i} ({share[i]}/{len(mem)})" for i in sorted(score, key=lambda i: -score[i])[:n]]
    return out


def main():
    TAB.mkdir(parents=True, exist_ok=True)
    kinds = pd.read_csv(TAB / "step1_proposed_kinds.csv")
    kind = dict(zip(kinds.cls, kinds.kind))
    recipes_json = json.load(open(ANN / "high-level" / "complete_recipes.json", encoding="utf-8"))
    recipes = sorted(recipes_json)
    person = np.array([r[:3] for r in recipes])
    dup_pp = Counter((recipes_json[r]["name"], r[:3]) for r in recipes)   # same name, same person: add the recipe id
    label = {r: f"{recipes_json[r]['name']} ({r[:3]}" + (f", {r[4:]})" if dup_pp[(recipes_json[r]['name'], r[:3])] > 1 else ")")
             for r in recipes}
    dup = Counter(recipes_json[r]["name"] for r in recipes)
    same_name = {n: [r for r in recipes if recipes_json[r]["name"] == n] for n, c in dup.items() if c > 1}
    blocks = load_blocks()
    key_of, phrase, conflicts = noun_tables()
    info = dict(recipes=len(recipes), same_name_recipes=same_name, phrase_conflicts_in_class_instances=conflicts)

    fetch_ok = lambda i, kinds_: kind.get(i) in kinds_ and i not in NOT_FETCHABLE
    summary = []
    for corr_name, corr in (("corrected", CORRECTIONS), ("uncorrected", {})):
        out = OUT / corr_name
        (out / "figures").mkdir(parents=True, exist_ok=True)
        match = make_matcher(phrase, corr)
        narr, n_corr = narration_counts(blocks, key_of, corr)
        move, unmatched, n_tracks, n_in = movement_counts(blocks, match)
        listed, unmatched_ing = listed_ingredients(recipes_json, match, kind)
        info[corr_name] = dict(movement_tracks=n_tracks, tracks_in_recipe_blocks=n_in,
                               corrections_applied={f"{p} | {a} -> {b}": n for (p, a, b), n in n_corr.items()},
                               unmatched_movement_parts=sum(unmatched.values()), unmatched_listed_parts=sum(unmatched_ing.values()))
        pd.DataFrame([dict(name=n, part=p, tracks=c) for (n, p), c in unmatched.most_common()]).to_csv(out / "unmatched_movement_names.csv", index=False)
        pd.DataFrame([dict(name=n, part=p, count=c) for (n, p), c in unmatched_ing.most_common()]).to_csv(out / "unmatched_listed_ingredients.csv", index=False)

        def only(counts, kinds_):
            return {r: Counter({i: n for i, n in counts.get(r, Counter()).items() if kind.get(i) in kinds_}) for r in recipes}

        tool_kinds, v3_kinds = {"tool"}, {"tool", "ingredient"}
        truth_tools = {r: {i for i in move.get(r, {}) if fetch_ok(i, tool_kinds)} for r in recipes}
        truth_v3 = {r: {i for i in move.get(r, {}) if fetch_ok(i, v3_kinds)} for r in recipes}
        profiles = {"V1": only(narr, tool_kinds), "V2": only(move, tool_kinds)}

        def run_version(v, prof, kinds_, truth):
            items = sorted({i for r in recipes for i in prof[r]})
            X = matrix(prof, recipes, items)
            top = pd.DataFrame(dict(item=items, recipes_using=(X > 0).sum(0).astype(int), total=X.sum(0).astype(int)))
            top.sort_values(["recipes_using", "total"], ascending=False).head(20).to_csv(out / f"{v}_top20_items.csv", index=False)
            fetch_counts = {r: Counter({i: n for i, n in prof[r].items() if fetch_ok(i, kinds_)}) for r in recipes}

            def build(rs, form):
                Xs = matrix(prof, rs, items)
                return Xs if form == "raw" else normalise(Xs, np.array([r[:3] for r in rs]))
            groups = {}
            for form in ("raw", "normalised"):
                Z = cluster(build(recipes, form))
                fig, ax = plt.subplots(figsize=(9, 15))
                dendrogram(Z, labels=[label[r] for r in recipes], orientation="left", ax=ax, leaf_font_size=7,
                           color_threshold=0)
                ax.set_title(f"{v}, {form}, {corr_name}: average linkage, cosine distance", loc="left", fontsize=10)
                fig.tight_layout(); fig.savefig(out / "figures" / f"dendrogram_{v}_{form}.png", dpi=130); plt.close(fig)
                for k in KS:
                    lab = cut(Z, k)
                    rng = np.random.default_rng(SEED)
                    ev = evaluate(lab, recipes, fetch_counts, truth, rng)
                    ev.to_csv(out / f"fetch_{v}_{form}_k{k}.csv", index=False)
                    groups[f"{form}|k{k}"] = dict(
                        sizes=sorted(Counter(lab).values(), reverse=True),
                        members={int(g): [label[r] for r, l in zip(recipes, lab) if l == g] for g in sorted(set(lab))},
                        distinctive=distinctive(lab, recipes, fetch_counts))
                    summary.append(dict(corrections=corr_name, version=v, form=form, k=k,
                                        nmi_person=normalized_mutual_info_score(person, lab),
                                        stability_ari=stability(build, recipes, list(person), k, form),
                                        n_eval=len(ev), n_singleton=int((ev.group_size == 1).sum()),
                                        **{f"{s}_{m}": ev[f"{s}_{m}"].mean() for s in ("group", "global", "random")
                                           for m in ("P", "R", "F1", "P5", "P10", "listlen")}))
            json.dump(groups, open(out / f"groups_{v}.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)

        for v in ("V1", "V2"):
            run_version(v, profiles[v], tool_kinds, truth_tools)
        s = pd.DataFrame(summary)
        cur = s[(s.corrections == corr_name) & s.version.isin(["V1", "V2"])].groupby("version").group_F1.mean()
        if corr_name == "corrected":
            best = cur.idxmax()                                         # pre-set rule; the same source is used for both runs
            info["V3_source"] = dict(rule="higher mean group fetch F1 over forms and k, with corrections",
                                     mean_F1=cur.round(4).to_dict(), chosen=best)
        best = info["V3_source"]["chosen"]
        src = narr if best == "V1" else move
        v3 = {}
        for r in recipes:
            c = Counter({i: n for i, n in src.get(r, Counter()).items() if kind.get(i) in v3_kinds})
            for i in listed.get(r, ()):
                c[i] += 1
            v3[r] = c
        run_version("V3", v3, v3_kinds, truth_v3)
        pd.DataFrame(summary).to_csv(TAB / "summary.csv", index=False)
        print(f"== {corr_name} done", flush=True)
    json.dump(info, open(OUT / "info.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False, default=str)
    print(json.dumps({k: v for k, v in info.items() if k != "same_name_recipes"}, indent=1, default=str)[:3000])


if __name__ == "__main__":
    main()
