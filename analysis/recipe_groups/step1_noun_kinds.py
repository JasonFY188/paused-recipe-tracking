"""
analysis/recipe_groups/step1_noun_kinds.py
Step 1 of the recipe-group task: which HD-EPIC noun classes are tools / ingredients / fixtures / cleaning / other?
Reads only; no LLM; HD-EPIC classes and phrases exactly as given.

  - per category: number of noun classes, the 10 classes mentioned most often inside recipe blocks, mentions inside
    recipe blocks (narration x noun pairs), number of recipes (of 69) in which the category is mentioned
  - per class: the same counts (for checking the proposed split class by class)
  - possible class errors: raw noun phrases in the narrations whose LAST word is the key of ANOTHER noun class
    (e.g. "cheese grater" stored as cheese; "oven glove" stored as oven), counted inside recipe blocks

Run (intention env): python analysis/recipe_groups/step1_noun_kinds.py
Outputs: analysis/recipe_groups/tables/step1_*.csv
"""
import ast
import glob
import pathlib
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[2]
ANN = ROOT / "annotations"
OUT = ROOT / "analysis" / "recipe_groups" / "tables"
VIDEO_ID_FIXES = {"P09-2024623-154547": "P09-20240623-154547"}


def recipe_block_narrations():
    """narration rows inside recipe blocks (start in [s, e)), with the recipe key; a narration in two recipes' blocks
    counts once per recipe"""
    d = pd.read_pickle(ANN / "narrations-and-action-segments" / "HD_EPIC_Narrations.pkl")
    dur = pd.read_csv(ANN / "youtube-links" / "HD_EPIC_YouTube_URLs.csv").set_index("video_id").duration
    b = pd.concat([pd.read_csv(f, dtype=str) for f in sorted(glob.glob(str(ANN / "high-level" / "activities" / "*.csv")))],
                  ignore_index=True)
    b["video_id"] = b.video_id.replace(VIDEO_ID_FIXES)
    b["s"] = b.start_time.astype(float)
    b["e"] = pd.to_numeric(b.end_time, errors="coerce")
    b.loc[b.e.isna(), "e"] = b.video_id.map(dur)
    b = b[b.recipe_id.notna() & (b.recipe_id.str.strip() != "")]
    by_v = {v: g for v, g in d.groupby("video_id")}
    rows = []
    for r in b.itertuples():
        g = by_v.get(r.video_id)
        if g is None:
            continue
        g = g[(g.start_timestamp >= r.s) & (g.start_timestamp < r.e)]
        for rid in r.recipe_id.split(","):
            key = f"{r.video_id[:3]}_{rid.strip()}"
            rows += [(key, n) for n in g.index]
    rows = pd.DataFrame(rows, columns=["recipe", "row"]).drop_duplicates()
    return d, rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    nc = pd.read_csv(ANN / "narrations-and-action-segments" / "HD_EPIC_noun_classes.csv")
    key_of, cat_of = dict(zip(nc.id, nc.key)), dict(zip(nc.key, nc.category))
    d, rr = recipe_block_narrations()
    n_recipes = rr.recipe.nunique()

    # mentions: every (narration, noun) pair, with the raw phrase and its class, inside recipe blocks
    mention = []
    for rec, row in zip(rr.recipe, rr.row):
        r = d.loc[row]
        for phrase, cid in zip(r.nouns, r.noun_classes):
            mention.append((rec, row, phrase, key_of[cid]))
    m = pd.DataFrame(mention, columns=["recipe", "row", "phrase", "cls"])
    m["category"] = m.cls.map(cat_of)

    cls = pd.DataFrame(dict(cls=nc.key, category=nc.category))
    cnt = m.groupby("cls").size()
    nrec = m.groupby("cls").recipe.nunique()
    cls["mentions_in_recipe_blocks"] = cls.cls.map(cnt).fillna(0).astype(int)
    cls["recipes_using"] = cls.cls.map(nrec).fillna(0).astype(int)
    cls = cls.sort_values(["category", "mentions_in_recipe_blocks"], ascending=[True, False])
    cls.to_csv(OUT / "step1_classes.csv", index=False)

    total = len(m)
    cats = []
    for c, g in cls.groupby("category"):
        cats.append(dict(category=c, n_classes=len(g), examples=", ".join(g.cls.head(10)),
                         mentions=int(g.mentions_in_recipe_blocks.sum()),
                         share_pct=round(100 * g.mentions_in_recipe_blocks.sum() / total, 1),
                         recipes_using=int(m[m.category == c].recipe.nunique())))
    cats = pd.DataFrame(cats).sort_values("mentions", ascending=False)
    cats.to_csv(OUT / "step1_categories.csv", index=False)

    # possible class errors: last word of the raw phrase is the key (first part) of a different class
    head_key = {}
    for k in nc.key:
        head_key.setdefault(k.split(":")[0], k)
    rows = []
    for (phrase, c), n in m.groupby(["phrase", "cls"]).size().items():
        p = str(phrase).lower().replace("'s", "")
        words = p.split(" of ")[0].split()       # "bottle of oil" -> head "bottle"; "cheese grater" -> "grater"
        if not words:
            continue
        last = words[-1]
        cand = head_key.get(last) or head_key.get(last[:-1] if last.endswith("s") else "")
        if cand and cand != c and cat_of[cand] != cat_of[c]:
            rows.append(dict(phrase=phrase, stored_as=c, stored_category=cat_of[c], head_word_class=cand,
                             head_word_category=cat_of[cand], mentions=int(n)))
    err = pd.DataFrame(rows).sort_values("mentions", ascending=False)
    err.to_csv(OUT / "step1_possible_class_errors.csv", index=False)
    pairs = err.groupby(["stored_as", "head_word_class"]).agg(mentions=("mentions", "sum"),
                                                               example=("phrase", "first")).sort_values("mentions", ascending=False)
    pairs.to_csv(OUT / "step1_possible_class_errors_by_pair.csv")
    # second screen: frequent phrases that share no word with their class key (e.g. "rotating motion" -> control:remote)
    rows = []
    for (phrase, c), n in m.groupby(["phrase", "cls"]).size().items():
        kw = set(c.replace(":", " ").split())
        pw = set(str(phrase).lower().replace("'s", "").split())
        stem = lambda w: w[:-1] if w.endswith("s") else w
        if not ({stem(w) for w in pw} & {stem(w) for w in kw}) and n >= 5:
            rows.append(dict(phrase=phrase, stored_as=c, category=cat_of[c], mentions=int(n)))
    pd.DataFrame(rows).sort_values("mentions", ascending=False).to_csv(OUT / "step1_phrases_unlike_class.csv", index=False)
    print(f"recipes {n_recipes} | noun mentions in recipe blocks {total}")
    print(cats.to_string(index=False))
    print(pairs.head(40).to_string())


if __name__ == "__main__":
    main()
