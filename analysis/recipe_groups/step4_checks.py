"""
analysis/recipe_groups/step4_checks.py
Extra checks for Step 4 (reads the outputs of recipe_groups.py):
  - paired group-minus-global difference per recipe (F1, P@5) with a bootstrap 95% CI over recipes (seed 0)
  - NMI(person) that random groups of the same sizes would get (mean of 200 shuffles, seed 0)
  - do same-name recipes (Coffee x4, Cappuccino x2) fall into the same group?
  - how many glove mentions in recipe blocks look like rubber gloves (keyword count only; no rule applied)
Output: tables/step4_checks.csv, tables/step4_same_name.csv, step4_checks.json
"""
import json
import pathlib
import re
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import normalized_mutual_info_score

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import recipe_groups as rg

KS = rg.KS


def boot(d, rng, n=2000):
    d = np.asarray(d)
    m = [rng.choice(d, len(d)).mean() for _ in range(n)]
    return [round(float(np.percentile(m, 2.5)), 3), round(float(np.percentile(m, 97.5)), 3)]


def main():
    rng = np.random.default_rng(0)
    recipes_json = json.load(open(rg.ANN / "high-level" / "complete_recipes.json", encoding="utf-8"))
    recipes = sorted(recipes_json)
    person = np.array([r[:3] for r in recipes])
    from collections import Counter
    dup_pp = Counter((recipes_json[r]["name"], r[:3]) for r in recipes)
    label = {f"{recipes_json[r]['name']} ({r[:3]}" + (f", {r[4:]})" if dup_pp[(recipes_json[r]['name'], r[:3])] > 1 else ")"): r
             for r in recipes}
    rows, same = [], []
    for corr in ("corrected", "uncorrected"):
        for v in ("V1", "V2", "V3"):
            groups = json.load(open(HERE / corr / f"groups_{v}.json", encoding="utf-8"))
            for form in ("raw", "normalised"):
                for k in KS:
                    ev = pd.read_csv(HERE / corr / f"fetch_{v}_{form}_k{k}.csv")
                    g = groups[f"{form}|k{k}"]
                    lab = {label[m]: gid for gid, mem in g["members"].items() for m in mem}
                    labs = np.array([lab[r] for r in recipes])
                    sizes = np.bincount(np.unique(labs, return_inverse=True)[1])
                    rand_nmi = np.mean([normalized_mutual_info_score(person, rng.permutation(labs)) for _ in range(200)])
                    dF, dP = ev.group_F1 - ev.global_F1, ev.group_P5 - ev.global_P5
                    rows.append(dict(corrections=corr, version=v, form=form, k=k,
                                     nmi_person=round(normalized_mutual_info_score(person, labs), 3),
                                     nmi_random_same_sizes=round(float(rand_nmi), 3), sizes=" / ".join(map(str, sorted(sizes, reverse=True))),
                                     F1_group_minus_global=round(float(dF.mean()), 3), F1_ci=boot(dF, rng),
                                     F1_wins=f"{int((dF > 0).sum())}/{int((dF < 0).sum())}",
                                     P5_group_minus_global=round(float(dP.mean()), 3), P5_ci=boot(dP, rng),
                                     P5_wins=f"{int((dP > 0).sum())}/{int((dP < 0).sum())}"))
                    if corr == "corrected":
                        for name in ("Coffee", "Cappuccino"):
                            mem = [r for r in recipes if recipes_json[r]["name"] == name]
                            same.append(dict(version=v, form=form, k=k, recipe=name, n=len(mem),
                                             groups=" ".join(f"{r[:3]}:{lab[r]}" for r in mem),
                                             all_same=len({lab[r] for r in mem}) == 1))
    t = pd.DataFrame(rows)
    t.to_csv(HERE / "tables" / "step4_checks.csv", index=False)
    pd.DataFrame(same).to_csv(HERE / "tables" / "step4_same_name.csv", index=False)

    # glove mentions in recipe blocks that look like rubber gloves (sentence keywords; reported only)
    blocks = rg.load_blocks()
    d = pd.read_pickle(rg.ANN / "narrations-and-action-segments" / "HD_EPIC_Narrations.pkl")
    by_v = {v: g for v, g in d.groupby("video_id")}
    seen, n_glove, n_rubber = set(), 0, 0
    for r in blocks.itertuples():
        g = by_v.get(r.video_id)
        if g is None:
            continue
        g = g[(g.start_timestamp >= r.s) & (g.start_timestamp < r.e)]
        for idx, nouns, text in zip(g.index, g.nouns, g.narration):
            if idx in seen:
                continue
            seen.add(idx)
            gl = [p for p in nouns if re.search(r"glove|mitt", p.lower())]
            if gl:
                n_glove += len(gl)
                if re.search(r"rubber|wash|rinse|sink|dish|scrub|sponge", text.lower()) and not re.search(r"oven|mitt", " ".join(gl).lower()):
                    n_rubber += len(gl)
    out = dict(glove_mentions_in_recipe_blocks=n_glove, look_like_rubber_gloves=n_rubber)
    json.dump(out, open(HERE / "step4_checks.json", "w"), indent=1)
    pd.set_option("display.width", 250); pd.set_option("display.max_rows", 100)
    print(t.to_string(index=False)); print(out)
    print(pd.DataFrame(same).groupby("recipe").all_same.mean())


if __name__ == "__main__":
    main()
