"""
analysis/object_hierarchy/trees_stats.py
Step 1 (both trees) and Step 2 (target counts) statistics for feasibility.md. Reads data/objects.pkl.
Descriptive tree statistics use all videos of a kitchen; the evaluation uses the leave-one-video-out maps instead.
Output: tables/step1_*.csv, tables/step2_*.csv, stats.json
"""
import ast
import json
import pathlib
import sys
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
TAB = HERE / "tables"
sys.path.insert(0, str(ROOT / "analysis" / "pause_resume"))


def resume_moments(mom):
    """moments among the first 3 narrations of a block in which a paused recipe resumes (pause_resume events)"""
    import pause_resume as pr
    blk, _, _ = pr.load_blocks()
    ev = pd.read_csv(ROOT / "analysis" / "pause_resume" / "tables" / "events.csv")
    s_of = dict(zip(blk.block_id, blk.s))
    keys = {(v, round(s_of[b], 3)) for v, b in zip(ev.video_id, ev.block_after)}
    return keys


def main():
    D = pd.read_pickle(HERE / "data" / "objects.pkl")
    pick, mom, kind = D["pick"], D["moments"], D["kind"]
    st = {}
    # ---- storage tree
    n = len(pick)
    ft = pick.ftype.value_counts()
    st["pickups"] = n
    st["pickup_fixture_known_pct"] = round(100 * (~pick.ftype.isin(["unknown", "mid-air"])).mean(), 1)
    st["pickup_counter_pct"] = round(100 * (pick.ftype == "counter").mean(), 1)
    st["pickup_storage_pct"] = round(100 * pick.storage.mean(), 1)
    ft.rename("pickups").to_frame().assign(pct=lambda d: (100 * d.pickups / n).round(1)).to_csv(TAB / "step1_pickup_fixture_types.csv")
    unk = pick[pick.ftype.isin(["unknown", "mid-air"])]
    unk.groupby(["ftype", "name"]).size().sort_values(ascending=False).rename("pickups").to_csv(TAB / "step1_unknown_fixture_pickups.csv")
    fp = pick[pick.fetch.apply(len) > 0]
    st["fetchable_pickups"] = len(fp)
    st["fetchable_pickups_storage_pct"] = round(100 * fp.storage.mean(), 1)
    st["fetchable_pickups_counter_pct"] = round(100 * (fp.ftype == "counter").mean(), 1)

    rows, stab, per_fix = [], [], []
    for pp, g in fp.groupby("participant"):
        cnt_all, cnt_st = defaultdict(Counter), defaultdict(Counter)
        for cls, f, s in zip(g.fetch, g.fixture, g.storage):
            for c in cls:
                cnt_all[c][f] += 1
                if s:
                    cnt_st[c][f] += 1
        home = {c: (cnt_st[c].most_common(1)[0][0] if cnt_st[c] else "always out") for c in cnt_all}
        n_st = sum(h != "always out" for h in home.values())
        fixtures = Counter(h for h in home.values() if h != "always out")
        rows.append(dict(kitchen=pp, classes=len(home), storage_home=n_st, always_out=len(home) - n_st,
                         storage_fixtures_used_as_home=len(fixtures),
                         fixture_types=len({h.split("_", 1)[1].rsplit(".", 1)[0] for h in fixtures}),
                         objects_per_fixture_median=float(np.median(list(fixtures.values()))) if fixtures else 0,
                         objects_per_fixture_max=max(fixtures.values()) if fixtures else 0))
        for c in [c for c in cnt_st if cnt_st[c]]:                       # (the home lookup created empty entries)
            tot = sum(cnt_st[c].values())
            stab.append(dict(kitchen=pp, cls=c, storage_pickups=tot, share_from_home=cnt_st[c].most_common(1)[0][1] / tot,
                             all_pickups=sum(cnt_all[c].values()),
                             share_all_from_top_fixture=cnt_all[c].most_common(1)[0][1] / sum(cnt_all[c].values())))
        for f, k in fixtures.items():
            per_fix.append(dict(kitchen=pp, fixture=f, objects=k, examples=", ".join(sorted(c for c, h in home.items() if h == f)[:8])))
    t = pd.DataFrame(rows); t.to_csv(TAB / "step1_storage_tree_per_kitchen.csv", index=False)
    s = pd.DataFrame(stab); s.to_csv(TAB / "step1_home_stability.csv", index=False)
    pd.DataFrame(per_fix).sort_values(["kitchen", "objects"], ascending=[True, False]).to_csv(TAB / "step1_home_fixtures.csv", index=False)
    st["storage_tree"] = dict(
        classes_total=int(t.classes.sum()), storage_home=int(t.storage_home.sum()), always_out=int(t.always_out.sum()),
        fixtures_per_kitchen=[int(t.storage_fixtures_used_as_home.min()), int(t.storage_fixtures_used_as_home.max())],
        objects_per_fixture_median=float(np.median([r["objects"] for r in per_fix])),
        objects_per_fixture_max=int(max(r["objects"] for r in per_fix)),
        home_stability_median=round(float(s.share_from_home.median()), 3),
        home_stability_weighted=round(float((s.share_from_home * s.storage_pickups).sum() / s.storage_pickups.sum()), 3),
        home_stability_classes_with_2plus=round(float(s[s.storage_pickups >= 2].share_from_home.median()), 3),
        all_pickup_top_fixture_share_median=round(float(s.share_all_from_top_fixture.median()), 3))
    a3 = pd.Series(D["area3d"])
    st["area3d"] = dict(fixtures=len(a3), areas=int(a3.nunique()),
                        areas_per_kitchen={k: int(v) for k, v in a3.groupby(a3.index.str[:3]).nunique().items()})

    # ---- semantic tree
    nc = pd.read_csv(ROOT / "annotations" / "narrations-and-action-segments" / "HD_EPIC_noun_classes.csv")
    nc["kind"] = nc.key.map(kind)
    nc["n_inst"] = nc.instances.apply(lambda x: len(ast.literal_eval(x)))
    fe = nc[nc.kind.isin(["tool", "ingredient"]) & ~nc.key.isin(["water", "food", "mixture"])]
    used = Counter(c for cl in fp.fetch for c in cl)
    cat = fe.groupby("category").agg(classes=("key", "size"), instances=("n_inst", "sum"),
                                     pickups=("key", lambda k: sum(used[x] for x in k))).sort_values("pickups", ascending=False)
    cat.to_csv(TAB / "step1_semantic_categories.csv")
    st["semantic_tree"] = dict(depth=3, categories=len(cat), classes=len(fe),
                               classes_per_category_median=float(cat.classes.median()), classes_per_category_max=int(cat.classes.max()),
                               instances_per_class_median=float(fe.n_inst.median()), instances_per_class_max=int(fe.n_inst.max()),
                               top5_by_pickups=cat.head(5).reset_index().to_dict("records"))

    # ---- targets
    keys = resume_moments(mom)
    bs = D["blocks"].set_index("block").s
    mom["resume"] = [(v, round(bs[b], 3)) in keys and p < 3 for v, b, p in zip(mom.video_id, mom.block, mom.pos)]
    tg = {}
    for W in (5, 10, 20):
        for tag in ("", "cls_"):
            c = f"{tag}target_W{W}"
            m = mom[c].notna()
            tg[f"{tag or 'physical_'}W{W}"] = dict(moments=int(m.sum()), share_of_moments=round(float(m.mean()), 3),
                                                   picked_from_storage=round(float(mom.loc[m, f"{tag}storage_W{W}"].mean()), 3))
    m = mom.target_W10.notna()
    for name, hm in (("main", D["home_main"]), ("causal", D["home_causal"])):
        kinds_ = Counter()
        for v, tgt in zip(mom.video_id[m], mom.target_W10[m]):
            h = [hm.get(v, {}).get(c, "unknown") for c in tgt]
            kinds_["storage home" if any(x not in ("always out", "unknown") for x in h) else
                   "always out" if any(x == "always out" for x in h) else "unknown (not seen in the map videos)"] += 1
        tg[f"home_kind_{name}"] = {k: f"{v} ({100 * v / m.sum():.1f}%)" for k, v in kinds_.items()}
    tg["resume_moments_with_target"] = int((mom.resume & m).sum())
    tg["resume_moments"] = int(mom.resume.sum())
    st["targets"] = tg
    mom[["video_id", "block", "pos", "resume"]].to_pickle(HERE / "data" / "resume_flags.pkl")
    json.dump(st, open(HERE / "stats.json", "w"), indent=1, default=str)
    print(json.dumps(st, indent=1, default=str))
    print(t.to_string(index=False))


if __name__ == "__main__":
    main()
