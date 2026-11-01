"""
analysis/object_hierarchy/object_data.py
Data for the object-hierarchy feasibility check (feasibility.md): pick-up events, the two trees, homes, and the
"next new object" targets. HD-EPIC labels + the documented phrase corrections; no LLM.

Pick-up event   one object-movement track (assoc_info.json): time = start of the track; pick-up fixture = fixture of
                its earliest mask (mask_info.json); object = the association (one physical object in one video);
                classes = noun classes of its name (recipe_groups matcher: container X and contents Y), kept only if
                fetchable (tool or ingredient, not water / food / mixture)
Fixture type    the fixture name without kitchen prefix and number ("P01_cupboard.003" -> cupboard)   [dataset]
Storage         fixture types cupboard, drawer, fridge, freezer, fridgefreezer, shelf, storage, dishwasher   [user;
                fridgefreezer added by Claude; there is no drying-rack fixture type in HD-EPIC]
Home            per kitchen and class: the storage fixture it is most often picked up from, using the person's OTHER
                videos (main) or EARLIER videos (causal check); "always out" if seen but never from storage; "unknown"
                if not seen                                                                                 [user]
Moments         every narration of every activity block (identical duplicate blocks merged); past = the block's
                narrations up to it; window = from the start of the next narration to the start of the narration
                W later (W = 10 main; 5, 20 checks), or the block end                                       [user / Claude]
Target          the first pick-up in the window, inside the block, of a fetchable object that was not moved earlier in
                the block: physical object (main) or class level (check)                                     [user]

Run (intention env): python analysis/object_hierarchy/object_data.py
Output: analysis/object_hierarchy/data/*.pkl, tables/step1_*.csv
"""
import glob
import json
import pathlib
import re
import sys
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "analysis" / "recipe_groups"))
import recipe_groups as rg                                             # matcher + the 15 corrections (unchanged)

ANN = ROOT / "annotations"
OUT = ROOT / "analysis" / "object_hierarchy"
DATA, TAB = OUT / "data", OUT / "tables"
CORRECTIONS = dict(rg.CORRECTIONS, **{"tablecloth": "clothes", "washing machine": "machine:washing", "tarts": "pie",
                                      "kitchen roll": "towel:kitchen", "hot water": "water"})
STORAGE = {"cupboard", "drawer", "fridge", "freezer", "fridgefreezer", "shelf", "storage", "dishwasher"}
NOT_FETCHABLE = {"water", "food", "mixture"}
WINDOWS = [5, 10, 20]
CLUSTER_DIST = 1.0                                                      # metres, 3D-location tree (secondary)


def fixture_type(f):
    if f is None or f in ("Null", "null", "None"):
        return "unknown"
    if f == "mid-air":
        return "mid-air"
    return re.sub(r"\.\d+$", "", f.split("_", 1)[1]) if "_" in f else f


def video_time(v):
    return v.split("-", 1)[1]                                           # "20240202-110250" sorts by recording time


def load_pickups(match, kind):
    a = json.load(open(ANN / "scene-and-object-movements" / "assoc_info.json"))
    m = json.load(open(ANN / "scene-and-object-movements" / "mask_info.json"))
    mask = {k: v for vid in m.values() for k, v in vid.items()}
    rows = []
    for vid, assocs in a.items():
        for aid, asc in assocs.items():
            cls = [k for _, k in match(asc["name"]) if k]
            fetch = sorted({k for k in cls if kind.get(k) in ("tool", "ingredient") and k not in NOT_FETCHABLE})
            for t in asc["tracks"]:
                ms = sorted([mask[x] for x in t["masks"] if x in mask], key=lambda x: x["frame_number"])
                f = ms[0]["fixture"] if ms else None
                rows.append(dict(video_id=vid, participant=vid[:3], assoc=aid, name=asc["name"], start=float(t["time_segment"][0]),
                                 end=float(t["time_segment"][1]), fixture=f if f not in ("Null", "null") else None,
                                 ftype=fixture_type(f), classes=tuple(sorted(set(cls))), fetch=tuple(fetch),
                                 xyz=ms[0]["3d_location"] if ms else None))
    p = pd.DataFrame(rows).sort_values(["video_id", "start"]).reset_index(drop=True)
    p["storage"] = p.ftype.isin(STORAGE)
    return p, mask


def homes(p, videos_for):
    """{test video: {class: home}}; videos_for(test video) -> the videos whose pick-ups build the map"""
    out = {}
    by_v = {v: g for v, g in p.groupby("video_id")}
    for v in p.video_id.unique():
        src = [by_v[o] for o in videos_for(v) if o in by_v]
        cnt = defaultdict(Counter)
        seen = set()
        for g in src:
            for cls, f, st in zip(g.fetch, g.fixture, g.storage):
                for c in cls:
                    seen.add(c)
                    if st:
                        cnt[c][f] += 1
        out[v] = {c: (cnt[c].most_common(1)[0][0] if cnt[c] else "always out") for c in seen}
    return out


def fixture_clusters(p):
    """3D-location tree: per kitchen, storage fixtures grouped by the mean 3D position of their masks (average linkage,
    distance threshold CLUSTER_DIST). Fixture positions are static geometry, so all videos are used."""
    out = {}
    s = p[p.storage & p.xyz.notna()]
    for pp, g in s.groupby("participant"):
        cen = g.groupby("fixture").xyz.apply(lambda l: np.mean(np.stack(l.values), 0))
        if len(cen) == 1:
            out.update({cen.index[0]: f"{pp}_area1"})
            continue
        lab = fcluster(linkage(np.stack(cen.values), "average"), CLUSTER_DIST, "distance")
        out.update({f: f"{pp}_area{l}" for f, l in zip(cen.index, lab)})
    return out


def load_blocks():
    dur = pd.read_csv(ANN / "youtube-links" / "HD_EPIC_YouTube_URLs.csv").set_index("video_id").duration
    b = pd.concat([pd.read_csv(f, dtype=str) for f in sorted(glob.glob(str(ANN / "high-level" / "activities" / "*.csv")))],
                  ignore_index=True)
    b["video_id"] = b.video_id.replace(rg.VIDEO_ID_FIXES)
    b["s"] = b.start_time.astype(float)
    b["e"] = pd.to_numeric(b.end_time, errors="coerce")
    b.loc[b.e.isna(), "e"] = b.video_id.map(dur)
    b["recipe_id"] = b.recipe_id.fillna("").str.strip()
    b = b.drop_duplicates(["video_id", "s", "e"]).reset_index(drop=True)   # identical duplicate blocks (P03) merged
    b["block"] = b.index
    return b[b.e > b.s]


def narrations(key_of, verb_of):
    d = pd.read_pickle(ANN / "narrations-and-action-segments" / "HD_EPIC_Narrations.pkl")
    ok = d.main_action_classes.apply(lambda m: len(m) == 1 and len(m[0]) == 2)
    d = d[ok]
    verb, noun = [], []
    for ma, nouns, ncls in zip(d.main_action_classes, d.nouns, d.noun_classes):
        v, n = verb_of[ma[0][0]], key_of[ma[0][1]]
        for p, c in zip(nouns, ncls):                                   # the main noun's own phrase, if corrected
            if c == ma[0][1]:
                n = CORRECTIONS.get(p.strip().lower(), n)
                break
        verb.append(v); noun.append(n)
    return pd.DataFrame(dict(video_id=d.video_id.values, start=d.start_timestamp.astype(float).values,
                             verb=verb, noun=noun)).sort_values(["video_id", "start"]).reset_index(drop=True)


def moments_and_targets(blocks, narr, pick):
    nby, pby = {v: g for v, g in narr.groupby("video_id")}, {v: g for v, g in pick.groupby("video_id")}
    rows = []
    for b in blocks.itertuples():
        n = nby.get(b.video_id)
        if n is None:
            continue
        n = n[(n.start >= b.s) & (n.start < b.e)]
        if not len(n):
            continue
        pk = pby.get(b.video_id)
        pk = pk[(pk.start >= b.s) & (pk.start < b.e)] if pk is not None else pick.iloc[:0]
        starts = n.start.tolist()
        acts = (n.verb + " " + n.noun).tolist()
        for i in range(len(starts)):
            w0 = starts[i + 1] if i + 1 < len(starts) else b.e
            before = pk[pk.start < w0]
            moved_obj, moved_cls = set(before.assoc), {c for cl in before.fetch for c in cl}
            row = dict(video_id=b.video_id, participant=b.video_id[:3], block=b.block, recipe=b.recipe_id, pos=i,
                       t=w0, past=acts[:i + 1], past_verbs=n.verb.tolist()[:i + 1], past_nouns=n.noun.tolist()[:i + 1])
            for W in WINDOWS:
                w1 = starts[i + 1 + W] if i + 1 + W < len(starts) else b.e
                win = pk[(pk.start >= w0) & (pk.start < w1) & (pk.fetch.apply(len) > 0)]
                phys = win[~win.assoc.isin(moved_obj)]
                cls = win[win.fetch.apply(lambda f: not (set(f) & moved_cls))]
                for tag, sel in (("", phys), ("cls_", cls)):
                    if len(sel):
                        r = sel.iloc[0]
                        row[f"{tag}target_W{W}"] = r.fetch
                        row[f"{tag}fixture_W{W}"] = r.fixture
                        row[f"{tag}ftype_W{W}"] = r.ftype
                        row[f"{tag}storage_W{W}"] = bool(r.storage)
                        row[f"{tag}name_W{W}"] = r["name"]
            rows.append(row)
    return pd.DataFrame(rows)


def main():
    DATA.mkdir(parents=True, exist_ok=True); TAB.mkdir(parents=True, exist_ok=True)
    kinds = pd.read_csv(ROOT / "analysis" / "recipe_groups" / "tables" / "step1_proposed_kinds.csv")
    kind = dict(zip(kinds.cls, kinds.kind))
    key_of, phrase, _ = rg.noun_tables()
    match = rg.make_matcher(phrase, CORRECTIONS)
    pick, mask = load_pickups(match, kind)
    vc = pd.read_csv(ANN / "narrations-and-action-segments" / "HD_EPIC_verb_classes.csv")
    narr = narrations(key_of, dict(zip(vc.id, vc.key)))
    blocks = load_blocks()
    videos = sorted(pick.video_id.unique())
    by_person = defaultdict(list)
    for v in videos:
        by_person[v[:3]].append(v)
    home_main = homes(pick, lambda v: [o for o in by_person[v[:3]] if o != v])
    home_causal = homes(pick, lambda v: [o for o in by_person[v[:3]] if video_time(o) < video_time(v)])
    area3d = fixture_clusters(pick)
    mom = moments_and_targets(blocks, narr, pick)
    pd.to_pickle(dict(pick=pick, narr=narr, blocks=blocks, moments=mom, home_main=home_main, home_causal=home_causal,
                      area3d=area3d, kind=kind, corrections=CORRECTIONS), DATA / "objects.pkl")
    print(f"pick-ups {len(pick)} | moments {len(mom)} | with target W10 {mom.target_W10.notna().sum()}")


if __name__ == "__main__":
    main()

