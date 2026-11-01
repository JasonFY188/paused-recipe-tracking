"""
analysis/pause_resume/pause_resume.py
When people leave a recipe and come back: what happens in between, and does the pause start right after a waiting action?
Plain data analysis, no LLM. HD-EPIC text is used exactly as written (only leading/trailing spaces are stripped).

Run (intention env): KMP_DUPLICATE_LIB_OK=TRUE python analysis/pause_resume/pause_resume.py
Outputs (analysis/pause_resume/): results.json, tables/*.csv, figures/*.png. The report (report.md) is written by hand from these.
"""
import glob
import json
import re
import textwrap
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
ANN = ROOT / "annotations"
OUT = ROOT / "analysis" / "pause_resume"
TAB, FIG = OUT / "tables", OUT / "figures"
VIDEO_ID_FIXES = {"P09-2024623-154547": "P09-20240623-154547"}      # typo in P09_recipe_timestamps.csv
WAIT_WORDS = ["oven", "microwave", "kettle", "boil", "simmer", "bake", "timer", "machine", "start", "switch on",
              "turn on", "heat", "toaster", "rice cooker", "wait"]
LOOSE_PHRASES = {"switch on": "switch", "turn on": "turn"}            # verb ... on, up to 3 words in between
MIN_COUNT = 10
N_EDGE = 3                                                          # last / first 3 narrations
SEG_OVERLAP = 0.5

# Manual check by Claude (the assistant), reading each raw narration against the written definition in report.md.
# The sentences to read are exported to tables/to_label_*.csv; the labels are stored in tables/claude_wait_labels.csv.
LABELS = OUT / "tables" / "claude_wait_labels.csv"
N_BASELINE_WINDOWS = 400
SEED = 0


# ------------------------------------------------------------------ loading
def load_blocks():
    dur = pd.read_csv(ANN / "youtube-links" / "HD_EPIC_YouTube_URLs.csv").set_index("video_id").duration
    b = pd.concat([pd.read_csv(f, dtype=str) for f in sorted(glob.glob(str(ANN / "high-level" / "activities" / "*.csv")))],
                  ignore_index=True)
    b["video_id"] = b.video_id.replace(VIDEO_ID_FIXES)
    b["name"] = b.high_level_activity_label.str.strip()
    b["s"] = b.start_time.astype(float)
    b["e"] = pd.to_numeric(b.end_time, errors="coerce")
    b.loc[b.e.isna(), "e"] = b.video_id.map(dur)                    # "end" = end of the video
    b["recs"] = b.recipe_id.fillna("").apply(lambda x: [f"{r.strip()}" for r in x.split(",") if r.strip()])
    n_raw = len(b)
    # Option 2: identical blocks (same video, start, end, name) listed once per recipe -> one block with both recipes
    b = (b.groupby(["video_id", "s", "e", "name"], as_index=False)
         .agg(recs=("recs", lambda l: sorted({r for x in l for r in x})), n_copies=("recs", "size")))
    b["participant"] = b.video_id.str[:3]
    b["recs"] = [[f"{p}_{r}" for r in rs] for p, rs in zip(b.participant, b.recs)]
    b["broken"] = b.e < b.s                                          # end before start
    b = b.sort_values(["video_id", "s", "e"]).reset_index(drop=True)
    b["block_id"] = b.index
    return b, n_raw, dur


def load_narrations():
    d = pd.read_pickle(ANN / "narrations-and-action-segments" / "HD_EPIC_Narrations.pkl")
    vc = pd.read_csv(ANN / "narrations-and-action-segments" / "HD_EPIC_verb_classes.csv")
    nc = pd.read_csv(ANN / "narrations-and-action-segments" / "HD_EPIC_noun_classes.csv")
    verbs, nouns = dict(zip(vc.id, vc.key)), dict(zip(nc.id, nc.key))
    ok = lambda m: len(m) > 0 and len(m[0]) == 2
    n = pd.DataFrame(dict(
        narration_id=d.unique_narration_id, video_id=d.video_id, participant=d.participant_id,
        start=d.start_timestamp.astype(float), end=d.end_timestamp.astype(float), narration=d.narration.str.strip(),
        main_verb=d.main_action_classes.apply(lambda m: verbs[m[0][0]] if ok(m) else None),
        main_noun=d.main_action_classes.apply(lambda m: nouns[m[0][1]] if ok(m) else None),
        nouns=d.noun_classes.apply(lambda l: sorted({nouns[i] for i in l}))))
    n["main_action"] = np.where(n.main_verb.notna(), n.main_verb + " " + n.main_noun.fillna(""), None)
    return n.sort_values(["video_id", "start", "end"]).reset_index(drop=True), vc, nc


def load_recipes():
    r = json.load(open(ANN / "high-level" / "complete_recipes.json", encoding="utf-8"))
    segs = [dict(recipe=k, video_id=s["video"], s=float(s["start"]), e=float(s["end"]))
            for k, v in r.items() for c in v["captures"] for kind in ("step_times", "prep_times")
            for lst in c[kind].values() for s in lst]
    return {k: v["name"] for k, v in r.items()}, pd.DataFrame(segs)


# ------------------------------------------------------------------ waiting words
def word_rx(k):
    return re.compile(r"\b" + r"\s+".join(map(re.escape, k.split())) + r"\w*", re.I)


STRICT = {k: word_rx(k) for k in WAIT_WORDS}
LOOSE = {k: re.compile(rf"\b{v}\w*\s+(?:[\w'-]+\s+){{1,3}}on\b", re.I) for k, v in LOOSE_PHRASES.items()}


def sentence_hits(text):
    """(strict keywords found, keywords found only by the loose 'verb ... on' rule)"""
    strict = {k for k, rx in STRICT.items() if rx.search(text)}
    loose = {k for k, rx in LOOSE.items() if k not in strict and rx.search(text)}
    return strict, loose


def class_keyword_map(vc, nc):
    """which HD-EPIC verb / noun classes each waiting word matches (word-start match on the class key)"""
    m = {}
    for k in WAIT_WORDS:
        rx = word_rx(k)
        norm = lambda x: x.replace(":", " ").replace("-", " ")
        m[k] = dict(verbs=[v for v in vc.key if rx.search(norm(v))], nouns=[v for v in nc.key if rx.search(norm(v))])
    return m


# ------------------------------------------------------------------ helpers
def q(x, qs=(.25, .5, .75, .9)):
    x = np.asarray(x, float)
    return {f"p{int(k * 100)}": round(float(np.quantile(x, k)), 1) for k in qs} if len(x) else {}


def assign(narr, blocks):
    """narration -> blocks whose [s, e) contains its start (a narration can be in several overlapping blocks)"""
    by_v = {v: g for v, g in blocks.groupby("video_id")}
    out = []
    for r in narr.itertuples():
        g = by_v.get(r.video_id)
        out.append([] if g is None else g.block_id[(g.s <= r.start) & (r.start < g.e)].tolist())
    return out


def ratio_table(items_a, items_b, min_count):
    """items_*: list of sets (one per narration). rate = share of narrations containing the class."""
    ca, cb = Counter(x for s in items_a for x in s), Counter(x for s in items_b for x in s)
    na, nb = len(items_a), len(items_b)
    rows = [dict(cls=k, n_before=c, pct_before=round(100 * c / na, 2), n_other=cb.get(k, 0),
                 pct_other=round(100 * cb.get(k, 0) / nb, 2),
                 ratio=round((c / na) / max(cb.get(k, 0) / nb, 0.5 / nb), 2)) for k, c in ca.items() if c >= min_count]
    return pd.DataFrame(rows).sort_values(["ratio", "n_before"], ascending=False).reset_index(drop=True) if rows else pd.DataFrame()


# ------------------------------------------------------------------ main
def main():
    TAB.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    blocks, n_raw_blocks, dur = load_blocks()
    narr, vc, nc = load_narrations()
    rname, segs = load_recipes()
    narr["blocks"] = assign(narr, blocks)
    blk = blocks.set_index("block_id")
    res = dict(data=dict(blocks_in_files=n_raw_blocks, blocks_after_merging_copies=len(blocks),
                         merged_copies=int((blocks.n_copies > 1).sum()), broken_blocks=blocks[blocks.broken][["video_id", "name", "s", "e"]].to_dict("records"),
                         videos_with_blocks=int(blocks.video_id.nunique()), narrations=len(narr),
                         narrations_in_no_block=int((narr.blocks.str.len() == 0).sum()),
                         narrations_in_2plus_blocks=int((narr.blocks.str.len() > 1).sum())))
    narr_by_block = defaultdict(list)
    for i, bl in zip(narr.index, narr.blocks):
        for b_ in bl:
            narr_by_block[b_].append(i)

    # ---------------- Step 1: events within one video
    events = []
    for v, g in blocks.groupby("video_id"):
        g = g.reset_index(drop=True)
        for R in sorted({r for l in g.recs for r in l}):
            idx = [i for i, l in enumerate(g.recs) if R in l]
            for i, j in zip(idx, idx[1:]):
                if j - i < 2:
                    continue
                bi, bj, gap = g.iloc[i], g.iloc[j], g.iloc[i + 1:j]
                ov = max([max(0.0, min(x.e, bi.e) - max(x.s, bi.s)) if not x.broken else 0 for x in gap.itertuples()] +
                         [max(0.0, min(x.e, bj.e) - max(x.s, bj.s)) if not x.broken else 0 for x in gap.itertuples()])
                broken = bool(g.iloc[i:j + 1].broken.any())
                gap_narr = sorted({n for x in gap.block_id for n in narr_by_block[x]} -
                                  set(narr_by_block[bi.block_id]) - set(narr_by_block[bj.block_id]))
                # last 3 narrations of R before the gap (R blocks of this video up to block i), first 3 after it
                r_before = sorted({n for x in g.iloc[:i + 1].itertuples() if R in x.recs for n in narr_by_block[x.block_id]},
                                  key=lambda n: (narr.start[n], n))
                r_after = sorted({n for x in g.iloc[j:].itertuples() if R in x.recs for n in narr_by_block[x.block_id]},
                                 key=lambda n: (narr.start[n], n))
                events.append(dict(
                    event_id=f"{v}|{R}|{bi.block_id}", video_id=v, participant=v[:3], recipe=R, recipe_name=rname.get(R, "?"),
                    block_before=int(bi.block_id), block_after=int(bj.block_id), gap_blocks=gap.block_id.tolist(),
                    gap_s=round(bj.s - bi.e, 1), n_gap_narr=len(gap_narr), gap_narr=gap_narr,
                    n_gap_blocks=len(gap), gap_has_other_recipe=bool(any(gap.recs.str.len() > 0)),
                    overlap_s=round(ov, 1), broken=broken,
                    flag="broken" if broken else ("parallel" if ov > 0 else "clean"),
                    last3=r_before[-N_EDGE:], first3=r_after[:N_EDGE], name_before=bi["name"], name_after=bj["name"]))
    ev = pd.DataFrame(events)
    clean = ev[ev.flag == "clean"]

    def step1(e):
        return dict(n=len(e), per_participant=e.participant.value_counts().sort_index().to_dict(),
                    participants_with_events=int(e.participant.nunique()),
                    top2_participants_share=round(100 * e.participant.value_counts().head(2).sum() / len(e), 1),
                    recipes_with_events=int(e.recipe.nunique()),
                    gap_seconds=q(e.gap_s), gap_narrations=q(e.n_gap_narr), gap_blocks=q(e.n_gap_blocks),
                    pct_gap_with_other_recipe=round(100 * e.gap_has_other_recipe.mean(), 1))
    res["step1"] = dict(all=step1(ev), clean_only=step1(clean), flags=ev.flag.value_counts().to_dict(),
                        parallel_overlap_seconds=sorted(ev[ev.flag == "parallel"].overlap_s.tolist()))
    per_recipe = (ev.groupby(["recipe", "recipe_name"]).agg(events=("event_id", "size"), clean=("flag", lambda f: (f == "clean").sum()),
                                                            median_gap_s=("gap_s", "median"))
                  .reset_index().sort_values("events", ascending=False))
    per_recipe.to_csv(TAB / "step1_events_per_recipe.csv", index=False)
    per_part = ev.groupby("participant").agg(events=("event_id", "size"), clean=("flag", lambda f: (f == "clean").sum()),
                                             recipes=("recipe", "nunique"), median_gap_s=("gap_s", "median"),
                                             median_gap_narr=("n_gap_narr", "median")).reset_index()
    per_part.to_csv(TAB / "step1_events_per_participant.csv", index=False)
    res["step1"]["recipes_total"] = len(rname)

    # resumes in a DIFFERENT video
    vt = lambda v: pd.Timestamp(pd.to_datetime(v[4:], format="%Y%m%d-%H%M%S"))
    cross = []
    for R, g in blocks.explode("recs").dropna(subset=["recs"]).groupby("recs"):
        vids = sorted(g.video_id.unique(), key=vt)
        for a, b_ in zip(vids, vids[1:]):
            last_a, first_b = g[g.video_id == a].e.max(), g[g.video_id == b_].s.min()
            ga = blocks[(blocks.video_id == a) & (blocks.s >= last_a - 1e-6)]
            gb = blocks[(blocks.video_id == b_) & (blocks.e <= first_b + 1e-6)]
            hours = (vt(b_) - (vt(a) + pd.Timedelta(seconds=float(dur[a])))).total_seconds() / 3600
            cross.append(dict(recipe=R, recipe_name=rname.get(R, "?"), participant=R[:3], video_from=a, video_to=b_,
                              same_day=a[4:12] == b_[4:12], hours_between=round(hours, 2),
                              blocks_after_in_old_video=" | ".join(ga.name), blocks_before_in_new_video=" | ".join(gb.name)))
    cr = pd.DataFrame(cross)
    cr.to_csv(TAB / "step1_resumes_in_another_video.csv", index=False)
    res["step1"]["other_video"] = dict(n=len(cr), same_day=int(cr.same_day.sum()), other_day=int((~cr.same_day).sum()),
                                       hours_between=q(cr.hours_between), per_participant=cr.participant.value_counts().sort_index().to_dict(),
                                       recipes=int(cr.recipe.nunique()))

    # cross-check: step/prep segments (union of recipe R's segments per video; pause = >= 1 narration between two pieces)
    seg_ev = []
    for (R, v), g in segs.groupby(["recipe", "video_id"]):
        iv = sorted(zip(g.s, g.e))
        merged = [list(iv[0])]
        for s, e in iv[1:]:
            if s <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], e)
            else:
                merged.append([s, e])
        nv = narr[narr.video_id == v]
        for (s1, e1), (s2, e2) in zip(merged, merged[1:]):
            k = int(((nv.start >= e1) & (nv.start < s2)).sum())
            if k > 0:
                seg_ev.append(dict(recipe=R, video_id=v, participant=v[:3], gap_s=s2 - e1, n_gap_narr=k))
    se = pd.DataFrame(seg_ev)
    res["step1"]["crosscheck_segments"] = dict(
        definition="per video, union of recipe R's step+prep segments; a pause = >= 1 narration starts between two consecutive pieces",
        n=len(se), per_participant=se.participant.value_counts().sort_index().to_dict(),
        gap_seconds=q(se.gap_s), gap_narrations=q(se.n_gap_narr),
        n_with_gap_ge_5_narrations=int((se.n_gap_narr >= 5).sum()), n_with_gap_ge_20_narrations=int((se.n_gap_narr >= 20).sum()))

    # ---------------- Step 2: what fills the gap
    gb_rows = [dict(event_id=e.event_id, participant=e.participant, block_id=b_, name=blk.name[b_],
                    is_other_recipe=bool(blk.recs[b_]), flag=e.flag) for e in ev.itertuples() for b_ in e.gap_blocks]
    gbl = pd.DataFrame(gb_rows)
    gbl.to_csv(TAB / "step2_all_gap_blocks.csv", index=False)

    def name_table(g):
        c = g.groupby("name").agg(count=("event_id", "size"), other_recipe=("is_other_recipe", "sum"),
                                  participants=("participant", lambda p: ", ".join(f"{k} {v}" for k, v in Counter(p).most_common())))
        c = c.reset_index().sort_values(["count", "name"], ascending=[False, True])
        low = g.name.str.lower()
        c["count_case_merged"] = c.name.str.lower().map(low.value_counts())
        return c

    nt_all, nt_clean = name_table(gbl), name_table(gbl[gbl.flag == "clean"])
    nt_all.to_csv(TAB / "step2_gap_block_names_all.csv", index=False)
    nt_clean.to_csv(TAB / "step2_gap_block_names_clean.csv", index=False)

    def step2(g, nt):
        return dict(gap_blocks=len(g), pct_non_recipe=round(100 * (~g.is_other_recipe).mean(), 1),
                    pct_other_recipe=round(100 * g.is_other_recipe.mean(), 1), unique_names=int(g.name.nunique()),
                    names_used_once=int((nt["count"] == 1).sum()),
                    top40_coverage_pct=round(100 * nt["count"].head(40).sum() / len(g), 1),
                    unique_names_case_merged=int(g.name.str.lower().nunique()))
    res["step2"] = dict(all=step2(gbl, nt_all), clean_only=step2(gbl[gbl.flag == "clean"], nt_clean))

    # proposed grouping (keyword rules on the lower-cased raw name; first matching rule wins) -> mapping table to edit
    RULES = [("another recipe", None),
             ("waiting", r"\bwait"),
             ("hand washing", r"\b(wash|dry|wipe|clean)\w*\b[^,]*\bhands?\b"),
             ("dishwasher", r"dishwasher"),
             ("washing up", r"\bwash\w*\b|\bwashing[- ]up\b|\bdishes\b|\bdry up\b|\bdrying\b"),
             ("wiping / cleaning surfaces", r"\bwip\w*|\bcounter|\bsurface|\bspill"),
             ("tidying / putting away", r"\btid\w*|\bput\w* (?:\w+ )*away|\bput stuff|\bstore|\bclear\b|\bfridge"),
             ("cleaning (unspecified)", r"\bclean"),
             ("eating / serving / table", r"\beat|\bserv|\btable\b|\bdish(?:ing)? up|\bplat(?:e|ing)\b|\blunch|\bdinner|\bbreakfast|\bsnack"),
             ("drink / other food (not a listed recipe)", r"\bcoffee|\btea\b|\bjuice|\bdrink|\bwater\b|\bprepare|\bmake|\bcook|\bheat|\bmicrowave|\btoast"),
             ("other", r".")]

    def group_of(name, is_recipe):
        if is_recipe:
            return "another recipe", "block has a recipe id"
        for gname, rx in RULES[1:]:
            if re.search(rx, name.lower()):
                return gname, rx
        return "other", "."
    mp = gbl.drop_duplicates("name")[["name", "is_other_recipe"]].copy()
    # a raw name can be a recipe block in one place and not in another: group by the majority
    maj = gbl.groupby("name").is_other_recipe.mean() >= 0.5
    mp["group"], mp["rule"] = zip(*[group_of(n, maj[n]) for n in mp.name])
    mp = mp.merge(nt_all[["name", "count"]], on="name").sort_values(["group", "count"], ascending=[True, False])
    mp[["name", "count", "group", "rule"]].to_csv(TAB / "step2_proposed_grouping_mapping.csv", index=False)
    gbl["group"] = gbl.name.map(dict(zip(mp.name, mp.group)))
    gt = (gbl.groupby("group").agg(blocks=("name", "size"), participants=("participant", "nunique"),
                                   top_participant=("participant", lambda p: f"{Counter(p).most_common(1)[0][0]} ({Counter(p).most_common(1)[0][1]})"))
          .sort_values("blocks", ascending=False))
    gt["pct"] = (100 * gt.blocks / gt.blocks.sum()).round(1)
    gt.to_csv(TAB / "step2_proposed_groups.csv")
    res["step2"]["proposed_groups"] = gt.reset_index().to_dict("records")
    res["step2"]["grouping_rules"] = [dict(group=g_, regex=r_) for g_, r_ in RULES]

    # action level inside the gap vs recipe R's own narrations
    ev_recipes = set(ev.recipe)
    r_narr = [i for i, bl in zip(narr.index, narr.blocks) if any(set(blk.recs[b_]) & ev_recipes for b_ in bl)]
    # action level: narrations of the NON-recipe gap blocks vs all narrations inside any recipe block
    # (gap blocks of another recipe are just that recipe's actions, so they are left out here)
    nonrec_gap_blocks = {b_ for l in ev.gap_blocks for b_ in l if not blk.recs[b_]}
    gap_ids = sorted({n for b_ in nonrec_gap_blocks for n in narr_by_block[b_]})
    any_rec = [i for i, bl in zip(narr.index, narr.blocks) if any(blk.recs[b_] for b_ in bl)]
    r_other = sorted(set(any_rec) - set(gap_ids))
    feats = dict(main_action=lambda i: {narr.main_action[i]} if narr.main_action[i] else set(),
                 noun=lambda i: set(narr.nouns[i]))
    res["step2"]["action_level"] = dict(gap_narrations=len(gap_ids), recipe_narrations=len(r_other))
    for f, fn in feats.items():
        t = ratio_table([fn(i) for i in gap_ids], [fn(i) for i in r_other], MIN_COUNT)
        t.to_csv(TAB / f"step2_gap_{f}_ratio.csv", index=False)
        top_count = Counter(x for i in gap_ids for x in fn(i)).most_common(20)
        res["step2"]["action_level"][f] = dict(top20_by_ratio=t.head(20).to_dict("records"),
                                               top20_by_count=[dict(cls=k, n=c, pct=round(100 * c / len(gap_ids), 1)) for k, c in top_count])

    # ---------------- Step 3: waiting before the gap
    cmap = class_keyword_map(vc, nc)
    wait_verbs = {v for m in cmap.values() for v in m["verbs"]}
    wait_nouns = {v for m in cmap.values() for v in m["nouns"]}

    def wait_info(i):
        strict, loose = sentence_hits(narr.narration[i])
        cls = {k for k, m in cmap.items() if narr.main_verb[i] in m["verbs"] or narr.main_noun[i] in m["nouns"]
               or set(narr.nouns[i]) & set(m["nouns"])}
        return dict(sent_strict=strict, sent_loose_extra=loose, cls=cls)
    W = {i: wait_info(i) for i in set(r_narr) | {n for l in ev.last3 for n in l} | {n for l in ev.first3 for n in l}}
    sent = lambda i, loose=True: bool(W[i]["sent_strict"] or (loose and W[i]["sent_loose_extra"]))
    clsw = lambda i: bool(W[i]["cls"])
    kw = lambda i: sent(i) or clsw(i)

    def step3(e, label):
        last = [n for l in e.last3 for n in l]
        last_set = set(last)
        other = [i for i in r_narr if i not in last_set and narr.blocks[i] and
                 any(set(blk.recs[b_]) & set(e.recipe) for b_ in narr.blocks[i])]
        # baseline for events: all windows of 3 consecutive R narrations (per recipe and video) in the event recipes
        win_any = []
        for (R, v), g in narr.loc[[i for i in r_narr]].assign(
                rec=[[r for b_ in narr.blocks[i] for r in blk.recs[b_]] for i in r_narr]).explode("rec").groupby(["rec", "video_id"]):
            if R not in set(e.recipe):
                continue
            ids = g.sort_values("start").index.tolist()
            for k in range(len(ids) - N_EDGE + 1):
                w = ids[k:k + N_EDGE]
                win_any.append((any(sent(i) for i in w), any(clsw(i) for i in w), any(sent(i, False) for i in w)))
        win_any = np.array(win_any) if win_any else np.zeros((0, 3))
        ev_sent = np.array([any(sent(i) for i in l) for l in e.last3])
        ev_sent_strict = np.array([any(sent(i, False) for i in l) for l in e.last3])
        ev_cls = np.array([any(clsw(i) for i in l) for l in e.last3])
        ev_only_sent = np.array([any(sent(i) for i in l) and not any(clsw(i) for i in l) for l in e.last3])
        pp = (pd.DataFrame(dict(p=e.participant, w=ev_sent | ev_cls)).groupby("p").w.agg(["size", "mean"])
              .rename(columns={"size": "events", "mean": "pct_wait"}))
        pp["pct_wait"] = (100 * pp.pct_wait).round(1)
        return dict(
            label=label, events=len(e), narrations_last3=len(last), other_recipe_narrations=len(other),
            narration_level=dict(
                pct_last3_sentence=round(100 * np.mean([sent(i) for i in last]), 1),
                pct_last3_sentence_strict_only=round(100 * np.mean([sent(i, False) for i in last]), 1),
                pct_last3_verb_noun=round(100 * np.mean([clsw(i) for i in last]), 1),
                pct_other_sentence=round(100 * np.mean([sent(i) for i in other]), 1),
                pct_other_sentence_strict_only=round(100 * np.mean([sent(i, False) for i in other]), 1),
                pct_other_verb_noun=round(100 * np.mean([clsw(i) for i in other]), 1),
                n_last3_found_only_via_sentence=int(sum(sent(i) and not clsw(i) for i in last)),
                n_last3_found_only_via_verb_noun=int(sum(clsw(i) and not sent(i) for i in last)),
                n_last3_found_by_both=int(sum(clsw(i) and sent(i) for i in last))),
            event_level=dict(
                pct_events_wait_sentence=round(100 * ev_sent.mean(), 1),
                pct_events_wait_sentence_strict_only=round(100 * ev_sent_strict.mean(), 1),
                pct_events_wait_verb_noun=round(100 * ev_cls.mean(), 1),
                pct_events_wait_either=round(100 * (ev_sent | ev_cls).mean(), 1),
                n_events_found_only_via_sentence=int(ev_only_sent.sum()),
                baseline_pct_random_3windows_sentence=round(100 * win_any[:, 0].mean(), 1) if len(win_any) else None,
                baseline_pct_random_3windows_verb_noun=round(100 * win_any[:, 1].mean(), 1) if len(win_any) else None,
                baseline_pct_random_3windows_either=round(100 * (win_any[:, 0] | win_any[:, 1]).mean(), 1) if len(win_any) else None,
                baseline_n_windows=len(win_any)),
            per_participant=pp.reset_index().to_dict("records"),
            per_keyword=[dict(keyword=k,
                              last3_sentence=int(sum(k in W[i]["sent_strict"] | W[i]["sent_loose_extra"] for i in last)),
                              last3_verb_noun=int(sum(k in W[i]["cls"] for i in last)),
                              other_sentence_pct=round(100 * np.mean([k in W[i]["sent_strict"] | W[i]["sent_loose_extra"] for i in other]), 2),
                              examples_last3=[narr.narration[i] for i in last if k in W[i]["sent_strict"] | W[i]["sent_loose_extra"]][:3])
                         for k in WAIT_WORDS],
            loose_extra_examples=[narr.narration[i] for i in last if W[i]["sent_loose_extra"] and not W[i]["sent_strict"]][:10])
    res["step3"] = dict(keyword_class_map=cmap, all=step3(ev, "all events"), clean=step3(clean, "clean events only"))

    # ---------------- Step 3b: manual check (Claude reads each raw narration; blind to which set it comes from)
    last_all = sorted({n for l in ev.last3 for n in l})
    windows = []                                     # every 3-narration stretch of the event recipes, not touching a last-3
    for (R, v), g in narr.loc[r_narr].assign(rec=[[r for b_ in narr.blocks[i] for r in blk.recs[b_]] for i in r_narr]) \
            .explode("rec").groupby(["rec", "video_id"]):
        if R not in ev_recipes:
            continue
        ids = g.sort_values("start").index.tolist()
        windows += [(R, ids[k:k + N_EDGE]) for k in range(len(ids) - N_EDGE + 1) if not set(ids[k:k + N_EDGE]) & set(last_all)]
    rng = np.random.default_rng(SEED)
    base_win = [windows[k] for k in sorted(rng.choice(len(windows), N_BASELINE_WINDOWS, replace=False))]
    base_ids = sorted({n for _, w in base_win for n in w})
    key = pd.DataFrame([dict(narration_id=narr.narration_id[i], set="before_gap") for i in last_all] +
                       [dict(narration_id=narr.narration_id[i], set="baseline") for i in base_ids if i not in set(last_all)])
    key = key.sample(frac=1, random_state=SEED).reset_index(drop=True)
    key["item"] = key.index + 1
    text_of = dict(zip(narr.narration_id, narr.narration))
    key.to_csv(TAB / "to_label_key.csv", index=False)
    key.assign(narration=key.narration_id.map(text_of))[["item", "narration"]].to_csv(TAB / "to_label_blind.csv", index=False)
    res["step3"]["manual"] = dict(n_items=len(key), n_before_gap=len(last_all), n_baseline_narrations=len(base_ids),
                                  n_baseline_windows=N_BASELINE_WINDOWS, n_windows_available=len(windows), seed=SEED)
    if LABELS.exists():
        lab = pd.read_csv(LABELS)                            # item, waiting (0/1)
        lab = key.merge(lab, on="item")
        wait_of = dict(zip(lab.narration_id, lab.waiting.astype(bool)))
        nid = lambda i: narr.narration_id[i]
        cw = lambda i: wait_of[nid(i)]

        def wilson(k, n):
            if n == 0:
                return None
            p, z = k / n, 1.96
            c = (p + z * z / (2 * n)) / (1 + z * z / n)
            h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
            return [round(100 * (c - h), 1), round(100 * (c + h), 1)]

        def man(e, label):
            evw = np.array([any(cw(i) for i in l) for l in e.last3])
            bw = np.array([any(cw(i) for i in w) for _, w in base_win])
            last = [n for l in e.last3 for n in l]
            pp = (pd.DataFrame(dict(p=e.participant, w=evw)).groupby("p").w.agg(["size", "sum"])
                  .rename(columns={"size": "events", "sum": "events_after_waiting"}))
            pp["pct"] = (100 * pp.events_after_waiting / pp.events).round(1)
            # which of the last 3: the very last narration, or earlier?
            pos = Counter(len(l) - 1 - max(k for k, i in enumerate(l) if cw(i)) for l in e.last3 if any(cw(i) for i in l))
            return dict(label=label, events=len(e),
                        pct_events_after_waiting=round(100 * evw.mean(), 1), ci95=wilson(int(evw.sum()), len(evw)),
                        baseline_pct_windows_with_waiting=round(100 * bw.mean(), 1), baseline_ci95=wilson(int(bw.sum()), len(bw)),
                        pct_last3_narrations_waiting=round(100 * np.mean([cw(i) for i in last]), 1),
                        pct_baseline_narrations_waiting=round(100 * np.mean([cw(i) for i in base_ids]), 1),
                        waiting_position_from_gap={("last" if k == 0 else f"{k} before last"): v for k, v in sorted(pos.items())},
                        per_participant=pp.reset_index().to_dict("records"))
        both = [i for i in last_all]
        agree = dict(both=sum(cw(i) and kw(i) for i in both), claude_only=sum(cw(i) and not kw(i) for i in both),
                     keyword_only=sum(kw(i) and not cw(i) for i in both), neither=sum(not cw(i) and not kw(i) for i in both))
        res["step3"]["manual"].update(
            all=man(ev, "all events"), clean=man(clean, "clean events only"), agreement_with_keywords_last3=agree,
            examples_claude_only=[narr.narration[i] for i in both if cw(i) and not kw(i)][:12],
            examples_keyword_only=[narr.narration[i] for i in both if kw(i) and not cw(i)][:12],
            examples_waiting=[narr.narration[i] for i in both if cw(i)][:20])
        ev["wait_claude"] = [any(cw(i) for i in l) for l in ev.last3]
        no_drip = ev[ev.recipe != "P03_R03"]
        res["step3"]["manual"]["without_P03_drip_coffee"] = dict(
            events=len(no_drip), pct_events_after_waiting=round(100 * no_drip.wait_claude.mean(), 1),
            ci95=wilson(int(no_drip.wait_claude.sum()), len(no_drip)))
        res["step3"]["manual"]["waiting_events_by_recipe"] = (ev[ev.wait_claude].groupby(["recipe", "recipe_name"]).size()
                                                              .sort_values(ascending=False).head(12).reset_index(name="n").to_dict("records"))
        # what is the waiting action (main action of the waiting narration in the last 3)
        res["step3"]["manual"]["waiting_main_actions"] = Counter(narr.main_action[i] for l in ev.last3 for i in l if cw(i)).most_common(20)

    # data-driven ratio tables: main action / main verb / all nouns in the last 3 vs all other narrations of recipe R
    last_ids = sorted({n for l in ev.last3 for n in l})
    first_ids = sorted({n for l in ev.first3 for n in l})
    other_ids = sorted(set(r_narr) - set(last_ids))
    feats3 = dict(main_action=feats["main_action"], main_verb=lambda i: {narr.main_verb[i]} if narr.main_verb[i] else set(),
                  noun=feats["noun"])
    res["step3"]["ratio"] = dict(n_last3=len(last_ids), n_other=len(other_ids), n_first3=len(first_ids))
    for f, fn in feats3.items():
        A, B = [fn(i) for i in last_ids], [fn(i) for i in other_ids]
        t = ratio_table(A, B, MIN_COUNT)
        t.to_csv(TAB / f"step3_last3_{f}_ratio.csv", index=False)
        top = set(t.head(20).cls) if len(t) else set()
        robust = {}
        for m in (5, 20):
            t2 = ratio_table(A, B, m)
            robust[f"min{m}_top20_overlap_with_min10"] = len(top & set(t2.head(20).cls)) if len(t2) else 0
            robust[f"min{m}_n_classes"] = len(t2)
        t_first = ratio_table([fn(i) for i in first_ids], [fn(i) for i in sorted(set(r_narr) - set(first_ids))], MIN_COUNT)
        t_first.to_csv(TAB / f"step3_first3_{f}_ratio.csv", index=False)
        res["step3"]["ratio"][f] = dict(top20=t.head(20).to_dict("records"), n_classes=len(t), robustness=robust,
                                        first3_top10=t_first.head(10).to_dict("records"))

    # block names with "wait"
    res["step3"]["block_names"] = dict(
        gap_blocks_named_wait=nt_all[nt_all.name.str.contains(r"\bwait", case=False)][["name", "count", "participants"]].to_dict("records"),
        block_before_gap_name_has_wait_word=int(sum(bool(sentence_hits(n)[0] | sentence_hits(n)[1]) for n in ev.name_before)),
        block_before_gap_examples=[n for n in ev.name_before if sentence_hits(n)[0] | sentence_hits(n)[1]][:15])

    # per-event table with raw narrations
    def txt(ids):
        return " || ".join(narr.narration[i] for i in ids)
    if "wait_claude" not in ev:
        ev["wait_claude"] = None
    ev_out = ev.assign(
        last3_narrations=ev.last3.apply(txt), first3_narrations=ev.first3.apply(txt),
        gap_block_names=ev.gap_blocks.apply(lambda l: " | ".join(blk.name[b_] for b_ in l)),
        wait_sentence=[any(sent(i) for i in l) for l in ev.last3], wait_verb_noun=[any(clsw(i) for i in l) for l in ev.last3],
        wait_keywords=[", ".join(sorted({k for i in l for k in W[i]["sent_strict"] | W[i]["sent_loose_extra"] | W[i]["cls"]})) for l in ev.last3])
    ev_out.drop(columns=["gap_narr", "last3", "first3", "gap_blocks"]).to_csv(TAB / "events.csv", index=False)

    # ---------------- Step 4: example timelines
    elig = ev_out[(ev_out.flag == "clean") & ev_out.n_gap_blocks.between(1, 3) & ev_out.gap_s.between(20, 600) &
                  (ev.last3.str.len() == N_EDGE) & (ev.first3.str.len() == N_EDGE)]
    med = ev.gap_s.median()
    picks = []
    for p in elig.participant.value_counts().index:
        g = elig[elig.participant == p]
        picks.append(g.iloc[(g.gap_s - med).abs().argsort()].iloc[0])
        if len(picks) == 5:
            break
    res["step4"] = dict(rule="clean events with 1-3 gap blocks, gap 20-600 s, >= 3 narrations before and after; "
                             "one per participant (participants with the most eligible events first); within a participant the event "
                             "whose gap length is closest to the overall median gap", median_gap_s=med,
                        picks=[dict(event_id=x.event_id, participant=x.participant, recipe=x.recipe, recipe_name=x.recipe_name,
                                    gap_s=x.gap_s, file=f"example_{k + 1}_{x.participant}.png") for k, x in enumerate(picks)])
    for k, x in enumerate(picks):
        draw(x, ev.set_index("event_id").loc[x.event_id], blocks, narr, FIG / f"example_{k + 1}_{x.participant}.png", rname)

    json.dump(res, open(OUT / "results.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False, default=str)
    print(json.dumps({k: res[k] for k in ("data", "step1")}, indent=1, ensure_ascii=False, default=str)[:6000])


def draw(x, e, blocks, narr, path, rname):
    bi, bj = blocks.loc[e.block_before], blocks.loc[e.block_after]
    pad = max(60.0, 1.5 * max(bj.s - bi.e, 1.0))                  # zoom: up to `pad` seconds of recipe R on each side
    t0, t1 = max(bi.s, bi.e - pad), min(bj.e, bj.s + pad)
    win =blocks[(blocks.video_id == x.video_id) & (blocks.e > t0) & (blocks.s < t1) & ~blocks.broken].sort_values("s")
    lanes, ends = {}, []
    for b_ in win.itertuples():                       # overlapping blocks go to separate lanes
        for li, le in enumerate(ends):
            if b_.s >= le - 1e-6:
                ends[li] = b_.e
                lanes[b_.block_id] = li
                break
        else:
            lanes[b_.block_id] = len(ends)
            ends.append(b_.e)
    fig = plt.figure(figsize=(15, 6.2))
    ax = fig.add_axes([0.04, 0.47, 0.94, 0.42])
    for b_ in win.itertuples():
        isR = x.recipe in b_.recs
        col = "#d62728" if isR else ("#ff9f40" if b_.recs else "#9e9e9e")
        y = -lanes[b_.block_id]
        s_, e_ = max(b_.s, t0), min(b_.e, t1)
        ax.barh(y, e_ - s_, left=s_, height=0.7, color=col, alpha=0.85, edgecolor="black", lw=0.5)
        label = "\n".join(textwrap.wrap(b_.name, max(14, int((e_ - s_) / (t1 - t0) * 150))))     # raw name, never cut
        ax.text(s_ + (e_ - s_) / 2, y, label, ha="center", va="center", fontsize=7.2, color="white" if isR else "black")
    ax.axvspan(bi.e, bj.s, color="#1f77b4", alpha=0.10, zorder=0)
    ax.annotate("", xy=(bj.s, 0.75), xytext=(bi.e, 0.75), arrowprops=dict(arrowstyle="<->", color="#1f77b4"))
    ax.text((bi.e + bj.s) / 2, 0.95, f"gap: {x.gap_s:.0f} s, {x.n_gap_narr} narrations", ha="center", color="#1f77b4", fontsize=9)
    ax.set_xlim(t0, t1), ax.set_ylim(-len(ends) + 0.4, 1.25), ax.set_yticks([])
    ax.set_xlabel("seconds in video")
    ax.set_title(f"{x.video_id}  |  recipe {x.recipe}: {rname.get(x.recipe, '')}  (red = this recipe, orange = another recipe, grey = no recipe)",
                 fontsize=10)

    def box(ids, title, xpos):
        lines = [title] + [f"{narr.start[i]:7.1f}s  " + "\n           ".join(textwrap.wrap(narr.narration[i], 62)) for i in ids]
        fig.text(xpos, 0.36, "\n".join(lines), va="top", ha="left", fontsize=8.2, family="monospace",
                 bbox=dict(boxstyle="round", fc="#fff5f5", ec="#d62728"))
    box(e.last3, "LAST 3 narrations of the recipe BEFORE the gap", 0.02)
    box(e.first3, "FIRST 3 narrations AFTER the recipe resumes", 0.51)
    fig.savefig(path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
