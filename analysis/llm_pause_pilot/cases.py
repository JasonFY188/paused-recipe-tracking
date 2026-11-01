"""
analysis/llm_pause_pilot/cases.py
Builds the query cases for the LLM pause pilot from the pause-and-resume events (analysis/pause_resume/).

For each event (recipe R paused, then resumed in the same video):
  A  middle of the gap: the middle narration (by count) among the narrations that start inside the gap
     [end of R's block before the gap, start of the resumed R block); history = narrations up to and including it
  B  right at the resume: the first narration of the resumed R block; history up to and including it
Ground truth from activity blocks: R is paused at A, and active / continued at B.
"next recipe" truth: the recipe of the first recipe block starting after the query narration (R at B by definition).
One-line rule (baseline): the most recent recipe seen in the past narrations' blocks, excluding the block(s) that contain
the query narration, is the paused one and the one continued next. It uses the true block labels of the past.

Output: data/cases.pkl (one row per event x point)
"""
import json
import pathlib
import sys

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "analysis" / "pause_resume"))
import pause_resume as pr                                               # loaders only (unchanged)


def main():
    blk, _, _ = pr.load_blocks()
    narr, _, _ = pr.load_narrations()                                   # raw text (outer spaces stripped), time-sorted
    raw = pd.read_pickle(ROOT / "annotations" / "narrations-and-action-segments" / "HD_EPIC_Narrations.pkl")
    raw_text = dict(zip(raw.unique_narration_id, raw.narration))        # exactly as written
    narr["text"] = narr.narration_id.map(raw_text)
    narr["blocks"] = pr.assign(narr, blk)
    ev = pd.read_csv(ROOT / "analysis" / "pause_resume" / "tables" / "events.csv")
    recipes = json.load(open(ROOT / "annotations" / "high-level" / "complete_recipes.json", encoding="utf-8"))
    bs, be = dict(zip(blk.block_id, blk.s)), dict(zip(blk.block_id, blk.e))
    recs_of = dict(zip(blk.block_id, blk.recs))
    rows = []
    for e in ev.itertuples():
        g = narr[narr.video_id == e.video_id]
        gap0, gap1 = be[e.block_before], bs[e.block_after]
        R = e.recipe
        person = e.participant
        recipe_list = {k[4:]: recipes[k]["name"] for k in sorted(recipes) if k.startswith(person)}
        in_gap = g[(g.start >= gap0) & (g.start < gap1)]
        resumed = g[(g.start >= gap1) & g.blocks.apply(lambda l: e.block_after in l)]
        points = {}
        if len(in_gap):
            points["A"] = in_gap.index[len(in_gap) // 2]
        if len(resumed):
            points["B"] = resumed.index[0]
        for pt, idx in points.items():
            hist = g.loc[:idx]
            cur_blocks = set(g.loc[idx, "blocks"])
            # one-line rule: most recent recipe among past narrations, excluding the query narration's own blocks
            rule = None
            for bl in reversed(hist.blocks.tolist()[:-1]):
                rs = [r for b_ in bl if b_ not in cur_blocks for r in recs_of[b_]]
                if rs:
                    rule = rs[0]
                    break
            if pt == "B":                                               # the person is continuing R right now
                nxt = R
            else:
                later = blk[(blk.video_id == e.video_id) & (blk.s > g.loc[idx, "start"]) & (blk.recs.apply(len) > 0)].sort_values("s")
                nxt = later.recs.iloc[0][0] if len(later) else None
            rows.append(dict(case=f"{e.event_id}_{pt}", event_id=e.event_id, point=pt, participant=person,
                             video_id=e.video_id, recipe=R, recipe_id=R[4:], recipe_name=e.recipe_name,
                             gap_s=e.gap_s, n_gap_narr=e.n_gap_narr, flag=e.flag,
                             gap_has_other_recipe=e.gap_has_other_recipe,
                             history=[(float(s), t) for s, t in zip(hist.start, hist.text)],
                             query_time=float(g.loc[idx, "start"]), recipe_list=recipe_list,
                             true_next=nxt[4:] if nxt else None, rule_recipe=rule[4:] if rule else None))
    c = pd.DataFrame(rows)
    (HERE / "data").mkdir(exist_ok=True)
    c.to_pickle(HERE / "data" / "cases.pkl")
    c["n_hist"] = c.history.apply(len)
    print(len(c), c.point.value_counts().to_dict(), c.event_id.nunique(), "events")
    print(c.n_hist.describe(percentiles=[.5, .9, .99]).round(0).to_dict())
    print("rule says R:", c.groupby("point").apply(lambda d: (d.rule_recipe == d.recipe_id).mean()).round(3).to_dict())
    print("true next == R:", c.groupby("point").apply(lambda d: (d.true_next == d.recipe_id).mean()).round(3).to_dict())


if __name__ == "__main__":
    main()
